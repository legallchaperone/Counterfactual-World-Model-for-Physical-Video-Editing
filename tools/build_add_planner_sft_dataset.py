#!/usr/bin/env python3
"""Build add-planner SFT data from Phase-1A self-insertion composites.

Self-insertion inversion (see docs/E2W_SPEC.md Add Planner Output Contract):
an object that was composited into a clean background becomes an *add* target.
For each ``operation == "add"`` self-insert row we emit one Qwen2.5-VL SFT row:

    user:  [background first frame (object absent)] + canonical add planner prompt
    assistant: {target_ref, edit_type:"add", vace_prompt, primary_point, primary_bbox}

where ``target_ref``/``vace_prompt`` come from the manifest (``metadata.object_name``
and the positive ``prompt``) and ``primary_point``/``primary_bbox`` are the norm1000
centroid/bbox of the inserted object's ``primary_mask.npy``. Every emitted assistant
object is validated against the current add contract; rows that fail are skipped and
recorded, never silently repaired.

Output matches the trainer (tools/train_qwen25vl_lora_sft.py): train.jsonl / eval.jsonl
with messages[0].content = [image, text] and messages[1].content = assistant JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import build_add_planner_user_prompt, extract_first_frame, validate_add_planner_output  # noqa: E402

DATASET_ROOT = Path("/data/cwx/E2W/data/phase1a_pexels_self_insert_v1")
DEFAULT_TRAIN_MANIFEST = DATASET_ROOT / "03_self_insert/manifests/self_insert_train.jsonl"
DEFAULT_EVAL_MANIFEST = DATASET_ROOT / "03_self_insert/manifests/eval_4.jsonl"
DEFAULT_OUTPUT_DIR = Path("/data/cwx/E2W/data/add_planner_sft")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    p.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def primary_mask_2d(mask: np.ndarray) -> np.ndarray:
    """Collapse a [T,H,W] (or [H,W]) primary mask to a 2D boolean frame-0 mask."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"primary mask must be 2D or 3D, got shape {arr.shape}")
    return arr.astype(bool)


def point_bbox_norm1000(mask2d: np.ndarray) -> tuple[list[int], list[int]]:
    """Centroid point and bbox of the nonzero region, in norm1000 [0,1000] coords."""
    ys, xs = np.where(mask2d)
    if xs.size == 0:
        raise ValueError("primary mask is empty; cannot derive grounding")
    h, w = mask2d.shape
    cx, cy = float(xs.mean()), float(ys.mean())
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    def nx(v: float) -> int:
        return int(round(min(1000.0, max(0.0, v / max(1, w - 1) * 1000.0))))

    def ny(v: float) -> int:
        return int(round(min(1000.0, max(0.0, v / max(1, h - 1) * 1000.0))))

    point = [nx(cx), ny(cy)]
    bbox = [nx(x1), ny(y1), nx(x2), ny(y2)]
    # guard against degenerate bbox after rounding
    if bbox[0] >= bbox[2]:
        bbox[2] = min(1000, bbox[0] + 1)
    if bbox[1] >= bbox[3]:
        bbox[3] = min(1000, bbox[1] + 1)
    return point, bbox


def build_add_assistant_obj(target_ref: str, vace_prompt: str, point: list[int], bbox: list[int]) -> dict[str, Any]:
    return {
        "target_ref": target_ref,
        "edit_type": "add",
        "vace_prompt": vace_prompt,
        "primary_point": point,
        "primary_bbox": bbox,
    }


def build_sft_row(manifest_row: dict[str, Any], *, dataset_root: Path, frames_dir: Path) -> dict[str, Any]:
    """Build one validated add-planner SFT row from a self-insert add manifest row.

    Raises ValueError if the row cannot produce a contract-valid assistant object."""
    meta = manifest_row.get("metadata") or {}
    target_ref = str(meta.get("object_name") or "").strip()
    if not target_ref:
        raise ValueError("manifest row missing metadata.object_name")
    vace_prompt = str(manifest_row.get("prompt") or "").strip()

    mask = np.load(dataset_root / manifest_row["primary_mask_npy"])
    point, bbox = point_bbox_norm1000(primary_mask_2d(mask))
    assistant_obj = build_add_assistant_obj(target_ref, vace_prompt, point, bbox)
    ok, err = validate_add_planner_output(assistant_obj)
    if not ok:
        raise ValueError(f"assistant object fails add contract: {err}")

    sample_id = str(manifest_row.get("sample_id") or manifest_row.get("composite_id") or "sample")
    src_video = dataset_root / manifest_row["src_video"]
    image_path = frames_dir / f"{sample_id}_background.png"
    extract_first_frame(src_video, image_path)

    user_request = f"Add {target_ref} to the scene."
    user_text = build_add_planner_user_prompt(user_request, sample_id=sample_id)
    return {
        "video_id": sample_id,
        "source": str(manifest_row.get("dataset_source") or "pexels_self_insertion"),
        "operation": "add",
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": user_text}]},
            {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def build_split(manifest: Path, *, dataset_root: Path, frames_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in load_jsonl(manifest):
        if row.get("operation") != "add":
            continue
        try:
            out.append(build_sft_row(row, dataset_root=dataset_root, frames_dir=frames_dir))
        except (ValueError, FileNotFoundError, KeyError) as exc:
            skipped.append({"sample_id": row.get("sample_id"), "error": f"{type(exc).__name__}: {exc}"})
    return out, skipped


def main() -> int:
    args = parse_args()
    frames_dir = args.output_dir / "frames"
    train_rows, train_skipped = build_split(args.train_manifest, dataset_root=args.dataset_root, frames_dir=frames_dir)
    eval_rows, eval_skipped = build_split(args.eval_manifest, dataset_root=args.dataset_root, frames_dir=frames_dir)
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "eval.jsonl", eval_rows)
    summary = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_skipped": train_skipped,
        "eval_skipped": eval_skipped,
        "source": "phase1a_self_insertion_inversion",
        "contract": "e2w add planner (validate_add_planner_output)",
        "note": "distribution shift: self-insertion teaches adding objects that were really composited in; "
        "hold out hand-authored novel-add cases for eval.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("train_rows", "eval_rows")}, indent=2))
    print(f"output_dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

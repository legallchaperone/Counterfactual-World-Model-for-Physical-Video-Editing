#!/usr/bin/env python3
"""Run or prepare the first-frame edit stage for the E2W v0 bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-e2w")

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import (  # noqa: E402
    DEFAULT_RUN_DIR,
    ensure_run_dirs,
    load_manifest,
    serialize_first_frame_prompt,
    write_json,
    write_manifest,
    write_text,
)


DEFAULT_QWEN_IMAGE_EDIT = Path("/data/cwx/Edit2World-unified/checkpoints/Qwen-Image-Edit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--mode", default="mode_c_full_predicted")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--backend",
        choices=["opencv_inpaint_debug", "qwen_image_edit"],
        default="opencv_inpaint_debug",
        help="opencv_inpaint_debug is a contract smoke backend; qwen_image_edit runs the real first-frame editor.",
    )
    parser.add_argument("--qwen-checkpoint", type=Path, default=DEFAULT_QWEN_IMAGE_EDIT)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def index_manifest(entries: list[dict[str, Any]]) -> dict[tuple[str | None, str, str | None], dict[str, Any]]:
    out: dict[tuple[str | None, str, str | None], dict[str, Any]] = {}
    for entry in entries:
        sample_id = entry.get("sample_id")
        if sample_id:
            out[(entry.get("stage"), sample_id, entry.get("mode"))] = entry
    return out


def lookup_planner_entry(
    indexed: dict[tuple[str | None, str, str | None], dict[str, Any]],
    sample_id: str,
    mode: str,
) -> dict[str, Any] | None:
    return (
        indexed.get(("planner_eval", sample_id, mode))
        or indexed.get((None, sample_id, mode))
        or indexed.get(("planner_eval", sample_id, "planner_pred"))
        or indexed.get((None, sample_id, "planner_pred"))
    )


def primary_mask_first_frame(mask_entry: dict[str, Any], out_path: Path) -> np.ndarray:
    primary_path = Path(mask_entry["paths"]["primary_mask"])
    cap = cv2.VideoCapture(str(primary_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read primary mask frame: {primary_path}")
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    mask = (gray > 127).astype(np.uint8) * 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out_path)
    return mask


def opencv_inpaint(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    edited = cv2.inpaint(bgr, dilated, 5, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(edited, cv2.COLOR_BGR2RGB))


def load_qwen_pipe(args: argparse.Namespace) -> Any:
    import torch
    from diffusers import QwenImageEditPipeline

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pipe = QwenImageEditPipeline.from_pretrained(str(args.qwen_checkpoint), torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe


def qwen_edit(pipe: Any, image: Image.Image, prompt: str, args: argparse.Namespace) -> Image.Image:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipe(
        image.convert("RGB"),
        prompt=prompt,
        negative_prompt=" ",
        num_inference_steps=args.steps,
        generator=generator,
        true_cfg_scale=args.true_cfg_scale,
        num_images_per_prompt=1,
    ).images[0]
    return result


def build_failed_entry(
    args: argparse.Namespace,
    sample_id: str,
    out_dir: Path,
    reason: str,
    failure_source: str,
    planner_entry: dict[str, Any] | None,
    mask_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    result = {
        "sample_id": sample_id,
        "mode": args.mode,
        "backend": args.backend,
        "first_frame_ok": False,
        "failure_source": failure_source,
        "reason": reason,
    }
    eval_path = out_dir / "first_frame_edit_metadata.json"
    write_json(eval_path, result)
    paths: dict[str, str] = {"first_frame_metadata": str(eval_path)}
    if planner_entry:
        paths.update({k: v for k, v in planner_entry.get("paths", {}).items() if k in {"edit_plan", "first_frame"}})
    if mask_entry:
        paths.update({k: v for k, v in mask_entry.get("paths", {}).items() if k in {"primary_mask", "mask_eval"}})
    return {
        "stage": "first_frame",
        "sample_id": sample_id,
        "mode": args.mode,
        "status": "failed",
        "failure_source": failure_source,
        "paths": paths,
        "metrics": result,
    }


def process_one(
    args: argparse.Namespace,
    sample_id: str,
    planner_entry: dict[str, Any] | None,
    mask_entry: dict[str, Any] | None,
    qwen_pipe: Any | None,
) -> dict[str, Any]:
    out_dir = args.run_dir / "first_frame" / sample_id / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    if planner_entry is None:
        return build_failed_entry(args, sample_id, out_dir, "missing planner entry", "planner", None, mask_entry)
    if mask_entry is None:
        return build_failed_entry(args, sample_id, out_dir, "missing mask entry", "mask", planner_entry, None)
    if mask_entry.get("status") != "ok":
        failure = mask_entry.get("failure_source") or "mask"
        return build_failed_entry(args, sample_id, out_dir, "upstream mask stage failed", failure, planner_entry, mask_entry)

    edit_plan = json.loads(Path(planner_entry["paths"]["edit_plan"]).read_text(encoding="utf-8"))
    prompt = serialize_first_frame_prompt(edit_plan)
    write_text(out_dir / "first_frame_prompt.txt", prompt)
    mask = primary_mask_first_frame(mask_entry, out_dir / "qwen_edit_target_mask.png")

    source_image_path = Path(planner_entry["paths"]["first_frame"])
    edited_path = out_dir / "edited_first_frame.png"
    started = time.time()
    if args.force or not edited_path.exists():
        image = Image.open(source_image_path).convert("RGB")
        source_size = image.size
        if args.backend == "opencv_inpaint_debug":
            edited = opencv_inpaint(image, mask)
        elif args.backend == "qwen_image_edit":
            if qwen_pipe is None:
                raise RuntimeError("qwen_image_edit backend selected without a loaded pipeline")
            edited = qwen_edit(qwen_pipe, image, prompt, args)
        else:
            raise ValueError(args.backend)
        raw_size = edited.size
        if raw_size != source_size:
            edited = edited.resize(source_size, Image.Resampling.LANCZOS)
        edited.save(edited_path)
    else:
        source_size = Image.open(source_image_path).size
        raw_size = Image.open(edited_path).size
    runtime = time.time() - started

    result = {
        "sample_id": sample_id,
        "mode": args.mode,
        "backend": args.backend,
        "first_frame_ok": True,
        "failure_source": None,
        "runtime_sec": runtime,
        "seed": args.seed,
        "steps": args.steps if args.backend == "qwen_image_edit" else None,
        "true_cfg_scale": args.true_cfg_scale if args.backend == "qwen_image_edit" else None,
        "source_size": list(source_size),
        "raw_output_size": list(raw_size),
        "edited_size": list(Image.open(edited_path).size),
        "target_mask_consumed_by_backend": args.backend == "opencv_inpaint_debug",
        "note": "opencv_inpaint_debug is not Qwen-Image-Edit; use --backend qwen_image_edit for the real editor."
        if args.backend == "opencv_inpaint_debug"
        else "Qwen-Image-Edit first-frame editor. Target mask is saved for QC/debug only and is not consumed by this backend.",
    }
    metadata_path = out_dir / "first_frame_edit_metadata.json"
    write_json(metadata_path, result)
    return {
        "stage": "first_frame",
        "sample_id": sample_id,
        "mode": args.mode,
        "status": "ok",
        "failure_source": None,
        "paths": {
            "source_first_frame": str(source_image_path),
            "primary_mask_frame0": str(out_dir / "qwen_edit_target_mask.png"),
            "first_frame_prompt": str(out_dir / "first_frame_prompt.txt"),
            "edited_first_frame": str(edited_path),
            "first_frame_metadata": str(metadata_path),
        },
        "metrics": result,
    }


def main() -> None:
    args = parse_args()
    ensure_run_dirs(args.run_dir)
    manifest_path = args.run_dir / "manifest.jsonl"
    manifest = load_manifest(manifest_path)
    indexed = index_manifest(manifest)
    planner_entries = [e for e in manifest if e.get("stage") in (None, "planner_eval")]
    sample_ids = [e["sample_id"] for e in planner_entries]
    if args.sample_id:
        wanted = set(args.sample_id)
        sample_ids = [sid for sid in sample_ids if sid in wanted]
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]
    if not sample_ids:
        raise ValueError("No samples selected")

    qwen_pipe = load_qwen_pipe(args) if args.backend == "qwen_image_edit" else None
    new_entries = []
    for idx, sample_id in enumerate(sample_ids, start=1):
        print(f"[{idx}/{len(sample_ids)}] first-frame {sample_id}", flush=True)
        planner_entry = lookup_planner_entry(indexed, sample_id, args.mode)
        mask_entry = indexed.get(("mask_builder", sample_id, args.mode))
        new_entries.append(process_one(args, sample_id, planner_entry, mask_entry, qwen_pipe))

    kept = [
        entry
        for entry in manifest
        if not (entry.get("stage") == "first_frame" and entry.get("mode") == args.mode and entry.get("sample_id") in set(sample_ids))
    ]
    all_entries = kept + new_entries
    write_manifest(manifest_path, all_entries)
    mode_entries = [e for e in all_entries if e.get("stage") == "first_frame" and e.get("mode") == args.mode]
    summary = {
        "count": len(mode_entries),
        "ok_count": sum(e.get("status") == "ok" for e in mode_entries),
        "last_backend": args.backend,
        "failure_sources": {},
        "backends": {},
    }
    for entry in mode_entries:
        fs = entry.get("failure_source") or "none"
        summary["failure_sources"][fs] = summary["failure_sources"].get(fs, 0) + 1
        backend = entry.get("metrics", {}).get("backend", "unknown")
        summary["backends"][backend] = summary["backends"].get(backend, 0) + 1
    write_json(args.run_dir / "first_frame" / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

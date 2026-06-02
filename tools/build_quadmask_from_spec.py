#!/usr/bin/env python3
"""Build E2W quadmasks from planner quadmask specs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import (  # noqa: E402
    DEFAULT_RUN_DIR,
    ensure_run_dirs,
    make_debug_contact_sheet,
    primary_grounding_from_spec,
    read_video_rgb,
    validate_quadmask_spec,
    video_meta,
    write_gray_video,
    write_json,
    write_rgb_video,
)


DEFAULT_SAM2_REPO = Path("/data/cwx/Edit2World-unified/external/sam2")
DEFAULT_SAM2_CKPT = Path("/data/cwx/Edit2World-unified/checkpoints/sam2/sam2.1_hiera_large.pt")
DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--mode", default="mode_c_full_predicted")
    parser.add_argument("--max-frames", type=int, default=81)
    parser.add_argument("--sam2-repo", type=Path, default=DEFAULT_SAM2_REPO)
    parser.add_argument("--sam2-ckpt", type=Path, default=DEFAULT_SAM2_CKPT)
    parser.add_argument("--sam2-cfg", default=DEFAULT_SAM2_CFG)
    parser.add_argument("--fallback-rect", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_manifest(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "manifest.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_manifest(run_dir: Path, entries: list[dict[str, Any]]) -> None:
    path = run_dir / "manifest.jsonl"
    planner_entries = [e for e in load_manifest(run_dir) if e.get("stage") != "mask_builder"]
    by_key = {(e.get("stage"), e.get("sample_id"), e.get("mode")): e for e in planner_entries}
    for entry in entries:
        by_key[(entry.get("stage"), entry.get("sample_id"), entry.get("mode"))] = entry
    with path.open("w", encoding="utf-8") as f:
        for entry in by_key.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def rasterize_grid_regions(spec: dict[str, Any], t: int, h: int, w: int) -> np.ndarray:
    out = np.zeros((t, h, w), dtype=bool)
    affected_v6 = spec.get("affected", {}) if isinstance(spec.get("affected"), dict) else {}
    grid_shape = affected_v6.get("grid_shape")
    if isinstance(grid_shape, list) and len(grid_shape) == 2:
        try:
            rows, cols = int(grid_shape[0]), int(grid_shape[1])
        except (TypeError, ValueError):
            rows, cols = 0, 0
        for item in affected_v6.get("frame_ranges", []) or []:
            if not isinstance(item, dict) or rows <= 0 or cols <= 0:
                continue
            start = item.get("start_frame")
            end = item.get("end_frame")
            cells = item.get("cells")
            if not isinstance(start, int) or not isinstance(end, int) or not isinstance(cells, list):
                continue
            start = max(0, min(t - 1, start))
            end = max(start, min(t - 1, end))
            for cell in cells:
                box = grid_cell_to_box(str(cell), rows, cols)
                if box is None:
                    continue
                r1, c1, r2, c2 = box
                y1, y2 = int(round(r1 * h / rows)), int(round(r2 * h / rows))
                x1, x2 = int(round(c1 * w / cols)), int(round(c2 * w / cols))
                out[start : end + 1, y1:y2, x1:x2] = True

    grid = spec.get("grid", {}) if isinstance(spec.get("grid"), dict) else {}
    rows = int(grid.get("rows") or 0)
    cols = int(grid.get("cols") or 0)
    if rows <= 0 or cols <= 0:
        return out
    affected = spec.get("affected_regions", [])
    if not isinstance(affected, list):
        return out
    cell_h = h / rows
    cell_w = w / cols
    for region in affected:
        if not isinstance(region, dict):
            continue
        for item in region.get("grid_boxes", []) or []:
            if not isinstance(item, dict):
                continue
            frame_idx = item.get("frame_index")
            box = item.get("box")
            if not isinstance(frame_idx, int) or frame_idx < 0 or frame_idx >= t:
                continue
            if not isinstance(box, list) or len(box) != 4:
                continue
            r1, c1, r2, c2 = [int(x) for x in box]
            r1, c1 = max(0, r1), max(0, c1)
            r2, c2 = min(rows, r2), min(cols, c2)
            if r1 >= r2 or c1 >= c2:
                continue
            y1, y2 = int(round(r1 * cell_h)), int(round(r2 * cell_h))
            x1, x2 = int(round(c1 * cell_w)), int(round(c2 * cell_w))
            out[frame_idx, y1:y2, x1:x2] = True
    return out


def grid_cell_to_box(cell: str, rows: int, cols: int) -> list[int] | None:
    match = re.match(r"^([A-Za-z]+)([1-9][0-9]*)$", cell.strip())
    if not match:
        return None
    letters, number = match.groups()
    row = 0
    for ch in letters.upper():
        row = row * 26 + (ord(ch) - ord("A") + 1)
    col = int(number)
    if not (1 <= row <= rows and 1 <= col <= cols):
        return None
    r0 = row - 1
    c0 = col - 1
    return [r0, c0, r0 + 1, c0 + 1]


def rect_mask_from_bbox(bbox: list[int], t: int, h: int, w: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(x) for x in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    mask = np.zeros((t, h, w), dtype=bool)
    if x1 < x2 and y1 < y2:
        mask[:, y1:y2, x1:x2] = True
    return mask


def sam2_propagate(args: argparse.Namespace, video_clip: Path, spec: dict[str, Any]) -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("SAM2 propagation requires CUDA in this setup")
    sys.path.insert(0, str(args.sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor  # noqa: WPS433

    predictor = build_sam2_video_predictor(args.sam2_cfg, str(args.sam2_ckpt), device="cuda")
    grounding = primary_grounding_from_spec(spec, video_meta(video_clip))
    bbox = grounding["bbox"]
    point = grounding["point"]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(
            video_path=str(video_clip),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        if bbox:
            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=1,
                box=np.array(bbox, dtype=np.float32),
            )
        elif point:
            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=1,
                points=np.array([point], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )
        else:
            raise ValueError("No primary bbox or point available for SAM2")

        video_segments: dict[int, np.ndarray] = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            if len(out_obj_ids) == 0:
                continue
            mask = (out_mask_logits[0] > 0.0).detach().cpu().numpy()
            if mask.ndim == 3:
                mask = mask[0]
            video_segments[int(out_frame_idx)] = mask.astype(bool)

    if not video_segments:
        raise RuntimeError("SAM2 returned no masks")
    frame_count = max(video_segments) + 1
    first = next(iter(video_segments.values()))
    out = np.zeros((frame_count, first.shape[0], first.shape[1]), dtype=bool)
    for idx, mask in video_segments.items():
        out[idx] = mask
    return out


def build_one(args: argparse.Namespace, planner_entry: dict[str, Any]) -> dict[str, Any]:
    sample_id = planner_entry["sample_id"]
    mask_dir = args.run_dir / "masks" / sample_id / args.mode
    contact_dir = args.run_dir / "contact_sheets" / sample_id
    mask_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    spec_path = Path(planner_entry["paths"]["quadmask_spec"])
    started = time.time()
    eval_path = mask_dir / "mask_eval.json"
    if planner_entry.get("status") not in (None, "ok"):
        result = {
            "sample_id": sample_id,
            "mode": args.mode,
            "mask_valid": False,
            "failure_source": "planner",
            "reason": f"upstream planner status is {planner_entry.get('status')}",
            "runtime_sec": time.time() - started,
        }
        write_json(eval_path, result)
        return {
            "stage": "mask_builder",
            "sample_id": sample_id,
            "mode": args.mode,
            "status": "failed",
            "failure_source": "planner",
            "paths": {"quadmask_spec": str(spec_path), "mask_eval": str(eval_path)},
            "metrics": result,
        }

    video_path = Path(planner_entry["paths"]["original_video"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    meta = video_meta(video_path)
    spec_metrics = validate_quadmask_spec(spec, meta)
    if not spec_metrics["quadmask_spec_executable"]:
        result = {
            "sample_id": sample_id,
            "mode": args.mode,
            "mask_valid": False,
            "failure_source": "planner_grounding",
            "reason": "quadmask_spec is not executable: missing or invalid primary bbox/point",
            **spec_metrics,
        }
        write_json(eval_path, result)
        return {
            "stage": "mask_builder",
            "sample_id": sample_id,
            "mode": args.mode,
            "status": "failed",
            "failure_source": "planner_grounding",
            "paths": {"quadmask_spec": str(spec_path), "mask_eval": str(eval_path)},
            "metrics": result,
        }

    frames, fps = read_video_rgb(video_path, max_frames=args.max_frames)
    if not frames:
        raise RuntimeError(f"No frames in {video_path}")
    h, w = frames[0].shape[:2]
    clip_path = mask_dir / "source_clip.mp4"
    if args.force or not clip_path.exists():
        write_rgb_video(clip_path, frames, fps=fps)

    try:
        grounding = primary_grounding_from_spec(spec, {"width": w, "height": h, "frame_count": len(frames)})
        if args.fallback_rect:
            primary = rect_mask_from_bbox(grounding["bbox"], len(frames), h, w)
            mask_source = "rect_fallback_from_bbox"
        else:
            primary = sam2_propagate(args, clip_path, spec)
            mask_source = "sam2_from_planner_grounding"
    except Exception as exc:
        result = {
            "sample_id": sample_id,
            "mode": args.mode,
            "mask_valid": False,
            "failure_source": "sam2",
            "reason": f"{type(exc).__name__}: {exc}",
            **spec_metrics,
        }
        write_json(eval_path, result)
        return {
            "stage": "mask_builder",
            "sample_id": sample_id,
            "mode": args.mode,
            "status": "failed",
            "failure_source": "sam2",
            "paths": {"quadmask_spec": str(spec_path), "source_clip": str(clip_path), "mask_eval": str(eval_path)},
            "metrics": result,
        }

    # Align SAM2 output with the decoded clip if needed.
    primary = primary[: len(frames)]
    if primary.shape[1:] != (h, w):
        resized = []
        import cv2

        for mask in primary:
            resized.append(cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
        primary = np.stack(resized, axis=0)

    affected = rasterize_grid_regions(spec, len(frames), h, w)
    editable = primary | affected
    quadmask = np.full((len(frames), h, w), 255, dtype=np.uint8)
    quadmask[primary & ~affected] = 0
    quadmask[primary & affected] = 63
    quadmask[affected & ~primary] = 127
    values = sorted(int(x) for x in np.unique(quadmask))
    editable_ratio = float((quadmask != 255).mean())
    mask_valid = (
        set(values).issubset({0, 63, 127, 255})
        and quadmask.ndim == 3
        and int(primary.sum()) > 0
        and editable_ratio < 0.70
    )

    np.save(mask_dir / "quadmask.npy", quadmask)
    write_gray_video(mask_dir / "primary_mask.mp4", primary.astype(np.uint8) * 255, fps=fps)
    write_gray_video(mask_dir / "affected_mask.mp4", affected.astype(np.uint8) * 255, fps=fps)
    write_gray_video(mask_dir / "editable_mask.mp4", editable.astype(np.uint8) * 255, fps=fps)
    write_gray_video(mask_dir / "quadmask.mp4", quadmask, fps=fps)
    make_debug_contact_sheet(
        contact_dir / f"{args.mode}_debug_contact_sheet.png",
        frames,
        primary,
        affected,
        quadmask,
    )

    result = {
        "sample_id": sample_id,
        "mode": args.mode,
        "mask_valid": mask_valid,
        "failure_source": None if mask_valid else "mask_builder",
        "mask_source": mask_source,
        "quadmask_values": values,
        "quadmask_shape": list(quadmask.shape),
        "primary_area_ratio": float(primary.mean()),
        "affected_area_ratio": float(affected.mean()),
        "editable_area_ratio": editable_ratio,
        "runtime_sec": time.time() - started,
        **spec_metrics,
    }
    write_json(eval_path, result)
    return {
        "stage": "mask_builder",
        "sample_id": sample_id,
        "mode": args.mode,
        "status": "ok" if mask_valid else "failed",
        "failure_source": result["failure_source"],
        "paths": {
            "quadmask_spec": str(spec_path),
            "source_clip": str(clip_path),
            "primary_mask": str(mask_dir / "primary_mask.mp4"),
            "affected_mask": str(mask_dir / "affected_mask.mp4"),
            "editable_mask": str(mask_dir / "editable_mask.mp4"),
            "quadmask_npy": str(mask_dir / "quadmask.npy"),
            "quadmask": str(mask_dir / "quadmask.mp4"),
            "debug_contact_sheet": str(contact_dir / f"{args.mode}_debug_contact_sheet.png"),
            "mask_eval": str(eval_path),
        },
        "metrics": result,
    }


def main() -> None:
    args = parse_args()
    ensure_run_dirs(args.run_dir)
    planner_entries = [e for e in load_manifest(args.run_dir) if e.get("stage") in (None, "planner_eval")]
    if args.sample_id:
        wanted = set(args.sample_id)
        planner_entries = [e for e in planner_entries if e.get("sample_id") in wanted]
    if not planner_entries:
        raise ValueError("No planner entries found. Run eval_vlm_planner.py first.")

    entries = []
    metrics = []
    for idx, entry in enumerate(planner_entries, start=1):
        print(f"[{idx}/{len(planner_entries)}] build mask {entry['sample_id']}", flush=True)
        out = build_one(args, entry)
        entries.append(out)
        metrics.append(out["metrics"])

    write_manifest(args.run_dir, entries)
    summary = {
        "count": len(metrics),
        "mask_valid_count": sum(bool(m.get("mask_valid")) for m in metrics),
        "mask_valid_rate": sum(bool(m.get("mask_valid")) for m in metrics) / len(metrics),
        "primary_nonempty_count": sum(float(m.get("primary_area_ratio", 0.0)) > 0 for m in metrics),
        "affected_nonempty_count": sum(float(m.get("affected_area_ratio", 0.0)) > 0 for m in metrics),
        "valid_quadmask_value_count": sum(
            bool(m.get("quadmask_values")) and set(m.get("quadmask_values", [])).issubset({0, 63, 127, 255})
            for m in metrics
        ),
        "avg_editable_area_ratio": float(np.mean([m.get("editable_area_ratio", 0.0) for m in metrics])),
        "failure_sources": {},
    }
    for m in metrics:
        fs = m.get("failure_source") or "none"
        summary["failure_sources"][fs] = summary["failure_sources"].get(fs, 0) + 1
    write_json(args.run_dir / "masks" / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

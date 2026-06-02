#!/usr/bin/env python3
"""Package v0.2 Qwen+VACE smoke artifacts into per-sample flat bundles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import load_manifest, write_json, write_text  # noqa: E402


DEFAULT_RUN = Path("/data/cwx/E2W/runs/e2w_v0_2_qwen_vace_smoke")
DEFAULT_MODE = "mode_v0_2_qwen_vace_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    return parser.parse_args()


def symlink_file(src: str | Path | None, dst: Path) -> str | None:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    if not src_path.is_absolute():
        src_path = src_path.absolute()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src_path, dst)
    return str(dst)


def index_manifest(entries: list[dict[str, Any]]) -> dict[tuple[str | None, str, str | None], dict[str, Any]]:
    out: dict[tuple[str | None, str, str | None], dict[str, Any]] = {}
    for entry in entries:
        sample_id = entry.get("sample_id")
        if sample_id:
            out[(entry.get("stage"), sample_id, entry.get("mode"))] = entry
    return out


def video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"exists": path.exists(), "opened": False}
    info = {
        "exists": path.exists(),
        "opened": True,
        "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
    }
    cap.release()
    return info


def frame_stats(path: Path, max_frames: int = 81) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    means: list[float] = []
    stds: list[float] = []
    frames = 0
    while frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        means.append(float(gray.mean()))
        stds.append(float(gray.std()))
        frames += 1
    cap.release()
    black_like = [i for i, (m, s) in enumerate(zip(means, stds)) if m < 8.0 and s < 4.0]
    return {
        "sampled_frames": frames,
        "mean_luma_min": min(means) if means else None,
        "mean_luma_max": max(means) if means else None,
        "std_luma_min": min(stds) if stds else None,
        "black_like_frame_indices": black_like,
        "has_black_frame_failure": len(black_like) > max(2, frames // 4),
    }


def first_frame_diff(edited_first_frame: Path, edited_video: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(edited_video))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        return {"computed": False}
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.open(edited_first_frame).convert("RGB").resize((frame.shape[1], frame.shape[0]), Image.Resampling.LANCZOS)
    arr = np.array(image)
    diff = np.abs(arr.astype(np.float32) - frame.astype(np.float32))
    return {
        "computed": True,
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
    }


def read_video_frames(path: Path, indices: list[int]) -> list[np.ndarray]:
    info = video_info(path)
    if not info.get("opened"):
        return []
    cap = cv2.VideoCapture(str(path))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def thumb(img: Image.Image | np.ndarray, label: str, size: tuple[int, int] = (240, 135)) -> Image.Image:
    if isinstance(img, np.ndarray):
        out = Image.fromarray(img)
    else:
        out = img.convert("RGB")
    out = out.resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, min(size[0], 180), 18], fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255))
    return out


def make_contact_sheet(out_path: Path, flat_dir: Path) -> None:
    original = flat_dir / "original.mp4"
    primary_mask = flat_dir / "primary_mask.mp4"
    affected_mask = flat_dir / "affected_mask.mp4"
    quadmask = flat_dir / "quadmask.mp4"
    vace = flat_dir / "edited_video.mp4"
    qwen = flat_dir / "edited_first_frame.png"
    rows: list[list[Image.Image]] = []
    original_info = video_info(original)
    n = int(original_info.get("num_frames") or 1)
    indices = [0, max(0, n // 2), max(0, n - 1)]
    rows.append([thumb(img, f"original {idx}") for img, idx in zip(read_video_frames(original, indices), indices)])
    rows.append([thumb(img, f"primary {idx}") for img, idx in zip(read_video_frames(primary_mask, indices), indices)])
    rows.append([thumb(img, f"affected {idx}") for img, idx in zip(read_video_frames(affected_mask, indices), indices)])
    rows.append([thumb(img, f"quadmask {idx}") for img, idx in zip(read_video_frames(quadmask, indices), indices)])
    if qwen.exists():
        rows.append([thumb(Image.open(qwen), "qwen first frame")])
    vace_info = video_info(vace)
    if vace_info.get("opened"):
        m = int(vace_info.get("num_frames") or 1)
        vindices = [0, max(0, m // 2), max(0, m - 1)]
        rows.append([thumb(img, f"vace {idx}") for img, idx in zip(read_video_frames(vace, vindices), vindices)])
    width = 240 * max(len(row) for row in rows if row)
    height = 135 * len(rows)
    sheet = Image.new("RGB", (width, height), (24, 24, 24))
    y = 0
    for row in rows:
        x = 0
        for img in row:
            sheet.paste(img, (x, y))
            x += 240
        y += 135
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def build_first_frame_qc(first_entry: dict[str, Any] | None) -> dict[str, Any]:
    metrics = (first_entry or {}).get("metrics", {})
    ok = bool(first_entry and first_entry.get("status") == "ok")
    source_size = metrics.get("source_size")
    edited_size = metrics.get("edited_size")
    resize_ok = bool(source_size and edited_size and source_size == edited_size)
    system_interface_ok = ok and resize_ok
    backend = metrics.get("backend")
    qwen_interface_ok = bool(system_interface_ok and backend == "qwen_image_edit")
    visual_review_status = metrics.get("visual_review_status") or "unreviewed"
    failure_source = "none"
    if not ok:
        failure_source = "qwen_artifact"
    elif not resize_ok:
        failure_source = "resize"
    return {
        "primary_removed": None,
        "background_fill": None,
        "keep_preservation": None,
        "artifact_level": None,
        "system_interface_ok": system_interface_ok,
        "qwen_interface_ok": qwen_interface_ok,
        "qwen_visual_review_status": visual_review_status,
        "visual_review_status": visual_review_status,
        "failure_source": failure_source,
        "notes": "System QC only. Scores require visual review: 0=failed, 1=partial, 2=good.",
        "backend": backend,
        "runtime_sec": metrics.get("runtime_sec"),
        "source_size": source_size,
        "edited_size": edited_size,
        "raw_output_size": metrics.get("raw_output_size"),
        "has_resize_mismatch": bool(ok and not resize_ok),
        "target_mask_consumed_by_backend": metrics.get("target_mask_consumed_by_backend"),
    }


def build_vace_runtime(vace_entry: dict[str, Any] | None, flat_dir: Path) -> dict[str, Any]:
    metadata = (vace_entry or {}).get("metrics", {})
    edited_video = flat_dir / "edited_video.mp4"
    info = video_info(edited_video)
    stats = frame_stats(edited_video)
    condition_info = video_info(flat_dir / "vace_conditioning.mp4")
    expected_frames = int(metadata.get("condition_shape", [81])[0] or 81)
    completed = bool(vace_entry and vace_entry.get("status") == "ok" and info.get("opened"))
    fps = float(info.get("fps") or 0.0)
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    aspect_video = width / height if height else 0.0
    aspect_condition = (
        float(condition_info.get("width", 0)) / float(condition_info.get("height", 1))
        if condition_info.get("opened")
        else 0.0
    )
    resize_mismatch = bool(completed and aspect_condition and abs(aspect_video - aspect_condition) / aspect_condition > 0.08)
    failure_source = "none"
    if not completed:
        failure_source = "vace_runtime"
    elif int(info.get("num_frames") or 0) != expected_frames:
        failure_source = "codec"
    elif stats.get("has_black_frame_failure"):
        failure_source = "generation_mask"
    elif resize_mismatch:
        failure_source = "condition_video"
    return {
        "completed": completed,
        "num_frames": info.get("num_frames"),
        "resolution": f"{width}x{height}" if width and height else "",
        "fps": fps,
        "runtime_sec": metadata.get("runtime_sec"),
        "has_black_frame_failure": bool(stats.get("has_black_frame_failure")),
        "has_resize_mismatch": resize_mismatch,
        "failure_source": failure_source,
        "video_info": info,
        "condition_info": condition_info,
        "frame_stats": stats,
        "first_frame_diff": first_frame_diff(flat_dir / "edited_first_frame.png", edited_video)
        if edited_video.exists() and (flat_dir / "edited_first_frame.png").exists()
        else {"computed": False},
    }


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.run_dir / "manifest.jsonl")
    indexed = index_manifest(manifest)
    sample_ids = sorted({entry["sample_id"] for entry in manifest if entry.get("stage") == "planner_eval"})
    rows = []
    for sample_id in sample_ids:
        planner = indexed.get(("planner_eval", sample_id, args.mode))
        mask = indexed.get(("mask_builder", sample_id, args.mode))
        first = indexed.get(("first_frame", sample_id, args.mode))
        vace = indexed.get(("vace_v0", sample_id, args.mode))
        if not planner or not mask:
            continue
        flat_dir = args.run_dir / sample_id
        flat_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        paths["original"] = symlink_file(planner["paths"].get("original_video"), flat_dir / "original.mp4")
        paths["edit_plan"] = symlink_file(planner["paths"].get("edit_plan"), flat_dir / "edit_plan.json")
        paths["quadmask_spec"] = symlink_file(planner["paths"].get("quadmask_spec"), flat_dir / "quadmask_spec.json")
        paths["primary_mask"] = symlink_file(mask["paths"].get("primary_mask"), flat_dir / "primary_mask.mp4")
        paths["affected_mask"] = symlink_file(mask["paths"].get("affected_mask"), flat_dir / "affected_mask.mp4")
        paths["quadmask_npy"] = symlink_file(mask["paths"].get("quadmask_npy"), flat_dir / "quadmask.npy")
        paths["quadmask"] = symlink_file(mask["paths"].get("quadmask"), flat_dir / "quadmask.mp4")
        if first:
            paths["qwen_prompt"] = symlink_file(first["paths"].get("first_frame_prompt"), flat_dir / "qwen_prompt.txt")
            paths["edited_first_frame"] = symlink_file(first["paths"].get("edited_first_frame"), flat_dir / "edited_first_frame.png")
        if vace:
            paths["vace_prompt"] = symlink_file(vace["paths"].get("vace_prompt"), flat_dir / "vace_prompt.txt")
            paths["vace_generation_mask"] = symlink_file(vace["paths"].get("vace_generation_mask"), flat_dir / "vace_generation_mask.mp4")
            paths["edited_video"] = symlink_file(vace["paths"].get("edited_video"), flat_dir / "edited_video.mp4")
            symlink_file(vace["paths"].get("vace_conditioning"), flat_dir / "vace_conditioning.mp4")
        first_qc = build_first_frame_qc(first)
        vace_runtime = build_vace_runtime(vace, flat_dir)
        failure_source = "none"
        if not first_qc["system_interface_ok"]:
            failure_source = first_qc["failure_source"]
        elif not vace_runtime["completed"]:
            failure_source = vace_runtime["failure_source"]
        elif vace_runtime["has_black_frame_failure"]:
            failure_source = "generation_mask"
        elif vace_runtime["has_resize_mismatch"]:
            failure_source = "condition_video"
        failure = {"sample_id": sample_id, "failure_source": failure_source}
        write_json(flat_dir / "first_frame_qc.json", first_qc)
        write_json(flat_dir / "vace_runtime.json", vace_runtime)
        write_json(flat_dir / "failure.json", failure)
        make_contact_sheet(flat_dir / "contact_sheet.png", flat_dir)
        rows.append(
            {
                "sample_id": sample_id,
                "system_interface_ok": first_qc["system_interface_ok"],
                "qwen_interface_ok": first_qc["qwen_interface_ok"],
                "qwen_visual_review_status": first_qc["qwen_visual_review_status"],
                "vace_completed": vace_runtime["completed"],
                "num_frames": vace_runtime["num_frames"],
                "resolution": vace_runtime["resolution"],
                "fps": vace_runtime["fps"],
                "black_frame_failure": vace_runtime["has_black_frame_failure"],
                "resize_mismatch": vace_runtime["has_resize_mismatch"],
                "failure_source": failure_source,
                "flat_dir": str(flat_dir),
            }
        )
    summary = {
        "count": len(rows),
        "system_interface_ok": sum(bool(r["system_interface_ok"]) for r in rows),
        "qwen_interface_ok": sum(bool(r["qwen_interface_ok"]) for r in rows),
        "qwen_visual_review_status": {
            status: sum(r["qwen_visual_review_status"] == status for r in rows)
            for status in sorted({str(r["qwen_visual_review_status"]) for r in rows})
        },
        "vace_completed": sum(bool(r["vace_completed"]) for r in rows),
        "black_frame_failures": sum(bool(r["black_frame_failure"]) for r in rows),
        "resize_mismatches": sum(bool(r["resize_mismatch"]) for r in rows),
        "rows": rows,
    }
    write_json(args.run_dir / "v0_2_smoke_summary.json", summary)
    lines = [
        "# E2W v0.2 Qwen + VACE Smoke",
        "",
        "E2W v0.2 verifies real first-frame editing and VACE executor integration on primary-good teacher-grounded samples.",
        "",
        f"- samples: {summary['count']}",
        f"- system interface ok: {summary['system_interface_ok']}/{summary['count']}",
        f"- qwen_image_edit interface ok: {summary['qwen_interface_ok']}/{summary['count']}",
        f"- qwen visual review status: `{json.dumps(summary['qwen_visual_review_status'], sort_keys=True)}`",
        f"- VACE completed: {summary['vace_completed']}/{summary['count']}",
        f"- black-frame failures: {summary['black_frame_failures']}",
        f"- resize mismatches: {summary['resize_mismatches']}",
        "- visual success is not inferred from size/interface checks; first-frame visual review defaults to `unreviewed`.",
        "",
        "| sample_id | system_interface_ok | qwen_interface_ok | qwen_visual_review | vace_completed | frames | resolution | fps | black_frame | resize_mismatch | failure_source | artifact_dir |",
        "|---|---:|---:|---|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {sample_id} | {system_interface_ok} | {qwen_interface_ok} | {qwen_visual_review_status} | {vace_completed} | {num_frames} | {resolution} | {fps:.1f} | {black_frame_failure} | {resize_mismatch} | {failure_source} | `{flat_dir}` |".format(
                **row
            )
        )
    write_text(args.run_dir / "report.md", "\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

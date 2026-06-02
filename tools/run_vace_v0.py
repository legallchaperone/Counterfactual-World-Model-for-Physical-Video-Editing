#!/usr/bin/env python3
"""Prepare or run the Wan-VACE legacy executor for E2W v0."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import (  # noqa: E402
    DEFAULT_RUN_DIR,
    ensure_run_dirs,
    load_manifest,
    read_video_rgb,
    write_gray_video,
    write_json,
    write_manifest,
    write_rgb_video,
    write_text,
)


DEFAULT_VACE_REPO = Path("/data/cwx/Edit2World-unified/external/VACE")
DEFAULT_VACE_CKPT = Path("/data/cwx/Edit2World-unified/checkpoints/Wan2.1-VACE-14B")
DEFAULT_PYTHON = Path("/data/cwx/conda/envs/edit2world-phase1-real/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--mode", default="mode_c_full_predicted")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--size", default="480p")
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--vace-repo", type=Path, default=DEFAULT_VACE_REPO)
    parser.add_argument("--vace-ckpt", type=Path, default=DEFAULT_VACE_CKPT)
    parser.add_argument("--run-vace", action="store_true")
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


def build_generation_mask(frame_num: int, h: int, w: int) -> np.ndarray:
    # VACE convention: black/0 is known retained condition, white/255 is generated.
    mask = np.ones((frame_num, h, w), dtype=np.uint8) * 255
    mask[0] = 0
    return mask


def build_condition_video(edited_first_frame: Path, frame_num: int) -> list[np.ndarray]:
    first = np.array(Image.open(edited_first_frame).convert("RGB"))
    black = np.zeros_like(first)
    return [first] + [black.copy() for _ in range(frame_num - 1)]


def quadmask_frame_check(mask_entry: dict[str, Any], frame_num: int) -> dict[str, Any]:
    quadmask_path = mask_entry.get("paths", {}).get("quadmask_npy")
    result: dict[str, Any] = {
        "quadmask_path": quadmask_path,
        "quadmask_frame_count": None,
        "requested_frame_num": frame_num,
        "quadmask_frame_mismatch": False,
        "warnings": [],
    }
    if not quadmask_path:
        result["warnings"].append("quadmask_npy missing from mask entry")
        return result
    try:
        quadmask = np.load(quadmask_path, mmap_mode="r")
        result["quadmask_frame_count"] = int(quadmask.shape[0])
        if int(quadmask.shape[0]) != int(frame_num):
            result["quadmask_frame_mismatch"] = True
            result["warnings"].append(
                f"v0 warning: quadmask frame count {int(quadmask.shape[0])} != requested frame_num {int(frame_num)}"
            )
    except Exception as exc:
        result["warnings"].append(f"could not inspect quadmask_npy: {type(exc).__name__}: {exc}")
    return result


def failed_entry(
    args: argparse.Namespace,
    sample_id: str,
    out_dir: Path,
    reason: str,
    failure_source: str,
    first_frame_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    result = {
        "sample_id": sample_id,
        "mode": args.mode,
        "executor": "E2W-VACE wrapper v0, no trained quadmask branch",
        "vace_completed": False,
        "failure_source": failure_source,
        "reason": reason,
        "quadmask_semantics": "stored/logged only; not consumed by legacy Wan-VACE",
    }
    metadata_path = out_dir / "vace_metadata.json"
    write_json(metadata_path, result)
    paths = {"vace_metadata": str(metadata_path)}
    if first_frame_entry:
        paths.update(first_frame_entry.get("paths", {}))
    return {
        "stage": "vace_v0",
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
    first_frame_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    out_dir = args.run_dir / "vace" / sample_id / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    if planner_entry is None:
        return failed_entry(args, sample_id, out_dir, "missing planner entry", "planner", first_frame_entry)
    if mask_entry is None or mask_entry.get("status") != "ok":
        return failed_entry(args, sample_id, out_dir, "upstream mask stage failed", "mask", first_frame_entry)
    if first_frame_entry is None or first_frame_entry.get("status") != "ok":
        return failed_entry(args, sample_id, out_dir, "upstream first-frame stage failed", "first_frame", first_frame_entry)

    edited_first_frame = Path(first_frame_entry["paths"]["edited_first_frame"])
    vace_prompt = Path(planner_entry["paths"]["vace_prompt"]).read_text(encoding="utf-8")
    write_text(out_dir / "vace_prompt.txt", vace_prompt)
    quadmask_check = quadmask_frame_check(mask_entry, args.frame_num)

    condition_path = out_dir / "vace_conditioning.mp4"
    generation_mask_path = out_dir / "vace_generation_mask.mp4"
    if args.force or not condition_path.exists() or not generation_mask_path.exists():
        cond_frames = build_condition_video(edited_first_frame, args.frame_num)
        h, w = cond_frames[0].shape[:2]
        generation_mask = build_generation_mask(args.frame_num, h, w)
        write_rgb_video(condition_path, cond_frames, fps=args.fps)
        write_gray_video(generation_mask_path, generation_mask, fps=args.fps)
    else:
        frames, _ = read_video_rgb(condition_path, max_frames=1)
        h, w = frames[0].shape[:2]

    edited_video = out_dir / "edited_video.mp4"
    condition_arg = condition_path.resolve()
    generation_mask_arg = generation_mask_path.resolve()
    out_dir_arg = out_dir.resolve()
    edited_video_arg = edited_video.resolve()
    command = [
        str(args.python.resolve()),
        str((args.vace_repo / "vace" / "vace_wan_inference.py").resolve()),
        "--model_name",
        "vace-14B",
        "--size",
        args.size,
        "--ckpt_dir",
        str(args.vace_ckpt.resolve()),
        "--src_video",
        str(condition_arg),
        "--src_mask",
        str(generation_mask_arg),
        "--prompt",
        vace_prompt,
        "--frame_num",
        str(args.frame_num),
        "--save_dir",
        str(out_dir_arg),
        "--save_file",
        str(edited_video_arg),
        "--use_prompt_extend",
        "plain",
        "--sample_steps",
        str(args.sample_steps),
        "--base_seed",
        str(args.seed),
    ]
    command_path = out_dir / "vace_command.json"
    write_json(command_path, {"argv": command, "cwd": str(args.vace_repo)})

    started = time.time()
    returncode: int | None = None
    if args.run_vace:
        env = os.environ.copy()
        env.setdefault("USE_TF", "0")
        env.setdefault("TRANSFORMERS_NO_TF", "1")
        proc = subprocess.run(
            command,
            cwd=str(args.vace_repo),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        returncode = proc.returncode
        write_text(out_dir / "vace_stdout.txt", proc.stdout)
        write_text(out_dir / "vace_stderr.txt", proc.stderr)

    completed = bool(args.run_vace and returncode == 0 and edited_video.exists())
    if completed:
        status = "ok"
        failure_source = None
        reason = None
    elif args.run_vace:
        status = "failed"
        failure_source = "vace"
        if returncode == 0 and not edited_video.exists():
            reason = "VACE command returned 0 but edited video was not created"
        else:
            reason = f"VACE command failed with returncode {returncode}"
    else:
        status = "prepared"
        failure_source = "not_run"
        reason = "VACE command prepared but not executed"
    result = {
        "sample_id": sample_id,
        "mode": args.mode,
        "executor": "E2W-VACE wrapper v0, no trained quadmask branch",
        "vace_completed": completed,
        "prepared_only": not args.run_vace,
        "returncode": returncode,
        "failure_source": failure_source,
        "reason": reason,
        "runtime_sec": time.time() - started,
        "condition_shape": [args.frame_num, h, w],
        "generation_mask_semantics": "0/black known-retained frame, 255/white generated frame",
        "generation_mask_rule": "frame 0 known, frames 1..T generated whole frame",
        "quadmask_semantics": "stored/logged only; not consumed by legacy Wan-VACE",
        **quadmask_check,
        "seed": args.seed,
        "sample_steps": args.sample_steps,
    }
    metadata_path = out_dir / "vace_metadata.json"
    write_json(metadata_path, result)
    paths = {
        "edited_first_frame": str(edited_first_frame),
        "quadmask_npy": mask_entry["paths"]["quadmask_npy"],
        "vace_prompt": str(out_dir / "vace_prompt.txt"),
        "vace_conditioning": str(condition_path),
        "vace_generation_mask": str(generation_mask_path),
        "vace_command": str(command_path),
        "vace_metadata": str(metadata_path),
    }
    if edited_video.exists():
        paths["edited_video"] = str(edited_video)
    return {
        "stage": "vace_v0",
        "sample_id": sample_id,
        "mode": args.mode,
        "status": status,
        "failure_source": result["failure_source"],
        "paths": paths,
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

    new_entries = []
    for idx, sample_id in enumerate(sample_ids, start=1):
        print(f"[{idx}/{len(sample_ids)}] vace-v0 {sample_id}", flush=True)
        planner_entry = lookup_planner_entry(indexed, sample_id, args.mode)
        mask_entry = indexed.get(("mask_builder", sample_id, args.mode))
        first_frame_entry = indexed.get(("first_frame", sample_id, args.mode))
        new_entries.append(process_one(args, sample_id, planner_entry, mask_entry, first_frame_entry))

    sample_set = set(sample_ids)
    kept = [
        entry
        for entry in manifest
        if not (entry.get("stage") == "vace_v0" and entry.get("mode") == args.mode and entry.get("sample_id") in sample_set)
    ]
    all_entries = kept + new_entries
    write_manifest(manifest_path, all_entries)
    mode_entries = [e for e in all_entries if e.get("stage") == "vace_v0" and e.get("mode") == args.mode]
    summary = {
        "count": len(mode_entries),
        "ok_count": sum(e.get("status") == "ok" for e in mode_entries),
        "prepared_count": sum(e.get("status") == "prepared" for e in mode_entries),
        "last_run_vace": args.run_vace,
        "failure_sources": {},
    }
    for entry in mode_entries:
        fs = entry.get("failure_source") or "none"
        summary["failure_sources"][fs] = summary["failure_sources"].get(fs, 0) + 1
    write_json(args.run_dir / "vace" / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

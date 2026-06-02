#!/usr/bin/env python3
"""Run a direct E2W v0.3 quadmask-consuming Wan-VACE experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/data/cwx/conda/envs/edit2world-phase1-real/bin/python")
DEFAULT_VACE_REPO = Path("/data/cwx/Edit2World-unified/external/VACE")
DEFAULT_VACE_CKPT = Path("/data/cwx/Edit2World-unified/checkpoints/Wan2.1-VACE-14B")
VOID_VALUES = {0, 63, 127, 255}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-video", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--quadmask-npy", type=Path, required=True)
    parser.add_argument("--operation", choices=["remove", "add"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--generation-mask-mode", choices=["quadmask-editable", "future-full-frame"], default="quadmask-editable")
    parser.add_argument("--align-quadmask", choices=["error", "nearest"], default="error")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--vace-repo", type=Path, default=DEFAULT_VACE_REPO)
    parser.add_argument("--vace-ckpt", type=Path, default=DEFAULT_VACE_CKPT)
    parser.add_argument("--model-name", default="vace-14B", choices=["vace-14B", "vace-1.3B"])
    parser.add_argument("--size", default="480p")
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--sample-shift", type=float, default=16.0)
    parser.add_argument("--sample-guide-scale", type=float, default=5.0)
    parser.add_argument("--base-seed", type=int, default=2025)
    parser.add_argument("--context-scale", type=float, default=1.0)
    parser.add_argument("--low-mem", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-vace", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def video_meta(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    meta = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 16.0),
    }
    cap.release()
    if meta["frame_count"] <= 0 or meta["width"] <= 0 or meta["height"] <= 0:
        raise RuntimeError(f"invalid video metadata for {video_path}: {meta}")
    if meta["frame_count"] % 4 != 1:
        raise RuntimeError(f"Wan2.1-VACE frame_count must be 4n+1, got {meta['frame_count']}")
    return meta


def load_quadmask(path: Path) -> np.ndarray:
    quad = np.load(path)
    if quad.ndim != 3:
        raise RuntimeError(f"quadmask must have shape [T,H,W], got {quad.shape}: {path}")
    values = sorted(int(x) for x in np.unique(quad))
    if not set(values).issubset(VOID_VALUES):
        raise RuntimeError(f"quadmask values must be {sorted(VOID_VALUES)}, got {values}: {path}")
    return quad.astype(np.uint8, copy=False)


def align_quadmask(quad: np.ndarray, meta: dict[str, Any], mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    target_shape = (int(meta["frame_count"]), int(meta["height"]), int(meta["width"]))
    source_shape = tuple(int(x) for x in quad.shape)
    info: dict[str, Any] = {
        "source_shape": list(source_shape),
        "target_shape": list(target_shape),
        "method": mode,
        "changed": source_shape != target_shape,
    }
    if source_shape == target_shape:
        info["frame_index_mapping_first_last"] = [0, source_shape[0] - 1]
        info["frame_index_mapping_unique_count"] = source_shape[0]
        return quad.copy(), info
    if mode == "error":
        raise RuntimeError(f"quadmask shape {source_shape} does not match video shape {target_shape}; pass --align-quadmask nearest to record deterministic alignment")
    if mode != "nearest":
        raise RuntimeError(f"unsupported quadmask alignment mode: {mode}")
    t_idx = np.rint(np.linspace(0, source_shape[0] - 1, target_shape[0])).astype(int)
    aligned = np.empty(target_shape, dtype=np.uint8)
    for out_i, src_i in enumerate(t_idx):
        aligned[out_i] = cv2.resize(quad[src_i], (target_shape[2], target_shape[1]), interpolation=cv2.INTER_NEAREST)
    values = sorted(int(x) for x in np.unique(aligned))
    if not set(values).issubset(VOID_VALUES):
        raise RuntimeError(f"aligned quadmask values must be {sorted(VOID_VALUES)}, got {values}")
    info["frame_index_mapping_first_last"] = [int(t_idx[0]), int(t_idx[-1])]
    info["frame_index_mapping_unique_count"] = int(len(set(int(x) for x in t_idx)))
    return aligned, info


def generation_mask_from_quadmask(quad: np.ndarray, mode: str) -> np.ndarray:
    if mode == "quadmask-editable":
        return np.where(quad != 255, 255, 0).astype(np.uint8)
    if mode == "future-full-frame":
        out = np.full_like(quad, 255, dtype=np.uint8)
        out[0] = 0
        return out
    raise RuntimeError(f"unsupported generation mask mode: {mode}")


def write_gray_video(path: Path, masks: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [np.repeat(frame[:, :, None], 3, axis=2) for frame in masks.astype(np.uint8)]
    imageio.mimwrite(str(path), frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)


def build_command(args: argparse.Namespace, aligned_quadmask: Path, generation_mask: Path, output_video: Path) -> list[str]:
    return [
        str(args.python.resolve()),
        str((TOOLS_DIR / "run_wan_vace_quad_i2v.py").resolve()),
        "--vace_repo",
        str(args.vace_repo.resolve()),
        "--ckpt_dir",
        str(args.vace_ckpt.resolve()),
        "--model_name",
        args.model_name,
        "--size",
        args.size,
        "--src_video",
        str(args.src_video.resolve()),
        "--generation_mask",
        str(generation_mask.resolve()),
        "--quadmask_npy",
        str(aligned_quadmask.resolve()),
        "--operation",
        args.operation,
        "--prompt",
        args.prompt,
        "--frame_num",
        str(video_meta(args.src_video)["frame_count"]),
        "--save_dir",
        str(args.run_dir.resolve()),
        "--save_file",
        str(output_video.resolve()),
        "--base_seed",
        str(args.base_seed),
        "--sample_steps",
        str(args.sample_steps),
        "--sample_shift",
        str(args.sample_shift),
        "--sample_guide_scale",
        str(args.sample_guide_scale),
        "--context_scale",
        str(args.context_scale),
        "--offload_model" if args.low_mem else "--no-offload_model",
    ]


def main() -> None:
    args = parse_args()
    started = time.time()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    meta = video_meta(args.src_video)
    raw_quad = load_quadmask(args.quadmask_npy)
    aligned, alignment = align_quadmask(raw_quad, meta, args.align_quadmask)
    aligned_path = args.run_dir / "quadmask_aligned.npy"
    np.save(aligned_path, aligned)
    generation_mask = generation_mask_from_quadmask(aligned, args.generation_mask_mode)
    generation_mask_path = args.run_dir / "vace_generation_mask_from_quadmask.mp4"
    write_gray_video(generation_mask_path, generation_mask, meta["fps"])
    prompt_path = args.run_dir / "prompt.txt"
    write_text(prompt_path, args.prompt + "\n")

    output_video = args.run_dir / "edited_video.mp4"
    command = build_command(args, aligned_path, generation_mask_path, output_video)
    write_json(args.run_dir / "vace_command.json", {"argv": command, "cwd": str(Path.cwd())})
    quadmask_passed_to_backend_command = all(flag in command for flag in ["--quadmask_npy", "--operation", "--generation_mask"])

    returncode: int | None = None
    if args.run_vace:
        env = os.environ.copy()
        env.setdefault("USE_TF", "0")
        env.setdefault("TRANSFORMERS_NO_TF", "1")
        if args.low_mem:
            env.setdefault("E2W_VACE_LOW_MEM", "1")
        if args.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        proc = subprocess.run(
            command,
            cwd=str(Path.cwd()),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        returncode = proc.returncode
        write_text(args.run_dir / "vace_stdout.txt", proc.stdout)
        write_text(args.run_dir / "vace_stderr.txt", proc.stderr)

    completed = bool(args.run_vace and returncode == 0 and output_video.exists())
    status = "ok" if completed else ("failed" if args.run_vace else "prepared")
    result = {
        "schema_version": "e2w.vace_quad_experiment.v0.3",
        "status": status,
        "vace_completed": completed,
        "returncode": returncode,
        "runtime_sec": time.time() - started,
        "src_video": str(args.src_video),
        "src_video_resolved": str(args.src_video.resolve()),
        "prompt": args.prompt,
        "operation": args.operation,
        "quadmask_passed_to_backend_command": quadmask_passed_to_backend_command,
        "quadmask_consumed_by_backend": completed,
        "quadmask_backend": "e2w_quad_i2v",
        "quadmask_semantics": {"0": "primary", "63": "primary_affected_overlap", "127": "affected", "255": "keep"},
        "generation_mask_mode": args.generation_mask_mode,
        "generation_mask_semantics": "255/generate, 0/keep",
        "video_meta": meta,
        "quadmask_source": str(args.quadmask_npy),
        "quadmask_source_resolved": str(args.quadmask_npy.resolve()),
        "quadmask_source_values": sorted(int(x) for x in np.unique(raw_quad)),
        "quadmask_alignment": alignment,
        "paths": {
            "prompt": str(prompt_path),
            "quadmask_aligned": str(aligned_path),
            "vace_generation_mask": str(generation_mask_path),
            "vace_command": str(args.run_dir / "vace_command.json"),
            "vace_stdout": str(args.run_dir / "vace_stdout.txt"),
            "vace_stderr": str(args.run_dir / "vace_stderr.txt"),
            "edited_video": str(output_video) if output_video.exists() else None,
        },
    }
    if args.run_vace and not completed:
        result["failure_source"] = "vace"
        result["reason"] = "VACE command failed or did not create edited_video.mp4"
    write_json(args.run_dir / "experiment_metadata.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.run_vace and not completed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run CONTROL-level perturbation tests against a completed bridge run.

Takes an existing bridge run directory (which has vace_conditioning_video,
quadmask_npy, generation_mask, vace_prompt, etc.) and runs four VACE inference
variants with perturbed inputs. Comparing outputs against the baseline
determines whether the model responds to operation and quadmask_npy signals.

Evidence level produced: STRUCTURAL_CONTROL_CANDIDATE
Visual review is required before any CONTROL claim is made.

Usage:
    python tools/run_control_perturbation_test.py \\
        --bridge-run-dir /data/cwx/E2W/runs/<run> \\
        --vace-repo /data/cwx/Edit2World-unified/external/VACE \\
        --vace-ckpt-dir /data/cwx/Edit2World-unified/checkpoints/Wan2.1-VACE-14B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (TOOLS, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_PYTHON = Path("/data/cwx/conda/envs/edit2world-phase1-real/bin/python")
DEFAULT_VACE_REPO = Path("/data/cwx/Edit2World-unified/external/VACE")
DEFAULT_VACE_CKPT = Path("/data/cwx/Edit2World-unified/checkpoints/Wan2.1-VACE-14B")


# --------------------------------------------------------------------------- #
# Quadmask perturbation helpers                                               #
# --------------------------------------------------------------------------- #

def build_q0_suppressed_quadmask(quadmask: np.ndarray) -> np.ndarray:
    """Replace all pixels with Q3 (255). Removes any edit region signal."""
    return np.full_like(quadmask, 255, dtype=np.uint8)


def build_q0_shifted_quadmask(quadmask: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    """Translate Q0 (0) and Q2 (127) regions by (shift_x, shift_y) pixels.

    Q3 (255) fills vacated pixels. Q1 (63) is treated as Q0 and also shifted.
    """
    result = np.full_like(quadmask, 255, dtype=np.uint8)
    T, H, W = quadmask.shape
    for t in range(T):
        frame = quadmask[t]
        edit_mask = (frame == 0) | (frame == 63) | (frame == 127)
        ys, xs = np.where(edit_mask)
        for y, x in zip(ys, xs):
            ny, nx = y + shift_y, x + shift_x
            if 0 <= ny < H and 0 <= nx < W:
                result[t, ny, nx] = frame[y, x]
    return result


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# VACE invocation                                                             #
# --------------------------------------------------------------------------- #

def run_vace_variant(
    *,
    out_dir: Path,
    vace_conditioning_video: Path,
    quadmask_npy: Path,
    generation_mask: Path,
    operation: str,
    vace_prompt: str,
    frame_num: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video = out_dir / "edited_video.mp4"
    command = [
        str(args.python.resolve()),
        str((TOOLS / "run_wan_vace_quad_i2v.py").resolve()),
        "--vace_repo", str(args.vace_repo.resolve()),
        "--ckpt_dir", str(args.vace_ckpt_dir.resolve()),
        "--model_name", args.model_name,
        "--size", args.size,
        "--src_video", str(vace_conditioning_video.resolve()),
        "--generation_mask", str(generation_mask.resolve()),
        "--quadmask_npy", str(quadmask_npy.resolve()),
        "--operation", operation,
        "--prompt", vace_prompt,
        "--frame_num", str(frame_num),
        "--save_dir", str(out_dir.resolve()),
        "--save_file", str(output_video.resolve()),
        "--base_seed", str(args.base_seed),
        "--sample_steps", str(args.sample_steps),
        "--offload_model",
    ]
    (out_dir / "command.json").write_text(
        json.dumps({"cmd": command, "cwd": str(ROOT)}, indent=2) + "\n", encoding="utf-8"
    )
    if args.skip_vace:
        return {"skipped": True, "output_video": str(output_video)}
    env = os.environ.copy()
    env.setdefault("USE_TF", "0")
    env.setdefault("TRANSFORMERS_NO_TF", "1")
    proc = subprocess.run(command, cwd=str(ROOT), env=env, text=True, capture_output=True, check=False)
    (out_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    return {
        "returncode": proc.returncode,
        "output_video": str(output_video),
        "output_video_exists": output_video.exists(),
        "output_video_bytes": output_video.stat().st_size if output_video.exists() else 0,
    }


# --------------------------------------------------------------------------- #
# Per-test runners                                                            #
# --------------------------------------------------------------------------- #

def run_test_operation_swap(
    baseline: dict[str, Any], out_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Flip operation; all other inputs identical to baseline."""
    original_op = baseline["operation"]
    swapped_op = "add" if original_op == "remove" else "remove"
    result = run_vace_variant(
        out_dir=out_dir,
        vace_conditioning_video=Path(baseline["vace_conditioning_video"]),
        quadmask_npy=Path(baseline["quadmask_npy"]),
        generation_mask=Path(baseline["generation_mask"]),
        operation=swapped_op,
        vace_prompt=baseline["vace_prompt"],
        frame_num=baseline["frame_num"],
        args=args,
    )
    return {"test": "operation_swap", "original_operation": original_op, "swapped_operation": swapped_op, **result}


def run_test_q0_suppressed(
    baseline: dict[str, Any], out_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Replace quadmask with all-255 (no Q0/Q2 edit signal)."""
    original_quad = np.load(baseline["quadmask_npy"])
    suppressed = build_q0_suppressed_quadmask(original_quad)
    quad_path = out_dir / "quadmask_suppressed.npy"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(quad_path, suppressed)
    result = run_vace_variant(
        out_dir=out_dir,
        vace_conditioning_video=Path(baseline["vace_conditioning_video"]),
        quadmask_npy=quad_path,
        generation_mask=Path(baseline["generation_mask"]),
        operation=baseline["operation"],
        vace_prompt=baseline["vace_prompt"],
        frame_num=baseline["frame_num"],
        args=args,
    )
    return {
        "test": "q0_suppressed",
        "original_quadmask_sha256": sha256_file(Path(baseline["quadmask_npy"])),
        "suppressed_quadmask_sha256": sha256_file(quad_path),
        "suppressed_unique_values": sorted(int(v) for v in np.unique(suppressed).tolist()),
        **result,
    }


def run_test_q0_shifted(
    baseline: dict[str, Any], out_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Shift Q0/Q2 region spatially; edit should follow the mask."""
    original_quad = np.load(baseline["quadmask_npy"])
    _, H, W = original_quad.shape
    shift_x = max(1, W // 5)
    shift_y = 0
    shifted = build_q0_shifted_quadmask(original_quad, shift_x=shift_x, shift_y=shift_y)
    quad_path = out_dir / "quadmask_shifted.npy"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(quad_path, shifted)
    result = run_vace_variant(
        out_dir=out_dir,
        vace_conditioning_video=Path(baseline["vace_conditioning_video"]),
        quadmask_npy=quad_path,
        generation_mask=Path(baseline["generation_mask"]),
        operation=baseline["operation"],
        vace_prompt=baseline["vace_prompt"],
        frame_num=baseline["frame_num"],
        args=args,
    )
    return {
        "test": "q0_shifted",
        "shift_x": shift_x,
        "shift_y": shift_y,
        "original_quadmask_sha256": sha256_file(Path(baseline["quadmask_npy"])),
        "shifted_quadmask_sha256": sha256_file(quad_path),
        **result,
    }


def run_test_q3_preservation(
    baseline: dict[str, Any], baseline_video: Path, out_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Record Q3 pixel stats for the baseline output.

    Visual comparison of Q3 pixels between the source and baseline output is
    required to claim preservation — this test only records the structural
    metadata needed for that comparison.
    """
    quad = np.load(baseline["quadmask_npy"])
    q3_fraction = float((quad == 255).mean())
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "test": "q3_preservation",
        "baseline_output_video": str(baseline_video),
        "baseline_output_exists": baseline_video.exists(),
        "q3_fraction_in_quadmask": q3_fraction,
        "q3_pixel_count": int((quad == 255).sum()),
        "visual_review_required": True,
        "note": "Q3 preservation requires human or model visual review of baseline output vs source frame at Q3 pixels.",
    }
    (out_dir / "q3_preservation_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #

def summarize_control_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    all_videos_exist = all(
        r.get("output_video_exists", False)
        for r in results
        if r.get("test") != "q3_preservation"
    )
    all_nonzero = all(
        r.get("output_video_bytes", 0) > 0
        for r in results
        if r.get("test") != "q3_preservation"
    )
    return {
        "passed_all_structural": all_videos_exist and all_nonzero,
        "evidence_level": "STRUCTURAL_CONTROL_CANDIDATE",
        "visual_review_required": True,
        "note": "File existence and non-zero size are structural checks only. CONTROL evidence requires visual review showing output differs meaningfully when operation or quadmask_npy is perturbed.",
        "tests": [r.get("test") for r in results],
    }


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge-run-dir", type=Path, required=True,
                   help="Completed bridge run dir containing vace_runtime_inputs in metadata.json")
    p.add_argument("--vace-repo", type=Path, default=DEFAULT_VACE_REPO)
    p.add_argument("--vace-ckpt-dir", type=Path, default=DEFAULT_VACE_CKPT)
    p.add_argument("--model-name", dest="model_name", default="vace-14B",
                   choices=["vace-14B", "vace-1.3B"])
    p.add_argument("--size", default="480p")
    p.add_argument("--base-seed", dest="base_seed", type=int, default=2025)
    p.add_argument("--sample-steps", dest="sample_steps", type=int, default=8)
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    p.add_argument("--skip-vace", action="store_true",
                   help="Dry run: skip actual VACE calls. Outputs will not exist.")
    return p.parse_args()


def load_baseline(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        # Try experiment_metadata.json (v03 runner convention)
        meta_path = run_dir / "experiment_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    vace = meta.get("vace_runtime_inputs") or {}
    if not vace:
        raise ValueError(f"No vace_runtime_inputs in {meta_path}")
    required = ("vace_conditioning_video", "quadmask_npy", "generation_mask", "operation", "vace_prompt", "frame_num")
    missing = [k for k in required if not vace.get(k)]
    if missing:
        raise ValueError(f"vace_runtime_inputs missing: {missing}")
    return {
        "vace_conditioning_video": str(vace["vace_conditioning_video"]),
        "quadmask_npy": str(vace["quadmask_npy"]),
        "generation_mask": str(vace["generation_mask"]),
        "operation": str(vace["operation"]),
        "vace_prompt": str(vace["vace_prompt"]),
        "frame_num": int(vace["frame_num"]),
    }


def main() -> int:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    run_dir = args.bridge_run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Bridge run dir not found: {run_dir}")

    baseline = load_baseline(run_dir)
    out_root = run_dir / "control_perturbation"
    out_root.mkdir(parents=True, exist_ok=True)

    # Baseline output video (used by q3_preservation)
    baseline_videos = list(run_dir.glob("edited_video*.mp4"))
    baseline_video = baseline_videos[0] if baseline_videos else run_dir / "edited_video.mp4"

    results = []
    results.append(run_test_operation_swap(baseline, out_root / "operation_swap", args))
    results.append(run_test_q0_suppressed(baseline, out_root / "q0_suppressed", args))
    results.append(run_test_q0_shifted(baseline, out_root / "q0_shifted", args))
    results.append(run_test_q3_preservation(baseline, baseline_video, out_root / "q3_preservation", args))

    summary = summarize_control_results(results)
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()
    summary["bridge_run_dir"] = str(run_dir)
    summary["baseline"] = baseline
    summary["test_results"] = results

    summary_path = out_root / "control_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"control_summary": str(summary_path), "passed_all_structural": summary["passed_all_structural"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

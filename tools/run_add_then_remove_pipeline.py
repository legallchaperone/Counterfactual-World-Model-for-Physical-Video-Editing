#!/usr/bin/env python3
"""Run a debug add-then-remove E2W pipeline.

This is an orchestration wrapper only:

source video -> existing add runner -> add edited video -> existing remove runner.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/data/cwx/conda/envs/edit2world-phase1-real/bin/python")
DEFAULT_RUN_ROOT = Path("/data/cwx/E2W/runs/add_then_remove")
CURRENT_VACE_INPUTS = {
    "vace_conditioning_video",
    "quadmask_npy",
    "generation_mask",
    "operation",
    "vace_prompt",
    "frame_num",
}


class PipelineError(RuntimeError):
    """Raised when an add-then-remove stage cannot produce valid artifacts."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--add-prompt", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--control-branch-checkpoint", type=Path, required=True)
    parser.add_argument("--vace-sample-steps", type=int, default=8)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    return parser.parse_args(argv)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PipelineError(f"missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env.update({k: v for k, v in env.items() if v is not None})
    proc = subprocess.run(cmd, cwd=str(cwd), env=merged_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    write_json(log_path, {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
    return proc


def extract_video_frames(video_path: Path, frame_dir: Path) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("*.jpg"):
        old.unlink()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PipelineError(f"failed to open video for frame extraction: {video_path}")
    frames: list[Path] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = frame_dir / f"{idx:05d}.jpg"
            if not cv2.imwrite(str(out), frame):
                raise PipelineError(f"failed to write extracted frame: {out}")
            frames.append(out)
            idx += 1
    finally:
        cap.release()
    if not frames:
        raise PipelineError(f"video has no readable frames: {video_path}")
    return frames


def build_remove_prompt(target_ref: str) -> str:
    target = target_ref.strip()
    if not target:
        raise PipelineError("add stage metadata missing target_ref")
    return f"Remove only the newly added {target}. Restore the scene as if this added object was not present."


def write_remove_eval_jsonl(path: Path, *, sample_id: str, first_frame: Path, target_ref: str) -> str:
    prompt = build_remove_prompt(target_ref)
    row = {
        "id": f"{sample_id}_remove_after_add",
        "video_id": f"{sample_id}_remove_after_add",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(first_frame)},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "metadata": {
            "operation": "remove",
            "chain_role": "remove_after_add",
            "remove_target_source": "add_stage.target_ref",
            "add_stage_target_ref": target_ref,
        },
    }
    write_jsonl(path, [row])
    return prompt


def path_from_add_metadata(add_meta: dict[str, Any], add_stage: Path) -> Path:
    edited = add_meta.get("edited_video")
    if isinstance(edited, dict) and edited.get("path"):
        return Path(edited["path"])
    return add_stage / "edited_video.mp4"


def require_existing_video(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise PipelineError(f"{label} missing or empty: {path}")


def require_current_runtime_inputs(value: Any, label: str) -> bool:
    if not isinstance(value, dict):
        raise PipelineError(f"{label} vace_runtime_inputs is not a dict")
    keys = set(value)
    if keys != CURRENT_VACE_INPUTS:
        raise PipelineError(f"{label} vace_runtime_inputs keys mismatch: {sorted(keys)}")
    return True


def assert_add_contract(add_meta: dict[str, Any], add_video: Path) -> dict[str, Any]:
    target_ref = str(add_meta.get("target_ref") or "").strip()
    if not target_ref:
        raise PipelineError("add stage metadata missing target_ref")
    require_existing_video(add_video, "add stage edited_video")
    runtime_inputs_ok = require_current_runtime_inputs(add_meta.get("vace_runtime_inputs"), "add stage")
    conditioning = add_meta.get("vace_conditioning_video") if isinstance(add_meta.get("vace_conditioning_video"), dict) else {}
    checks = {
        "target_ref_present": True,
        "edited_video_exists": True,
        "trained_control_branch_used": bool(add_meta.get("trained_control_branch_used")),
        "control_branch_checkpoint_loaded": bool(add_meta.get("control_branch_checkpoint_loaded")),
        "control_branch_step": add_meta.get("control_branch_step"),
        "control_branch_gate": add_meta.get("control_branch_gate"),
        "control_branch_installed_in_forward_vace": bool(add_meta.get("control_branch_installed_in_forward_vace")),
        "source_video_passed_to_vace": bool(add_meta.get("source_video_passed_to_vace")),
        "vace_runtime_inputs_current_contract": runtime_inputs_ok,
        "conditioning_future_frames_are_zero_filled": bool(conditioning.get("future_frames_are_zero_filled")),
        "conditioning_future_frames_source_video_used": bool(conditioning.get("future_frames_source_video_used")),
    }
    required_false = ["source_video_passed_to_vace", "conditioning_future_frames_source_video_used"]
    required_true = [
        "trained_control_branch_used",
        "control_branch_checkpoint_loaded",
        "control_branch_installed_in_forward_vace",
        "conditioning_future_frames_are_zero_filled",
    ]
    failed = [key for key in required_true if not checks[key]] + [key for key in required_false if checks[key]]
    if failed:
        raise PipelineError(f"add stage failed contract checks: {failed}")
    return checks


def first_ok_remove_result(summary: dict[str, Any]) -> dict[str, Any]:
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise PipelineError("remove summary must contain exactly one result")
    result = results[0]
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise PipelineError(f"remove stage did not finish ok: {result.get('error') if isinstance(result, dict) else result}")
    return result


def generation_mask_values_from_remove(result: dict[str, Any]) -> list[int]:
    metadata = result.get("generation_mask_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("generation_mask_values"), list):
        return [int(v) for v in metadata["generation_mask_values"]]
    vace_info = result.get("vace_info") if isinstance(result.get("vace_info"), dict) else {}
    runtime_meta = vace_info.get("vace_runtime_input_metadata") if isinstance(vace_info.get("vace_runtime_input_metadata"), dict) else {}
    generation = runtime_meta.get("generation_mask") if isinstance(runtime_meta.get("generation_mask"), dict) else {}
    if isinstance(generation.get("generation_mask_values"), list):
        return [int(v) for v in generation["generation_mask_values"]]
    return []


def remove_conditioning_metadata(result: dict[str, Any]) -> dict[str, Any]:
    vace_info = result.get("vace_info") if isinstance(result.get("vace_info"), dict) else {}
    runtime_meta = vace_info.get("vace_runtime_input_metadata") if isinstance(vace_info.get("vace_runtime_input_metadata"), dict) else {}
    conditioning = runtime_meta.get("vace_conditioning_video")
    return conditioning if isinstance(conditioning, dict) else {}


def assert_remove_contract(result: dict[str, Any], final_video: Path) -> dict[str, Any]:
    require_existing_video(final_video, "remove stage edited video")
    runtime_inputs_ok = require_current_runtime_inputs(result.get("vace_runtime_inputs"), "remove stage")
    vace_info = result.get("vace_info") if isinstance(result.get("vace_info"), dict) else {}
    conditioning = remove_conditioning_metadata(result)
    generation_values = generation_mask_values_from_remove(result)
    checks = {
        "edited_video_exists": True,
        "trained_control_branch_used": bool(vace_info.get("trained_control_branch_used")),
        "control_branch_checkpoint_loaded": bool(vace_info.get("control_branch_checkpoint_loaded")),
        "control_branch_step": vace_info.get("control_branch_step"),
        "control_branch_gate": vace_info.get("control_branch_gate"),
        "control_branch_installed_in_forward_vace": bool(vace_info.get("control_branch_installed_in_forward_vace")),
        "source_video_passed_to_vace": bool(result.get("source_video_passed_to_vace")),
        "vace_runtime_inputs_current_contract": runtime_inputs_ok,
        "conditioning_future_frames_are_zero_filled": bool(conditioning.get("future_frames_are_zero_filled")),
        "conditioning_future_frames_source_video_used": bool(conditioning.get("future_frames_source_video_used")),
        "generation_mask_values": generation_values,
        "generation_mask_full_domain": generation_values == [255],
    }
    required_false = ["source_video_passed_to_vace", "conditioning_future_frames_source_video_used"]
    required_true = [
        "trained_control_branch_used",
        "control_branch_checkpoint_loaded",
        "control_branch_installed_in_forward_vace",
        "conditioning_future_frames_are_zero_filled",
        "generation_mask_full_domain",
    ]
    failed = [key for key in required_true if not checks[key]] + [key for key in required_false if checks[key]]
    if failed:
        raise PipelineError(f"remove stage failed contract checks: {failed}")
    return checks


def _read_all_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PipelineError(f"failed to open video for comparison grid: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 6.0
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise PipelineError(f"video has no readable frames for comparison grid: {video_path}")
    return frames, float(fps)


def make_comparison_grid(original: Path, after_add: Path, after_remove: Path, out_path: Path) -> None:
    videos = [original, after_add, after_remove]
    loaded = [_read_all_frames(path) for path in videos]
    frame_sets = [item[0] for item in loaded]
    fps = loaded[0][1] or 6.0
    min_h = min(frames[0].shape[0] for frames in frame_sets)
    min_w = min(frames[0].shape[1] for frames in frame_sets)
    labels = ["original", "after add", "after add-remove"]
    frame_count = max(len(frames) for frames in frame_sets)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (min_w * 3, min_h))
    if not writer.isOpened():
        raise PipelineError(f"failed to open comparison grid writer: {out_path}")
    try:
        for idx in range(frame_count):
            cells: list[np.ndarray] = []
            for frames, label in zip(frame_sets, labels):
                frame = frames[min(idx, len(frames) - 1)]
                cell = cv2.resize(frame, (min_w, min_h), interpolation=cv2.INTER_AREA)
                cv2.rectangle(cell, (0, 0), (min_w, 34), (0, 0, 0), -1)
                cv2.putText(cell, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                cells.append(cell)
            writer.write(np.concatenate(cells, axis=1))
    finally:
        writer.release()
    require_existing_video(out_path, "comparison_grid")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    source_video = args.source_video.resolve()
    checkpoint = args.control_branch_checkpoint.resolve()
    if not source_video.exists():
        raise PipelineError(f"source video not found: {source_video}")
    if not checkpoint.exists():
        raise PipelineError(f"control branch checkpoint not found: {checkpoint}")

    run_dir = (args.run_dir or (DEFAULT_RUN_ROOT / args.sample_id)).resolve()
    add_stage = run_dir / "add_stage"
    remove_stage = run_dir / "remove_stage"
    remove_input_frames = run_dir / "remove_input_frames"
    remove_eval = run_dir / "remove_eval.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        run_dir / "inputs.json",
        {
            "source_video": str(source_video),
            "add_prompt": args.add_prompt,
            "sample_id": args.sample_id,
            "control_branch_checkpoint": str(checkpoint),
            "vace_sample_steps": args.vace_sample_steps,
            "cuda_visible_devices": args.cuda_visible_devices,
        },
    )

    add_cmd = [
        str(args.python.resolve()),
        str((TOOLS / "run_add_pipeline_interface.py").resolve()),
        "--source-video",
        str(source_video),
        "--user-prompt",
        args.add_prompt,
        "--sample-id",
        f"{args.sample_id}_add",
        "--run-dir",
        str(add_stage),
        "--vace-sample-steps",
        str(args.vace_sample_steps),
        "--control-branch-checkpoint",
        str(checkpoint),
    ]
    if args.cuda_visible_devices:
        add_cmd.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    proc = run_cmd(
        add_cmd,
        cwd=ROOT,
        env={"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices},
        log_path=run_dir / "add_stage_command.json",
    )
    if proc.returncode != 0:
        raise PipelineError(f"add stage failed rc={proc.returncode}; see {run_dir / 'add_stage_command.json'}")

    add_meta = read_json(add_stage / "metadata.json")
    target_ref = str(add_meta.get("target_ref") or "").strip()
    add_video = path_from_add_metadata(add_meta, add_stage).resolve()
    add_checks = assert_add_contract(add_meta, add_video)

    frames = extract_video_frames(add_video, remove_input_frames)
    remove_prompt = write_remove_eval_jsonl(remove_eval, sample_id=args.sample_id, first_frame=frames[0], target_ref=target_ref)

    remove_cmd = [
        str(args.python.resolve()),
        str((TOOLS / "run_counterfactual_planner_pipeline.py").resolve()),
        "--eval-jsonl",
        str(remove_eval),
        "--output-dir",
        str(remove_stage),
        "--sample-count",
        "1",
        "--seed",
        "0",
        "--vace-sample-steps",
        str(args.vace_sample_steps),
        "--control-branch-checkpoint",
        str(checkpoint),
    ]
    proc = run_cmd(
        remove_cmd,
        cwd=ROOT,
        env={"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices},
        log_path=run_dir / "remove_stage_command.json",
    )
    if proc.returncode != 0:
        raise PipelineError(f"remove stage failed rc={proc.returncode}; see {run_dir / 'remove_stage_command.json'}")

    remove_summary = read_json(remove_stage / "summary.json")
    remove_result = first_ok_remove_result(remove_summary)
    remove_video = Path(str(remove_result.get("vace_output_path") or "")).resolve()
    remove_checks = assert_remove_contract(remove_result, remove_video)

    final_video = run_dir / "final_add_then_remove_video.mp4"
    shutil.copy2(remove_video, final_video)
    comparison_grid = run_dir / "comparison_grid.mp4"
    make_comparison_grid(source_video, add_video, final_video, comparison_grid)

    metadata = {
        "sample_id": args.sample_id,
        "evidence_level": "VISUAL_CANDIDATE_ONLY",
        "control_claimed": False,
        "research_claimed": False,
        "operation_chain": ["add", "remove"],
        "remove_target_source": "add_stage.target_ref",
        "source_video": str(source_video),
        "add_prompt": args.add_prompt,
        "remove_prompt": remove_prompt,
        "add_stage_edited_video": str(add_video),
        "final_edited_video": str(final_video),
        "comparison_grid": str(comparison_grid),
        "add_target_ref": target_ref,
        "add_vace_prompt": add_meta.get("vace_prompt"),
        "remove_vace_prompt": remove_result.get("vace_prompt"),
        "contract_checks": {
            "add": add_checks,
            "remove": remove_checks,
        },
        "add_stage_metadata": str(add_stage / "metadata.json"),
        "remove_stage_summary": str(remove_stage / "summary.json"),
        "remove_eval_jsonl": str(remove_eval),
    }
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = run_pipeline(args)
    print(
        json.dumps(
            {
                "run_dir": str((args.run_dir or (DEFAULT_RUN_ROOT / args.sample_id)).resolve()),
                "metadata": str((args.run_dir or (DEFAULT_RUN_ROOT / args.sample_id)).resolve() / "metadata.json"),
                "comparison_grid": metadata["comparison_grid"],
                "final_edited_video": metadata["final_edited_video"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

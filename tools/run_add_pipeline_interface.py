#!/usr/bin/env python3
"""Run the E2W add pipeline at INTERFACE evidence level.

This runner orchestrates the real pipeline from original video + user prompt. It
must not author or repair planner output. If planner inference fails contract
validation after the configured retries, the runner fails loudly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (TOOLS, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_add_quadmask_from_edited_first_frame import (  # noqa: E402
    build_quadmask_from_primary,
    load_rgb,
    sam2_primary_from_edited_frame,
    write_gray_video,
    write_quadmask_preview,
    write_rgb_video,
)
from e2w_v0_common import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_PLANNER,
    extract_first_frame,
    serialize_first_frame_prompt,
    video_meta,
    write_json,
    write_text,
)
from run_first_frame_edit import DEFAULT_QWEN_IMAGE_EDIT  # noqa: E402

DEFAULT_SOURCE_VIDEO = Path("/data/cwx/E2W/data/phase1a_pexels_self_insert_v1/02_background_clean/videos_mp4/bg_000001.mp4")
DEFAULT_USER_PROMPT = "Add a red mug on the table near the center of the image."
DEFAULT_RUN_ROOT = Path("/data/cwx/E2W/runs")
DEFAULT_VACE_REPO = Path("/data/cwx/Edit2World-unified/external/VACE")
DEFAULT_VACE_CKPT = Path("/data/cwx/Edit2World-unified/checkpoints/Wan2.1-VACE-14B")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-video", type=Path, default=DEFAULT_SOURCE_VIDEO)
    p.add_argument("--user-prompt", default=DEFAULT_USER_PROMPT)
    p.add_argument("--sample-id", default="add_bg_000001")
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--frame-num", type=int, default=21)
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--planner-attempts", type=int, default=3)
    p.add_argument("--planner-adapter", type=Path, default=DEFAULT_PLANNER)
    p.add_argument("--planner-base-model", type=Path, default=DEFAULT_BASE_MODEL)
    p.add_argument("--planner-no-adapter", action="store_true")
    p.add_argument("--planner-max-new-tokens", type=int, default=1536)
    p.add_argument("--qwen-checkpoint", type=Path, default=DEFAULT_QWEN_IMAGE_EDIT)
    p.add_argument("--qwen-steps", type=int, default=20)
    p.add_argument("--qwen-true-cfg-scale", type=float, default=4.0)
    p.add_argument("--qwen-seed", type=int, default=2025)
    p.add_argument("--qwen-cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vace-repo", type=Path, default=DEFAULT_VACE_REPO)
    p.add_argument("--vace-ckpt-dir", type=Path, default=DEFAULT_VACE_CKPT)
    p.add_argument("--vace-model-name", default="vace-14B", choices=["vace-14B", "vace-1.3B"])
    p.add_argument("--vace-size", default="480p")
    p.add_argument("--vace-sample-steps", type=int, default=8)
    p.add_argument("--vace-base-seed", type=int, default=2025)
    p.add_argument("--control-branch-checkpoint", type=Path)
    p.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    p.add_argument("--skip-vace", action="store_true", help="debug only: do not use for acceptance")
    return p.parse_args()


def ensure_frame_num(frame_num: int) -> None:
    if frame_num <= 0 or frame_num % 4 != 1:
        raise ValueError(f"frame_num must be Wan-compatible 4n+1, got {frame_num}")


def build_add_planner_user_prompt(sample_id: str, user_prompt: str, *, attempt: int) -> str:
    rules = [
        "Return only one complete top-level JSON object. Do not include markdown, prose, comments, or fallback text.",
        "The operation must be add.",
        "The planner/model output must include vace_prompt as a final field produced by this inference attempt.",
        "Do not prefill, copy, or substitute a teacher/manual vace_prompt.",
        "vace_prompt should describe the edited scene after the addition and may mention the object to add.",
        "vace_prompt must not contain removal-residue language such as absent, missing, gone, no longer visible, removed, erased, deleted, without the added object, or where the object was.",
        "Provide target_ref as a concise visual reference to the object being added.",
        "Provide first-frame grounding for the insertion region as primary_point_norm1000 [x, y]; include primary_bbox_norm1000 [x1, y1, x2, y2] if visible or inferable.",
        "Use norm1000 coordinates with [0, 0] at the top-left and [1000, 1000] at the bottom-right.",
        "Provide affected_regions as short text labels for local contact, shadow, reflection, or support regions that may change.",
    ]
    if attempt > 1:
        rules.extend(
            [
                f"This is retry attempt {attempt} for an add-operation interface run.",
                "Fix only contract failures from the previous attempt by returning a fresh complete planner JSON.",
                "Keep vace_prompt positive for add; do not describe removal, disappearance, absence, or cleanup.",
            ]
        )
    schema = {
        "sample_id": sample_id,
        "operation": "add",
        "target_ref": "short visual reference to the object being added",
        "vace_prompt": "positive edited-scene prompt produced by this planner/model inference",
        "primary_point_norm1000": [500, 500],
        "primary_bbox_norm1000": [450, 450, 550, 550],
        "affected_regions": ["local contact shadow or nearby surface"],
    }
    return (
        "You are the Edit2World add-operation planner. Given the video and user request, "
        "return a current-spec add planner JSON for the first-frame edit, grounding bridge, and VACE prompt path.\n\n"
        f"Required JSON shape:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Rules:\n- "
        + "\n- ".join(rules)
        + "\n\n"
        f"Set sample_id exactly to {sample_id}.\n"
        f"User request: {user_prompt.strip()}\n"
    )


def write_split_jsonl(path: Path, *, sample_id: str, video: Path, user_prompt: str, attempt: int) -> None:
    content = build_add_planner_user_prompt(sample_id, user_prompt, attempt=attempt)
    row = {"id": sample_id, "video": str(video), "messages": [{"role": "user", "content": content}], "metadata": {"operation": "add"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update({k: v for k, v in env.items() if v is not None})
    proc = subprocess.run(cmd, cwd=str(cwd), env=merged, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps({"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}, indent=2) + "\n",
            encoding="utf-8",
        )
    return proc


def run_planner(args: argparse.Namespace, run_dir: Path) -> tuple[dict[str, Any], list[str], Path]:
    errors: list[str] = []
    last_entry: dict[str, Any] | None = None
    for attempt in range(1, args.planner_attempts + 1):
        attempt_dir = run_dir / "planner" / f"attempt_{attempt}"
        split_path = attempt_dir / "split.jsonl"
        write_split_jsonl(split_path, sample_id=args.sample_id, video=args.source_video, user_prompt=args.user_prompt, attempt=attempt)
        cmd = [
            sys.executable,
            str(TOOLS / "eval_vlm_planner.py"),
            "--run-dir",
            str(attempt_dir),
            "--split-jsonl",
            str(split_path),
            "--operation",
            "add",
            "--sample-id",
            args.sample_id,
            "--max-new-tokens",
            str(args.planner_max_new_tokens),
            "--allow-failures",
            "--force",
            "--adapter",
            str(args.planner_adapter),
            "--base-model",
            str(args.planner_base_model),
        ]
        if args.planner_no_adapter:
            cmd.append("--no-adapter")
        proc = run_cmd(cmd, cwd=ROOT, env={"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices}, log_path=attempt_dir / "planner_command.json")
        manifest = attempt_dir / "manifest.jsonl"
        if manifest.exists():
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            last_entry = rows[0] if rows else None
            if last_entry:
                metrics = last_entry.get("metrics") or {}
                planner_usable_for_add_interface = bool(
                    metrics.get("schema_valid")
                    and metrics.get("operation_accuracy")
                    and metrics.get("vace_prompt_contract_ok")
                    and metrics.get("primary_point_valid")
                )
                # The archived executable-planner validator marks missing bbox as
                # non-executable, but the add interface mask path can prompt SAM2 with
                # a planner/model point on the edited_first_frame. Accept that real
                # model output without manual repair.
                if last_entry.get("status") == "ok" or planner_usable_for_add_interface:
                    last_entry.setdefault("metrics", {})["accepted_for_add_interface"] = True
                    last_entry["status"] = "ok" if last_entry.get("status") == "ok" else "accepted_point_only_for_add_interface"
                    return last_entry, errors, attempt_dir
                errors.append(json.dumps({"attempt": attempt, "status": last_entry.get("status"), "metrics": metrics}, ensure_ascii=False, sort_keys=True))
            else:
                errors.append(f"attempt {attempt}: manifest empty")
        else:
            errors.append(f"attempt {attempt}: planner command failed rc={proc.returncode}: {proc.stderr[-1000:]}")
    raise RuntimeError("planner/model inference failed after retries: " + " | ".join(errors))


def build_conditioning_video(edited_first_frame: Path, out_path: Path, *, frame_num: int, fps: float) -> dict[str, Any]:
    image = load_rgb(edited_first_frame)
    # Current VACE conditioning carries only the edited first frame; future frames are blank placeholders, not source-video frames.
    frames = [np.zeros_like(image) for _ in range(frame_num)]
    frames[0] = image
    write_rgb_video(out_path, frames, fps=fps)
    return {
        "frame_0_is_edited_first_frame": True,
        "future_frames_are_zero_filled": True,
        "future_frames_source_video_used": False,
        "conditioning_strategy": "edited_first_frame_plus_zero_filled_future_frames",
        "frame_count": frame_num,
    }


def load_qwen_edit_pipe_for_interface(args: argparse.Namespace) -> Any:
    import torch
    from diffusers import QwenImageEditPipeline

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pipe = QwenImageEditPipeline.from_pretrained(str(args.qwen_checkpoint), torch_dtype=dtype)
    if torch.cuda.is_available():
        if args.qwen_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    return pipe


def qwen_edit_for_interface(pipe: Any, image: Image.Image, prompt: str, args: argparse.Namespace) -> Image.Image:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(args.qwen_seed)
    result = pipe(
        image.convert("RGB"),
        prompt=prompt,
        negative_prompt=" ",
        num_inference_steps=args.qwen_steps,
        generator=generator,
        true_cfg_scale=args.qwen_true_cfg_scale,
        num_images_per_prompt=1,
    ).images[0]
    return result


def build_generation_mask(out_npy: Path, out_mp4: Path, *, frame_num: int, height: int, width: int, fps: float) -> dict[str, Any]:
    arr = np.full((frame_num, height, width), 255, dtype=np.uint8)
    np.save(out_npy, arr)
    write_gray_video(out_mp4, arr, fps=fps)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "values": sorted(int(x) for x in np.unique(arr)),
        "generation_mask_is_full_domain": True,
        "generation_mask_encodes_quadmask_semantics": False,
    }


def planner_target_ref_from_raw(raw_pred_path: Path) -> str | None:
    value = json.loads(raw_pred_path.read_text(encoding="utf-8")).get("target_ref")
    return str(value).strip() if str(value or "").strip() else None


def main() -> int:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    ensure_frame_num(args.frame_num)
    if not args.source_video.exists():
        raise FileNotFoundError(args.source_video)
    run_dir = args.run_dir or (args.run_root / f"add_pipeline_interface_{args.sample_id}_{utc_stamp()}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_text(run_dir / "user_prompt.txt", args.user_prompt)

    source_meta = video_meta(args.source_video)
    input_first_frame = run_dir / "input_first_frame.png"
    extract_first_frame(args.source_video, input_first_frame)

    planner_entry, planner_errors, planner_attempt_dir = run_planner(args, run_dir)
    planner_paths = planner_entry["paths"]
    edit_plan_path = Path(planner_paths["edit_plan"])
    quadmask_spec_path = Path(planner_paths["quadmask_spec"])
    vace_prompt_path = Path(planner_paths["vace_prompt"])
    edit_plan = json.loads(edit_plan_path.read_text(encoding="utf-8"))
    vace_prompt = vace_prompt_path.read_text(encoding="utf-8")
    shutil.copy2(edit_plan_path, run_dir / "planner_output.json")
    shutil.copy2(vace_prompt_path, run_dir / "vace_prompt.txt")

    qwen_pipe = load_qwen_edit_pipe_for_interface(args)
    qwen_prompt = serialize_first_frame_prompt(edit_plan)
    write_text(run_dir / "qwen_edit_prompt.txt", qwen_prompt)
    input_image = Image.open(input_first_frame).convert("RGB")
    input_size = input_image.size
    edited = qwen_edit_for_interface(qwen_pipe, input_image, qwen_prompt, args)
    edited_first_frame = run_dir / "edited_first_frame.png"
    edited = edited.resize(input_size, Image.Resampling.LANCZOS)
    edited.save(edited_first_frame)

    vace_conditioning = run_dir / "vace_conditioning_video.mp4"
    conditioning_meta = build_conditioning_video(edited_first_frame, vace_conditioning, frame_num=args.frame_num, fps=args.fps)

    mask_dir = run_dir / "add_quadmask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    primary, sam2_info = sam2_primary_from_edited_frame(
        edited_first_frame=edited_first_frame,
        quadmask_spec=json.loads(quadmask_spec_path.read_text(encoding="utf-8")),
        out_dir=mask_dir,
        frame_num=args.frame_num,
        fps=args.fps,
    )
    quadmask, quad_meta = build_quadmask_from_primary(
        primary,
        frame_num=args.frame_num,
        original_first_frame=load_rgb(input_first_frame),
        edited_first_frame=load_rgb(edited_first_frame),
    )
    quadmask_npy = run_dir / "quadmask.npy"
    np.save(quadmask_npy, quadmask)
    np.save(run_dir / "primary_mask.npy", primary.astype(np.uint8))
    Image.fromarray(primary.astype(np.uint8) * 255).save(run_dir / "primary_mask.png")
    write_quadmask_preview(run_dir / "quadmask_preview.mp4", quadmask, fps=args.fps)

    h, w = quadmask.shape[1:]
    gen_npy = run_dir / "generation_mask.npy"
    gen_mp4 = run_dir / "generation_mask.mp4"
    gen_meta = build_generation_mask(gen_npy, gen_mp4, frame_num=args.frame_num, height=h, width=w, fps=args.fps)

    edited_video = run_dir / "edited_video.mp4"
    vace_cmd = [
        sys.executable,
        str(TOOLS / "run_wan_vace_quad_i2v.py"),
        "--vace_repo",
        str(args.vace_repo),
        "--ckpt_dir",
        str(args.vace_ckpt_dir),
        "--model_name",
        args.vace_model_name,
        "--size",
        args.vace_size,
        "--src_video",
        str(vace_conditioning),
        "--generation_mask",
        str(gen_mp4),
        "--quadmask_npy",
        str(quadmask_npy),
        "--operation",
        "add",
        "--prompt",
        vace_prompt,
        "--frame_num",
        str(args.frame_num),
        "--save_dir",
        str(run_dir / "vace_backend"),
        "--save_file",
        str(edited_video),
        "--base_seed",
        str(args.vace_base_seed),
        "--sample_steps",
        str(args.vace_sample_steps),
        "--offload_model",
    ]
    if args.control_branch_checkpoint:
        vace_cmd.extend(["--control_branch_checkpoint", str(args.control_branch_checkpoint)])
    write_json(run_dir / "run_command.json", {"vace_cmd": vace_cmd, "cwd": str(ROOT)})
    if args.skip_vace:
        raise RuntimeError("--skip-vace is debug-only and cannot satisfy acceptance criteria")
    proc = run_cmd(vace_cmd, cwd=ROOT, env={"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices}, log_path=run_dir / "vace_command_result.json")
    if proc.returncode != 0:
        raise RuntimeError(f"VACE command failed rc={proc.returncode}; see {run_dir / 'vace_command_result.json'}")
    context_info_path = run_dir / "vace_backend" / "e2w_quad_context.json"
    context_info = json.loads(context_info_path.read_text(encoding="utf-8")) if context_info_path.exists() else {}

    metadata = {
        "evidence_level": "INTERFACE",
        "visual_quality_evaluated": False,
        "operation": "add",
        "source_video_for_upstream_context": str(args.source_video),
        "source_video_passed_to_vace": False,
        "user_prompt": args.user_prompt,
        "planner": {
            "used": True,
            "attempt_dir": str(planner_attempt_dir),
            "planner_attempt_count": len(planner_errors) + 1,
            "planner_invalid_attempt_errors": planner_errors,
            "accepted_for_add_interface": bool((planner_entry.get("metrics") or {}).get("accepted_for_add_interface")),
            "accepted_point_only_for_add_interface": planner_entry.get("status") == "accepted_point_only_for_add_interface",
            "planner_output_manually_modified": False,
            "vace_prompt_source": "planner_model",
            "vace_prompt_passed_through_unchanged": True,
            "manual_or_teacher_vace_prompt_used": False,
            "learned_planner_add_quality_claimed": False,
            "output_path": str(run_dir / "planner_output.json"),
        },
        "target_ref": planner_target_ref_from_raw(Path(planner_paths["raw_pred"])),
        "vace_prompt": vace_prompt,
        "frame_num": args.frame_num,
        "qwen_edit": {"used": True, "edited_first_frame": str(edited_first_frame), "prompt_path": str(run_dir / "qwen_edit_prompt.txt")},
        "vace_conditioning_video": conditioning_meta,
        "sam2": sam2_info,
        "quadmask": quad_meta,
        "generation_mask": gen_meta,
        "control_branch": context_info,
        "control_branch_checkpoint_loaded": bool(context_info.get("control_branch_checkpoint_loaded")),
        "trained_control_branch_used": bool(context_info.get("trained_control_branch_used")),
        "control_branch_step": context_info.get("control_branch_step"),
        "control_branch_gate": context_info.get("control_branch_gate"),
        "control_branch_installed_in_forward_vace": bool(context_info.get("control_branch_installed_in_forward_vace")),
        "vace_runtime_inputs": {
            "vace_conditioning_video": str(vace_conditioning),
            "quadmask_npy": str(quadmask_npy),
            "generation_mask": str(gen_mp4),
            "operation": "add",
            "vace_prompt": vace_prompt,
            "frame_num": args.frame_num,
        },
        "legacy_backend_arg_adapter_used": True,
        "edited_video": {"path": str(edited_video), "exists": edited_video.exists(), "size_bytes": edited_video.stat().st_size if edited_video.exists() else 0},
        "success_criteria": {
            "edited_first_frame_exists": edited_first_frame.exists(),
            "quadmask_shape_ok": list(quadmask.shape) == [args.frame_num, h, w],
            "quadmask_values_ok": set(np.unique(quadmask).astype(int).tolist()).issubset({0, 63, 127, 255}),
            "generation_mask_full_domain": gen_meta["values"] == [255],
            "edited_video_exists": edited_video.exists() and edited_video.stat().st_size > 0,
        },
        "source_video_metadata": source_meta,
    }
    write_json(run_dir / "metadata.json", metadata)
    print(json.dumps({"run_dir": str(run_dir), "metadata": str(run_dir / "metadata.json"), "edited_video": str(edited_video)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

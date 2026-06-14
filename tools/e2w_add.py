#!/usr/bin/env python3
"""E2W add interface: Counterfactual/add planner -> edit first frame -> SAM2 on the
edited frame -> VACE(add).

Unified add interface on tools/e2w_pipeline_core.py. Unlike remove, add grounds the
EDITED first frame (where the new object now exists): inline add-planner inference
produces target_ref + object-naming vace_prompt + primary_point (norm1000); the first
frame is edited to add the object; SAM2 is seeded by primary_point on the edited frame;
then quadmask / conditioning / generation mask / VACE(add).

This interface does NOT use the legacy v6 eval_vlm_planner route. All model checkpoints
are injectable; the add planner adapter must be supplied explicitly (see --planner-adapter).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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

import e2w_pipeline_core as core  # noqa: E402
from build_add_quadmask_from_edited_first_frame import (  # noqa: E402
    DEFAULT_SAM2_CFG,
    DEFAULT_SAM2_CKPT,
    DEFAULT_SAM2_REPO,
    build_quadmask_from_primary,
    load_rgb,
    sam2_primary_from_edited_frame,
    write_quadmask_preview,
)
from e2w_v0_common import (  # noqa: E402
    build_add_planner_user_prompt,
    extract_first_frame,
    validate_add_planner_output,
    video_meta,
    write_json,
    write_text,
)

DEFAULT_BASE_MODEL = Path("/data/cwx/Edit2World-unified/checkpoints/Qwen2.5-VL-7B-Instruct")
DEFAULT_QWEN_IMAGE_EDIT = Path("/data/cwx/Edit2World-unified/checkpoints/Qwen-Image-Edit")
DEFAULT_VACE_REPO = Path("/data/cwx/Edit2World-unified/external/VACE")
DEFAULT_VACE_CKPT = Path("/data/cwx/Edit2World-unified/checkpoints/Wan2.1-VACE-14B")
DEFAULT_PYTHON = Path("/data/cwx/conda/envs/edit2world-phase1-real/bin/python")
DEFAULT_RUN_ROOT = Path("/data/cwx/E2W/runs")
DEFAULT_SOURCE_VIDEO = Path("/data/cwx/E2W/data/phase1a_pexels_self_insert_v1/02_background_clean/videos_mp4/bg_000001.mp4")
DEFAULT_USER_PROMPT = "Add a red mug on the table near the center of the image."


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-video", type=Path, default=DEFAULT_SOURCE_VIDEO)
    p.add_argument("--user-prompt", default=DEFAULT_USER_PROMPT)
    p.add_argument("--sample-id", default="add_bg_000001")
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--frame-num", type=int, default=21)
    p.add_argument("--fps", type=float, default=12.0)
    # Planner (inline add-planner inference). Adapter must be an add-capable checkpoint.
    p.add_argument("--planner-base-model", type=Path, default=DEFAULT_BASE_MODEL)
    p.add_argument("--planner-adapter", type=Path, required=True, help="add-planner LoRA adapter (no archived default)")
    p.add_argument("--planner-attempts", type=int, default=3)
    p.add_argument("--planner-max-new-tokens", type=int, default=768)
    p.add_argument("--planner-temperature", type=float, default=0.0)
    # First-frame edit (Qwen-Image-Edit).
    p.add_argument("--qwen-checkpoint", type=Path, default=DEFAULT_QWEN_IMAGE_EDIT)
    p.add_argument("--qwen-steps", type=int, default=20)
    p.add_argument("--qwen-true-cfg-scale", type=float, default=4.0)
    p.add_argument("--qwen-seed", type=int, default=2025)
    # SAM2 (add grounding on the edited frame).
    p.add_argument("--sam2-repo", type=Path, default=DEFAULT_SAM2_REPO)
    p.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CKPT)
    p.add_argument("--sam2-config", default=DEFAULT_SAM2_CFG)
    # VACE backend.
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--vace-repo", type=Path, default=DEFAULT_VACE_REPO)
    p.add_argument("--vace-ckpt", type=Path, default=DEFAULT_VACE_CKPT)
    p.add_argument("--vace-model-name", default="vace-14B", choices=["vace-14B", "vace-1.3B"])
    p.add_argument("--vace-size", default="480p")
    p.add_argument("--vace-sample-steps", type=int, default=8)
    p.add_argument("--vace-base-seed", type=int, default=2025)
    p.add_argument("--control-branch-checkpoint", type=Path)
    p.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    p.add_argument("--skip-vace", action="store_true", help="debug only: not for acceptance")
    return p.parse_args()


def ensure_frame_num(frame_num: int) -> None:
    if frame_num <= 0 or frame_num % 4 != 1:
        raise ValueError(f"frame_num must be Wan-compatible 4n+1, got {frame_num}")


def add_quadmask_spec_from_planner(parsed: dict[str, Any]) -> dict[str, Any]:
    """Map validated add planner output (primary_point/bbox in norm1000) into the
    quadmask_spec keyframe shape that primary_grounding_from_spec / SAM2 expect."""
    keyframe: dict[str, Any] = {
        "frame_index": 0,
        "positive_points_norm1000": [list(parsed["primary_point"])],
    }
    if parsed.get("primary_bbox") is not None:
        keyframe["bbox_xyxy_norm1000"] = list(parsed["primary_bbox"])
    return {
        "schema_version": "e2w.quadmask_spec.v1",
        "operation": "add",
        "primary": {"keyframes": [keyframe]},
    }


def run_add_planner(
    image_path: Path,
    user_prompt: str,
    sample_id: str,
    *,
    base_model: Path,
    adapter: Path,
    attempts: int,
    max_new_tokens: int,
    temperature: float,
    run_dir: Path,
) -> dict[str, Any]:
    """Inline add-planner inference + add-contract validation, with retries."""
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        content = [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": build_add_planner_user_prompt(user_prompt, sample_id=sample_id, attempt=attempt)},
        ]
        row = {"video_id": sample_id, "messages": [{"role": "user", "content": content}]}
        planned = core.run_planner(base_model, adapter, [row], max_new_tokens=max_new_tokens, temperature=temperature)
        item = planned[0]
        attempt_dir = run_dir / "planner" / f"attempt_{attempt}"
        write_json(attempt_dir / "planner_output_raw.json", {"raw_output": item["raw_output"], "parsed": item["parsed"]})
        parsed = item["parsed"]
        if parsed is None:
            errors.append(f"attempt {attempt}: json parse failed: {item.get('parse_error')}")
            continue
        ok, err = validate_add_planner_output(parsed)
        if ok:
            return parsed
        errors.append(f"attempt {attempt}: {err}")
    raise RuntimeError("add planner failed contract validation after retries: " + " | ".join(errors))


def main() -> int:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    ensure_frame_num(args.frame_num)
    if not args.source_video.exists():
        raise FileNotFoundError(args.source_video)
    run_dir = args.run_dir or (args.run_root / f"e2w_add_{args.sample_id}_{utc_stamp()}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_text(run_dir / "user_prompt.txt", args.user_prompt)

    source_meta = video_meta(args.source_video)
    input_first_frame = run_dir / "input_first_frame.png"
    extract_first_frame(args.source_video, input_first_frame)

    parsed = run_add_planner(
        input_first_frame,
        args.user_prompt,
        args.sample_id,
        base_model=args.planner_base_model,
        adapter=args.planner_adapter,
        attempts=args.planner_attempts,
        max_new_tokens=args.planner_max_new_tokens,
        temperature=args.planner_temperature,
        run_dir=run_dir,
    )
    write_json(run_dir / "planner_output.json", parsed)
    target_ref = str(parsed["target_ref"]).strip()
    vace_prompt = str(parsed["vace_prompt"]).strip()
    write_text(run_dir / "vace_prompt.txt", vace_prompt)

    # First-frame edit: add the object, using the user's add request as the edit instruction.
    edited_first_frame = run_dir / "edited_first_frame.png"
    edit_info = core.edit_first_frame(
        input_first_frame,
        args.user_prompt,
        edited_first_frame,
        qwen_checkpoint=args.qwen_checkpoint,
        seed=args.qwen_seed,
        steps=args.qwen_steps,
        true_cfg_scale=args.qwen_true_cfg_scale,
    )

    # Add grounding: SAM2 on the EDITED first frame seeded by the planner point (tested path).
    mask_dir = run_dir / "add_quadmask"
    quadmask_spec = add_quadmask_spec_from_planner(parsed)
    write_json(mask_dir / "quadmask_spec.json", quadmask_spec)
    primary, sam2_info = sam2_primary_from_edited_frame(
        edited_first_frame=edited_first_frame,
        quadmask_spec=quadmask_spec,
        out_dir=mask_dir,
        frame_num=args.frame_num,
        fps=args.fps,
        sam2_repo=args.sam2_repo,
        sam2_ckpt=args.sam2_checkpoint,
        sam2_cfg=args.sam2_config,
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

    height, width = quadmask.shape[1:]
    vace_conditioning = run_dir / "vace_conditioning_video.mp4"
    conditioning_meta = core.build_conditioning_video(
        edited_first_frame, vace_conditioning, frame_count=args.frame_num, width=width, height=height, fps=args.fps
    )
    generation_mask = core.build_generation_mask(args.frame_num, height, width)
    gen_npy = run_dir / "generation_mask.npy"
    gen_mp4 = run_dir / "generation_mask.mp4"
    np.save(gen_npy, generation_mask)
    core.write_gray_video(generation_mask, gen_mp4, args.fps)

    if args.skip_vace:
        raise RuntimeError("--skip-vace is debug-only and cannot satisfy acceptance criteria")
    edited_video = run_dir / "edited_video.mp4"
    vace_output, vace_info = core.run_vace(
        "add",
        args.sample_id,
        python=args.python,
        vace_repo=args.vace_repo,
        vace_ckpt=args.vace_ckpt,
        model_name=args.vace_model_name,
        size=args.vace_size,
        conditioning_video=vace_conditioning,
        generation_mask_video=gen_mp4,
        quadmask_npy=quadmask_npy,
        vace_prompt=vace_prompt,
        frame_num=args.frame_num,
        save_dir=run_dir / "vace_backend",
        output_video=edited_video,
        base_seed=args.vace_base_seed,
        sample_steps=args.vace_sample_steps,
        control_branch_checkpoint=args.control_branch_checkpoint,
    )
    if vace_output is None:
        raise RuntimeError(f"VACE command failed; see {vace_info.get('stderr_path')}")
    control_branch = core.control_branch_info_from_context(vace_info)

    metadata = {
        "evidence_level": "INTERFACE",
        "visual_quality_evaluated": False,
        "operation": "add",
        "source_video_for_upstream_context": str(args.source_video),
        "source_video_passed_to_vace": False,
        "user_prompt": args.user_prompt,
        "planner": {
            "used": True,
            "route": "inline_add_planner",
            "adapter": str(args.planner_adapter),
            "planner_output_manually_modified": False,
            "vace_prompt_source": "planner_model",
            "manual_or_teacher_vace_prompt_used": False,
            "learned_planner_add_quality_claimed": False,
            "output_path": str(run_dir / "planner_output.json"),
        },
        "target_ref": target_ref,
        "vace_prompt": vace_prompt,
        "frame_num": args.frame_num,
        "qwen_edit": {"used": True, "edited_first_frame": str(edited_first_frame), "info": edit_info},
        "vace_conditioning_video": conditioning_meta,
        "sam2": sam2_info,
        "quadmask": quad_meta,
        "generation_mask": core.generation_mask_metadata(generation_mask),
        "control_branch": control_branch,
        "control_branch_checkpoint_loaded": control_branch["control_branch_checkpoint_loaded"],
        "trained_control_branch_used": control_branch["trained_control_branch_used"],
        "control_branch_installed_in_forward_vace": control_branch["control_branch_installed_in_forward_vace"],
        "vace_runtime_inputs": {
            "vace_conditioning_video": str(vace_conditioning),
            "quadmask_npy": str(quadmask_npy),
            "generation_mask": str(gen_mp4),
            "operation": "add",
            "vace_prompt": vace_prompt,
            "frame_num": args.frame_num,
        },
        "legacy_backend_arg_adapter_used": True,
        "edited_video": {
            "path": str(edited_video),
            "exists": edited_video.exists(),
            "size_bytes": edited_video.stat().st_size if edited_video.exists() else 0,
        },
        "success_criteria": {
            "edited_first_frame_exists": edited_first_frame.exists(),
            "quadmask_shape_ok": list(quadmask.shape) == [args.frame_num, height, width],
            "quadmask_values_ok": set(np.unique(quadmask).astype(int).tolist()).issubset({0, 63, 127, 255}),
            "generation_mask_full_domain": sorted(int(x) for x in np.unique(generation_mask)) == [255],
            "edited_video_exists": edited_video.exists() and edited_video.stat().st_size > 0,
        },
        "source_video_metadata": source_meta,
    }
    write_json(run_dir / "metadata.json", metadata)
    print(json.dumps({"run_dir": str(run_dir), "metadata": str(run_dir / "metadata.json"), "edited_video": str(edited_video)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""E2W remove interface: Counterfactual Planner -> GroundingDINO -> SAM2 -> VACE(remove).

Thin orchestrator over tools/e2w_pipeline_core.py. Grounding order (remove):
ground the original frame (GroundingDINO bbox -> SAM2 propagation), edit the first
frame ("remove <target_ref>"), then build quadmask / conditioning / generation mask
and call the VACE backend with operation=remove.

All model checkpoints are injectable; there are no archived/version-named defaults
beyond the current Counterfactual Planner adapter.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2w_pipeline_core as core  # noqa: E402
from e2w_pipeline_core import (  # noqa: E402  (re-exported for tests/back-compat)
    build_conditioning_video,
    build_generation_mask,
    build_quadmask,
    edit_first_frame,
    load_jsonl,
    parse_json_object,
    write_json,
)
from e2w_v0_common import serialize_vace_prompt, validate_counterfactual_planner_output  # noqa: E402

DEFAULT_EVAL = Path("/data/cwx/E2W/data/counterfactual_planner_sft/eval.jsonl")
DEFAULT_BASE_MODEL = Path("/data/cwx/Edit2World-unified/checkpoints/Qwen2.5-VL-7B-Instruct")
DEFAULT_ADAPTER = Path("/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3")
DEFAULT_OUT_DIR = Path("/data/cwx/E2W/runs/e2w_remove_debug")
DEFAULT_DINO_CKPT = Path("/data/cwx/edit2world-models/phase1/groundingdino_swint_ogc.pth")
DEFAULT_SAM2_REPO = Path("/data/cwx/Edit2World-unified/external/sam2")
DEFAULT_SAM2_CKPT = Path("/data/cwx/Edit2World-unified/checkpoints/sam2/sam2.1_hiera_large.pt")
DEFAULT_SAM2_CFG = Path("/data/cwx/Edit2World-unified/external/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
DEFAULT_PYTHON = Path("/data/cwx/conda/envs/edit2world-phase1-real/bin/python")
DEFAULT_VACE_REPO = Path("/data/cwx/Edit2World-unified/external/VACE")
DEFAULT_VACE_CKPT = Path("/data/cwx/Edit2World-unified/checkpoints/Wan2.1-VACE-14B")
DEFAULT_QWEN_IMAGE_EDIT = Path("/data/cwx/Edit2World-unified/checkpoints/Qwen-Image-Edit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dino-checkpoint", type=Path, default=DEFAULT_DINO_CKPT)
    parser.add_argument("--dino-config", type=Path, default=None)
    parser.add_argument("--dino-box-threshold", type=float, default=0.25)
    parser.add_argument("--dino-text-threshold", type=float, default=0.25)
    parser.add_argument("--sam2-repo", type=Path, default=DEFAULT_SAM2_REPO)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CKPT)
    parser.add_argument("--sam2-config", type=Path, default=DEFAULT_SAM2_CFG)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--vace-repo", type=Path, default=DEFAULT_VACE_REPO)
    parser.add_argument("--vace-ckpt", type=Path, default=DEFAULT_VACE_CKPT)
    parser.add_argument("--vace-model-name", default="vace-14B", choices=["vace-14B", "vace-1.3B"])
    parser.add_argument("--vace-size", default="480p")
    parser.add_argument("--vace-sample-steps", type=int, default=8)
    parser.add_argument("--vace-base-seed", type=int, default=2025)
    parser.add_argument("--vace-fps", type=float, default=6.0)
    parser.add_argument("--control-branch-checkpoint", type=Path)
    parser.add_argument("--skip-vace", action="store_true")
    parser.add_argument("--qwen-image-edit-checkpoint", type=Path, default=DEFAULT_QWEN_IMAGE_EDIT)
    parser.add_argument("--qwen-image-edit-seed", type=int, default=2025)
    parser.add_argument("--qwen-image-edit-steps", type=int, default=12)
    parser.add_argument("--qwen-image-edit-true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--skip-first-frame-edit", action="store_true")
    return parser.parse_args()


def run_planner(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inline Counterfactual Planner inference (via core) + remove-schema validation."""
    raw = core.run_planner(args.base_model, args.adapter, rows, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    planned: list[dict[str, Any]] = []
    for item in raw:
        parsed = item["parsed"]
        schema_valid = False
        validation_error = item.get("parse_error")
        if parsed is not None:
            schema_valid, validation_error = validate_counterfactual_planner_output(
                parsed, source_video_id=str(item["row"].get("video_id") or "unknown")
            )
        target_ref = str(parsed.get("target_ref") or "").strip() if parsed else ""
        planned.append(
            {
                "row": item["row"],
                "raw_output": item["raw_output"],
                "parsed": parsed,
                "json_parse_ok": item["json_parse_ok"],
                "schema_valid": schema_valid,
                "validation_error": validation_error,
                "target_ref": target_ref,
            }
        )
    return planned


def render_vace_prompt_from_v8(parsed: dict[str, Any], video_id: str) -> tuple[str, bool, str | None]:
    state = parsed.get("counterfactual_state") if isinstance(parsed.get("counterfactual_state"), dict) else {}
    keys = ["surface", "lighting", "shadow", "temporal", "interaction", "geometry"]
    parts = [str(state.get(key) or "").strip().rstrip(".") for key in keys if str(state.get(key) or "").strip()]
    prompt = ". ".join(parts).strip()
    if prompt and not prompt.endswith("."):
        prompt += "."
    edit_plan = {
        "source_video_id": video_id,
        "operation": "remove",
        "edit_subject": {"label": str(parsed.get("target_ref") or "").strip(), "aliases": [], "included_parts": []},
        "operation_details": {
            "target_object": {"label": str(parsed.get("target_ref") or "").strip()},
            "physical_consequences": [],
            "protected_objects": [],
        },
        "edited_scene": {"caption": prompt, "outcome_effects": []},
    }
    try:
        serialize_vace_prompt(edit_plan)
    except Exception as exc:
        return prompt, False, f"{type(exc).__name__}: {exc}"
    return prompt, True, None


def run_first_frame_edit(args: argparse.Namespace, anchor: Path, target_ref: str, video_id: str) -> tuple[bool, str | None, dict[str, Any]]:
    """Back-compat wrapper around core.edit_first_frame for the remove prompt."""
    if args.skip_first_frame_edit:
        return False, None, {"skipped": True, "reason": "--skip-first-frame-edit"}
    out_path = args.output_dir / f"edited_frame_{video_id}.jpg"
    info = core.edit_first_frame(
        anchor,
        f"remove {target_ref}",
        out_path,
        qwen_checkpoint=args.qwen_image_edit_checkpoint,
        seed=args.qwen_image_edit_seed,
        steps=args.qwen_image_edit_steps,
        true_cfg_scale=args.qwen_image_edit_true_cfg_scale,
    )
    return True, str(out_path), info


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dino_config = core.find_groundingdino_config(args.dino_config)
    print(f"GroundingDINO config: {dino_config}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pipeline")
    rows = load_jsonl(args.eval_jsonl)
    if len(rows) < args.sample_count:
        raise ValueError(f"eval set has {len(rows)} rows, cannot sample {args.sample_count}")
    sampled = random.Random(args.seed).sample(rows, args.sample_count)
    print(f"sampled_video_ids: {', '.join(str(r.get('video_id')) for r in sampled)}", flush=True)

    planned = run_planner(args, sampled)

    from groundingdino.util.inference import load_model  # noqa: WPS433

    dino_model = load_model(str(dino_config), str(args.dino_checkpoint), device="cuda")
    results: list[dict[str, Any]] = []
    for item in planned:
        row = item["row"]
        video_id = str(row.get("video_id") or "unknown")
        target_ref = item["target_ref"]
        anchor = core.image_path_from_row(row)
        frames, anchor_idx = core.frame_paths_for_anchor(anchor)
        record: dict[str, Any] = {
            "video_id": video_id,
            "anchor_frame": str(anchor),
            "anchor_index": anchor_idx,
            "target_ref": target_ref,
            "planner_json_parse_ok": item["json_parse_ok"],
            "planner_schema_valid": item["schema_valid"],
            "planner_validation_error": item["validation_error"],
            "raw_output": item["raw_output"],
            "parsed": item["parsed"],
        }
        try:
            if not target_ref:
                raise RuntimeError("planner did not produce target_ref")
            bbox, confidence = core.detect_bbox(
                dino_model, anchor, target_ref, box_threshold=args.dino_box_threshold, text_threshold=args.dino_text_threshold
            )
            grounding_path = args.output_dir / f"debug_grounding_{video_id}.jpg"
            core.save_grounding_debug(anchor, bbox, target_ref, confidence, grounding_path)
            masks = core.sam2_masks_from_box(
                frames[0].parent,
                anchor_idx,
                bbox,
                sam2_repo=args.sam2_repo,
                sam2_checkpoint=args.sam2_checkpoint,
                sam2_config=args.sam2_config,
            )
            mask_areas = [float(mask.mean()) for mask in masks.values()]
            wanted_indices = sorted({0, len(frames) // 2, len(frames) - 1})
            mask_debug_paths: list[str] = []
            for frame_idx in wanted_indices:
                if frame_idx not in masks:
                    continue
                out_path = args.output_dir / f"debug_mask_{video_id}_{frame_idx}.jpg"
                core.save_mask_overlay(frames[frame_idx], masks[frame_idx], out_path)
                mask_debug_paths.append(str(out_path))

            anchor_image = cv2.imread(str(anchor), cv2.IMREAD_COLOR)
            if anchor_image is None:
                raise RuntimeError(f"failed to read image: {anchor}")
            height, width = anchor_image.shape[:2]
            quadmask, generation_mask = build_quadmask(masks, len(frames), height, width)
            quadmask_path = args.output_dir / f"quadmask_{video_id}.npy"
            generation_mask_path = args.output_dir / f"generation_mask_{video_id}.npy"
            np.save(quadmask_path, quadmask)
            np.save(generation_mask_path, generation_mask)
            mid_idx = len(frames) // 2
            quadmask_debug_path = args.output_dir / f"debug_quadmask_{video_id}_mid.jpg"
            core.save_quadmask_preview(frames[mid_idx], quadmask[mid_idx], quadmask_debug_path)

            vace_prompt = ""
            vace_prompt_valid = False
            vace_prompt_error = "planner output was not parsed"
            vace_output_path: str | None = None
            vace_info: dict[str, Any] = {}
            parsed = item["parsed"]
            if isinstance(parsed, dict):
                vace_prompt, vace_prompt_valid, vace_prompt_error = render_vace_prompt_from_v8(parsed, video_id)

            first_frame_edit_ok = False
            edited_frame_path: str | None = None
            first_frame_edit_info: dict[str, Any] = {}
            if vace_prompt_valid and not args.skip_vace:
                first_frame_edit_ok, edited_frame_path, first_frame_edit_info = run_first_frame_edit(args, anchor, target_ref, video_id)

            if vace_prompt_valid and not args.skip_vace:
                if not first_frame_edit_ok or edited_frame_path is None:
                    raise RuntimeError("first frame edit did not produce an edited frame for VACE")
                vace_frame_count = core.next_vace_frame_count(len(frames))
                vace_quadmask = core.pad_time_axis(quadmask, vace_frame_count)
                vace_generation_mask = build_generation_mask(vace_frame_count, height, width)
                vace_quadmask_path = args.output_dir / f"quadmask_{video_id}_vace.npy"
                vace_generation_mask_npy_path = args.output_dir / f"generation_mask_{video_id}_vace.npy"
                vace_generation_mask_video_path = args.output_dir / f"generation_mask_{video_id}.mp4"
                vace_conditioning_video_path = args.output_dir / f"vace_conditioning_video_{video_id}.mp4"
                np.save(vace_quadmask_path, vace_quadmask)
                np.save(vace_generation_mask_npy_path, vace_generation_mask)
                core.write_gray_video(vace_generation_mask, vace_generation_mask_video_path, args.vace_fps)
                conditioning_metadata = build_conditioning_video(
                    Path(edited_frame_path),
                    vace_conditioning_video_path,
                    frame_count=vace_frame_count,
                    width=width,
                    height=height,
                    fps=args.vace_fps,
                )
                vace_output_path, vace_info = core.run_vace(
                    "remove",
                    video_id,
                    python=args.python,
                    vace_repo=args.vace_repo,
                    vace_ckpt=args.vace_ckpt,
                    model_name=args.vace_model_name,
                    size=args.vace_size,
                    conditioning_video=vace_conditioning_video_path,
                    generation_mask_video=vace_generation_mask_video_path,
                    quadmask_npy=vace_quadmask_path,
                    vace_prompt=vace_prompt,
                    frame_num=vace_frame_count,
                    save_dir=args.output_dir / f"vace_{video_id}",
                    output_video=args.output_dir / f"edited_video_{video_id}.mp4",
                    base_seed=args.vace_base_seed,
                    sample_steps=args.vace_sample_steps,
                    control_branch_checkpoint=args.control_branch_checkpoint,
                )
                vace_info.update(
                    {
                        "legacy_backend_arg_adapter_used": True,
                        "vace_frame_count": vace_frame_count,
                        "vace_runtime_inputs": {
                            "vace_conditioning_video": str(vace_conditioning_video_path),
                            "quadmask_npy": str(vace_quadmask_path),
                            "generation_mask": str(vace_generation_mask_video_path),
                            "operation": "remove",
                            "vace_prompt": vace_prompt,
                            "frame_num": vace_frame_count,
                        },
                        "vace_runtime_input_metadata": {
                            "vace_conditioning_video_first_frame": edited_frame_path,
                            "vace_conditioning_video": conditioning_metadata,
                            "quadmask": core.mask_metadata(vace_quadmask),
                            "generation_mask": core.generation_mask_metadata(vace_generation_mask),
                        },
                    }
                )
                vace_info.update(core.control_branch_info_from_context(vace_info))
            elif vace_prompt_valid and args.skip_vace:
                vace_info = {"skipped": True, "reason": "--skip-vace"}
            else:
                vace_info = {"skipped": True, "reason": vace_prompt_error}
            record.update(
                {
                    "status": "ok",
                    "bbox_xyxy": list(bbox),
                    "bbox_confidence": confidence,
                    "mask_mean_area": float(np.mean(mask_areas)),
                    "quadmask_q0_mean_area": float(np.mean(quadmask == 0)),
                    "quadmask_q2_mean_area": float(np.mean(quadmask == 127)),
                    "vace_prompt": vace_prompt,
                    "vace_prompt_preview": vace_prompt[:60],
                    "vace_prompt_valid": vace_prompt_valid,
                    "vace_prompt_error": vace_prompt_error,
                    "first_frame_edit_ok": first_frame_edit_ok,
                    "edited_first_frame": edited_frame_path,
                    "first_frame_edit_info": first_frame_edit_info,
                    "vace_output_path": vace_output_path,
                    "vace_info": vace_info,
                    "debug_grounding": str(grounding_path),
                    "debug_masks": mask_debug_paths,
                    "debug_quadmask": str(quadmask_debug_path),
                    "quadmask_npy": str(quadmask_path),
                    "generation_mask_npy": str(generation_mask_path),
                    "quadmask_metadata": core.mask_metadata(quadmask),
                    "generation_mask_metadata": core.generation_mask_metadata(generation_mask),
                    "vace_runtime_inputs": {
                        "vace_conditioning_video": (
                            vace_info.get("vace_runtime_inputs", {}).get("vace_conditioning_video")
                            if isinstance(vace_info.get("vace_runtime_inputs"), dict)
                            else None
                        ),
                        "quadmask_npy": (
                            vace_info.get("vace_runtime_inputs", {}).get("quadmask_npy")
                            if isinstance(vace_info.get("vace_runtime_inputs"), dict)
                            else str(quadmask_path)
                        ),
                        "generation_mask": (
                            vace_info.get("vace_runtime_inputs", {}).get("generation_mask")
                            if isinstance(vace_info.get("vace_runtime_inputs"), dict)
                            else str(generation_mask_path)
                        ),
                        "operation": "remove",
                        "vace_prompt": vace_prompt,
                        "frame_num": (
                            vace_info.get("vace_runtime_inputs", {}).get("frame_num")
                            if isinstance(vace_info.get("vace_runtime_inputs"), dict)
                            else len(frames)
                        ),
                    },
                    "source_video_passed_to_vace": False,
                    "frame_count": len(frames),
                    "mask_frame_count": len(masks),
                }
            )
            print(
                f"{video_id} | {target_ref[:40]} | bbox{tuple(bbox)} | {confidence:.4f} | "
                f"first_frame={first_frame_edit_ok} vace_valid={vace_prompt_valid} vace_output={vace_output_path}",
                flush=True,
            )
        except Exception as exc:
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"{video_id} | {target_ref[:40]} | FAILED | {type(exc).__name__}: {exc}", flush=True)
        results.append(record)

    write_json(args.output_dir / "summary.json", {"results": results})
    print(f"output_dir: {args.output_dir}", flush=True)
    return 0 if all(r.get("status") == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

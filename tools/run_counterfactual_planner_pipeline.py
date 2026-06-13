#!/usr/bin/env python3
"""Debug Counterfactual Planner -> GroundingDINO -> SAM2 propagation on eval samples."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import serialize_vace_prompt, validate_counterfactual_planner_output  # noqa: E402


DEFAULT_EVAL = Path("/data/cwx/E2W/data/counterfactual_planner_sft/eval.jsonl")
DEFAULT_BASE_MODEL = Path("/data/cwx/Edit2World-unified/checkpoints/Qwen2.5-VL-7B-Instruct")
DEFAULT_ADAPTER = Path("/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3")
DEFAULT_OUT_DIR = Path("/data/cwx/E2W/runs/e2w_counterfactual_planner_pipeline_debug")
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        return (obj, None) if isinstance(obj, dict) else (None, "decoded JSON is not an object")
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(cleaned[start : end + 1])
            return (obj, None) if isinstance(obj, dict) else (None, "decoded JSON snippet is not an object")
        except json.JSONDecodeError as exc:
            return None, str(exc)
    return None, "no JSON object found"


def find_groundingdino_config(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit.resolve()
        raise FileNotFoundError(f"GroundingDINO config not found: {explicit}")

    roots: list[Path] = []
    try:
        import groundingdino  # noqa: WPS433

        roots.append(Path(groundingdino.__file__).resolve().parent)
    except Exception as exc:
        print(f"import groundingdino failed while searching config: {exc}", flush=True)
    roots.append(Path("/data/cwx/edit2world-models/phase1"))

    candidates: list[Path] = []
    for root in roots:
        print(f"GroundingDINO config search root: {root}", flush=True)
        if not root.exists():
            continue
        for pattern in ("GroundingDINO_SwinT_OGC.py", "*SwinT*OGC*.py"):
            candidates.extend(root.rglob(pattern))

    candidates = sorted({p.resolve() for p in candidates})
    if not candidates:
        raise FileNotFoundError("GroundingDINO config not found in site-packages/groundingdino or /data/cwx/edit2world-models/phase1")
    return candidates[0]


def image_path_from_row(row: dict[str, Any]) -> Path:
    for message in row.get("messages", []):
        for item in message.get("content", []):
            if isinstance(item, dict) and item.get("type") == "image" and item.get("image"):
                return Path(item["image"])
    raise ValueError(f"row {row.get('id') or row.get('video_id')} has no image content")


def frame_paths_for_anchor(anchor: Path) -> tuple[list[Path], int]:
    frames = sorted(anchor.parent.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no jpg frames under {anchor.parent}")
    anchor_resolved = anchor.resolve()
    for idx, frame in enumerate(frames):
        if frame.resolve() == anchor_resolved:
            return frames, idx
    raise FileNotFoundError(f"anchor frame not found in frame directory: {anchor}")


def run_planner(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    planned: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        messages = [row["messages"][0]]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        gen_kwargs: dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
        if args.temperature and args.temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": args.temperature})
        else:
            gen_kwargs.update({"do_sample": False})
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        parsed, parse_error = parse_json_object(text)
        schema_valid = False
        validation_error = parse_error
        if parsed is not None:
            schema_valid, validation_error = validate_counterfactual_planner_output(parsed, source_video_id=str(row.get("video_id") or "unknown"))
        target_ref = str(parsed.get("target_ref") or "").strip() if parsed else ""
        planned.append(
            {
                "row": row,
                "raw_output": text,
                "parsed": parsed,
                "json_parse_ok": parsed is not None,
                "schema_valid": schema_valid,
                "validation_error": validation_error,
                "target_ref": target_ref,
            }
        )
        print(
            f"planner {idx}/{len(rows)} video_id={row.get('video_id')} "
            f"parse={parsed is not None} valid={schema_valid} target_ref={target_ref[:60]!r}",
            flush=True,
        )

    del model
    del base
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return planned


def cxcywh_norm_to_xyxy_pixels(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = [float(x) for x in box.tolist()]
    x1 = int(round((cx - bw / 2.0) * width))
    y1 = int(round((cy - bh / 2.0) * height))
    x2 = int(round((cx + bw / 2.0) * width))
    y2 = int(round((cy + bh / 2.0) * height))
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(1, min(width, x2)),
        max(1, min(height, y2)),
    )


def save_grounding_debug(image_path: Path, bbox: tuple[int, int, int, int], label: str, confidence: float, out_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    text = f"{label[:48]} {confidence:.3f}"
    cv2.putText(image, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def save_mask_overlay(frame_path: Path, mask: np.ndarray, out_path: Path) -> None:
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {frame_path}")
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    overlay = image.copy()
    overlay[mask.astype(bool)] = (0.35 * overlay[mask.astype(bool)] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    blended = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blended)


def write_video_from_frames(
    frame_paths: list[Path],
    out_path: Path,
    fps: float,
    target_frame_count: int | None = None,
    first_frame_override: Path | None = None,
) -> None:
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"failed to read image: {frame_paths[0]}")
    height, width = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {out_path}")
    count = target_frame_count or len(frame_paths)
    try:
        for idx in range(count):
            frame_path = first_frame_override if idx == 0 and first_frame_override is not None else frame_paths[min(idx, len(frame_paths) - 1)]
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"failed to read image: {frame_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def write_vace_conditioning_video(
    edited_first_frame: Path,
    out_path: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
) -> dict[str, Any]:
    image = cv2.imread(str(edited_first_frame), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read edited first frame: {edited_first_frame}")
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {out_path}")
    try:
        zero = np.zeros_like(image)
        for idx in range(frame_count):
            writer.write(image if idx == 0 else zero)
    finally:
        writer.release()
    return {
        "conditioning_strategy": "edited_first_frame_plus_zero_filled_future_frames",
        "frame_0_is_edited_first_frame": True,
        "future_frames_are_zero_filled": True,
        "future_frames_source_video_used": False,
        "frame_count": frame_count,
    }


def write_gray_video(masks: np.ndarray, out_path: Path, fps: float) -> None:
    height, width = masks.shape[1:]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {out_path}")
    try:
        for mask in masks.astype(np.uint8):
            writer.write(np.repeat(mask[:, :, None], 3, axis=2))
    finally:
        writer.release()


def next_vace_frame_count(frame_count: int) -> int:
    remainder = frame_count % 4
    return frame_count if remainder == 1 else frame_count + ((1 - remainder) % 4)


def pad_time_axis(array: np.ndarray, target_frame_count: int) -> np.ndarray:
    if array.shape[0] == target_frame_count:
        return array
    if array.shape[0] > target_frame_count:
        raise ValueError(f"cannot pad array with T={array.shape[0]} down to {target_frame_count}")
    tail = np.repeat(array[-1:],
                     target_frame_count - array.shape[0],
                     axis=0)
    return np.concatenate([array, tail], axis=0)


def affected_region_from_mask(mask: np.ndarray, radius: int = 8) -> np.ndarray:
    mo = mask.astype(bool)
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mo.astype(np.uint8), kernel, iterations=1) > 0
    ys, xs = np.where(mo)
    if len(xs) == 0:
        return dilated
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    width = max(1, x2 - x1 + 1)
    height = max(1, y2 - y1 + 1)
    center = (int(round((x1 + x2) / 2.0)), int(round(y2 + 10)))
    axes = (max(1, int(round(width * 1.2 / 2.0))), max(1, int(round(height * 0.2 / 2.0))))
    ellipse = np.zeros_like(mo, dtype=np.uint8)
    cv2.ellipse(ellipse, center, axes, 0, 0, 360, 255, -1)
    blur_ksize = max(3, radius * 2 + 1)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(ellipse, (blur_ksize, blur_ksize), 0) > 16
    return dilated | blurred


def build_quadmask(masks: dict[int, np.ndarray], frame_count: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    quad = np.full((frame_count, height, width), 255, dtype=np.uint8)
    for frame_idx in range(frame_count):
        mask = masks.get(frame_idx)
        if mask is None:
            mo = np.zeros((height, width), dtype=bool)
        else:
            mo = mask.astype(bool)
            if mo.shape != (height, width):
                mo = cv2.resize(mo.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        ma = affected_region_from_mask(mo)
        quad[frame_idx][ma & ~mo] = 127
        quad[frame_idx][mo] = 0
    generation_mask = np.full((frame_count, height, width), 255, dtype=np.uint8)
    return quad, generation_mask


def mask_metadata(mask: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "values": sorted(int(value) for value in np.unique(mask)),
    }


def generation_mask_metadata(mask: np.ndarray) -> dict[str, Any]:
    values = sorted(int(value) for value in np.unique(mask))
    return {
        "generation_mask_shape": list(mask.shape),
        "generation_mask_values": values,
        "generation_mask_is_full_domain": values == [255],
        "generation_mask_encodes_quadmask_semantics": False,
    }


def save_quadmask_preview(frame_path: Path, quad_frame: np.ndarray, out_path: Path) -> None:
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {frame_path}")
    if quad_frame.shape[:2] != image.shape[:2]:
        quad_frame = cv2.resize(quad_frame, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    colors = {
        0: np.array([0, 0, 255], dtype=np.uint8),
        63: np.array([0, 128, 255], dtype=np.uint8),
        127: np.array([0, 255, 255], dtype=np.uint8),
        255: np.array([160, 160, 160], dtype=np.uint8),
    }
    overlay = image.copy()
    for value, color in colors.items():
        region = quad_frame == value
        overlay[region] = (0.35 * overlay[region] + 0.65 * color).astype(np.uint8)
    blended = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blended)


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
    if args.skip_first_frame_edit:
        return False, None, {"skipped": True, "reason": "--skip-first-frame-edit"}

    from diffusers import QwenImageEditPipeline  # noqa: WPS433

    image = Image.open(anchor).convert("RGB")
    prompt = f"remove {target_ref}"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pipe = QwenImageEditPipeline.from_pretrained(str(args.qwen_image_edit_checkpoint), torch_dtype=dtype)
    if torch.cuda.is_available():
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(args.qwen_image_edit_seed)
    out_path = args.output_dir / f"edited_frame_{video_id}.jpg"
    result = pipe(
        image,
        prompt=prompt,
        negative_prompt=" ",
        num_inference_steps=args.qwen_image_edit_steps,
        generator=generator,
        true_cfg_scale=args.qwen_image_edit_true_cfg_scale,
        num_images_per_prompt=1,
    ).images[0]
    raw_size = result.size
    if raw_size != image.size:
        result = result.resize(image.size, Image.Resampling.LANCZOS)
    result.save(out_path, quality=95)
    with Image.open(out_path) as edited_image:
        edited_size = list(edited_image.size)
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True, str(out_path), {
        "instruction": prompt,
        "source_image": str(anchor),
        "edited_frame": str(out_path),
        "seed": args.qwen_image_edit_seed,
        "steps": args.qwen_image_edit_steps,
        "true_cfg_scale": args.qwen_image_edit_true_cfg_scale,
        "source_size": list(image.size),
        "raw_output_size": list(raw_size),
        "edited_size": edited_size,
        "target_mask_consumed_by_backend": False,
    }


def run_vace(
    args: argparse.Namespace,
    video_id: str,
    vace_conditioning_video: Path,
    generation_mask_video: Path,
    quadmask_npy: Path,
    vace_prompt: str,
    frame_num: int,
) -> tuple[str | None, dict[str, Any]]:
    save_dir = args.output_dir / f"vace_{video_id}"
    output_video = args.output_dir / f"edited_video_{video_id}.mp4"
    command = [
        str(args.python.resolve()),
        str((Path(__file__).resolve().parent / "run_wan_vace_quad_i2v.py").resolve()),
        "--vace_repo",
        str(args.vace_repo.resolve()),
        "--ckpt_dir",
        str(args.vace_ckpt.resolve()),
        "--model_name",
        args.vace_model_name,
        "--size",
        args.vace_size,
        "--src_video",
        str(vace_conditioning_video.resolve()),
        "--generation_mask",
        str(generation_mask_video.resolve()),
        "--quadmask_npy",
        str(quadmask_npy.resolve()),
        "--operation",
        "remove",
        "--prompt",
        vace_prompt,
        "--frame_num",
        str(frame_num),
        "--save_dir",
        str(save_dir.resolve()),
        "--save_file",
        str(output_video.resolve()),
        "--base_seed",
        str(args.vace_base_seed),
        "--sample_steps",
        str(args.vace_sample_steps),
        "--offload_model",
    ]
    if args.control_branch_checkpoint:
        command.extend(["--control_branch_checkpoint", str(args.control_branch_checkpoint.resolve())])
    save_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        save_dir / "vace_backend_adapter_command.json",
        {
            "argv": command,
            "cwd": str(Path.cwd()),
            "legacy_backend_arg_adapter_used": True,
            "backend_visual_condition_arg": "--src_video",
            "e2w_visual_condition_input": "vace_conditioning_video",
            "backend_text_arg": "--prompt",
            "e2w_text_input": "vace_prompt",
        },
    )
    env = os.environ.copy()
    env.setdefault("USE_TF", "0")
    env.setdefault("TRANSFORMERS_NO_TF", "1")
    proc = subprocess.run(command, cwd=str(Path.cwd()), env=env, text=True, capture_output=True, check=False)
    (save_dir / "vace_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (save_dir / "vace_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    info = {
        "returncode": proc.returncode,
        "backend_adapter_command_path": str(save_dir / "vace_backend_adapter_command.json"),
        "stdout_path": str(save_dir / "vace_stdout.txt"),
        "stderr_path": str(save_dir / "vace_stderr.txt"),
    }
    if proc.returncode != 0:
        info["error"] = f"VACE command failed with return code {proc.returncode}"
        return None, info
    context_info_path = save_dir / "e2w_quad_context.json"
    if context_info_path.exists():
        info["context_info"] = json.loads(context_info_path.read_text(encoding="utf-8"))
    return str(output_video), info


def detect_bbox(args: argparse.Namespace, dino_model: Any, image_path: Path, target_ref: str) -> tuple[tuple[int, int, int, int], float]:
    from groundingdino.util.inference import load_image, predict  # noqa: WPS433

    image_source, image = load_image(str(image_path))
    boxes, logits, _phrases = predict(
        model=dino_model,
        image=image,
        caption=target_ref,
        box_threshold=args.dino_box_threshold,
        text_threshold=args.dino_text_threshold,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if len(boxes) == 0:
        raise RuntimeError(f"GroundingDINO returned no boxes for prompt {target_ref!r}")
    best_idx = int(torch.argmax(logits).item())
    height, width = image_source.shape[:2]
    bbox = cxcywh_norm_to_xyxy_pixels(boxes[best_idx].detach().cpu().numpy(), width, height)
    confidence = float(logits[best_idx].detach().cpu().item())
    return bbox, confidence


def propagate_masks(args: argparse.Namespace, frame_dir: Path, anchor_idx: int, bbox: tuple[int, int, int, int]) -> dict[int, np.ndarray]:
    sys.path.insert(0, str(args.sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor  # noqa: WPS433

    sam2_config_name = resolve_sam2_config_for_api(args.sam2_config, args.sam2_repo)
    predictor = build_sam2_video_predictor(sam2_config_name, str(args.sam2_checkpoint), device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        predictor.add_new_points_or_box(
            state,
            frame_idx=anchor_idx,
            obj_id=1,
            box=np.array(bbox, dtype=np.float32),
        )
        masks: dict[int, np.ndarray] = {}
        for reverse in (False, True):
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                state,
                start_frame_idx=anchor_idx,
                reverse=reverse,
            ):
                if len(out_obj_ids) == 0:
                    continue
                mask = (out_mask_logits[0] > 0.0).detach().cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                masks[int(out_frame_idx)] = mask.astype(bool)
    if not masks:
        raise RuntimeError("SAM2 returned no propagated masks")
    return masks


def resolve_sam2_config_for_api(config_path: Path, sam2_repo: Path) -> str:
    """SAM2's Hydra compose expects package-relative config names."""
    if not config_path.is_absolute():
        return str(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"SAM2 absolute config not found: {config_path}")
    package_root = sam2_repo / "sam2"
    try:
        return config_path.resolve().relative_to(package_root.resolve()).as_posix()
    except ValueError:
        return str(config_path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dino_config = find_groundingdino_config(args.dino_config)
    print(f"GroundingDINO config: {dino_config}", flush=True)
    print(f"SAM2 config absolute: {args.sam2_config}", flush=True)
    print(f"SAM2 config API name: {resolve_sam2_config_for_api(args.sam2_config, args.sam2_repo)}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this debug pipeline")
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
        anchor = image_path_from_row(row)
        frames, anchor_idx = frame_paths_for_anchor(anchor)
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
            bbox, confidence = detect_bbox(args, dino_model, anchor, target_ref)
            grounding_path = args.output_dir / f"debug_grounding_{video_id}.jpg"
            save_grounding_debug(anchor, bbox, target_ref, confidence, grounding_path)
            masks = propagate_masks(args, frames[0].parent, anchor_idx, bbox)
            mask_areas = [float(mask.mean()) for mask in masks.values()]
            wanted_indices = sorted({0, len(frames) // 2, len(frames) - 1})
            mask_debug_paths: list[str] = []
            for frame_idx in wanted_indices:
                if frame_idx not in masks:
                    continue
                out_path = args.output_dir / f"debug_mask_{video_id}_{frame_idx}.jpg"
                save_mask_overlay(frames[frame_idx], masks[frame_idx], out_path)
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
            save_quadmask_preview(frames[mid_idx], quadmask[mid_idx], quadmask_debug_path)

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
                vace_frame_count = next_vace_frame_count(len(frames))
                vace_quadmask = pad_time_axis(quadmask, vace_frame_count)
                vace_generation_mask = np.full_like(vace_quadmask, 255, dtype=np.uint8)
                vace_quadmask_path = args.output_dir / f"quadmask_{video_id}_vace.npy"
                vace_generation_mask_npy_path = args.output_dir / f"generation_mask_{video_id}_vace.npy"
                vace_generation_mask_video_path = args.output_dir / f"generation_mask_{video_id}.mp4"
                vace_conditioning_video_path = args.output_dir / f"vace_conditioning_video_{video_id}.mp4"
                np.save(vace_quadmask_path, vace_quadmask)
                np.save(vace_generation_mask_npy_path, vace_generation_mask)
                write_gray_video(vace_generation_mask, vace_generation_mask_video_path, args.vace_fps)
                conditioning_metadata = write_vace_conditioning_video(
                    Path(edited_frame_path),
                    vace_conditioning_video_path,
                    frame_count=vace_frame_count,
                    width=width,
                    height=height,
                    fps=args.vace_fps,
                )
                vace_output_path, vace_info = run_vace(
                    args,
                    video_id,
                    vace_conditioning_video_path,
                    vace_generation_mask_video_path,
                    vace_quadmask_path,
                    vace_prompt,
                    vace_frame_count,
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
                            "quadmask": mask_metadata(vace_quadmask),
                            "generation_mask": generation_mask_metadata(vace_generation_mask),
                        },
                    }
                )
                context_info = vace_info.get("context_info") if isinstance(vace_info.get("context_info"), dict) else {}
                vace_info.update(
                    {
                        "control_branch_checkpoint_loaded": bool(context_info.get("control_branch_checkpoint_loaded")),
                        "trained_control_branch_used": bool(context_info.get("trained_control_branch_used")),
                        "control_branch_step": context_info.get("control_branch_step"),
                        "control_branch_gate": context_info.get("control_branch_gate"),
                        "control_branch_installed_in_forward_vace": bool(
                            context_info.get("control_branch_installed_in_forward_vace")
                        ),
                    }
                )
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
                    "quadmask_metadata": mask_metadata(quadmask),
                    "generation_mask_metadata": generation_mask_metadata(generation_mask),
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
                    "evidence_chain": {
                        "planner_json": {
                            "json_parse_ok": item["json_parse_ok"],
                            "schema_valid": item["schema_valid"],
                            "raw_output_recorded": True,
                            "parsed_recorded": isinstance(item["parsed"], dict),
                            "target_ref": target_ref,
                        },
                        "grounding": {
                            "target_ref": target_ref,
                            "bbox_xyxy": list(bbox),
                            "bbox_confidence": confidence,
                            "debug_grounding": str(grounding_path),
                            "mask_frame_count": len(masks),
                        },
                        "quadmask": {
                            "path": str(quadmask_path),
                            **mask_metadata(quadmask),
                        },
                        "vace_prompt": {
                            "value": vace_prompt,
                            "valid": vace_prompt_valid,
                            "error": vace_prompt_error,
                            "source": "planner_json_counterfactual_state",
                        },
                        "vace_runtime_inputs": (
                            vace_info.get("vace_runtime_inputs")
                            if isinstance(vace_info.get("vace_runtime_inputs"), dict)
                            else {
                                "vace_conditioning_video": None,
                                "quadmask_npy": str(quadmask_path),
                                "generation_mask": str(generation_mask_path),
                                "operation": "remove",
                                "vace_prompt": vace_prompt,
                                "frame_num": len(frames),
                            }
                        ),
                    },
                    "frame_count": len(frames),
                    "mask_frame_count": len(masks),
                }
            )
            print(
                f"{video_id} | {target_ref[:40]} | "
                f"bbox({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) | "
                f"{confidence:.4f} | {float(np.mean(mask_areas)):.6f} | "
                f"q0={float(np.mean(quadmask == 0)):.6f} q2={float(np.mean(quadmask == 127)):.6f} | "
                f"first_frame={first_frame_edit_ok} vace_valid={vace_prompt_valid} vace_output={vace_output_path}",
                flush=True,
            )
        except Exception as exc:
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"{video_id} | {target_ref[:40]} | FAILED | {type(exc).__name__}: {exc}", flush=True)
        results.append(record)

    write_json(args.output_dir / "summary.json", {"results": results})
    print(f"debug_output_dir: {args.output_dir}", flush=True)
    return 0 if all(r.get("status") == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

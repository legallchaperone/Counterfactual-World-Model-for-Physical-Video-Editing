#!/usr/bin/env python3
"""Shared core stages for the E2W remove / add / add->remove interfaces.

This module holds single, operation-parameterized, checkpoint-injected
implementations of the pipeline stages that were previously duplicated across
`run_counterfactual_planner_pipeline.py` (remove) and
`run_add_pipeline_interface.py` (add):

    planner inference -> first-frame edit -> grounding -> quadmask ->
    conditioning video -> generation mask -> VACE backend

Functions take explicit parameters (no argparse coupling) so the three thin
entry points can compose them. Remove and add differ only in their grounding
order (remove grounds the original frame; add grounds the edited frame), so
grounding has operation-specific helpers while everything else is shared.

The current VACE runtime contract (see docs/E2W_SPEC.md) is exactly:
    vace_conditioning_video, quadmask_npy, generation_mask, operation,
    vace_prompt, frame_num
"""

from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CURRENT_VACE_INPUTS = {
    "vace_conditioning_video",
    "quadmask_npy",
    "generation_mask",
    "operation",
    "vace_prompt",
    "frame_num",
}
QUADMASK_VALUES = {0, 63, 127, 255}


# --------------------------------------------------------------------------- #
# IO / JSON helpers
# --------------------------------------------------------------------------- #
def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


# --------------------------------------------------------------------------- #
# Planner inference (inline Qwen2.5-VL + PEFT) — operation-agnostic.
# Validation of the parsed JSON is the caller's responsibility (operation
# decides remove vs add schema), keeping this stage checkpoint-only.
# --------------------------------------------------------------------------- #
def run_planner(
    base_model: Path,
    adapter: Path | None,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int = 768,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    import torch  # local import keeps array-only callers light
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    base = model
    if adapter is not None:
        model = PeftModel.from_pretrained(base, adapter)
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
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if temperature and temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            gen_kwargs.update({"do_sample": False})
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        parsed, parse_error = parse_json_object(text)
        planned.append(
            {
                "row": row,
                "raw_output": text,
                "parsed": parsed,
                "json_parse_ok": parsed is not None,
                "parse_error": parse_error,
            }
        )
        print(
            f"planner {idx}/{len(rows)} video_id={row.get('video_id')} parse={parsed is not None}",
            flush=True,
        )

    del model
    del base
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return planned


# --------------------------------------------------------------------------- #
# First-frame edit (Qwen-Image-Edit) — one implementation; caller supplies the
# operation-specific prompt (e.g. "remove X" or an add scene instruction).
# --------------------------------------------------------------------------- #
def edit_first_frame(
    image_path: Path,
    prompt: str,
    out_path: Path,
    *,
    qwen_checkpoint: Path,
    seed: int = 2025,
    steps: int = 12,
    true_cfg_scale: float = 4.0,
) -> dict[str, Any]:
    import torch
    from diffusers import QwenImageEditPipeline
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pipe = QwenImageEditPipeline.from_pretrained(str(qwen_checkpoint), torch_dtype=dtype)
    if torch.cuda.is_available():
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        image,
        prompt=prompt,
        negative_prompt=" ",
        num_inference_steps=steps,
        generator=generator,
        true_cfg_scale=true_cfg_scale,
        num_images_per_prompt=1,
    ).images[0]
    raw_size = result.size
    if raw_size != image.size:
        result = result.resize(image.size, Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path, quality=95)
    with Image.open(out_path) as edited_image:
        edited_size = list(edited_image.size)
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "instruction": prompt,
        "source_image": str(image_path),
        "edited_frame": str(out_path),
        "seed": seed,
        "steps": steps,
        "true_cfg_scale": true_cfg_scale,
        "source_size": list(image.size),
        "raw_output_size": list(raw_size),
        "edited_size": edited_size,
        "target_mask_consumed_by_backend": False,
    }


def edit_first_frame_inpaint(
    image_path: Path,
    prompt: str,
    mask: np.ndarray,
    out_path: Path,
    *,
    qwen_checkpoint: Path,
    seed: int = 2025,
    steps: int = 20,
    true_cfg_scale: float = 4.0,
    strength: float = 1.0,
) -> dict[str, Any]:
    """Masked add edit: the object is generated ONLY inside `mask` (HxW uint8, 255 =
    region to paint). This makes the first-frame edit obey the planner's chosen
    location, so SAM2 (seeded at the planner point inside the mask) grounds the
    actually-added object. Uses QwenImageEditInpaintPipeline."""
    import torch
    from diffusers import QwenImageEditInpaintPipeline
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    if mask.shape[:2] != (image.size[1], image.size[0]):
        raise ValueError(f"mask shape {mask.shape} does not match image HxW {(image.size[1], image.size[0])}")
    mask_img = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pipe = QwenImageEditInpaintPipeline.from_pretrained(str(qwen_checkpoint), torch_dtype=dtype)
    if torch.cuda.is_available():
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        image=image,
        mask_image=mask_img,
        prompt=prompt,
        negative_prompt=" ",
        num_inference_steps=steps,
        generator=generator,
        true_cfg_scale=true_cfg_scale,
        strength=strength,
        num_images_per_prompt=1,
    ).images[0]
    raw_size = result.size
    if raw_size != image.size:
        result = result.resize(image.size, Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path, quality=95)
    with Image.open(out_path) as edited_image:
        edited_size = list(edited_image.size)
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "instruction": prompt,
        "source_image": str(image_path),
        "edited_frame": str(out_path),
        "edit_mode": "masked_inpaint_at_planner_region",
        "seed": seed,
        "steps": steps,
        "true_cfg_scale": true_cfg_scale,
        "strength": strength,
        "mask_area_px": int((np.asarray(mask) > 0).sum()),
        "source_size": list(image.size),
        "raw_output_size": list(raw_size),
        "edited_size": edited_size,
        "target_mask_consumed_by_backend": False,
    }


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #
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


def find_groundingdino_config(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit.resolve()
        raise FileNotFoundError(f"GroundingDINO config not found: {explicit}")
    roots: list[Path] = []
    try:
        import groundingdino

        roots.append(Path(groundingdino.__file__).resolve().parent)
    except Exception as exc:
        print(f"import groundingdino failed while searching config: {exc}", flush=True)
    roots.append(Path("/data/cwx/edit2world-models/phase1"))
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("GroundingDINO_SwinT_OGC.py", "*SwinT*OGC*.py"):
            candidates.extend(root.rglob(pattern))
    candidates = sorted({p.resolve() for p in candidates})
    if not candidates:
        raise FileNotFoundError("GroundingDINO config not found in site-packages/groundingdino or /data/cwx/edit2world-models/phase1")
    return candidates[0]


def detect_bbox(
    dino_model: Any,
    image_path: Path,
    target_ref: str,
    *,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
) -> tuple[tuple[int, int, int, int], float]:
    import torch
    from groundingdino.util.inference import load_image, predict

    image_source, image = load_image(str(image_path))
    boxes, logits, _phrases = predict(
        model=dino_model,
        image=image,
        caption=target_ref,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if len(boxes) == 0:
        raise RuntimeError(f"GroundingDINO returned no boxes for prompt {target_ref!r}")
    best_idx = int(torch.argmax(logits).item())
    height, width = image_source.shape[:2]
    bbox = cxcywh_norm_to_xyxy_pixels(boxes[best_idx].detach().cpu().numpy(), width, height)
    confidence = float(logits[best_idx].detach().cpu().item())
    return bbox, confidence


def resolve_sam2_config_for_api(config_path: Path, sam2_repo: Path) -> str:
    """SAM2's Hydra compose expects package-relative config names."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        return str(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"SAM2 absolute config not found: {config_path}")
    package_root = Path(sam2_repo) / "sam2"
    try:
        return config_path.resolve().relative_to(package_root.resolve()).as_posix()
    except ValueError:
        return str(config_path)


def sam2_masks_from_box(
    frame_dir: Path,
    anchor_idx: int,
    bbox: tuple[int, int, int, int],
    *,
    sam2_repo: Path,
    sam2_checkpoint: Path,
    sam2_config: Path,
) -> dict[int, np.ndarray]:
    """Remove-side grounding: SAM2 video propagation from a bbox on the original frames."""
    import torch

    sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    sam2_config_name = resolve_sam2_config_for_api(sam2_config, sam2_repo)
    predictor = build_sam2_video_predictor(sam2_config_name, str(sam2_checkpoint), device="cuda")
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


def sam2_primary_from_edited_frame(
    edited_first_frame: Path,
    *,
    point: list[float] | None = None,
    bbox: list[float] | None = None,
    out_dir: Path,
    frame_num: int,
    fps: float,
    sam2_repo: Path,
    sam2_checkpoint: Path,
    sam2_config: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add-side grounding: SAM2 on the edited first frame (repeated as a short clip),
    seeded by the planner-provided insertion point/bbox. Reuses the tested
    `build_quadmask_from_spec.sam2_propagate` SAM2 utility (not the v6 planner)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_quadmask_from_spec import sam2_propagate
    from PIL import Image as _Image
    import argparse as _argparse

    if point is None and bbox is None:
        raise ValueError("add grounding requires a primary point or bbox")
    edited = np.asarray(_Image.open(edited_first_frame).convert("RGB"))
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_path = out_dir / "edited_first_frame_sam2_clip.mp4"
    import imageio.v2 as imageio

    imageio.mimwrite(str(clip_path), [edited.copy() for _ in range(frame_num)], fps=fps, codec="libx264", quality=8, macro_block_size=1)
    propagation_spec = {"primary": {"first_frame_bbox": bbox, "point": point, "negative_points": []}}
    ns = _argparse.Namespace(sam2_repo=sam2_repo, sam2_ckpt=sam2_checkpoint, sam2_cfg=resolve_sam2_config_for_api(sam2_config, sam2_repo))
    propagated = sam2_propagate(ns, clip_path, propagation_spec)
    primary = propagated[0].astype(bool)
    info = {
        "used_for_quadmask": True,
        "input_image": str(edited_first_frame),
        "clip_path": str(clip_path),
        "prompt_source": "planner_model_grounding",
        "bbox_xyxy": bbox,
        "point_xy": point,
        "primary_mask_shape": list(primary.shape),
        "primary_mask_area": int(primary.sum()),
    }
    return primary, info


# --------------------------------------------------------------------------- #
# Quadmask / generation mask
# --------------------------------------------------------------------------- #
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
    """Build the [T,H,W] quadmask (Q0=0 target/insertion, Q2=127 affected, Q3=255 keep)
    from per-frame primary masks, plus a full-domain generation mask."""
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
    generation_mask = build_generation_mask(frame_count, height, width)
    return quad, generation_mask


def build_generation_mask(frame_count: int, height: int, width: int) -> np.ndarray:
    return np.full((frame_count, height, width), 255, dtype=np.uint8)


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


def next_vace_frame_count(frame_count: int) -> int:
    remainder = frame_count % 4
    return frame_count if remainder == 1 else frame_count + ((1 - remainder) % 4)


def pad_time_axis(array: np.ndarray, target_frame_count: int) -> np.ndarray:
    if array.shape[0] == target_frame_count:
        return array
    if array.shape[0] > target_frame_count:
        raise ValueError(f"cannot pad array with T={array.shape[0]} down to {target_frame_count}")
    tail = np.repeat(array[-1:], target_frame_count - array.shape[0], axis=0)
    return np.concatenate([array, tail], axis=0)


# --------------------------------------------------------------------------- #
# Conditioning video (edited frame 0 + zero-filled future frames) + masks video
# --------------------------------------------------------------------------- #
def build_conditioning_video(
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


# --------------------------------------------------------------------------- #
# VACE backend invocation (operation-parameterized). The backend CLI still uses
# legacy arg names (--src_video, --prompt) internally; the E2W-level inputs are
# vace_conditioning_video and vace_prompt. The original source video is never
# passed here.
# --------------------------------------------------------------------------- #
def run_vace(
    operation: str,
    video_id: str,
    *,
    python: Path,
    vace_repo: Path,
    vace_ckpt: Path,
    model_name: str,
    size: str,
    conditioning_video: Path,
    generation_mask_video: Path,
    quadmask_npy: Path,
    vace_prompt: str,
    frame_num: int,
    save_dir: Path,
    output_video: Path,
    base_seed: int,
    sample_steps: int,
    control_branch_checkpoint: Path | None = None,
) -> tuple[str | None, dict[str, Any]]:
    if operation not in ("remove", "add"):
        raise ValueError(f"operation must be 'remove' or 'add', got {operation!r}")
    command = [
        str(Path(python).resolve()),
        str((Path(__file__).resolve().parent / "run_wan_vace_quad_i2v.py").resolve()),
        "--vace_repo",
        str(Path(vace_repo).resolve()),
        "--ckpt_dir",
        str(Path(vace_ckpt).resolve()),
        "--model_name",
        model_name,
        "--size",
        size,
        "--src_video",
        str(Path(conditioning_video).resolve()),
        "--generation_mask",
        str(Path(generation_mask_video).resolve()),
        "--quadmask_npy",
        str(Path(quadmask_npy).resolve()),
        "--operation",
        operation,
        "--prompt",
        vace_prompt,
        "--frame_num",
        str(frame_num),
        "--save_dir",
        str(Path(save_dir).resolve()),
        "--save_file",
        str(Path(output_video).resolve()),
        "--base_seed",
        str(base_seed),
        "--sample_steps",
        str(sample_steps),
        "--offload_model",
    ]
    if control_branch_checkpoint:
        command.extend(["--control_branch_checkpoint", str(Path(control_branch_checkpoint).resolve())])
    save_dir = Path(save_dir)
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
    info: dict[str, Any] = {
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
    return str(Path(output_video)), info


def control_branch_info_from_context(vace_info: dict[str, Any]) -> dict[str, Any]:
    context_info = vace_info.get("context_info") if isinstance(vace_info.get("context_info"), dict) else {}
    return {
        "control_branch_checkpoint_loaded": bool(context_info.get("control_branch_checkpoint_loaded")),
        "trained_control_branch_used": bool(context_info.get("trained_control_branch_used")),
        "control_branch_step": context_info.get("control_branch_step"),
        "control_branch_gate": context_info.get("control_branch_gate"),
        "control_branch_installed_in_forward_vace": bool(context_info.get("control_branch_installed_in_forward_vace")),
    }


# --------------------------------------------------------------------------- #
# Debug previews (for manual review of intermediate artifacts)
# --------------------------------------------------------------------------- #
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

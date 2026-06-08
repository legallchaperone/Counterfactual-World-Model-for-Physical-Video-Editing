#!/usr/bin/env python3
"""Debug v8 planner -> GroundingDINO -> SAM2 propagation on eval samples."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import validate_planner_output_v8  # noqa: E402


DEFAULT_EVAL = Path("/data/cwx/E2W/data/planner_sft_v8/eval.jsonl")
DEFAULT_BASE_MODEL = Path("/data/cwx/Edit2World-unified/checkpoints/Qwen2.5-VL-7B-Instruct")
DEFAULT_ADAPTER = Path("/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3")
DEFAULT_OUT_DIR = Path("/data/cwx/E2W/runs/e2w_v8_pipeline_debug")
DEFAULT_DINO_CKPT = Path("/data/cwx/edit2world-models/phase1/groundingdino_swint_ogc.pth")
DEFAULT_SAM2_REPO = Path("/data/cwx/Edit2World-unified/external/sam2")
DEFAULT_SAM2_CKPT = Path("/data/cwx/Edit2World-unified/checkpoints/sam2/sam2.1_hiera_large.pt")
DEFAULT_SAM2_CFG = Path("/data/cwx/Edit2World-unified/external/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")


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
            schema_valid, validation_error = validate_planner_output_v8(parsed, source_video_id=str(row.get("video_id") or "unknown"))
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
            record.update(
                {
                    "status": "ok",
                    "bbox_xyxy": list(bbox),
                    "bbox_confidence": confidence,
                    "mask_mean_area": float(np.mean(mask_areas)),
                    "debug_grounding": str(grounding_path),
                    "debug_masks": mask_debug_paths,
                    "frame_count": len(frames),
                    "mask_frame_count": len(masks),
                }
            )
            print(
                f"{video_id} | {target_ref[:40]} | "
                f"bbox({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) | "
                f"{confidence:.4f} | {float(np.mean(mask_areas)):.6f}",
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

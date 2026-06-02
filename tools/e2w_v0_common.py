#!/usr/bin/env python3
"""Shared helpers for the E2W v0 Physics-IQ integration freeze."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_RUN_DIR = Path("/data/cwx/E2W/runs/e2w_v0_physics_iq")
DEFAULT_SPLIT = Path("/data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_eval.jsonl")
DEFAULT_PLANNER = Path("/data/cwx/E2W/checkpoints/vlm_planner_lora_physics_iq_v5_split_eval")
DEFAULT_BASE_MODEL = Path("/data/cwx/Edit2World-unified/checkpoints/Qwen2.5-VL-7B-Instruct")
DEFAULT_TEACHER_DIR = Path("/data/cwx/E2W/data/physics_iq_vlm_sft/teacher_labels/parsed")

RUN_DIRS = [
    "source",
    "planner_pred",
    "masks",
    "first_frame",
    "vace",
    "judge",
    "contact_sheets",
]

FORBIDDEN_VACE_WORDS = {
    "remove": "make absent",
    "removed": "absent",
    "removing": "making absent",
    "delete": "make absent",
    "deleted": "absent",
    "erase": "make absent",
    "erased": "absent",
}
TARGET_TERM_SPLIT_RE = re.compile(
    r"\b(?:of|with|containing|inside|filled with|holding|attached to)\b",
    flags=re.I,
)
COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "clear",
    "gray",
    "grey",
    "green",
    "orange",
    "purple",
    "red",
    "white",
    "yellow",
}
WEAK_MODIFIERS = {
    "bright",
    "clear",
    "dark",
    "light",
    "shallow",
    "small",
    "tall",
}
TARGET_HEAD_NOUNS = {
    "arm",
    "ball",
    "balloon",
    "board",
    "candle",
    "container",
    "cup",
    "dish",
    "duck",
    "frame",
    "glass",
    "liquid",
    "mug",
    "paper",
    "plate",
    "rod",
    "spotlight",
    "stick",
    "table",
    "tube",
    "water",
}
QUANTITY_PREFIX_RE = re.compile(
    r"^(?:"
    r"\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"a|an|both|all|multiple|several|many|few"
    r")\b",
    flags=re.I,
)
SUPPORTED_OPERATIONS = {"remove", "add"}
REMOVE_OPERATION_RE = re.compile(
    r"\b(?:remove|delete|erase|take\s+away|make\s+absent|get\s+rid\s+of)\b",
    flags=re.I,
)
ADD_OPERATION_RE = re.compile(
    r"\b(?:add|insert|place|put|create|introduce|include)\b",
    flags=re.I,
)


class VacePromptContractError(ValueError):
    """Raised when planner text cannot produce a contract-safe VACE prompt."""


def ensure_run_dirs(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for rel in RUN_DIRS:
        (run_dir / rel).mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def video_meta(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


def read_video_rgb(video_path: Path, max_frames: int | None = None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 16.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def write_rgb_video(path: Path, frames: list[np.ndarray], fps: float = 16.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to write")
    imageio.mimwrite(str(path), frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)


def write_gray_video(path: Path, masks: np.ndarray, fps: float = 16.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = masks.astype(np.uint8)
    frames = [np.repeat(frame[:, :, None], 3, axis=2) for frame in arr]
    imageio.mimwrite(str(path), frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)


def extract_first_frame(video_path: Path, out_path: Path) -> np.ndarray:
    frames, _ = read_video_rgb(video_path, max_frames=1)
    if not frames:
        raise RuntimeError(f"No frames in {video_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frames[0]).save(out_path)
    return frames[0]


def sample_indices(frame_count: int, n: int = 6) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= n:
        return list(range(frame_count))
    return sorted({int(round(i * (frame_count - 1) / (n - 1))) for i in range(n)})


def extract_json_text(text: str) -> tuple[str | None, str | None]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            _, end = decoder.raw_decode(cleaned[i:])
            return cleaned[i : i + end], None
        except json.JSONDecodeError as exc:
            last_error = str(exc)
    return None, last_error if "last_error" in locals() else "no JSON object found"


def parse_json_output(text: str) -> tuple[dict[str, Any] | None, str | None]:
    json_text, error = extract_json_text(text)
    if json_text is None:
        return None, error
    try:
        return json.loads(json_text), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _strings(value: Any) -> list[str]:
    return [str(x).strip() for x in _as_list(value) if str(x).strip()]


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = re.sub(r"\s+", " ", value.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value.strip())
    return out


def _singularize_word(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _target_term_variants(values: list[str]) -> list[str]:
    variants: list[str] = []
    for value in values:
        normalized = re.sub(r"[^A-Za-z0-9\s-]", " ", value).strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        if not normalized:
            continue
        variants.append(normalized)
        for part in TARGET_TERM_SPLIT_RE.split(normalized):
            part = re.sub(r"\s+", " ", part).strip()
            if not part:
                continue
            variants.append(part)
            tokens = part.split()
            if len(tokens) >= 2 and tokens[0] in WEAK_MODIFIERS:
                without_weak_modifier = " ".join(tokens[1:])
                variants.append(without_weak_modifier)
                weak_tokens = without_weak_modifier.split()
                if len(weak_tokens) >= 2 and weak_tokens[0] in COLOR_WORDS:
                    variants.append(without_weak_modifier)
            if len(tokens) >= 2 and tokens[0] in COLOR_WORDS:
                variants.append(" ".join(tokens))
            head = _singularize_word(tokens[-1]) if tokens else ""
            if head in TARGET_HEAD_NOUNS:
                variants.append(tokens[-1])
                variants.append(head)
    return _dedupe_strings([x for x in variants if len(x) >= 3])


def _object_count_constraint(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return None
    low = normalized.lower()
    if QUANTITY_PREFIX_RE.match(low):
        if low.startswith(("a ", "an ")):
            return "one " + normalized.split(" ", 1)[1]
        return normalized
    if re.search(r"\b(objects|balls|dominoes|pillows|strings|blocks|pieces)\b", low):
        return None
    return f"one {normalized}"


def _object_count_constraints(values: list[str]) -> list[str]:
    return _dedupe_strings([x for item in values if (x := _object_count_constraint(item))])


def infer_target_from_sample(sample: dict[str, Any]) -> str:
    user = sample.get("messages", [{}])[0].get("content", "")
    match = re.search(r"User request:\s*remove\s+(.+?)(?:\.|$)", user, re.I)
    if match:
        return match.group(1).strip()
    return "target object"


def infer_operation_from_text(text: str | None) -> str | None:
    """Infer an edit operation from user-facing text when it is unambiguous."""
    if not text:
        return None
    has_add = ADD_OPERATION_RE.search(text) is not None
    has_remove = REMOVE_OPERATION_RE.search(text) is not None
    if has_add and not has_remove:
        return "add"
    if has_remove and not has_add:
        return "remove"
    return None


def infer_operation_from_sample(sample: dict[str, Any]) -> str | None:
    for message in sample.get("messages", []) or []:
        if message.get("role") in {"user", "system"}:
            inferred = infer_operation_from_text(str(message.get("content") or ""))
            if inferred:
                return inferred
    return infer_operation_from_text(str(sample.get("prompt") or sample.get("text") or ""))


def resolve_expected_operation(
    explicit_operation: str | None,
    sample: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> str | None:
    if explicit_operation and explicit_operation != "auto":
        if explicit_operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f"Unsupported operation: {explicit_operation}")
        return explicit_operation
    if sample is not None:
        inferred = infer_operation_from_sample(sample)
        if inferred:
            return inferred
    if plan is not None:
        for key in ("user_prompt", "scene_summary"):
            inferred = infer_operation_from_text(str(plan.get(key) or ""))
            if inferred:
                return inferred
    return None


def _first_target(raw: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    targets = raw.get("target_objects")
    if isinstance(targets, list) and targets and isinstance(targets[0], dict):
        return targets[0]
    return {"name": infer_target_from_sample(sample), "aliases": []}


def _bbox_from_raw(raw: dict[str, Any], width: int, height: int) -> list[int] | None:
    candidates: list[Any] = []
    q = raw.get("quadmask_spec", {}) if isinstance(raw.get("quadmask_spec"), dict) else {}
    primary = q.get("primary", {}) if isinstance(q.get("primary"), dict) else {}
    target = _first_target(raw, {})
    for key in ("first_frame_bbox", "bbox_first_frame", "bbox_2d", "bbox", "box"):
        candidates.append(primary.get(key))
        if isinstance(target, dict):
            candidates.append(target.get(key))
    for cand in candidates:
        if not isinstance(cand, list) or len(cand) != 4:
            continue
        try:
            box = [int(round(float(x))) for x in cand]
        except (TypeError, ValueError):
            continue
        if box_is_valid(box, width, height):
            return box
    return None


def _point_from_raw(raw: dict[str, Any], bbox: list[int] | None, width: int, height: int) -> list[int] | None:
    q = raw.get("quadmask_spec", {}) if isinstance(raw.get("quadmask_spec"), dict) else {}
    primary = q.get("primary", {}) if isinstance(q.get("primary"), dict) else {}
    for key in ("point", "point_first_frame", "center", "prompt_point"):
        pt = primary.get(key)
        if isinstance(pt, list) and len(pt) == 2:
            try:
                out = [int(round(float(pt[0]))), int(round(float(pt[1])))]
            except (TypeError, ValueError):
                continue
            if point_is_valid(out, width, height):
                return out
    if bbox is not None:
        return [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]
    return None


def normalize_to_e2w_contract(
    raw: dict[str, Any],
    sample: dict[str, Any],
    meta: dict[str, Any],
    source: str = "planner_pred",
    operation: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    op = resolve_expected_operation(operation, sample=sample) or "remove"
    target = _first_target(raw, sample)
    label = str(target.get("name") or infer_target_from_sample(sample)).strip()
    aliases = _strings(target.get("aliases"))
    protected_objects = _strings(raw.get("protected_objects"))
    target_id = "subject_1"

    chain = raw.get("physical_causal_chain", [])
    observed_dynamics: list[str] = []
    if isinstance(chain, list):
        observed_dynamics.extend(str(step.get("description", "")).strip() for step in chain if isinstance(step, dict))
    cfe = raw.get("counterfactual_expectation", {})
    counterfactual_outcomes: list[str] = []
    if isinstance(cfe, dict):
        counterfactual_outcomes.extend(_strings(cfe.get("if_removed")))
    counterfactual_outcomes = _dedupe_strings([x for x in counterfactual_outcomes if x])
    observed_dynamics = _dedupe_strings([x for x in observed_dynamics if x])

    scene_caption = ""
    outcome_effects: list[str] = []
    if isinstance(cfe, dict):
        scene_caption = str(cfe.get("if_removed") or "").strip()
        outcome_effects.extend(counterfactual_outcomes)
    preserve_regions = _strings(cfe.get("unchanged_regions") if isinstance(cfe, dict) else [])
    if op == "add":
        local_fill_instruction = "Place the target object naturally in the specified local area while preserving neighboring pixels."
        role = "add_target"
        user_prompt = f"add {label}"
    else:
        local_fill_instruction = "Fill target pixels with plausible local background consistent with neighboring pixels."
        role = "remove_target"
        user_prompt = f"remove {label}"

    edit_plan = {
        "schema_version": "e2w.edit_plan.v1",
        "operation": op,
        "user_prompt": user_prompt,
        "scene_summary": str(raw.get("event_summary") or "").strip(),
        "observed_dynamics": [x for x in observed_dynamics[:3]],
        "edit_subject": {
            "id": target_id,
            "label": label,
            "aliases": aliases,
            "role": role,
            "scope": "single_object",
            "visual_descriptor": str(target.get("location_description") or label),
            "included_parts": [label],
            "excluded_non_target_parts": protected_objects,
        },
        "operation_details": {
            "target_object": {"id": target_id, "label": label},
            "added_object": {},
            "placement_reference": {},
            "object_interactions": _strings(cfe.get("affected_regions") if isinstance(cfe, dict) else []),
            "protected_objects": protected_objects,
            "physical_consequences": counterfactual_outcomes,
            "preserve_regions": preserve_regions,
            "local_fill_instruction": local_fill_instruction,
            "expected_background": "; ".join(preserve_regions) or "background consistent with surrounding scene",
            "visual_effects_to_remove": [],
            "visual_effects_to_add": [],
        },
        "edited_scene": {
            "caption": scene_caption,
            "outcome_effects": outcome_effects,
            "preserve": preserve_regions,
        },
        "source": source,
        "source_video_id": sample.get("id") or raw.get("video_id"),
    }

    width, height = int(meta["width"]), int(meta["height"])
    bbox = _bbox_from_raw(raw, width, height)
    point = _point_from_raw(raw, bbox, width, height)
    raw_q = raw.get("quadmask_spec", {}) if isinstance(raw.get("quadmask_spec"), dict) else {}
    raw_affected = raw_q.get("affected", {}) if isinstance(raw_q.get("affected"), dict) else {}
    affected_labels = _strings(raw_affected.get("objects"))
    if isinstance(cfe, dict):
        affected_labels.extend(_strings(cfe.get("affected_regions")))

    spec = {
        "schema_version": "e2w.quadmask_spec.v1",
        "operation": op,
        "source": source,
        "grid": {
            "rows": 14,
            "cols": 8,
            "frame_count": int(meta["frame_count"]),
            "reference_frame_indices": sample_indices(int(meta["frame_count"]), n=6),
        },
        "primary": {
            "label": label,
            "source": "planner_grounding",
            "first_frame_bbox": bbox,
            "point": point,
            "negative_points": [],
            "prompt": label,
            "grid_timeline": [],
        },
        "affected_regions": [
            {
                "id": f"affected_{i+1}",
                "label": region,
                "category": "planner_text",
                "reason": "predicted by planner",
                "grid_timeline": [],
                "grid_boxes": [],
                "will_move": True,
                "movement_description": region,
            }
            for i, region in enumerate(dict.fromkeys(affected_labels))
        ],
        "keep": raw_q.get("keep", {}) if isinstance(raw_q.get("keep"), dict) else {},
    }
    return edit_plan, spec


def box_is_valid(box: list[int] | None, width: int, height: int) -> bool:
    if box is None or len(box) != 4:
        return False
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def point_is_valid(point: list[int] | None, width: int, height: int) -> bool:
    if point is None or len(point) != 2:
        return False
    x, y = point
    return 0 <= x < width and 0 <= y < height


def norm1000_box_to_pixel(box: list[Any] | None, width: int, height: int) -> list[int] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(x) for x in box]
    except (TypeError, ValueError):
        return None
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    out = [
        int(round(x1 / 1000 * width)),
        int(round(y1 / 1000 * height)),
        int(round(x2 / 1000 * width)),
        int(round(y2 / 1000 * height)),
    ]
    return out if box_is_valid(out, width, height) else None


def norm1000_point_to_pixel(point: list[Any] | None, width: int, height: int) -> list[int] | None:
    if not isinstance(point, list) or len(point) != 2:
        return None
    try:
        x, y = [float(v) for v in point]
    except (TypeError, ValueError):
        return None
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    out = [int(round(x / 1000 * width)), int(round(y / 1000 * height))]
    return out if point_is_valid(out, width, height) else None


def primary_grounding_from_spec(spec: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    width, height = int(meta["width"]), int(meta["height"])
    primary = spec.get("primary", {}) if isinstance(spec.get("primary"), dict) else {}
    bbox = primary.get("first_frame_bbox")
    point = primary.get("point")
    negative_points = primary.get("negative_points") or []
    frame_index = 0
    coordinate_source = "pixel"

    if not box_is_valid(bbox, width, height):
        bbox = None
    if not point_is_valid(point, width, height):
        point = None

    keyframes = primary.get("keyframes", [])
    if isinstance(keyframes, list) and keyframes:
        keyframe0 = keyframes[0] if isinstance(keyframes[0], dict) else {}
        frame_index = int(keyframe0.get("frame_index") or 0)
        norm_box = norm1000_box_to_pixel(keyframe0.get("bbox_xyxy_norm1000"), width, height)
        if norm_box is not None:
            bbox = norm_box
            coordinate_source = "norm1000"
        points = keyframe0.get("positive_points_norm1000")
        if isinstance(points, list) and points:
            norm_point = norm1000_point_to_pixel(points[0], width, height)
            if norm_point is not None:
                point = norm_point
        neg = []
        for item in keyframe0.get("negative_points_norm1000", []) or []:
            neg_point = norm1000_point_to_pixel(item, width, height)
            if neg_point is not None:
                neg.append(neg_point)
        negative_points = neg

    if point is None and bbox is not None:
        point = [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]

    return {
        "bbox": bbox,
        "point": point,
        "negative_points": negative_points,
        "frame_index": frame_index,
        "coordinate_source": coordinate_source,
    }


def _cell_is_valid(cell: str, rows: int, cols: int) -> bool:
    match = re.match(r"^([A-Za-z]+)([1-9][0-9]*)$", str(cell).strip())
    if not match:
        return False
    letters, number = match.groups()
    row = 0
    for ch in letters.upper():
        row = row * 26 + (ord(ch) - ord("A") + 1)
    col = int(number)
    return 1 <= row <= rows and 1 <= col <= cols


def validate_quadmask_spec_executor(spec: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    width, height = int(meta["width"]), int(meta["height"])
    frame_count = int(meta["frame_count"])
    primary = spec.get("primary", {}) if isinstance(spec.get("primary"), dict) else {}
    grounding = primary_grounding_from_spec(spec, meta)
    bbox = grounding["bbox"]
    point = grounding["point"]

    affected = spec.get("affected", {}) if isinstance(spec.get("affected"), dict) else {}
    grid_shape = affected.get("grid_shape")
    if isinstance(grid_shape, list) and len(grid_shape) == 2:
        try:
            rows, cols = int(grid_shape[0]), int(grid_shape[1])
        except (TypeError, ValueError):
            rows, cols = 0, 0
    else:
        grid = spec.get("grid", {}) if isinstance(spec.get("grid"), dict) else {}
        rows, cols = int(grid.get("rows") or 0), int(grid.get("cols") or 0)

    affected_grid_seen = False
    affected_grid_valid = True
    frame_ranges_valid = True

    frame_ranges = affected.get("frame_ranges", [])
    if isinstance(frame_ranges, list) and frame_ranges:
        affected_grid_seen = True
        for item in frame_ranges:
            if not isinstance(item, dict):
                affected_grid_valid = False
                continue
            start = item.get("start_frame")
            end = item.get("end_frame")
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start or end >= frame_count:
                frame_ranges_valid = False
            cells = item.get("cells")
            if not isinstance(cells, list) or not cells:
                affected_grid_valid = False
                continue
            if rows <= 0 or cols <= 0:
                affected_grid_valid = False
                continue
            for cell in cells:
                if not _cell_is_valid(str(cell), rows, cols):
                    affected_grid_valid = False

    affected_regions = spec.get("affected_regions", [])
    if isinstance(affected_regions, list):
        for region in affected_regions:
            if not isinstance(region, dict):
                continue
            grid_boxes = region.get("grid_boxes", [])
            if isinstance(grid_boxes, list) and grid_boxes:
                affected_grid_seen = True

    checks = {
        "has_primary": isinstance(primary, dict) and bool(primary),
        "has_bbox": bbox is not None,
        "bbox_in_range": box_is_valid(bbox, width, height),
        "has_point": point is not None,
        "point_in_range": point_is_valid(point, width, height),
        "has_affected_grid": affected_grid_seen,
        "affected_grid_valid": affected_grid_seen and affected_grid_valid,
        "frame_ranges_valid": frame_ranges_valid,
    }
    failure = None
    if not checks["has_bbox"]:
        failure = "missing_primary_bbox"
    elif not checks["bbox_in_range"]:
        failure = "invalid_coordinate_space"
    elif not checks["has_point"]:
        failure = "missing_primary_point"
    elif not checks["point_in_range"]:
        failure = "invalid_coordinate_space"
    elif not checks["has_affected_grid"]:
        failure = "affected_grid_empty"
    elif not checks["affected_grid_valid"]:
        failure = "affected_grid_invalid"
    elif not checks["frame_ranges_valid"]:
        failure = "affected_grid_invalid"
    return {
        **checks,
        "executor_valid": failure is None,
        "executor_failure": failure,
    }


def validate_edit_plan(plan: dict[str, Any], expected_operation: str | None = None) -> dict[str, Any]:
    details = plan.get("operation_details", {})
    scene = plan.get("edited_scene", {})
    expected = resolve_expected_operation(expected_operation, plan=plan)
    if expected is None:
        expected = "remove"
    actual = str(plan.get("operation") or "").strip()
    return {
        "schema_valid": all(k in plan for k in ["schema_version", "operation", "edit_subject", "operation_details", "edited_scene"]),
        "operation_accuracy": actual == expected,
        "expected_operation": expected,
        "actual_operation": actual,
        "physical_consequences_nonempty": bool(_strings(details.get("physical_consequences") if isinstance(details, dict) else [])),
        "edited_scene_caption_nonempty": bool(str(scene.get("caption") if isinstance(scene, dict) else "").strip()),
        "edited_scene_outcome_effects_nonempty": bool(_strings(scene.get("outcome_effects") if isinstance(scene, dict) else [])),
    }


def validate_quadmask_spec(spec: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    width, height = int(meta["width"]), int(meta["height"])
    frame_count = int(meta["frame_count"])
    primary = spec.get("primary", {}) if isinstance(spec.get("primary"), dict) else {}
    grid = spec.get("grid", {}) if isinstance(spec.get("grid"), dict) else {}
    rows = int(grid.get("rows") or 0)
    cols = int(grid.get("cols") or 0)

    grounding = primary_grounding_from_spec(spec, meta)
    bbox_ok = box_is_valid(grounding["bbox"], width, height)
    point_ok = point_is_valid(grounding["point"], width, height)
    prompt_ok = bool(str(primary.get("prompt") or primary.get("label") or primary.get("object_name") or "").strip())

    affected = spec.get("affected_regions", [])
    grid_boxes_seen = 0
    grid_boxes_valid = 0
    frame_indices_valid = True
    if isinstance(affected, list):
        for region in affected:
            if not isinstance(region, dict):
                continue
            for item in region.get("grid_boxes", []) or []:
                if not isinstance(item, dict):
                    continue
                grid_boxes_seen += 1
                idx = item.get("frame_index")
                box = item.get("box")
                if not isinstance(idx, int) or idx < 0 or idx >= frame_count:
                    frame_indices_valid = False
                if (
                    isinstance(box, list)
                    and len(box) == 4
                    and rows > 0
                    and cols > 0
                    and 0 <= int(box[0]) < int(box[2]) <= rows
                    and 0 <= int(box[1]) < int(box[3]) <= cols
                ):
                    grid_boxes_valid += 1

    coordinate_range_valid = bbox_ok or point_ok
    affected_grid_valid = grid_boxes_seen == 0 or grid_boxes_seen == grid_boxes_valid
    executable = prompt_ok and coordinate_range_valid and affected_grid_valid and frame_indices_valid
    executor_checks = validate_quadmask_spec_executor(spec, meta)
    schema_ok = all(k in spec for k in ["schema_version", "operation", "grid", "primary", "affected_regions"])
    schema_ok = schema_ok or all(k in spec for k in ["schema_version", "primary", "affected"])
    return {
        "quadmask_schema_valid": schema_ok,
        "primary_prompt_valid": prompt_ok,
        "primary_bbox_valid": bbox_ok,
        "primary_point_valid": point_ok,
        "affected_grid_valid": affected_grid_valid,
        "affected_grid_boxes_seen": grid_boxes_seen,
        "affected_grid_boxes_valid": grid_boxes_valid,
        "frame_index_valid": frame_indices_valid,
        "coordinate_range_valid": coordinate_range_valid,
        "quadmask_spec_executable": executable,
        "quadmask_spec_executor_valid": executor_checks["executor_valid"],
        **executor_checks,
    }


def summarize_boolean_metrics(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows)}
    for key in keys:
        vals = [bool(r.get(key)) for r in rows]
        out[key] = {
            "count": sum(vals),
            "rate": (sum(vals) / len(vals)) if vals else 0.0,
        }
    return out


def serialize_vace_prompt(edit_plan: dict[str, Any]) -> str:
    scene = edit_plan.get("edited_scene", {}) if isinstance(edit_plan.get("edited_scene"), dict) else {}
    details = edit_plan.get("operation_details", {}) if isinstance(edit_plan.get("operation_details"), dict) else {}
    operation = str(edit_plan.get("operation") or "remove").strip().lower()
    caption = str(scene.get("caption") or "").strip()
    outcomes = _strings(scene.get("outcome_effects"))
    consequences = _strings(details.get("physical_consequences"))
    subject = edit_plan.get("edit_subject", {}) if isinstance(edit_plan.get("edit_subject"), dict) else {}
    target = details.get("target_object", {}) if isinstance(details.get("target_object"), dict) else {}
    target_terms = _target_term_variants(
        _strings(subject.get("label"))
        + _strings(subject.get("aliases"))
        + _strings(subject.get("included_parts"))
        + _strings(target.get("label"))
    )
    protected_objects = _dedupe_strings(
        _strings(details.get("protected_objects")) + _strings(subject.get("excluded_non_target_parts"))
    )
    protected_count_pairs = [
        (item, counted)
        for item in protected_objects
        if (counted := _object_count_constraint(item)) and counted.lower() != item.lower()
    ]

    def preserve_case(replacement: str, matched: str) -> str:
        return replacement[:1].upper() + replacement[1:] if matched[:1].isupper() else replacement

    def apply_count_wording(text: str) -> str:
        out = text
        for item, counted in protected_count_pairs:
            pattern = rf"(?<!\bone\s)\b(?:the\s+)?{re.escape(item)}\b"
            out = re.sub(pattern, lambda m: preserve_case(counted, m.group(0)), out, flags=re.I)
        return out

    def mentions_target(text: str) -> bool:
        if operation == "add":
            return False
        low = text.lower()
        return any(len(term) >= 3 and re.search(rf"\b{re.escape(term.lower())}\b", low) for term in target_terms)

    def split_fragments(text: str) -> list[str]:
        return [fragment.strip() for fragment in re.split(r"(?<=[.!?])\s+", text.strip()) if fragment.strip()]

    target_hits: list[dict[str, str]] = []
    for source_name, values in [
        ("edited_scene.caption", [caption]),
        ("edited_scene.outcome_effects", outcomes),
        ("operation_details.physical_consequences", consequences),
    ]:
        for value in values:
            for fragment in split_fragments(value):
                if mentions_target(fragment):
                    target_hits.append({"source": source_name, "fragment": fragment})
    if target_hits:
        sample_id = edit_plan.get("source_video_id") or "unknown"
        preview = "; ".join(f"{hit['source']}: {hit['fragment']}" for hit in target_hits[:3])
        raise VacePromptContractError(
            "VACE prompt contract violation: target-contaminated planner text "
            f"for sample {sample_id}. Regenerate or fix the planner label; "
            f"do not silently drop or fallback. Hits: {preview}"
        )

    def clean(text: str) -> str:
        out = text
        for bad, repl in FORBIDDEN_VACE_WORDS.items():
            out = re.sub(rf"\b{re.escape(bad)}\b", repl, out, flags=re.I)
        if operation != "add":
            for term in sorted(target_terms, key=len, reverse=True):
                if len(term) < 3:
                    continue
                out = re.sub(rf"\b{re.escape(term)}\b", "the edited subject", out, flags=re.I)
        out = apply_count_wording(out)
        return out.strip()

    def looks_like_region_fragment(text: str) -> bool:
        low = text.strip().lower()
        if not low:
            return True
        spatial_starts = (
            "area ",
            "the area ",
            "central area",
            "left side",
            "right side",
            "lower portion",
            "shadow on",
            "highlights on",
            "overall scene brightness",
            "trajectory of",
            "strings",
        )
        if low.startswith(spatial_starts):
            return True
        if re.search(r"\bwhere\b.+\b(was|were)\b.+\b(located|previously|occupied)\b", low):
            return True
        return False

    def vace_lines(values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            for fragment in split_fragments(value):
                if mentions_target(fragment):
                    raise VacePromptContractError(f"Unexpected target term after precheck: {fragment}")
                if looks_like_region_fragment(fragment):
                    continue
                line = clean(fragment)
                if line:
                    cleaned.append(line)
        return _dedupe_strings(cleaned)

    caption_lines = vace_lines([caption])
    safe_outcomes = vace_lines(outcomes)
    safe_consequences = vace_lines(consequences)

    if caption_lines:
        safe_caption = " ".join(caption_lines)
    elif safe_outcomes:
        safe_caption = safe_outcomes.pop(0)
    elif safe_consequences:
        safe_caption = safe_consequences.pop(0)
    else:
        sample_id = edit_plan.get("source_video_id") or "unknown"
        raise VacePromptContractError(
            "VACE prompt contract violation: no target-free semantic line "
            f"for sample {sample_id}; refusing neutral fallback."
        )

    def line_key(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    caption_key = line_key(safe_caption)
    safe_outcomes = [x for x in safe_outcomes if line_key(x) != caption_key and line_key(x) not in caption_key]
    outcome_keys = {line_key(x) for x in safe_outcomes}
    safe_consequences = [
        x
        for x in safe_consequences
        if line_key(x) not in outcome_keys and line_key(x) != caption_key and line_key(x) not in caption_key
    ]

    lines = ["Scene after editing:", safe_caption]
    if safe_outcomes:
        lines.extend(["", "Expected outcome:"])
        lines.extend(f"- {x}" for x in safe_outcomes)
    if safe_consequences:
        lines.extend(["", "Physical consequences:"])
        lines.extend(f"- {x}" for x in safe_consequences)
    count_sources = protected_objects
    if operation == "add":
        count_sources = _dedupe_strings(_strings(subject.get("label")) + count_sources)
    count_constraints = _object_count_constraints(count_sources)
    if count_constraints:
        lines.extend(["", "Visible non-target object counts to preserve:"])
        lines.extend(f"- {x}" for x in count_constraints)
    lines.extend(
        [
            "",
            "Maintain natural temporal motion, camera consistency, lighting, scene identity, and existing object layout. Do not introduce people, hands, or unrelated new objects.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def serialize_first_frame_prompt(edit_plan: dict[str, Any]) -> str:
    subject = edit_plan.get("edit_subject", {}) if isinstance(edit_plan.get("edit_subject"), dict) else {}
    details = edit_plan.get("operation_details", {}) if isinstance(edit_plan.get("operation_details"), dict) else {}
    scene = edit_plan.get("edited_scene", {}) if isinstance(edit_plan.get("edited_scene"), dict) else {}
    target = details.get("target_object", {}) if isinstance(details.get("target_object"), dict) else {}
    operation = str(edit_plan.get("operation") or "remove").strip().lower()
    target_label = str(subject.get("label") or target.get("label") or "the specified target object").strip()
    target_phrase = target_label if target_label.lower().startswith(("the ", "a ", "an ")) else f"the {target_label}"
    aliases = [x for x in _strings(subject.get("aliases")) if x.lower() != target_label.lower()]
    visual_descriptor = str(subject.get("visual_descriptor") or "").strip()

    def prompt_items(*values: Any) -> list[str]:
        parts: list[str] = []
        for value in values:
            for text in _strings(value):
                parts.extend(x.strip() for x in re.split(r"\s*;\s*", text) if x.strip())
        return _dedupe_strings(parts)

    preserve_regions = prompt_items(details.get("preserve_regions"), scene.get("preserve"))
    if not preserve_regions:
        preserve_regions = prompt_items(details.get("expected_background"))
    protected_objects = prompt_items(details.get("protected_objects"), subject.get("excluded_non_target_parts"))
    if operation == "add":
        effects = _strings(details.get("visual_effects_to_add"))
        local_fill_instruction = str(details.get("local_fill_instruction") or "").strip()
        if not local_fill_instruction:
            local_fill_instruction = f"Place {target_phrase} naturally in the scene with plausible contact, scale, lighting, and shadows."
        lines = [
            f"Add {target_phrase} to this image.",
        ]
    else:
        effects = _strings(details.get("visual_effects_to_remove"))
        local_fill_instruction = str(details.get("local_fill_instruction") or "").strip()
        if not local_fill_instruction:
            local_fill_instruction = "Fill only the pixels where the target object was with plausible local background consistent with neighboring pixels."
        lines = [
            f"Remove only {target_phrase} from this image.",
        ]
    if aliases:
        lines.append(f"Target aliases: {'; '.join(aliases)}.")
    if visual_descriptor and visual_descriptor.lower() != target_label.lower():
        lines.append(f"Target visual description: {visual_descriptor.rstrip('.')}.")
    lines.append(local_fill_instruction.rstrip(".") + ".")
    if effects and operation == "remove":
        lines.append(f"Also remove local visual effects if visible: {'; '.join(effects)}.")
    elif effects:
        lines.append(f"Also add local visual effects if needed: {'; '.join(effects)}.")
    if protected_objects:
        lines.append(f"Protected non-target objects, do not remove or alter: {'; '.join(protected_objects)}.")
    if preserve_regions:
        lines.append(f"Non-target content to keep visible and unchanged: {'; '.join(preserve_regions)}.")
    lines.append("Preserve camera perspective, lighting continuity, and object layout outside the target area.")
    return "\n".join(lines).strip() + "\n"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_manifest(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def quadmask_to_rgb(quadmask: np.ndarray) -> np.ndarray:
    colors = {
        0: np.array([255, 40, 40], dtype=np.uint8),
        63: np.array([255, 180, 40], dtype=np.uint8),
        127: np.array([60, 120, 255], dtype=np.uint8),
        255: np.array([25, 25, 25], dtype=np.uint8),
    }
    out = np.zeros((*quadmask.shape, 3), dtype=np.uint8)
    for value, color in colors.items():
        out[quadmask == value] = color
    return out


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = frame.copy()
    if mask.dtype != bool:
        mask = mask > 0
    col = np.array(color, dtype=np.float32)
    out[mask] = np.clip(out[mask].astype(np.float32) * (1 - alpha) + col * alpha, 0, 255).astype(np.uint8)
    return out


def make_debug_contact_sheet(
    out_path: Path,
    frames: list[np.ndarray],
    primary: np.ndarray,
    affected: np.ndarray,
    quadmask: np.ndarray,
    max_cols: int = 6,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indices = sample_indices(len(frames), max_cols)
    if not indices:
        return
    thumbs: list[list[Image.Image]] = []
    for row_name, maker in [
        ("original", lambda i: frames[i]),
        ("primary", lambda i: overlay_mask(frames[i], primary[i], (255, 30, 30))),
        ("affected", lambda i: overlay_mask(frames[i], affected[i], (40, 120, 255))),
        ("quadmask", lambda i: quadmask_to_rgb(quadmask[i])),
    ]:
        row_imgs: list[Image.Image] = []
        for i in indices:
            img = Image.fromarray(maker(i)).resize((240, 135))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, 100, 18], fill=(0, 0, 0))
            draw.text((4, 3), f"{row_name} {i}", fill=(255, 255, 255))
            row_imgs.append(img)
        thumbs.append(row_imgs)
    width = 240 * len(indices)
    height = 135 * len(thumbs)
    sheet = Image.new("RGB", (width, height), (20, 20, 20))
    for r, row in enumerate(thumbs):
        for c, img in enumerate(row):
            sheet.paste(img, (c * 240, r * 135))
    sheet.save(out_path)

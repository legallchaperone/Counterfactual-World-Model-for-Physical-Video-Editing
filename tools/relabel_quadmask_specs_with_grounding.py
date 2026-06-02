#!/usr/bin/env python3
"""Relabel Physics-IQ planner SFT rows with executable grounding fields."""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import (  # noqa: E402
    DEFAULT_RUN_DIR,
    load_jsonl,
    read_video_rgb,
    sample_indices,
    validate_quadmask_spec,
    video_meta,
    write_json,
    write_text,
)


DEFAULT_MODEL = "qwen/qwen3.5-plus-20260420"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_DEBUG_DIR = Path("/data/cwx/E2W/data/physics_iq_vlm_sft/grounding_debug")


SYSTEM_PROMPT = """You are a visual grounding labeler for Edit2World.
Return only valid JSON. Do not include markdown or explanations.
Use the human/user remove target and existing planner text as source of truth.
Your job is spatial grounding, not rewriting the physics plan."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--http-referer", default="https://openai.com")
    parser.add_argument("--title", default="E2W Physics-IQ v6 Grounded Relabeling")
    parser.add_argument("--provider-only", action="append")
    parser.add_argument("--allow-fallbacks", action="store_true")
    parser.add_argument("--require-parameters", action="store_true")
    return parser.parse_args()


def parse_assistant_label(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["messages"][1]["content"])


def user_request(row: dict[str, Any]) -> str:
    content = row["messages"][0]["content"]
    match = re.search(r"User request:\s*(.+?)(?:\.\s*$|$)", content)
    if match:
        return match.group(1).strip()
    return content.strip()


def draw_grid(frame: Image.Image, frame_index: int, grid_size: int) -> Image.Image:
    img = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    line_color = (255, 255, 255)
    shadow = (0, 0, 0)
    for i in range(grid_size + 1):
        x = round(i * w / grid_size)
        y = round(i * h / grid_size)
        draw.line([(x, 0), (x, h)], fill=shadow, width=3)
        draw.line([(x, 0), (x, h)], fill=line_color, width=1)
        draw.line([(0, y), (w, y)], fill=shadow, width=3)
        draw.line([(0, y), (w, y)], fill=line_color, width=1)
    for r in range(grid_size):
        for c in range(grid_size):
            label = f"{chr(ord('A') + r)}{c + 1}"
            x = round((c + 0.04) * w / grid_size)
            y = round((r + 0.04) * h / grid_size)
            draw.text((x + 1, y + 1), label, fill=shadow)
            draw.text((x, y), label, fill=(255, 230, 80))
    draw.rectangle([0, 0, 108, 22], fill=(0, 0, 0))
    draw.text((4, 4), f"frame {frame_index}", fill=(255, 255, 255))
    return img


def make_contact_sheet(video_path: Path, out_path: Path, grid_size: int) -> dict[str, Any]:
    frames, fps = read_video_rgb(video_path)
    if not frames:
        raise RuntimeError(f"No frames in {video_path}")
    indices = sample_indices(len(frames), 5)
    thumbs = []
    for idx in indices:
        img = Image.fromarray(frames[idx])
        img = draw_grid(img, idx, grid_size)
        img.thumbnail((360, 240), Image.Resampling.LANCZOS)
        thumbs.append(img.copy())
    width = max(img.width for img in thumbs) * len(thumbs)
    height = max(img.height for img in thumbs)
    sheet = Image.new("RGB", (width, height), (20, 20, 20))
    x = 0
    for img in thumbs:
        sheet.paste(img, (x, 0))
        x += img.width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return {
        "frame_indices": indices,
        "fps": fps,
        "frame_count": len(frames),
        "width": frames[0].shape[1],
        "height": frames[0].shape[0],
    }


def image_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_teacher_prompt(row: dict[str, Any], label: dict[str, Any], sheet_meta: dict[str, Any], grid_size: int) -> str:
    text_quadmask = label.get("quadmask_spec", {})
    return f"""Relabel this Edit2World planner example with executable spatial grounding.

Video/sample:
- sample_id: {row["id"]}
- user_request: {user_request(row)}
- video_width: {sheet_meta["width"]}
- video_height: {sheet_meta["height"]}
- frame_count: {sheet_meta["frame_count"]}
- sampled_frames_in_contact_sheet: {sheet_meta["frame_indices"]}

Existing text planner label:
{json.dumps(label, ensure_ascii=False, indent=2)}

Existing text-only quadmask_spec:
{json.dumps(text_quadmask, ensure_ascii=False, indent=2)}

Coordinate rules:
- Use norm1000 coordinates, not pixels.
- x_norm1000 = round(x_pixel / video_width * 1000).
- y_norm1000 = round(y_pixel / video_height * 1000).
- bbox_xyxy_norm1000 is [x1, y1, x2, y2], tight around the primary remove target in frame 0.
- positive_points_norm1000 should be inside the primary target, preferably near its visual center.
- negative_points_norm1000 should be nearby non-target points if useful; use [] if unsure.
- The contact sheet grid is {grid_size}x{grid_size}. A1 is top-left, H8 is bottom-right for an 8x8 grid.

Affected region rules:
- Mark coarse grid cells where counterfactual visual/physical effects should occur.
- Include contact regions, object trajectory changes, collision regions, dynamic shadows/reflections/effects, and background revealed by removing the primary object.
- Use frame ranges in original video frame indices.
- If there is truly no visible affected region beyond the primary removal, set frame_ranges to [] and confidence <= 0.35.

Return exactly this JSON object:
{{
  "schema_version": "e2w.quadmask_spec.v1",
  "primary": {{
    "object_name": "...",
    "description": "...",
    "visibility": "clear|partial|brief|unclear",
    "keyframes": [
      {{
        "frame_index": 0,
        "bbox_xyxy_norm1000": [0, 0, 0, 0],
        "positive_points_norm1000": [[0, 0]],
        "negative_points_norm1000": []
      }}
    ],
    "confidence": 0.0
  }},
  "affected": {{
    "objects": ["..."],
    "grid_shape": [{grid_size}, {grid_size}],
    "frame_ranges": [
      {{
        "start_frame": 0,
        "end_frame": 0,
        "cells": ["A1"]
      }}
    ],
    "confidence": 0.0,
    "reason": "..."
  }},
  "keep": {{
    "description": "..."
  }},
  "quality_flags": {{
    "needs_human_review": false,
    "reasons": []
  }}
}}"""


def call_openrouter(args: argparse.Namespace, prompt: str, contact_sheet: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(contact_sheet)}},
                ],
            },
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    if args.provider_only:
        payload["provider"] = {
            "only": args.provider_only,
            "allow_fallbacks": args.allow_fallbacks,
            "require_parameters": args.require_parameters,
        }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": args.http_referer,
            "X-OpenRouter-Title": args.title,
        },
        method="POST",
    )
    last_exc: Exception | None = None
    for attempt in range(1, args.retries + 2):
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (
            urllib.error.URLError,
            http.client.IncompleteRead,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            last_exc = exc
            if attempt > args.retries:
                break
            sleep_for = args.retry_sleep_seconds * attempt
            print(f"  transient error on attempt {attempt}: {exc}; retrying in {sleep_for:.1f}s")
            time.sleep(sleep_for)
    raise RuntimeError(f"OpenRouter request failed after retries: {last_exc!r}")


def extract_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    return json.loads(text)


def normalize_teacher_spec(obj: dict[str, Any]) -> dict[str, Any]:
    if "quadmask_spec" in obj and isinstance(obj["quadmask_spec"], dict):
        obj = obj["quadmask_spec"]
    obj.setdefault("schema_version", "e2w.quadmask_spec.v1")
    obj.setdefault("keep", {"description": "background and unrelated objects that should stay unchanged"})
    return obj


def merge_label(base_label: dict[str, Any], grounded_spec: dict[str, Any]) -> dict[str, Any]:
    out = dict(base_label)
    old_spec = base_label.get("quadmask_spec", {}) if isinstance(base_label.get("quadmask_spec"), dict) else {}
    if "keep" not in grounded_spec and isinstance(old_spec.get("keep"), dict):
        grounded_spec["keep"] = old_spec["keep"]
    out["quadmask_spec"] = grounded_spec
    out.setdefault("quality_flags", {})
    out["quality_flags"]["grounded_spatial_supervision"] = True
    return out


def process_row(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any] | None:
    sample_id = row["id"]
    out_dir = args.debug_dir / sample_id
    validated_path = out_dir / "validated.json"
    if args.skip_existing and validated_path.exists():
        return json.loads(validated_path.read_text(encoding="utf-8"))["sft_row"]

    label = parse_assistant_label(row)
    video_path = Path(row["video"])
    contact_sheet = out_dir / "contact_sheet.png"
    sheet_meta = make_contact_sheet(video_path, contact_sheet, args.grid_size)
    prompt = build_teacher_prompt(row, label, sheet_meta, args.grid_size)
    write_text(out_dir / "prompt.txt", prompt)

    if args.dry_run:
        write_json(
            out_dir / "dry_run.json",
            {
                "sample_id": sample_id,
                "contact_sheet": str(contact_sheet),
                "prompt": str(out_dir / "prompt.txt"),
                "dry_run": True,
            },
        )
        return None
    if not args.api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required unless --dry-run is used")

    raw = call_openrouter(args, prompt, contact_sheet)
    write_json(out_dir / "teacher_raw.json", raw)
    content = raw["choices"][0]["message"]["content"]
    grounded_spec = normalize_teacher_spec(extract_json(content))
    write_json(out_dir / "teacher_grounding.json", grounded_spec)
    merged = merge_label(label, grounded_spec)
    metrics = validate_quadmask_spec(grounded_spec, video_meta(video_path))
    out_row = {
        **row,
        "messages": [
            row["messages"][0],
            {"role": "assistant", "content": json.dumps(merged, ensure_ascii=False)},
        ],
        "metadata": {
            **row.get("metadata", {}),
            "grounded_spatial_supervision": True,
            "grounding_model": args.model,
            "grounding_debug_dir": str(out_dir),
        },
    }
    validated = {
        "sample_id": sample_id,
        "sft_row": out_row,
        "metrics": metrics,
        "grounded_spec": grounded_spec,
    }
    write_json(validated_path, validated)
    return out_row


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl)
    if args.sample_id:
        wanted = set(args.sample_id)
        rows = [row for row in rows if row["id"] in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No rows selected")

    args.debug_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] relabel {row['id']}", flush=True)
        try:
            out_row = process_row(args, row)
            if out_row is not None:
                output_rows.append(out_row)
        except Exception as exc:
            failures.append({"sample_id": row["id"], "error": repr(exc)})
            write_json(args.debug_dir / row["id"] / "error.json", failures[-1])
            if not args.keep_going:
                raise
            print(f"  error: {exc}", file=sys.stderr)

    if not args.dry_run:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as f:
            for row in output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "debug_dir": str(args.debug_dir),
        "selected": len(rows),
        "written": len(output_rows),
        "failures": failures,
        "dry_run": args.dry_run,
        "model": args.model,
    }
    write_json(args.debug_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

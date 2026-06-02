#!/usr/bin/env python3
"""Rewrite planner SFT JSONL user prompts to the executable v6 schema."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from e2w_v0_common import (
    PLANNER_IO_SCHEMA_VERSION,
    build_planner_user_prompt,
    infer_actual_operation_from_raw,
    load_jsonl,
    validate_quadmask_spec,
    video_meta,
    write_json,
)


OLD_EMPTY_QUADMASK_SCHEMA = '"quadmask_spec": {"primary": {}, "affected": {}, "keep": {}}'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="If set, copy the input JSONL here before rewriting.",
    )
    parser.add_argument(
        "--operation",
        choices=["auto", "remove", "add"],
        default="auto",
        help="Expected operation for rewritten prompts. auto infers from assistant/sample.",
    )
    parser.add_argument(
        "--validate-assistant-executable",
        action="store_true",
        help="Validate assistant quadmask_spec using local video metadata.",
    )
    parser.add_argument(
        "--allow-validation-failures",
        action="store_true",
        help="Write output even if executable validation fails.",
    )
    return parser.parse_args()


def assistant_json(row: dict[str, Any]) -> dict[str, Any]:
    for message in row.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            return json.loads(content)
    raise ValueError(f"No assistant message for row {row.get('id')}")


def infer_operation(row: dict[str, Any], label: dict[str, Any], explicit: str) -> tuple[str, str]:
    if explicit != "auto":
        return explicit, "cli"
    return infer_actual_operation_from_raw(label, row)


def request_text(row: dict[str, Any], label: dict[str, Any], operation: str) -> str:
    edit_prompt = str(label.get("edit_prompt") or "").strip()
    if edit_prompt:
        return edit_prompt
    user = str((row.get("messages") or [{}])[0].get("content") or "").strip()
    marker = "User request:"
    if marker in user:
        return user.split(marker, 1)[1].strip().rstrip(".")
    target = label.get("target_objects", [{}])
    if isinstance(target, list) and target and isinstance(target[0], dict):
        name = str(target[0].get("name") or "target object").strip()
    else:
        name = "target object"
    return f"{operation} {name}"


def rewrite_row(row: dict[str, Any], operation_arg: str) -> tuple[dict[str, Any], dict[str, Any]]:
    label = assistant_json(row)
    operation, operation_source = infer_operation(row, label, operation_arg)
    messages = list(row.get("messages", []))
    if not messages:
        messages = [{"role": "user", "content": ""}]
    old_prompt = str(messages[0].get("content") or "")
    new_prompt = build_planner_user_prompt(
        str(row.get("id") or label.get("video_id") or ""),
        request_text(row, label, operation),
        operation=operation,
    )
    messages[0] = {**messages[0], "role": "user", "content": new_prompt}
    # Patch assistant quadmask_spec.operation if missing
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            try:
                assistant_data = json.loads(msg["content"]) if isinstance(msg["content"], str) else msg["content"]
                qs = assistant_data.get("quadmask_spec")
                if isinstance(qs, dict) and not qs.get("operation"):
                    qs["operation"] = operation
                    assistant_data["quadmask_spec"] = qs
                    messages[i] = {**msg, "content": json.dumps(assistant_data, ensure_ascii=False)}
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
            break
    out = {
        **row,
        "messages": messages,
        "metadata": {
            **row.get("metadata", {}),
            "planner_user_prompt_schema": PLANNER_IO_SCHEMA_VERSION,
            "operation": operation,
            "operation_source": operation_source,
        },
    }
    return out, {
        "sample_id": row.get("id"),
        "operation": operation,
        "operation_source": operation_source,
        "old_prompt_had_empty_quadmask_schema": OLD_EMPTY_QUADMASK_SCHEMA in old_prompt,
        "new_prompt_had_empty_quadmask_schema": OLD_EMPTY_QUADMASK_SCHEMA in new_prompt,
        "new_prompt_has_keyframes": "keyframes" in new_prompt,
        "new_prompt_has_norm1000": "bbox_xyxy_norm1000" in new_prompt and "positive_points_norm1000" in new_prompt,
        "new_prompt_has_grid": "grid_shape" in new_prompt and "frame_ranges" in new_prompt,
    }


def archive_input(input_jsonl: Path, archive_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{input_jsonl.name}.{stamp}.old_schema"
    shutil.copy2(input_jsonl, archive_path)
    return archive_path


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl)
    archive_path = archive_input(args.input_jsonl, args.archive_dir) if args.archive_dir else None
    output_rows: list[dict[str, Any]] = []
    row_metrics: list[dict[str, Any]] = []
    validation_failures: list[dict[str, Any]] = []

    for row in rows:
        out_row, metrics = rewrite_row(row, args.operation)
        if args.validate_assistant_executable:
            label = assistant_json(out_row)
            spec = label.get("quadmask_spec") if isinstance(label.get("quadmask_spec"), dict) else {}
            if "operation" not in spec:
                spec = {**spec, "operation": metrics["operation"]}
            spec_metrics = validate_quadmask_spec(spec, video_meta(Path(out_row["video"])))
            metrics.update(spec_metrics)
            if not spec_metrics["quadmask_spec_executable"]:
                validation_failures.append(
                    {
                        "sample_id": row.get("id"),
                        "executor_failure": spec_metrics.get("executor_failure"),
                        "metrics": spec_metrics,
                    }
                )
        row_metrics.append(metrics)
        output_rows.append(out_row)

    if validation_failures and not args.allow_validation_failures:
        summary = {
            "input_jsonl": str(args.input_jsonl),
            "output_jsonl": str(args.output_jsonl),
            "archive_path": str(archive_path) if archive_path else None,
            "rows": len(rows),
            "validation_failures": validation_failures,
            "status": "failed",
        }
        write_json(args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".summary.json"), summary)
        raise SystemExit(json.dumps(summary, indent=2, ensure_ascii=False))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "archive_path": str(archive_path) if archive_path else None,
        "rows": len(rows),
        "planner_user_prompt_schema": PLANNER_IO_SCHEMA_VERSION,
        "old_prompt_empty_quadmask_schema_count": sum(bool(m["old_prompt_had_empty_quadmask_schema"]) for m in row_metrics),
        "new_prompt_empty_quadmask_schema_count": sum(bool(m["new_prompt_had_empty_quadmask_schema"]) for m in row_metrics),
        "new_prompt_keyframes_count": sum(bool(m["new_prompt_has_keyframes"]) for m in row_metrics),
        "new_prompt_norm1000_count": sum(bool(m["new_prompt_has_norm1000"]) for m in row_metrics),
        "new_prompt_grid_count": sum(bool(m["new_prompt_has_grid"]) for m in row_metrics),
        "validation_failures": validation_failures,
        "status": "ok",
    }
    write_json(args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".summary.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

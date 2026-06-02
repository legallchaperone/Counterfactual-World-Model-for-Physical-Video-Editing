#!/usr/bin/env python3
"""Export teacher-grounded SFT rows as E2W run-bundle planner artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import (  # noqa: E402
    ensure_run_dirs,
    extract_first_frame,
    infer_actual_operation_from_raw,
    link_or_copy,
    load_jsonl,
    normalize_to_e2w_contract,
    resolve_expected_operation_with_source,
    serialize_vace_prompt,
    summarize_boolean_metrics,
    validate_edit_plan,
    validate_quadmask_spec,
    video_meta,
    write_json,
    write_text,
)


DEFAULT_INPUT = Path("/data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_eval_v6_teacher_grounded.jsonl")
DEFAULT_RUN_DIR = Path("/data/cwx/E2W/runs/e2w_v0_1_physics_iq_teacher_grounded_eval")
DEFAULT_MODE = "mode_a_lite_teacher_grounded"

METRIC_KEYS = [
    "json_parse_ok",
    "schema_valid",
    "quadmask_schema_valid",
    "operation_accuracy",
    "physical_consequences_nonempty",
    "edited_scene_caption_nonempty",
    "edited_scene_outcome_effects_nonempty",
    "primary_prompt_valid",
    "primary_bbox_valid",
    "primary_point_valid",
    "affected_grid_valid",
    "frame_index_valid",
    "coordinate_range_valid",
    "quadmask_spec_executable",
    "quadmask_spec_executor_valid",
    "executor_valid",
    "has_affected_grid",
    "frame_ranges_valid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--operation",
        choices=["auto", "remove", "add"],
        default="auto",
        help="Expected edit operation. auto infers add/remove from the sample prompt.",
    )
    parser.add_argument("--force", action="store_true")
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
    raise ValueError(f"No assistant JSON content for sample {row.get('id')}")


def write_source_artifacts(run_dir: Path, row: dict[str, Any], force: bool) -> dict[str, str]:
    sample_id = row["id"]
    src_dir = run_dir / "source" / sample_id
    video_src = Path(row["video"])
    video_dst = src_dir / "original_video.mp4"
    link_or_copy(video_src, video_dst)
    query = row.get("messages", [{}])[0].get("content", "")
    write_text(src_dir / "text_query.txt", query)
    first_frame = src_dir / "first_frame.png"
    if force or not first_frame.exists():
        extract_first_frame(video_src, first_frame)
    return {
        "original_video": str(video_dst),
        "text_query": str(src_dir / "text_query.txt"),
        "first_frame": str(first_frame),
    }


def process_row(args: argparse.Namespace, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = row["id"]
    raw = assistant_json(row)
    meta = video_meta(Path(row["video"]))
    expected_operation, expected_operation_source = resolve_expected_operation_with_source(args.operation, sample=row)
    _actual_operation, actual_operation_source = infer_actual_operation_from_raw(raw, row)
    edit_plan, quadmask_spec = normalize_to_e2w_contract(
        raw,
        row,
        meta,
        source="teacher_executable_v6",
    )
    if not isinstance(quadmask_spec, dict):
        raise ValueError(f"Missing quadmask_spec for sample {sample_id}")

    source_paths = write_source_artifacts(args.run_dir, row, args.force)
    pred_dir = args.run_dir / "planner_pred" / sample_id
    pred_dir.mkdir(parents=True, exist_ok=True)

    plan_metrics = validate_edit_plan(
        edit_plan,
        expected_operation=expected_operation,
        expected_operation_source=expected_operation_source,
        actual_operation_source=actual_operation_source,
    )
    spec_metrics = validate_quadmask_spec(quadmask_spec, meta)
    metrics = {
        "sample_id": sample_id,
        "json_parse_ok": True,
        "parse_error": None,
        "generation_runtime_sec": 0.0,
        "video": meta,
        "label_source": "teacher_executable_v6",
        **plan_metrics,
        **spec_metrics,
    }

    write_json(pred_dir / "raw.teacher.json", raw)
    write_json(pred_dir / "edit_plan.teacher.json", edit_plan)
    write_json(pred_dir / "edit_plan.json", edit_plan)
    write_json(pred_dir / "quadmask_spec.teacher.json", quadmask_spec)
    write_json(pred_dir / "quadmask_spec.json", quadmask_spec)
    write_text(pred_dir / "vace_prompt.txt", serialize_vace_prompt(edit_plan))
    write_json(pred_dir / "planner_eval.json", metrics)
    if not metrics["schema_valid"] or not metrics["executor_valid"]:
        status = "teacher_schema_failed"
    elif not metrics["operation_accuracy"]:
        status = "operation_mismatch"
    else:
        status = "ok"

    entry = {
        "stage": "planner_eval",
        "sample_id": sample_id,
        "mode": args.mode,
        "status": status,
        "paths": {
            **source_paths,
            "raw_teacher": str(pred_dir / "raw.teacher.json"),
            "edit_plan": str(pred_dir / "edit_plan.teacher.json"),
            "quadmask_spec": str(pred_dir / "quadmask_spec.teacher.json"),
            "vace_prompt": str(pred_dir / "vace_prompt.txt"),
            "planner_eval": str(pred_dir / "planner_eval.json"),
        },
        "metrics": metrics,
    }
    return entry, metrics


def main() -> None:
    args = parse_args()
    ensure_run_dirs(args.run_dir)
    rows = load_jsonl(args.input_jsonl)
    if args.sample_id:
        wanted = set(args.sample_id)
        rows = [row for row in rows if row.get("id") in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No rows selected")

    entries: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] export teacher {row['id']}", flush=True)
        entry, row_metrics = process_row(args, row)
        entries.append(entry)
        metrics.append(row_metrics)

    with (args.run_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary = summarize_boolean_metrics(metrics, METRIC_KEYS)
    summary.update(
        {
            "run_dir": str(args.run_dir),
            "input_jsonl": str(args.input_jsonl),
            "mode": args.mode,
            "label_source": "teacher_executable_v6",
        }
    )
    write_json(args.run_dir / "planner_pred" / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

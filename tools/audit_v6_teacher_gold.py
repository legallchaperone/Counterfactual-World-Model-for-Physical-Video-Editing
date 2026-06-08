#!/usr/bin/env python3
"""Audit teacher-grounded planner labels by treating teacher assistant responses as predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import (  # noqa: E402
    ensure_run_dirs,
    normalize_to_e2w_contract,
    parse_json_output,
    resolve_expected_operation_with_source,
    serialize_vace_prompt,
    summarize_boolean_metrics,
    validate_edit_plan,
    validate_quadmask_spec,
    video_meta,
    write_json,
    write_text,
    link_or_copy,
)


PLANNER_METRIC_KEYS = [
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
    "vace_prompt_contract_ok",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True, help="Input JSONL to audit.")
    parser.add_argument("--out", type=Path, required=True, help="Output run dir.")
    parser.add_argument("--mode", default="teacher_gold")
    parser.add_argument("--operation", choices=["auto", "remove", "add"], default="auto")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Write artifacts and return 0 even when selected samples fail metric gates.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def load_messages_last_assistant(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    if not isinstance(messages, list):
        return ""
    if not messages:
        return ""
    last = messages[-1]
    if not isinstance(last, dict):
        return ""
    text = last.get("content")
    if text is None:
        return ""
    return str(text)


def write_source_artifacts(out: Path, sample: dict[str, Any]) -> dict[str, str]:
    sample_id = sample["id"]
    src_dir = out / "source" / sample_id
    video_src = Path(sample["video"])
    video_dst = src_dir / "original_video.mp4"
    link_or_copy(video_src, video_dst)
    query = sample["messages"][0]["content"]
    write_text(src_dir / "text_query.txt", query)
    return {
        "original_video": str(video_dst),
        "text_query": str(src_dir / "text_query.txt"),
    }


def process_sample(args: argparse.Namespace, sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = sample["id"]
    pred_dir = args.out / "planner_pred" / sample_id
    pred_dir.mkdir(parents=True, exist_ok=True)

    source_paths = write_source_artifacts(args.out, sample)

    raw_text = load_messages_last_assistant(sample)
    raw_path = pred_dir / "raw_output.txt"
    write_text(raw_path, raw_text)

    raw_json, parse_error = parse_json_output(raw_text)
    meta = video_meta(Path(sample["video"]))
    metrics: dict[str, Any] = {
        "sample_id": sample_id,
        "json_parse_ok": raw_json is not None,
        "parse_error": parse_error,
        "video": meta,
    }

    if raw_json is None:
        metrics["vace_prompt_contract_ok"] = False
        metrics["vace_prompt_contract_error"] = "parse_failed"
        status = "planner_parse_failed"
        metrics["schema_valid"] = False
        metrics["quadmask_spec_executable"] = False
        for key in [
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
            "quadmask_schema_valid",
        ]:
            metrics[key] = False
        entry = {
            "stage": "planner_eval",
            "sample_id": sample_id,
            "mode": args.mode,
            "status": status,
            "paths": {
                **source_paths,
                "raw_output": str(raw_path),
                "planner_eval": str(pred_dir / "planner_eval.json"),
            },
            "metrics": metrics,
        }
        write_json(pred_dir / "planner_eval.json", metrics)
        return entry, metrics

    expected_operation, expected_operation_source = resolve_expected_operation_with_source(args.operation, sample=sample)
    actual_operation_source = "teacher_raw"
    edit_plan, quadmask_spec = normalize_to_e2w_contract(
        raw_json,
        sample,
        meta,
        source="teacher_raw",
    )
    plan_metrics = validate_edit_plan(
        edit_plan,
        expected_operation=expected_operation,
        expected_operation_source=expected_operation_source,
        actual_operation_source=actual_operation_source,
    )
    spec_metrics = validate_quadmask_spec(quadmask_spec, meta)
    metrics.update(plan_metrics)
    metrics.update(spec_metrics)

    # parse_json_output already validates raw planner JSON contract.
    metrics["schema_valid"] = True

    write_json(pred_dir / "raw.pred.json", raw_json)
    write_json(pred_dir / "edit_plan.pred.json", edit_plan)
    write_json(pred_dir / "edit_plan.json", edit_plan)
    write_json(pred_dir / "quadmask_spec.pred.json", quadmask_spec)
    write_json(pred_dir / "quadmask_spec.json", quadmask_spec)

    vace_prompt_error = None
    try:
        write_text(pred_dir / "vace_prompt.txt", serialize_vace_prompt(edit_plan))
        metrics["vace_prompt_contract_ok"] = True
        metrics["vace_prompt_contract_error"] = None
    except Exception as exc:
        vace_prompt_error = str(exc)
        metrics["vace_prompt_contract_ok"] = False
        metrics["vace_prompt_contract_error"] = vace_prompt_error

    if not metrics["schema_valid"]:
        status = "planner_schema_failed"
    elif not metrics["operation_accuracy"]:
        status = "operation_mismatch"
    elif not metrics["quadmask_spec_executable"]:
        status = "planner_quadmask_failed"
    elif vace_prompt_error:
        status = "vace_prompt_contract_failed"
    else:
        status = "ok"

    entry = {
        "stage": "planner_eval",
        "sample_id": sample_id,
        "mode": args.mode,
        "status": status,
        "paths": {
            **source_paths,
            "raw_output": str(raw_path),
            "raw_pred": str(pred_dir / "raw.pred.json"),
            "edit_plan": str(pred_dir / "edit_plan.pred.json"),
            "quadmask_spec": str(pred_dir / "quadmask_spec.pred.json"),
            "planner_eval": str(pred_dir / "planner_eval.json"),
        },
        "metrics": metrics,
    }
    write_json(pred_dir / "planner_eval.json", metrics)
    return entry, metrics


def main() -> None:
    args = parse_args()
    ensure_run_dirs(args.out)
    rows = load_jsonl(args.jsonl)
    if args.sample_id:
        wanted = set(args.sample_id)
        rows = [r for r in rows if r["id"] in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No samples selected")

    manifest_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for sample in rows:
        entry, metrics = process_sample(args, sample)
        manifest_rows.append(entry)
        metric_rows.append(metrics)

    summary = summarize_boolean_metrics(metric_rows, PLANNER_METRIC_KEYS)
    summary.update(
        {
            "run_dir": str(args.out),
            "split_jsonl": str(args.jsonl),
            "mode": args.mode,
            "label_source": "teacher_gold",
            "planner_backend": "teacher_gold",
            "adapter": None,
            "base_model": None,
        }
    )
    write_json(args.out / "planner_pred" / "summary.json", summary)

    manifest_path = args.out / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in manifest_rows:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.allow_failures:
        failed = [entry for entry in manifest_rows if entry.get("status") != "ok"]
        if failed:
            print(
                json.dumps(
                    {
                        "planner_failures": [
                            {
                                "sample_id": entry.get("sample_id"),
                                "status": entry.get("status"),
                                "parse_error": (entry.get("metrics") or {}).get("parse_error"),
                                "operation_accuracy": (entry.get("metrics") or {}).get("operation_accuracy"),
                                "schema_valid": (entry.get("metrics") or {}).get("schema_valid"),
                                "quadmask_spec_executable": (entry.get("metrics") or {}).get("quadmask_spec_executable"),
                                "vace_prompt_contract_ok": (entry.get("metrics") or {}).get("vace_prompt_contract_ok"),
                                "vace_prompt_contract_error": (entry.get("metrics") or {}).get("vace_prompt_contract_error"),
                            }
                            for entry in failed
                        ]
                    },
                    indent=2,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            raise SystemExit(1)


if __name__ == "__main__":
    main()

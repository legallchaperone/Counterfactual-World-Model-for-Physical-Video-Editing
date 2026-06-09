#!/usr/bin/env python3
"""Artifact-level eval for the E2W v0 VLM planner checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_PLANNER,
    DEFAULT_RUN_DIR,
    DEFAULT_SPLIT,
    VacePromptContractError,
    ensure_run_dirs,
    extract_first_frame,
    infer_actual_operation_from_raw,
    link_or_copy,
    load_jsonl,
    normalize_to_e2w_contract,
    parse_add_planner_json,
    parse_json_output,
    resolve_expected_operation_with_source,
    serialize_vace_prompt,
    summarize_boolean_metrics,
    validate_edit_plan,
    validate_quadmask_spec,
    video_meta,
    write_json,
    write_text,
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--split-jsonl", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_PLANNER)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Run the base VLM without loading a LoRA adapter.",
    )
    parser.add_argument("--mode", default="mode_vlm_planner_pred")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--min-pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument(
        "--operation",
        choices=["auto", "remove", "add"],
        default="auto",
        help="Expected edit operation. auto infers add/remove from the sample prompt.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Write planner artifacts but return 0 even when one or more selected samples fail planner validation.",
    )
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    processor_dir = args.base_model if args.no_adapter else args.adapter
    processor = AutoProcessor.from_pretrained(processor_dir, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    if not args.no_adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    return processor, model.eval()


def generate_one(args: argparse.Namespace, processor: Any, model: Any, sample: dict[str, Any]) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": sample["video"],
                    "fps": args.video_fps,
                    "min_pixels": args.min_pixels,
                    "max_pixels": args.max_pixels,
                },
                {"type": "text", "text": sample["messages"][0]["content"]},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    if torch.cuda.is_available():
        inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    new_tokens = out[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def write_source_artifacts(args: argparse.Namespace, sample: dict[str, Any]) -> dict[str, str]:
    sample_id = sample["id"]
    src_dir = args.run_dir / "source" / sample_id
    video_src = Path(sample["video"])
    video_dst = src_dir / "original_video.mp4"
    link_or_copy(video_src, video_dst)
    query = sample["messages"][0]["content"]
    write_text(src_dir / "text_query.txt", query)
    first_frame = src_dir / "first_frame.png"
    if args.force or not first_frame.exists():
        extract_first_frame(video_src, first_frame)
    return {
        "original_video": str(video_dst),
        "text_query": str(src_dir / "text_query.txt"),
        "first_frame": str(first_frame),
    }


def process_sample(args: argparse.Namespace, processor: Any, model: Any, sample: dict[str, Any]) -> dict[str, Any]:
    sample_id = sample["id"]
    pred_dir = args.run_dir / "planner_pred" / sample_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    source_paths = write_source_artifacts(args, sample)

    raw_path = pred_dir / "raw_output.txt"
    if args.force or not raw_path.exists():
        started = time.time()
        raw_text = generate_one(args, processor, model, sample)
        write_text(raw_path, raw_text)
        runtime = time.time() - started
    else:
        raw_text = raw_path.read_text(encoding="utf-8")
        runtime = 0.0

    meta = video_meta(Path(sample["video"]))
    raw_json, parse_error = parse_json_output(raw_text)
    add_contract_parse_fallback_used = False
    if raw_json is None and args.operation == "add":
        raw_json_add, parse_error_add = parse_add_planner_json(raw_text)
        if raw_json_add is not None:
            raw_json = raw_json_add
            parse_error = None
            add_contract_parse_fallback_used = True
    metrics: dict[str, Any] = {
        "sample_id": sample_id,
        "json_parse_ok": raw_json is not None,
        "parse_error": parse_error,
        "add_contract_parse_fallback_used": add_contract_parse_fallback_used,
        "generation_runtime_sec": runtime,
        "video": meta,
    }

    if raw_json is None:
        write_json(pred_dir / "planner_eval.json", metrics)
        return {
            "stage": "planner_eval",
            "sample_id": sample_id,
            "mode": args.mode,
            "status": "planner_parse_failed",
            "paths": {**source_paths, "raw_output": str(raw_path), "planner_eval": str(pred_dir / "planner_eval.json")},
            "metrics": metrics,
        }

    expected_operation, expected_operation_source = resolve_expected_operation_with_source(args.operation, sample=sample)
    _actual_operation, actual_operation_source = infer_actual_operation_from_raw(raw_json, sample)
    edit_plan, quadmask_spec = normalize_to_e2w_contract(
        raw_json,
        sample,
        meta,
        source="planner_pred",
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

    write_json(pred_dir / "raw.pred.json", raw_json)
    write_json(pred_dir / "edit_plan.pred.json", edit_plan)
    write_json(pred_dir / "edit_plan.json", edit_plan)
    write_json(pred_dir / "quadmask_spec.pred.json", quadmask_spec)
    write_json(pred_dir / "quadmask_spec.json", quadmask_spec)
    vace_prompt_path = pred_dir / "vace_prompt.txt"
    vace_prompt_error = None
    try:
        write_text(vace_prompt_path, serialize_vace_prompt(edit_plan))
        metrics["vace_prompt_contract_ok"] = True
        metrics["vace_prompt_contract_error"] = None
    except VacePromptContractError as exc:
        vace_prompt_error = str(exc)
        metrics["vace_prompt_contract_ok"] = False
        metrics["vace_prompt_contract_error"] = vace_prompt_error
    write_json(pred_dir / "planner_eval.json", metrics)
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

    return {
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
            "vace_prompt": str(vace_prompt_path) if not vace_prompt_error else None,
            "planner_eval": str(pred_dir / "planner_eval.json"),
        },
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    ensure_run_dirs(args.run_dir)
    rows = load_jsonl(args.split_jsonl)
    if args.sample_id:
        wanted = set(args.sample_id)
        rows = [r for r in rows if r["id"] in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No samples selected")

    processor, model = load_model(args)
    manifest_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] planner eval {sample['id']}", flush=True)
        entry = process_sample(args, processor, model, sample)
        manifest_rows.append(entry)
        metric_rows.append(entry["metrics"])

    summary = summarize_boolean_metrics(metric_rows, PLANNER_METRIC_KEYS)
    summary.update(
        {
            "run_dir": str(args.run_dir),
            "split_jsonl": str(args.split_jsonl),
        "adapter": None if args.no_adapter else str(args.adapter),
        "base_model": str(args.base_model),
        "mode": args.mode,
        "planner_backend": "vlm_base_qwen2.5_vl" if args.no_adapter else "vlm_sft_qwen2.5_vl_lora",
            "label_source": "planner_pred",
            "generation": {
                "temperature": 0,
                "top_p": 1,
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
            },
        }
    )
    write_json(args.run_dir / "planner_pred" / "summary.json", summary)

    manifest_path = args.run_dir / "manifest.jsonl"
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

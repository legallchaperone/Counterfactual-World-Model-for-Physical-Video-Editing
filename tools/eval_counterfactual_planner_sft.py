#!/usr/bin/env python3
"""Run Counterfactual Planner LoRA inference and validate eval JSON outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
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
from e2w_v0_common import validate_counterfactual_planner_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-jsonl", type=Path, required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sample-count", type=int, default=5)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        snippet = cleaned[start : end + 1]
        try:
            obj = json.loads(snippet)
            return (obj, None) if isinstance(obj, dict) else (None, "decoded JSON snippet is not an object")
        except json.JSONDecodeError as exc:
            return None, str(exc)
    return None, "no JSON object found"


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.eval_jsonl)
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

    outputs: list[dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
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
        gen_kwargs = {"max_new_tokens": args.max_new_tokens}
        if args.temperature and args.temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": args.temperature})
        else:
            gen_kwargs.update({"do_sample": False})
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        obj, parse_error = parse_json_object(text)
        valid = False
        validation_error = parse_error
        if obj is not None:
            valid, validation_error = validate_counterfactual_planner_output(obj, source_video_id=str(row.get("video_id") or "unknown"))
        out = {
            "index": i - 1,
            "source": row.get("source"),
            "video_id": row.get("video_id"),
            "raw_output": text,
            "parsed": obj,
            "json_parse_ok": obj is not None,
            "schema_valid": valid,
            "validation_error": validation_error,
        }
        outputs.append(out)
        print(f"{i:02d}/{len(rows)} {row.get('source')}/{row.get('video_id')} parse={obj is not None} valid={valid}", flush=True)

    write_jsonl(args.output_jsonl, outputs)
    parsed = [r["parsed"] for r in outputs if isinstance(r.get("parsed"), dict)]
    valid_outputs = [r for r in outputs if r.get("schema_valid")]
    fill_counts = Counter(
        r["parsed"]["counterfactual_state"]["fill_type"]
        for r in outputs
        if r.get("schema_valid") and isinstance(r.get("parsed"), dict)
    )
    summary = {
        "count": len(outputs),
        "json_parse_ok": sum(bool(r.get("json_parse_ok")) for r in outputs),
        "schema_valid": len(valid_outputs),
        "if_removed_pass": len(valid_outputs),
        "fill_type_distribution": dict(fill_counts),
        "output_jsonl": str(args.output_jsonl),
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print("SAMPLE_OUTPUTS")
    for item in outputs[: args.sample_count]:
        printable = item.get("parsed") if item.get("parsed") is not None else {"raw_output": item.get("raw_output"), "error": item.get("validation_error")}
        print(json.dumps({"video_id": item.get("video_id"), "source": item.get("source"), "output": printable}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

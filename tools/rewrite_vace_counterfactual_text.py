#!/usr/bin/env python3
"""Rewrite planner SFT counterfactual text to satisfy the VACE prompt contract."""

from __future__ import annotations

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import (  # noqa: E402
    PLANNER_IO_SCHEMA_VERSION,
    VacePromptContractError,
    infer_actual_operation_from_raw,
    load_jsonl,
    normalize_to_e2w_contract,
    serialize_vace_prompt,
    validate_quadmask_spec,
    video_meta,
    write_json,
    write_text,
    _strings,
    _target_term_variants,
)


DEFAULT_MODEL = "qwen/qwen3.5-plus-20260420"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = """You are an Edit2World planner-label editor.
Return only valid JSON. Do not include markdown or explanations.
Your job is to rewrite VACE-facing counterfactual text without changing the edit target, object grounding, or quadmask fields."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--rewrite-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--http-referer", default="https://openai.com")
    parser.add_argument("--title", default="E2W VACE Counterfactual Text Rewrite")
    parser.add_argument("--provider-only", action="append")
    parser.add_argument("--allow-fallbacks", action="store_true")
    parser.add_argument("--require-parameters", action="store_true")
    return parser.parse_args()


def parse_assistant_label(row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    messages = row.get("messages", [])
    for idx, message in enumerate(messages):
        if message.get("role") == "assistant":
            return idx, json.loads(message.get("content") or "{}")
    raise ValueError(f"row {row.get('id')} has no assistant message")


def user_request(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    if not messages:
        return ""
    content = str(messages[0].get("content") or "")
    match = re.search(r"User request:\s*(.+?)(?:\n|$)", content)
    return match.group(1).strip() if match else content.strip()


def target_terms(label: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    targets = label.get("target_objects")
    if isinstance(targets, list):
        for target in targets:
            if not isinstance(target, dict):
                continue
            if target.get("name"):
                terms.append(str(target["name"]))
            aliases = target.get("aliases")
            if isinstance(aliases, list):
                terms.extend(str(alias) for alias in aliases if str(alias).strip())
    return list(dict.fromkeys(term.strip() for term in terms if term.strip()))


def validator_forbidden_terms(row: dict[str, Any], label: dict[str, Any]) -> list[str]:
    """Return the exact target-term expansion used by serialize_vace_prompt()."""
    plan, _, _ = make_plan(row, label)
    details = plan.get("operation_details", {}) if isinstance(plan.get("operation_details"), dict) else {}
    subject = plan.get("edit_subject", {}) if isinstance(plan.get("edit_subject"), dict) else {}
    target = details.get("target_object", {}) if isinstance(details.get("target_object"), dict) else {}
    return _target_term_variants(
        _strings(subject.get("label"))
        + _strings(subject.get("aliases"))
        + _strings(subject.get("included_parts"))
        + _strings(target.get("label"))
    )


def make_plan(row: dict[str, Any], label: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    meta = video_meta(Path(row["video"]))
    plan, spec = normalize_to_e2w_contract(label, row, meta, source="sft_label")
    return plan, spec, meta


def validate_row(row: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    plan, spec, meta = make_plan(row, label)
    quadmask = validate_quadmask_spec(spec, meta)
    try:
        prompt = serialize_vace_prompt(plan)
        return {
            "vace_prompt_contract_ok": True,
            "vace_prompt": prompt,
            "vace_prompt_contract_error": "",
            **quadmask,
        }
    except VacePromptContractError as exc:
        return {
            "vace_prompt_contract_ok": False,
            "vace_prompt": "",
            "vace_prompt_contract_error": str(exc),
            **quadmask,
        }


def build_prompt(row: dict[str, Any], label: dict[str, Any], previous_errors: list[str]) -> str:
    operation, _ = infer_actual_operation_from_raw(label, row)
    cfe = label.get("counterfactual_expectation", {})
    target = target_terms(label)
    forbidden = validator_forbidden_terms(row, label)
    return f"""Rewrite only the planner counterfactual_expectation text so it is safe for a target-free VACE video generation prompt.

Sample:
- sample_id: {row.get("id")}
- schema_version: {PLANNER_IO_SCHEMA_VERSION}
- operation: {operation}
- user_request: {user_request(row)}
- target terms from planner label: {json.dumps(target, ensure_ascii=False)}
- exact validator forbidden list generated by _target_term_variants(): {json.dumps(forbidden, ensure_ascii=False)}
- 以下词语及其任何变体一律不得出现: {json.dumps(forbidden, ensure_ascii=False)}
- protected_objects: {json.dumps(label.get("protected_objects", []), ensure_ascii=False)}

Planner context:
- event_summary: {label.get("event_summary", "")}
- physical_causal_chain: {json.dumps(label.get("physical_causal_chain", []), ensure_ascii=False)}
- current counterfactual_expectation: {json.dumps(cfe, ensure_ascii=False)}

Rules:
- Preserve the same physical meaning and non-target objects.
- Do not change target_objects, protected_objects, physical_causal_chain, quadmask_spec, or quality_flags except counterfactual text review reasons if needed.
- For remove operations, rewrite if_removed so it does not contain any item from the exact validator forbidden list above, including case, plural, possessive, hyphenated, or punctuation variants.
- For remove operations, the forbidden list can include generic head nouns or material words; avoid those words even when describing protected non-target objects, and use location, motion, illumination, support, contact, shadow, reflection, or surface descriptions instead.
- For remove operations, also avoid negative/removal wording such as "no <target>", "without <target>", "where <target> was", "missing", "gone", "absent", "removed", "deleted", or "erased".
- For remove operations, describe only target-free visible outcomes, such as changed motion, support, illumination, contact, shadows, surfaces, or unchanged non-target objects.
- For add operations, rewrite if_added so it describes the inserted object and physically expected effects, including count or contact shadow when clear.
- Preserve explicit counts for visible non-target objects when clear.
- If you cannot make the text safe without losing the physical meaning, set needs_human_review true and explain why.

Previous validation errors:
{json.dumps(previous_errors, ensure_ascii=False)}

Return exactly this JSON object:
{{
  "counterfactual_expectation": {{
    "if_removed": "...",
    "if_added": "...",
    "affected_regions": ["..."],
    "unchanged_regions": ["..."],
    "uncertainties": ["..."]
  }},
  "quality_flags_patch": {{
    "needs_human_review": false,
    "reasons": []
  }}
}}"""


def call_openrouter(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
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
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
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
            print(f"  transient error on attempt {attempt}: {exc}; retrying in {sleep_for:.1f}s", flush=True)
            time.sleep(sleep_for)
    raise RuntimeError(f"OpenRouter request failed after retries: {last_exc!r}")


def extract_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    return json.loads(text)


def merge_counterfactual(label: dict[str, Any], patch: dict[str, Any], model: str) -> dict[str, Any]:
    cfe_patch = patch.get("counterfactual_expectation")
    if not isinstance(cfe_patch, dict):
        raise ValueError("teacher output missing counterfactual_expectation")
    out = json.loads(json.dumps(label, ensure_ascii=False))
    current = out.get("counterfactual_expectation", {})
    if not isinstance(current, dict):
        current = {}
    merged = dict(current)
    for key in ["if_removed", "if_added", "affected_regions", "unchanged_regions", "uncertainties"]:
        if key in cfe_patch:
            merged[key] = cfe_patch[key]
    out["counterfactual_expectation"] = merged
    out.setdefault("quality_flags", {})
    out["quality_flags"]["vace_counterfactual_text_rewritten"] = True
    out["quality_flags"]["vace_counterfactual_text_model"] = model
    qpatch = patch.get("quality_flags_patch")
    if isinstance(qpatch, dict):
        if qpatch.get("needs_human_review"):
            out["quality_flags"]["needs_human_review"] = True
        reasons = qpatch.get("reasons")
        if isinstance(reasons, list) and reasons:
            existing = out["quality_flags"].get("reasons", [])
            if not isinstance(existing, list):
                existing = []
            out["quality_flags"]["reasons"] = list(dict.fromkeys([*existing, *[str(x) for x in reasons]]))
    return out


def write_row(row: dict[str, Any], label: dict[str, Any], assistant_idx: int, metadata: dict[str, Any]) -> dict[str, Any]:
    messages = list(row.get("messages", []))
    messages[assistant_idx] = {
        **messages[assistant_idx],
        "role": "assistant",
        "content": json.dumps(label, ensure_ascii=False),
    }
    return {
        **row,
        "messages": messages,
        "metadata": {
            **row.get("metadata", {}),
            **metadata,
        },
    }


def process_row(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any] | None:
    sample_id = str(row.get("id"))
    out_dir = args.debug_dir / sample_id
    validated_path = out_dir / "validated.json"
    if args.skip_existing and validated_path.exists():
        return json.loads(validated_path.read_text(encoding="utf-8"))["sft_row"]

    assistant_idx, label = parse_assistant_label(row)
    initial_metrics = validate_row(row, label)
    write_json(out_dir / "initial_metrics.json", initial_metrics)
    if initial_metrics["vace_prompt_contract_ok"]:
        out_row = write_row(
            row,
            label,
            assistant_idx,
            {
                "vace_counterfactual_text_rewritten": False,
                "vace_prompt_contract_ok": True,
                "accepted_for_sft": bool(initial_metrics.get("quadmask_spec_executable")),
            },
        )
        write_json(out_dir / "validated.json", {"sample_id": sample_id, "sft_row": out_row, "metrics": initial_metrics})
        return out_row

    if args.dry_run:
        write_text(out_dir / "prompt.txt", build_prompt(row, label, [initial_metrics["vace_prompt_contract_error"]]))
        return None
    if not args.api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required unless --dry-run is used")

    errors = [initial_metrics["vace_prompt_contract_error"]]
    for attempt in range(1, args.rewrite_attempts + 1):
        prompt = build_prompt(row, label, errors)
        write_text(out_dir / f"prompt_attempt_{attempt}.txt", prompt)
        raw = call_openrouter(args, prompt)
        write_json(out_dir / f"teacher_raw_attempt_{attempt}.json", raw)
        content = raw["choices"][0]["message"]["content"]
        patch = extract_json(content)
        write_json(out_dir / f"counterfactual_patch_attempt_{attempt}.json", patch)
        candidate = merge_counterfactual(label, patch, args.model)
        metrics = validate_row(row, candidate)
        write_json(out_dir / f"metrics_attempt_{attempt}.json", metrics)
        if metrics["vace_prompt_contract_ok"] and metrics.get("quadmask_spec_executable"):
            out_row = write_row(
                row,
                candidate,
                assistant_idx,
                {
                    "vace_counterfactual_text_rewritten": True,
                    "vace_counterfactual_text_model": args.model,
                    "vace_counterfactual_debug_dir": str(out_dir),
                    "vace_prompt_contract_ok": True,
                    "accepted_for_sft": True,
                },
            )
            write_json(
                validated_path,
                {"sample_id": sample_id, "sft_row": out_row, "metrics": metrics, "attempt": attempt},
            )
            return out_row
        errors.append(metrics.get("vace_prompt_contract_error") or "quadmask_spec_executable failed")

    review = {
        "sample_id": sample_id,
        "initial_metrics": initial_metrics,
        "errors": errors,
        "debug_dir": str(out_dir),
    }
    write_json(out_dir / "review_required.json", review)
    return None


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl)
    if args.sample_id:
        wanted = set(args.sample_id)
        rows = [row for row in rows if str(row.get("id")) in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No rows selected")

    args.debug_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    review_queue: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        sample_id = str(row.get("id"))
        print(f"[{idx}/{len(rows)}] rewrite VACE text {sample_id}", flush=True)
        try:
            out_row = process_row(args, row)
            if out_row is not None:
                output_rows.append(out_row)
            validated_path = args.debug_dir / sample_id / "validated.json"
            if validated_path.exists():
                validation_rows.append(json.loads(validated_path.read_text(encoding="utf-8")))
            else:
                review_queue.append({"sample_id": sample_id, "debug_dir": str(args.debug_dir / sample_id)})
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": repr(exc)})
            write_json(args.debug_dir / sample_id / "error.json", failures[-1])
            if not args.keep_going:
                raise
            print(f"  error: {exc}", file=sys.stderr, flush=True)
            review_queue.append({"sample_id": sample_id, "debug_dir": str(args.debug_dir / sample_id), "error": repr(exc)})

    if not args.dry_run:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as f:
            for row in output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_rows = [v.get("metrics", {}) for v in validation_rows]
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "debug_dir": str(args.debug_dir),
        "selected": len(rows),
        "written": len(output_rows),
        "failures": failures,
        "review_queue": str(args.debug_dir / "review_queue.json"),
        "review_queue_count": len(review_queue),
        "dry_run": args.dry_run,
        "model": args.model,
        "validation": {
            "rows": len(validation_rows),
            "quadmask_spec_executable": sum(bool(m.get("quadmask_spec_executable")) for m in metrics_rows),
            "vace_prompt_contract_ok": sum(bool(m.get("vace_prompt_contract_ok")) for m in metrics_rows),
            "vace_prompt_contract_failed": sum(not bool(m.get("vace_prompt_contract_ok")) for m in metrics_rows),
        },
    }
    write_json(args.debug_dir / "review_queue.json", review_queue)
    write_json(args.debug_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

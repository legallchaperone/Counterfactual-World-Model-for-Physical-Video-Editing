#!/usr/bin/env python3
"""Create debug judge records and report.md for the E2W v0 bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import DEFAULT_RUN_DIR, ensure_run_dirs, load_manifest, write_json, write_text  # noqa: E402


SCORE_KEYS = [
    "interaction_physics",
    "object_removal",
    "background_artifacts",
    "temporal_consistency",
    "preservation",
    "sharpness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--mode", default="mode_c_full_predicted")
    return parser.parse_args()


def stage_index(entries: list[dict[str, Any]]) -> dict[tuple[str | None, str, str | None], dict[str, Any]]:
    out: dict[tuple[str | None, str, str | None], dict[str, Any]] = {}
    for entry in entries:
        sample_id = entry.get("sample_id")
        if sample_id:
            out[(entry.get("stage"), sample_id, entry.get("mode"))] = entry
    return out


def lookup_planner_entry(
    indexed: dict[tuple[str | None, str, str | None], dict[str, Any]],
    sample_id: str,
    mode: str,
) -> dict[str, Any] | None:
    return (
        indexed.get(("planner_eval", sample_id, mode))
        or indexed.get((None, sample_id, mode))
        or indexed.get(("planner_eval", sample_id, "planner_pred"))
        or indexed.get((None, sample_id, "planner_pred"))
    )


def load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def first_present(*entries: dict[str, Any] | None) -> dict[str, Any] | None:
    for entry in entries:
        if entry is not None:
            return entry
    return None


def normalize_failure(entry: dict[str, Any] | None, default: str) -> tuple[str, str]:
    detail = (entry or {}).get("failure_source") or default
    if detail in {"planner", "planner_grounding", "planner_parse_failed", "planner_schema_failed"}:
        return "planner", detail
    if detail in {"mask", "sam2", "mask_builder"}:
        return "mask", detail
    if detail in {"first_frame"}:
        return "first_frame", detail
    if detail in {"vace", "not_run"}:
        return "vace", detail
    return "unclear", str(detail)


def judge_one(
    run_dir: Path,
    sample_id: str,
    mode: str,
    planner: dict[str, Any] | None,
    mask: dict[str, Any] | None,
    first_frame: dict[str, Any] | None,
    vace: dict[str, Any] | None,
) -> dict[str, Any]:
    planner_metrics = (planner or {}).get("metrics", {})
    if not planner or not planner_metrics.get("schema_valid"):
        failure_source, detail = normalize_failure(planner, "planner")
    elif not mask or mask.get("status") != "ok":
        failure_source, detail = normalize_failure(mask, "mask")
    elif not first_frame or first_frame.get("status") != "ok":
        failure_source, detail = normalize_failure(first_frame, "first_frame")
    elif not vace or vace.get("status") != "ok":
        failure_source, detail = normalize_failure(vace, "not_run")
    else:
        failure_source, detail = "unclear", "vlm_judge_not_run"

    edited_video = (vace or {}).get("paths", {}).get("edited_video")
    scores = {key: 0 for key in SCORE_KEYS}
    judge = {
        "sample_id": sample_id,
        "mode": mode,
        "judge_backend": "artifact_heuristic_no_vlm",
        "not_final_score": True,
        "scores": scores,
        "total": sum(scores.values()),
        "failure_source": failure_source,
        "failure_source_detail": detail,
        "notes": "No VLM judge call was made; this record is for artifact gating and failure attribution.",
        "edited_video": edited_video,
    }
    out_path = run_dir / "judge" / sample_id / mode / "judge.json"
    write_json(out_path, judge)
    return judge


def rate(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key, {})
    if isinstance(value, dict):
        return float(value.get("rate", 0.0))
    return float(value or 0.0)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def aggregate_mask_metrics(mask_entries: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [entry.get("metrics", {}) for entry in mask_entries]
    valid_metrics = [m for m in metrics if m.get("mask_valid")]
    values_ok = [
        bool(m.get("quadmask_values")) and set(m.get("quadmask_values", [])).issubset({0, 63, 127, 255})
        for m in metrics
    ]
    return {
        "count": len(metrics),
        "mask_valid_count": sum(bool(m.get("mask_valid")) for m in metrics),
        "primary_nonempty_count": sum(float(m.get("primary_area_ratio", 0.0)) > 0 for m in metrics),
        "affected_nonempty_count": sum(float(m.get("affected_area_ratio", 0.0)) > 0 for m in metrics),
        "valid_quadmask_value_count": sum(values_ok),
        "avg_editable_area_ratio": float(np.mean([m.get("editable_area_ratio", 0.0) for m in valid_metrics])) if valid_metrics else 0.0,
        "failure_sources": {},
    }


def main() -> None:
    args = parse_args()
    ensure_run_dirs(args.run_dir)
    manifest = load_manifest(args.run_dir / "manifest.jsonl")
    indexed = stage_index(manifest)
    planner_entries = [e for e in manifest if e.get("stage") in (None, "planner_eval")]
    sample_ids = [e["sample_id"] for e in planner_entries]

    rows: list[dict[str, Any]] = []
    judges: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        planner = lookup_planner_entry(indexed, sample_id, args.mode)
        mask = indexed.get(("mask_builder", sample_id, args.mode))
        first_frame = indexed.get(("first_frame", sample_id, args.mode))
        vace = indexed.get(("vace_v0", sample_id, args.mode))
        judge = judge_one(args.run_dir, sample_id, args.mode, planner, mask, first_frame, vace)
        judges.append(judge)
        artifact_entry = first_present(vace, first_frame, mask, planner)
        rows.append(
            {
                "sample_id": sample_id,
                "mode": args.mode,
                "schema_valid": bool((planner or {}).get("metrics", {}).get("schema_valid")),
                "mask_valid": bool((mask or {}).get("metrics", {}).get("mask_valid")),
                "first_frame_status": (first_frame or {}).get("status", "missing"),
                "vace_status": (vace or {}).get("status", "missing"),
                "judge_total": judge["total"],
                "failure_source": judge["failure_source_detail"],
                "artifact_path": str(args.run_dir / "judge" / sample_id / args.mode / "judge.json")
                if artifact_entry
                else "",
            }
        )

    planner_summary = load_json(args.run_dir / "planner_pred" / "summary.json")
    label_source = planner_summary.get("label_source", "planner_pred")
    mask_entries = [e for e in manifest if e.get("stage") == "mask_builder" and e.get("mode") == args.mode]
    mask_summary = aggregate_mask_metrics(mask_entries)
    for entry in mask_entries:
        fs = entry.get("failure_source") or "none"
        mask_summary["failure_sources"][fs] = mask_summary["failure_sources"].get(fs, 0) + 1
    first_entries = [e for e in manifest if e.get("stage") == "first_frame" and e.get("mode") == args.mode]
    vace_entries = [e for e in manifest if e.get("stage") == "vace_v0" and e.get("mode") == args.mode]

    first_summary = {
        "count": len(first_entries),
        "ok_count": sum(e.get("status") == "ok" for e in first_entries),
        "failure_sources": {},
        "ok_backends": {},
    }
    for entry in first_entries:
        fs = entry.get("failure_source") or "none"
        first_summary["failure_sources"][fs] = first_summary["failure_sources"].get(fs, 0) + 1
        if entry.get("status") == "ok":
            backend = entry.get("metrics", {}).get("backend", "unknown")
            first_summary["ok_backends"][backend] = first_summary["ok_backends"].get(backend, 0) + 1

    vace_summary = {
        "count": len(vace_entries),
        "completed_videos": sum(e.get("status") == "ok" for e in vace_entries),
        "prepared_only": sum(e.get("status") == "prepared" for e in vace_entries),
        "failure_sources": {},
    }
    for entry in vace_entries:
        fs = entry.get("failure_source") or "none"
        vace_summary["failure_sources"][fs] = vace_summary["failure_sources"].get(fs, 0) + 1

    failure_counts: dict[str, int] = {}
    for judge in judges:
        fs = judge["failure_source_detail"]
        failure_counts[fs] = failure_counts.get(fs, 0) + 1

    lines = [
        "# E2W v0.1 Physics-IQ Teacher-Grounded Integration",
        "",
        f"Run bundle: `{args.run_dir}`",
        "",
        "## Status",
        "",
        f"- Scope: `{label_source}` planner artifacts + SAM2/code quadmask + first-frame interface + legacy VACE v0 adapter + debug judge/report.",
        f"- Mode executed: `{args.mode}`.",
        "- VACE v0 semantics: quadmask is saved/logged only; legacy Wan-VACE consumes `vace_generation_mask.mp4` where frame 0 is known and frames 1..T are generated.",
        "",
        "## Gates",
        "",
        f"- JSON parse rate: {pct(rate(planner_summary, 'json_parse_ok'))} (target >= 95%)",
        f"- schema valid rate: {pct(rate(planner_summary, 'schema_valid'))} (target >= 90%)",
        f"- physical consequences non-empty: {pct(rate(planner_summary, 'physical_consequences_nonempty'))} (target = 100%)",
        f"- quadmask_spec executable rate: {pct(rate(planner_summary, 'quadmask_spec_executable'))} (target >= 85%)",
        "",
        "## VLM",
        "",
        f"- JSON parse: {planner_summary.get('json_parse_ok', {}).get('count', 0)}/{planner_summary.get('count', 0)}",
        f"- schema valid: {planner_summary.get('schema_valid', {}).get('count', 0)}/{planner_summary.get('count', 0)}",
        f"- operation accuracy: {planner_summary.get('operation_accuracy', {}).get('count', 0)}/{planner_summary.get('count', 0)}",
        f"- primary bbox valid: {planner_summary.get('primary_bbox_valid', {}).get('count', 0)}/{planner_summary.get('count', 0)}",
        f"- primary point valid: {planner_summary.get('primary_point_valid', {}).get('count', 0)}/{planner_summary.get('count', 0)}",
        f"- quadmask_spec executable: {planner_summary.get('quadmask_spec_executable', {}).get('count', 0)}/{planner_summary.get('count', 0)}",
        f"- executor valid: {planner_summary.get('executor_valid', {}).get('count', 0)}/{planner_summary.get('count', 0)}",
        "",
        "## Mask",
        "",
        f"- built valid quadmasks: {mask_summary['mask_valid_count']}/{mask_summary['count']}",
        f"- primary non-empty: {mask_summary['primary_nonempty_count']}/{mask_summary['count']}",
        f"- affected non-empty: {mask_summary['affected_nonempty_count']}/{mask_summary['count']}",
        f"- valid quadmask value sets: {mask_summary['valid_quadmask_value_count']}/{mask_summary['count']}",
        f"- average editable area ratio among valid masks: {mask_summary['avg_editable_area_ratio']:.6f}",
        f"- failure sources: `{json.dumps(mask_summary['failure_sources'], sort_keys=True)}`",
        "",
        "## First Frame",
        "",
        f"- records: {first_summary['count']}",
        f"- ok: {first_summary['ok_count']}",
        f"- ok backends: `{json.dumps(first_summary['ok_backends'], sort_keys=True)}`",
        f"- failure sources: `{json.dumps(first_summary['failure_sources'], sort_keys=True)}`",
        "",
        "## VACE",
        "",
        f"- records: {vace_summary['count']}",
        f"- completed videos: {vace_summary['completed_videos']}",
        f"- prepared-only condition bundles: {vace_summary['prepared_only']}",
        f"- failure sources: `{json.dumps(vace_summary['failure_sources'], sort_keys=True)}`",
        "",
        "## Judge",
        "",
        "- Backend: `artifact_heuristic_no_vlm`; scores are debug placeholders, not paper metrics.",
        f"- failure attribution: `{json.dumps(failure_counts, sort_keys=True)}`",
        "",
        "## Samples",
        "",
        "| sample_id | mode | schema_valid | mask_valid | first_frame | vace | judge_total | failure_source | artifact_path |",
        "|---|---|---:|---:|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {sample_id} | {mode} | {schema_valid} | {mask_valid} | {first_frame_status} | "
            "{vace_status} | {judge_total} | {failure_source} | `{artifact_path}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Next Bottleneck",
            "",
            "The current teacher-grounded executor path is contract-executable. The remaining v0.1 bottlenecks are visual spot-checking the teacher/SAM2 masks, handling large-edit samples such as 0239, and replacing the debug first-frame backend with measured Qwen-Image-Edit runs.",
        ]
    )
    write_text(args.run_dir / "report.md", "\n".join(lines) + "\n")
    write_json(
        args.run_dir / "judge" / "summary.json",
        {
            "count": len(judges),
            "backend": "artifact_heuristic_no_vlm",
            "failure_counts": failure_counts,
            "not_final_score": True,
        },
    )
    print(f"wrote {args.run_dir / 'report.md'}")


if __name__ == "__main__":
    main()

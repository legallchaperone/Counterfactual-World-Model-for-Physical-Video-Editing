#!/usr/bin/env python3
"""Deprecated stale-artifact v0.2 bundle creator.

Use run_v02_qwen_vace_smoke.py instead. This historical script used symlinks to
reuse v0.1 planner, mask, and prompt artifacts, which violates the v0.2 smoke
freshness contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import ensure_run_dirs, load_manifest, write_json  # noqa: E402


DEFAULT_SOURCE_RUN = Path("/data/cwx/E2W/runs/e2w_v0_1_physics_iq_teacher_grounded_eval")
DEFAULT_OUT_RUN = Path("/data/cwx/E2W/runs/e2w_v0_2_qwen_vace_smoke")
DEFAULT_SOURCE_MODE = "mode_a_lite_teacher_grounded"
DEFAULT_MODE = "mode_v0_2_qwen_vace_smoke"
DEFAULT_SAMPLES = ["0052", "0056", "0070", "0076", "0112", "0077", "0341", "0128"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--out-run", type=Path, default=DEFAULT_OUT_RUN)
    parser.add_argument("--source-mode", default=DEFAULT_SOURCE_MODE)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--sample-id", action="append", default=[])
    return parser.parse_args()


def symlink_file(src: str | Path, dst: Path) -> str:
    src_path = Path(src).resolve()
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src_path, dst)
    return str(dst)


def index_manifest(entries: list[dict[str, Any]]) -> dict[tuple[str | None, str, str | None], dict[str, Any]]:
    out: dict[tuple[str | None, str, str | None], dict[str, Any]] = {}
    for entry in entries:
        sample_id = entry.get("sample_id")
        if sample_id:
            out[(entry.get("stage"), sample_id, entry.get("mode"))] = entry
    return out


def load_qc(source_run: Path) -> dict[str, dict[str, str]]:
    path = source_run / "qc" / "mask_qc.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def copy_planner_entry(
    out_run: Path,
    mode: str,
    source_entry: dict[str, Any],
    sample_id: str,
) -> dict[str, Any]:
    source_paths = source_entry["paths"]
    src_dir = out_run / "source" / sample_id
    pred_dir = out_run / "planner_pred" / sample_id
    paths = {
        "original_video": symlink_file(source_paths["original_video"], src_dir / "original_video.mp4"),
        "text_query": symlink_file(source_paths["text_query"], src_dir / "text_query.txt"),
        "first_frame": symlink_file(source_paths["first_frame"], src_dir / "first_frame.png"),
        "raw_teacher": symlink_file(source_paths["raw_teacher"], pred_dir / "raw.teacher.json"),
        "edit_plan": symlink_file(source_paths["edit_plan"], pred_dir / "edit_plan.teacher.json"),
        "quadmask_spec": symlink_file(source_paths["quadmask_spec"], pred_dir / "quadmask_spec.teacher.json"),
        "vace_prompt": symlink_file(source_paths["vace_prompt"], pred_dir / "vace_prompt.txt"),
        "planner_eval": symlink_file(source_paths["planner_eval"], pred_dir / "planner_eval.json"),
    }
    symlink_file(source_paths["edit_plan"], pred_dir / "edit_plan.json")
    symlink_file(source_paths["quadmask_spec"], pred_dir / "quadmask_spec.json")
    return {
        **source_entry,
        "mode": mode,
        "paths": paths,
        "metrics": {**source_entry.get("metrics", {}), "v0_2_subset": True},
    }


def copy_mask_entry(
    out_run: Path,
    mode: str,
    source_entry: dict[str, Any],
    sample_id: str,
) -> dict[str, Any]:
    source_paths = source_entry["paths"]
    mask_dir = out_run / "masks" / sample_id / mode
    paths: dict[str, str] = {}
    mapping = {
        "quadmask_spec": "quadmask_spec.teacher.json",
        "source_clip": "source_clip.mp4",
        "primary_mask": "primary_mask.mp4",
        "affected_mask": "affected_mask.mp4",
        "editable_mask": "editable_mask.mp4",
        "quadmask_npy": "quadmask.npy",
        "quadmask": "quadmask.mp4",
        "mask_eval": "mask_eval.json",
    }
    for key, filename in mapping.items():
        if key in source_paths:
            paths[key] = symlink_file(source_paths[key], mask_dir / filename)
    contact_src = source_paths.get("debug_contact_sheet")
    if contact_src:
        contact_dst = out_run / "contact_sheets" / sample_id / f"{mode}_debug_contact_sheet.png"
        paths["debug_contact_sheet"] = symlink_file(contact_src, contact_dst)
    return {
        **source_entry,
        "mode": mode,
        "paths": paths,
        "metrics": {**source_entry.get("metrics", {}), "v0_2_subset": True},
    }


def main() -> None:
    raise SystemExit(
        "Deprecated: create_v02_qwen_vace_smoke_bundle.py symlinks old v0.1 artifacts. "
        "Use tools/run_v02_qwen_vace_smoke.py to regenerate planner, mask, first-frame, "
        "and VACE artifacts for each v0.2 smoke run."
    )
    args = parse_args()
    sample_ids = args.sample_id or DEFAULT_SAMPLES
    ensure_run_dirs(args.out_run)
    qc = load_qc(args.source_run)
    source_manifest = load_manifest(args.source_run / "manifest.jsonl")
    indexed = index_manifest(source_manifest)
    entries: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        planner = indexed.get(("planner_eval", sample_id, args.source_mode))
        mask = indexed.get(("mask_builder", sample_id, args.source_mode))
        if planner is None or mask is None:
            raise ValueError(f"Missing source planner/mask entry for {sample_id}")
        if mask.get("status") != "ok":
            raise ValueError(f"Source mask is not ok for {sample_id}: {mask.get('status')}")
        entries.append(copy_planner_entry(args.out_run, args.mode, planner, sample_id))
        entries.append(copy_mask_entry(args.out_run, args.mode, mask, sample_id))
        selected.append(
            {
                "sample_id": sample_id,
                "user_edit_query": qc.get(sample_id, {}).get("user_edit_query", ""),
                "primary_ok": qc.get(sample_id, {}).get("primary_ok", ""),
                "affected_ok": qc.get(sample_id, {}).get("affected_ok", ""),
                "overlap_ok": qc.get(sample_id, {}).get("overlap_ok", ""),
                "selection_role": {
                    "0052": "clean_primary",
                    "0056": "clean_primary",
                    "0070": "clean_primary",
                    "0076": "clean_primary",
                    "0112": "clean_primary",
                    "0077": "affected_coarse",
                    "0341": "affected_coarse",
                    "0128": "fluid_stress",
                }.get(sample_id, "selected"),
            }
        )

    with (args.out_run / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    write_json(
        args.out_run / "selection.json",
        {
            "source_run": str(args.source_run),
            "source_mode": args.source_mode,
            "mode": args.mode,
            "sample_ids": sample_ids,
            "selected": selected,
            "note": "v0.2 primary-good Qwen-Image-Edit + legacy VACE runtime smoke subset.",
        },
    )
    print(json.dumps({"out_run": str(args.out_run), "mode": args.mode, "samples": sample_ids}, indent=2))


if __name__ == "__main__":
    main()

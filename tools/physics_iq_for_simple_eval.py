#!/usr/bin/env python3
"""Build and review the Physics-IQ simple E2W visual eval set.

This tool is intentionally separate from the old v0 Physics-IQ judge. It builds
a leakage-filtered manifest, validates current E2W artifact contracts, creates
strict VLM judge prompts, and launches a Gradio human review dashboard.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent

DEFAULT_DESCRIPTIONS_CSV = Path("/data/cwx/physics-iq-benchmark-src/descriptions/descriptions.csv")
DEFAULT_PHYSICS_IQ_ROOT = Path("/data/cwx/physics-iq/physics-IQ-benchmark")
DEFAULT_OUT_ROOT = Path("/data/cwx/E2W/data/physics_iq_for_simple_eval")
DEFAULT_RUN_ROOT = Path("/data/cwx/E2W/runs/physics_iq_for_simple_eval")
DEFAULT_TEACHER_LABEL_DIR = Path("/data/cwx/E2W/data/physics_iq_vlm_sft/teacher_labels/parsed")
DEFAULT_OLD_RUN_DIR = Path("/data/cwx/E2W/runs/e2w_v0_physics_iq")
DEFAULT_PLANNER_MANIFESTS = (
    Path("/data/cwx/E2W/data/planner_sft_v8_seed_v3/train.jsonl"),
    Path("/data/cwx/E2W/data/planner_sft_v8_seed_v3/eval.jsonl"),
    Path("/data/cwx/E2W/data/planner_sft_v8/train.jsonl"),
    Path("/data/cwx/E2W/data/planner_sft_v8/eval.jsonl"),
)

CURRENT_VACE_INPUT_KEYS = (
    "vace_conditioning_video",
    "quadmask_npy",
    "generation_mask",
    "operation",
    "vace_prompt",
    "frame_num",
)

VLM_JUDGE_SCHEMA = {
    "target_success": 0,
    "physical_effect_success": 0,
    "preservation_success": 0,
    "temporal_consistency": 0,
    "major_artifacts": 0,
    "overall_pass": 0,
    "short_reason": "",
}

HUMAN_E2W_DEFAULTS = {
    "human_e2w_target_success": 0,
    "human_e2w_physical_effect_success": 0,
    "human_e2w_preservation_success": 0,
    "human_e2w_temporal_consistency": 0,
    "human_e2w_major_artifacts": 0,
    "human_e2w_overall_pass": 0,
    "human_e2w_notes": "",
}

HUMAN_VLM_DEFAULTS = {
    "human_vlm_correct_about_target": 0,
    "human_vlm_correct_about_physics": 0,
    "human_vlm_correct_about_preservation": 0,
    "human_vlm_overall_agree": 0,
    "human_vlm_missed_failure": 0,
    "human_vlm_notes": "",
}


@dataclass(frozen=True)
class BenchmarkSpec:
    physics_iq_id: str
    operation: str
    user_prompt: str
    target_object: str
    expected_visible_outcome: str
    expected_physical_effect: str
    must_preserve: tuple[str, ...]


BENCHMARK_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        "0018",
        "remove",
        "Remove the tennis ball released above the green kinetic sand.",
        "tennis ball",
        "The tennis ball is absent while the green kinetic sand, grabber, table, and wall remain visible.",
        "The sand should not be struck or dented by the falling ball.",
        ("green kinetic sand", "blue grabber tool", "wooden table", "plain wall"),
    ),
    BenchmarkSpec(
        "0037",
        "remove",
        "Remove the black balloon attached to the air pump hose.",
        "black balloon",
        "The black balloon is absent while the pump hose, table, and wall remain visible.",
        "No balloon expansion should occur.",
        ("air pump hose", "table", "plain wall", "camera framing"),
    ),
    BenchmarkSpec(
        "0048",
        "remove",
        "Remove the white domino before it drops into the blue mug.",
        "white domino",
        "The white domino is absent while the blue mug and grabber tool remain visible.",
        "The dark liquid should remain still with no splash or ripple from the domino.",
        ("blue mug", "dark liquid", "grabber tool", "wooden surface"),
    ),
    BenchmarkSpec(
        "0053",
        "remove",
        "Remove the two released metal balls from the Newton's cradle.",
        "two released metal balls",
        "The released metal balls are absent while the cradle frame and remaining balls stay visible.",
        "The cradle should not receive an impact from the removed balls.",
        ("Newton's cradle frame", "remaining metal balls", "table", "grabber tool"),
    ),
    BenchmarkSpec(
        "0063",
        "remove",
        "Remove the yellow rubber duck from the table.",
        "yellow rubber duck",
        "The duck is absent from the tabletop.",
        "The tabletop remains stable with no new moving object in the duck location.",
        ("wooden table", "plain background", "lighting"),
    ),
    BenchmarkSpec(
        "0246",
        "remove",
        "Remove the white domino before it falls into the liquid.",
        "white domino",
        "The white domino is absent while the mug and liquid remain visible.",
        "The liquid should not splash or ripple from a domino impact.",
        ("blue mug", "dark liquid", "wooden surface", "grabber tool"),
    ),
    BenchmarkSpec(
        "0038",
        "add",
        "Add a red clamp around the black balloon so the balloon cannot expand normally.",
        "red clamp",
        "A red clamp appears around the balloon.",
        "The balloon expansion should be visibly constrained by the clamp.",
        ("black balloon", "air pump hose", "table", "plain wall"),
    ),
    BenchmarkSpec(
        "0050",
        "add",
        "Add a white domino bridge across the gap between the two domino rows.",
        "white domino bridge",
        "An extra white domino appears bridging the gap between the rows.",
        "The falling domino chain should be able to continue across the gap through the added domino.",
        ("two domino rows", "rotating stick", "wooden table", "camera framing"),
    ),
    BenchmarkSpec(
        "0051",
        "add",
        "Add a heavy red block in front of the first domino so the rotating stick cannot start the domino chain normally.",
        "heavy red block",
        "A red block appears at the first domino contact area.",
        "The rotating stick or first domino should be blocked or visibly altered.",
        ("domino rows", "rotating platform", "wooden table", "background"),
    ),
    BenchmarkSpec(
        "0248",
        "add",
        "Add a black domino in the gap between the two domino rows.",
        "black domino",
        "A black domino appears in the gap between the rows.",
        "The domino chain should transfer across the gap through the added domino.",
        ("two domino rows", "rotating stick", "wooden table", "camera framing"),
    ),
    BenchmarkSpec(
        "0251",
        "add",
        "Add a small clamp holding the released cradle balls so they cannot swing into the other balls.",
        "small clamp",
        "A clamp appears holding the released Newton's cradle balls.",
        "The cradle collision should be prevented or visibly altered.",
        ("Newton's cradle frame", "metal balls", "table", "grabber tool"),
    ),
    BenchmarkSpec(
        "0261",
        "add",
        "Add a rolling orange ball aimed at the yellow duck.",
        "rolling orange ball",
        "An orange ball appears rolling toward the duck.",
        "The duck should be contacted or visibly affected by the added rolling ball.",
        ("yellow duck", "wooden table", "plain background", "camera framing"),
    ),
)


class BenchmarkError(ValueError):
    """Raised when benchmark artifacts violate the accepted plan."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_media_probe_path() -> None:
    """Make Gradio video probing work in non-interactive launcher envs."""
    if shutil.which("ffprobe"):
        return

    candidates = [
        Path("/usr/bin"),
        Path("/usr/local/bin"),
        Path("/home/cwx/.local/bin"),
        Path("/data/cwx/conda/envs/edit2world-phase1-real/bin"),
    ]
    try:
        import imageio_ffmpeg

        candidates.append(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
    except Exception:
        pass
    candidates.extend(sorted(Path("/data/cwx/conda/pkgs").glob("ffmpeg-*/bin"), reverse=True))

    for directory in candidates:
        if (directory / "ffprobe").exists():
            os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"
            return


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_descriptions(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            match = re.match(r"(\d{4})_", row.get("scenario", ""))
            if match:
                rows[match.group(1)] = row
    return rows


def official_ids(descriptions_csv: Path) -> set[str]:
    return set(load_descriptions(descriptions_csv))


def collect_leakage_ids(
    *,
    descriptions_csv: Path = DEFAULT_DESCRIPTIONS_CSV,
    teacher_label_dir: Path = DEFAULT_TEACHER_LABEL_DIR,
    old_run_dir: Path = DEFAULT_OLD_RUN_DIR,
    planner_manifests: tuple[Path, ...] = DEFAULT_PLANNER_MANIFESTS,
) -> dict[str, list[str]]:
    ids = official_ids(descriptions_csv)
    evidence: dict[str, list[str]] = {}

    if teacher_label_dir.exists():
        for path in teacher_label_dir.glob("*.json"):
            if path.stem in ids:
                evidence.setdefault(path.stem, []).append(str(path))

    if old_run_dir.exists():
        for path in old_run_dir.glob("*/*"):
            if path.name in ids:
                evidence.setdefault(path.name, []).append(str(path))

    for manifest in planner_manifests:
        for row in read_jsonl(manifest):
            row_id = str(row.get("id", ""))
            match = re.search(r"(?<!\d)(\d{4})(?!\d)", row_id)
            if match and match.group(1) in ids:
                evidence.setdefault(match.group(1), []).append(str(manifest))
    return evidence


def find_source_full_video(physics_iq_root: Path, row: dict[str, str]) -> Path | None:
    scenario = row["scenario"]
    sid = scenario[:4]
    candidates: list[Path] = []
    for root in [
        physics_iq_root / "full-videos",
        physics_iq_root / "full-videos" / "take-1",
        physics_iq_root / "full-videos" / "take-2",
    ]:
        if root.exists():
            candidates.extend(root.rglob(f"{sid}_*.mp4"))
    if not candidates:
        # The official bucket filename convention may include "full-videos" in
        # the filename or may use the descriptions.csv scenario basename.
        for root in [physics_iq_root / "full-videos"]:
            if root.exists():
                candidates.extend(root.rglob(f"*{scenario}"))
    candidates = [p for p in candidates if is_allowed_source_video_path(p)]
    if candidates:
        return sorted(candidates)[0]

    for root in [
        physics_iq_root / "split-videos" / "testing" / "30FPS",
        physics_iq_root / "split-videos" / "testing" / "16FPS",
    ]:
        if root.exists():
            candidates.extend(root.rglob(f"{sid}_*.mp4"))
    candidates = [p for p in candidates if is_allowed_source_video_path(p)]
    return sorted(candidates)[0] if candidates else None


def expected_gcloud_uri(row: dict[str, str], fps: int = 30) -> str:
    scenario = row["scenario"]
    take = "take-2" if "_take-2_" in scenario else "take-1"
    sid = scenario[:4]
    return f"gs://physics-iq-benchmark/full-videos/{take}/{fps}FPS/{sid}_*.mp4"


def local_full_video_dir(physics_iq_root: Path, row: dict[str, str], fps: int = 30) -> Path:
    take = "take-2" if "_take-2_" in row["scenario"] else "take-1"
    return physics_iq_root / "full-videos" / take / f"{fps}FPS"


def is_allowed_source_video_path(path: Path) -> bool:
    text = str(path)
    return (
        "/full-videos/" in text
        or (
            "/physics-IQ-benchmark/split-videos/testing/" in text
            and "testing-videos" in path.name
            and "conditioning-videos" not in path.name
        )
    )


def is_full_video_path(path: Path) -> bool:
    return is_allowed_source_video_path(path)


def source_kind(path: Path | None) -> str:
    if path is None:
        return "missing"
    text = str(path)
    if "/full-videos/" in text:
        return "physics_iq_full_video"
    if "/physics-IQ-benchmark/split-videos/testing/" in text:
        return "physics_iq_official_testing_video"
    return "invalid"


def converted_video_path(out_root: Path, spec: BenchmarkSpec) -> Path:
    return out_root / "converted" / f"{spec.physics_iq_id}_{spec.operation}.mp4"


def build_vlm_judge_prompt(row: dict[str, Any]) -> str:
    return (
        "You are judging an Edit2World counterfactual video edit.\n\n"
        "Inputs:\n"
        "1. Original video.\n"
        "2. Edited video.\n"
        f"3. Operation: {row['operation']}\n"
        f"4. Edit request: {row['user_prompt']}\n"
        f"5. Expected visible outcome: {row['expected_visible_outcome']}\n"
        f"6. Expected physical effect: {row['expected_physical_effect']}\n"
        f"7. Must preserve: {', '.join(row['must_preserve'])}\n\n"
        "Answer only JSON with exactly this shape:\n"
        f"{json.dumps(VLM_JUDGE_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        "Rules:\n"
        "- Judge only the edited video against the edit request and expected outcome.\n"
        "- Do not reward the video merely for being non-black or plausible.\n"
        "- For add, the added object must appear and affect the described physical interaction.\n"
        "- For remove, the target must disappear and directly affected regions should change consistently.\n"
        "- Penalize if unrelated objects disappear, the scene identity changes, or the edit only works in one frame.\n"
    )


def make_manifest_row(
    spec: BenchmarkSpec,
    *,
    description: dict[str, str],
    source_full_video: Path | None,
    out_root: Path,
    leakage_evidence: dict[str, list[str]],
) -> dict[str, Any]:
    leaked = spec.physics_iq_id in leakage_evidence
    row = {
        "sample_id": f"piq_simple_eval_{spec.physics_iq_id}_{spec.operation}",
        "physics_iq_id": spec.physics_iq_id,
        "operation": spec.operation,
        "user_prompt": spec.user_prompt,
        "target_object": spec.target_object,
        "expected_visible_outcome": spec.expected_visible_outcome,
        "expected_physical_effect": spec.expected_physical_effect,
        "must_preserve": list(spec.must_preserve),
        "physics_iq_description": description.get("description", ""),
        "physics_iq_category": description.get("category", ""),
        "physics_iq_scenario": description.get("scenario", ""),
        "source_full_video": str(source_full_video) if source_full_video else None,
        "converted_video": str(converted_video_path(out_root, spec)),
        "source_metadata": {
            "dataset": "Physics-IQ",
            "official_description_csv": str(DEFAULT_DESCRIPTIONS_CSV),
            "expected_full_video_gs_uri": expected_gcloud_uri(description),
            "source_kind": source_kind(source_full_video),
            "allowed_source_roots": [
                "physics-IQ-benchmark/full-videos",
                "physics-IQ-benchmark/split-videos/testing",
            ],
            "review_proxy_forbidden": True,
            "old_e2w_run_forbidden": True,
        },
        "leakage_exclusion_evidence": {
            "leakage_checked": True,
            "leaked": leaked,
            "matched_paths": leakage_evidence.get(spec.physics_iq_id, []),
        },
        "vlm_judge_prompt": "",
    }
    row["vlm_judge_prompt"] = build_vlm_judge_prompt(row)
    return row


def build_manifest(
    *,
    descriptions_csv: Path = DEFAULT_DESCRIPTIONS_CSV,
    physics_iq_root: Path = DEFAULT_PHYSICS_IQ_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
    allow_missing_source: bool = False,
) -> list[dict[str, Any]]:
    descriptions = load_descriptions(descriptions_csv)
    leakage_evidence = collect_leakage_ids(descriptions_csv=descriptions_csv)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    seen_ids: set[str] = set()
    for spec in BENCHMARK_SPECS:
        if spec.physics_iq_id in seen_ids:
            errors.append(f"duplicate benchmark id: {spec.physics_iq_id}")
        seen_ids.add(spec.physics_iq_id)
        description = descriptions.get(spec.physics_iq_id)
        if not description:
            errors.append(f"missing descriptions.csv row for {spec.physics_iq_id}")
            continue
        if spec.physics_iq_id in leakage_evidence:
            errors.append(f"selected id {spec.physics_iq_id} appears in leakage sources: {leakage_evidence[spec.physics_iq_id]}")
        source = find_source_full_video(physics_iq_root, description)
        if source is None and not allow_missing_source:
            errors.append(f"missing full-video source for {spec.physics_iq_id}; expected {expected_gcloud_uri(description)}")
        if source is not None and not is_allowed_source_video_path(source):
            errors.append(f"source is not an allowed official Physics-IQ source for {spec.physics_iq_id}: {source}")
        rows.append(make_manifest_row(spec, description=description, source_full_video=source, out_root=out_root, leakage_evidence=leakage_evidence))

    if len(rows) != 12:
        errors.append(f"manifest must contain 12 rows, got {len(rows)}")
    if errors:
        raise BenchmarkError("\n".join(errors))
    return rows


def convert_source_video(source: Path, dest: Path, *, fps: int = 12, width: int = 832, height: int = 480, frame_num: int = 21) -> None:
    if not is_allowed_source_video_path(source):
        raise BenchmarkError(f"refusing to convert non-official Physics-IQ source: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        vf,
        "-r",
        str(fps),
        "-frames:v",
        str(frame_num),
        "-an",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def materialize_converted_videos(rows: list[dict[str, Any]], *, fps: int = 12, width: int = 832, height: int = 480, frame_num: int = 21) -> None:
    for row in rows:
        source = Path(row["source_full_video"] or "")
        dest = Path(row["converted_video"])
        if not source.exists():
            raise BenchmarkError(f"source_full_video missing for {row['sample_id']}: {source}")
        convert_source_video(source, dest, fps=fps, width=width, height=height, frame_num=frame_num)


def download_full_videos(rows: list[dict[str, Any]], physics_iq_root: Path, *, descriptions_csv: Path = DEFAULT_DESCRIPTIONS_CSV, fps: int = 30) -> None:
    descriptions = load_descriptions(descriptions_csv)
    for row in rows:
        desc = descriptions[row["physics_iq_id"]]
        dest = local_full_video_dir(physics_iq_root, desc, fps=fps)
        dest.mkdir(parents=True, exist_ok=True)
        cmd = ["gcloud", "storage", "cp", expected_gcloud_uri(desc, fps=fps), str(dest)]
        subprocess.run(cmd, check=True)


def validate_manifest_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(rows) != 12:
        errors.append(f"manifest must contain 12 rows, got {len(rows)}")
    ids = [str(row.get("physics_iq_id")) for row in rows]
    if len(set(ids)) != len(ids):
        errors.append("manifest contains duplicate physics_iq_id values")
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        operation = row.get("operation")
        if operation not in {"add", "remove"}:
            errors.append(f"{sample_id}: invalid operation {operation!r}")
        if row.get("leakage_exclusion_evidence", {}).get("leaked"):
            errors.append(f"{sample_id}: selected id is marked leaked")
        source = row.get("source_full_video")
        if source and not is_allowed_source_video_path(Path(source)):
            errors.append(f"{sample_id}: source_full_video is not an allowed official Physics-IQ source: {source}")
        for key in ["user_prompt", "target_object", "expected_visible_outcome", "expected_physical_effect"]:
            if not str(row.get(key) or "").strip():
                errors.append(f"{sample_id}: missing {key}")
        if not isinstance(row.get("must_preserve"), list) or not row["must_preserve"]:
            errors.append(f"{sample_id}: must_preserve must be a non-empty list")
        if "Answer only JSON" not in str(row.get("vlm_judge_prompt") or ""):
            errors.append(f"{sample_id}: missing strict VLM judge prompt")
    return errors


def load_vlm_judge(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_vlm_judge(data)
    return data


def validate_vlm_judge(data: dict[str, Any]) -> None:
    required = set(VLM_JUDGE_SCHEMA)
    missing = sorted(required - set(data))
    extra = sorted(set(data) - required)
    if missing or extra:
        raise BenchmarkError(f"VLM judge JSON keys mismatch missing={missing} extra={extra}")
    for key in required - {"short_reason"}:
        if data[key] not in (0, 1):
            raise BenchmarkError(f"VLM judge field {key} must be 0 or 1")
    if not isinstance(data["short_reason"], str):
        raise BenchmarkError("VLM judge short_reason must be a string")


def find_sample_artifacts(run_root: Path, row: dict[str, Any]) -> dict[str, str | None]:
    sample_id = row["sample_id"]
    candidates = [
        run_root / sample_id,
        run_root / row["physics_iq_id"],
        run_root,
    ]
    edited_video = None
    metadata = None
    vlm_judge = None
    for root in candidates:
        if not root.exists():
            continue
        for name in ("edited_video.mp4", f"edited_video_{row['physics_iq_id']}.mp4"):
            p = root / name
            if p.exists():
                edited_video = str(p)
                break
        for name in ("metadata.json", "summary.json"):
            p = root / name
            if p.exists():
                metadata = str(p)
                break
        for name in ("vlm_judge.json", "judge.json"):
            p = root / name
            if p.exists():
                vlm_judge = str(p)
                break
    return {"edited_video": edited_video, "metadata": metadata, "vlm_judge": vlm_judge}


def validate_run_contract(run_root: Path, manifest_rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in manifest_rows:
        artifacts = find_sample_artifacts(run_root, row)
        meta_path = artifacts.get("metadata")
        if not meta_path:
            errors.append(f"{row['sample_id']}: missing metadata.json/summary.json")
            continue
        data = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        runtime = data.get("vace_runtime_inputs") or data.get("runtime_inputs") or {}
        missing = [key for key in CURRENT_VACE_INPUT_KEYS if key not in runtime]
        if missing:
            errors.append(f"{row['sample_id']}: missing current VACE runtime keys {missing}")
        if data.get("source_video_passed_to_vace") is not False:
            errors.append(f"{row['sample_id']}: source_video_passed_to_vace must be false")
        conditioning_meta = data.get("vace_conditioning_video") or {}
        if conditioning_meta.get("future_frames_source_video_used") is not False:
            errors.append(f"{row['sample_id']}: vace_conditioning_video future frames must not use source video")
        if conditioning_meta.get("future_frames_are_zero_filled") is not True:
            errors.append(f"{row['sample_id']}: vace_conditioning_video future frames must be zero-filled")
        gen_values = data.get("generation_mask_values")
        gen_meta = data.get("generation_mask_metadata") or {}
        if gen_values is None:
            gen_values = gen_meta.get("generation_mask_values")
        if gen_values != [255]:
            errors.append(f"{row['sample_id']}: generation_mask_values must be [255], got {gen_values}")
        if data.get("control_branch_checkpoint_loaded") is not True:
            errors.append(f"{row['sample_id']}: control_branch_checkpoint_loaded must be true")
        if data.get("trained_control_branch_used") is not True:
            errors.append(f"{row['sample_id']}: trained_control_branch_used must be true")
        if data.get("control_branch_installed_in_forward_vace") is not True:
            errors.append(f"{row['sample_id']}: control_branch_installed_in_forward_vace must be true")
        gate = data.get("control_branch_gate")
        if not isinstance(gate, (int, float)) or float(gate) == 0.0:
            errors.append(f"{row['sample_id']}: control_branch_gate must be nonzero, got {gate}")
        if artifacts.get("vlm_judge"):
            load_vlm_judge(Path(artifacts["vlm_judge"]))
    return errors


def append_human_judgment(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"created_at": utc_now(), **row}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_latest_human_judgments(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        if sample_id:
            latest[sample_id] = row
    return latest


def export_human_summary(judgment_path: Path, out_json: Path, out_csv: Path) -> None:
    latest = load_latest_human_judgments(judgment_path)
    rows = list(latest.values())
    write_json(out_json, {"count": len(rows), "rows": rows})
    if rows:
        keys = sorted({key for row in rows for key in row})
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, log_path: Path) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update({k: v for k, v in env.items() if v is not None})
    proc = subprocess.run(cmd, cwd=str(cwd), env=merged, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    write_json(log_path, {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
    return proc


def extract_video_frames(video_path: Path, frame_dir: Path) -> list[Path]:
    import cv2

    frame_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frame_dir.glob("*.jpg"))
    if existing:
        return existing
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise BenchmarkError(f"failed to open video for frame extraction: {video_path}")
    paths: list[Path] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = frame_dir / f"{idx:05d}.jpg"
            if not cv2.imwrite(str(out), frame):
                raise BenchmarkError(f"failed to write extracted frame: {out}")
            paths.append(out)
            idx += 1
    finally:
        cap.release()
    if not paths:
        raise BenchmarkError(f"video has no readable frames: {video_path}")
    return paths


def write_remove_split(row: dict[str, Any], sample_dir: Path) -> Path:
    from e2w_v0_common import build_counterfactual_planner_user_prompt

    frames = extract_video_frames(Path(row["converted_video"]), sample_dir / "source_frames")
    instruction = f"Remove only the {row['target_object']}. Original request: {row['user_prompt']}"
    extra_rules = (
        f"target_ref must refer only to the target object: {row['target_object']}. "
        f"Do not include preserved/context objects in target_ref: {', '.join(row['must_preserve'])}. "
        "The counterfactual text may describe preserved/context objects, but must not make them part of the removed target."
    )
    prompt = build_counterfactual_planner_user_prompt(row["sample_id"], instruction, extra_rules=extra_rules)
    split_row = {
        "id": row["sample_id"],
        "video_id": row["sample_id"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(frames[0])},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "metadata": {
            "physics_iq_id": row["physics_iq_id"],
            "operation": "remove",
            "converted_video": row["converted_video"],
        },
    }
    split = sample_dir / "remove_eval.jsonl"
    write_jsonl(split, [split_row])
    return split


def normalize_add_artifacts(sample_dir: Path, row: dict[str, Any]) -> None:
    meta_path = sample_dir / "metadata.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data.update(
        {
            "sample_id": row["sample_id"],
            "physics_iq_id": row["physics_iq_id"],
            "expected_visible_outcome": row["expected_visible_outcome"],
            "expected_physical_effect": row["expected_physical_effect"],
            "must_preserve": row["must_preserve"],
            "visual_quality_evaluated": False,
        }
    )
    gen = data.get("generation_mask") or {}
    data["generation_mask_values"] = gen.get("generation_mask_values") or gen.get("values")
    data["generation_mask_is_full_domain"] = gen.get("generation_mask_is_full_domain")
    write_json(meta_path, data)


def normalize_remove_artifacts(sample_dir: Path, row: dict[str, Any]) -> None:
    summary_path = sample_dir / "remove_pipeline" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = summary.get("results") or []
    if len(results) != 1 or results[0].get("status") != "ok":
        raise BenchmarkError(f"remove pipeline did not produce exactly one ok result: {summary_path}")
    result = results[0]
    if not result.get("vace_output_path"):
        raise BenchmarkError(
            f"remove pipeline did not produce edited video for {row['sample_id']}: "
            f"{result.get('vace_prompt_error') or result.get('error') or result.get('vace_info')}"
        )
    edited = Path(result["vace_output_path"])
    if not edited.exists():
        raise BenchmarkError(f"remove pipeline edited video missing: {edited}")
    shutil.copy2(edited, sample_dir / "edited_video.mp4")
    runtime = result.get("vace_runtime_inputs") or {}
    if runtime.get("quadmask_npy"):
        shutil.copy2(runtime["quadmask_npy"], sample_dir / "quadmask.npy")
    if runtime.get("generation_mask"):
        shutil.copy2(runtime["generation_mask"], sample_dir / "generation_mask.mp4")
    runtime_meta = (result.get("vace_info") or {}).get("vace_runtime_input_metadata") or {}
    generation_meta = runtime_meta.get("generation_mask") or {}
    conditioning_meta = runtime_meta.get("vace_conditioning_video") or {}
    context_info = ((result.get("vace_info") or {}).get("context_info") or {})
    metadata = {
        "sample_id": row["sample_id"],
        "physics_iq_id": row["physics_iq_id"],
        "operation": "remove",
        "source_video_for_upstream_context": row["converted_video"],
        "source_video_passed_to_vace": False,
        "user_prompt": row["user_prompt"],
        "target_object": row["target_object"],
        "expected_visible_outcome": row["expected_visible_outcome"],
        "expected_physical_effect": row["expected_physical_effect"],
        "must_preserve": row["must_preserve"],
        "visual_quality_evaluated": False,
        "vace_conditioning_video": conditioning_meta,
        "vace_prompt": result.get("vace_prompt"),
        "vace_runtime_inputs": runtime,
        "generation_mask_values": generation_meta.get("generation_mask_values"),
        "generation_mask_is_full_domain": generation_meta.get("generation_mask_is_full_domain"),
        "quadmask": result.get("quadmask_metadata"),
        "control_branch": context_info,
        "control_branch_checkpoint_loaded": bool(context_info.get("control_branch_checkpoint_loaded")),
        "trained_control_branch_used": bool(context_info.get("trained_control_branch_used")),
        "control_branch_step": context_info.get("control_branch_step"),
        "control_branch_gate": context_info.get("control_branch_gate"),
        "control_branch_installed_in_forward_vace": bool(context_info.get("control_branch_installed_in_forward_vace")),
        "pipeline_result": result,
    }
    write_json(sample_dir / "metadata.json", metadata)


def run_pipeline_row(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    sample_dir = args.run_root / row["sample_id"]
    if args.skip_existing and (sample_dir / "edited_video.mp4").exists() and (sample_dir / "metadata.json").exists():
        return {"sample_id": row["sample_id"], "status": "skipped_existing", "run_dir": str(sample_dir)}
    sample_dir.mkdir(parents=True, exist_ok=True)
    if row["operation"] == "add":
        if not args.add_planner_adapter:
            raise BenchmarkError("add rows require --add-planner-adapter (no archived default for the unified add interface)")
        cmd = [
            sys.executable,
            str(TOOLS / "e2w_add.py"),
            "--source-video",
            row["converted_video"],
            "--user-prompt",
            row["user_prompt"],
            "--sample-id",
            row["sample_id"],
            "--run-dir",
            str(sample_dir),
            "--planner-adapter",
            str(args.add_planner_adapter),
            "--vace-sample-steps",
            str(args.vace_sample_steps),
            "--control-branch-checkpoint",
            str(args.control_branch_checkpoint),
        ]
        proc = run_command(cmd, cwd=ROOT, env={"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices}, log_path=sample_dir / "pipeline_command.json")
        if proc.returncode != 0:
            raise BenchmarkError(f"add pipeline failed for {row['sample_id']}; see {sample_dir / 'pipeline_command.json'}")
        normalize_add_artifacts(sample_dir, row)
    elif row["operation"] == "remove":
        split = write_remove_split(row, sample_dir)
        cmd = [
            sys.executable,
            str(TOOLS / "e2w_remove.py"),
            "--eval-jsonl",
            str(split),
            "--output-dir",
            str(sample_dir / "remove_pipeline"),
            "--sample-count",
            "1",
            "--seed",
            "0",
            "--vace-sample-steps",
            str(args.vace_sample_steps),
            "--control-branch-checkpoint",
            str(args.control_branch_checkpoint),
        ]
        proc = run_command(cmd, cwd=ROOT, env={"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices}, log_path=sample_dir / "pipeline_command.json")
        if proc.returncode != 0:
            raise BenchmarkError(f"remove pipeline failed for {row['sample_id']}; see {sample_dir / 'pipeline_command.json'}")
        normalize_remove_artifacts(sample_dir, row)
    else:
        raise BenchmarkError(f"unsupported operation for {row['sample_id']}: {row['operation']}")
    return {"sample_id": row["sample_id"], "status": "ok", "run_dir": str(sample_dir)}


def make_dashboard(manifest_path: Path, run_root: Path, judgment_path: Path) -> Any:
    import gradio as gr

    rows = read_jsonl(manifest_path)
    if not rows:
        raise BenchmarkError(f"empty manifest: {manifest_path}")
    by_sample = {row["sample_id"]: row for row in rows}
    latest = load_latest_human_judgments(judgment_path)

    def sample_choices() -> list[str]:
        return [row["sample_id"] for row in rows]

    def load_sample(sample_id: str) -> tuple[Any, ...]:
        row = by_sample[sample_id]
        artifacts = find_sample_artifacts(run_root, row)
        vlm_json: dict[str, Any] = {}
        if artifacts.get("vlm_judge"):
            try:
                vlm_json = load_vlm_judge(Path(artifacts["vlm_judge"]))
            except Exception as exc:  # pragma: no cover - UI display path
                vlm_json = {"error": str(exc)}
        meta_text = ""
        if artifacts.get("metadata"):
            metadata_obj = json.loads(Path(artifacts["metadata"]).read_text(encoding="utf-8"))
            meta_text = json.dumps(metadata_obj, ensure_ascii=False, indent=2)[:6000]
            runtime_inputs = metadata_obj.get("vace_runtime_inputs") or {}
            vace_prompt = metadata_obj.get("vace_prompt") or runtime_inputs.get("vace_prompt") or ""
        else:
            vace_prompt = ""
        saved = latest.get(sample_id, {})
        return (
            row.get("converted_video") or row.get("source_full_video"),
            artifacts.get("edited_video"),
            row["user_prompt"],
            vace_prompt,
            row["expected_visible_outcome"],
            row["expected_physical_effect"],
            "\n".join(row["must_preserve"]),
            vlm_json,
            vlm_json.get("short_reason", ""),
            meta_text,
            int(saved.get("human_e2w_target_success", 0)),
            int(saved.get("human_e2w_physical_effect_success", 0)),
            int(saved.get("human_e2w_preservation_success", 0)),
            int(saved.get("human_e2w_temporal_consistency", 0)),
            int(saved.get("human_e2w_major_artifacts", 0)),
            int(saved.get("human_e2w_overall_pass", 0)),
            str(saved.get("human_e2w_notes", "")),
            int(saved.get("human_vlm_correct_about_target", 0)),
            int(saved.get("human_vlm_correct_about_physics", 0)),
            int(saved.get("human_vlm_correct_about_preservation", 0)),
            int(saved.get("human_vlm_overall_agree", 0)),
            int(saved.get("human_vlm_missed_failure", 0)),
            str(saved.get("human_vlm_notes", "")),
        )

    def save_sample(
        sample_id: str,
        e2w_target: int,
        e2w_physics: int,
        e2w_preserve: int,
        e2w_temporal: int,
        e2w_artifacts: int,
        e2w_overall: int,
        e2w_notes: str,
        vlm_target: int,
        vlm_physics: int,
        vlm_preserve: int,
        vlm_agree: int,
        vlm_missed: int,
        vlm_notes: str,
    ) -> str:
        record = {
            "sample_id": sample_id,
            "physics_iq_id": by_sample[sample_id]["physics_iq_id"],
            "operation": by_sample[sample_id]["operation"],
            "human_e2w_target_success": int(e2w_target),
            "human_e2w_physical_effect_success": int(e2w_physics),
            "human_e2w_preservation_success": int(e2w_preserve),
            "human_e2w_temporal_consistency": int(e2w_temporal),
            "human_e2w_major_artifacts": int(e2w_artifacts),
            "human_e2w_overall_pass": int(e2w_overall),
            "human_e2w_notes": e2w_notes,
            "human_vlm_correct_about_target": int(vlm_target),
            "human_vlm_correct_about_physics": int(vlm_physics),
            "human_vlm_correct_about_preservation": int(vlm_preserve),
            "human_vlm_overall_agree": int(vlm_agree),
            "human_vlm_missed_failure": int(vlm_missed),
            "human_vlm_notes": vlm_notes,
        }
        append_human_judgment(judgment_path, record)
        latest[sample_id] = record
        export_human_summary(judgment_path, run_root / "human_judgments_summary.json", run_root / "human_judgments_summary.csv")
        return f"Saved {sample_id} to {judgment_path}"

    with gr.Blocks(title="E2W Physics-IQ Human/VLM Judge") as app:
        gr.Markdown("# E2W Physics-IQ Human/VLM Judge\nHuman scores are VISUAL evidence; VLM agreement is evaluated separately.")
        sample = gr.Dropdown(choices=sample_choices(), value=sample_choices()[0], label="Sample")
        with gr.Row():
            original_video = gr.Video(label="Original full-video source")
            edited_video = gr.Video(label="Edited video")
        with gr.Row():
            with gr.Column():
                user_prompt = gr.Textbox(label="E2W prompt", lines=2)
                vace_prompt = gr.Textbox(label="VACE prompt", lines=6)
                expected_visible = gr.Textbox(label="Expected visible outcome", lines=2)
                expected_physics = gr.Textbox(label="Expected physical effect", lines=2)
                must_preserve = gr.Textbox(label="Must preserve", lines=4)
            with gr.Column():
                vlm_json = gr.JSON(label="VLM judge JSON")
                vlm_reason = gr.Textbox(label="VLM short reason", lines=2)
                metadata = gr.Textbox(label="Pipeline metadata excerpt", lines=12)
        gr.Markdown("## Human score: E2W output")
        with gr.Row():
            e2w_target = gr.Radio([0, 1], value=0, label="target_success")
            e2w_physics = gr.Radio([0, 1], value=0, label="physical_effect_success")
            e2w_preserve = gr.Radio([0, 1], value=0, label="preservation_success")
            e2w_temporal = gr.Radio([0, 1], value=0, label="temporal_consistency")
            e2w_artifacts = gr.Radio([0, 1], value=0, label="major_artifacts")
            e2w_overall = gr.Radio([0, 1], value=0, label="overall_pass")
        e2w_notes = gr.Textbox(label="E2W notes", lines=3)
        gr.Markdown("## Human score: VLM judge")
        with gr.Row():
            vlm_target = gr.Radio([0, 1], value=0, label="correct_about_target")
            vlm_physics = gr.Radio([0, 1], value=0, label="correct_about_physics")
            vlm_preserve = gr.Radio([0, 1], value=0, label="correct_about_preservation")
            vlm_agree = gr.Radio([0, 1], value=0, label="overall_agree")
            vlm_missed = gr.Radio([0, 1], value=0, label="missed_failure")
        vlm_notes = gr.Textbox(label="VLM notes", lines=3)
        save = gr.Button("Save judgment")
        status = gr.Textbox(label="Save status")

        load_outputs = [
            original_video,
            edited_video,
            user_prompt,
            vace_prompt,
            expected_visible,
            expected_physics,
            must_preserve,
            vlm_json,
            vlm_reason,
            metadata,
            e2w_target,
            e2w_physics,
            e2w_preserve,
            e2w_temporal,
            e2w_artifacts,
            e2w_overall,
            e2w_notes,
            vlm_target,
            vlm_physics,
            vlm_preserve,
            vlm_agree,
            vlm_missed,
            vlm_notes,
        ]
        sample.change(load_sample, inputs=[sample], outputs=load_outputs)
        app.load(load_sample, inputs=[sample], outputs=load_outputs)
        save.click(
            save_sample,
            inputs=[
                sample,
                e2w_target,
                e2w_physics,
                e2w_preserve,
                e2w_temporal,
                e2w_artifacts,
                e2w_overall,
                e2w_notes,
                vlm_target,
                vlm_physics,
                vlm_preserve,
                vlm_agree,
                vlm_missed,
                vlm_notes,
            ],
            outputs=[status],
        )
    return app


def cmd_build_manifest(args: argparse.Namespace) -> None:
    rows = build_manifest(
        descriptions_csv=args.descriptions_csv,
        physics_iq_root=args.physics_iq_root,
        out_root=args.out_root,
        allow_missing_source=args.allow_missing_source,
    )
    errors = validate_manifest_rows(rows)
    if errors:
        raise BenchmarkError("\n".join(errors))
    write_jsonl(args.manifest, rows)
    write_jsonl(args.vlm_prompts, [{"sample_id": row["sample_id"], "prompt": row["vlm_judge_prompt"]} for row in rows])
    write_json(
        args.summary,
        {
            "created_at": utc_now(),
            "evidence_level": "VISUAL_CANDIDATE_ONLY",
            "manifest": str(args.manifest),
            "vlm_prompts": str(args.vlm_prompts),
            "row_count": len(rows),
            "allowed_source_roots": [
                "physics-IQ-benchmark/full-videos",
                "physics-IQ-benchmark/split-videos/testing",
            ],
            "review_proxy_forbidden": True,
        },
    )
    if args.convert:
        materialize_converted_videos(rows, fps=args.fps, width=args.width, height=args.height, frame_num=args.frame_num)
    print(f"wrote {args.manifest}")


def cmd_validate_manifest(args: argparse.Namespace) -> None:
    errors = validate_manifest_rows(read_jsonl(args.manifest))
    if errors:
        raise BenchmarkError("\n".join(errors))
    print(f"manifest ok: {args.manifest}")


def cmd_validate_run(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.manifest)
    errors = validate_manifest_rows(rows) + validate_run_contract(args.run_root, rows)
    if errors:
        raise BenchmarkError("\n".join(errors))
    print(f"run contract ok: {args.run_root}")


def cmd_run_pipeline(args: argparse.Namespace) -> None:
    if not args.control_branch_checkpoint.exists():
        raise BenchmarkError(f"control branch checkpoint missing: {args.control_branch_checkpoint}")
    rows = read_jsonl(args.manifest)
    errors = validate_manifest_rows(rows)
    if errors:
        raise BenchmarkError("\n".join(errors))
    if not args.all and not args.sample_id:
        raise BenchmarkError("run-pipeline requires --all or at least one --sample-id")
    selected = rows if args.all else [row for row in rows if row["sample_id"] in set(args.sample_id)]
    missing = sorted(set(args.sample_id) - {row["sample_id"] for row in selected})
    if missing:
        raise BenchmarkError(f"unknown sample_id(s): {missing}")
    args.run_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] {row['sample_id']} {row['operation']}", flush=True)
        summary_rows.append(run_pipeline_row(args, row))
    write_json(
        args.run_root / "pipeline_summary.json",
        {
            "created_at": utc_now(),
            "manifest": str(args.manifest),
            "run_root": str(args.run_root),
            "skip_vlm_judge": bool(args.skip_vlm_judge),
            "control_branch_checkpoint": str(args.control_branch_checkpoint),
            "rows": summary_rows,
        },
    )
    print(f"wrote {args.run_root / 'pipeline_summary.json'}")


def cmd_download_full_videos(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.manifest)
    if not rows:
        rows = build_manifest(
            descriptions_csv=args.descriptions_csv,
            physics_iq_root=args.physics_iq_root,
            out_root=args.out_root,
            allow_missing_source=True,
        )
    download_full_videos(rows, args.physics_iq_root, descriptions_csv=args.descriptions_csv, fps=args.fps)


def cmd_launch_dashboard(args: argparse.Namespace) -> None:
    ensure_media_probe_path()
    app = make_dashboard(args.manifest, args.run_root, args.judgments)
    allowed_paths = [
        str(args.manifest.parent),
        str(args.run_root),
        str(args.judgments.parent),
        str(DEFAULT_PHYSICS_IQ_ROOT),
    ]
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=False,
        allowed_paths=allowed_paths,
        show_error=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build-manifest")
    build.add_argument("--descriptions-csv", type=Path, default=DEFAULT_DESCRIPTIONS_CSV)
    build.add_argument("--physics-iq-root", type=Path, default=DEFAULT_PHYSICS_IQ_ROOT)
    build.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    build.add_argument("--manifest", type=Path, default=DEFAULT_OUT_ROOT / "manifest.jsonl")
    build.add_argument("--vlm-prompts", type=Path, default=DEFAULT_OUT_ROOT / "vlm_judge_prompts.jsonl")
    build.add_argument("--summary", type=Path, default=DEFAULT_OUT_ROOT / "summary.json")
    build.add_argument("--allow-missing-source", action="store_true")
    build.add_argument("--convert", action="store_true")
    build.add_argument("--fps", type=int, default=12)
    build.add_argument("--width", type=int, default=832)
    build.add_argument("--height", type=int, default=480)
    build.add_argument("--frame-num", type=int, default=21)
    build.set_defaults(func=cmd_build_manifest)

    validate = sub.add_parser("validate-manifest")
    validate.add_argument("--manifest", type=Path, default=DEFAULT_OUT_ROOT / "manifest.jsonl")
    validate.set_defaults(func=cmd_validate_manifest)

    validate_run = sub.add_parser("validate-run")
    validate_run.add_argument("--manifest", type=Path, default=DEFAULT_OUT_ROOT / "manifest.jsonl")
    validate_run.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    validate_run.set_defaults(func=cmd_validate_run)

    run_pipeline = sub.add_parser("run-pipeline")
    run_pipeline.add_argument("--manifest", type=Path, default=DEFAULT_OUT_ROOT / "manifest.jsonl")
    run_pipeline.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    run_pipeline.add_argument("--sample-id", action="append", default=[])
    run_pipeline.add_argument("--all", action="store_true")
    run_pipeline.add_argument("--skip-existing", action="store_true")
    run_pipeline.add_argument("--control-branch-checkpoint", type=Path, required=True)
    run_pipeline.add_argument("--add-planner-adapter", type=Path, default=None, help="add-planner LoRA adapter; required when the manifest has add rows (no archived default)")
    run_pipeline.add_argument("--vace-sample-steps", type=int, default=8)
    run_pipeline.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    run_pipeline.add_argument("--skip-vlm-judge", action="store_true")
    run_pipeline.set_defaults(func=cmd_run_pipeline)

    download = sub.add_parser("download-full-videos")
    download.add_argument("--manifest", type=Path, default=DEFAULT_OUT_ROOT / "manifest.jsonl")
    download.add_argument("--descriptions-csv", type=Path, default=DEFAULT_DESCRIPTIONS_CSV)
    download.add_argument("--physics-iq-root", type=Path, default=DEFAULT_PHYSICS_IQ_ROOT)
    download.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    download.add_argument("--fps", type=int, default=30)
    download.set_defaults(func=cmd_download_full_videos)

    dash = sub.add_parser("launch-dashboard")
    dash.add_argument("--manifest", type=Path, default=DEFAULT_OUT_ROOT / "manifest.jsonl")
    dash.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    dash.add_argument("--judgments", type=Path, default=DEFAULT_RUN_ROOT / "human_judgments.jsonl")
    dash.add_argument("--server-name", default="127.0.0.1")
    dash.add_argument("--server-port", type=int, default=7860)
    dash.set_defaults(func=cmd_launch_dashboard)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

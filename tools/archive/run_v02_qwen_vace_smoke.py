#!/usr/bin/env python3
"""Run a fresh E2W VLM-planner + Qwen + VACE smoke pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import ensure_run_dirs, load_manifest, write_json  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
RUNS_ROOT = REPO_ROOT / "runs"
DEFAULT_INPUT = REPO_ROOT / "data/physics_iq_vlm_sft/vlm_planner_sft_eval_v6_teacher_grounded.jsonl"
DEFAULT_MODE = "mode_v0_2_qwen_vace_smoke"
DEFAULT_SAMPLES = ["0052", "0056", "0070", "0076", "0112", "0077", "0341", "0128"]
DEFAULT_PLANNER_ADAPTER = Path("/data/cwx/E2W/checkpoints/vlm_planner_lora_physics_iq_v5_split_eval")
DEFAULT_BASE_MODEL = Path("/data/cwx/Edit2World-unified/checkpoints/Qwen2.5-VL-7B-Instruct")
DEFAULT_QWEN_IMAGE_EDIT = Path("/data/cwx/Edit2World-unified/checkpoints/Qwen-Image-Edit")
DEFAULT_VACE_REPO = Path("/data/cwx/Edit2World-unified/external/VACE")
DEFAULT_VACE_CKPT = Path("/data/cwx/Edit2World-unified/checkpoints/Wan2.1-VACE-14B")
DEFAULT_VACE_PYTHON = Path("/data/cwx/conda/envs/edit2world-phase1-real/bin/python")

STAGE_ENTRY_NAMES = {
    "planner": "planner_eval",
    "mask": "mask_builder",
    "first_frame": "first_frame",
    "vace": "vace_v0",
}
CRITICAL_KEYS = {
    "planner": {"raw_output", "raw_pred", "edit_plan", "quadmask_spec", "vace_prompt", "planner_eval", "text_query", "first_frame"},
    "mask": {"source_clip", "primary_mask", "affected_mask", "editable_mask", "quadmask_npy", "quadmask", "mask_eval"},
    "first_frame": {"primary_mask_frame0", "first_frame_prompt", "edited_first_frame", "first_frame_metadata"},
    "vace": {"vace_prompt", "vace_conditioning", "vace_generation_mask", "vace_command", "vace_metadata"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--planner-adapter", type=Path, default=DEFAULT_PLANNER_ADAPTER)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--operation", choices=["auto", "remove", "add"], default="auto")
    parser.add_argument("--planner-max-new-tokens", type=int, default=1536)
    parser.add_argument("--planner-video-fps", type=float, default=1.0)
    parser.add_argument("--planner-min-pixels", type=int, default=50176)
    parser.add_argument("--planner-max-pixels", type=int, default=100352)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--max-frames", type=int, default=81)
    parser.add_argument("--fallback-rect", action="store_true")
    parser.add_argument("--first-frame-backend", choices=["qwen_image_edit", "opencv_inpaint_debug"], default="qwen_image_edit")
    parser.add_argument("--qwen-checkpoint", type=Path, default=DEFAULT_QWEN_IMAGE_EDIT)
    parser.add_argument("--qwen-steps", type=int, default=12)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--size", default="480p")
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--vace-python", type=Path, default=DEFAULT_VACE_PYTHON)
    parser.add_argument("--vace-repo", type=Path, default=DEFAULT_VACE_REPO)
    parser.add_argument("--vace-ckpt", type=Path, default=DEFAULT_VACE_CKPT)
    parser.add_argument(
        "--cuda-visible-devices",
        help="Set CUDA_VISIBLE_DEVICES for every pipeline subprocess, e.g. a free physical GPU index like 4.",
    )
    parser.add_argument("--run-vace", action="store_true")
    parser.add_argument("--no-package", action="store_true")
    parser.add_argument("--no-force", action="store_true")
    return parser.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return args.run_dir.resolve()
    run_name = args.run_name
    if not run_name:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"e2w_v0_2_qwen_vace_smoke_{stamp}"
    return RUNS_ROOT / run_name


def sample_args(sample_ids: list[str]) -> list[str]:
    out: list[str] = []
    for sample_id in sample_ids:
        out.extend(["--sample-id", sample_id])
    return out


def build_stage_env(cuda_visible_devices: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def stage_env_metadata(env: dict[str, str]) -> dict[str, str]:
    keys = ["CUDA_VISIBLE_DEVICES"]
    return {key: env[key] for key in keys if key in env}


def run_stage(
    stage: str,
    command: list[str],
    run_dir: Path,
    records: dict[str, dict[str, Any]],
    env: dict[str, str],
) -> None:
    log_dir = run_dir / "stage_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started_ts = time.time()
    record: dict[str, Any] = {
        "stage": stage,
        "source": Path(command[1]).name if len(command) > 1 else "",
        "command": command,
        "env": stage_env_metadata(env),
        "started_at": utc_now(),
        "started_ts": started_ts,
        "stdout": str(log_dir / f"{stage}.stdout.txt"),
        "stderr": str(log_dir / f"{stage}.stderr.txt"),
    }
    print(f"[v0.2] {stage}: {' '.join(command)}", flush=True)
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (log_dir / f"{stage}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (log_dir / f"{stage}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    record.update(
        {
            "ended_at": utc_now(),
            "returncode": proc.returncode,
            "runtime_sec": time.time() - started_ts,
        }
    )
    records[stage] = record
    if proc.returncode != 0:
        raise RuntimeError(f"{stage} failed with returncode {proc.returncode}; see {record['stderr']}")


def direct_symlink_target(path: Path) -> str | None:
    if not path.is_symlink():
        return None
    target = Path(os.readlink(path))
    if not target.is_absolute():
        target = path.parent / target
    return str(target)


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def artifact_record(path_text: str, run_dir: Path, started_ts: float, critical: bool) -> dict[str, Any]:
    path = Path(path_text)
    is_symlink = path.is_symlink()
    direct_target = direct_symlink_target(path)
    resolved = path.resolve(strict=False)
    exists = path.exists()
    mtime = path.stat().st_mtime if exists else None
    run_root = run_dir.resolve()
    runs_root = RUNS_ROOT.resolve()
    direct_inside_run = bool(direct_target and is_under(Path(direct_target), run_root))
    resolved_inside_run = is_under(resolved, run_root)
    points_to_other_run = is_symlink and is_under(resolved, runs_root) and not resolved_inside_run
    regenerated = bool(
        critical
        and exists
        and not is_symlink
        and resolved_inside_run
        and mtime is not None
        and mtime >= started_ts - 2.0
    )
    return {
        "path": str(path),
        "exists": exists,
        "is_symlink": is_symlink,
        "direct_symlink_target": direct_target,
        "resolved_path": str(resolved),
        "direct_target_inside_run": direct_inside_run,
        "resolved_inside_run": resolved_inside_run,
        "points_to_other_run": points_to_other_run,
        "mtime": mtime,
        "critical": critical,
        "regenerated": regenerated,
    }


def inspect_stage(
    run_dir: Path,
    mode: str,
    sample_ids: list[str],
    stage: str,
    record: dict[str, Any] | None,
    include_edited_video: bool,
) -> dict[str, Any]:
    manifest = load_manifest(run_dir / "manifest.jsonl")
    entry_stage = STAGE_ENTRY_NAMES[stage]
    critical_keys = set(CRITICAL_KEYS[stage])
    if stage == "vace" and include_edited_video:
        critical_keys.add("edited_video")
    started_ts = float((record or {}).get("started_ts") or 0.0)
    artifacts: list[dict[str, Any]] = []
    for entry in manifest:
        if entry.get("stage") != entry_stage or entry.get("mode") != mode or entry.get("sample_id") not in sample_ids:
            continue
        for key, path_text in sorted((entry.get("paths") or {}).items()):
            if not path_text:
                continue
            artifacts.append(
                {
                    "sample_id": entry.get("sample_id"),
                    "key": key,
                    **artifact_record(str(path_text), run_dir, started_ts, key in critical_keys),
                }
            )
    critical = [a for a in artifacts if a["critical"]]
    return {
        "source": (record or {}).get("source"),
        "started_at": (record or {}).get("started_at"),
        "ended_at": (record or {}).get("ended_at"),
        "returncode": (record or {}).get("returncode"),
        "regenerated": bool(critical) and all(bool(a["regenerated"]) for a in critical),
        "critical_artifacts": len(critical),
        "critical_artifacts_regenerated": sum(bool(a["regenerated"]) for a in critical),
        "symlink_artifacts": sum(bool(a["is_symlink"]) for a in artifacts),
        "other_run_symlink_artifacts": [a for a in artifacts if a["points_to_other_run"]],
        "artifacts": artifacts,
    }


def inspect_flat_package(run_dir: Path, sample_ids: list[str]) -> dict[str, Any]:
    symlinks: list[dict[str, Any]] = []
    run_root = run_dir.resolve()
    runs_root = RUNS_ROOT.resolve()
    for sample_id in sample_ids:
        flat_dir = run_dir / sample_id
        if not flat_dir.exists():
            continue
        for path in sorted(flat_dir.iterdir()):
            if not path.is_symlink():
                continue
            direct_target = direct_symlink_target(path)
            resolved = path.resolve(strict=False)
            resolved_inside_run = is_under(resolved, run_root)
            points_to_other_run = is_under(resolved, runs_root) and not resolved_inside_run
            symlinks.append(
                {
                    "sample_id": sample_id,
                    "path": str(path),
                    "direct_symlink_target": direct_target,
                    "resolved_path": str(resolved),
                    "direct_target_inside_run": bool(
                        direct_target and is_under(Path(direct_target), run_root)
                    ),
                    "resolved_inside_run": resolved_inside_run,
                    "points_to_other_run": points_to_other_run,
                }
            )
    return {
        "symlink_count": len(symlinks),
        "direct_internal_symlink_count": sum(bool(item["direct_target_inside_run"]) for item in symlinks),
        "other_run_symlink_count": sum(bool(item["points_to_other_run"]) for item in symlinks),
        "other_run_symlinks": [item for item in symlinks if item["points_to_other_run"]],
        "symlinks": symlinks,
    }


def build_freshness(
    run_dir: Path,
    mode: str,
    sample_ids: list[str],
    records: dict[str, dict[str, Any]],
    run_vace: bool,
    packaged: bool,
) -> dict[str, Any]:
    stages = {
        stage: inspect_stage(run_dir, mode, sample_ids, stage, records.get(stage), include_edited_video=run_vace)
        for stage in ["planner", "mask", "first_frame", "vace"]
    }
    flat_package = inspect_flat_package(run_dir, sample_ids) if packaged else {"skipped": True}
    return {
        "schema_version": "e2w.artifact_freshness.v1",
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "mode": mode,
        "sample_ids": sample_ids,
        "stage_commands": records,
        "stages": stages,
        "flat_package": flat_package,
        "all_critical_stages_regenerated": all(bool(stage["regenerated"]) for stage in stages.values()),
        "no_other_run_symlinks": not any(stage["other_run_symlink_artifacts"] for stage in stages.values())
        and not flat_package.get("other_run_symlink_count", 0),
    }


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args)
    sample_ids = args.sample_id or DEFAULT_SAMPLES
    force_args = [] if args.no_force else ["--force"]
    stage_env = build_stage_env(args.cuda_visible_devices)
    if args.cuda_visible_devices is not None:
        print(f"[v0.2] CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}", flush=True)
    ensure_run_dirs(run_dir)
    records: dict[str, dict[str, Any]] = {}
    packaged = False
    py = sys.executable
    try:
        run_stage(
            "planner",
            [
                py,
                str(TOOLS_DIR / "eval_vlm_planner.py"),
                "--split-jsonl",
                str(args.input_jsonl),
                "--run-dir",
                str(run_dir),
                "--mode",
                args.mode,
                "--adapter",
                str(args.planner_adapter),
                "--base-model",
                str(args.base_model),
                "--operation",
                args.operation,
                "--max-new-tokens",
                str(args.planner_max_new_tokens),
                "--video-fps",
                str(args.planner_video_fps),
                "--min-pixels",
                str(args.planner_min_pixels),
                "--max-pixels",
                str(args.planner_max_pixels),
                *sample_args(sample_ids),
                *force_args,
            ],
            run_dir,
            records,
            stage_env,
        )
        mask_command = [
            py,
            str(TOOLS_DIR / "build_quadmask_from_spec.py"),
            "--run-dir",
            str(run_dir),
            "--mode",
            args.mode,
            "--max-frames",
            str(args.max_frames),
            *sample_args(sample_ids),
            *force_args,
        ]
        if args.fallback_rect:
            mask_command.append("--fallback-rect")
        run_stage("mask", mask_command, run_dir, records, stage_env)
        run_stage(
            "first_frame",
            [
                py,
                str(TOOLS_DIR / "run_first_frame_edit.py"),
                "--run-dir",
                str(run_dir),
                "--mode",
                args.mode,
                "--backend",
                args.first_frame_backend,
                "--qwen-checkpoint",
                str(args.qwen_checkpoint),
                "--seed",
                str(args.seed),
                "--steps",
                str(args.qwen_steps),
                "--true-cfg-scale",
                str(args.true_cfg_scale),
                *sample_args(sample_ids),
                *force_args,
            ],
            run_dir,
            records,
            stage_env,
        )
        vace_command = [
            py,
            str(TOOLS_DIR / "run_vace_v0.py"),
            "--run-dir",
            str(run_dir),
            "--mode",
            args.mode,
            "--frame-num",
            str(args.frame_num),
            "--fps",
            str(args.fps),
            "--size",
            args.size,
            "--sample-steps",
            str(args.sample_steps),
            "--seed",
            str(args.seed),
            "--python",
            str(args.vace_python),
            "--vace-repo",
            str(args.vace_repo),
            "--vace-ckpt",
            str(args.vace_ckpt),
            *sample_args(sample_ids),
            *force_args,
        ]
        if args.run_vace:
            vace_command.append("--run-vace")
        run_stage("vace", vace_command, run_dir, records, stage_env)
        if not args.no_package:
            run_stage(
                "package",
                [
                    py,
                    str(TOOLS_DIR / "package_v02_qwen_vace_smoke.py"),
                    "--run-dir",
                    str(run_dir),
                    "--mode",
                    args.mode,
                ],
                run_dir,
                records,
                stage_env,
            )
            packaged = True
    finally:
        freshness = build_freshness(run_dir, args.mode, sample_ids, records, args.run_vace, packaged)
        write_json(run_dir / "artifact_freshness.json", freshness)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "artifact_freshness": str(run_dir / "artifact_freshness.json"),
                "cuda_visible_devices": stage_env.get("CUDA_VISIBLE_DEVICES"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

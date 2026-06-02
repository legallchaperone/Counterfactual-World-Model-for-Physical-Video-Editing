#!/usr/bin/env python3
"""Prepare Physics-IQ videos and teacher labels for VLM planner SFT.

This pipeline treats Physics-IQ as weak planner supervision only. The source
videos are real continuations, not object-removal counterfactual targets.
"""

from __future__ import annotations

import argparse
import base64
import csv
import http.client
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DIR = Path(
    "/data/cwx/physics-iq/physics-IQ-benchmark/split-videos/testing/16FPS"
)
DEFAULT_DESCRIPTIONS_CSV = Path("/data/cwx/physics-iq/descriptions.csv")
DEFAULT_WORK_DIR = Path("/data/cwx/E2W/data/physics_iq_vlm_sft")
DEFAULT_FULL_ANNOTATIONS_CSV = DEFAULT_WORK_DIR / "human_remove_annotation_template.csv"
DEFAULT_MODEL = "google/gemini-2.5-pro"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


VIDEO_RE = re.compile(
    r"^(?P<id>\d{4})_testing-videos_(?P<fps>\d+FPS)_"
    r"(?P<perspective>perspective-[^_]+)_(?P<take>take-\d+)_"
    r"trimmed-(?P<scenario_slug>.+)\.mp4$"
)


SYSTEM_PROMPT = """You produce JSON labels for Edit2World VLM planner SFT.
The video is a real Physics-IQ full/testing video, not a counterfactual target.
Use the human annotation as the source of truth for which object(s) should be removed.
Return only valid JSON. Do not include markdown."""

PLANNER_VACE_CONTRACT_RULES = """Planner/VACE prompt contract:
- In counterfactual_expectation.if_removed, do not name the removed target object or any visible target subpart/material, even in negative wording such as "no <target>" or "without <target>".
- Describe only the non-target objects, local background, revealed surface, and resulting physical motion.
- When the visible count of a non-target object is clear and relevant, include it explicitly in protected_objects and counterfactual text, for example "one potato" or "two dominoes". Do not invent counts when uncertain."""


USER_PROMPT_TEMPLATE = """Analyze the attached Physics-IQ video for a remove-object Edit2World planner example.

Dataset caveat:
- This source video is the original real-world event.
- It is NOT the object-removal counterfactual target video.
- Your output will supervise only the VLM planner/edit-plan stage.

{planner_contract_rules}

Video metadata:
- video_id: {video_id}
- category: {category}
- perspective: {perspective}
- take: {take}
- scenario_slug: {scenario_slug}
- description: {description}

Human annotation:
- remove_objects: {remove_objects}
- protected_objects: {protected_objects}
- notes: {human_notes}

Return JSON with this exact top-level structure:
{{
  "video_id": "...",
  "task_type": "remove",
  "edit_prompt": "remove ...",
  "target_objects": [
    {{
      "name": "...",
      "aliases": ["..."],
      "role": "causal_initiator|affected_object|distractor|unknown",
      "visibility": "clear|partial|brief|unclear",
      "temporal_span": {{"start_sec": 0.0, "end_sec": 8.0}},
      "location_description": "..."
    }}
  ],
  "protected_objects": ["..."],
  "event_summary": "...",
  "physical_causal_chain": [
    {{
      "step": 1,
      "description": "...",
      "objects": ["..."],
      "approx_time_sec": 0.0
    }}
  ],
  "counterfactual_expectation": {{
    "if_removed": "...",
    "affected_regions": ["..."],
    "unchanged_regions": ["..."],
    "uncertainties": ["..."]
  }},
  "quadmask_spec": {{
    "primary": {{
      "objects": ["..."],
      "description": "pixels of removed object(s) across all frames"
    }},
    "affected": {{
      "objects": ["..."],
      "description": "regions whose motion/state may change if target is removed"
    }},
    "keep": {{
      "description": "background and unrelated objects that should stay unchanged"
    }}
  }},
  "quality_flags": {{
    "usable_for_planner_sft": true,
    "needs_human_review": false,
    "reasons": []
  }}
}}"""


def read_descriptions(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["scenario"]] = row
    return rows


def parse_video(path: Path) -> dict[str, str]:
    match = VIDEO_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected Physics-IQ filename: {path.name}")
    data = match.groupdict()
    data["video_id"] = data.pop("id")
    data["scenario"] = (
        f"{data['video_id']}_{data['perspective']}_{data['take']}_"
        f"trimmed-{data['scenario_slug']}.mp4"
    )
    data["scenario_group"] = data["scenario_slug"]
    return data


def prepare(args: argparse.Namespace) -> None:
    source_dir = args.source_dir
    work_dir = args.work_dir
    links_dir = work_dir / "full_videos"
    work_dir.mkdir(parents=True, exist_ok=True)
    links_dir.mkdir(parents=True, exist_ok=True)

    descriptions = read_descriptions(args.descriptions_csv)
    rows: list[dict[str, Any]] = []
    missing_description = 0

    for source_path in sorted(source_dir.glob("*.mp4")):
        meta = parse_video(source_path)
        desc = descriptions.get(meta["scenario"])
        if desc is None:
            missing_description += 1
            desc = {
                "description": "",
                "category": "",
                "generated_video_name": "",
            }

        link_path = links_dir / source_path.name
        if link_path.exists() or link_path.is_symlink():
            if link_path.resolve() != source_path.resolve():
                raise RuntimeError(f"Refusing to overwrite existing link: {link_path}")
        else:
            link_path.symlink_to(source_path)

        rows.append(
            {
                "video_id": meta["video_id"],
                "scenario": meta["scenario"],
                "scenario_group": meta["scenario_group"],
                "category": desc["category"],
                "perspective": meta["perspective"].removeprefix("perspective-"),
                "take": meta["take"].removeprefix("take-"),
                "fps": meta["fps"],
                "description": desc["description"],
                "source_video_path": str(source_path),
                "linked_video_path": str(link_path),
                "generated_video_name": desc["generated_video_name"],
                "include_for_labeling": "",
                "remove_objects": "",
                "protected_objects": "",
                "human_notes": "",
                "label_status": "unlabeled",
            }
        )

    manifest = {
        "dataset": "physics-iq",
        "purpose": "VLM planner weak-supervision generation only",
        "source_dir": str(source_dir),
        "linked_video_dir": str(links_dir),
        "descriptions_csv": str(args.descriptions_csv),
        "available_full_videos": len(rows),
        "description_rows": len(descriptions),
        "missing_description_rows": missing_description,
        "source_is_counterfactual_target": False,
        "caveat": (
            "Physics-IQ testing videos are real full/testing continuations; "
            "do not use them as object-removal target videos for VACE training."
        ),
    }
    (work_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    template_csv = work_dir / "human_remove_annotation_template.csv"
    with template_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {template_csv}")


def load_annotation_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_full_metadata(work_dir: Path) -> dict[str, dict[str, str]]:
    path = work_dir / "human_remove_annotation_template.csv"
    if not path.exists():
        return {}
    return {row["video_id"]: row for row in load_annotation_rows(path)}


def enrich_rows(rows: list[dict[str, str]], work_dir: Path) -> list[dict[str, str]]:
    full_rows = load_annotation_rows(work_dir / "human_remove_annotation_template.csv")
    full_metadata = {row["video_id"]: row for row in full_rows}
    full_by_seq = {str(i): row for i, row in enumerate(full_rows, start=1)}
    enriched: list[dict[str, str]] = []
    for row in rows:
        base = dict(full_metadata.get(row.get("video_id", ""), {}))
        if not base and row.get("seq"):
            base = dict(full_by_seq.get(str(int(row["seq"])), {}))
        base.update(row)
        enriched.append(base)
    return enriched


def should_label(row: dict[str, str]) -> bool:
    include = row.get("include_for_labeling", "").strip().lower()
    remove_objects = row.get("remove_objects", "").strip()
    status = row.get("label_status", "").strip().lower()
    excluded = include in {"0", "no", "n", "false", "skip", "exclude", "excluded"}
    return not excluded and bool(remove_objects) and status != "done"


def encode_video_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{data}"


def extract_json_text(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def call_openrouter(args: argparse.Namespace, row: dict[str, str]) -> dict[str, Any]:
    if args.api_video_source == "review":
        seq = int(row["seq"]) if row.get("seq") else int(row["video_id"])
        video_path = args.work_dir / "review_videos_by_seq" / f"{seq:04d}.mp4"
    else:
        video_path = Path(row["linked_video_path"])
    size = video_path.stat().st_size
    if args.max_video_mb and size > args.max_video_mb * 1024 * 1024:
        raise RuntimeError(
            f"Video exceeds --max-video-mb={args.max_video_mb}: {video_path} ({size} bytes)"
        )

    prompt = USER_PROMPT_TEMPLATE.format(
        video_id=row["video_id"],
        category=row["category"],
        perspective=row["perspective"],
        take=row["take"],
        scenario_slug=row["scenario_group"],
        description=row["description"],
        remove_objects=row["remove_objects"],
        protected_objects=row.get("protected_objects", ""),
        human_notes=row.get("human_notes", ""),
        planner_contract_rules=PLANNER_VACE_CONTRACT_RULES,
    )
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": encode_video_data_url(video_path)}},
                ],
            },
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
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
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
            print(f"  transient error on attempt {attempt}: {exc}; retrying in {sleep_for:.1f}s")
            time.sleep(sleep_for)
    raise RuntimeError(f"OpenRouter request failed after retries: {last_exc!r}")


def label_openrouter(args: argparse.Namespace) -> None:
    if not args.api_key and not args.dry_run:
        raise SystemExit("OPENROUTER_API_KEY is required unless --dry-run is used")

    rows = enrich_rows(load_annotation_rows(args.annotations_csv), args.work_dir)
    output_dir = args.work_dir / "teacher_labels"
    raw_dir = output_dir / "raw"
    parsed_dir = output_dir / "parsed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    selected = [r for r in rows if should_label(r)]
    if args.limit is not None:
        selected = selected[: args.limit]
    print(f"selected {len(selected)} rows for labeling")

    for i, row in enumerate(selected, start=1):
        video_id = row["video_id"]
        raw_path = raw_dir / f"{video_id}.json"
        parsed_path = parsed_dir / f"{video_id}.json"
        if args.skip_existing and parsed_path.exists():
            print(f"[{i}/{len(selected)}] skip existing {video_id}")
            continue

        if args.dry_run:
            print(f"[dry-run] would label {video_id}: {row['remove_objects']}")
            continue

        print(f"[{i}/{len(selected)}] labeling {video_id}")
        try:
            raw = call_openrouter(args, row)
            raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            content = raw["choices"][0]["message"]["content"]
            parsed = extract_json_text(content)
            parsed_path.write_text(
                json.dumps(parsed, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            stale_error = raw_dir / f"{video_id}.error.json"
            if stale_error.exists():
                stale_error.unlink()
            print(f"  wrote {parsed_path}")
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            http.client.IncompleteRead,
            RuntimeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            error_path = raw_dir / f"{video_id}.error.json"
            detail = {"video_id": video_id, "error": repr(exc)}
            if isinstance(exc, urllib.error.HTTPError):
                detail["status"] = exc.code
                detail["body"] = exc.read().decode("utf-8", errors="replace")
            error_path.write_text(json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            if not args.keep_going:
                raise
            print(f"  error: {exc}", file=sys.stderr)
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)


def build_sft(args: argparse.Namespace) -> None:
    rows = enrich_rows(load_annotation_rows(args.annotations_csv), args.work_dir)
    parsed_dir = args.work_dir / "teacher_labels" / "parsed"
    out_path = args.work_dir / "vlm_planner_sft.jsonl"
    review_dir = args.work_dir / "review_videos_by_seq"
    count = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row in rows:
            parsed_path = parsed_dir / f"{row['video_id']}.json"
            if not parsed_path.exists():
                continue
            video_path = row["linked_video_path"]
            if args.sft_video_source == "review":
                candidate = review_dir / f"{int(row['seq']):04d}.mp4"
                if candidate.exists():
                    video_path = str(candidate)
            label = json.loads(parsed_path.read_text(encoding="utf-8"))
            prompt = (
                "You are the Edit2World VLM planner. Given the video and user request, "
                "return only valid JSON, without markdown. The JSON must match this planner schema: "
                '{"video_id": "...", "task_type": "remove", "edit_prompt": "remove ...", '
                '"target_objects": [{"name": "...", "aliases": [], "role": "...", '
                '"visibility": "...", "temporal_span": {"start_sec": 0.0, "end_sec": 0.0}, '
                '"location_description": "..."}], "protected_objects": [], '
                '"event_summary": "...", "physical_causal_chain": [{"step": 1, '
                '"description": "...", "objects": [], "approx_time_sec": 0.0}], '
                '"counterfactual_expectation": {"if_removed": "...", "affected_regions": [], '
                '"unchanged_regions": [], "uncertainties": []}, "quadmask_spec": {"primary": {}, '
                '"affected": {}, "keep": {}}, "quality_flags": {"usable_for_planner_sft": true, '
                '"needs_human_review": false, "reasons": []}}. '
                f"{PLANNER_VACE_CONTRACT_RULES} "
                f"Set video_id exactly to {row['video_id']}. "
                f"User request: remove {row['remove_objects']}."
            )
            example = {
                "id": row["video_id"],
                "video": video_path,
                "messages": [
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": json.dumps(label, ensure_ascii=False),
                    },
                ],
                "metadata": {
                    "source": "physics-iq",
                    "category": row["category"],
                    "scenario_group": row["scenario_group"],
                    "counterfactual_target_available": False,
                },
            }
            out.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} examples to {out_path}")


def validate_annotations(args: argparse.Namespace) -> None:
    rows = enrich_rows(load_annotation_rows(args.annotations_csv), args.work_dir)
    selected = [r for r in rows if should_label(r)]
    missing_video = [r["video_id"] for r in selected if not Path(r["linked_video_path"]).exists()]
    print(
        json.dumps(
            {
                "rows": len(rows),
                "selected_for_labeling": len(selected),
                "missing_linked_video": len(missing_video),
                "missing_video_ids": missing_video[:20],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def make_review_proxies(args: argparse.Namespace) -> None:
    rows = load_annotation_rows(args.annotations_csv)
    proxy_dir = args.work_dir / "review_proxies_h264_720p"
    numbered_dir = args.work_dir / "review_videos_by_seq"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    numbered_dir.mkdir(parents=True, exist_ok=True)
    review_rows: list[dict[str, str]] = []
    previous_by_id = {}
    previous_by_seq = {}
    if args.output_csv.exists():
        previous_rows = load_annotation_rows(args.output_csv)
        previous_by_id = {
            row["video_id"]: row for row in previous_rows if row.get("video_id")
        }
        previous_by_seq = {
            row["seq"]: row for row in previous_rows if row.get("seq")
        }

    selected_rows = rows
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]

    selected_ids = {row["video_id"] for row in selected_rows}
    for index, row in enumerate(rows, start=1):
        source = Path(row["linked_video_path"])
        proxy = proxy_dir / source.name
        numbered_link = numbered_dir / f"{index:04d}.mp4"
        if numbered_link.exists() or numbered_link.is_symlink():
            if numbered_link.resolve() != proxy.resolve():
                numbered_link.unlink()
                numbered_link.symlink_to(proxy)
        else:
            numbered_link.symlink_to(proxy)

        previous = previous_by_id.get(row["video_id"], previous_by_seq.get(str(index), {}))
        review_rows.append(
            {
                "seq": str(index),
                "include_for_labeling": previous.get("include_for_labeling", "1") or "1",
                "remove_objects": previous.get("remove_objects", ""),
                "protected_objects": previous.get("protected_objects", ""),
                "human_notes": previous.get("human_notes", ""),
                "label_status": previous.get("label_status", "unlabeled") or "unlabeled",
            }
        )

        if row["video_id"] not in selected_ids:
            continue
        if proxy.exists() and not args.overwrite:
            print(f"[{index}/{len(rows)}] exists {proxy.name}")
            continue

        tmp = proxy.with_suffix(".tmp.mp4")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"scale={args.width}:-2",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp),
        ]
        print(f"[{index}/{len(rows)}] proxy {source.name}")
        subprocess.run(cmd, check=True)
        tmp.replace(proxy)

    out_csv = args.output_csv
    fieldnames = [
        "seq",
        "include_for_labeling",
        "remove_objects",
        "protected_objects",
        "human_notes",
        "label_status",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"wrote {out_csv}")
    print(f"review proxies dir: {proxy_dir}")
    print(f"numbered review videos dir: {numbered_dir}")


def add_common_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--descriptions-csv", type=Path, default=DEFAULT_DESCRIPTIONS_CSV)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="Create full-video symlinks, manifest, and annotation CSV")
    add_common_prepare_args(p)
    p.set_defaults(func=prepare)

    p = sub.add_parser("validate-annotations", help="Check completed annotation CSV before API calls")
    p.add_argument("--annotations-csv", type=Path, default=DEFAULT_FULL_ANNOTATIONS_CSV)
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.set_defaults(func=validate_annotations)

    p = sub.add_parser("make-review-proxies", help="Create VSCode/browser-friendly H.264 proxy videos")
    p.add_argument("--annotations-csv", type=Path, default=DEFAULT_FULL_ANNOTATIONS_CSV)
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_WORK_DIR / "human_remove_review_sheet.csv",
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--preset", default="veryfast")
    p.add_argument("--limit", type=int)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=make_review_proxies)

    p = sub.add_parser("label-openrouter", help="Call OpenRouter/Gemini for selected annotation rows")
    p.add_argument("--annotations-csv", type=Path, default=DEFAULT_WORK_DIR / "human_remove_review_sheet.csv")
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    p.add_argument("--http-referer", default="https://localhost/e2w")
    p.add_argument("--title", default="E2W Physics-IQ VLM Labels")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    p.add_argument("--max-video-mb", type=float, default=32.0)
    p.add_argument("--api-video-source", choices=["review", "original"], default="review")
    p.add_argument("--provider-only", action="append", default=[])
    p.add_argument("--allow-fallbacks", action="store_true")
    p.add_argument("--require-parameters", action="store_true")
    p.add_argument("--sleep-seconds", type=float, default=1.0)
    p.add_argument("--limit", type=int)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=label_openrouter)

    p = sub.add_parser("build-sft", help="Build a JSONL SFT file from parsed teacher labels")
    p.add_argument("--annotations-csv", type=Path, default=DEFAULT_WORK_DIR / "human_remove_review_sheet.csv")
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--sft-video-source", choices=["review", "original"], default="review")
    p.set_defaults(func=build_sft)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

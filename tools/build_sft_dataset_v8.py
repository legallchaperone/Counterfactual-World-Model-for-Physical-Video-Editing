#!/usr/bin/env python3
"""Build Qwen2.5-VL SFT data for v8 tool-augmented E2W planner.

Input rows come from Line C annotations with grounded targets and v2 7-field
counterfactual_state. The v8 planner target no longer emits bbox/mask fields;
it emits only target_ref, edit_type=remove, counterfactual_state, and if_removed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2w_v0_common import validate_planner_output_v8  # noqa: E402

DEFAULT_INPUT = Path("/data/cwx/E2W/data/line_c_annotations/seed_150_v2.jsonl")
DEFAULT_OUTPUT_DIR = Path("/data/cwx/E2W/data/planner_sft_v8")
DAVIS_FRAME_ROOT = Path("/data/cwx/E2W/data/raw_datasets/DAVIS/DAVIS/JPEGImages/480p")
RYVOS_FRAME_ROOT = Path("/data/cwx/E2W/data/raw_datasets/refer-youtube-vos/rvos/valid/JPEGImages")
PLANNER_V8_KEYS = ("target_ref", "edit_type", "counterfactual_state", "if_removed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized_source(row: dict[str, Any]) -> str:
    source = str(row.get("source") or "").strip().lower()
    original = str(row.get("original_source") or "").strip().lower()
    merged = f"{source} {original}"
    if "ryvos" in merged or "refer-youtube" in merged:
        return "ryvos"
    if "davis" in merged:
        return "davis"
    raise ValueError(f"Cannot infer source for video_id={row.get('video_id')}: source={row.get('source')!r}")


def frame_name(anchor_frame: Any) -> str:
    text = str(anchor_frame).strip()
    stem = Path(text).stem if Path(text).suffix else text
    try:
        return f"{int(stem):05d}.jpg"
    except ValueError:
        return text if text.endswith(".jpg") else f"{text}.jpg"


def anchor_frame_path(row: dict[str, Any], source: str) -> Path:
    video_id = str(row.get("video_id") or "").strip()
    if not video_id:
        raise ValueError("video_id must be non-empty")
    name = frame_name(row.get("anchor_frame"))
    if source == "davis":
        return DAVIS_FRAME_ROOT / video_id / name
    if source == "ryvos":
        return RYVOS_FRAME_ROOT / video_id / name
    raise ValueError(f"Unsupported source: {source}")


def planner_output(row: dict[str, Any]) -> dict[str, Any]:
    obj = {key: row.get(key) for key in PLANNER_V8_KEYS}
    obj["edit_type"] = "remove"
    return obj


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    source = normalized_source(row)
    image_path = anchor_frame_path(row, source)
    if not image_path.exists():
        raise FileNotFoundError(f"Missing anchor frame for {source}/{row.get('video_id')}: {image_path}")
    assistant_obj = planner_output(row)
    ok, err = validate_planner_output_v8(assistant_obj, source_video_id=str(row.get("video_id") or "unknown"))
    if not ok:
        raise ValueError(f"Invalid planner v8 output for {source}/{row.get('video_id')}: {err}")
    instruction = str(row.get("instruction") or f"remove {row.get('target_ref')}").strip()
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": f"Edit instruction: {instruction}"},
                ],
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_obj, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "source": source,
        "video_id": str(row.get("video_id") or ""),
    }


def stratified_split(rows: list[dict[str, Any]], seed: int, train_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for source in sorted(by_source):
        bucket = list(by_source[source])
        rng.shuffle(bucket)
        train_count = round(len(bucket) * train_ratio)
        train.extend(bucket[:train_count])
        eval_rows.extend(bucket[train_count:])
    rng.shuffle(train)
    rng.shuffle(eval_rows)
    return train, eval_rows


def validate_dataset(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        user = row["messages"][0]
        assistant = row["messages"][1]
        image_path = Path(user["content"][0]["image"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        obj = json.loads(assistant["content"])
        ok, err = validate_planner_output_v8(obj, source_video_id=row.get("video_id", "unknown"))
        if not ok:
            raise ValueError(f"Invalid assistant JSON for {row.get('source')}/{row.get('video_id')}: {err}")


def main() -> int:
    args = parse_args()
    source_rows = load_jsonl(args.input)
    converted = [convert_row(row) for row in source_rows]
    train, eval_rows = stratified_split(converted, args.seed, args.train_ratio)
    validate_dataset(train)
    validate_dataset(eval_rows)

    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(eval_path, eval_rows)

    print(f"input={args.input} rows={len(source_rows)} source={dict(Counter(row['source'] for row in converted))}")
    print(f"train={len(train)} source={dict(Counter(row['source'] for row in train))} path={train_path}")
    print(f"eval={len(eval_rows)} source={dict(Counter(row['source'] for row in eval_rows))} path={eval_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

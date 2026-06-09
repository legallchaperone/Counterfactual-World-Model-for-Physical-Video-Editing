#!/usr/bin/env python3
"""Build add-operation E2W quadmasks from an edited first frame.

The runtime acceptance path uses SAM2 on the edited first frame (repeated as a
short conditioning clip) with planner/model-provided grounding. Pure array helpers
are kept small and testable, but tests must not be confused with the real SAM2
acceptance path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (TOOLS, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_quadmask_from_spec import (  # noqa: E402
    DEFAULT_SAM2_CFG,
    DEFAULT_SAM2_CKPT,
    DEFAULT_SAM2_REPO,
    primary_grounding_from_spec,
    sam2_propagate,
)
from e2w_v0_common import quadmask_to_rgb, video_meta  # noqa: E402

VOID_VALUES = {0, 63, 127, 255}


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def write_rgb_video(path: str | Path, frames: list[np.ndarray], fps: float = 12.0) -> None:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)


def write_gray_video(path: str | Path, arr: np.ndarray, fps: float = 12.0) -> None:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [np.repeat(frame[:, :, None], 3, axis=2).astype(np.uint8) for frame in arr]
    imageio.mimwrite(str(path), frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)


def write_quadmask_preview(path: str | Path, quadmask: np.ndarray, fps: float = 12.0) -> None:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = quadmask_to_rgb(quadmask)
    imageio.mimwrite(str(path), list(rgb), fps=fps, codec="libx264", quality=8, macro_block_size=1)


def repeat_image_video(image: np.ndarray, frame_num: int) -> list[np.ndarray]:
    if frame_num <= 0:
        raise ValueError(f"frame_num must be positive, got {frame_num}")
    return [image.copy() for _ in range(frame_num)]


def dilate_mask(mask: np.ndarray, kernel_size: int = 21, iterations: int = 1) -> np.ndarray:
    import cv2

    if mask.dtype != np.uint8:
        work = mask.astype(np.uint8)
    else:
        work = mask.copy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(work, kernel, iterations=iterations).astype(bool)


def diff_mask(original: np.ndarray, edited: np.ndarray, threshold: int = 24) -> np.ndarray:
    if original.shape != edited.shape:
        edited_img = Image.fromarray(edited).resize((original.shape[1], original.shape[0]), Image.Resampling.LANCZOS)
        edited = np.asarray(edited_img)
    delta = np.abs(edited.astype(np.int16) - original.astype(np.int16)).max(axis=2)
    return delta >= threshold


def contact_band(mask: np.ndarray, band_px: int = 18, width_px: int = 14) -> np.ndarray:
    """Small support/contact band below the Q0 mask; deterministic heuristic."""

    ys, xs = np.where(mask)
    out = np.zeros(mask.shape, dtype=bool)
    if ys.size == 0:
        return out
    y0 = int(ys.max())
    x1 = max(0, int(xs.min()) - width_px)
    x2 = min(mask.shape[1], int(xs.max()) + width_px + 1)
    y1 = min(mask.shape[0], y0 + 1)
    y2 = min(mask.shape[0], y0 + band_px + 1)
    if y1 < y2 and x1 < x2:
        out[y1:y2, x1:x2] = True
    return out


def build_quadmask_from_primary(
    primary_2d: np.ndarray,
    *,
    frame_num: int,
    original_first_frame: np.ndarray | None = None,
    edited_first_frame: np.ndarray | None = None,
    diff_threshold: int = 24,
    dilation_kernel: int = 21,
) -> tuple[np.ndarray, dict[str, Any]]:
    primary = primary_2d.astype(bool)
    if not primary.any():
        raise ValueError("primary mask is empty; cannot build add quadmask")
    q2 = dilate_mask(primary, kernel_size=dilation_kernel) & ~primary
    if original_first_frame is not None and edited_first_frame is not None:
        q2 |= diff_mask(original_first_frame, edited_first_frame, threshold=diff_threshold) & ~primary
    q2 |= contact_band(primary) & ~primary

    quad2d = np.full(primary.shape, 255, dtype=np.uint8)
    quad2d[q2] = 127
    quad2d[primary] = 0
    quadmask = np.repeat(quad2d[None], frame_num, axis=0).astype(np.uint8)
    values = sorted(int(x) for x in np.unique(quadmask))
    bad = set(values) - VOID_VALUES
    if bad:
        raise ValueError(f"invalid quadmask values: {sorted(bad)}")
    meta = {
        "shape": list(quadmask.shape),
        "dtype": str(quadmask.dtype),
        "values": values,
        "q0_area_mean": float((quadmask == 0).mean()),
        "q2_area_mean": float((quadmask == 127).mean()),
        "q3_area_mean": float((quadmask == 255).mean()),
        "temporal_strategy": "repeat_first_frame",
        "q1_used": 63 in values,
    }
    return quadmask, meta


def sam2_primary_from_edited_frame(
    *,
    edited_first_frame: Path,
    quadmask_spec: dict[str, Any],
    out_dir: Path,
    frame_num: int,
    fps: float,
    sam2_repo: Path = DEFAULT_SAM2_REPO,
    sam2_ckpt: Path = DEFAULT_SAM2_CKPT,
    sam2_cfg: str = DEFAULT_SAM2_CFG,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run SAM2 on edited_first_frame repeated as a short clip.

    Uses planner/model-provided add grounding (bbox/point) from quadmask_spec.
    """

    edited = load_rgb(edited_first_frame)
    clip_path = out_dir / "edited_first_frame_sam2_clip.mp4"
    write_rgb_video(clip_path, repeat_image_video(edited, frame_num), fps=fps)
    # Validate grounding against edited clip metadata.
    meta = video_meta(clip_path)
    grounding = primary_grounding_from_spec(quadmask_spec, meta)
    ns = argparse.Namespace(sam2_repo=sam2_repo, sam2_ckpt=sam2_ckpt, sam2_cfg=sam2_cfg)
    propagated = sam2_propagate(ns, clip_path, quadmask_spec)
    primary = propagated[0].astype(bool)
    info = {
        "used_for_quadmask": True,
        "input_image": str(edited_first_frame),
        "clip_path": str(clip_path),
        "prompt_source": "planner_model_grounding",
        "bbox_xyxy": grounding.get("bbox"),
        "point_xy": grounding.get("point"),
        "primary_mask_shape": list(primary.shape),
        "primary_mask_area": int(primary.sum()),
    }
    return primary, info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--original-first-frame", type=Path, required=True)
    p.add_argument("--edited-first-frame", type=Path, required=True)
    p.add_argument("--quadmask-spec", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--frame-num", type=int, default=21)
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--sam2-repo", type=Path, default=DEFAULT_SAM2_REPO)
    p.add_argument("--sam2-ckpt", type=Path, default=DEFAULT_SAM2_CKPT)
    p.add_argument("--sam2-cfg", default=DEFAULT_SAM2_CFG)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.quadmask_spec.read_text(encoding="utf-8"))
    primary, sam2_info = sam2_primary_from_edited_frame(
        edited_first_frame=args.edited_first_frame,
        quadmask_spec=spec,
        out_dir=args.out_dir,
        frame_num=args.frame_num,
        fps=args.fps,
        sam2_repo=args.sam2_repo,
        sam2_ckpt=args.sam2_ckpt,
        sam2_cfg=args.sam2_cfg,
    )
    quadmask, quad_meta = build_quadmask_from_primary(
        primary,
        frame_num=args.frame_num,
        original_first_frame=load_rgb(args.original_first_frame),
        edited_first_frame=load_rgb(args.edited_first_frame),
    )
    np.save(args.out_dir / "primary_mask.npy", primary.astype(np.uint8))
    Image.fromarray(primary.astype(np.uint8) * 255).save(args.out_dir / "primary_mask.png")
    np.save(args.out_dir / "quadmask.npy", quadmask)
    write_quadmask_preview(args.out_dir / "quadmask_preview.mp4", quadmask, fps=args.fps)
    write_gray_video(args.out_dir / "primary_mask_preview.mp4", np.repeat(primary[None].astype(np.uint8) * 255, args.frame_num, axis=0), fps=args.fps)
    (args.out_dir / "quadmask_metadata.json").write_text(
        json.dumps({"sam2": sam2_info, "quadmask": quad_meta}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

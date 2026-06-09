from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from run_control_perturbation_test import (
    build_q0_shifted_quadmask,
    build_q0_suppressed_quadmask,
    load_baseline,
    summarize_control_results,
)


class ControlPerturbationTests(unittest.TestCase):
    def _make_quad(self) -> np.ndarray:
        q = np.full((3, 8, 10), 255, dtype=np.uint8)
        q[:, 2:5, 3:7] = 0
        q[:, 1:6, 2:8] = np.where(q[:, 1:6, 2:8] == 255, 127, q[:, 1:6, 2:8])
        return q

    def test_q0_suppressed_is_all_255(self) -> None:
        quad = self._make_quad()
        suppressed = build_q0_suppressed_quadmask(quad)
        self.assertEqual(suppressed.shape, quad.shape)
        self.assertEqual(suppressed.dtype, np.uint8)
        self.assertEqual(sorted(int(v) for v in np.unique(suppressed).tolist()), [255])

    def test_q0_shifted_moves_edit_region(self) -> None:
        quad = np.full((2, 10, 20), 255, dtype=np.uint8)
        quad[:, 4:6, 4:6] = 0
        shifted = build_q0_shifted_quadmask(quad, shift_x=4, shift_y=0)
        self.assertEqual(shifted.shape, quad.shape)
        self.assertEqual(shifted.dtype, np.uint8)
        self.assertTrue(np.any(shifted[:, 4:6, 8:10] == 0), "Q0 should appear at shifted position")
        self.assertTrue(np.all(shifted[:, 4:6, 4:6] == 255), "Original Q0 position should now be Q3")

    def test_q0_shifted_out_of_bounds_pixels_dropped(self) -> None:
        quad = np.full((1, 4, 4), 255, dtype=np.uint8)
        quad[0, 0, 0] = 0
        shifted = build_q0_shifted_quadmask(quad, shift_x=10, shift_y=10)
        self.assertEqual(sorted(int(v) for v in np.unique(shifted).tolist()), [255])

    def test_summarize_all_pass(self) -> None:
        results = [
            {"test": "operation_swap", "output_video_exists": True, "output_video_bytes": 1000},
            {"test": "q0_suppressed", "output_video_exists": True, "output_video_bytes": 900},
            {"test": "q0_shifted", "output_video_exists": True, "output_video_bytes": 950},
            {"test": "q3_preservation", "visual_review_required": True},
        ]
        summary = summarize_control_results(results)
        self.assertTrue(summary["passed_all_structural"])
        self.assertTrue(summary["visual_review_required"])
        self.assertEqual(summary["evidence_level"], "STRUCTURAL_CONTROL_CANDIDATE")

    def test_summarize_fails_if_any_video_missing(self) -> None:
        results = [
            {"test": "operation_swap", "output_video_exists": False, "output_video_bytes": 0},
            {"test": "q0_suppressed", "output_video_exists": True, "output_video_bytes": 900},
            {"test": "q0_shifted", "output_video_exists": True, "output_video_bytes": 950},
            {"test": "q3_preservation", "visual_review_required": True},
        ]
        summary = summarize_control_results(results)
        self.assertFalse(summary["passed_all_structural"])

    def test_load_baseline_reads_vace_runtime_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            quad = root / "quadmask.npy"
            np.save(quad, np.zeros((1, 2, 2), dtype=np.uint8))
            cond = root / "vace_conditioning_video.mp4"
            cond.write_bytes(b"placeholder")
            gen = root / "generation_mask.mp4"
            gen.write_bytes(b"placeholder")
            meta = {
                "vace_runtime_inputs": {
                    "vace_conditioning_video": str(cond),
                    "quadmask_npy": str(quad),
                    "generation_mask": str(gen),
                    "operation": "remove",
                    "vace_prompt": "The background is revealed.",
                    "frame_num": 21,
                }
            }
            (root / "metadata.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
            baseline = load_baseline(root)
            self.assertEqual(baseline["operation"], "remove")
            self.assertEqual(baseline["frame_num"], 21)
            self.assertEqual(baseline["vace_prompt"], "The background is revealed.")

    def test_load_baseline_raises_on_missing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "metadata.json").write_text(json.dumps({"vace_runtime_inputs": {"operation": "remove"}}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_baseline(root)


if __name__ == "__main__":
    unittest.main()

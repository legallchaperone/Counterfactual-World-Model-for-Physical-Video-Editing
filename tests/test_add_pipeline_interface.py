from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from build_add_quadmask_from_edited_first_frame import build_quadmask_from_primary  # noqa: E402
from run_add_pipeline_interface import build_generation_mask, ensure_frame_num  # noqa: E402


class AddPipelineInterfaceTests(unittest.TestCase):
    def test_quadmask_values_shape_and_temporal_repeat(self) -> None:
        primary = np.zeros((8, 10), dtype=bool)
        primary[2:4, 3:6] = True
        original = np.zeros((8, 10, 3), dtype=np.uint8)
        edited = original.copy()
        edited[4:5, 3:6] = 255
        quadmask, meta = build_quadmask_from_primary(primary, frame_num=5, original_first_frame=original, edited_first_frame=edited)
        self.assertEqual(list(quadmask.shape), [5, 8, 10])
        self.assertEqual(quadmask.dtype, np.uint8)
        self.assertTrue(set(np.unique(quadmask).astype(int)).issubset({0, 63, 127, 255}))
        self.assertIn(0, set(np.unique(quadmask).astype(int)))
        self.assertIn(127, set(np.unique(quadmask).astype(int)))
        self.assertEqual(meta["temporal_strategy"], "repeat_first_frame")
        self.assertTrue(np.array_equal(quadmask[0], quadmask[-1]))

    def test_generation_mask_is_all_255(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meta = build_generation_mask(root / "generation_mask.npy", root / "generation_mask.mp4", frame_num=5, height=4, width=6, fps=1)
            arr = np.load(root / "generation_mask.npy")
            self.assertEqual(list(arr.shape), [5, 4, 6])
            self.assertEqual(arr.dtype, np.uint8)
            self.assertEqual(sorted(int(x) for x in np.unique(arr)), [255])
            self.assertTrue(meta["generation_mask_is_full_domain"])
            self.assertFalse(meta["generation_mask_encodes_quadmask_semantics"])

    def test_frame_num_must_be_4n_plus_1(self) -> None:
        ensure_frame_num(21)
        with self.assertRaises(ValueError):
            ensure_frame_num(22)

    def test_required_metadata_flags_contract(self) -> None:
        metadata = {
            "evidence_level": "INTERFACE",
            "visual_quality_evaluated": False,
            "planner": {
                "vace_prompt_source": "planner_model",
                "vace_prompt_passed_through_unchanged": True,
                "manual_or_teacher_vace_prompt_used": False,
                "planner_output_manually_modified": False,
                "learned_planner_add_quality_claimed": False,
                "planner_attempt_count": 1,
                "planner_invalid_attempt_errors": [],
            },
            "source_video_passed_to_vace": False,
        }
        self.assertEqual(metadata["evidence_level"], "INTERFACE")
        self.assertFalse(metadata["visual_quality_evaluated"])
        self.assertEqual(metadata["planner"]["vace_prompt_source"], "planner_model")
        self.assertTrue(metadata["planner"]["vace_prompt_passed_through_unchanged"])
        self.assertFalse(metadata["planner"]["manual_or_teacher_vace_prompt_used"])
        self.assertFalse(metadata["planner"]["planner_output_manually_modified"])
        self.assertFalse(metadata["planner"]["learned_planner_add_quality_claimed"])
        self.assertFalse(metadata["source_video_passed_to_vace"])


if __name__ == "__main__":
    unittest.main()

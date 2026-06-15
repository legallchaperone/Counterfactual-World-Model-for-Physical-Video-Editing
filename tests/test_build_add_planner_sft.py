from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import build_add_planner_sft_dataset as b  # noqa: E402
from e2w_v0_common import validate_add_planner_output  # noqa: E402


class BuildAddPlannerSftTests(unittest.TestCase):
    def test_primary_mask_2d_collapses_time_axis(self) -> None:
        m = np.zeros((5, 10, 12), dtype=np.uint8)
        m[:, 2:4, 3:5] = 1
        out = b.primary_mask_2d(m)
        self.assertEqual(out.shape, (10, 12))
        self.assertTrue(out.dtype == bool)

    def test_point_bbox_norm1000_centroid_and_range(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:61, 40:61] = True  # centered square
        point, bbox = b.point_bbox_norm1000(mask)
        self.assertTrue(all(0 <= v <= 1000 for v in point + bbox))
        self.assertAlmostEqual(point[0], 505, delta=15)
        self.assertAlmostEqual(point[1], 505, delta=15)
        self.assertLess(bbox[0], bbox[2])
        self.assertLess(bbox[1], bbox[3])

    def test_empty_mask_raises(self) -> None:
        with self.assertRaises(ValueError):
            b.point_bbox_norm1000(np.zeros((20, 20), dtype=bool))

    def test_built_assistant_obj_passes_add_contract(self) -> None:
        mask = np.zeros((80, 120), dtype=bool)
        mask[30:50, 50:80] = True
        point, bbox = b.point_bbox_norm1000(mask)
        obj = b.build_add_assistant_obj(
            "red ball",
            "A clean surface with a red ball resting near the center and a soft contact shadow.",
            point,
            bbox,
        )
        ok, err = validate_add_planner_output(obj)
        self.assertTrue(ok, err)

    def test_degenerate_bbox_is_widened(self) -> None:
        mask = np.zeros((50, 50), dtype=bool)
        mask[25, 25] = True  # single pixel
        _point, bbox = b.point_bbox_norm1000(mask)
        self.assertLess(bbox[0], bbox[2])
        self.assertLess(bbox[1], bbox[3])


if __name__ == "__main__":
    unittest.main()

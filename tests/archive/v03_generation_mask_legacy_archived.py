from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_vace_v03_quad_experiment as v03  # noqa: E402


class ArchivedV03GenerationMaskTests(unittest.TestCase):
    def test_legacy_generation_mask_modes_are_binary(self) -> None:
        quad = np.array(
            [
                [[255, 0], [63, 127]],
                [[255, 255], [0, 127]],
            ],
            dtype=np.uint8,
        )
        local = v03.generation_mask_from_quadmask(quad, "quadmask-editable")
        self.assertEqual(sorted(int(x) for x in np.unique(local)), [0, 255])
        self.assertEqual(local[0, 0, 0], 0)
        self.assertEqual(local[0, 0, 1], 255)
        self.assertEqual(local[0, 1, 0], 255)
        self.assertEqual(local[0, 1, 1], 255)

        future = v03.generation_mask_from_quadmask(quad, "future-full-frame")
        self.assertEqual(int(future[0].max()), 0)
        self.assertEqual(int(future[1].min()), 255)


if __name__ == "__main__":
    unittest.main()

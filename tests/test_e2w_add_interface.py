from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import e2w_add  # noqa: E402


def _valid_parsed() -> dict[str, object]:
    return {
        "target_ref": "a red mug",
        "edit_type": "add",
        "vace_prompt": "A table with a red mug placed near the center, consistent lighting and shadow.",
        "primary_point": [500, 480],
        "primary_bbox": [450, 440, 560, 540],
    }


class E2WAddInterfaceTests(unittest.TestCase):
    def test_quadmask_spec_maps_norm1000_point_into_keyframes(self) -> None:
        spec = e2w_add.add_quadmask_spec_from_planner(_valid_parsed())
        self.assertEqual(spec["operation"], "add")
        self.assertEqual(spec["schema_version"], "e2w.quadmask_spec.v1")
        kf = spec["primary"]["keyframes"][0]
        self.assertEqual(kf["frame_index"], 0)
        self.assertEqual(kf["positive_points_norm1000"], [[500, 480]])
        self.assertEqual(kf["bbox_xyxy_norm1000"], [450, 440, 560, 540])

    def test_quadmask_spec_omits_bbox_when_absent(self) -> None:
        parsed = _valid_parsed()
        parsed.pop("primary_bbox")
        spec = e2w_add.add_quadmask_spec_from_planner(parsed)
        self.assertNotIn("bbox_xyxy_norm1000", spec["primary"]["keyframes"][0])

    def test_add_edit_instruction_is_planner_driven_with_region(self) -> None:
        # names the object and encodes a region derived from the planner point
        instr = e2w_add.add_edit_instruction("a red mug", [500, 500])
        self.assertIn("a red mug", instr)
        self.assertIn("center", instr)
        self.assertIn("left", e2w_add.add_edit_instruction("a vase", [100, 100]))
        self.assertIn("top", e2w_add.add_edit_instruction("a vase", [100, 100]))
        self.assertIn("bottom", e2w_add.add_edit_instruction("a vase", [900, 900]))
        self.assertIn("right", e2w_add.add_edit_instruction("a vase", [900, 900]))

    def test_inpaint_mask_from_bbox_is_white_inside_expanded_box(self) -> None:
        import numpy as np
        parsed = _valid_parsed()  # bbox [450,440,560,540] norm1000
        mask, box = e2w_add.build_inpaint_mask_from_planner(parsed, height=480, width=832, margin=0.0)
        self.assertEqual(mask.shape, (480, 832))
        self.assertEqual(set(np.unique(mask).tolist()), {0, 255})
        # box ~ bbox in pixels: x in [450/1000*832, 560/1000*832], y in [440/1000*480, 540/1000*480]
        self.assertAlmostEqual(box[0], round(450 / 1000 * 832), delta=2)
        self.assertAlmostEqual(box[3], round(540 / 1000 * 480), delta=2)
        self.assertTrue(mask[box[1] + 2, box[0] + 2] == 255)
        self.assertTrue(mask[0, 0] == 0)

    def test_inpaint_mask_margin_expands(self) -> None:
        parsed = _valid_parsed()
        _m0, b0 = e2w_add.build_inpaint_mask_from_planner(parsed, 480, 832, margin=0.0)
        _m1, b1 = e2w_add.build_inpaint_mask_from_planner(parsed, 480, 832, margin=0.5)
        self.assertLess(b1[0], b0[0])
        self.assertGreater(b1[2], b0[2])

    def test_inpaint_mask_falls_back_to_point_box_without_bbox(self) -> None:
        parsed = _valid_parsed()
        parsed.pop("primary_bbox")
        mask, box = e2w_add.build_inpaint_mask_from_planner(parsed, 480, 832, margin=0.0)
        self.assertLess(box[0], box[2])
        self.assertLess(box[1], box[3])

    def test_ensure_frame_num_requires_4n_plus_1(self) -> None:
        e2w_add.ensure_frame_num(21)
        for bad in (20, 0, -1, 22):
            with self.assertRaises(ValueError):
                e2w_add.ensure_frame_num(bad)

    def test_wan_frame_num_at_most_matches_source(self) -> None:
        self.assertEqual(e2w_add.wan_frame_num_at_most(81), 81)  # already 4n+1
        self.assertEqual(e2w_add.wan_frame_num_at_most(80), 77)  # round down to 4n+1
        self.assertEqual(e2w_add.wan_frame_num_at_most(20), 17)
        self.assertEqual(e2w_add.wan_frame_num_at_most(1), 1)
        # every result is Wan-compatible
        for n in range(1, 130):
            e2w_add.ensure_frame_num(e2w_add.wan_frame_num_at_most(n))

    def test_run_add_planner_returns_valid_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                e2w_add.core,
                "run_planner",
                return_value=[{"raw_output": "{}", "parsed": _valid_parsed(), "json_parse_ok": True}],
            ) as planner:
                out = e2w_add.run_add_planner(
                    Path(td) / "frame.png",
                    "Add a red mug.",
                    "s1",
                    base_model=Path("/base"),
                    adapter=Path("/add-lora"),
                    attempts=3,
                    max_new_tokens=64,
                    temperature=0.0,
                    run_dir=Path(td),
                )
            self.assertEqual(out["target_ref"], "a red mug")
            planner.assert_called_once()

    def test_run_add_planner_raises_after_invalid_retries(self) -> None:
        bad = {"target_ref": "a red mug", "edit_type": "remove", "vace_prompt": "x", "primary_point": [1, 1]}
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                e2w_add.core,
                "run_planner",
                return_value=[{"raw_output": "{}", "parsed": bad, "json_parse_ok": True}],
            ) as planner:
                with self.assertRaisesRegex(RuntimeError, "contract validation"):
                    e2w_add.run_add_planner(
                        Path(td) / "frame.png",
                        "Add a red mug.",
                        "s1",
                        base_model=Path("/base"),
                        adapter=Path("/add-lora"),
                        attempts=2,
                        max_new_tokens=64,
                        temperature=0.0,
                        run_dir=Path(td),
                    )
            self.assertEqual(planner.call_count, 2)


if __name__ == "__main__":
    unittest.main()

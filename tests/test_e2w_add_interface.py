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

    def test_ensure_frame_num_requires_4n_plus_1(self) -> None:
        e2w_add.ensure_frame_num(21)
        for bad in (20, 0, -1, 22):
            with self.assertRaises(ValueError):
                e2w_add.ensure_frame_num(bad)

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

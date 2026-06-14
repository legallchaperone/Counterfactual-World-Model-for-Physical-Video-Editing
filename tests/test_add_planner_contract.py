from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from e2w_v0_common import (  # noqa: E402
    build_add_planner_user_prompt,
    validate_add_planner_output,
)


def _valid() -> dict[str, object]:
    return {
        "target_ref": "a red mug",
        "edit_type": "add",
        "vace_prompt": "A cozy table with a red mug placed near the center, consistent lighting and shadow.",
        "primary_point": [500, 480],
        "primary_bbox": [450, 440, 560, 540],
    }


class AddPlannerContractTests(unittest.TestCase):
    def test_prompt_requests_add_contract_fields_not_v6(self) -> None:
        prompt = build_add_planner_user_prompt("Add a red mug on the table.", sample_id="s1")
        self.assertIn("edit_type", prompt)
        self.assertIn("primary_point", prompt)
        self.assertIn("vace_prompt", prompt)
        self.assertIn("Add a red mug on the table.", prompt)
        # must not leak the archived v6 executable-planner schema keys
        for v6_key in ("quadmask_spec", "physical_causal_chain", "protected_objects", "task_type"):
            self.assertNotIn(v6_key, prompt)

    def test_valid_add_output_passes(self) -> None:
        ok, err = validate_add_planner_output(_valid())
        self.assertTrue(ok, err)
        self.assertIsNone(err)

    def test_bbox_is_optional(self) -> None:
        obj = _valid()
        obj.pop("primary_bbox")
        ok, err = validate_add_planner_output(obj)
        self.assertTrue(ok, err)

    def test_edit_type_must_be_add(self) -> None:
        obj = _valid()
        obj["edit_type"] = "remove"
        ok, err = validate_add_planner_output(obj)
        self.assertFalse(ok)
        self.assertIn("edit_type", err)

    def test_vace_prompt_must_name_object(self) -> None:
        obj = _valid()
        obj["vace_prompt"] = "A cozy table with soft lighting near the center."  # no 'mug'/'red'
        ok, err = validate_add_planner_output(obj)
        self.assertFalse(ok)
        self.assertIn("name the added object", err)

    def test_removal_residue_wording_rejected(self) -> None:
        obj = _valid()
        obj["vace_prompt"] = "The red mug is now gone and the area is empty."
        ok, err = validate_add_planner_output(obj)
        self.assertFalse(ok)
        self.assertIn("removal-residue", err)

    def test_primary_point_must_be_norm1000(self) -> None:
        for bad in ([1200, 500], [500], "x", [500, -1]):
            obj = _valid()
            obj["primary_point"] = bad
            ok, err = validate_add_planner_output(obj)
            self.assertFalse(ok, f"expected reject for primary_point={bad!r}")
            self.assertIn("primary_point", err)

    def test_bad_bbox_rejected(self) -> None:
        obj = _valid()
        obj["primary_bbox"] = [560, 440, 450, 540]  # x1 > x2
        ok, err = validate_add_planner_output(obj)
        self.assertFalse(ok)
        self.assertIn("primary_bbox", err)


if __name__ == "__main__":
    unittest.main()

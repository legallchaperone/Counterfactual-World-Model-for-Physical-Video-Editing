from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import build_add_quadmask_from_edited_first_frame as add_masks  # noqa: E402
from build_add_quadmask_from_edited_first_frame import build_quadmask_from_primary  # noqa: E402
from e2w_v0_common import normalize_to_e2w_contract, serialize_vace_prompt  # noqa: E402
from run_add_pipeline_interface import (  # noqa: E402
    build_add_planner_user_prompt,
    build_conditioning_video,
    build_generation_mask,
    ensure_frame_num,
    planner_target_ref_from_raw,
    write_split_jsonl,
)


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

    def test_add_planner_prompt_uses_current_add_contract_not_v6_executable_schema(self) -> None:
        prompt = build_add_planner_user_prompt("add_001", "Add a red mug on the table.", attempt=2)
        self.assertIn('"vace_prompt"', prompt)
        self.assertIn('"target_ref"', prompt)
        self.assertIn("primary_point_norm1000", prompt)
        self.assertNotIn("e2w.planner_io.v6_executable.v1", prompt)
        self.assertNotIn("executable planner schema", prompt.lower())
        self.assertNotIn("quadmask_spec.primary", prompt)
        self.assertNotIn("if_removed", prompt)
        self.assertIn("must not contain removal-residue language", prompt)

    def test_split_jsonl_prompt_does_not_inject_archived_schema_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "split.jsonl"
            write_split_jsonl(path, sample_id="add_001", video=Path("/tmp/source.mp4"), user_prompt="Add a red mug.", attempt=1)
            row = json.loads(path.read_text(encoding="utf-8"))
            content = row["messages"][0]["content"]
            self.assertIn('"vace_prompt"', content)
            self.assertNotIn("e2w.planner_io.v6_executable.v1", content)
            self.assertNotIn("quadmask_spec", content)

    def test_current_add_raw_output_normalizes_to_planner_vace_prompt(self) -> None:
        raw = {
            "operation": "add",
            "target_ref": "red mug",
            "vace_prompt": "A red mug sits naturally on the table near the center of the image.",
            "primary_point_norm1000": [500, 520],
            "affected_regions": ["table contact shadow"],
        }
        sample = {"id": "add_001", "messages": [{"role": "user", "content": "User request: Add a red mug."}]}
        meta = {"width": 1000, "height": 1000, "frame_count": 21}
        edit_plan, spec = normalize_to_e2w_contract(raw, sample, meta, source="planner_pred")
        self.assertEqual(edit_plan["edit_subject"]["label"], "red mug")
        self.assertEqual(edit_plan["edited_scene"]["caption"], raw["vace_prompt"])
        self.assertEqual(serialize_vace_prompt(edit_plan).splitlines()[1], raw["vace_prompt"])
        self.assertEqual(spec["primary"]["point"], [500, 520])

    def test_planner_target_ref_comes_from_top_level_raw_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw_path = Path(td) / "raw.pred.json"
            raw_path.write_text(json.dumps({"target_ref": "red mug", "target_objects": [{"name": "wrong legacy label"}]}), encoding="utf-8")
            self.assertEqual(planner_target_ref_from_raw(raw_path), "red mug")

    def test_conditioning_video_records_zero_filled_future_frames(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "edited_first_frame.png"
            out = root / "conditioning.mp4"
            image = np.zeros((8, 10, 3), dtype=np.uint8)
            image[0, 0] = [255, 0, 0]
            Image.fromarray(image).save(first)
            meta = build_conditioning_video(first, out, frame_num=5, fps=1)
            self.assertTrue(out.exists())
            self.assertTrue(meta["frame_0_is_edited_first_frame"])
            self.assertTrue(meta["future_frames_are_zero_filled"])
            self.assertFalse(meta["future_frames_source_video_used"])

    def test_frame_num_must_be_4n_plus_1(self) -> None:
        ensure_frame_num(21)
        with self.assertRaises(ValueError):
            ensure_frame_num(22)

    def test_sam2_propagation_uses_validated_grounding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            edited_first_frame = root / "edited.png"
            Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(edited_first_frame)
            validated_grounding = {
                "bbox": [1, 1, 4, 3],
                "point": [2, 2],
                "negative_points": [[0, 0]],
                "frame_index": 0,
                "coordinate_source": "pixel",
            }
            seen_specs = []

            def fake_propagate(_ns: object, _clip_path: Path, spec: dict[str, object]) -> np.ndarray:
                seen_specs.append(spec)
                return np.ones((5, 4, 6), dtype=bool)

            with (
                mock.patch.object(add_masks, "video_meta", return_value={"width": 6, "height": 4, "frame_count": 5}),
                mock.patch.object(add_masks, "primary_grounding_from_spec", return_value=validated_grounding),
                mock.patch.object(add_masks, "sam2_propagate", side_effect=fake_propagate),
            ):
                primary, info = add_masks.sam2_primary_from_edited_frame(
                    edited_first_frame=edited_first_frame,
                    quadmask_spec={"primary": {"point": [9999, 9999]}},
                    out_dir=root,
                    frame_num=5,
                    fps=1,
                )

            self.assertEqual(seen_specs, [{"primary": {"first_frame_bbox": [1, 1, 4, 3], "point": [2, 2], "negative_points": [[0, 0]]}}])
            self.assertEqual(primary.shape, (4, 6))
            self.assertEqual(info["bbox_xyxy"], [1, 1, 4, 3])
            self.assertEqual(info["point_xy"], [2, 2])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from e2w_v0_common import (  # noqa: E402
    PLANNER_IO_SCHEMA_VERSION,
    VacePromptContractError,
    build_planner_user_prompt,
    infer_actual_operation_from_raw,
    infer_operation_from_sample,
    infer_operation_from_text,
    normalize_to_e2w_contract,
    parse_json_output,
    resolve_expected_operation,
    serialize_first_frame_prompt,
    serialize_vace_prompt,
    validate_edit_plan,
    validate_quadmask_spec,
)
import build_quadmask_from_spec as build_quadmask  # noqa: E402
import export_teacher_grounded_bundle as export_teacher  # noqa: E402
import package_v02_qwen_vace_smoke as package_v02  # noqa: E402
import physics_iq_vlm_pipeline  # noqa: E402
import relabel_quadmask_specs_with_grounding as relabel_grounding  # noqa: E402
import rewrite_planner_user_prompt_schema as rewrite_prompt_schema  # noqa: E402
import run_v02_qwen_vace_smoke as run_v02  # noqa: E402
import run_vace_v0  # noqa: E402


SMOKE_IDS = ["0052", "0056", "0070", "0076", "0112", "0077", "0341", "0128"]
INPUT_JSONL = ROOT / "tests/fixtures/vlm_planner_sft_eval_v6_teacher_grounded.jsonl"


def load_smoke_rows() -> dict[str, dict]:
    rows = {}
    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") in SMOKE_IDS:
                rows[row["id"]] = row
    missing = sorted(set(SMOKE_IDS) - set(rows))
    if missing:
        raise AssertionError(f"Missing smoke rows: {missing}")
    return rows


def assistant_json(row: dict) -> dict:
    for message in row.get("messages", []):
        if message.get("role") == "assistant":
            content = message.get("content")
            return json.loads(content) if isinstance(content, str) else content
    raise AssertionError(f"No assistant message for {row.get('id')}")


def smoke_edit_plans() -> dict[str, dict]:
    rows = load_smoke_rows()
    meta = {"width": 832, "height": 480, "frame_count": 81}
    plans = {}
    for sample_id, row in rows.items():
        raw = assistant_json(row)
        plan, _ = normalize_to_e2w_contract(raw, row, meta, source="teacher_executable_v6")
        plans[sample_id] = {"raw": raw, "plan": plan}
    return plans


class V02PromptContractTests(unittest.TestCase):
    def test_first_frame_prompt_names_target_and_protected_objects(self) -> None:
        for sample_id, bundle in smoke_edit_plans().items():
            raw = bundle["raw"]
            plan = bundle["plan"]
            prompt = serialize_first_frame_prompt(plan)
            target = plan["edit_subject"]["label"]
            aliases = plan["edit_subject"]["aliases"]
            descriptor = plan["edit_subject"]["visual_descriptor"]

            self.assertNotIn("primary subject", prompt.lower(), sample_id)
            self.assertIn(target.lower(), prompt.lower(), sample_id)
            for alias in aliases:
                self.assertIn(alias.lower(), prompt.lower(), sample_id)
            if descriptor and descriptor.lower() != target.lower():
                self.assertIn(descriptor.rstrip(".").lower(), prompt.lower(), sample_id)
            self.assertIn("plausible local background", prompt.lower(), sample_id)
            self.assertNotIn("fill the revealed area as", prompt.lower(), sample_id)

            protected = raw.get("protected_objects") or []
            self.assertEqual(protected, plan["operation_details"]["protected_objects"], sample_id)
            self.assertEqual(protected, plan["edit_subject"]["excluded_non_target_parts"], sample_id)

        prompt_0077 = serialize_first_frame_prompt(smoke_edit_plans()["0077"]["plan"])
        first_line_0077 = prompt_0077.splitlines()[0].lower()
        self.assertIn("spotlight", prompt_0077.lower())
        self.assertIn("yellow mug", prompt_0077.lower())
        self.assertIn("turntable", prompt_0077.lower())
        self.assertNotIn("yellow mug", first_line_0077)
        self.assertNotIn("turntable", first_line_0077)

    def test_vace_prompt_is_neutral_and_target_free(self) -> None:
        forbidden = re.compile(r"\b(remove|delete|erase|removed)\b", re.IGNORECASE)
        hard_fail_ids = {"0076", "0077", "0128"}
        for sample_id, bundle in smoke_edit_plans().items():
            plan = bundle["plan"]
            if sample_id in hard_fail_ids:
                with self.assertRaisesRegex(VacePromptContractError, "target-contaminated planner text"):
                    serialize_vace_prompt(plan)
                continue
            prompt = serialize_vace_prompt(plan)
            self.assertIsNone(forbidden.search(prompt), sample_id)
            self.assertNotIn("removed subject", prompt.lower(), sample_id)
            target_terms = [
                plan["edit_subject"]["label"],
                *plan["edit_subject"]["aliases"],
                *plan["edit_subject"]["included_parts"],
                plan["operation_details"]["target_object"]["label"],
            ]
            for term in target_terms:
                if len(term) >= 3:
                    self.assertIsNone(re.search(rf"\b{re.escape(term)}\b", prompt, re.IGNORECASE), (sample_id, term))

        prompt_0341 = serialize_vace_prompt(smoke_edit_plans()["0341"]["plan"]).lower()
        self.assertIn("one potato", prompt_0341)

    def test_vace_prompt_hard_fails_instead_of_fallback(self) -> None:
        with self.assertRaisesRegex(VacePromptContractError, "sample 0128"):
            serialize_vace_prompt(smoke_edit_plans()["0128"]["plan"])
        with self.assertRaisesRegex(VacePromptContractError, "sample 0076"):
            serialize_vace_prompt(smoke_edit_plans()["0076"]["plan"])
        with self.assertRaisesRegex(VacePromptContractError, "sample 0077"):
            serialize_vace_prompt(smoke_edit_plans()["0077"]["plan"])

    def test_vace_prompt_rejects_neutral_fallback(self) -> None:
        plan = {
            "schema_version": "e2w.edit_plan.v1",
            "operation": "remove",
            "source_video_id": "unit",
            "edit_subject": {"label": "red ball", "aliases": [], "included_parts": ["red ball"]},
            "operation_details": {
                "target_object": {"label": "red ball"},
                "physical_consequences": [],
                "protected_objects": [],
            },
            "edited_scene": {"caption": "", "outcome_effects": []},
        }
        with self.assertRaisesRegex(VacePromptContractError, "no target-free semantic line"):
            serialize_vace_prompt(plan)

    def test_planner_prompt_requests_target_free_vace_text_and_counts(self) -> None:
        rules = physics_iq_vlm_pipeline.PLANNER_VACE_CONTRACT_RULES.lower()
        self.assertIn("do not name the removed target object", rules)
        self.assertIn("visible target subpart", rules)
        self.assertIn("visible count of a non-target object", rules)
        self.assertIn("one potato", rules)

    def test_planner_user_prompt_schema_is_v6_executable(self) -> None:
        prompt = build_planner_user_prompt("0077", "remove the spotlight", operation="remove")
        self.assertIn(PLANNER_IO_SCHEMA_VERSION, prompt)
        self.assertIn("bbox_xyxy_norm1000", prompt)
        self.assertIn("positive_points_norm1000", prompt)
        self.assertIn("grid_shape", prompt)
        self.assertIn("frame_ranges", prompt)
        self.assertIn("quadmask_spec.operation", prompt)
        self.assertNotIn('"quadmask_spec": {"primary": {}, "affected": {}, "keep": {}}', prompt)

    def test_operation_validation_supports_explicit_and_inferred_add(self) -> None:
        self.assertEqual(
            infer_operation_from_text("Add a yellow mug on the rotating turntable in front of the spotlight"),
            "add",
        )
        plan = {
            "schema_version": "e2w.edit_plan.v1",
            "operation": "add",
            "user_prompt": "Add a yellow mug on the rotating turntable in front of the spotlight",
            "edit_subject": {
                "label": "yellow mug",
                "aliases": ["mug"],
                "visual_descriptor": "a yellow mug on the rotating turntable",
                "excluded_non_target_parts": ["turntable", "spotlight"],
            },
            "operation_details": {
                "target_object": {"label": "yellow mug"},
                "protected_objects": ["turntable", "spotlight"],
                "physical_consequences": ["The mug stays on the rotating turntable."],
                "local_fill_instruction": "Place the yellow mug naturally on the turntable.",
                "visual_effects_to_add": ["contact shadow"],
            },
            "edited_scene": {
                "caption": "A yellow mug is on the rotating turntable in front of the spotlight.",
                "outcome_effects": ["The turntable continues rotating with the mug visible."],
            },
        }
        metrics = validate_edit_plan(plan, expected_operation="add")
        self.assertTrue(metrics["operation_accuracy"])
        self.assertEqual(metrics["expected_operation"], "add")
        prompt = serialize_first_frame_prompt(plan).lower()
        self.assertIn("add the yellow mug", prompt)
        self.assertNotIn("remove only", prompt)
        self.assertNotIn("delete", prompt)
        vace_prompt = serialize_vace_prompt(plan).lower()
        self.assertIn("yellow mug", vace_prompt)
        self.assertIn("added target object count:", vace_prompt)
        self.assertIn("one yellow mug", vace_prompt)
        self.assertNotIn("visible non-target object counts to preserve:\n- one yellow mug", vace_prompt)
        self.assertNotRegex(vace_prompt, r"\b(remove|delete|erase|removed)\b")

    def test_expected_add_does_not_coerce_remove_raw_operation(self) -> None:
        rows = load_smoke_rows()
        row = rows["0341"]
        raw = assistant_json(row)
        actual, source = infer_actual_operation_from_raw(raw, row)
        self.assertEqual(actual, "remove")
        self.assertEqual(source, "raw.task_type")

        meta = {"width": 832, "height": 480, "frame_count": 81}
        plan, spec = normalize_to_e2w_contract(raw, row, meta, source="teacher_executable_v6", operation="add")
        metrics = validate_edit_plan(plan, expected_operation="add", actual_operation_source=source)
        self.assertEqual(plan["operation"], "remove")
        self.assertEqual(spec["operation"], "remove")
        self.assertFalse(metrics["operation_accuracy"])
        self.assertEqual(metrics["actual_operation"], "remove")
        self.assertEqual(metrics["expected_operation"], "add")
        self.assertIsNone(resolve_expected_operation("auto", plan=plan))

    def test_forced_add_cannot_bypass_remove_vace_hard_fail(self) -> None:
        rows = load_smoke_rows()
        row = rows["0076"]
        raw = assistant_json(row)
        meta = {"width": 832, "height": 480, "frame_count": 81}
        plan, _ = normalize_to_e2w_contract(raw, row, meta, source="teacher_executable_v6", operation="add")
        self.assertEqual(plan["operation"], "remove")
        with self.assertRaisesRegex(VacePromptContractError, "target-contaminated planner text"):
            serialize_vace_prompt(plan)

    def test_add_vace_prompt_rejects_removal_residue(self) -> None:
        plan = {
            "schema_version": "e2w.edit_plan.v1",
            "operation": "add",
            "source_video_id": "unit_add",
            "edit_subject": {"label": "yellow mug", "aliases": ["mug"], "included_parts": ["yellow mug"]},
            "operation_details": {
                "target_object": {"label": "yellow mug"},
                "physical_consequences": ["The spotlight illuminates the wall where the mug was."],
                "protected_objects": ["turntable"],
            },
            "edited_scene": {"caption": "A yellow mug is on the turntable.", "outcome_effects": []},
        }
        with self.assertRaisesRegex(VacePromptContractError, "removal-residue planner text"):
            serialize_vace_prompt(plan)

        for bad_line in [
            "The yellow mug is absent from the turntable.",
            "The mug is missing from the turntable.",
            "The mug is gone from the turntable.",
            "The mug is no longer visible.",
            "The mug has been made absent.",
        ]:
            plan["operation_details"]["physical_consequences"] = [bad_line]
            with self.assertRaisesRegex(VacePromptContractError, "removal-residue planner text"):
                serialize_vace_prompt(plan)

    def test_operation_inference_ignores_system_create_include_words(self) -> None:
        sample = {
            "messages": [
                {"role": "system", "content": "Create JSON. Include required fields."},
                {"role": "user", "content": "remove red ball"},
            ]
        }
        self.assertEqual(infer_operation_from_sample(sample), "remove")
        self.assertIsNone(infer_operation_from_text("Create JSON. Include required fields."))

        ambiguous = {"messages": [{"role": "user", "content": "Add a mug. Do not remove the table."}]}
        self.assertIsNone(infer_operation_from_sample(ambiguous))

    def test_teacher_export_writes_normalized_operation_spec(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_dir = Path(tmp)
            video_path = run_dir / "source.mp4"
            import imageio.v2 as imageio

            imageio.mimwrite(
                str(video_path),
                [np.zeros((16, 16, 3), dtype=np.uint8)],
                fps=1.0,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
            raw = {
                "task_type": "add",
                "edit_prompt": "add yellow mug",
                "event_summary": "A turntable sits under a spotlight.",
                "target_objects": [{"name": "yellow mug", "location_description": "on the turntable"}],
                "protected_objects": ["turntable"],
                "counterfactual_expectation": {
                    "if_removed": "One yellow mug is visible on the turntable.",
                    "affected_regions": ["mug contact area"],
                    "unchanged_regions": ["turntable"],
                },
                "quadmask_spec": {
                    "schema_version": "e2w.quadmask_spec.v1",
                    "primary": {
                        "object_name": "yellow mug insertion area",
                        "keyframes": [
                            {
                                "frame_index": 0,
                                "bbox_xyxy_norm1000": [250, 250, 750, 750],
                                "positive_points_norm1000": [[500, 500]],
                            }
                        ],
                    },
                    "affected": {"grid_shape": [2, 2], "frame_ranges": [{"start_frame": 0, "end_frame": 0, "cells": ["A1"]}]},
                },
            }
            row = {
                "id": "unit_add",
                "video": str(video_path),
                "messages": [
                    {"role": "user", "content": "Add a yellow mug on the turntable"},
                    {"role": "assistant", "content": json.dumps(raw)},
                ],
            }
            args = argparse.Namespace(run_dir=run_dir, mode="mode_test", operation="auto", force=True)
            entry, metrics = export_teacher.process_row(args, row)
            spec = json.loads(Path(entry["paths"]["quadmask_spec"]).read_text(encoding="utf-8"))
            self.assertEqual(spec["operation"], "add")
            self.assertIn("affected", spec)
            self.assertIn("keyframes", spec["primary"])
            self.assertTrue(metrics["operation_accuracy"])

    def test_operation_mismatch_blocks_manifest_and_mask_stage(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_dir = Path(tmp)
            video_path = run_dir / "source.mp4"
            import imageio.v2 as imageio

            imageio.mimwrite(
                str(video_path),
                [np.zeros((16, 16, 3), dtype=np.uint8)],
                fps=1.0,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
            raw = {
                "task_type": "remove",
                "edit_prompt": "remove yellow mug",
                "event_summary": "A mug sits on a table.",
                "target_objects": [{"name": "yellow mug", "location_description": "on the table"}],
                "counterfactual_expectation": {
                    "if_removed": "The table remains visible.",
                    "affected_regions": ["table area"],
                    "unchanged_regions": ["background"],
                },
                "quadmask_spec": {
                    "schema_version": "e2w.quadmask_spec.v1",
                    "primary": {
                        "object_name": "yellow mug",
                        "keyframes": [
                            {
                                "frame_index": 0,
                                "bbox_xyxy_norm1000": [250, 250, 750, 750],
                                "positive_points_norm1000": [[500, 500]],
                            }
                        ],
                    },
                    "affected": {"grid_shape": [2, 2], "frame_ranges": [{"start_frame": 0, "end_frame": 0, "cells": ["A1"]}]},
                },
            }
            row = {
                "id": "unit_mismatch",
                "video": str(video_path),
                "messages": [
                    {"role": "user", "content": "Add a yellow mug on the table"},
                    {"role": "assistant", "content": json.dumps(raw)},
                ],
            }
            args = argparse.Namespace(run_dir=run_dir, mode="mode_test", operation="add", force=True)
            entry, metrics = export_teacher.process_row(args, row)
            self.assertEqual(entry["status"], "operation_mismatch")
            self.assertFalse(metrics["operation_accuracy"])

            mask_args = argparse.Namespace(run_dir=run_dir, mode="mode_test")
            mask_entry = build_quadmask.build_one(mask_args, entry)
            self.assertEqual(mask_entry["status"], "failed")
            self.assertEqual(mask_entry["failure_source"], "planner")

    def test_planner_json_parser_rejects_nested_object_salvage(self) -> None:
        raw, error = parse_json_output(
            'The answer is {"name": "yellow mug", "aliases": [], "role": "target"}'
        )
        self.assertIsNone(raw)
        self.assertIn("Expecting value", error)

        raw, error = parse_json_output(
            '{"name": "yellow mug", "aliases": [], "role": "target"}'
        )
        self.assertIsNone(raw)
        self.assertIn("missing top-level keys", error)

    def test_old_text_quadmask_schema_is_not_executable(self) -> None:
        metrics = validate_quadmask_spec(
            {
                "schema_version": "e2w.quadmask_spec.v1",
                "operation": "remove",
                "primary": {"objects": ["mug"], "description": "pixels of target"},
                "affected": {"objects": ["table"], "description": "table area"},
                "keep": {"description": "background"},
            },
            {"width": 832, "height": 480, "frame_count": 81},
        )
        self.assertFalse(metrics["quadmask_schema_valid"])
        self.assertFalse(metrics["quadmask_spec_executable"])
        self.assertEqual(metrics["executor_failure"], "missing_primary_bbox")

    def test_relabel_and_rewrite_helpers_use_v6_prompt_schema(self) -> None:
        row = {
            "id": "unit",
            "video": "/tmp/unit.mp4",
            "messages": [
                {
                    "role": "user",
                    "content": 'Old prompt "quadmask_spec": {"primary": {}, "affected": {}, "keep": {}}. User request: remove the mug.',
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "video_id": "unit",
                            "task_type": "remove",
                            "edit_prompt": "remove the mug",
                            "target_objects": [{"name": "mug"}],
                            "protected_objects": ["turntable"],
                            "event_summary": "A mug sits on a turntable.",
                            "physical_causal_chain": [],
                            "counterfactual_expectation": {"if_removed": "The turntable remains visible."},
                            "quadmask_spec": {"primary": {}, "affected": {}, "keep": {}},
                            "quality_flags": {},
                        }
                    ),
                },
            ],
        }
        label = assistant_json(row)
        rewritten = relabel_grounding.rewrite_row_user_prompt_schema(row, label, "remove")
        prompt = rewritten["messages"][0]["content"]
        self.assertIn(PLANNER_IO_SCHEMA_VERSION, prompt)
        self.assertIn("bbox_xyxy_norm1000", prompt)
        self.assertNotIn('"quadmask_spec": {"primary": {}, "affected": {}, "keep": {}}', prompt)

        rewritten2, metrics = rewrite_prompt_schema.rewrite_row(row, "auto")
        self.assertEqual(rewritten2["metadata"]["planner_user_prompt_schema"], PLANNER_IO_SCHEMA_VERSION)
        self.assertTrue(metrics["old_prompt_had_empty_quadmask_schema"])
        self.assertFalse(metrics["new_prompt_had_empty_quadmask_schema"])


class V02ReportContractTests(unittest.TestCase):
    def test_first_frame_qc_separates_interface_from_visual_review(self) -> None:
        first_entry = {
            "status": "ok",
            "metrics": {
                "backend": "qwen_image_edit",
                "source_size": [832, 480],
                "edited_size": [832, 480],
                "raw_output_size": [832, 480],
                "target_mask_consumed_by_backend": False,
            },
        }
        qc = package_v02.build_first_frame_qc(first_entry)
        self.assertTrue(qc["system_interface_ok"])
        self.assertTrue(qc["qwen_interface_ok"])
        self.assertEqual(qc["qwen_visual_review_status"], "unreviewed")
        self.assertNotIn("usable_for_vace", qc)

        debug_entry = {"status": "ok", "metrics": {**first_entry["metrics"], "backend": "opencv_inpaint_debug"}}
        debug_qc = package_v02.build_first_frame_qc(debug_entry)
        self.assertTrue(debug_qc["system_interface_ok"])
        self.assertFalse(debug_qc["qwen_interface_ok"])

    def test_run_vace_nonzero_returncode_is_failed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_dir = Path(tmp)
            sample_dir = run_dir / "sample"
            sample_dir.mkdir(parents=True)
            image_path = sample_dir / "edited_first_frame.png"
            Image.new("RGB", (16, 16), (128, 128, 128)).save(image_path)
            prompt_path = sample_dir / "vace_prompt.txt"
            prompt_path.write_text("The edited scene remains physically plausible.\n", encoding="utf-8")
            quadmask_path = sample_dir / "quadmask.npy"
            np.save(quadmask_path, np.zeros((3, 16, 16), dtype=np.uint8))
            vace_repo = run_dir / "missing_vace_repo"
            vace_repo.mkdir()
            args = argparse.Namespace(
                run_dir=run_dir,
                mode="mode_test",
                frame_num=3,
                fps=16.0,
                size="480p",
                sample_steps=1,
                seed=1,
                python=Path(sys.executable),
                vace_repo=vace_repo,
                vace_ckpt=run_dir / "missing_ckpt",
                run_vace=True,
                force=True,
            )
            entry = run_vace_v0.process_one(
                args,
                "0001",
                {"paths": {"vace_prompt": str(prompt_path)}},
                {"status": "ok", "paths": {"quadmask_npy": str(quadmask_path)}},
                {"status": "ok", "paths": {"edited_first_frame": str(image_path)}},
            )
            self.assertEqual(entry["status"], "failed")
            self.assertEqual(entry["failure_source"], "vace")
            self.assertFalse(entry["metrics"]["vace_completed"])
            self.assertFalse(entry["metrics"]["prepared_only"])


class V02RunnerContractTests(unittest.TestCase):
    def test_cuda_visible_devices_is_explicit_stage_env(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            env = run_v02.build_stage_env("4")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "4")
        self.assertEqual(run_v02.stage_env_metadata(env), {"CUDA_VISIBLE_DEVICES": "4"})

    def test_canonical_runner_requires_vlm_planner_prediction_artifacts(self) -> None:
        self.assertEqual(run_v02.DEFAULT_INPUT.name, "vlm_planner_sft_eval_v6_teacher_grounded.jsonl")
        self.assertIn("raw_output", run_v02.CRITICAL_KEYS["planner"])
        self.assertIn("raw_pred", run_v02.CRITICAL_KEYS["planner"])
        self.assertNotIn("raw_teacher", run_v02.CRITICAL_KEYS["planner"])


if __name__ == "__main__":
    unittest.main()

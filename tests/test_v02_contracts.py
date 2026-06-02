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
    VacePromptContractError,
    infer_operation_from_text,
    normalize_to_e2w_contract,
    serialize_first_frame_prompt,
    serialize_vace_prompt,
    validate_edit_plan,
)
import package_v02_qwen_vace_smoke as package_v02  # noqa: E402
import physics_iq_vlm_pipeline  # noqa: E402
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
        metrics = validate_edit_plan(plan)
        self.assertTrue(metrics["operation_accuracy"])
        self.assertEqual(metrics["expected_operation"], "add")
        prompt = serialize_first_frame_prompt(plan).lower()
        self.assertIn("add the yellow mug", prompt)
        self.assertNotIn("remove only", prompt)
        self.assertNotIn("delete", prompt)
        vace_prompt = serialize_vace_prompt(plan).lower()
        self.assertIn("yellow mug", vace_prompt)
        self.assertIn("one yellow mug", vace_prompt)
        self.assertNotRegex(vace_prompt, r"\b(remove|delete|erase|removed)\b")


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


if __name__ == "__main__":
    unittest.main()

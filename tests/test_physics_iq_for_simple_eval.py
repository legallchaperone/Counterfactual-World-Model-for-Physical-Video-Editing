from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import physics_iq_for_simple_eval as bench  # noqa: E402


class PhysicsIqForSimpleEvalTests(unittest.TestCase):
    def _write_descriptions(self, root: Path) -> Path:
        path = root / "descriptions.csv"
        rows = []
        for spec in bench.BENCHMARK_SPECS:
            rows.append(
                {
                    "scenario": f"{spec.physics_iq_id}_perspective-center_take-1_trimmed-test.mp4",
                    "description": f"Description for {spec.physics_iq_id}",
                    "category": "Solid Mechanics",
                    "generated_video_name": f"{spec.physics_iq_id}_perspective-center_trimmed-test.mp4",
                }
            )
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["scenario", "description", "category", "generated_video_name"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _write_full_video_placeholders(self, root: Path) -> None:
        for spec in bench.BENCHMARK_SPECS:
            take = "take-2" if int(spec.physics_iq_id) >= 199 else "take-1"
            path = root / "full-videos" / take / "30FPS" / f"{spec.physics_iq_id}_perspective-center_take-1_trimmed-test.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"placeholder")

    def test_build_manifest_has_12_unleaked_rows_and_strict_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            descriptions = self._write_descriptions(root)
            self._write_full_video_placeholders(root)

            rows = bench.build_manifest(descriptions_csv=descriptions, physics_iq_root=root, out_root=root / "out")
            errors = bench.validate_manifest_rows(rows)

        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["physics_iq_id"] for row in rows}), 12)
        self.assertTrue(all(not row["leakage_exclusion_evidence"]["leaked"] for row in rows))
        self.assertTrue(all("/full-videos/" in row["source_full_video"] for row in rows))
        self.assertTrue(all("Answer only JSON" in row["vlm_judge_prompt"] for row in rows))
        self.assertTrue(all("Do not reward the video merely for being non-black" in row["vlm_judge_prompt"] for row in rows))

    def test_manifest_allows_official_split_testing_source(self) -> None:
        row = {
            "sample_id": "ok",
            "physics_iq_id": "0001",
            "operation": "remove",
            "user_prompt": "Remove the ball.",
            "target_object": "ball",
            "expected_visible_outcome": "Ball absent.",
            "expected_physical_effect": "No collision.",
            "must_preserve": ["table"],
            "source_full_video": "/data/cwx/physics-iq/physics-IQ-benchmark/split-videos/testing/30FPS/0001_testing-videos_30FPS_x.mp4",
            "leakage_exclusion_evidence": {"leaked": False},
            "vlm_judge_prompt": "Answer only JSON",
        }
        rows = [{**row, "sample_id": f"ok_{idx}", "physics_iq_id": f"{idx:04d}"} for idx in range(12)]
        errors = bench.validate_manifest_rows(rows)
        self.assertFalse(any("source_full_video is not an allowed official Physics-IQ source" in err for err in errors))

    def test_manifest_rejects_review_proxy_source(self) -> None:
        row = {
            "sample_id": "bad",
            "physics_iq_id": "0001",
            "operation": "remove",
            "user_prompt": "Remove the ball.",
            "target_object": "ball",
            "expected_visible_outcome": "Ball absent.",
            "expected_physical_effect": "No collision.",
            "must_preserve": ["table"],
            "source_full_video": "/data/cwx/E2W/data/physics_iq_vlm_sft/review_proxies_h264_720p/0001_testing-videos_16FPS_x.mp4",
            "leakage_exclusion_evidence": {"leaked": False},
            "vlm_judge_prompt": "Answer only JSON",
        }
        rows = [{**row, "sample_id": f"bad_{idx}", "physics_iq_id": f"{idx:04d}"} for idx in range(12)]
        errors = bench.validate_manifest_rows(rows)
        self.assertTrue(any("source_full_video is not an allowed official Physics-IQ source" in err for err in errors))

    def test_validate_vlm_judge_requires_exact_binary_schema(self) -> None:
        valid = dict(bench.VLM_JUDGE_SCHEMA)
        valid.update({"target_success": 1, "short_reason": "The object appears."})
        bench.validate_vlm_judge(valid)

        invalid = dict(valid)
        invalid["overall_pass"] = 2
        with self.assertRaises(bench.BenchmarkError):
            bench.validate_vlm_judge(invalid)

        invalid_extra = dict(valid)
        invalid_extra["score"] = 0.5
        with self.assertRaises(bench.BenchmarkError):
            bench.validate_vlm_judge(invalid_extra)

    def test_append_and_export_human_judgments_keeps_latest_per_sample(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            judgments = root / "human_judgments.jsonl"
            bench.append_human_judgment(judgments, {"sample_id": "s1", "human_e2w_overall_pass": 0})
            bench.append_human_judgment(judgments, {"sample_id": "s1", "human_e2w_overall_pass": 1})
            bench.export_human_summary(judgments, root / "summary.json", root / "summary.csv")
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["rows"][0]["human_e2w_overall_pass"], 1)

    def test_validate_run_contract_requires_current_vace_inputs_and_full_generation_mask(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [
                {
                    "sample_id": "piq_simple_eval_0037_remove",
                    "physics_iq_id": "0037",
                    "operation": "remove",
                    "user_prompt": "Remove the black balloon.",
                    "target_object": "black balloon",
                    "expected_visible_outcome": "Balloon absent.",
                    "expected_physical_effect": "No expansion.",
                    "must_preserve": ["table"],
                    "leakage_exclusion_evidence": {"leaked": False},
                    "vlm_judge_prompt": "Answer only JSON",
                }
            ]
            sample_dir = root / "piq_simple_eval_0037_remove"
            sample_dir.mkdir()
            (sample_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "vace_runtime_inputs": {
                            "vace_conditioning_video": "cond.mp4",
                            "quadmask_npy": "quad.npy",
                            "generation_mask": "generation_mask.npy",
                            "operation": "remove",
                            "vace_prompt": "The pump hose remains on the table.",
                            "frame_num": 21,
                        },
                        "source_video_passed_to_vace": False,
                        "vace_conditioning_video": {
                            "frame_0_is_edited_first_frame": True,
                            "future_frames_are_zero_filled": True,
                            "future_frames_source_video_used": False,
                        },
                        "generation_mask_values": [255],
                        "control_branch_checkpoint_loaded": True,
                        "trained_control_branch_used": True,
                        "control_branch_installed_in_forward_vace": True,
                        "control_branch_gate": 0.125,
                    }
                ),
                encoding="utf-8",
            )

            errors = bench.validate_run_contract(root, rows)

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import e2w_add_then_remove as pipe  # noqa: E402


def _six_inputs(operation: str, prompt: str) -> dict[str, object]:
    return {
        "vace_conditioning_video": "cond.mp4",
        "quadmask_npy": "quad.npy",
        "generation_mask": "generation.mp4",
        "operation": operation,
        "vace_prompt": prompt,
        "frame_num": 21,
    }


class AddThenRemovePipelineTests(unittest.TestCase):
    def _args(self, root: Path) -> SimpleNamespace:
        source = root / "source.mp4"
        source.write_bytes(b"source")
        checkpoint = root / "latest.pt"
        checkpoint.write_bytes(b"checkpoint")
        add_planner_adapter = root / "add-lora.pt"
        add_planner_adapter.write_bytes(b"add-lora")
        return SimpleNamespace(
            source_video=source,
            add_prompt="Add a red cube on the table.",
            sample_id="debug_add_remove_0001",
            run_dir=root / "run",
            control_branch_checkpoint=checkpoint,
            add_planner_adapter=add_planner_adapter,
            vace_sample_steps=8,
            cuda_visible_devices="5",
            python=Path("/python"),
        )

    def _fake_run_cmd_factory(self, commands: list[list[str]]):
        def fake_run_cmd(cmd, *, cwd, env, log_path):
            commands.append(cmd)
            run_dir = log_path.parent
            if "e2w_add.py" in cmd[1]:
                add_stage = run_dir / "add_stage"
                add_stage.mkdir(parents=True, exist_ok=True)
                (add_stage / "edited_video.mp4").write_bytes(b"add-video")
                pipe.write_json(
                    add_stage / "metadata.json",
                    {
                        "target_ref": "red cube",
                        "vace_prompt": "A red cube sits on the table.",
                        "edited_video": {"path": str(add_stage / "edited_video.mp4")},
                        "source_video_passed_to_vace": False,
                        "vace_conditioning_video": {
                            "future_frames_are_zero_filled": True,
                            "future_frames_source_video_used": False,
                        },
                        "control_branch_checkpoint_loaded": True,
                        "trained_control_branch_used": True,
                        "control_branch_installed_in_forward_vace": True,
                        "vace_runtime_inputs": _six_inputs("add", "A red cube sits on the table."),
                    },
                )
            elif "e2w_remove.py" in cmd[1]:
                remove_stage = run_dir / "remove_stage"
                remove_stage.mkdir(parents=True, exist_ok=True)
                edited = remove_stage / "edited_video_debug_add_remove_0001_remove_after_add.mp4"
                edited.write_bytes(b"remove-video")
                pipe.write_json(
                    remove_stage / "summary.json",
                    {
                        "results": [
                            {
                                "status": "ok",
                                "vace_output_path": str(edited),
                                "vace_prompt": "The tabletop remains visible and stable.",
                                "source_video_passed_to_vace": False,
                                "vace_runtime_inputs": _six_inputs("remove", "The tabletop remains visible and stable."),
                                "generation_mask_metadata": {"generation_mask_values": [255]},
                                "vace_info": {
                                    "control_branch_checkpoint_loaded": True,
                                    "trained_control_branch_used": True,
                                    "control_branch_installed_in_forward_vace": True,
                                    "vace_runtime_input_metadata": {
                                        "vace_conditioning_video": {
                                            "future_frames_are_zero_filled": True,
                                            "future_frames_source_video_used": False,
                                        },
                                        "generation_mask": {"generation_mask_values": [255]},
                                    },
                                },
                            }
                        ]
                    },
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        return fake_run_cmd

    def test_orchestrates_add_before_remove_and_records_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self._args(root)
            commands: list[list[str]] = []

            with (
                mock.patch.object(pipe, "run_cmd", side_effect=self._fake_run_cmd_factory(commands)),
                mock.patch.object(pipe, "extract_video_frames", return_value=[args.run_dir / "remove_input_frames" / "00000.jpg"]),
                mock.patch.object(pipe, "make_comparison_grid", side_effect=lambda _a, _b, _c, out: out.write_bytes(b"grid")),
            ):
                metadata = pipe.run_pipeline(args)

            self.assertEqual(len(commands), 2)
            self.assertIn("e2w_add.py", commands[0][1])
            self.assertIn("e2w_remove.py", commands[1][1])
            self.assertIn(str(args.run_dir / "add_stage" / "edited_video.mp4"), metadata["add_stage_edited_video"])
            self.assertEqual(metadata["operation_chain"], ["add", "remove"])
            self.assertEqual(metadata["add_target_ref"], "red cube")
            self.assertEqual(metadata["add_vace_prompt"], "A red cube sits on the table.")
            self.assertEqual(metadata["remove_vace_prompt"], "The tabletop remains visible and stable.")
            self.assertTrue(metadata["contract_checks"]["add"]["trained_control_branch_used"])
            self.assertTrue(metadata["contract_checks"]["remove"]["trained_control_branch_used"])

            eval_rows = [
                json.loads(line)
                for line in (args.run_dir / "remove_eval.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            remove_text = eval_rows[0]["messages"][0]["content"][1]["text"]
            image_path = eval_rows[0]["messages"][0]["content"][0]["image"]
            self.assertIn("newly added red cube", remove_text)
            self.assertIn("remove_input_frames/00000.jpg", image_path)

    def test_remove_stage_receives_add_edited_video_as_frame_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self._args(root)
            commands: list[list[str]] = []
            extracted_from: list[Path] = []

            def fake_extract(video_path: Path, frame_dir: Path) -> list[Path]:
                extracted_from.append(video_path)
                return [frame_dir / "00000.jpg"]

            with (
                mock.patch.object(pipe, "run_cmd", side_effect=self._fake_run_cmd_factory(commands)),
                mock.patch.object(pipe, "extract_video_frames", side_effect=fake_extract),
                mock.patch.object(pipe, "make_comparison_grid", side_effect=lambda _a, _b, _c, out: out.write_bytes(b"grid")),
            ):
                pipe.run_pipeline(args)

            self.assertEqual(extracted_from, [(args.run_dir / "add_stage" / "edited_video.mp4").resolve()])
            self.assertNotEqual(extracted_from[0], args.source_video.resolve())

    def test_missing_add_target_ref_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self._args(root)

            def fake_run_cmd(cmd, *, cwd, env, log_path):
                add_stage = log_path.parent / "add_stage"
                add_stage.mkdir(parents=True, exist_ok=True)
                (add_stage / "edited_video.mp4").write_bytes(b"add-video")
                pipe.write_json(
                    add_stage / "metadata.json",
                    {
                        "target_ref": "",
                        "edited_video": {"path": str(add_stage / "edited_video.mp4")},
                        "source_video_passed_to_vace": False,
                        "vace_conditioning_video": {
                            "future_frames_are_zero_filled": True,
                            "future_frames_source_video_used": False,
                        },
                        "control_branch_checkpoint_loaded": True,
                        "trained_control_branch_used": True,
                        "control_branch_installed_in_forward_vace": True,
                        "vace_runtime_inputs": _six_inputs("add", "A red cube sits on the table."),
                    },
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(pipe, "run_cmd", side_effect=fake_run_cmd):
                with self.assertRaisesRegex(pipe.PipelineError, "target_ref"):
                    pipe.run_pipeline(args)


if __name__ == "__main__":
    unittest.main()

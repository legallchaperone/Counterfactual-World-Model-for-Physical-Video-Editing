from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import e2w_vace_quad_i2v as quad_i2v  # noqa: E402
import run_vace_v03_quad_experiment as v03  # noqa: E402


class V03QuadVaceContractTests(unittest.TestCase):
    def test_quadmask_loader_rejects_raw_values_before_uint8_cast(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "bad_quadmask.npy"
            np.save(path, np.array([[[256]]], dtype=np.int16))
            with self.assertRaisesRegex(quad_i2v.VACEQuadI2VError, "256"):
                quad_i2v.load_quadmask_npy(path)

    def test_quadmask_context_uses_count_key_not_semantic_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "quadmask.npy"
            np.save(path, np.array([[[0, 63], [127, 255]]], dtype=np.uint8))
            import torch

            class FakeWan:
                def vace_latent(self) -> None:
                    return None

            info = quad_i2v.install_quad_latent_hook(
                FakeWan(),
                quadmask_npy=path,
                src_mask=[torch.zeros((1, 1, 2, 2), dtype=torch.float32)],
                src_ref_images=[None],
            )
            self.assertEqual(info["quadmask_value_counts"], {"0": 1, "63": 1, "127": 1, "255": 1})
            self.assertNotIn("quadmask_values", info)

    def test_quadmask_alignment_requires_explicit_mode(self) -> None:
        quad = np.zeros((2, 4, 6), dtype=np.uint8)
        meta = {"frame_count": 3, "height": 2, "width": 3}
        with self.assertRaisesRegex(RuntimeError, "pass --align-quadmask nearest"):
            v03.align_quadmask(quad, meta, "error")

        aligned, info = v03.align_quadmask(quad, meta, "nearest")
        self.assertEqual(aligned.shape, (3, 2, 3))
        self.assertEqual(info["method"], "nearest")
        self.assertTrue(info["changed"])
        self.assertEqual(info["frame_index_mapping_first_last"], [0, 1])

    def test_generation_mask_default_mode_is_full_domain(self) -> None:
        quad = np.array(
            [
                [[255, 0], [63, 127]],
                [[255, 255], [0, 127]],
            ],
            dtype=np.uint8,
        )
        mask = v03.generation_mask_from_quadmask(quad, "full-domain")
        self.assertEqual(mask.shape, quad.shape)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(sorted(int(x) for x in np.unique(mask)), [255])

    def test_quad_runner_command_contains_required_v03_inputs(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            args = argparse.Namespace(
                python=Path("/env/python"),
                vace_repo=tmp_path / "VACE",
                vace_ckpt=tmp_path / "ckpt",
                model_name="vace-14B",
                size="480p",
                src_video=tmp_path / "source.mp4",
                operation="add",
                prompt="Add a yellow mug on the rotating turntable in front of the spotlight",
                run_dir=tmp_path / "run",
                base_seed=2025,
                sample_steps=8,
                sample_shift=16.0,
                sample_guide_scale=5.0,
                context_scale=1.0,
                low_mem=True,
            )
            args.vace_repo.mkdir()
            args.vace_ckpt.mkdir()
            args.src_video.write_bytes(b"placeholder")
            quad = tmp_path / "quadmask.npy"
            gen = tmp_path / "generation_mask.mp4"
            out = tmp_path / "edited_video.mp4"
            quad.write_bytes(b"quad")
            gen.write_bytes(b"mask")
            with mock.patch.object(v03, "video_meta", return_value={"frame_count": 81, "height": 480, "width": 832, "fps": 16.0}):
                command = v03.build_command(args, quad, gen, out)

        self.assertIn("--quadmask_npy", command)
        self.assertIn(str(quad.resolve()), command)
        self.assertIn("--operation", command)
        self.assertIn("add", command)
        self.assertIn("--generation_mask", command)
        self.assertIn(str(gen.resolve()), command)
        self.assertIn("--context_scale", command)

    def test_prepared_run_does_not_claim_backend_consumption(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            src_video = tmp_path / "source.mp4"
            src_video.write_bytes(b"placeholder")
            quad = tmp_path / "quadmask.npy"
            np.save(quad, np.full((1, 2, 2), 255, dtype=np.uint8))
            argv = [
                "run_vace_v03_quad_experiment.py",
                "--src-video",
                str(src_video),
                "--prompt",
                "Add a yellow mug on the rotating turntable in front of the spotlight",
                "--quadmask-npy",
                str(quad),
                "--operation",
                "add",
                "--run-dir",
                str(run_dir),
            ]
            meta = {"frame_count": 1, "height": 2, "width": 2, "fps": 16.0}
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(v03, "video_meta", return_value=meta),
            ):
                v03.main()

            payload = __import__("json").loads((run_dir / "experiment_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "prepared")
            self.assertTrue(payload["quadmask_passed_to_backend_command"])
            self.assertFalse(payload["quadmask_consumed_by_backend"])
            self.assertEqual(payload["generation_mask_mode"], "full-domain")
            self.assertEqual(payload["generation_mask_values"], [255])
            self.assertTrue(payload["generation_mask_is_full_domain"])
            self.assertFalse(payload["generation_mask_encodes_quadmask_semantics"])


if __name__ == "__main__":
    unittest.main()

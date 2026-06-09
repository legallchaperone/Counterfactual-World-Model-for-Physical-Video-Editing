from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import run_counterfactual_planner_pipeline as bridge  # noqa: E402
from run_counterfactual_planner_pipeline import build_quadmask  # noqa: E402


class CounterfactualPlannerBridgeTests(unittest.TestCase):
    def test_build_quadmask_preserves_primary_as_q0(self) -> None:
        primary = np.zeros((12, 14), dtype=bool)
        primary[4:7, 5:9] = True

        quadmask, generation_mask = build_quadmask({0: primary}, frame_count=1, height=12, width=14)

        values = set(np.unique(quadmask).astype(int).tolist())
        self.assertTrue(values.issubset({0, 63, 127, 255}))
        self.assertIn(0, values)
        self.assertIn(127, values)
        self.assertTrue(np.all(quadmask[0][primary] == 0))
        self.assertEqual(sorted(int(x) for x in np.unique(generation_mask)), [255])
        self.assertEqual(generation_mask.dtype, np.uint8)

    def test_first_frame_edit_uses_model_cpu_offload_when_available(self) -> None:
        calls: list[str] = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, _checkpoint: str, torch_dtype: object) -> "FakePipeline":
                calls.append(f"dtype={torch_dtype}")
                return cls()

            def enable_model_cpu_offload(self) -> None:
                calls.append("offload")

            def to(self, device: str) -> "FakePipeline":
                calls.append(f"to={device}")
                return self

            def __call__(self, *_args: object, **_kwargs: object) -> object:
                return types.SimpleNamespace(images=[Image.new("RGB", (4, 4), "white")])

        fake_diffusers = types.SimpleNamespace(QwenImageEditPipeline=FakePipeline)
        fake_generator = mock.Mock()
        fake_generator.manual_seed.return_value = "generator"

        with tempfile.TemporaryDirectory(dir="/tmp") as td:
            root = Path(td)
            anchor = root / "anchor.jpg"
            Image.new("RGB", (4, 4), "black").save(anchor)
            args = types.SimpleNamespace(
                skip_first_frame_edit=False,
                qwen_image_edit_checkpoint=root / "fake-qwen",
                output_dir=root,
                qwen_image_edit_seed=1,
                qwen_image_edit_steps=1,
                qwen_image_edit_true_cfg_scale=1.0,
            )
            with (
                mock.patch.dict(sys.modules, {"diffusers": fake_diffusers}),
                mock.patch.object(bridge.torch.cuda, "is_available", return_value=True),
                mock.patch.object(bridge.torch.cuda, "is_bf16_supported", return_value=True),
                mock.patch.object(bridge.torch, "Generator", return_value=fake_generator),
                mock.patch.object(bridge.torch.cuda, "empty_cache"),
            ):
                ok, edited_frame, info = bridge.run_first_frame_edit(args, anchor, "sink", "sample")

        self.assertTrue(ok)
        self.assertIsNotNone(edited_frame)
        self.assertEqual(info["edited_size"], [4, 4])
        self.assertIn("offload", calls)
        self.assertNotIn("to=cuda", calls)


if __name__ == "__main__":
    unittest.main()

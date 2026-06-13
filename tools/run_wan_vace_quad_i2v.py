#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for path in (TOOLS, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from e2w_vace_quad_i2v import (  # noqa: E402
    generate_with_quad_context,
    install_quad_latent_hook,
    install_quad_vace_controls,
    install_trained_control_branch,
    operation_to_id,
    write_context_debug,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E2W quadmask-conditioned Wan2.1-VACE I2V.")
    parser.add_argument("--vace_repo", required=True, help="Path to ali-vilab/VACE repo.")
    parser.add_argument("--ckpt_dir", required=True, help="Wan2.1-VACE checkpoint directory.")
    parser.add_argument("--model_name", default="vace-14B", choices=["vace-14B", "vace-1.3B"])
    parser.add_argument("--size", default="480p")
    parser.add_argument("--src_video", required=True, help="Conditioning video: frame 0 edited frame, future black frames.")
    parser.add_argument("--generation_mask", required=True, help="Binary G mask video: frame 0 known, future frames generate.")
    parser.add_argument("--quadmask_npy", required=True, help="Formal VOID-style quadmask.npy with values {0,63,127,255}.")
    parser.add_argument("--operation", required=True, choices=["remove", "add"])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--save_file", required=True)
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_shift", type=float, default=16.0)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--sample_solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--context_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--offload_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--t5_cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--control_branch_checkpoint", type=Path)
    return parser.parse_args()


def add_vace_paths(vace_repo: Path) -> None:
    for path in (vace_repo / "vace", vace_repo):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    vace_repo = Path(args.vace_repo).expanduser().resolve()
    add_vace_paths(vace_repo)

    import torch  # noqa: E402
    from wan.utils.utils import cache_video  # noqa: E402
    from models.wan import WanVace  # noqa: E402
    from models.wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS  # noqa: E402

    if args.model_name not in WAN_CONFIGS:
        raise ValueError(f"unsupported model_name: {args.model_name}")
    if args.size not in SUPPORTED_SIZES[args.model_name]:
        raise ValueError(f"unsupported size {args.size!r} for {args.model_name}")
    if args.frame_num % 4 != 1:
        raise ValueError(f"Wan2.1-VACE frame_num must be 4n+1, got {args.frame_num}")
    if not torch.cuda.is_available():
        raise RuntimeError("Wan2.1-VACE requires CUDA for this runner")

    device_id = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    cfg = WAN_CONFIGS[args.model_name]
    logging.info("Loading WanVace model.")
    wan_vace = WanVace(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=args.t5_cpu,
    )
    control_branch_info = {
        "control_branch_checkpoint": None,
        "control_branch_checkpoint_loaded": False,
        "trained_control_branch_used": False,
        "control_branch_installed_in_forward_vace": False,
    }
    if args.control_branch_checkpoint:
        control_branch_info = install_trained_control_branch(
            wan_vace.model,
            checkpoint_path=args.control_branch_checkpoint,
            operation=args.operation,
        )
    else:
        install_quad_vace_controls(wan_vace.model, args.operation)

    logging.info("Preparing source conditioning video, generation mask, and quadmask context.")
    src_video, src_mask, src_ref_images = wan_vace.prepare_source(
        [args.src_video],
        [args.generation_mask],
        [None],
        args.frame_num,
        SIZE_CONFIGS[args.size],
        device,
    )
    context_info = install_quad_latent_hook(
        wan_vace,
        quadmask_npy=args.quadmask_npy,
        src_mask=src_mask,
        src_ref_images=src_ref_images,
    )
    context_info.update(
        {
            "version": "e2w.vace_quad_i2v.v1",
            "operation": args.operation,
            "operation_id": operation_to_id(args.operation),
            "generation_mask_semantics": {"0": "known_or_condition", "1": "generate"},
            "quadmask_semantics": {"0": "primary", "63": "primary_affected_overlap", "127": "affected", "255": "keep"},
            "src_video": args.src_video,
            "generation_mask": args.generation_mask,
            "quadmask_npy": args.quadmask_npy,
            **control_branch_info,
        }
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    write_context_debug(save_dir / "e2w_quad_context.json", context_info)

    logging.info("Generating edited video with quadmask-conditioned VACE context.")
    video = generate_with_quad_context(
        wan_vace,
        prompt=args.prompt,
        input_frames=src_video,
        input_masks=src_mask,
        input_ref_images=src_ref_images,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        context_scale=args.context_scale,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        negative_prompt=args.negative_prompt,
        seed=args.base_seed,
        offload_model=args.offload_model,
    )
    if video is None:
        raise RuntimeError("WanVace returned no video")

    cache_video(
        tensor=video[None],
        save_file=args.save_file,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    cache_video(
        tensor=src_video[0][None],
        save_file=str(save_dir / "src_video.mp4"),
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    cache_video(
        tensor=src_mask[0][None],
        save_file=str(save_dir / "src_generation_mask.mp4"),
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(0, 1),
    )
    logging.info("Saved edited video to %s", args.save_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

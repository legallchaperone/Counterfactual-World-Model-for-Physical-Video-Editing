from __future__ import annotations

import json
import math
import random
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


E2W_QUAD_VACE_IN_DIM = 416
LEGACY_VACE_IN_DIM = 96
VOID_VALUES = (0, 63, 127, 255)
Operation = Literal["remove", "add"]


class VACEQuadI2VError(RuntimeError):
    pass


def operation_to_id(operation: Operation | str) -> int:
    if operation == "remove":
        return 0
    if operation == "add":
        return 1
    raise VACEQuadI2VError(f"unsupported operation: {operation!r}")


def build_generation_mask_array(frame_count: int, height: int, width: int) -> np.ndarray:
    if frame_count <= 0 or height <= 0 or width <= 0:
        raise VACEQuadI2VError(f"invalid video shape: frames={frame_count}, size={width}x{height}")
    mask = np.full((frame_count, height, width), 255, dtype=np.uint8)
    mask[0] = 0
    return mask


def write_generation_mask_video(output_path: str | Path, *, frame_count: int, height: int, width: int, fps: float) -> Path:
    import imageio.v2 as imageio

    mask = build_generation_mask_array(frame_count=frame_count, height=height, width=width)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [np.repeat(frame[:, :, None], 3, axis=2) for frame in mask]
    imageio.mimwrite(str(output_path), frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    return output_path


def load_quadmask_npy(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise VACEQuadI2VError(f"quadmask npy does not exist: {path}")
    arr = np.load(path)
    if arr.ndim != 3:
        raise VACEQuadI2VError(f"quadmask must have shape [T,H,W], got {arr.shape}: {path}")
    values = np.unique(arr)
    bad = values[~np.isin(values, np.asarray(VOID_VALUES, dtype=values.dtype))]
    if bad.size:
        raise VACEQuadI2VError(f"quadmask values must be {VOID_VALUES}, got {bad[:20].tolist()}: {path}")
    return arr.astype(np.uint8, copy=False)


def expand_vace_patch_embedding(model: nn.Module, *, new_in_dim: int = E2W_QUAD_VACE_IN_DIM) -> None:
    old_conv = getattr(model, "vace_patch_embedding", None)
    if not isinstance(old_conv, nn.Conv3d):
        raise VACEQuadI2VError("model.vace_patch_embedding must be nn.Conv3d")
    old_in = int(old_conv.in_channels)
    if old_in == new_in_dim:
        setattr(model, "vace_in_dim", new_in_dim)
        return
    if old_in != LEGACY_VACE_IN_DIM:
        raise VACEQuadI2VError(f"expected legacy VACE input dim {LEGACY_VACE_IN_DIM}, got {old_in}")
    if new_in_dim <= old_in:
        raise VACEQuadI2VError(f"new_in_dim must be greater than {old_in}, got {new_in_dim}")

    new_conv = nn.Conv3d(
        in_channels=new_in_dim,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :old_in].copy_(old_conv.weight)
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.vace_patch_embedding = new_conv
    setattr(model, "vace_in_dim", new_in_dim)
    config = getattr(model, "config", None)
    if config is not None:
        try:
            setattr(config, "vace_in_dim", new_in_dim)
        except Exception:
            pass


def install_operation_embedding(model: nn.Module, operation: Operation | str) -> None:
    operation_id = operation_to_id(operation)
    dim = int(getattr(model, "dim"))
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if not hasattr(model, "e2w_operation_embedding"):
        embedding = nn.Embedding(2, dim, device=device, dtype=dtype)
        nn.init.zeros_(embedding.weight)
        model.e2w_operation_embedding = embedding
    setattr(model, "e2w_operation_id", operation_id)
    model.forward_vace = types.MethodType(_forward_vace_with_operation, model)


def install_quad_vace_controls(model: nn.Module, operation: Operation | str) -> None:
    expand_vace_patch_embedding(model)
    install_operation_embedding(model, operation)


def _forward_vace_with_operation(self: nn.Module, x: Any, vace_context: list[torch.Tensor], seq_len: int, kwargs: dict[str, Any]):
    c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
    c = [u.flatten(2).transpose(1, 2) for u in c]
    c = torch.cat(
        [
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in c
        ]
    )

    operation_id = getattr(self, "e2w_operation_id", None)
    if operation_id is not None:
        op = torch.full((c.size(0),), int(operation_id), dtype=torch.long, device=c.device)
        op_embedding = self.e2w_operation_embedding(op).to(dtype=c.dtype)
        c = c + op_embedding[:, None, :]

    new_kwargs = dict(x=x)
    new_kwargs.update(kwargs)
    for block in self.vace_blocks:
        c = block(c, **new_kwargs)
    return torch.unbind(c)[:-1]


def install_quad_latent_hook(
    wan_vace: Any,
    *,
    quadmask_npy: str | Path,
    src_mask: list[torch.Tensor],
    src_ref_images: list[Any],
) -> dict[str, Any]:
    if not src_mask or src_mask[0] is None:
        raise VACEQuadI2VError("generation mask is required for quad VACE context")
    base_mask = src_mask[0]
    if base_mask.ndim != 4 or base_mask.shape[0] != 1:
        raise VACEQuadI2VError(f"generation mask must have shape [1,T,H,W], got {tuple(base_mask.shape)}")
    quad = _quadmask_tensor_for_source(quadmask_npy, base_mask)
    q0 = (quad == 0).to(base_mask.dtype)
    q1 = (quad == 63).to(base_mask.dtype)
    q2 = (quad == 127).to(base_mask.dtype)
    q3 = (quad == 255).to(base_mask.dtype)
    editable = (q0 + q1 + q2).clamp_(0, 1)

    extras = [
        [editable],
        [q0],
        [q1],
        [q2],
        [q3],
    ]
    wan_vace.e2w_quad_extra_masks = extras
    wan_vace.e2w_quad_ref_images = src_ref_images
    wan_vace.e2w_original_vace_latent = getattr(wan_vace, "vace_latent")
    wan_vace.vace_latent = types.MethodType(_vace_latent_with_quadmask, wan_vace)

    counts = {str(value): int((quad == value).sum().item()) for value in VOID_VALUES}
    return {
        "quadmask_shape": list(quad.shape),
        "quadmask_value_counts": counts,
        "context_channels": E2W_QUAD_VACE_IN_DIM,
        "semantic_channels": ["G", "E", "Q0_primary", "Q1_overlap", "Q2_affected", "Q3_keep"],
    }


def _quadmask_tensor_for_source(quadmask_npy: str | Path, base_mask: torch.Tensor) -> torch.Tensor:
    arr = load_quadmask_npy(quadmask_npy)
    device = base_mask.device
    target_t, target_h, target_w = [int(v) for v in base_mask.shape[1:]]
    tensor = torch.from_numpy(arr.astype(np.float32, copy=False)).to(device=device)
    tensor = tensor[None, None]
    if tuple(tensor.shape[-3:]) != (target_t, target_h, target_w):
        tensor = F.interpolate(tensor, size=(target_t, target_h, target_w), mode="nearest-exact")
    tensor = tensor[0, 0].round().to(torch.uint8)
    values = torch.unique(tensor).detach().cpu().numpy().astype(np.uint8)
    bad = values[~np.isin(values, np.asarray(VOID_VALUES, dtype=np.uint8))]
    if bad.size:
        raise VACEQuadI2VError(f"resized quadmask has invalid values: {bad[:20].tolist()}")
    return tensor[None].to(device=device)


def _vace_latent_with_quadmask(self: Any, z: list[torch.Tensor], m: list[torch.Tensor]) -> list[torch.Tensor]:
    extra_packed = [
        self.vace_encode_masks(mask_list, self.e2w_quad_ref_images)
        for mask_list in self.e2w_quad_extra_masks
    ]
    out: list[torch.Tensor] = []
    for idx, (z_item, g_pack) in enumerate(zip(z, m)):
        channels = [z_item, g_pack]
        channels.extend(packed[idx] for packed in extra_packed)
        context = torch.cat(channels, dim=0)
        if context.shape[0] != E2W_QUAD_VACE_IN_DIM:
            raise VACEQuadI2VError(
                f"quad VACE context must have {E2W_QUAD_VACE_IN_DIM} channels, got {context.shape[0]}"
            )
        out.append(context)
    return out


def generate_with_quad_context(
    wan_vace: Any,
    *,
    prompt: str,
    input_frames: list[torch.Tensor],
    input_masks: list[torch.Tensor],
    input_ref_images: list[Any],
    size: tuple[int, int],
    frame_num: int,
    context_scale: float = 1.0,
    shift: float = 16.0,
    sample_solver: str = "unipc",
    sampling_steps: int = 50,
    guide_scale: float = 5.0,
    negative_prompt: str = "",
    seed: int = -1,
    offload_model: bool = True,
) -> torch.Tensor | None:
    # This mirrors upstream WanVace.generate, but keeps the quadmask latent hook active.
    if negative_prompt == "":
        negative_prompt = wan_vace.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=wan_vace.device)
    seed_g.manual_seed(seed)

    if not wan_vace.t5_cpu:
        wan_vace.text_encoder.model.to(wan_vace.device)
        context = wan_vace.text_encoder([prompt], wan_vace.device)
        context_null = wan_vace.text_encoder([negative_prompt], wan_vace.device)
        if offload_model:
            wan_vace.text_encoder.model.cpu()
    else:
        context = wan_vace.text_encoder([prompt], torch.device("cpu"))
        context_null = wan_vace.text_encoder([negative_prompt], torch.device("cpu"))
        context = [item.to(wan_vace.device) for item in context]
        context_null = [item.to(wan_vace.device) for item in context_null]

    z0 = wan_vace.vace_encode_frames(input_frames, input_ref_images, masks=input_masks)
    g0 = wan_vace.vace_encode_masks(input_masks, input_ref_images)
    z = wan_vace.vace_latent(z0, g0)

    target_shape = list(z0[0].shape)
    target_shape[0] = int(target_shape[0] / 2)
    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=wan_vace.device,
            generator=seed_g,
        )
    ]
    seq_len = math.ceil(
        (target_shape[2] * target_shape[3])
        / (wan_vace.patch_size[1] * wan_vace.patch_size[2])
        * target_shape[1]
        / wan_vace.sp_size
    ) * wan_vace.sp_size

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(wan_vace.model, "no_sync", noop_no_sync)

    with amp.autocast(dtype=wan_vace.param_dtype), torch.no_grad(), no_sync():
        if sample_solver == "unipc":
            from wan.text2video import FlowUniPCMultistepScheduler

            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=wan_vace.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(sampling_steps, device=wan_vace.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            from wan.text2video import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps

            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=wan_vace.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=wan_vace.device, sigmas=sampling_sigmas)
        else:
            raise VACEQuadI2VError(f"unsupported sample_solver: {sample_solver!r}")

        latents = noise
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        for t in tqdm(timesteps):
            timestep = torch.stack([t])
            wan_vace.model.to(wan_vace.device)
            noise_pred_cond = wan_vace.model(
                latents,
                t=timestep,
                vace_context=z,
                vace_context_scale=context_scale,
                **arg_c,
            )[0]
            noise_pred_uncond = wan_vace.model(
                latents,
                t=timestep,
                vace_context=z,
                vace_context_scale=context_scale,
                **arg_null,
            )[0]
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latents = [temp_x0.squeeze(0)]

        x0 = latents
        if offload_model:
            wan_vace.model.cpu()
            torch.cuda.empty_cache()
        videos = wan_vace.decode_latent(x0, input_ref_images) if wan_vace.rank == 0 else None

    del noise, latents, sample_scheduler
    if offload_model and torch.cuda.is_available():
        torch.cuda.synchronize()
    return videos[0] if wan_vace.rank == 0 else None


def write_context_debug(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

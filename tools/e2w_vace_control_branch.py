#!/usr/bin/env python3
"""E2W Phase 1 v0.4 gated control branch for Wan-VACE quadmask training.

Phase 1 keeps the legacy 96-channel VACE context path frozen and routes the
extra v0.3 packed quadmask/control channels through a trainable residual branch.
The 416-channel v0.3 requirement applies to the raw pre-adapter packed context;
the trainable residual is produced in embedded context space and must match the
legacy patch-embedding output shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

LEGACY_VACE_CONTEXT_CHANNELS = 96
V03_PACKED_CONTEXT_CHANNELS = 416
V03_EXTRA_QUAD_CONTEXT_CHANNELS = V03_PACKED_CONTEXT_CHANNELS - LEGACY_VACE_CONTEXT_CHANNELS
SEMANTIC_MASK_NAMES = ("editable", "q0", "q1", "q2", "q3")
ALLOWED_QUADMASK_VALUES = (0, 63, 127, 255)
GENERATION_MASK_MODE_QUADMASK_EDITABLE = "quadmask-editable"


class ControlBranchContractError(ValueError):
    """Raised when the Phase 1 control branch contract is violated."""


class QuadmaskContractError(ValueError):
    """Raised when a quadmask violates the v0.4 control-branch contract."""


def validate_operation(operation: str) -> str:
    op = str(operation).strip().lower()
    if op not in {"remove", "add"}:
        raise ControlBranchContractError(f"unsupported operation value {operation!r}; expected remove/add")
    return op


def validate_quadmask_values(arr: np.ndarray) -> None:
    values = np.unique(arr)
    bad = [int(value) for value in values if int(value) not in ALLOWED_QUADMASK_VALUES]
    if bad:
        raise QuadmaskContractError(f"quadmask values must be {ALLOWED_QUADMASK_VALUES}, got {bad[:20]}")


def quadmask_semantic_masks(quadmask: np.ndarray) -> dict[str, np.ndarray]:
    validate_quadmask_values(quadmask)
    q0 = quadmask == 0
    q1 = quadmask == 63
    q2 = quadmask == 127
    q3 = quadmask == 255
    return {
        "editable": q0 | q1 | q2,
        "q0": q0,
        "q1": q1,
        "q2": q2,
        "q3": q3,
    }


def build_generation_mask_from_quadmask(quadmask: np.ndarray, mode: str = GENERATION_MASK_MODE_QUADMASK_EDITABLE) -> np.ndarray:
    del mode
    validate_quadmask_values(quadmask)
    return np.full_like(quadmask, 255, dtype=np.uint8)


def _first_conv3d(module: nn.Module) -> nn.Conv3d | None:
    if isinstance(module, nn.Conv3d):
        return module
    for child in module.modules():
        if isinstance(child, nn.Conv3d):
            return child
    return None


def freeze_module(module: nn.Module) -> nn.Module:
    """Set all parameters in ``module`` to ``requires_grad=False``."""

    for param in module.parameters():
        param.requires_grad_(False)
    return module


def grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().pow(2).sum().item())
    return float(total**0.5)


def split_packed_context(
    packed_context: torch.Tensor,
    legacy_channels: int = LEGACY_VACE_CONTEXT_CHANNELS,
    packed_channels: int = V03_PACKED_CONTEXT_CHANNELS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split raw v0.3-compatible packed context into legacy and E2W branches."""

    if packed_context.ndim != 5:
        raise ControlBranchContractError(
            f"packed_context must be [B,C,T,H,W], got shape={list(packed_context.shape)}"
        )
    if int(packed_context.shape[1]) != int(packed_channels):
        raise ControlBranchContractError(
            f"packed_context must have {packed_channels} raw channels, got {int(packed_context.shape[1])}"
        )
    return packed_context[:, :legacy_channels], packed_context[:, legacy_channels:packed_channels]


def _validate_quadmask_tensor_values(quadmask: torch.Tensor) -> None:
    arr = quadmask.detach().cpu().numpy()
    if arr.ndim == 3:
        validate_quadmask_values(arr)
        return
    if arr.ndim == 4:
        for item in arr:
            validate_quadmask_values(item)
        return
    raise ControlBranchContractError(f"quadmask must be [T,H,W] or [B,T,H,W], got {list(arr.shape)}")


def semantic_masks_from_quadmask_tensor(quadmask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return bool semantic masks for tensor quadmasks without semantic collapse."""

    if quadmask.ndim not in (3, 4):
        raise ControlBranchContractError(f"quadmask must be [T,H,W] or [B,T,H,W], got {list(quadmask.shape)}")
    _validate_quadmask_tensor_values(quadmask)
    return {
        "editable": torch.isin(quadmask, torch.tensor([0, 63, 127], device=quadmask.device, dtype=quadmask.dtype)),
        "q0": quadmask == 0,
        "q1": quadmask == 63,
        "q2": quadmask == 127,
        "q3": quadmask == 255,
    }


def build_v03_compatible_packed_context(
    legacy_context_input: torch.Tensor,
    quadmask: torch.Tensor,
    *,
    extra_channels: int = V03_EXTRA_QUAD_CONTEXT_CHANNELS,
) -> torch.Tensor:
    """Construct a v0.3-compatible 416-channel raw packed context for tests/training.

    The legacy 96 channels are passed through unchanged.  The extra 320 channels
    reserve space for E2W semantic control by repeating the five semantic masks
    (editable, Q0, Q1, Q2, Q3) across the existing VACE mask-packing surface.
    """

    if legacy_context_input.ndim != 5:
        raise ControlBranchContractError("legacy_context_input must be [B,96,T,H,W]")
    b, c, t, h, w = legacy_context_input.shape
    if int(c) != LEGACY_VACE_CONTEXT_CHANNELS:
        raise ControlBranchContractError(f"legacy_context_input must have 96 channels, got {int(c)}")
    if quadmask.ndim == 3:
        quadmask_b = quadmask.unsqueeze(0).expand(b, -1, -1, -1)
    elif quadmask.ndim == 4:
        quadmask_b = quadmask
    else:
        raise ControlBranchContractError(f"quadmask must be [T,H,W] or [B,T,H,W], got {list(quadmask.shape)}")
    if list(quadmask_b.shape) != [b, t, h, w]:
        raise ControlBranchContractError(
            f"quadmask shape {list(quadmask_b.shape)} must align with legacy input {[b, t, h, w]}"
        )
    masks = semantic_masks_from_quadmask_tensor(quadmask_b)
    ordered = [masks[name].to(dtype=legacy_context_input.dtype) for name in SEMANTIC_MASK_NAMES]
    channels: list[torch.Tensor] = []
    for idx in range(int(extra_channels)):
        channels.append(ordered[idx % len(ordered)].unsqueeze(1))
    quad_extra = torch.cat(channels, dim=1).to(device=legacy_context_input.device)
    return torch.cat([legacy_context_input, quad_extra], dim=1)


@dataclass(frozen=True)
class ControlBranchInitMetadata:
    control_branch_init: str = "Option B"
    quad_adapter_init: str = "small_random"
    residual_gate_init: float = 1.0e-3
    operation_embedding_init: str = "small_random"

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_branch_init": self.control_branch_init,
            "quad_adapter_init": self.quad_adapter_init,
            "residual_gate_init": self.residual_gate_init,
            "operation_embedding_init": self.operation_embedding_init,
        }


class E2WGatedCausalControlBranch(nn.Module):
    """Trainable E2W residual/control branch with a frozen legacy context path.

    Parameters in ``legacy_patch_embedding`` are frozen in-place.  Forward returns
    the trainable quad/control residual in embedded context space.  Use
    :meth:`compose_with_legacy` to add it to the frozen legacy embedding with the
    trainable residual gate.
    """

    def __init__(
        self,
        legacy_patch_embedding: nn.Module,
        quad_input_channels: int = V03_EXTRA_QUAD_CONTEXT_CHANNELS,
        operation_vocab: Sequence[str] = ("remove", "add"),
        reserve_cycle_fields: bool = True,
        operation_embedding_dim: int = 8,
        residual_gate_init: float = 1.0e-3,
        init_std: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if int(quad_input_channels) <= 0:
            raise ControlBranchContractError("quad_input_channels must be positive")
        self.legacy_patch_embedding = freeze_module(legacy_patch_embedding)
        legacy_conv = _first_conv3d(legacy_patch_embedding)
        if legacy_conv is None:
            raise ControlBranchContractError("legacy_patch_embedding must contain an nn.Conv3d for Phase 1 bootstrap")
        if int(legacy_conv.in_channels) != LEGACY_VACE_CONTEXT_CHANNELS:
            raise ControlBranchContractError(
                f"legacy patch embedding must consume 96 channels, got {int(legacy_conv.in_channels)}"
            )
        self.legacy_conv = legacy_conv
        self.quad_input_channels = int(quad_input_channels)
        self.operation_vocab = tuple(validate_operation(op) for op in operation_vocab)
        self.operation_to_index = {op: idx for idx, op in enumerate(self.operation_vocab)}
        if set(self.operation_vocab) != {"remove", "add"}:
            raise ControlBranchContractError("operation_vocab must contain exactly remove/add in Phase 1")
        self.reserve_cycle_fields = bool(reserve_cycle_fields)
        self.operation_embedding_dim = int(operation_embedding_dim)
        if self.operation_embedding_dim <= 0:
            raise ControlBranchContractError("operation_embedding_dim must be positive")

        self.operation_embedding = nn.Embedding(len(self.operation_vocab), self.operation_embedding_dim)
        self.inverse_operation_vocab = ("null", *self.operation_vocab)
        self.inverse_operation_to_index = {op: idx for idx, op in enumerate(self.inverse_operation_vocab)}
        self.inverse_operation_embedding = nn.Embedding(len(self.inverse_operation_vocab), self.operation_embedding_dim)

        adapter_in_channels = self.quad_input_channels + self.operation_embedding_dim
        if self.reserve_cycle_fields:
            adapter_in_channels += self.operation_embedding_dim
        self.quad_adapter = nn.Conv3d(
            adapter_in_channels,
            legacy_conv.out_channels,
            kernel_size=legacy_conv.kernel_size,
            stride=legacy_conv.stride,
            padding=legacy_conv.padding,
            dilation=legacy_conv.dilation,
            groups=1,
            bias=legacy_conv.bias is not None,
            padding_mode=legacy_conv.padding_mode,
        )
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init), dtype=torch.float32))
        self._init_metadata = ControlBranchInitMetadata(residual_gate_init=float(residual_gate_init))
        self.reset_control_parameters(init_std=float(init_std))

    def reset_control_parameters(self, init_std: float = 1.0e-3) -> None:
        """Option B initialization: small random adapter/op embeddings, nonzero gate."""

        nn.init.normal_(self.quad_adapter.weight, mean=0.0, std=float(init_std))
        if self.quad_adapter.bias is not None:
            nn.init.normal_(self.quad_adapter.bias, mean=0.0, std=float(init_std))
        nn.init.normal_(self.operation_embedding.weight, mean=0.0, std=float(init_std))
        nn.init.normal_(self.inverse_operation_embedding.weight, mean=0.0, std=float(init_std))

    def init_metadata(self) -> dict[str, Any]:
        return self._init_metadata.as_dict()

    def gate(self) -> torch.Tensor:
        """Return the trainable nonzero residual gate scalar."""

        return self.residual_gate

    def _indices_from_operations(
        self,
        operation: str | Sequence[str] | torch.Tensor,
        batch_size: int,
        mapping: Mapping[str, int],
        *,
        allow_none: bool = False,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(operation, torch.Tensor):
            indices = operation.to(device=device, dtype=torch.long)
            if indices.ndim == 0:
                indices = indices.expand(batch_size)
            if indices.numel() != batch_size:
                raise ControlBranchContractError(
                    f"operation tensor has {indices.numel()} entries but batch size is {batch_size}"
                )
            return indices.reshape(batch_size)
        if operation is None and allow_none:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if isinstance(operation, str) or operation is None:
            ops = ["null" if operation is None else str(operation)] * batch_size
        else:
            ops = ["null" if item is None else str(item) for item in operation]
            if len(ops) != batch_size:
                raise ControlBranchContractError(f"got {len(ops)} operations for batch size {batch_size}")
        out = []
        for op in ops:
            key = op.strip().lower()
            if key not in mapping:
                raise ControlBranchContractError(f"unsupported operation value {op!r}; expected {sorted(mapping)}")
            out.append(mapping[key])
        return torch.tensor(out, dtype=torch.long, device=device)

    def _conditioning_channels(
        self,
        *,
        batch_size: int,
        spatial_shape: Sequence[int],
        operation: str | Sequence[str] | torch.Tensor,
        inverse_operation: str | Sequence[str] | torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        t, h, w = [int(x) for x in spatial_shape]
        op_idx = self._indices_from_operations(
            operation,
            batch_size,
            self.operation_to_index,
            device=device,
        )
        op_emb = self.operation_embedding(op_idx).to(dtype=dtype).view(batch_size, -1, 1, 1, 1)
        channels = [op_emb.expand(batch_size, op_emb.shape[1], t, h, w)]
        if self.reserve_cycle_fields:
            inv_idx = self._indices_from_operations(
                inverse_operation,
                batch_size,
                self.inverse_operation_to_index,
                allow_none=True,
                device=device,
            )
            inv_emb = self.inverse_operation_embedding(inv_idx).to(dtype=dtype).view(batch_size, -1, 1, 1, 1)
            channels.append(inv_emb.expand(batch_size, inv_emb.shape[1], t, h, w))
        return torch.cat(channels, dim=1)

    def legacy_context(self, legacy_input: torch.Tensor) -> torch.Tensor:
        if legacy_input.ndim != 5 or int(legacy_input.shape[1]) != LEGACY_VACE_CONTEXT_CHANNELS:
            raise ControlBranchContractError(
                f"legacy_input must be [B,96,T,H,W], got {list(legacy_input.shape)}"
            )
        return self.legacy_patch_embedding(legacy_input)

    def forward(
        self,
        quad_input: torch.Tensor,
        operation: str | Sequence[str] | torch.Tensor,
        inverse_operation: str | Sequence[str] | torch.Tensor | None = None,
        object_side_packet: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Return the trainable control residual in embedded context space.

        ``object_side_packet`` is accepted for cycle-compatible interface
        reservation.  Phase 1 may pass ``None``; real side-packet conditioning is
        intentionally not required for bootstrap tests.
        """

        del object_side_packet  # Reserved Phase 1 interface field.
        if quad_input.ndim != 5:
            raise ControlBranchContractError(f"quad_input must be [B,C,T,H,W], got {list(quad_input.shape)}")
        if int(quad_input.shape[1]) != self.quad_input_channels:
            raise ControlBranchContractError(
                f"quad_input must have {self.quad_input_channels} channels, got {int(quad_input.shape[1])}"
            )
        b, _, t, h, w = quad_input.shape
        cond = self._conditioning_channels(
            batch_size=b,
            spatial_shape=(t, h, w),
            operation=operation,
            inverse_operation=inverse_operation,
            device=quad_input.device,
            dtype=quad_input.dtype,
        )
        return self.quad_adapter(torch.cat([quad_input, cond], dim=1))

    def compose_with_legacy(
        self,
        packed_context: torch.Tensor,
        operation: str | Sequence[str] | torch.Tensor,
        inverse_operation: str | Sequence[str] | torch.Tensor | None = None,
        object_side_packet: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Split 416 raw channels and combine frozen legacy + gated control."""

        legacy_input, quad_input = split_packed_context(packed_context)
        legacy_context = self.legacy_context(legacy_input)
        quad_context = self.forward(
            quad_input=quad_input,
            operation=operation,
            inverse_operation=inverse_operation,
            object_side_packet=object_side_packet,
        )
        if list(legacy_context.shape) != list(quad_context.shape):
            raise ControlBranchContractError(
                "quad/control branch output shape must match legacy embedded context shape: "
                f"legacy={list(legacy_context.shape)}, quad={list(quad_context.shape)}"
            )
        return {
            "legacy_context": legacy_context,
            "quad_context": quad_context,
            "gate": self.gate(),
            "context": legacy_context + self.gate().to(dtype=quad_context.dtype, device=quad_context.device) * quad_context,
        }

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def frozen_parameter_names(self) -> list[str]:
        return [name for name, param in self.named_parameters() if not param.requires_grad]

    def trainable_control_parameters(self) -> list[nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]


def assert_legacy_parameters_frozen(module: E2WGatedCausalControlBranch) -> None:
    leaking = [name for name, param in module.legacy_patch_embedding.named_parameters() if param.requires_grad]
    if leaking:
        raise ControlBranchContractError(f"legacy patch embedding parameters still trainable: {leaking}")


class LegacySliceFreezeGuard:
    """Fallback guard for temporary monolithic widened Conv3D experiments.

    The recommended Phase 1 path is split/frozen legacy + trainable E2W branch.
    If a widened 416-channel Conv3D must be used temporarily, this guard zeros
    gradients for the old 96-channel slice and restores that slice after every
    optimizer step.
    """

    def __init__(self, conv: nn.Conv3d, legacy_channels: int = LEGACY_VACE_CONTEXT_CHANNELS) -> None:
        if not isinstance(conv, nn.Conv3d):
            raise TypeError("LegacySliceFreezeGuard expects nn.Conv3d")
        if int(conv.in_channels) < int(legacy_channels):
            raise ControlBranchContractError(
                f"conv.in_channels={conv.in_channels} is smaller than legacy_channels={legacy_channels}"
            )
        self.conv = conv
        self.legacy_channels = int(legacy_channels)
        self._frozen_weight = conv.weight.detach()[:, : self.legacy_channels].clone()

    def zero_legacy_slice_grad(self) -> None:
        if self.conv.weight.grad is not None:
            self.conv.weight.grad[:, : self.legacy_channels].zero_()

    def restore_legacy_slice(self) -> None:
        with torch.no_grad():
            self.conv.weight[:, : self.legacy_channels].copy_(self._frozen_weight.to(self.conv.weight.device))

    def legacy_delta_max(self) -> float:
        current = self.conv.weight.detach()[:, : self.legacy_channels]
        ref = self._frozen_weight.to(current.device, dtype=current.dtype)
        return float((current - ref).abs().max().item())


__all__ = [
    "ALLOWED_QUADMASK_VALUES",
    "GENERATION_MASK_MODE_QUADMASK_EDITABLE",
    "LEGACY_VACE_CONTEXT_CHANNELS",
    "V03_PACKED_CONTEXT_CHANNELS",
    "V03_EXTRA_QUAD_CONTEXT_CHANNELS",
    "ControlBranchContractError",
    "E2WGatedCausalControlBranch",
    "LegacySliceFreezeGuard",
    "assert_legacy_parameters_frozen",
    "build_generation_mask_from_quadmask",
    "build_v03_compatible_packed_context",
    "freeze_module",
    "grad_norm",
    "quadmask_semantic_masks",
    "semantic_masks_from_quadmask_tensor",
    "split_packed_context",
    "validate_quadmask_values",
]

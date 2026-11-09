"""Triton inference-only dispatch for MoE-LoRA expert pools.

This path is intentionally optional and only used for latency experiments.
It avoids materializing per-token A/B expert weights with index_select.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _lora_a_kernel(
    h_ptr,
    a_ptr,
    idx_ptr,
    z_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_r = tl.arange(0, BLOCK_R)
    mask_m = offs_m < M
    mask_r = offs_r < R

    token_ids = offs_m // K
    expert_ids = tl.load(idx_ptr + offs_m, mask=mask_m, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_R), tl.float32)
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        h = tl.load(
            h_ptr + token_ids[:, None] * D + offs_d[None, :],
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        )
        a = tl.load(
            a_ptr + expert_ids[:, None, None] * (R * D) + offs_r[None, :, None] * D + offs_d[None, None, :],
            mask=mask_m[:, None, None] & mask_r[None, :, None] & mask_d[None, None, :],
            other=0.0,
        )
        acc += tl.sum(h[:, None, :] * a, axis=2)

    tl.store(
        z_ptr + offs_m[:, None] * R + offs_r[None, :],
        acc,
        mask=mask_m[:, None] & mask_r[None, :],
    )


@triton.jit
def _lora_b_scatter_kernel(
    z_ptr,
    b_ptr,
    vals_ptr,
    idx_ptr,
    out_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    SCALING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_r = tl.arange(0, BLOCK_R)
    mask_m = offs_m < M
    mask_d = offs_d < D
    mask_r = offs_r < R

    token_ids = offs_m // K
    expert_ids = tl.load(idx_ptr + offs_m, mask=mask_m, other=0).to(tl.int64)
    weights = tl.load(vals_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32) * SCALING

    z = tl.load(
        z_ptr + offs_m[:, None] * R + offs_r[None, :],
        mask=mask_m[:, None] & mask_r[None, :],
        other=0.0,
    ).to(tl.float32)
    b = tl.load(
        b_ptr + expert_ids[:, None, None] * (D * R) + offs_d[None, :, None] * R + offs_r[None, None, :],
        mask=mask_m[:, None, None] & mask_d[None, :, None] & mask_r[None, None, :],
        other=0.0,
    ).to(tl.float32)
    y = tl.sum(z[:, None, :] * b, axis=2) * weights[:, None]

    tl.atomic_add(
        out_ptr + token_ids[:, None] * D + offs_d[None, :],
        y,
        sem="relaxed",
        mask=mask_m[:, None] & mask_d[None, :],
    )


def triton_lora_dispatch(
    h_flat: torch.Tensor,
    topk_vals: torch.Tensor,
    topk_idx: torch.Tensor,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    if not h_flat.is_cuda:
        raise RuntimeError("triton_lora_dispatch requires CUDA tensors.")
    if h_flat.dim() != 2:
        raise ValueError(f"h_flat must be 2D, got {tuple(h_flat.shape)}")

    h_flat = h_flat.contiguous()
    topk_vals = topk_vals.contiguous()
    topk_idx = topk_idx.contiguous()
    a_weight = a_weight.contiguous()
    b_weight = b_weight.contiguous()

    N, D = h_flat.shape
    K = topk_idx.shape[-1]
    R = a_weight.shape[1]
    M = N * K

    idx_flat = topk_idx.reshape(M)
    vals_flat = topk_vals.reshape(M)
    z = torch.empty((M, R), device=h_flat.device, dtype=h_flat.dtype)
    out_fp32 = torch.zeros((N, D), device=h_flat.device, dtype=torch.float32)

    try:
        block_m = int(os.environ.get("MOE_LORA_TRITON_BLOCK_M", "16"))
    except ValueError:
        block_m = 16
    if block_m not in {16, 32, 64}:
        block_m = 16
    block_d = 64
    block_r = triton.next_power_of_2(R)

    grid_a = (triton.cdiv(M, block_m),)
    _lora_a_kernel[grid_a](
        h_flat,
        a_weight,
        idx_flat,
        z,
        M,
        D,
        K,
        R,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        num_warps=4,
    )

    grid_b = (triton.cdiv(M, block_m), triton.cdiv(D, block_d))
    _lora_b_scatter_kernel[grid_b](
        z,
        b_weight,
        vals_flat,
        idx_flat,
        out_fp32,
        M,
        D,
        K,
        R,
        float(scaling),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        num_warps=4,
    )
    return out_fp32.to(h_flat.dtype)

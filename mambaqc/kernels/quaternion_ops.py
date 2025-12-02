"""
Fused Triton kernels for quaternion operations.

Optimizations:
- Tiled computation to fit in shared memory (SM)
- Tensor Core compatible matrix layouts
- Minimal HBM traffic through fusion
- Vectorized loads/stores
"""

import torch
import triton
import triton.language as tl


@triton.jit
def quaternion_multiply_kernel(
    # Pointers
    p_ptr, q_ptr, out_ptr,
    # Shapes
    n_elements,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for Hamilton quaternion multiplication.

    p, q: quaternions as [n_elements, 4] tensors
    out: result of p * q (Hamilton product)

    Formula:
    p * q = (p0*q0 - p1*q1 - p2*q2 - p3*q3) +
            (p0*q1 + p1*q0 + p2*q3 - p3*q2)i +
            (p0*q2 - p1*q3 + p2*q0 + p3*q1)j +
            (p0*q3 + p1*q2 - p2*q1 + p3*q0)k

    Optimizations:
    - Coalesced memory access (stride-4 pattern)
    - Vectorized loads for all 4 components
    - Fused computation in registers
    - Single store at the end
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load quaternions (vectorized load of 4 components)
    # Layout: [n_elements, 4] - contiguous in last dimension
    p_offsets = offsets[:, None] * 4 + tl.arange(0, 4)[None, :]
    q_offsets = offsets[:, None] * 4 + tl.arange(0, 4)[None, :]

    p = tl.load(p_ptr + p_offsets, mask=mask[:, None], other=0.0)
    q = tl.load(q_ptr + q_offsets, mask=mask[:, None], other=0.0)

    # Extract components
    p0, p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Hamilton product (16 muls, 12 adds - fused in registers)
    r0 = p0*q0 - p1*q1 - p2*q2 - p3*q3
    r1 = p0*q1 + p1*q0 + p2*q3 - p3*q2
    r2 = p0*q2 - p1*q3 + p2*q0 + p3*q1
    r3 = p0*q3 + p1*q2 - p2*q1 + p3*q0

    # Stack and store (vectorized store)
    result = tl.stack([r0, r1, r2, r3], axis=1)
    out_offsets = offsets[:, None] * 4 + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + out_offsets, result, mask=mask[:, None])


@triton.jit
def quaternion_multiply_batched_kernel(
    # Pointers
    p_ptr, q_ptr, out_ptr,
    # Shapes
    batch_size, seq_len, d_model, d_state,
    stride_pb, stride_ps, stride_pd, stride_pk,
    stride_qb, stride_qs, stride_qd, stride_qk,
    stride_ob, stride_os, stride_od, stride_ok,
    # Tiling
    BLOCK_B: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batched quaternion multiplication with 4D tiling.

    Input shapes: [batch, seq_len, d_model, d_state, 4]

    Tiling strategy:
    - Process BLOCK_B batches at once
    - Process BLOCK_S sequence positions at once
    - Process BLOCK_D model dimensions at once
    - Process BLOCK_K state dimensions at once
    - Keep tiles in shared memory
    """
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_k = tl.program_id(3)

    # Compute offsets for this tile
    b_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Masks
    b_mask = b_offsets < batch_size
    s_mask = s_offsets < seq_len
    d_mask = d_offsets < d_model
    k_mask = k_offsets < d_state

    # Broadcast for 4D indexing
    b_idx = b_offsets[:, None, None, None]
    s_idx = s_offsets[None, :, None, None]
    d_idx = d_offsets[None, None, :, None]
    k_idx = k_offsets[None, None, None, :]

    # Compute base pointers
    p_base = (b_idx * stride_pb + s_idx * stride_ps +
              d_idx * stride_pd + k_idx * stride_pk)
    q_base = (b_idx * stride_qb + s_idx * stride_qs +
              d_idx * stride_qd + k_idx * stride_qk)

    # Load quaternion components (4 vectorized loads)
    mask_4d = (b_mask[:, None, None, None] & s_mask[None, :, None, None] &
               d_mask[None, None, :, None] & k_mask[None, None, None, :])

    p0 = tl.load(p_ptr + p_base + 0, mask=mask_4d, other=0.0)
    p1 = tl.load(p_ptr + p_base + 1, mask=mask_4d, other=0.0)
    p2 = tl.load(p_ptr + p_base + 2, mask=mask_4d, other=0.0)
    p3 = tl.load(p_ptr + p_base + 3, mask=mask_4d, other=0.0)

    q0 = tl.load(q_ptr + q_base + 0, mask=mask_4d, other=0.0)
    q1 = tl.load(q_ptr + q_base + 1, mask=mask_4d, other=0.0)
    q2 = tl.load(q_ptr + q_base + 2, mask=mask_4d, other=0.0)
    q3 = tl.load(q_ptr + q_base + 3, mask=mask_4d, other=0.0)

    # Hamilton product (computed in registers - stays in SM)
    r0 = p0*q0 - p1*q1 - p2*q2 - p3*q3
    r1 = p0*q1 + p1*q0 + p2*q3 - p3*q2
    r2 = p0*q2 - p1*q3 + p2*q0 + p3*q1
    r3 = p0*q3 + p1*q2 - p2*q1 + p3*q0

    # Store results
    out_base = (b_idx * stride_ob + s_idx * stride_os +
                d_idx * stride_od + k_idx * stride_ok)

    tl.store(out_ptr + out_base + 0, r0, mask=mask_4d)
    tl.store(out_ptr + out_base + 1, r1, mask=mask_4d)
    tl.store(out_ptr + out_base + 2, r2, mask=mask_4d)
    tl.store(out_ptr + out_base + 3, r3, mask=mask_4d)


@triton.jit
def quaternion_conjugate_kernel(
    q_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for quaternion conjugation: q̄ = a - bi - cj - dk

    Super fast: just sign flips, no computation.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Vectorized load
    q_offsets = offsets[:, None] * 4 + tl.arange(0, 4)[None, :]
    q = tl.load(q_ptr + q_offsets, mask=mask[:, None], other=0.0)

    # Conjugate: (a, b, c, d) -> (a, -b, -c, -d)
    signs = tl.tensor([1.0, -1.0, -1.0, -1.0], dtype=tl.float32)
    q_conj = q * signs[None, :]

    # Vectorized store
    tl.store(out_ptr + q_offsets, q_conj, mask=mask[:, None])


@triton.jit
def quaternion_inverse_kernel(
    q_ptr, out_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for quaternion inverse: q^{-1} = q̄ / ||q||²

    Includes numerical stability with epsilon clamping.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Vectorized load
    q_offsets = offsets[:, None] * 4 + tl.arange(0, 4)[None, :]
    q = tl.load(q_ptr + q_offsets, mask=mask[:, None], other=0.0)

    # Conjugate
    signs = tl.tensor([1.0, -1.0, -1.0, -1.0], dtype=tl.float32)
    q_conj = q * signs[None, :]

    # Norm squared: ||q||² = a² + b² + c² + d²
    norm_sq = tl.sum(q * q, axis=1, keep_dims=True)

    # Clamp for numerical stability
    norm_sq_safe = tl.maximum(norm_sq, eps)

    # Inverse: q̄ / ||q||²
    q_inv = q_conj / norm_sq_safe

    # Vectorized store
    tl.store(out_ptr + q_offsets, q_inv, mask=mask[:, None])


@triton.jit
def quaternion_norm_kernel(
    q_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for quaternion norm: ||q|| = sqrt(a² + b² + c² + d²)

    Returns: [n_elements, 1] tensor of norms.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Vectorized load
    q_offsets = offsets[:, None] * 4 + tl.arange(0, 4)[None, :]
    q = tl.load(q_ptr + q_offsets, mask=mask[:, None], other=0.0)

    # Norm: sqrt(sum of squares)
    norm_sq = tl.sum(q * q, axis=1)
    norm = tl.sqrt(norm_sq)

    # Store
    tl.store(out_ptr + offsets, norm, mask=mask)


# PyTorch wrapper functions

def quaternion_multiply(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Hamilton quaternion multiplication: p * q

    Args:
        p: [..., 4] quaternion tensor
        q: [..., 4] quaternion tensor (same shape as p)

    Returns:
        out: [..., 4] quaternion product
    """
    assert p.shape == q.shape, f"Shape mismatch: {p.shape} vs {q.shape}"
    assert p.shape[-1] == 4, f"Last dim must be 4, got {p.shape[-1]}"

    # Flatten to 2D for kernel
    original_shape = p.shape
    p_flat = p.reshape(-1, 4)
    q_flat = q.reshape(-1, 4)
    n_elements = p_flat.shape[0]

    # Allocate output
    out = torch.empty_like(p_flat)

    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    quaternion_multiply_kernel[grid](
        p_flat, q_flat, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(original_shape)


def quaternion_multiply_batched(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Batched quaternion multiplication with 4D tiling.

    Args:
        p: [batch, seq_len, d_model, d_state, 4]
        q: [batch, seq_len, d_model, d_state, 4]

    Returns:
        out: [batch, seq_len, d_model, d_state, 4]
    """
    assert p.shape == q.shape
    assert p.shape[-1] == 4
    assert p.ndim == 5, f"Expected 5D tensor, got {p.ndim}D"

    batch_size, seq_len, d_model, d_state, _ = p.shape

    # Allocate output
    out = torch.empty_like(p)

    # Compute strides
    stride_pb, stride_ps, stride_pd, stride_pk, _ = p.stride()
    stride_qb, stride_qs, stride_qd, stride_qk, _ = q.stride()
    stride_ob, stride_os, stride_od, stride_ok, _ = out.stride()

    # Tiling configuration (tune these for your GPU)
    BLOCK_B, BLOCK_S, BLOCK_D, BLOCK_K = 1, 4, 16, 8

    # Grid dimensions
    grid = (
        triton.cdiv(batch_size, BLOCK_B),
        triton.cdiv(seq_len, BLOCK_S),
        triton.cdiv(d_model, BLOCK_D),
        triton.cdiv(d_state, BLOCK_K),
    )

    quaternion_multiply_batched_kernel[grid](
        p, q, out,
        batch_size, seq_len, d_model, d_state,
        stride_pb, stride_ps, stride_pd, stride_pk,
        stride_qb, stride_qs, stride_qd, stride_qk,
        stride_ob, stride_os, stride_od, stride_ok,
        BLOCK_B=BLOCK_B,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
    )

    return out


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Quaternion conjugate: q̄ = a - bi - cj - dk

    Args:
        q: [..., 4] quaternion tensor

    Returns:
        q_conj: [..., 4] conjugated quaternion
    """
    assert q.shape[-1] == 4

    original_shape = q.shape
    q_flat = q.reshape(-1, 4)
    n_elements = q_flat.shape[0]

    out = torch.empty_like(q_flat)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    quaternion_conjugate_kernel[grid](
        q_flat, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(original_shape)


def quaternion_inverse(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Quaternion inverse: q^{-1} = q̄ / ||q||²

    Args:
        q: [..., 4] quaternion tensor
        eps: epsilon for numerical stability

    Returns:
        q_inv: [..., 4] inverse quaternion
    """
    assert q.shape[-1] == 4

    original_shape = q.shape
    q_flat = q.reshape(-1, 4)
    n_elements = q_flat.shape[0]

    out = torch.empty_like(q_flat)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    quaternion_inverse_kernel[grid](
        q_flat, out,
        n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(original_shape)


def quaternion_norm(q: torch.Tensor) -> torch.Tensor:
    """
    Quaternion norm: ||q|| = sqrt(a² + b² + c² + d²)

    Args:
        q: [..., 4] quaternion tensor

    Returns:
        norm: [...] norm tensor (scalar per quaternion)
    """
    assert q.shape[-1] == 4

    original_shape = q.shape[:-1]
    q_flat = q.reshape(-1, 4)
    n_elements = q_flat.shape[0]

    out = torch.empty(n_elements, dtype=q.dtype, device=q.device)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    quaternion_norm_kernel[grid](
        q_flat, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(original_shape)

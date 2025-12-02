"""
Fused Cayley discretization kernel.

Implements: q = (1 - 0.5*z)^{-1} * (1 + 0.5*z)

Where z = Δ * Λ is the continuous dynamics parameter.

This kernel fuses:
1. Computation of numerator: 1 + 0.5*z
2. Computation of denominator: 1 - 0.5*z
3. Quaternion inverse of denominator
4. Quaternion multiplication: denom_inv * numerator

All operations stay in registers/shared memory.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def cayley_discretization_kernel(
    # Inputs
    z_ptr,  # [batch*seq_len, d_model, d_state, 4] - dynamics (flattened)
    # Outputs
    q_ptr,  # [batch*seq_len, d_model, d_state, 4] - discrete operators (flattened)
    # Shapes
    n_batch_seq, d_model, d_state,
    # Strides
    stride_zbs, stride_zd, stride_zk,
    stride_qbs, stride_qd, stride_qk,
    # Constants
    eps: tl.constexpr,
    # Tiling
    BLOCK_BS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused Cayley transform kernel (3D grid).

    Computes: q = (1 - 0.5*z)^{-1} * (1 + 0.5*z)

    All intermediate results stay in registers (no HBM roundtrip).

    Grid: (batch*seq_len, d_model, d_state)
    """
    pid_bs = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute offsets
    bs_offsets = pid_bs * BLOCK_BS + tl.arange(0, BLOCK_BS)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Masks
    bs_mask = bs_offsets < n_batch_seq
    d_mask = d_offsets < d_model
    k_mask = k_offsets < d_state

    # Broadcast for 3D indexing
    bs_idx = bs_offsets[:, None, None]
    d_idx = d_offsets[None, :, None]
    k_idx = k_offsets[None, None, :]

    # Compute base pointer for z
    z_base = (bs_idx * stride_zbs + d_idx * stride_zd + k_idx * stride_zk)

    # Load z components
    mask_3d = bs_mask[:, None, None] & d_mask[None, :, None] & k_mask[None, None, :]

    z0 = tl.load(z_ptr + z_base + 0, mask=mask_3d, other=0.0)
    z1 = tl.load(z_ptr + z_base + 1, mask=mask_3d, other=0.0)
    z2 = tl.load(z_ptr + z_base + 2, mask=mask_3d, other=0.0)
    z3 = tl.load(z_ptr + z_base + 3, mask=mask_3d, other=0.0)

    # Identity quaternion: (1, 0, 0, 0)
    one_0 = 1.0
    one_1 = 0.0
    one_2 = 0.0
    one_3 = 0.0

    # Compute numerator: 1 + 0.5*z (stays in registers)
    num_0 = one_0 + 0.5 * z0
    num_1 = one_1 + 0.5 * z1
    num_2 = one_2 + 0.5 * z2
    num_3 = one_3 + 0.5 * z3

    # Compute denominator: 1 - 0.5*z (stays in registers)
    den_0 = one_0 - 0.5 * z0
    den_1 = one_1 - 0.5 * z1
    den_2 = one_2 - 0.5 * z2
    den_3 = one_3 - 0.5 * z3

    # Inverse of denominator: den^{-1} = den̄ / ||den||²
    # Conjugate: (a, b, c, d) -> (a, -b, -c, -d)
    den_conj_0 = den_0
    den_conj_1 = -den_1
    den_conj_2 = -den_2
    den_conj_3 = -den_3

    # Norm squared
    den_norm_sq = den_0*den_0 + den_1*den_1 + den_2*den_2 + den_3*den_3
    den_norm_sq_safe = tl.maximum(den_norm_sq, eps)

    # Inverse
    den_inv_0 = den_conj_0 / den_norm_sq_safe
    den_inv_1 = den_conj_1 / den_norm_sq_safe
    den_inv_2 = den_conj_2 / den_norm_sq_safe
    den_inv_3 = den_conj_3 / den_norm_sq_safe

    # Quaternion multiplication: q = den_inv * numerator
    # Hamilton product formula (16 muls, 12 adds)
    q0 = den_inv_0*num_0 - den_inv_1*num_1 - den_inv_2*num_2 - den_inv_3*num_3
    q1 = den_inv_0*num_1 + den_inv_1*num_0 + den_inv_2*num_3 - den_inv_3*num_2
    q2 = den_inv_0*num_2 - den_inv_1*num_3 + den_inv_2*num_0 + den_inv_3*num_1
    q3 = den_inv_0*num_3 + den_inv_1*num_2 - den_inv_2*num_1 + den_inv_3*num_0

    # Store results (4 coalesced stores)
    q_base = (bs_idx * stride_qbs + d_idx * stride_qd + k_idx * stride_qk)

    tl.store(q_ptr + q_base + 0, q0, mask=mask_3d)
    tl.store(q_ptr + q_base + 1, q1, mask=mask_3d)
    tl.store(q_ptr + q_base + 2, q2, mask=mask_3d)
    tl.store(q_ptr + q_base + 3, q3, mask=mask_3d)


def cayley_discretization_fused(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Fused Cayley discretization.

    Args:
        z: [batch, seq_len, d_model, d_state, 4] - continuous dynamics
        eps: epsilon for numerical stability in inverse

    Returns:
        q: [batch, seq_len, d_model, d_state, 4] - discrete operators

    Formula:
        q = (1 - 0.5*z)^{-1} * (1 + 0.5*z)

    Properties:
        - If Re(z) < 0, then |q| < 1 (stability)
        - Higher order than ZOH discretization
        - Preserves unitary structure
    """
    assert z.shape[-1] == 4, f"Last dim must be 4, got {z.shape[-1]}"
    assert z.ndim == 5, f"Expected 5D tensor, got {z.ndim}D"

    batch_size, seq_len, d_model, d_state, _ = z.shape
    original_shape = z.shape

    # Flatten batch and seq_len for 3D grid
    z_flat = z.reshape(batch_size * seq_len, d_model, d_state, 4)
    n_batch_seq = batch_size * seq_len

    # Allocate output (flattened)
    q_flat = torch.empty_like(z_flat)

    # Compute strides (for flattened tensors)
    stride_zbs, stride_zd, stride_zk, _ = z_flat.stride()
    stride_qbs, stride_qd, stride_qk, _ = q_flat.stride()

    # Tiling configuration (tune for your GPU)
    BLOCK_BS, BLOCK_D, BLOCK_K = 4, 16, 8

    # Grid dimensions (3D only)
    grid = (
        triton.cdiv(n_batch_seq, BLOCK_BS),
        triton.cdiv(d_model, BLOCK_D),
        triton.cdiv(d_state, BLOCK_K),
    )

    cayley_discretization_kernel[grid](
        z_flat, q_flat,
        n_batch_seq, d_model, d_state,
        stride_zbs, stride_zd, stride_zk,
        stride_qbs, stride_qd, stride_qk,
        eps=eps,
        BLOCK_BS=BLOCK_BS,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
    )

    # Reshape back to original shape
    q = q_flat.reshape(original_shape)

    return q


# Verification function for testing
def cayley_discretization_reference(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Reference implementation for testing (not optimized).

    Uses the quaternion ops from quaternion_ops.py
    """
    from .quaternion_ops import quaternion_inverse, quaternion_multiply

    # Identity quaternion
    one = torch.tensor([1.0, 0.0, 0.0, 0.0], device=z.device, dtype=z.dtype)
    one = one.view(1, 1, 1, 1, 4).expand_as(z)

    # Numerator: 1 + 0.5*z
    numerator = one + 0.5 * z

    # Denominator: 1 - 0.5*z
    denominator = one - 0.5 * z

    # Inverse of denominator
    denominator_inv = quaternion_inverse(denominator, eps=eps)

    # Multiply: q = denominator_inv * numerator
    q = quaternion_multiply(denominator_inv, numerator)

    return q

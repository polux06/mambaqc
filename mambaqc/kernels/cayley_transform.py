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
    z_ptr,  # [batch, seq_len, d_model, d_state, 4] - dynamics
    # Outputs
    q_ptr,  # [batch, seq_len, d_model, d_state, 4] - discrete operators
    # Shapes
    batch_size, seq_len, d_model, d_state,
    # Strides
    stride_zb, stride_zs, stride_zd, stride_zk,
    stride_qb, stride_qs, stride_qd, stride_qk,
    # Constants
    eps: tl.constexpr,
    # Tiling
    BLOCK_B: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused Cayley transform kernel.

    Computes: q = (1 - 0.5*z)^{-1} * (1 + 0.5*z)

    All intermediate results stay in registers (no HBM roundtrip).
    """
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_k = tl.program_id(3)

    # Compute offsets
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

    # Compute base pointer for z
    z_base = (b_idx * stride_zb + s_idx * stride_zs +
              d_idx * stride_zd + k_idx * stride_zk)

    # Load z components
    mask_4d = (b_mask[:, None, None, None] & s_mask[None, :, None, None] &
               d_mask[None, None, :, None] & k_mask[None, None, None, :])

    z0 = tl.load(z_ptr + z_base + 0, mask=mask_4d, other=0.0)
    z1 = tl.load(z_ptr + z_base + 1, mask=mask_4d, other=0.0)
    z2 = tl.load(z_ptr + z_base + 2, mask=mask_4d, other=0.0)
    z3 = tl.load(z_ptr + z_base + 3, mask=mask_4d, other=0.0)

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
    q_base = (b_idx * stride_qb + s_idx * stride_qs +
              d_idx * stride_qd + k_idx * stride_qk)

    tl.store(q_ptr + q_base + 0, q0, mask=mask_4d)
    tl.store(q_ptr + q_base + 1, q1, mask=mask_4d)
    tl.store(q_ptr + q_base + 2, q2, mask=mask_4d)
    tl.store(q_ptr + q_base + 3, q3, mask=mask_4d)


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

    # Allocate output
    q = torch.empty_like(z)

    # Compute strides
    stride_zb, stride_zs, stride_zd, stride_zk, _ = z.stride()
    stride_qb, stride_qs, stride_qd, stride_qk, _ = q.stride()

    # Tiling configuration (tune for your GPU)
    BLOCK_B, BLOCK_S, BLOCK_D, BLOCK_K = 1, 4, 16, 8

    # Grid dimensions
    grid = (
        triton.cdiv(batch_size, BLOCK_B),
        triton.cdiv(seq_len, BLOCK_S),
        triton.cdiv(d_model, BLOCK_D),
        triton.cdiv(d_state, BLOCK_K),
    )

    cayley_discretization_kernel[grid](
        z, q,
        batch_size, seq_len, d_model, d_state,
        stride_zb, stride_zs, stride_zd, stride_zk,
        stride_qb, stride_qs, stride_qd, stride_qk,
        eps=eps,
        BLOCK_B=BLOCK_B,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
    )

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

"""
Geometric quaternion layer normalization.

Key principle: Normalize the NORMS of quaternions, not individual components.

This preserves the geometric structure:
- Direction (rotation encoded in quaternion) is preserved
- Only the magnitude is normalized

Analogy: For complex numbers z = r·e^(iθ), we normalize r but keep θ.

For quaternions: q = ||q|| · (unit quaternion), we normalize ||q|| but preserve direction.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def quaternion_layer_norm_kernel(
    # Inputs
    q_ptr,  # [batch, seq_len, d_model, 4]
    gamma_ptr,  # [d_model] - learnable scale
    beta_ptr,  # [d_model, 4] - learnable shift
    # Outputs
    out_ptr,  # [batch, seq_len, d_model, 4]
    # Shapes
    batch_size, seq_len, d_model,
    # Strides
    stride_qb, stride_qs, stride_qd,
    stride_ob, stride_os, stride_od,
    # Constants
    eps: tl.constexpr,
    # Tiling
    BLOCK_B: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Quaternion layer normalization kernel.

    Algorithm:
    1. Compute ||q|| = sqrt(a² + b² + c² + d²) for each quaternion
    2. Compute mean and variance of NORMS across d_model dimension
    3. Normalize norms: norm_normalized = (norm - mean) / sqrt(var + eps)
    4. Preserve direction: direction = q / ||q||
    5. Reconstruct: q_out = gamma * norm_normalized * direction + beta

    All operations fused in a single kernel.
    """
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    # This kernel processes one (batch, seq) pair
    # and normalizes across the d_model dimension

    b_idx = pid_b * BLOCK_B
    s_idx = pid_s * BLOCK_S

    if b_idx >= batch_size or s_idx >= seq_len:
        return

    # Load all quaternions for this (b, s) position
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < d_model

    # Base pointers
    q_base = b_idx * stride_qb + s_idx * stride_qs + d_offsets * stride_qd
    out_base = b_idx * stride_ob + s_idx * stride_os + d_offsets * stride_od

    # Load quaternion components (4 vectorized loads)
    q0 = tl.load(q_ptr + q_base + 0, mask=d_mask, other=0.0)
    q1 = tl.load(q_ptr + q_base + 1, mask=d_mask, other=0.0)
    q2 = tl.load(q_ptr + q_base + 2, mask=d_mask, other=0.0)
    q3 = tl.load(q_ptr + q_base + 3, mask=d_mask, other=0.0)

    # Compute norms: ||q|| = sqrt(q0² + q1² + q2² + q3²)
    norm_sq = q0*q0 + q1*q1 + q2*q2 + q3*q3
    norm = tl.sqrt(norm_sq + eps)  # Add eps to avoid sqrt(0)

    # Compute statistics of norms across d_model
    # (This is where we differ from naive component-wise normalization)
    norm_mean = tl.sum(norm, axis=0) / d_model
    norm_centered = norm - norm_mean
    norm_var = tl.sum(norm_centered * norm_centered, axis=0) / d_model

    # Normalize norms
    norm_normalized = norm_centered / tl.sqrt(norm_var + eps)

    # Compute unit direction (preserves rotation information)
    # direction = q / ||q||
    dir0 = q0 / (norm + eps)
    dir1 = q1 / (norm + eps)
    dir2 = q2 / (norm + eps)
    dir3 = q3 / (norm + eps)

    # Load learnable parameters
    gamma = tl.load(gamma_ptr + d_offsets, mask=d_mask, other=1.0)
    beta0 = tl.load(beta_ptr + d_offsets * 4 + 0, mask=d_mask, other=0.0)
    beta1 = tl.load(beta_ptr + d_offsets * 4 + 1, mask=d_mask, other=0.0)
    beta2 = tl.load(beta_ptr + d_offsets * 4 + 2, mask=d_mask, other=0.0)
    beta3 = tl.load(beta_ptr + d_offsets * 4 + 3, mask=d_mask, other=0.0)

    # Reconstruct with learned scale and shift
    # q_out = gamma * norm_normalized * direction + beta
    out0 = gamma * norm_normalized * dir0 + beta0
    out1 = gamma * norm_normalized * dir1 + beta1
    out2 = gamma * norm_normalized * dir2 + beta2
    out3 = gamma * norm_normalized * dir3 + beta3

    # Store results (4 vectorized stores)
    tl.store(out_ptr + out_base + 0, out0, mask=d_mask)
    tl.store(out_ptr + out_base + 1, out1, mask=d_mask)
    tl.store(out_ptr + out_base + 2, out2, mask=d_mask)
    tl.store(out_ptr + out_base + 3, out3, mask=d_mask)


def quaternion_layer_norm_fused(
    q: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Fused quaternion layer normalization.

    Args:
        q: [batch, seq_len, d_model, 4] - input quaternions
        gamma: [d_model] - learnable scale parameter
        beta: [d_model, 4] - learnable shift parameter
        eps: epsilon for numerical stability

    Returns:
        out: [batch, seq_len, d_model, 4] - normalized quaternions

    Properties:
        - Preserves quaternion direction (rotation information)
        - Normalizes quaternion magnitudes
        - Stabilizes training by preventing norm explosion
    """
    assert q.shape[-1] == 4
    assert q.ndim == 4, f"Expected [B, T, D, 4], got {q.shape}"
    assert gamma.shape == (q.shape[2],), f"gamma shape mismatch: {gamma.shape}"
    assert beta.shape == (q.shape[2], 4), f"beta shape mismatch: {beta.shape}"

    batch_size, seq_len, d_model, _ = q.shape

    # Allocate output
    out = torch.empty_like(q)

    # Compute strides
    stride_qb, stride_qs, stride_qd, _ = q.stride()
    stride_ob, stride_os, stride_od, _ = out.stride()

    # Tiling (process one d_model dimension at a time)
    BLOCK_B, BLOCK_S = 1, 1
    BLOCK_D = triton.next_power_of_2(d_model)

    # Grid: one thread block per (batch, seq_len) position
    grid = (batch_size, seq_len)

    quaternion_layer_norm_kernel[grid](
        q, gamma, beta, out,
        batch_size, seq_len, d_model,
        stride_qb, stride_qs, stride_qd,
        stride_ob, stride_os, stride_od,
        eps=eps,
        BLOCK_B=BLOCK_B,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
    )

    return out


class QuaternionLayerNorm(torch.nn.Module):
    """
    Learnable quaternion layer normalization module.

    Normalizes quaternion norms while preserving directions.

    Args:
        d_model: model dimension
        eps: epsilon for numerical stability

    Shape:
        - Input: [batch, seq_len, d_model, 4]
        - Output: [batch, seq_len, d_model, 4]
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable parameters
        self.gamma = torch.nn.Parameter(torch.ones(d_model))
        self.beta = torch.nn.Parameter(torch.zeros(d_model, 4))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: [batch, seq_len, d_model, 4]

        Returns:
            out: [batch, seq_len, d_model, 4]
        """
        return quaternion_layer_norm_fused(q, self.gamma, self.beta, self.eps)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"


# Reference implementation for testing
def quaternion_layer_norm_reference(
    q: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Reference implementation (not optimized) for testing.
    """
    # Compute norms: [B, T, D]
    norm = torch.sqrt((q ** 2).sum(dim=-1) + eps)

    # Statistics on norms (across d_model dimension)
    mean_norm = norm.mean(dim=-1, keepdim=True)  # [B, T, 1]
    var_norm = norm.var(dim=-1, keepdim=True, unbiased=False)  # [B, T, 1]

    # Normalize norms
    norm_normalized = (norm - mean_norm) / torch.sqrt(var_norm + eps)  # [B, T, D]

    # Unit direction
    direction = q / (norm.unsqueeze(-1) + eps)  # [B, T, D, 4]

    # Reconstruct
    out = (gamma.view(1, 1, -1, 1) * norm_normalized.unsqueeze(-1) * direction +
           beta.view(1, 1, -1, 4))

    return out

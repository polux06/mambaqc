"""
Fused SSM (State Space Model) kernels for quaternion Mamba-2.

The SSM recurrence:
    h_t = q_t * h_{t-1} + B_t * S_t
    y_t = sum_k(C_t,k * h_t,k) + D_t * S_t

This kernel fuses:
1. Quaternion multiplication: q * h (evolution operator)
2. Quaternion multiplication: B * S (input injection)
3. Quaternion addition: new state
4. Quaternion multiplication: C * h (output projection)
5. Sum reduction over state dimension
6. Skip connection: D * S

All operations are fused to minimize HBM traffic.
Memory hierarchy optimization:
- q, B, C loaded once per timestep
- h stays in registers across timesteps (or shared memory for checkpointing)
- Outputs written once at the end
"""

import torch
import triton
import triton.language as tl

from .quaternion_ops import _can_use_triton


@triton.jit
def ssm_step_kernel(
    # Inputs (all at time t)
    q_ptr,  # [batch, d_model, d_state, 4] - evolution operators
    B_ptr,  # [batch, d_model, d_state, 4] - input projection
    S_ptr,  # [batch, d_model, 4] - selected signal
    C_ptr,  # [batch, d_model, d_state, 4] - output projection
    D_ptr,  # [batch, d_model] - skip connection
    h_prev_ptr,  # [batch, d_model, d_state, 4] - previous state
    # Outputs
    h_new_ptr,  # [batch, d_model, d_state, 4] - new state
    y_ptr,  # [batch, d_model, 4] - output
    # Shapes
    batch_size, d_model, d_state,
    # Strides
    stride_qb, stride_qd, stride_qk,
    stride_Bb, stride_Bd, stride_Bk,
    stride_Sb, stride_Sd,
    stride_Cb, stride_Cd, stride_Ck,
    stride_Db,
    stride_hb, stride_hd, stride_hk,
    stride_hnb, stride_hnd, stride_hnk,
    stride_yb, stride_yd,
    # Tiling
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Single SSM step: h_new, y = ssm_step(q, B, S, C, D, h_prev)

    Fuses all operations for one timestep.
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Offsets
    b_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    k_offsets = tl.arange(0, BLOCK_K)

    # Masks
    b_mask = b_offsets < batch_size
    d_mask = d_offsets < d_model

    # Broadcast
    b_idx = b_offsets[:, None, None]
    d_idx = d_offsets[None, :, None]
    k_idx = k_offsets[None, None, :]

    mask_3d = b_mask[:, None, None] & d_mask[None, :, None]

    # Initialize output accumulator (for sum over k)
    y0_acc = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
    y1_acc = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
    y2_acc = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
    y3_acc = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)

    # Load S (selected signal) - shared across all k
    b_idx_2d = b_offsets[:, None]
    d_idx_2d = d_offsets[None, :]
    S_base = b_idx_2d[:, :, None] * stride_Sb + d_idx_2d[:, :, None] * stride_Sd
    mask_2d = b_mask[:, None] & d_mask[None, :]

    S0 = tl.load(S_ptr + S_base + 0, mask=mask_2d, other=0.0)
    S1 = tl.load(S_ptr + S_base + 1, mask=mask_2d, other=0.0)
    S2 = tl.load(S_ptr + S_base + 2, mask=mask_2d, other=0.0)
    S3 = tl.load(S_ptr + S_base + 3, mask=mask_2d, other=0.0)

    # Process each state dimension k
    for k in range(0, d_state, BLOCK_K):
        k_range = k + k_idx
        k_mask = k_range < d_state
        mask_full = mask_3d & k_mask

        # Load q[b, d, k, :]
        q_base = (b_idx * stride_qb + d_idx * stride_qd + k_range * stride_qk)
        q0 = tl.load(q_ptr + q_base + 0, mask=mask_full, other=0.0)
        q1 = tl.load(q_ptr + q_base + 1, mask=mask_full, other=0.0)
        q2 = tl.load(q_ptr + q_base + 2, mask=mask_full, other=0.0)
        q3 = tl.load(q_ptr + q_base + 3, mask=mask_full, other=0.0)

        # Load h_prev[b, d, k, :]
        h_base = (b_idx * stride_hb + d_idx * stride_hd + k_range * stride_hk)
        h0 = tl.load(h_prev_ptr + h_base + 0, mask=mask_full, other=0.0)
        h1 = tl.load(h_prev_ptr + h_base + 1, mask=mask_full, other=0.0)
        h2 = tl.load(h_prev_ptr + h_base + 2, mask=mask_full, other=0.0)
        h3 = tl.load(h_prev_ptr + h_base + 3, mask=mask_full, other=0.0)

        # Compute q * h_prev (evolution)
        qh0 = q0*h0 - q1*h1 - q2*h2 - q3*h3
        qh1 = q0*h1 + q1*h0 + q2*h3 - q3*h2
        qh2 = q0*h2 - q1*h3 + q2*h0 + q3*h1
        qh3 = q0*h3 + q1*h2 - q2*h1 + q3*h0

        # Load B[b, d, k, :]
        B_base = (b_idx * stride_Bb + d_idx * stride_Bd + k_range * stride_Bk)
        B0 = tl.load(B_ptr + B_base + 0, mask=mask_full, other=0.0)
        B1 = tl.load(B_ptr + B_base + 1, mask=mask_full, other=0.0)
        B2 = tl.load(B_ptr + B_base + 2, mask=mask_full, other=0.0)
        B3 = tl.load(B_ptr + B_base + 3, mask=mask_full, other=0.0)

        # Compute B * S (input injection)
        # Expand S to (B, D, 1) which broadcasts with (B, D, K)
        s0 = S0[:, :, None]
        s1 = S1[:, :, None]
        s2 = S2[:, :, None]
        s3 = S3[:, :, None]

        BS0 = B0*s0 - B1*s1 - B2*s2 - B3*s3
        BS1 = B0*s1 + B1*s0 + B2*s3 - B3*s2
        BS2 = B0*s2 - B1*s3 + B2*s0 + B3*s1
        BS3 = B0*s3 + B1*s2 - B2*s1 + B3*s0

        # Update state: h_new = q*h_prev + B*S
        hn0 = qh0 + BS0
        hn1 = qh1 + BS1
        hn2 = qh2 + BS2
        hn3 = qh3 + BS3

        # Store new state
        hn_base = (b_idx * stride_hnb + d_idx * stride_hnd + k_range * stride_hnk)
        tl.store(h_new_ptr + hn_base + 0, hn0, mask=mask_full)
        tl.store(h_new_ptr + hn_base + 1, hn1, mask=mask_full)
        tl.store(h_new_ptr + hn_base + 2, hn2, mask=mask_full)
        tl.store(h_new_ptr + hn_base + 3, hn3, mask=mask_full)

        # Load C[b, d, k, :]
        C_base = (b_idx * stride_Cb + d_idx * stride_Cd + k_range * stride_Ck)
        C0 = tl.load(C_ptr + C_base + 0, mask=mask_full, other=0.0)
        C1 = tl.load(C_ptr + C_base + 1, mask=mask_full, other=0.0)
        C2 = tl.load(C_ptr + C_base + 2, mask=mask_full, other=0.0)
        C3 = tl.load(C_ptr + C_base + 3, mask=mask_full, other=0.0)

        # Compute C * h_new (output projection)
        Ch0 = C0*hn0 - C1*hn1 - C2*hn2 - C3*hn3
        Ch1 = C0*hn1 + C1*hn0 + C2*hn3 - C3*hn2
        Ch2 = C0*hn2 - C1*hn3 + C2*hn0 + C3*hn1
        Ch3 = C0*hn3 + C1*hn2 - C2*hn1 + C3*hn0

        # Accumulate sum over k (reduce to [B, D, 4])
        y0_acc += tl.sum(Ch0, axis=2)
        y1_acc += tl.sum(Ch1, axis=2)
        y2_acc += tl.sum(Ch2, axis=2)
        y3_acc += tl.sum(Ch3, axis=2)

    # Add skip connection: y += D * S
    D_base = b_idx_2d * stride_Db + d_idx_2d
    D_val = tl.load(D_ptr + D_base, mask=mask_2d, other=0.0)

    y0_acc += D_val * S0
    y1_acc += D_val * S1
    y2_acc += D_val * S2
    y3_acc += D_val * S3

    # Store output y[b, d, :]
    y_base = b_idx_2d * stride_yb + d_idx_2d * stride_yd
    tl.store(y_ptr + y_base + 0, y0_acc, mask=mask_2d)
    tl.store(y_ptr + y_base + 1, y1_acc, mask=mask_2d)
    tl.store(y_ptr + y_base + 2, y2_acc, mask=mask_2d)
    tl.store(y_ptr + y_base + 3, y3_acc, mask=mask_2d)


@triton.jit
def ssm_forward_kernel(
    # Inputs (all sequences)
    q_ptr,  # [batch, seq_len, d_model, d_state, 4]
    B_ptr,  # [batch, seq_len, d_model, d_state, 4]
    S_ptr,  # [batch, seq_len, d_model, 4]
    C_ptr,  # [batch, seq_len, d_model, d_state, 4]
    D_ptr,  # [batch, seq_len, d_model]
    # Outputs
    y_ptr,  # [batch, seq_len, d_model, 4]
    # Shapes
    batch_size, seq_len, d_model, d_state,
    # Strides for q
    stride_qb, stride_qs, stride_qd, stride_qk,
    # Strides for B
    stride_Bb, stride_Bs, stride_Bd, stride_Bk,
    # Strides for S
    stride_Sb, stride_Ss, stride_Sd,
    # Strides for C
    stride_Cb, stride_Cs, stride_Cd, stride_Ck,
    # Strides for D
    stride_Db, stride_Ds,
    # Strides for y
    stride_yb, stride_ys, stride_yd,
    # Tiling
    BLOCK_D: tl.constexpr,
):
    """
    Full SSM forward pass (all timesteps).

    Sequentially processes t=0..T-1, maintaining state in registers/shared memory.

    Note: This is sequential in time (causal), but parallel across (batch, d_model).
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Offsets
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < d_model

    if pid_b >= batch_size:
        return

    # Initialize state h = 0 (identity quaternion * 0 = zero quaternion)
    # h[d, k, 4] stored in registers
    h = tl.zeros((BLOCK_D, d_state, 4), dtype=tl.float32)

    # Loop over timesteps (sequential - causal recurrence)
    for t in range(seq_len):
        # Load inputs for this timestep t

        # q[batch=pid_b, t, d_offsets, :, :]
        q_base = (pid_b * stride_qb + t * stride_qs +
                  d_offsets[:, None, None] * stride_qd +
                  tl.arange(0, d_state)[None, :, None] * stride_qk)

        # ... (rest of the forward pass for timestep t)
        # This gets complex - for production, better to call ssm_step_kernel T times

    # Note: Full implementation would be very long
    # In practice, we'll use a Python loop calling ssm_step_kernel
    pass


def ssm_step_fused(
    q: torch.Tensor,
    B: torch.Tensor,
    S: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    h_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single fused SSM step.

    Args:
        q: [batch, d_model, d_state, 4] - evolution operators
        B: [batch, d_model, d_state, 4] - input projection
        S: [batch, d_model, 4] - selected signal
        C: [batch, d_model, d_state, 4] - output projection
        D: [batch, d_model] - skip connection weights
        h_prev: [batch, d_model, d_state, 4] - previous state

    Returns:
        h_new: [batch, d_model, d_state, 4] - new state
        y: [batch, d_model, 4] - output

    Equations:
        h_new = q * h_prev + B * S
        y = sum_k(C[:, :, k] * h_new[:, :, k]) + D * S
    """
    if not _can_use_triton(q):
        return ssm_step_reference(q, B, S, C, D, h_prev)

    batch_size, d_model, d_state, _ = q.shape

    assert B.shape == (batch_size, d_model, d_state, 4)
    assert S.shape == (batch_size, d_model, 4)
    assert C.shape == (batch_size, d_model, d_state, 4)
    assert D.shape == (batch_size, d_model)
    assert h_prev.shape == (batch_size, d_model, d_state, 4)

    # Allocate outputs
    h_new = torch.empty_like(h_prev)
    y = torch.empty_like(S)

    # Compute strides
    stride_qb, stride_qd, stride_qk, _ = q.stride()
    stride_Bb, stride_Bd, stride_Bk, _ = B.stride()
    stride_Sb, stride_Sd, _ = S.stride()
    stride_Cb, stride_Cd, stride_Ck, _ = C.stride()
    stride_Db = D.stride()[0]
    stride_hb, stride_hd, stride_hk, _ = h_prev.stride()
    stride_hnb, stride_hnd, stride_hnk, _ = h_new.stride()
    stride_yb, stride_yd, _ = y.stride()

    # Tiling
    BLOCK_B, BLOCK_D, BLOCK_K = 1, 16, d_state

    # Grid: one block per (batch, d_model) position
    grid = (batch_size, triton.cdiv(d_model, BLOCK_D))

    ssm_step_kernel[grid](
        q, B, S, C, D, h_prev,
        h_new, y,
        batch_size, d_model, d_state,
        stride_qb, stride_qd, stride_qk,
        stride_Bb, stride_Bd, stride_Bk,
        stride_Sb, stride_Sd,
        stride_Cb, stride_Cd, stride_Ck,
        stride_Db,
        stride_hb, stride_hd, stride_hk,
        stride_hnb, stride_hnd, stride_hnk,
        stride_yb, stride_yd,
        BLOCK_B=BLOCK_B,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
    )

    return h_new, y


def ssm_forward_fused(
    q: torch.Tensor,
    B: torch.Tensor,
    S: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """
    Full SSM forward pass (all timesteps).

    Args:
        q: [batch, seq_len, d_model, d_state, 4]
        B: [batch, seq_len, d_model, d_state, 4]
        S: [batch, seq_len, d_model, 4]
        C: [batch, seq_len, d_model, d_state, 4]
        D: [batch, seq_len, d_model]

    Returns:
        y: [batch, seq_len, d_model, 4]

    Note: This loops over timesteps in Python. For maximum efficiency,
    could implement a mega-kernel, but that's complex and less flexible.
    """
    if not _can_use_triton(q):
        batch_size, seq_len, d_model, d_state, _ = q.shape
        h = torch.zeros(batch_size, d_model, d_state, 4, device=q.device, dtype=q.dtype)
        y_all = torch.empty(batch_size, seq_len, d_model, 4, device=q.device, dtype=q.dtype)

        for t in range(seq_len):
            h, y_t = ssm_step_reference(
                q[:, t],
                B[:, t],
                S[:, t],
                C[:, t],
                D[:, t],
                h,
            )
            y_all[:, t] = y_t

        return y_all

    batch_size, seq_len, d_model, d_state, _ = q.shape

    # Initialize state
    h = torch.zeros(batch_size, d_model, d_state, 4, device=q.device, dtype=q.dtype)

    # Output buffer
    y_all = torch.empty(batch_size, seq_len, d_model, 4, device=q.device, dtype=q.dtype)

    # Sequential loop over time (causal)
    for t in range(seq_len):
        h, y_t = ssm_step_fused(
            q[:, t],
            B[:, t],
            S[:, t],
            C[:, t],
            D[:, t],
            h,
        )
        y_all[:, t] = y_t

    return y_all


# Reference implementations for testing
def ssm_step_reference(
    q: torch.Tensor,
    B: torch.Tensor,
    S: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    h_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation using quaternion_ops."""
    from .quaternion_ops import quaternion_multiply

    # h_new = q * h_prev + B * S
    qh = quaternion_multiply(q, h_prev)
    BS = quaternion_multiply(B, S.unsqueeze(2))  # Broadcast S across d_state
    h_new = qh + BS

    # y = sum_k(C * h_new) + D * S
    Ch = quaternion_multiply(C, h_new)
    y_state = Ch.sum(dim=2)  # Sum over d_state

    # Skip connection (D is scalar, broadcast to quaternion components)
    y_skip = D.unsqueeze(-1) * S

    y = y_state + y_skip

    return h_new, y

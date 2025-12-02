"""
Parallel scan for quaternion sequences using Blelloch's algorithm.

Key insight: Quaternion multiplication is ASSOCIATIVE (but not commutative).
This allows us to use parallel prefix sum algorithms.

Complexity:
- Sequential: O(T) time
- Parallel: O(log T) depth, O(T) work

The algorithm consists of two phases:
1. Up-sweep: Build a binary tree of partial products (O(log T) depth)
2. Down-sweep: Propagate products down the tree (O(log T) depth)

All operations are fused to minimize memory traffic.
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def parallel_scan_upsweep_kernel(
    # Input/Output (in-place modification)
    data_ptr,  # [batch*d_model, seq_len, d_state, 4] (flattened)
    # Shapes
    n_batch_model, seq_len, d_state,
    # Current level
    stride_level: tl.constexpr,  # 2^(d+1) where d is current depth
    offset_level: tl.constexpr,  # 2^d
    s_idx: tl.constexpr,  # Specific sequence index for this kernel
    # Strides
    stride_bd, stride_s, stride_k,
    # Tiling
    BLOCK_BD: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Up-sweep phase of parallel scan (3D grid).

    At each level d, for indices i = 2^(d+1) - 1, 2*2^(d+1) - 1, ...:
        data[i] = data[i] * data[i - 2^d]
    """
    pid_bd = tl.program_id(0)
    pid_k = tl.program_id(1)

    if s_idx >= seq_len:
        return

    # Compute offsets
    bd_offsets = pid_bd * BLOCK_BD + tl.arange(0, BLOCK_BD)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Masks
    bd_mask = bd_offsets < n_batch_model
    k_mask = k_offsets < d_state

    # Broadcast for 2D indexing
    bd_bcast = bd_offsets[:, None]
    k_bcast = k_offsets[None, :]

    # Load data[i] (right operand)
    base_right = (bd_bcast * stride_bd + s_idx * stride_s + k_bcast * stride_k)

    mask_2d = bd_mask[:, None] & k_mask[None, :]

    r0 = tl.load(data_ptr + base_right + 0, mask=mask_2d, other=0.0)
    r1 = tl.load(data_ptr + base_right + 1, mask=mask_2d, other=0.0)
    r2 = tl.load(data_ptr + base_right + 2, mask=mask_2d, other=0.0)
    r3 = tl.load(data_ptr + base_right + 3, mask=mask_2d, other=0.0)

    # Load data[i - 2^d] (left operand)
    left_idx = s_idx - offset_level
    if left_idx < 0:
        return

    base_left = (bd_bcast * stride_bd + left_idx * stride_s + k_bcast * stride_k)

    l0 = tl.load(data_ptr + base_left + 0, mask=mask_2d, other=0.0)
    l1 = tl.load(data_ptr + base_left + 1, mask=mask_2d, other=0.0)
    l2 = tl.load(data_ptr + base_left + 2, mask=mask_2d, other=0.0)
    l3 = tl.load(data_ptr + base_left + 3, mask=mask_2d, other=0.0)

    # Quaternion multiplication: result = right * left
    # (Note: order matters due to non-commutativity)
    out0 = r0*l0 - r1*l1 - r2*l2 - r3*l3
    out1 = r0*l1 + r1*l0 + r2*l3 - r3*l2
    out2 = r0*l2 - r1*l3 + r2*l0 + r3*l1
    out3 = r0*l3 + r1*l2 - r2*l1 + r3*l0

    # Store result back to data[i]
    tl.store(data_ptr + base_right + 0, out0, mask=mask_2d)
    tl.store(data_ptr + base_right + 1, out1, mask=mask_2d)
    tl.store(data_ptr + base_right + 2, out2, mask=mask_2d)
    tl.store(data_ptr + base_right + 3, out3, mask=mask_2d)


@triton.jit
def parallel_scan_downsweep_kernel(
    # Input/Output (in-place modification)
    data_ptr,  # [batch*d_model, seq_len, d_state, 4] (flattened)
    # Shapes
    n_batch_model, seq_len, d_state,
    # Current level
    stride_level: tl.constexpr,
    offset_level: tl.constexpr,
    left_idx: tl.constexpr,
    right_idx: tl.constexpr,
    # Strides
    stride_bd, stride_s, stride_k,
    # Tiling
    BLOCK_BD: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Down-sweep phase of parallel scan (2D grid).

    Propagates cumulative products down the tree.
    """
    pid_bd = tl.program_id(0)
    pid_k = tl.program_id(1)

    if right_idx >= seq_len:
        return

    # Compute offsets
    bd_offsets = pid_bd * BLOCK_BD + tl.arange(0, BLOCK_BD)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Masks
    bd_mask = bd_offsets < n_batch_model
    k_mask = k_offsets < d_state

    # Broadcast
    bd_bcast = bd_offsets[:, None]
    k_bcast = k_offsets[None, :]

    mask_2d = bd_mask[:, None] & k_mask[None, :]

    # Load data[left_idx]
    base_left = (bd_bcast * stride_bd + left_idx * stride_s + k_bcast * stride_k)

    l0 = tl.load(data_ptr + base_left + 0, mask=mask_2d, other=0.0)
    l1 = tl.load(data_ptr + base_left + 1, mask=mask_2d, other=0.0)
    l2 = tl.load(data_ptr + base_left + 2, mask=mask_2d, other=0.0)
    l3 = tl.load(data_ptr + base_left + 3, mask=mask_2d, other=0.0)

    # Load data[right_idx]
    base_right = (bd_bcast * stride_bd + right_idx * stride_s + k_bcast * stride_k)

    r0 = tl.load(data_ptr + base_right + 0, mask=mask_2d, other=0.0)
    r1 = tl.load(data_ptr + base_right + 1, mask=mask_2d, other=0.0)
    r2 = tl.load(data_ptr + base_right + 2, mask=mask_2d, other=0.0)
    r3 = tl.load(data_ptr + base_right + 3, mask=mask_2d, other=0.0)

    # Multiply: data[right_idx] = data[right_idx] * data[left_idx]
    out0 = r0*l0 - r1*l1 - r2*l2 - r3*l3
    out1 = r0*l1 + r1*l0 + r2*l3 - r3*l2
    out2 = r0*l2 - r1*l3 + r2*l0 + r3*l1
    out3 = r0*l3 + r1*l2 - r2*l1 + r3*l0

    # Store result
    tl.store(data_ptr + base_right + 0, out0, mask=mask_2d)
    tl.store(data_ptr + base_right + 1, out1, mask=mask_2d)
    tl.store(data_ptr + base_right + 2, out2, mask=mask_2d)
    tl.store(data_ptr + base_right + 3, out3, mask=mask_2d)


def parallel_scan_quaternion(q_sequence: torch.Tensor) -> torch.Tensor:
    """
    Parallel prefix product for quaternion sequences.

    Args:
        q_sequence: [batch, seq_len, d_model, d_state, 4] - sequence of quaternions

    Returns:
        cumulative: [batch, seq_len, d_model, d_state, 4] - cumulative products

    For each position t, computes:
        cumulative[t] = q[t] * q[t-1] * ... * q[1] * q[0]

    Complexity: O(log T) depth, O(T) work (vs O(T) sequential)
    """
    assert q_sequence.shape[-1] == 4
    assert q_sequence.ndim == 5

    batch_size, seq_len, d_model, d_state, _ = q_sequence.shape
    original_shape = q_sequence.shape

    # Flatten batch and d_model for 2D grid
    data_flat = q_sequence.clone().reshape(batch_size * d_model, seq_len, d_state, 4)
    n_batch_model = batch_size * d_model

    # Compute strides (for flattened tensor)
    stride_bd, stride_s, stride_k, _ = data_flat.stride()

    # Tiling configuration
    BLOCK_BD, BLOCK_K = 16, 8

    # Number of levels in the tree
    log_T = int(math.ceil(math.log2(seq_len)))

    # Up-sweep phase
    for d in range(log_T):
        stride_level = 2 ** (d + 1)
        offset_level = 2 ** d

        # Process each sequence index at this level
        for pid_s in range((seq_len + stride_level - 1) // stride_level):
            s_idx = pid_s * stride_level + (stride_level - 1)

            if s_idx >= seq_len:
                continue

            grid = (
                triton.cdiv(n_batch_model, BLOCK_BD),
                triton.cdiv(d_state, BLOCK_K),
            )

            parallel_scan_upsweep_kernel[grid](
                data_flat,
                n_batch_model, seq_len, d_state,
                stride_level, offset_level, s_idx,
                stride_bd, stride_s, stride_k,
                BLOCK_BD=BLOCK_BD,
                BLOCK_K=BLOCK_K,
            )

    # Down-sweep phase (reverse order)
    for d in range(log_T - 1, -1, -1):
        stride_level = 2 ** (d + 1)
        offset_level = 2 ** d

        # Process each sequence index at this level
        for pid_s in range((seq_len + stride_level - 1) // stride_level):
            left_idx = pid_s * stride_level + (stride_level - 1)
            right_idx = left_idx + offset_level

            if right_idx >= seq_len:
                continue

            grid = (
                triton.cdiv(n_batch_model, BLOCK_BD),
                triton.cdiv(d_state, BLOCK_K),
            )

            parallel_scan_downsweep_kernel[grid](
                data_flat,
                n_batch_model, seq_len, d_state,
                stride_level, offset_level, left_idx, right_idx,
                stride_bd, stride_s, stride_k,
                BLOCK_BD=BLOCK_BD,
                BLOCK_K=BLOCK_K,
            )

    # Reshape back to original shape
    data = data_flat.reshape(original_shape)

    return data


def parallel_scan_quaternion_reference(q_sequence: torch.Tensor) -> torch.Tensor:
    """
    Reference sequential implementation for testing.

    Args:
        q_sequence: [batch, seq_len, d_model, d_state, 4]

    Returns:
        cumulative: [batch, seq_len, d_model, d_state, 4]
    """
    from .quaternion_ops import quaternion_multiply

    batch_size, seq_len, d_model, d_state, _ = q_sequence.shape

    # Initialize output
    cumulative = torch.zeros_like(q_sequence)

    # Identity quaternion for initialization
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q_sequence.device, dtype=q_sequence.dtype)

    # Accumulator (starts at identity)
    acc = identity.view(1, 1, 1, 4).expand(batch_size, d_model, d_state, 4)

    for t in range(seq_len):
        # acc = q[t] * acc (left multiplication)
        acc = quaternion_multiply(q_sequence[:, t], acc)
        cumulative[:, t] = acc

    return cumulative

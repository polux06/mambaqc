"""
Quaternion utility functions.

Provides conversions, visualizations, and helper functions for quaternions.
"""

import torch
import numpy as np


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        q: [..., 4] quaternion tensor (a, b, c, d)

    Returns:
        R: [..., 3, 3] rotation matrix
    """
    # Normalize quaternion
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

    a, b, c, d = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Rotation matrix formula
    R = torch.stack([
        torch.stack([
            1 - 2*(c**2 + d**2), 2*(b*c - a*d), 2*(b*d + a*c)
        ], dim=-1),
        torch.stack([
            2*(b*c + a*d), 1 - 2*(b**2 + d**2), 2*(c*d - a*b)
        ], dim=-1),
        torch.stack([
            2*(b*d - a*c), 2*(c*d + a*b), 1 - 2*(b**2 + c**2)
        ], dim=-1),
    ], dim=-2)

    return R


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to quaternion.

    Args:
        R: [..., 3, 3] rotation matrix

    Returns:
        q: [..., 4] quaternion (a, b, c, d)
    """
    # Shepperd's method (numerically stable)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    q = torch.zeros((*R.shape[:-2], 4), device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    mask1 = trace > 0
    s = torch.sqrt(trace[mask1] + 1.0) * 2
    q[mask1, 0] = 0.25 * s
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

    # Case 2: R[0,0] is largest diagonal
    mask2 = (~mask1) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
    q[mask2, 1] = 0.25 * s
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

    # Case 3: R[1,1] is largest diagonal
    mask3 = (~mask1) & (~mask2) & (R[..., 1, 1] > R[..., 2, 2])
    s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
    q[mask3, 2] = 0.25 * s
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

    # Case 4: R[2,2] is largest diagonal
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
    q[mask4, 3] = 0.25 * s

    return q


def quaternion_rotation(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector v by quaternion q.

    Uses formula: v' = q * v * q̄
    where v is embedded as (0, v_x, v_y, v_z)

    Args:
        q: [..., 4] quaternion
        v: [..., 3] vector

    Returns:
        v_rotated: [..., 3] rotated vector
    """
    from mambaqc.kernels.quaternion_ops import quaternion_multiply, quaternion_conjugate

    # Embed v as pure quaternion (0, v_x, v_y, v_z)
    v_quat = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)

    # q * v
    qv = quaternion_multiply(q, v_quat)

    # q̄
    q_conj = quaternion_conjugate(q)

    # q * v * q̄
    qvq = quaternion_multiply(qv, q_conj)

    # Extract imaginary part (rotated vector)
    v_rotated = qvq[..., 1:]

    return v_rotated


def visualize_quaternion_dynamics(q_sequence: torch.Tensor, save_path: str = None):
    """
    Visualize quaternion sequence as rotation trajectories.

    Args:
        q_sequence: [seq_len, 4] quaternion sequence
        save_path: Path to save plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available for visualization")
        return

    q_np = q_sequence.detach().cpu().numpy()

    # Convert to rotation matrices
    R_sequence = quaternion_to_matrix(q_sequence).detach().cpu().numpy()

    # Extract x-axis direction from each rotation
    x_dirs = R_sequence[:, :, 0]  # [seq_len, 3]

    # Plot trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_dirs[:, 0], x_dirs[:, 1], x_dirs[:, 2], 'b-', alpha=0.6)
    ax.scatter(x_dirs[0, 0], x_dirs[0, 1], x_dirs[0, 2], c='g', s=100, label='Start')
    ax.scatter(x_dirs[-1, 0], x_dirs[-1, 1], x_dirs[-1, 2], c='r', s=100, label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Quaternion Dynamics Trajectory')
    ax.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

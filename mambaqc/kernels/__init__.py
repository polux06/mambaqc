"""
Optimized Triton kernels for quaternion operations.

All kernels are designed to:
1. Maximize shared memory usage (stay in SM cache)
2. Exploit Tensor Cores for 4x4 matrix operations
3. Minimize HBM access through fusion
"""

from .quaternion_ops import (
    quaternion_multiply,
    quaternion_multiply_batched,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_norm,
)
from .cayley_transform import cayley_discretization_fused
from .parallel_scan import parallel_scan_quaternion
from .quaternion_norm import quaternion_layer_norm_fused
from .ssm_fused import ssm_step_fused, ssm_forward_fused

__all__ = [
    "quaternion_multiply",
    "quaternion_multiply_batched",
    "quaternion_conjugate",
    "quaternion_inverse",
    "quaternion_norm",
    "cayley_discretization_fused",
    "parallel_scan_quaternion",
    "quaternion_layer_norm_fused",
    "ssm_step_fused",
    "ssm_forward_fused",
]

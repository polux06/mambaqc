"""
Quaternion layer normalization module.

Re-exports the kernel-based implementation from kernels.quaternion_norm.
"""

from mambaqc.kernels.quaternion_norm import (
    QuaternionLayerNorm,
    quaternion_layer_norm_fused,
    quaternion_layer_norm_reference,
)

__all__ = [
    "QuaternionLayerNorm",
    "quaternion_layer_norm_fused",
    "quaternion_layer_norm_reference",
]

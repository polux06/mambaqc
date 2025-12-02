"""
Quaternion Mamba-2: A Quaternion State Space Dual Model with Cayley Dynamics

This package implements a non-commutative extension of Mamba-2 using quaternion algebra.
"""

__version__ = "0.1.0"

from .models.quaternion_mamba2 import QuaternionMamba2
from .models.quaternion_mamba2_block import QuaternionMamba2Block

__all__ = ["QuaternionMamba2", "QuaternionMamba2Block"]

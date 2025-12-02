"""
Kernels Triton optimisés pour Quaternion Mamba-2
================================================

Ce package fournit des implementations GPU hautement optimisées pour les
opérations quaternioniques nécessaires au modèle Quaternion Mamba-2.

Architecture:
- Opérations de base: multiplication, inverse, norme (quaternion_ops.py)
- SSM et Cayley: discrétisation, récurrence, scan (cayley_ssm.py)

Optimisations:
- Tiling pour garder les données en shared memory
- Fusion des opérations pour minimiser les accès HBM
- Utilisation des tensor cores via opérations matricielles
- Vectorisation et coalesced memory access
"""

from .quaternion_ops import (
    quat_mul_triton,
    quat_inv_triton,
    quat_norm_triton,
)

from .cayley_ssm import (
    cayley_discretization_triton,
    ssm_step_triton,
    parallel_scan_triton,
    full_ssm_triton,
)

__all__ = [
    # Opérations quaternioniques de base
    'quat_mul_triton',
    'quat_inv_triton',
    'quat_norm_triton',
    # SSM et Cayley
    'cayley_discretization_triton',
    'ssm_step_triton',
    'parallel_scan_triton',
    'full_ssm_triton',
]

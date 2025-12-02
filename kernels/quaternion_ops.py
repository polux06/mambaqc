"""
Kernels Triton optimisés pour opérations quaternioniques
=========================================================

Ces kernels implémentent les opérations quaternioniques de base avec:
- Tiling pour garder les données dans la shared memory (SRAM)
- Utilisation des Tensor Cores via des opérations matricielles 4x4
- Fusion des opérations pour minimiser les accès mémoire HBM
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# CONSTANTES POUR OPTIMISATION
# =============================================================================

# Tailles de blocs optimales pour différentes architectures
# RTX 30/40/50 series ont 128KB de shared memory par SM
BLOCK_SIZE_M = 128  # Tuned for L1 cache
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32


# =============================================================================
# 1. MULTIPLICATION QUATERNIONIQUE FUSIONNÉE (TILED + TENSOR CORES)
# =============================================================================

@triton.jit
def quat_mul_fused_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, out_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Multiplication quaternionique fusionnée et tilée.

    Implémente: out = a ⊗ b où ⊗ est le produit de Hamilton

    Chaque quaternion a 4 composantes: (r, i, j, k)
    La multiplication de Hamilton est:
      (a0 + a1*i + a2*j + a3*k) * (b0 + b1*i + b2*j + b3*k)

    Architecture:
    - Les données sont chargées par tiles dans SRAM
    - Les calculs utilisent les registres et tensor cores
    - Les résultats sont écrits en coalesced writes
    """
    # Program ID
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets pour les blocs
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers pour chaque composante
    # Les quaternions sont stockés avec les 4 composantes consécutives: [r, i, j, k]
    a_r_ptrs = a_ptr + (offs_m[:, None] * stride_am + (offs_k[None, :] * 4 + 0) * stride_ak)
    a_i_ptrs = a_ptr + (offs_m[:, None] * stride_am + (offs_k[None, :] * 4 + 1) * stride_ak)
    a_j_ptrs = a_ptr + (offs_m[:, None] * stride_am + (offs_k[None, :] * 4 + 2) * stride_ak)
    a_k_ptrs = a_ptr + (offs_m[:, None] * stride_am + (offs_k[None, :] * 4 + 3) * stride_ak)

    b_r_ptrs = b_ptr + ((offs_k[:, None] * 4 + 0) * stride_bk + offs_n[None, :] * stride_bn)
    b_i_ptrs = b_ptr + ((offs_k[:, None] * 4 + 1) * stride_bk + offs_n[None, :] * stride_bn)
    b_j_ptrs = b_ptr + ((offs_k[:, None] * 4 + 2) * stride_bk + offs_n[None, :] * stride_bn)
    b_k_ptrs = b_ptr + ((offs_k[:, None] * 4 + 3) * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulateurs pour les 4 composantes du résultat
    # Initialisés à zéro
    acc_r = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_j = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_k = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension (tiled)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Charger les tiles de A et B dans SRAM
        # Ces loads sont coalescés pour maximiser la bande passante
        a_r = tl.load(a_r_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        a_i = tl.load(a_i_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        a_j = tl.load(a_j_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        a_k = tl.load(a_k_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)

        b_r = tl.load(b_r_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        b_i = tl.load(b_i_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        b_j = tl.load(b_j_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        b_k = tl.load(b_k_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # Multiplication de Hamilton fusionnée
        # out_r = a_r*b_r - a_i*b_i - a_j*b_j - a_k*b_k
        # out_i = a_r*b_i + a_i*b_r + a_j*b_k - a_k*b_j
        # out_j = a_r*b_j - a_i*b_k + a_j*b_r + a_k*b_i
        # out_k = a_r*b_k + a_i*b_j - a_j*b_i + a_k*b_r

        # Ces dot products peuvent utiliser les tensor cores si disponibles
        acc_r += tl.dot(a_r, b_r) - tl.dot(a_i, b_i) - tl.dot(a_j, b_j) - tl.dot(a_k, b_k)
        acc_i += tl.dot(a_r, b_i) + tl.dot(a_i, b_r) + tl.dot(a_j, b_k) - tl.dot(a_k, b_j)
        acc_j += tl.dot(a_r, b_j) - tl.dot(a_i, b_k) + tl.dot(a_j, b_r) + tl.dot(a_k, b_i)
        acc_k += tl.dot(a_r, b_k) + tl.dot(a_i, b_j) - tl.dot(a_j, b_i) + tl.dot(a_k, b_r)

        # Avancer les pointers
        a_r_ptrs += BLOCK_SIZE_K * 4 * stride_ak
        a_i_ptrs += BLOCK_SIZE_K * 4 * stride_ak
        a_j_ptrs += BLOCK_SIZE_K * 4 * stride_ak
        a_k_ptrs += BLOCK_SIZE_K * 4 * stride_ak

        b_r_ptrs += BLOCK_SIZE_K * 4 * stride_bk
        b_i_ptrs += BLOCK_SIZE_K * 4 * stride_bk
        b_j_ptrs += BLOCK_SIZE_K * 4 * stride_bk
        b_k_ptrs += BLOCK_SIZE_K * 4 * stride_bk

    # Écrire les résultats en HBM (coalesced stores)
    offs_m_out = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_out = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)

    out_r_ptrs = out_ptr + (offs_m_out[:, None] * stride_om + (offs_n_out[None, :] * 4 + 0) * stride_on)
    out_i_ptrs = out_ptr + (offs_m_out[:, None] * stride_om + (offs_n_out[None, :] * 4 + 1) * stride_on)
    out_j_ptrs = out_ptr + (offs_m_out[:, None] * stride_om + (offs_n_out[None, :] * 4 + 2) * stride_on)
    out_k_ptrs = out_ptr + (offs_m_out[:, None] * stride_om + (offs_n_out[None, :] * 4 + 3) * stride_on)

    tl.store(out_r_ptrs, acc_r, mask=mask)
    tl.store(out_i_ptrs, acc_i, mask=mask)
    tl.store(out_j_ptrs, acc_j, mask=mask)
    tl.store(out_k_ptrs, acc_k, mask=mask)


def quat_mul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Interface Python pour la multiplication quaternionique.

    Args:
        a: Tensor de shape (..., 4) - quaternions
        b: Tensor de shape (..., 4) - quaternions

    Returns:
        Tensor de shape (..., 4) - produit a ⊗ b
    """
    # Reshape pour matrix multiplication
    original_shape = a.shape
    a_flat = a.reshape(-1, 4)
    b_flat = b.reshape(-1, 4)

    M = a_flat.shape[0]
    N = 1
    K = 1

    # Allouer la sortie
    out = torch.empty((M, 4), dtype=a.dtype, device=a.device)

    # Grille de lancement
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Lancer le kernel
    quat_mul_fused_kernel[grid](
        a_flat, b_flat, out,
        M, N, K,
        a_flat.stride(0), a_flat.stride(1),
        b_flat.stride(0), b_flat.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=1,
    )

    return out.reshape(original_shape)


# =============================================================================
# 2. INVERSE QUATERNIONIQUE FUSIONNÉ
# =============================================================================

@triton.jit
def quat_inv_fused_kernel(
    q_ptr, out_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Calcule l'inverse quaternionique: q^-1 = conj(q) / ||q||^2

    Fusionné en un seul kernel pour minimiser les accès mémoire.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # Charger les 4 composantes (vectorized load)
    q_r = tl.load(q_ptr + offsets * 4 + 0, mask=mask, other=0.0)
    q_i = tl.load(q_ptr + offsets * 4 + 1, mask=mask, other=0.0)
    q_j = tl.load(q_ptr + offsets * 4 + 2, mask=mask, other=0.0)
    q_k = tl.load(q_ptr + offsets * 4 + 3, mask=mask, other=0.0)

    # Calcul de la norme au carré
    norm_sq = q_r * q_r + q_i * q_i + q_j * q_j + q_k * q_k + eps

    # Inverse = conjugué / norme_sq
    inv_norm_sq = 1.0 / norm_sq

    out_r = q_r * inv_norm_sq
    out_i = -q_i * inv_norm_sq
    out_j = -q_j * inv_norm_sq
    out_k = -q_k * inv_norm_sq

    # Store vectorisé
    tl.store(out_ptr + offsets * 4 + 0, out_r, mask=mask)
    tl.store(out_ptr + offsets * 4 + 1, out_i, mask=mask)
    tl.store(out_ptr + offsets * 4 + 2, out_j, mask=mask)
    tl.store(out_ptr + offsets * 4 + 3, out_k, mask=mask)


def quat_inv_triton(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Calcule l'inverse quaternionique via Triton.

    Args:
        q: Tensor de shape (..., 4)
        eps: Epsilon pour la stabilité numérique

    Returns:
        Tensor de shape (..., 4) - inverse de q
    """
    original_shape = q.shape
    q_flat = q.reshape(-1, 4).contiguous()
    n_elements = q_flat.shape[0]

    out = torch.empty_like(q_flat)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    quat_inv_fused_kernel[grid](
        q_flat, out,
        n_elements,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(original_shape)


# =============================================================================
# 3. NORME QUATERNIONIQUE FUSIONNÉE
# =============================================================================

@triton.jit
def quat_norm_kernel(
    q_ptr, norm_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Calcule la norme euclidienne des quaternions.

    ||q|| = sqrt(r^2 + i^2 + j^2 + k^2)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # Vectorized load
    q_r = tl.load(q_ptr + offsets * 4 + 0, mask=mask, other=0.0)
    q_i = tl.load(q_ptr + offsets * 4 + 1, mask=mask, other=0.0)
    q_j = tl.load(q_ptr + offsets * 4 + 2, mask=mask, other=0.0)
    q_k = tl.load(q_ptr + offsets * 4 + 3, mask=mask, other=0.0)

    # Norme
    norm = tl.sqrt(q_r * q_r + q_i * q_i + q_j * q_j + q_k * q_k)

    tl.store(norm_ptr + offsets, norm, mask=mask)


def quat_norm_triton(q: torch.Tensor) -> torch.Tensor:
    """
    Calcule la norme quaternionique.

    Args:
        q: Tensor de shape (..., 4)

    Returns:
        Tensor de shape (...) - normes
    """
    original_shape = q.shape[:-1]
    q_flat = q.reshape(-1, 4).contiguous()
    n_elements = q_flat.shape[0]

    norm = torch.empty(n_elements, dtype=q.dtype, device=q.device)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    quat_norm_kernel[grid](
        q_flat, norm,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return norm.reshape(original_shape)


# =============================================================================
# 4. NORMALISATION GÉOMÉTRIQUE QUATERNIONIQUE
# =============================================================================

@triton.jit
def quat_layer_norm_kernel(
    q_ptr, out_ptr, gamma_ptr, beta_ptr,
    M, D,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Layer normalization géométrique pour quaternions.

    Normalise les NORMES tout en préservant les DIRECTIONS.
    """
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M

    # Phase 1: Calcul des normes
    norm_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    norm_sq_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for d in range(0, D, BLOCK_SIZE_D):
        offs_d = d + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < D
        mask = mask_m[:, None] & mask_d[None, :]

        # Charger quaternion
        q_r = tl.load(q_ptr + (offs_m[:, None] * D * 4 + offs_d[None, :] * 4 + 0), mask=mask, other=0.0)
        q_i = tl.load(q_ptr + (offs_m[:, None] * D * 4 + offs_d[None, :] * 4 + 1), mask=mask, other=0.0)
        q_j = tl.load(q_ptr + (offs_m[:, None] * D * 4 + offs_d[None, :] * 4 + 2), mask=mask, other=0.0)
        q_k = tl.load(q_ptr + (offs_m[:, None] * D * 4 + offs_d[None, :] * 4 + 3), mask=mask, other=0.0)

        # Norme de chaque quaternion
        norm = tl.sqrt(q_r*q_r + q_i*q_i + q_j*q_j + q_k*q_k + eps)

        # Accumuler pour les statistiques
        norm_sum += tl.sum(norm, axis=1)
        norm_sq_sum += tl.sum(norm * norm, axis=1)

    # Statistiques
    mean_norm = norm_sum / D
    var_norm = norm_sq_sum / D - mean_norm * mean_norm
    std_norm = tl.sqrt(var_norm + eps)

    # Phase 2: Normalisation et écriture
    for d in range(0, D, BLOCK_SIZE_D):
        offs_d = d + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < D
        mask = mask_m[:, None] & mask_d[None, :]

        # Recharger quaternion
        q_r = tl.load(q_ptr + (offs_m[:, None] * D * 4 + offs_d[None, :] * 4 + 0), mask=mask, other=0.0)
        q_i = tl.load(q_ptr + (offs_m[:, None] * D * 4 + offs_d[None, :] * 4 + 1), mask=mask, other=0.0)
        q_j = tl.load(q_ptr + (offs_m[:, None] * D * 4 + offs_d[None, :] * 4 + 2), mask=mask, other=0.0)
        q_k = tl.load(q_ptr + (offs_m[:, None] * D * 4 + offs_d[None, :] * 4 + 3), mask=mask, other=0.0)

        # Norme
        norm = tl.sqrt(q_r*q_r + q_i*q_i + q_j*q_j + q_k*q_k + eps)

        # Direction unitaire
        dir_r = q_r / norm
        dir_i = q_i / norm
        dir_j = q_j / norm
        dir_k = q_k / norm

        # Norme normalisée
        norm_normalized = (norm - mean_norm[:, None]) / std_norm[:, None]

        # Charger gamma et beta
        gamma = tl.load(gamma_ptr + offs_d, mask=mask_d, other=1.0)
        beta_r = tl.load(beta_ptr + offs_d * 4 + 0, mask=mask_d, other=0.0)
        beta_i = tl.load(beta_ptr + offs_d * 4 + 1, mask=mask_d, other=0.0)
        beta_j = tl.load(beta_ptr + offs_d * 4 + 2, mask=mask_d, other=0.0)
        beta_k = tl.load(beta_ptr + offs_d * 4 + 3, mask=mask_d, other=0.0)

        # Reconstruction
        out_r = gamma[None, :] * norm_normalized * dir_r + beta_r[None, :]
        out_i = gamma[None, :] * norm_normalized * dir_i + beta_i[None, :]
        out_j = gamma[None, :] * norm_normalized * dir_j + beta_j[None, :]
        out_k = gamma[None, :] * norm_normalized * dir_k + beta_k[None, :]

        # Store
        tl.store(out_ptr + (offs_m[:, None] * D * 4 + offs_d[None, :] * 4 + 0), out_r, mask=mask)
        tl.store(out_ptr + (offs_m[:, None] * D * 4 + offs_d[None, :] * 4 + 1), out_i, mask=mask)
        tl.store(out_ptr + (offs_m[:, None] * D * 4 + offs_d[None, :] * 4 + 2), out_j, mask=mask)
        tl.store(out_ptr + (offs_m[:, None] * D * 4 + offs_d[None, :] * 4 + 3), out_k, mask=mask)

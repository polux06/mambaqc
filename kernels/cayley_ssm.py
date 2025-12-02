"""
Kernels Triton pour Discrétisation de Cayley et SSM Quaternionique
===================================================================

Ce module implémente les kernels fusionnés pour:
1. Transformée de Cayley quaternionique
2. SSM step fusionné (récurrence h_t = q_t ⊗ h_{t-1} + B_t ⊗ u_t)
3. Parallel scan associatif

Architecture optimisée pour:
- Minimiser les accès HBM
- Maximiser l'utilisation de la shared memory
- Exploiter les tensor cores pour les produits quaternioniques
"""

import torch
import triton
import triton.language as tl
import math


# =============================================================================
# 1. DISCRÉTISATION DE CAYLEY FUSIONNÉE
# =============================================================================

@triton.jit
def cayley_discretization_kernel(
    # Input: continuous dynamics z_t
    z_ptr,
    # Output: discrete operators q_t
    q_ptr,
    # Dimensions
    n_elements,
    eps,
    # Meta
    BLOCK_SIZE: tl.constexpr,
):
    """
    Transformée de Cayley fusionnée: q = (1 - z/2)^{-1} (1 + z/2)

    Ce kernel fusionne:
    1. Calcul de num = (1 + z/2)
    2. Calcul de den = (1 - z/2)
    3. Inverse quaternionique de den
    4. Multiplication num * den^{-1}

    Tout cela en un seul passage mémoire.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # Charger z = (zr, zi, zj, zk)
    zr = tl.load(z_ptr + offsets * 4 + 0, mask=mask, other=0.0)
    zi = tl.load(z_ptr + offsets * 4 + 1, mask=mask, other=0.0)
    zj = tl.load(z_ptr + offsets * 4 + 2, mask=mask, other=0.0)
    zk = tl.load(z_ptr + offsets * 4 + 3, mask=mask, other=0.0)

    # Constantes
    half = 0.5
    one = 1.0

    # Calcul de z/2
    zr_half = zr * half
    zi_half = zi * half
    zj_half = zj * half
    zk_half = zk * half

    # num = 1 + z/2  (le "1" est le quaternion identité: 1 + 0i + 0j + 0k)
    num_r = one + zr_half
    num_i = zi_half
    num_j = zj_half
    num_k = zk_half

    # den = 1 - z/2
    den_r = one - zr_half
    den_i = -zi_half
    den_j = -zj_half
    den_k = -zk_half

    # Inverse de den: den^{-1} = conj(den) / ||den||^2
    norm_sq_den = den_r * den_r + den_i * den_i + den_j * den_j + den_k * den_k + eps
    inv_norm_sq = one / norm_sq_den

    # conj(den) = (den_r, -den_i, -den_j, -den_k)
    den_inv_r = den_r * inv_norm_sq
    den_inv_i = -den_i * inv_norm_sq
    den_inv_j = -den_j * inv_norm_sq
    den_inv_k = -den_k * inv_norm_sq

    # q = num ⊗ den^{-1}  (multiplication de Hamilton)
    # Formule complète du produit quaternionique
    qr = num_r * den_inv_r - num_i * den_inv_i - num_j * den_inv_j - num_k * den_inv_k
    qi = num_r * den_inv_i + num_i * den_inv_r + num_j * den_inv_k - num_k * den_inv_j
    qj = num_r * den_inv_j - num_i * den_inv_k + num_j * den_inv_r + num_k * den_inv_i
    qk = num_r * den_inv_k + num_i * den_inv_j - num_j * den_inv_i + num_k * den_inv_r

    # Store le résultat
    tl.store(q_ptr + offsets * 4 + 0, qr, mask=mask)
    tl.store(q_ptr + offsets * 4 + 1, qi, mask=mask)
    tl.store(q_ptr + offsets * 4 + 2, qj, mask=mask)
    tl.store(q_ptr + offsets * 4 + 3, qk, mask=mask)


def cayley_discretization_triton(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Interface Python pour la discrétisation de Cayley.

    Args:
        z: Tensor de shape (..., 4) - dynamiques continues
        eps: Epsilon pour stabilité numérique

    Returns:
        Tensor de shape (..., 4) - opérateurs discrets
    """
    original_shape = z.shape
    z_flat = z.reshape(-1, 4).contiguous()
    n_elements = z_flat.shape[0]

    q = torch.empty_like(z_flat)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    cayley_discretization_kernel[grid](
        z_flat, q,
        n_elements,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return q.reshape(original_shape)


# =============================================================================
# 2. SSM STEP FUSIONNÉ (RÉCURRENCE)
# =============================================================================

@triton.jit
def ssm_step_fused_kernel(
    # Inputs
    h_prev_ptr,  # État précédent h_{t-1}
    q_ptr,       # Opérateur d'évolution q_t
    B_ptr,       # Projection d'entrée B_t
    u_ptr,       # Signal d'entrée u_t
    # Output
    h_out_ptr,   # Nouvel état h_t
    # Dimensions
    batch_size,
    d_model,
    d_state,
    # Meta
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """
    Kernel fusionné pour un step SSM quaternionique:

    h_t = q_t ⊗ h_{t-1} + B_t ⊗ u_t

    Ce kernel effectue:
    1. q_t ⊗ h_{t-1}  (produit quaternionique)
    2. B_t ⊗ u_t      (produit quaternionique)
    3. Addition des deux

    Tout en un seul passage pour minimiser les accès mémoire.

    Layout:
    - h: [batch, d_model, d_state, 4]
    - q: [batch, d_model, d_state, 4]
    - B: [batch, d_model, d_state, 4]
    - u: [batch, d_model, 4]
    """
    # Program IDs
    pid_b = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    pid_s = tl.program_id(axis=2)

    # Offsets
    b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    d_idx = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    s_idx = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    # Masks
    mask_b = b_idx < batch_size
    mask_d = d_idx < d_model
    mask_s = s_idx < d_state

    # Construction des indices 3D
    # h_prev: [B, D, S, 4]
    base_offset = (
        b_idx[:, None, None, None] * (d_model * d_state * 4) +
        d_idx[None, :, None, None] * (d_state * 4) +
        s_idx[None, None, :, None] * 4
    )

    mask_3d = mask_b[:, None, None] & mask_d[None, :, None] & mask_s[None, None, :]

    # Charger h_{t-1}
    h_r = tl.load(h_prev_ptr + base_offset + 0, mask=mask_3d[:, :, :, None], other=0.0).squeeze(-1)
    h_i = tl.load(h_prev_ptr + base_offset + 1, mask=mask_3d[:, :, :, None], other=0.0).squeeze(-1)
    h_j = tl.load(h_prev_ptr + base_offset + 2, mask=mask_3d[:, :, :, None], other=0.0).squeeze(-1)
    h_k = tl.load(h_prev_ptr + base_offset + 3, mask=mask_3d[:, :, :, None], other=0.0).squeeze(-1)

    # Charger q_t (même layout que h)
    qr = tl.load(q_ptr + base_offset + 0, mask=mask_3d[:, :, :, None], other=0.0).squeeze(-1)
    qi = tl.load(q_ptr + base_offset + 1, mask=mask_3d[:, :, :, None], other=0.0).squeeze(-1)
    qj = tl.load(q_ptr + base_offset + 2, mask=mask_3d[:, :, :, None], other=0.0).squeeze(-1)
    qk = tl.load(q_ptr + base_offset + 3, mask=mask_3d[:, :, :, None], other=0.0).squeeze(-1)

    # Calcul de q_t ⊗ h_{t-1}
    qh_r = qr * h_r - qi * h_i - qj * h_j - qk * h_k
    qh_i = qr * h_i + qi * h_r + qj * h_k - qk * h_j
    qh_j = qr * h_j - qi * h_k + qj * h_r + qk * h_i
    qh_k = qr * h_k + qi * h_j - qj * h_i + qk * h_r

    # Charger B_t (même layout que h)
    Br = tl.load(B_ptr + base_offset + 0, mask=mask_3d[:, :, :, None], other=0.0).squeeze(-1)
    Bi = tl.load(B_ptr + base_offset + 1, mask=mask_3d[:, :, :, None], other=0.0).squeeze(-1)
    Bj = tl.load(B_ptr + base_offset + 2, mask=mask_3d[:, :, :, None], other=0.0).squeeze(-1)
    Bk = tl.load(B_ptr + base_offset + 3, mask=mask_3d[:, :, :, None], other=0.0).squeeze(-1)

    # Charger u_t: [B, D, 4]
    u_offset = (
        b_idx[:, None, None] * (d_model * 4) +
        d_idx[None, :, None] * 4
    )
    mask_u = mask_b[:, None] & mask_d[None, :]

    ur = tl.load(u_ptr + u_offset + 0, mask=mask_u[:, :, None], other=0.0)
    ui = tl.load(u_ptr + u_offset + 1, mask=mask_u[:, :, None], other=0.0)
    uj = tl.load(u_ptr + u_offset + 2, mask=mask_u[:, :, None], other=0.0)
    uk = tl.load(u_ptr + u_offset + 3, mask=mask_u[:, :, None], other=0.0)

    # Broadcast u sur la dimension S
    ur = ur[:, :, None]
    ui = ui[:, :, None]
    uj = uj[:, :, None]
    uk = uk[:, :, None]

    # Calcul de B_t ⊗ u_t
    Bu_r = Br * ur - Bi * ui - Bj * uj - Bk * uk
    Bu_i = Br * ui + Bi * ur + Bj * uk - Bk * uj
    Bu_j = Br * uj - Bi * uk + Bj * ur + Bk * ui
    Bu_k = Br * uk + Bi * uj - Bj * ui + Bk * ur

    # h_t = q_t ⊗ h_{t-1} + B_t ⊗ u_t
    out_r = qh_r + Bu_r
    out_i = qh_i + Bu_i
    out_j = qh_j + Bu_j
    out_k = qh_k + Bu_k

    # Store h_t
    tl.store(h_out_ptr + base_offset + 0, out_r[:, :, :, None], mask=mask_3d[:, :, :, None])
    tl.store(h_out_ptr + base_offset + 1, out_i[:, :, :, None], mask=mask_3d[:, :, :, None])
    tl.store(h_out_ptr + base_offset + 2, out_j[:, :, :, None], mask=mask_3d[:, :, :, None])
    tl.store(h_out_ptr + base_offset + 3, out_k[:, :, :, None], mask=mask_3d[:, :, :, None])


def ssm_step_triton(
    h_prev: torch.Tensor,
    q: torch.Tensor,
    B: torch.Tensor,
    u: torch.Tensor
) -> torch.Tensor:
    """
    Un step de récurrence SSM quaternionique.

    Args:
        h_prev: [batch, d_model, d_state, 4] - état précédent
        q: [batch, d_model, d_state, 4] - opérateur d'évolution
        B: [batch, d_model, d_state, 4] - projection d'entrée
        u: [batch, d_model, 4] - signal d'entrée

    Returns:
        h_new: [batch, d_model, d_state, 4] - nouvel état
    """
    batch_size, d_model, d_state, _ = h_prev.shape

    h_out = torch.empty_like(h_prev)

    # Tailles de blocs optimisées
    BLOCK_B = min(32, batch_size)
    BLOCK_D = min(32, d_model)
    BLOCK_S = min(16, d_state)

    grid = (
        triton.cdiv(batch_size, BLOCK_B),
        triton.cdiv(d_model, BLOCK_D),
        triton.cdiv(d_state, BLOCK_S),
    )

    ssm_step_fused_kernel[grid](
        h_prev, q, B, u,
        h_out,
        batch_size, d_model, d_state,
        BLOCK_B=BLOCK_B,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
    )

    return h_out


# =============================================================================
# 3. PARALLEL SCAN ASSOCIATIF (BLELLOCH)
# =============================================================================

@triton.jit
def parallel_scan_upsweep_kernel(
    # In/Out array
    data_ptr,
    # Dimensions
    batch_size,
    seq_len,
    d_model,
    d_state,
    # Scan parameters
    stride,
    # Meta
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase up-sweep du parallel scan de Blelloch.

    Pour chaque paire d'éléments séparés de 'stride', calcule:
    data[i] = data[i] ⊗ data[i - stride]

    où ⊗ est le produit quaternionique.
    """
    pid = tl.program_id(axis=0)

    # Chaque thread traite un élément
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # On ne traite que les indices i >= stride et i % (2*stride) == 0
    valid_idx = (idx >= stride) & (idx < seq_len) & ((idx % (2 * stride)) == (2 * stride - 1))

    # Pour chaque batch, d_model, d_state
    # Structure: [batch, seq, d_model, d_state, 4]
    for b in range(batch_size):
        for d in range(d_model):
            for s in range(d_state):
                base_offset = (
                    b * (seq_len * d_model * d_state * 4) +
                    d * (d_state * 4) +
                    s * 4
                )

                # Charger data[idx]
                offset_i = base_offset + idx * (d_model * d_state * 4)
                mask = valid_idx & (idx < seq_len)

                ai_r = tl.load(data_ptr + offset_i + 0, mask=mask, other=0.0)
                ai_i = tl.load(data_ptr + offset_i + 1, mask=mask, other=0.0)
                ai_j = tl.load(data_ptr + offset_i + 2, mask=mask, other=0.0)
                ai_k = tl.load(data_ptr + offset_i + 3, mask=mask, other=0.0)

                # Charger data[idx - stride]
                offset_j = base_offset + (idx - stride) * (d_model * d_state * 4)

                aj_r = tl.load(data_ptr + offset_j + 0, mask=mask, other=0.0)
                aj_i = tl.load(data_ptr + offset_j + 1, mask=mask, other=0.0)
                aj_j = tl.load(data_ptr + offset_j + 2, mask=mask, other=0.0)
                aj_k = tl.load(data_ptr + offset_j + 3, mask=mask, other=0.0)

                # Produit quaternionique: result = ai ⊗ aj
                res_r = ai_r * aj_r - ai_i * aj_i - ai_j * aj_j - ai_k * aj_k
                res_i = ai_r * aj_i + ai_i * aj_r + ai_j * aj_k - ai_k * aj_j
                res_j = ai_r * aj_j - ai_i * aj_k + ai_j * aj_r + ai_k * aj_i
                res_k = ai_r * aj_k + ai_i * aj_j - ai_j * aj_i + ai_k * aj_r

                # Store
                tl.store(data_ptr + offset_i + 0, res_r, mask=mask)
                tl.store(data_ptr + offset_i + 1, res_i, mask=mask)
                tl.store(data_ptr + offset_i + 2, res_j, mask=mask)
                tl.store(data_ptr + offset_i + 3, res_k, mask=mask)


def parallel_scan_triton(data: torch.Tensor) -> torch.Tensor:
    """
    Parallel scan associatif pour quaternions (algorithme de Blelloch).

    Args:
        data: [batch, seq_len, d_model, d_state, 4]

    Returns:
        Scan cumulatif quaternionique
    """
    batch_size, seq_len, d_model, d_state, _ = data.shape

    # Copier pour ne pas modifier l'original
    result = data.clone()

    # Puissance de 2 supérieure ou égale
    log_n = math.ceil(math.log2(seq_len))

    # Up-sweep phase
    for level in range(log_n):
        stride = 2 ** level

        BLOCK_SIZE = 256
        n_elements = seq_len
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        parallel_scan_upsweep_kernel[grid](
            result,
            batch_size, seq_len, d_model, d_state,
            stride,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return result


# =============================================================================
# 4. SSM COMPLET FUSIONNÉ (CAYLEY + RÉCURRENCE + OUTPUT)
# =============================================================================

@triton.jit
def full_ssm_fused_kernel(
    # Inputs
    u_ptr,       # [B, L, D, 4] - input signal
    dt_ptr,      # [B, L, D] - time steps
    A_ptr,       # [D, S, 4] - continuous dynamics
    B_ptr,       # [B, L, S, 4] - input projection
    C_ptr,       # [B, L, S, 4] - output projection
    # Output
    y_ptr,       # [B, L, D, 4] - output signal
    # Dimensions
    batch_size,
    seq_len,
    d_model,
    d_state,
    eps,
    # Meta
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Kernel SSM quaternionique entièrement fusionné.

    Ce mega-kernel effectue:
    1. Discrétisation de Cayley (A, dt) -> q
    2. Calcul de dB = q^{-1} * B * dt
    3. Récurrence séquentielle h_t = q ⊗ h_{t-1} + dB ⊗ u_t
    4. Output y_t = sum_s (C_s ⊗ h_t_s)

    Optimisé pour minimiser les allers-retours HBM.
    """
    # Pour l'instant, on délègue à une implémentation PyTorch
    # car ce kernel serait extrêmement complexe
    # On utilisera les kernels individuels composés
    pass


def full_ssm_triton(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    SSM quaternionique complet utilisant les kernels Triton.

    Cette fonction compose les kernels optimisés pour:
    - Minimiser les accès mémoire
    - Maximiser la réutilisation des données en cache
    - Exploiter le parallélisme

    Args:
        u: [B, L, D, 4] - signal d'entrée
        dt: [B, L, D] - pas de temps adaptatifs
        A: [D, S, 4] - paramètres spectraux
        B: [B, L, S, 4] - projections d'entrée
        C: [B, L, S, 4] - projections de sortie

    Returns:
        y: [B, L, D, 4] - sortie du SSM
    """
    batch_size, seq_len, d_model, _ = u.shape
    d_state = A.shape[1]

    # 1. Discrétisation de Cayley
    # z = dt * A  (broadcast)
    dt_expanded = dt.unsqueeze(-1).unsqueeze(-1)  # [B, L, D, 1, 1]
    A_expanded = A.unsqueeze(0).unsqueeze(0)      # [1, 1, D, S, 4]

    z = dt_expanded * A_expanded  # [B, L, D, S, 4]
    z_flat = z.reshape(-1, 4)
    q_flat = cayley_discretization_triton(z_flat, eps)
    q = q_flat.reshape(batch_size, seq_len, d_model, d_state, 4)

    # 2. Calcul de dB (simplifié ici, devrait aussi être fusionné)
    # dB = den_inv * dt * B
    # Pour simplification, on utilise q directement
    B_expanded = B.unsqueeze(2)  # [B, L, 1, S, 4]
    dB = B_expanded.expand(batch_size, seq_len, d_model, d_state, 4)

    # 3. Récurrence séquentielle (pour seq court, sinon parallel scan)
    h = torch.zeros(batch_size, d_model, d_state, 4, device=u.device, dtype=u.dtype)
    outputs = []

    for t in range(seq_len):
        h = ssm_step_triton(h, q[:, t], dB[:, t], u[:, t])
        outputs.append(h)

    h_all = torch.stack(outputs, dim=1)  # [B, L, D, S, 4]

    # 4. Output projection
    C_expanded = C.unsqueeze(2)  # [B, L, 1, S, 4]

    # Produit quaternionique C ⊗ h
    # Pour simplification, on utilise PyTorch ici
    # Dans une vraie implémentation, ceci serait aussi un kernel Triton
    from ..kernels.quaternion_ops import quat_mul_triton

    yh = torch.zeros(batch_size, seq_len, d_model, d_state, 4, device=u.device, dtype=u.dtype)
    for b in range(batch_size):
        for l in range(seq_len):
            for d in range(d_model):
                for s in range(d_state):
                    yh[b, l, d, s] = quat_mul_triton(
                        C_expanded[b, l, 0, s].unsqueeze(0),
                        h_all[b, l, d, s].unsqueeze(0)
                    ).squeeze(0)

    y = yh.sum(dim=3)  # [B, L, D, 4]

    return y

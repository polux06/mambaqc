import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# =============================================================================
# 1. CONFIGURATION DU MODÈLE (ARCHITECTURE)
# =============================================================================

@dataclass
class QuaternionMambaConfig:
    d_model: int = 384          # Dimension divisible par 4
    n_layers: int = 24
    vocab_size: int = 2048      # Ajusté pour TinyStories (ou 50257 pour GPT2)
    d_state: int = 16           # 4 quaternions d'état
    d_conv: int = 4
    expand: int = 2
    dt_rank: int = "auto"       # "auto" -> ceil(d_model / 16)
    dropout: float = 0.0
    use_bias: bool = False
    
    def __post_init__(self):
        assert self.d_model % 4 == 0, "d_model doit être divisible par 4 (Quaternions)"
        assert self.d_state % 4 == 0, "d_state doit être divisible par 4"
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


# =============================================================================
# 2. NOYAUX QUATERNIONIQUES + GRADIENTS ANALYTIQUES
# =============================================================================

@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ar, ai, aj, ak = a.unbind(-1)
    br, bi, bj, bk = b.unbind(-1)
    cr = ar * br - ai * bi - aj * bj - ak * bk
    ci = ar * bi + ai * br + aj * bk - ak * bj
    cj = ar * bj - ai * bk + aj * br + ak * bi
    ck = ar * bk + ai * bj - aj * bi + ak * br
    return torch.stack((cr, ci, cj, ck), dim=-1)


@torch.jit.script
def quat_inv(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r, i, j, k = q.unbind(-1)
    norm_sq = r * r + i * i + j * j + k * k + eps
    scale = 1.0 / norm_sq
    return torch.stack((r * scale, -i * scale, -j * scale, -k * scale), dim=-1)


def quat_mul_backward(a: torch.Tensor, b: torch.Tensor, gy: torch.Tensor):
    """
    Gradients analytiques de y = quat_mul(a, b) par rapport à a et b.
    Toutes les opérations sont vectorisées.
    """
    ar, ai, aj, ak = a.unbind(-1)
    br, bi, bj, bk = b.unbind(-1)
    gr, gi, gj, gk = gy.unbind(-1)

    # dL/da
    dqr = gr * br + gi * bi + gj * bj + gk * bk
    dqi = -gr * bi + gi * br - gj * bk + gk * bj
    dqj = -gr * bj + gi * bk + gj * br - gk * bi
    dqk = -gr * bk - gi * bj + gj * bi + gk * br

    # dL/db
    dpr = gr * ar + gi * ai + gj * aj + gk * ak
    dpi = -gr * ai + gi * ar + gj * ak - gk * aj
    dpj = -gr * aj - gi * ak + gj * ar + gk * ai
    dpk = -gr * ak + gi * aj - gj * ai + gk * ar

    grad_a = torch.stack((dqr, dqi, dqj, dqk), dim=-1)
    grad_b = torch.stack((dpr, dpi, dpj, dpk), dim=-1)
    return grad_a, grad_b


def quat_inv_backward(q: torch.Tensor, gy: torch.Tensor, eps: float = 1e-6):
    """
    Gradient analytique de y = quat_inv(q) par rapport à q.
    y = conj(q) / (||q||^2 + eps)
    """
    r, i, j, k = q.unbind(-1)
    gr, gi, gj, gk = gy.unbind(-1)

    norm_sq = r * r + i * i + j * j + k * k + eps
    g = 1.0 / norm_sq

    # conj(q) = (r, -i, -j, -k)
    cr, ci, cj, ck = r, -i, -j, -k

    # y = c * g
    gc_r = gr * g
    gc_i = gi * g
    gc_j = gj * g
    gc_k = gk * g

    # dL/dg
    gg = gr * cr + gi * ci + gj * cj + gk * ck

    # g = 1 / n  =>  dg/dn = -1 / n^2 = -g^2
    dL_dn = -gg * (g * g)

    # n = r^2 + i^2 + j^2 + k^2 + eps
    grad_r = gc_r + dL_dn * 2 * r
    grad_i = -gc_i + dL_dn * 2 * i
    grad_j = -gc_j + dL_dn * 2 * j
    grad_k = -gc_k + dL_dn * 2 * k

    return torch.stack((grad_r, grad_i, grad_j, grad_k), dim=-1)


# =============================================================================
# 3. SCAN PARALLÈLE / SÉQUENTIEL POUR SSM QUATERNIONIQUE
# =============================================================================

def parallel_quaternion_scan(dA: torch.Tensor, Bu: torch.Tensor) -> torch.Tensor:
    """
    Calcule h_t = dA_t h_{t-1} + Bu_t via un scan parallèle (style Blelloch).

    Args:
        dA: (B, L, D_q, S_q, 4)
        Bu: (B, L, D_q, S_q, 4)

    Returns:
        h: (B, L, D_q, S_q, 4)
    """
    B_sz, L_val, Dq, Sq, _ = dA.shape
    total = 1 << (L_val - 1).bit_length()

    A_pad = torch.zeros(B_sz, total, Dq, Sq, 4, device=dA.device, dtype=dA.dtype)
    A_pad[..., 0] = 1.0
    B_pad = torch.zeros_like(A_pad)

    A_pad[:, :L_val] = dA
    B_pad[:, :L_val] = Bu

    step = 1
    while step < total:
        A_left = A_pad[:, :-step]
        B_left = B_pad[:, :-step]
        A_right = A_pad[:, step:]
        B_right = B_pad[:, step:]

        new_A = quat_mul(A_right, A_left)
        new_B = B_right + quat_mul(A_right, B_left)

        A_pad[:, step:] = new_A
        B_pad[:, step:] = new_B
        step <<= 1

    return B_pad[:, :L_val]


def sequential_quaternion_scan(dA: torch.Tensor, Bu: torch.Tensor) -> torch.Tensor:
    """Référence séquentielle pour h_t = dA_t h_{t-1} + Bu_t."""
    h = torch.zeros(dA.shape[0], dA.shape[2], dA.shape[3], 4, device=dA.device, dtype=dA.dtype)
    outs = []
    for t in range(dA.shape[1]):
        h = quat_mul(dA[:, t], h) + Bu[:, t]
        outs.append(h)
    return torch.stack(outs, dim=1)


def sequential_quaternion_ssm_autograd(
    u: torch.Tensor, dt: torch.Tensor, A_quat: torch.Tensor, B_quat: torch.Tensor, C_quat: torch.Tensor
) -> torch.Tensor:
    """
    Référence entièrement autograd pour vérifier le backward analytique.

    Args:
        u: (B, L, D_q, 4)
        dt: (B, L, D_q)
        A_quat: (1, 1, D_q, S_q, 4)
        B_quat: (B, L, S_q, 4)
        C_quat: (B, L, S_q, 4)

    Returns:
        y: (B, L, D_q, 4)
    """
    B_sz, L, D_q, _ = u.shape
    S_q = A_quat.shape[3]

    A_core = A_quat.view(D_q, S_q, 4)
    ones = torch.zeros(D_q, S_q, 4, device=u.device, dtype=u.dtype)
    ones[..., 0] = 1.0

    dt_broad = dt.unsqueeze(-1).unsqueeze(-1)  # (B, L, D_q, 1, 1)
    half_dt = 0.5 * dt_broad

    A_scaled = A_core.unsqueeze(0).unsqueeze(0) * half_dt
    ones_b = ones.unsqueeze(0).unsqueeze(0)

    num = ones_b + A_scaled
    den = ones_b - A_scaled
    den_inv = quat_inv(den)

    dA = quat_mul(num, den_inv)
    dB_scale = den_inv * dt_broad

    B_unsq = B_quat.unsqueeze(2)
    dB = quat_mul(dB_scale, B_unsq)

    u_unsq = u.unsqueeze(3)
    Bu = quat_mul(dB, u_unsq)

    h = torch.zeros(B_sz, D_q, S_q, 4, device=u.device, dtype=u.dtype)
    outs = []
    for t in range(L):
        h = quat_mul(dA[:, t], h) + Bu[:, t]
        outs.append(h)
    h_all = torch.stack(outs, dim=1)

    C_unsq = C_quat.unsqueeze(2)
    yh = quat_mul(C_unsq, h_all)
    return yh.sum(dim=3)


# =============================================================================
# 4. SSM QUATERNIONIQUE AVEC BACKWARD ANALYTIQUE
# =============================================================================

class QuaternionSSMFunction(torch.autograd.Function):
    """
    Scan SSM quaternionique :
      - Forward : boucle en temps sans autograd (aucun graphe énorme).
      - Backward : BPTT manuel, entièrement analytique, vectorisé.
    """

    @staticmethod
    def forward(ctx, u, dt, A_quat, B_quat, C_quat):
        """
        u:      (B, L, D_q, 4)
        dt:     (B, L, D_q)
        A_quat: (1, 1, D_q, S_q, 4)
        B_quat: (B, L, S_q, 4)
        C_quat: (B, L, S_q, 4)
        """
        B_sz, L, D_q, _ = u.shape
        S_q = A_quat.shape[3]
        device = u.device
        dtype = u.dtype

        A_core = A_quat.view(D_q, S_q, 4)      # (D_q, S_q, 4)
        ones = torch.zeros(D_q, S_q, 4, device=device, dtype=dtype)
        ones[..., 0] = 1.0

        with torch.no_grad():
            dt_broad = dt.unsqueeze(-1).unsqueeze(-1)         # (B, L, D_q, 1, 1)
            half_dt = 0.5 * dt_broad                         # (B, L, D_q, 1, 1)

            A_scaled = A_core.unsqueeze(0).unsqueeze(0) * half_dt  # (B, L, D_q, S_q, 4)
            ones_b = ones.unsqueeze(0).unsqueeze(0)               # (1, 1, D_q, S_q, 4)

            num = ones_b + A_scaled
            den = ones_b - A_scaled
            den_inv = quat_inv(den)

            dA = quat_mul(num, den_inv)                          # (B, L, D_q, S_q, 4)
            dB_scale = den_inv * dt_broad                        # (B, L, D_q, S_q, 4)

            B_unsq = B_quat.unsqueeze(2)                         # (B, L, 1, S_q, 4)
            dB = quat_mul(dB_scale, B_unsq)                      # (B, L, D_q, S_q, 4)

            u_unsq = u.unsqueeze(3)                              # (B, L, D_q, 1, 4)
            Bu = quat_mul(dB, u_unsq)                            # (B, L, D_q, S_q, 4)

            h_all = parallel_quaternion_scan(dA, Bu)             # (B, L, D_q, S_q, 4)

            # Recompute a sequential state trajectory for the backward pass
            # to ensure the stored history exactly follows the recurrence,
            # independently of the parallel scan implementation.
            h_seq_all = sequential_quaternion_scan(dA, Bu)
            h_list = [h_seq_all[:, t].clone() for t in range(L)]

            C_unsq = C_quat.unsqueeze(2)                         # (B, L, 1, S_q, 4)
            yh = quat_mul(C_unsq, h_all)                         # (B, L, D_q, S_q, 4)
            y = yh.sum(dim=3)                                    # (B, L, D_q, 4)

        # On sauvegarde uniquement les inputs + quelques structures
        ctx.save_for_backward(u, dt, A_quat, B_quat, C_quat)
        ctx.h_list = [h_.detach() for h_ in h_list]
        ctx.ones = ones
        ctx.L = L
        ctx.D_q = D_q
        ctx.S_q = S_q

        return y

    @staticmethod
    def backward(ctx, grad_y):
        """
        grad_y: (B, L, D_q, 4)

        Retour :
          grad_u      : (B, L, D_q, 4)
          grad_dt     : (B, L, D_q)
          grad_A_quat : (1, 1, D_q, S_q, 4)
          grad_B_quat : (B, L, S_q, 4)
          grad_C_quat : (B, L, S_q, 4)
        """
        u, dt, A_quat, B_quat, C_quat = ctx.saved_tensors
        h_list = ctx.h_list
        ones = ctx.ones
        L = ctx.L
        D_q = ctx.D_q
        S_q = ctx.S_q

        B_sz, _, _, _ = u.shape
        device = u.device
        dtype = u.dtype

        A_core = A_quat.view(D_q, S_q, 4)      # (D_q, S_q, 4)

        grad_u = torch.zeros_like(u)
        grad_dt = torch.zeros_like(dt)
        grad_A_core = torch.zeros_like(A_core)
        grad_B = torch.zeros_like(B_quat)
        grad_C = torch.zeros_like(C_quat)

        # dh_next : gradient accumulé sur h_t depuis le futur
        dh_next = torch.zeros(B_sz, D_q, S_q, 4, device=device, dtype=dtype)

        for t in reversed(range(L)):
            dt_t = dt[:, t]            # (B, D_q)
            u_t = u[:, t]              # (B, D_q, 4)
            B_t = B_quat[:, t]         # (B, S_q, 4)
            C_t = C_quat[:, t]         # (B, S_q, 4)
            h_prev = h_list[t - 1] if t > 0 else torch.zeros_like(h_list[0])
            gy_t = grad_y[:, t]        # (B, D_q, 4)

            # --- Recompute intermédiaires pour le pas t ---
            dt_broad = dt_t.unsqueeze(-1).unsqueeze(-1)      # (B, D_q, 1, 1)
            half_dt = 0.5 * dt_broad                         # (B, D_q, 1, 1)

            A_scaled = A_core.unsqueeze(0) * half_dt         # (B, D_q, S_q, 4)
            ones_b = ones.unsqueeze(0)                       # (1, D_q, S_q, 4)

            num = ones_b + A_scaled                          # (B, D_q, S_q, 4)
            den = ones_b - A_scaled                          # (B, D_q, S_q, 4)
            den_inv = quat_inv(den)                          # (B, D_q, S_q, 4)

            dA = quat_mul(num, den_inv)                      # (B, D_q, S_q, 4)
            dB_scale = den_inv * dt_broad                    # (B, D_q, S_q, 4)

            B_unsq = B_t.unsqueeze(1).expand(B_sz, D_q, S_q, 4)
            dB = quat_mul(dB_scale, B_unsq)                  # (B, D_q, S_q, 4)

            u_unsq = u_t.unsqueeze(2).expand(B_sz, D_q, S_q, 4)
            Bu = quat_mul(dB, u_unsq)                        # (B, D_q, S_q, 4)

            Ah = quat_mul(dA, h_prev)                        # (B, D_q, S_q, 4)
            h = Ah + Bu                                      # (B, D_q, S_q, 4)

            C_unsq = C_t.unsqueeze(1).expand(B_sz, D_q, S_q, 4)
            yh = quat_mul(C_unsq, h)                         # (B, D_q, S_q, 4)

            # ---------- Backward analytique pour le pas t ----------

            # y_t = sum_s yh  =>  dL/dyh = gy_t broadcasté sur S_q
            gyh = gy_t.unsqueeze(2).expand_as(yh)            # (B, D_q, S_q, 4)

            # yh = quat_mul(C_unsq, h)
            gC_unsq, gh = quat_mul_backward(C_unsq, h, gyh)

            # contribution du futur sur h_t
            gh = gh + dh_next                                # (B, D_q, S_q, 4)

            # h = Ah + Bu
            gAh = gh
            gBu = gh

            # Ah = quat_mul(dA, h_prev)
            gdA, gh_prev = quat_mul_backward(dA, h_prev, gAh)

            # Bu = quat_mul(dB, u_unsq)
            gdB, gu_unsq = quat_mul_backward(dB, u_unsq, gBu)
            gu_t = gu_unsq.sum(dim=2)                        # somme sur S_q
            grad_u[:, t].add_(gu_t)

            # dB = quat_mul(dB_scale, B_unsq)
            gdB_scale, gB_unsq = quat_mul_backward(dB_scale, B_unsq, gdB)
            gB_t = gB_unsq.sum(dim=1)                        # somme sur D_q
            grad_B[:, t].add_(gB_t)

            # dB_scale = den_inv * dt_broad
            g_den_inv = gdB_scale * dt_broad
            g_dt_broad = (gdB_scale * den_inv).sum(dim=(2, 3), keepdim=True)  # (B, D_q, 1, 1)

            # dA = quat_mul(num, den_inv)
            gnum, g_den_inv2 = quat_mul_backward(num, den_inv, gdA)
            g_den_inv = g_den_inv + g_den_inv2

            # den_inv = quat_inv(den)
            gden = quat_inv_backward(den, g_den_inv)

            # den = ones - A_scaled
            gA_scaled = gnum - gden                          # (B, D_q, S_q, 4)

            # A_scaled = A_core * half_dt
            # grad A_core : somme sur batch
            grad_A_core.add_((gA_scaled * half_dt).sum(dim=0))   # (D_q, S_q, 4)

            # grad half_dt
            g_half_dt = (gA_scaled * A_core.unsqueeze(0)).sum(dim=(2, 3), keepdim=True)  # (B, D_q, 1, 1)

            # half_dt = 0.5 * dt_broad
            g_dt_broad = g_dt_broad + 0.5 * g_half_dt

            # dt_broad -> dt_t
            gdt_t = g_dt_broad.sum(dim=(2, 3))               # (B, D_q)
            grad_dt[:, t].add_(gdt_t)

            # C_unsq -> C_t (broadcast sur D_q)
            gC_t = gC_unsq.sum(dim=1)                        # (B, S_q, 4)
            grad_C[:, t].add_(gC_t)

            # h_prev reçoit gh_prev
            dh_next = gh_prev

        # A_core (D_q, S_q, 4) -> A_quat (1, 1, D_q, S_q, 4)
        grad_A_quat = torch.zeros_like(A_quat)
        grad_A_quat[0, 0].copy_(grad_A_core)

        return grad_u, grad_dt, grad_A_quat, grad_B, grad_C


def quaternion_ssm(u, dt, A_quat, B_quat, C_quat):
    return QuaternionSSMFunction.apply(u, dt, A_quat, B_quat, C_quat)


# =============================================================================
# 4. LAYERS QUATERNIONIQUES (RMSNorm, Conv, Linear)
# =============================================================================

class QuaternionRMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        x_q = x.view(*x.shape[:-1], -1, 4)
        norm_sq = (x_q ** 2).sum(dim=-1, keepdim=True)
        rms = torch.rsqrt(norm_sq.mean(dim=-2, keepdim=True) + self.eps)
        x_norm = x_q * rms
        return x_norm.flatten(start_dim=-2) * self.weight


class QuaternionConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0
        self.in_q, self.out_q = in_channels // 4, out_channels // 4
        self.kernel_size = kernel_size
        self.groups = self.in_q
        
        shape = (self.out_q, 1, kernel_size)
        self.r = nn.Parameter(torch.randn(shape))
        self.i = nn.Parameter(torch.randn(shape))
        self.j = nn.Parameter(torch.randn(shape))
        self.k = nn.Parameter(torch.randn(shape))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        std = 1.0 / math.sqrt(in_channels * kernel_size)
        for p in [self.r, self.i, self.j, self.k]:
            nn.init.normal_(p, 0.0, std)

    def forward(self, x):
        # x : (B, D, L)
        x_pad = F.pad(x, (self.kernel_size - 1, 0))
        B, D, L_pad = x_pad.shape
        x_reshaped = x_pad.view(B, 4, self.in_q, L_pad)
        xr, xi, xj, xk = x_reshaped.unbind(1)
        
        def conv(input_c, weight_c):
            return F.conv1d(input_c, weight_c, groups=self.groups)

        yr = conv(xr, self.r) - conv(xi, self.i) - conv(xj, self.j) - conv(xk, self.k)
        yi = conv(xi, self.r) + conv(xr, self.i) + conv(xj, self.k) - conv(xk, self.j)
        yj = conv(xj, self.r) - conv(xk, self.i) + conv(xr, self.j) + conv(xi, self.k)
        yk = conv(xk, self.r) + conv(xj, self.i) - conv(xi, self.j) + conv(xr, self.k)
        
        y = torch.stack([yr, yi, yj, yk], dim=1).view(B, -1, yr.shape[-1])
        return y + self.bias.view(1, -1, 1)


class QuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        assert in_features % 4 == 0 and out_features % 4 == 0
        self.in_q, self.out_q = in_features // 4, out_features // 4
        self.r = nn.Parameter(torch.empty(self.out_q, self.in_q))
        self.i = nn.Parameter(torch.empty(self.out_q, self.in_q))
        self.j = nn.Parameter(torch.empty(self.out_q, self.in_q))
        self.k = nn.Parameter(torch.empty(self.out_q, self.in_q))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        std = 1.0 / math.sqrt(in_features)
        for p in [self.r, self.i, self.j, self.k]:
            nn.init.normal_(p, 0, std)

    def forward(self, x):
        r, i, j, k = x.chunk(4, dim=-1)
        out_r = F.linear(r, self.r) - F.linear(i, self.i) - F.linear(j, self.j) - F.linear(k, self.k)
        out_i = F.linear(r, self.i) + F.linear(i, self.r) + F.linear(j, self.k) - F.linear(k, self.j)
        out_j = F.linear(r, self.j) - F.linear(i, self.k) + F.linear(j, self.r) + F.linear(k, self.i)
        out_k = F.linear(r, self.k) + F.linear(i, self.j) - F.linear(j, self.i) + F.linear(k, self.r)
        out = torch.cat([out_r, out_i, out_j, out_k], dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out


# =============================================================================
# 5. BLOC QUATERNION MAMBA
# =============================================================================

class QuaternionMambaBlock(nn.Module):
    def __init__(self, config: QuaternionMambaConfig):
        super().__init__()
        self.config = config
        self.d_inner = config.d_model * config.expand
        self.d_inner_q = self.d_inner // 4
        self.d_state_q = config.d_state // 4
        
        self.in_proj = QuaternionLinear(config.d_model, self.d_inner * 2, bias=config.use_bias)
        self.conv1d = QuaternionConv1d(self.d_inner, self.d_inner, config.d_conv)

        # Pour dt, B, C
        self.x_proj = nn.Linear(self.d_inner, config.dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, self.d_inner_q, bias=True)
        
        # Paramètres de A quaternionique
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, self.d_state_q + 1).float().repeat(self.d_inner_q, 1)
            )
        )
        self.A_i = nn.Parameter(torch.randn(self.d_inner_q, self.d_state_q) * 0.01)
        self.A_j = nn.Parameter(torch.randn(self.d_inner_q, self.d_state_q) * 0.01)
        self.A_k = nn.Parameter(torch.randn(self.d_inner_q, self.d_state_q) * 0.01)

        # Skip/diagonal
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = QuaternionLinear(self.d_inner, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        x: (B, L, d_model)
        """
        batch, seq_len, _ = x.shape

        # Input projection quaternionique -> x_ssm, z
        xz = self.in_proj(x)                          # (B, L, 2 * d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)               # (B, L, d_inner)

        # Convolution locale
        x_conv = self.act(self.conv1d(x_ssm.transpose(1, 2)).transpose(1, 2))  # (B, L, d_inner)
        
        # Proj vers dt, B, C
        x_dbl = self.x_proj(x_conv)                  # (B, L, dt_rank + 2*d_state)
        dt_rank, B_raw, C_raw = x_dbl.split(
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1
        )

        # dt : (B, L, D_q)
        dt = F.softplus(self.dt_proj(dt_rank))       # d_inner_q = D_q

        # B, C en quaternions
        B_quat = B_raw.view(batch, seq_len, 4, self.d_state_q)\
                      .permute(0, 1, 3, 2).contiguous()  # (B, L, S_q, 4)
        C_quat = C_raw.view(batch, seq_len, 4, self.d_state_q)\
                      .permute(0, 1, 3, 2).contiguous()  # (B, L, S_q, 4)

        # u_quat : (B, L, D_q, 4)
        u_quat = x_conv.view(batch, seq_len, 4, self.d_inner_q)\
                         .permute(0, 1, 3, 2).contiguous()

        # A : (1, 1, D_q, S_q, 4)
        A_r = -torch.exp(self.A_log)  # (D_q, S_q)
        A_quat = torch.stack(
            (A_r, self.A_i, self.A_j, self.A_k),
            dim=-1
        ).unsqueeze(0).unsqueeze(0)

        # SSM quaternionique (custom autograd analytique)
        y_ssm = quaternion_ssm(u_quat, dt, A_quat, B_quat, C_quat)   # (B, L, D_q, 4)

        # Remise en forme : (B, L, d_inner)
        y = y_ssm.permute(0, 1, 3, 2).contiguous().reshape(batch, seq_len, self.d_inner)

        # Skip connection diagonale
        y = y + x_conv * self.D

        # Gating par z
        y = y * self.act(z)

        # Projection de sortie
        return self.dropout(self.out_proj(y))


# =============================================================================
# 6. MODÈLE LM COMPLET
# =============================================================================

class QuaternionMambaLMHeadModel(nn.Module):
    def __init__(self, config: QuaternionMambaConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([QuaternionMambaBlock(config) for _ in range(config.n_layers)])
        self.norm_f = QuaternionRMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        idx: (B, L) indices de tokens
        targets: (B, L) ou None
        """
        x = self.embedding(idx)  # (B, L, d_model)

        for layer in self.layers:
            x = x + layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            
        return logits, loss


if __name__ == "__main__":
    # Tests rapides : scan parallèle vs séquentiel + passe avant du modèle
    torch.manual_seed(0)

    B, L, D_q, S_q = 2, 5, 3, 2
    dA = torch.randn(B, L, D_q, S_q, 4)
    Bu = torch.randn(B, L, D_q, S_q, 4)

    h_parallel = parallel_quaternion_scan(dA, Bu)
    h_seq = sequential_quaternion_scan(dA, Bu)
    torch.testing.assert_close(h_parallel, h_seq, rtol=1e-5, atol=1e-5)
    print("[OK] Scan parallèle équivaut au scan séquentiel sur un exemple synthétique.")

    cfg = QuaternionMambaConfig(d_model=64, n_layers=2, vocab_size=100, d_state=16, expand=2)
    model = QuaternionMambaLMHeadModel(cfg)
    x = torch.randint(0, 100, (2, 32))
    logits, loss = model(x, targets=x)
    print(f"[OK] Passe avant du modèle : logits {logits.shape}, loss {loss.item():.4f}")

    # Vérification du backward : comparaison analytique vs autograd naïf
    torch.manual_seed(1)
    B, L, D_q, S_q = 1, 3, 2, 2
    dtype = torch.double

    def rand_quat(shape, scale=0.1):
        return (torch.randn(*shape, dtype=dtype) * scale).requires_grad_()

    u = (torch.randn(B, L, D_q, 4, dtype=dtype) * 0.2).requires_grad_()
    dt = (torch.rand(B, L, D_q, dtype=dtype) * 0.1 + 0.05).requires_grad_()
    A_quat = rand_quat((1, 1, D_q, S_q, 4), scale=0.05)
    B_quat = rand_quat((B, L, S_q, 4), scale=0.1)
    C_quat = rand_quat((B, L, S_q, 4), scale=0.1)

    # Custom autograd
    y_custom = quaternion_ssm(u, dt, A_quat, B_quat, C_quat)
    loss_custom = (y_custom ** 2).sum()
    grads_custom = torch.autograd.grad(loss_custom, [u, dt, A_quat, B_quat, C_quat], create_graph=False)

    # Référence autograd séquentielle
    u_ref, dt_ref, A_ref, B_ref, C_ref = [t.detach().clone().requires_grad_() for t in (u, dt, A_quat, B_quat, C_quat)]
    y_ref = sequential_quaternion_ssm_autograd(u_ref, dt_ref, A_ref, B_ref, C_ref)
    loss_ref = (y_ref ** 2).sum()
    grads_ref = torch.autograd.grad(loss_ref, [u_ref, dt_ref, A_ref, B_ref, C_ref], create_graph=False)

    for name, g_c, g_r in zip(["u", "dt", "A", "B", "C"], grads_custom, grads_ref):
        torch.testing.assert_close(g_c, g_r, rtol=1e-5, atol=1e-6)
    print("[OK] Backward analytique concorde avec autograd séquentiel.")

    # Gradcheck complet sur un cas minuscule pour valider le backward analytique
    torch.autograd.gradcheck(
        lambda *inp: quaternion_ssm(*inp),
        (u.detach().double().requires_grad_(), dt.detach().double().requires_grad_(),
         A_quat.detach().double().requires_grad_(), B_quat.detach().double().requires_grad_(),
         C_quat.detach().double().requires_grad_()),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    print("[OK] gradcheck torch sur quaternion_ssm passé.")

"""
Quaternion Mamba-2 Block.

Implements the complete Quaternion Mamba-2 architecture as described in the paper:

Input x_t ∈ ℝ^{d_model}
    ↓
[Projection + Split] → (x'_t, z_t)
    ↓
[Conv1D Causale (kernel=4)] → x''_t
    ↓
[Activation SiLU] → x'''_t
    ↓
[Conversion Quaternionique] → x_quat ∈ ℍ^{d_model}
    ↓
[S6 Gate + SSM Quaternionique] → y_quat ∈ ℍ^{d_model}
    ↓
[Normalisation Géométrique]
    ↓
[Projection vers ℝ] → y'_t ∈ ℝ^{d_model}
    ↓
[Gating : z_t ⊙ y'_t]
    ↓
[Projection de sortie + Residual]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mambaqc.layers.causal_conv1d import CausalConv1d
from mambaqc.layers.quaternion_linear import QuaternionLinear, real_to_quaternion_projection, quaternion_to_real_projection
from mambaqc.layers.quaternion_norm import QuaternionLayerNorm
from mambaqc.kernels.ssm_fused import ssm_forward_fused


class QuaternionMamba2Block(nn.Module):
    """
    Single Quaternion Mamba-2 block.

    Args:
        d_model: Model dimension
        d_state: Internal state dimension per mode
        d_conv: Convolution kernel size (default: 4)
        expand_factor: Expansion factor for inner dimension (default: 2)
        dt_rank: Rank of Delta projection (default: "auto" = d_model // 16)
        bias: Whether to use bias in projections

    Shape:
        - Input: [batch, seq_len, d_model]
        - Output: [batch, seq_len, d_model]
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_rank: str | int = "auto",
        bias: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand_factor

        if dt_rank == "auto":
            self.dt_rank = max(16, d_model // 16)
        else:
            self.dt_rank = dt_rank

        # === Input projection ===
        # Projects x → (x', z) where x' goes through SSM and z is for gating
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # === Conv1D causale ===
        self.conv1d = CausalConv1d(self.d_inner, kernel_size=d_conv, bias=True)

        # === Paramètres spectraux Λ ===
        # Λ = -softplus(α) + tanh(v)i + tanh(w)j + tanh(z)k
        # Shape: [d_inner, d_state]
        self.Lambda_alpha = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.5 + 1.0)
        self.Lambda_imag = nn.Parameter(torch.randn(self.d_inner, d_state, 3) * 0.1)  # [v, w, z]

        # === Projections pour Δ, B, C, D ===
        # Δ: time-varying step size (real-valued, positive)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # B: input → state projection (quaternion-valued)
        self.B_proj = real_to_quaternion_projection(
            self.d_inner,
            self.d_inner * d_state,
            bias=False,
        )

        # C: state → output projection (quaternion-valued)
        self.C_proj = real_to_quaternion_projection(
            self.d_inner,
            self.d_inner * d_state,
            bias=False,
        )

        # D: skip connection (real-valued)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # === S6 Gate ===
        self.gate_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)

        # === Normalisation quaternionique ===
        self.norm = QuaternionLayerNorm(self.d_inner, eps=1e-5)

        # === Projection de sortie ===
        # Converts quaternion → real
        self.out_proj = quaternion_to_real_projection(
            self.d_inner,
            d_model,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Quaternion Mamba-2 block.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            out: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # === 1. Input projection and split ===
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_inner, z = xz.chunk(2, dim=-1)  # Each: [B, T, d_inner]

        # === 2. Causal convolution ===
        x_conv = self.conv1d(x_inner)  # [B, T, d_inner]

        # === 3. Activation ===
        x_act = F.silu(x_conv)  # [B, T, d_inner]

        # === 4. Convert to quaternions ===
        # For now, we'll project real → quaternion via B_proj during SSM
        # x_act stays real until SSM input

        # === 5. SSM forward pass ===
        y_quat = self._ssm_forward(x_act)  # [B, T, d_inner, 4]

        # === 6. Geometric normalization ===
        y_norm = self.norm(y_quat)  # [B, T, d_inner, 4]

        # === 7. Project back to real ===
        y_real = self.out_proj(y_norm)  # [B, T, d_model]

        # === 8. Gating ===
        z_gate = torch.sigmoid(z)
        y_gated = z_gate * y_real

        # === 9. Residual connection ===
        out = x + y_gated

        return out

    def _ssm_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SSM forward pass: compute quaternion dynamics and recurrence.

        Args:
            x: [batch, seq_len, d_inner] - real-valued input

        Returns:
            y: [batch, seq_len, d_inner, 4] - quaternion output
        """
        batch_size, seq_len, d_inner = x.shape

        # === Generate time-varying parameters ===

        # Δ: step size (real, positive)
        Delta = F.softplus(self.dt_proj(x))  # [B, T, d_inner]

        # B: input projection (real → quaternion)
        B = self.B_proj(x)  # [B, T, d_inner*d_state, 4]
        B = B.view(batch_size, seq_len, d_inner, self.d_state, 4)

        # C: output projection (real → quaternion)
        C = self.C_proj(x)  # [B, T, d_inner*d_state, 4]
        C = C.view(batch_size, seq_len, d_inner, self.d_state, 4)

        # D: skip connection
        D = self.D.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)  # [B, T, d_inner]

        # S: selected signal (S6 gate)
        gate = torch.sigmoid(self.gate_proj(x))  # [B, T, d_inner]
        S_real = gate * x  # [B, T, d_inner]

        # Convert S to quaternion (embed in real component)
        S = torch.zeros(batch_size, seq_len, d_inner, 4, device=x.device, dtype=x.dtype)
        S[..., 0] = S_real  # (s, 0, 0, 0)

        # === Compute spectral parameters Λ ===
        # Λ = -softplus(α) + tanh(v)i + tanh(w)j + tanh(z)k
        Lambda_real = -F.softplus(self.Lambda_alpha)  # [d_inner, d_state]
        Lambda_imag = torch.tanh(self.Lambda_imag)  # [d_inner, d_state, 3]

        # Construct quaternion Λ: [d_inner, d_state, 4]
        Lambda = torch.zeros(d_inner, self.d_state, 4, device=x.device, dtype=x.dtype)
        Lambda[..., 0] = Lambda_real
        Lambda[..., 1:] = Lambda_imag

        # === Compute z = Δ * Λ (dynamics) ===
        # Δ: [B, T, d_inner, 1, 1]
        # Λ: [1, 1, d_inner, d_state, 4]
        Delta_expanded = Delta.unsqueeze(-1).unsqueeze(-1)  # [B, T, d_inner, 1, 1]
        Lambda_expanded = Lambda.unsqueeze(0).unsqueeze(0)  # [1, 1, d_inner, d_state, 4]

        # Scalar-quaternion multiplication: Δ * Λ
        z = Delta_expanded * Lambda_expanded  # [B, T, d_inner, d_state, 4]

        # === Cayley discretization ===
        from mambaqc.kernels.cayley_transform import cayley_discretization_fused
        q = cayley_discretization_fused(z, eps=1e-6)  # [B, T, d_inner, d_state, 4]

        # === SSM recurrence ===
        y = ssm_forward_fused(q, B, S, C, D)  # [B, T, d_inner, 4]

        return y

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"d_inner={self.d_inner}, d_conv={self.d_conv}, "
            f"dt_rank={self.dt_rank}"
        )

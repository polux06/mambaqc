"""Quaternion Mamba-2 Lite block.

This block keeps the scalar state dynamics of Mamba-2 (bilinear discretization
with associative scalar A_t) while introducing quaternionic input injections
and output projections. The internal recurrent state remains real-valued to
preserve commutativity for the prefix scan, but B_t and C_t operate in
quaternion space to add rotational structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mambaqc.layers.causal_conv1d import CausalConv1d
from mambaqc.layers.quaternion_linear import (
    quaternion_to_real_projection,
    real_to_quaternion_projection,
)
from mambaqc.layers.quaternion_norm import QuaternionLayerNorm


class QuaternionMamba2LiteBlock(nn.Module):
    """Single Quaternion Mamba-2 Lite block.

    Args:
        d_model: Model dimension.
        d_state: Number of state modes (real-valued) per channel.
        d_conv: Causal convolution kernel size.
        expand_factor: Expansion factor for the inner dimension.
        bias: Whether to use bias terms in projections.

    Shape:
        - Input: ``[batch, seq_len, d_model]``
        - Output: ``[batch, seq_len, d_model]``
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand_factor: int = 2,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand_factor

        # === Input projection ===
        # Projects x → (x', z) where x' goes through SSM and z is for gating
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # === Conv1D causale ===
        self.conv1d = CausalConv1d(self.d_inner, kernel_size=d_conv, bias=True)

        # === Spectral parameters Λ (real, negative) ===
        self.Lambda_alpha = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.5 + 1.0)

        # === Projections for Δ, B, C, D ===
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Quaternionic injections and projections
        self.B_proj = real_to_quaternion_projection(
            self.d_inner,
            self.d_inner * d_state,
            bias=False,
        )

        self.C_proj = real_to_quaternion_projection(
            self.d_inner,
            self.d_inner * d_state,
            bias=False,
        )

        # Skip connection (real)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # === S6 Gate ===
        self.gate_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)

        # === Quaternion normalization ===
        self.norm = QuaternionLayerNorm(self.d_inner, eps=1e-5)

        # === Output projection (quaternion → real) ===
        self.out_proj = quaternion_to_real_projection(
            self.d_inner,
            d_model,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block."""

        batch_size, seq_len, _ = x.shape

        # === 1. Input projection and split ===
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # === 2. Causal convolution ===
        x_conv = self.conv1d(x_inner)

        # === 3. Activation ===
        x_act = F.silu(x_conv)

        # === 4. Generate time-varying parameters ===
        Delta = F.softplus(self.dt_proj(x_act))  # [B, T, d_inner]

        Lambda = -F.softplus(self.Lambda_alpha)  # [d_inner, d_state]
        Lambda_expanded = Lambda.unsqueeze(0).unsqueeze(0)  # [1, 1, d_inner, d_state]

        z_dyn = Delta.unsqueeze(-1) * Lambda_expanded  # [B, T, d_inner, d_state]

        # Bilinear discretization: A = (1 + 0.5 z) / (1 - 0.5 z)
        denom = 1 - 0.5 * z_dyn
        denom = torch.clamp(denom, min=1e-4)
        A = (1 + 0.5 * z_dyn) / denom

        # === 5. Quaternion parameters ===
        B = self.B_proj(x_act).view(batch_size, seq_len, self.d_inner, self.d_state, 4)
        C = self.C_proj(x_act).view(batch_size, seq_len, self.d_inner, self.d_state, 4)

        # === 6. Gate and selected signal ===
        gate = torch.sigmoid(self.gate_proj(x_act))
        S_real = gate * x_act

        S_quat = torch.zeros(
            batch_size, seq_len, self.d_inner, 4, device=x.device, dtype=x.dtype
        )
        S_quat[..., 0] = S_real

        # === 7. Recurrence over sequence ===
        h = torch.zeros(
            batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype
        )
        y_outputs: list[torch.Tensor] = []

        for t in range(seq_len):
            A_t = A[:, t]  # [B, d_inner, d_state]
            B_t = B[:, t]  # [B, d_inner, d_state, 4]
            C_t = C[:, t]  # [B, d_inner, d_state, 4]
            S_t = S_quat[:, t]  # [B, d_inner, 4]

            h = A_t * h + self._quaternion_injection_real(B_t, S_t)

            y_state = (C_t * h.unsqueeze(-1)).sum(dim=-2)
            y_skip = self.D.view(1, self.d_inner, 1) * S_t
            y = y_state + y_skip
            y_outputs.append(y)

        y_quat = torch.stack(y_outputs, dim=1)  # [B, T, d_inner, 4]

        # === 8. Quaternion norm and projection back to real ===
        y_norm = self.norm(y_quat)
        y_real = self.out_proj(y_norm)

        # === 9. Gating ===
        z_gate = torch.sigmoid(z[..., : self.d_model])
        y_gated = z_gate * y_real

        return x + y_gated

    @staticmethod
    def _quaternion_injection_real(
        B_t: torch.Tensor, S_t: torch.Tensor
    ) -> torch.Tensor:
        """Real-valued injection term from quaternion multiplication.

        Computes the real component of the Hamilton product B_t * S_t to keep
        the recurrent state real-valued while letting B capture quaternionic
        rotations of the selected signal.
        """

        B0, B1, B2, B3 = B_t.unbind(dim=-1)
        S0, S1, S2, S3 = S_t.unsqueeze(-2).unbind(dim=-1)

        return B0 * S0 - B1 * S1 - B2 * S2 - B3 * S3

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"d_inner={self.d_inner}, d_conv={self.d_conv}"
        )

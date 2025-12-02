"""
Causal 1D convolution for local pattern extraction.

Implements depthwise causal convolution as in Mamba-2.

Key properties:
- Causal: output at time t depends only on inputs at times <= t
- Depthwise: each channel processed independently (efficient)
- Local receptive field: kernel_size tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with depthwise groups.

    Args:
        d_model: Number of channels
        kernel_size: Size of convolutional kernel (default: 4)
        bias: Whether to use bias (default: True)

    Shape:
        - Input: [batch, seq_len, d_model]
        - Output: [batch, seq_len, d_model]

    Note:
        - Uses depthwise convolution (groups=d_model) for efficiency
        - Padding is applied on the left (past) only to maintain causality
        - Compatible with autoregressive generation
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 4,
        bias: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # Left padding for causality

        # Depthwise convolution (each channel processed independently)
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=0,  # We'll apply manual padding
            groups=d_model,  # Depthwise
            bias=bias,
        )

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize convolution weights."""
        # Xavier/Glorot initialization for conv weights
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            out: [batch, seq_len, d_model]
        """
        # Conv1d expects [batch, channels, time]
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]

        # Apply causal padding (on the left)
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0), mode='constant', value=0.0)

        # Convolve
        x = self.conv(x)

        # Back to [batch, seq_len, d_model]
        x = x.transpose(1, 2)

        return x

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, kernel_size={self.kernel_size}, "
            f"padding={self.padding}, bias={self.conv.bias is not None}"
        )


class CausalConv1dTriton(nn.Module):
    """
    Triton-optimized causal convolution.

    TODO: Implement custom Triton kernel for fused causal conv.

    Benefits:
    - Fuse padding + convolution
    - Optimize for short kernels (k=4)
    - Reduce memory traffic

    For now, uses the standard PyTorch implementation above.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        # Placeholder: delegate to standard implementation
        self.conv = CausalConv1d(d_model, kernel_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# Alias for backward compatibility
CausalConv1D = CausalConv1d

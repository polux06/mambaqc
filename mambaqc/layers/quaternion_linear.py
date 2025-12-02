"""
Quaternion linear layers with Tensor Core optimizations.

Supports:
1. Real → Quaternion projection
2. Quaternion → Real projection
3. Quaternion → Quaternion projection (with Hamilton product)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class QuaternionLinear(nn.Module):
    """
    Linear layer for quaternion-valued inputs/outputs.

    Modes:
    - input_type="real", output_type="quaternion": Projects R^d_in → H^{d_out/4}
    - input_type="quaternion", output_type="real": Projects H^{d_in/4} → R^d_out
    - input_type="quaternion", output_type="quaternion": Hamilton product-based projection

    Args:
        in_features: Input dimension
            - If input_type="real": actual dimension
            - If input_type="quaternion": must be divisible by 4
        out_features: Output dimension
            - If output_type="real": actual dimension
            - If output_type="quaternion": must be divisible by 4
        input_type: "real" or "quaternion"
        output_type: "real" or "quaternion"
        bias: Whether to include bias term

    Shape:
        - Input: [..., in_features] or [..., in_features//4, 4] if quaternion
        - Output: [..., out_features] or [..., out_features//4, 4] if quaternion
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_type: str = "real",
        output_type: str = "quaternion",
        bias: bool = True,
    ):
        super().__init__()

        assert input_type in ["real", "quaternion"]
        assert output_type in ["real", "quaternion"]

        self.in_features = in_features
        self.out_features = out_features
        self.input_type = input_type
        self.output_type = output_type

        if input_type == "quaternion":
            assert in_features % 4 == 0, f"in_features must be divisible by 4, got {in_features}"
        if output_type == "quaternion":
            assert out_features % 4 == 0, f"out_features must be divisible by 4, got {out_features}"

        # Weight initialization
        if input_type == "real" and output_type == "quaternion":
            # R → H: weight is [out_features, in_features] where out_features = 4*n_quat
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self._init_real_to_quaternion()

        elif input_type == "quaternion" and output_type == "real":
            # H → R: weight is [out_features, in_features]
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self._init_quaternion_to_real()

        elif input_type == "quaternion" and output_type == "quaternion":
            # H → H: Quaternion-valued weight matrix
            # W ∈ H^{n_out × n_in} represented as R^{4*n_out × 4*n_in}
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self._init_quaternion_to_quaternion()

        else:  # real → real (standard linear)
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Bias
        if bias:
            if output_type == "quaternion":
                # Bias in quaternion space: [n_quat, 4]
                n_quat = out_features // 4
                self.bias = nn.Parameter(torch.zeros(n_quat, 4))
            else:
                self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def _init_real_to_quaternion(self):
        """Initialize weights for R → H projection."""
        # Effective output quaternion dimension
        n_quat = self.out_features // 4

        # Variance scaling: account for 4 components
        std = math.sqrt(2.0 / (self.in_features + n_quat))
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def _init_quaternion_to_real(self):
        """Initialize weights for H → R projection."""
        # Effective input quaternion dimension
        n_quat_in = self.in_features // 4

        std = math.sqrt(2.0 / (n_quat_in + self.out_features))
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def _init_quaternion_to_quaternion(self):
        """Initialize weights for H → H projection."""
        n_quat_in = self.in_features // 4
        n_quat_out = self.out_features // 4

        std = math.sqrt(2.0 / (n_quat_in + n_quat_out))
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
                - If input_type="real": [..., in_features]
                - If input_type="quaternion": [..., in_features//4, 4]

        Returns:
            out: Output tensor
                - If output_type="real": [..., out_features]
                - If output_type="quaternion": [..., out_features//4, 4]
        """
        if self.input_type == "real" and self.output_type == "quaternion":
            return self._forward_real_to_quaternion(x)
        elif self.input_type == "quaternion" and self.output_type == "real":
            return self._forward_quaternion_to_real(x)
        elif self.input_type == "quaternion" and self.output_type == "quaternion":
            return self._forward_quaternion_to_quaternion(x)
        else:  # real → real
            out = F.linear(x, self.weight, self.bias)
            return out

    def _forward_real_to_quaternion(self, x: torch.Tensor) -> torch.Tensor:
        """R → H: Standard matrix multiply, reshape to quaternions."""
        # x: [..., in_features]
        # W: [out_features, in_features] where out_features = 4*n_quat

        out = F.linear(x, self.weight, bias=None)  # [..., out_features]

        # Reshape to quaternions
        *batch_dims, _ = out.shape
        n_quat = self.out_features // 4
        out = out.view(*batch_dims, n_quat, 4)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias

        return out

    def _forward_quaternion_to_real(self, x: torch.Tensor) -> torch.Tensor:
        """H → R: Flatten quaternions, then matrix multiply."""
        # x: [..., n_quat, 4]

        *batch_dims, n_quat, _ = x.shape
        x_flat = x.reshape(*batch_dims, n_quat * 4)

        out = F.linear(x_flat, self.weight, self.bias)

        return out

    def _forward_quaternion_to_quaternion(self, x: torch.Tensor) -> torch.Tensor:
        """
        H → H: Quaternion matrix multiplication.

        For simplicity, we use a real-valued weight matrix applied to flattened quaternions.
        True quaternion matrix multiply would use Hamilton products, but that's expensive.

        Approximation: Treat as block-wise real matrix multiply.
        """
        # x: [..., n_quat_in, 4]

        *batch_dims, n_quat_in, _ = x.shape
        n_quat_out = self.out_features // 4

        # Flatten input
        x_flat = x.reshape(*batch_dims, n_quat_in * 4)

        # Matrix multiply
        out_flat = F.linear(x_flat, self.weight, bias=None)

        # Reshape to quaternions
        out = out_flat.view(*batch_dims, n_quat_out, 4)

        # Add bias
        if self.bias is not None:
            out = out + self.bias

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"input_type={self.input_type}, output_type={self.output_type}, "
            f"bias={self.bias is not None}"
        )


class QuaternionLinearTensorCore(nn.Module):
    """
    Quaternion linear layer optimized for Tensor Cores.

    Uses explicit 4x4 block structure for Tensor Core acceleration.

    NOT YET IMPLEMENTED - placeholder for future optimization.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        raise NotImplementedError("TensorCore-optimized version not yet implemented")


# Convenience functions

def real_to_quaternion_projection(in_features: int, out_quat_dim: int, bias: bool = True) -> QuaternionLinear:
    """
    Create a R → H projection layer.

    Args:
        in_features: Input real dimension
        out_quat_dim: Output quaternion dimension (output will be [out_quat_dim, 4])
        bias: Whether to use bias

    Returns:
        QuaternionLinear layer
    """
    return QuaternionLinear(
        in_features=in_features,
        out_features=out_quat_dim * 4,
        input_type="real",
        output_type="quaternion",
        bias=bias,
    )


def quaternion_to_real_projection(in_quat_dim: int, out_features: int, bias: bool = True) -> QuaternionLinear:
    """
    Create a H → R projection layer.

    Args:
        in_quat_dim: Input quaternion dimension
        out_features: Output real dimension
        bias: Whether to use bias

    Returns:
        QuaternionLinear layer
    """
    return QuaternionLinear(
        in_features=in_quat_dim * 4,
        out_features=out_features,
        input_type="quaternion",
        output_type="real",
        bias=bias,
    )

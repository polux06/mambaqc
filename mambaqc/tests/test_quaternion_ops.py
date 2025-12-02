"""
Unit tests for quaternion operations.

Tests mathematical correctness of:
- Quaternion multiplication (Hamilton product)
- Quaternion conjugation
- Quaternion inverse
- Quaternion norm
- Cayley discretization
- Parallel scan
"""

import torch
import pytest
import math

from mambaqc.kernels.quaternion_ops import (
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_norm,
)
from mambaqc.kernels.cayley_transform import (
    cayley_discretization_fused,
    cayley_discretization_reference,
)
from mambaqc.kernels.parallel_scan import (
    parallel_scan_quaternion,
    parallel_scan_quaternion_reference,
)


class TestQuaternionMultiplication:
    """Test Hamilton quaternion multiplication."""

    def test_identity_multiplication(self):
        """Test q * 1 = q"""
        q = torch.randn(10, 4)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(q)

        result = quaternion_multiply(q, identity)

        assert torch.allclose(result, q, atol=1e-6)

    def test_conjugate_product(self):
        """Test q * q̄ = ||q||² * (1, 0, 0, 0)"""
        q = torch.randn(10, 4)
        q_conj = quaternion_conjugate(q)

        result = quaternion_multiply(q, q_conj)

        # Expected: (||q||², 0, 0, 0)
        norm_sq = (q ** 2).sum(dim=-1, keepdim=True)
        expected = torch.cat([norm_sq, torch.zeros(10, 3)], dim=-1)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_inverse_property(self):
        """Test q * q^{-1} = 1"""
        q = torch.randn(10, 4) + 0.5  # Avoid near-zero quaternions
        q_inv = quaternion_inverse(q)

        result = quaternion_multiply(q, q_inv)

        identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(result)

        assert torch.allclose(result, identity, atol=1e-5)

    def test_associativity(self):
        """Test (p * q) * r = p * (q * r)"""
        p = torch.randn(5, 4)
        q = torch.randn(5, 4)
        r = torch.randn(5, 4)

        # Left association
        pq = quaternion_multiply(p, q)
        pq_r = quaternion_multiply(pq, r)

        # Right association
        qr = quaternion_multiply(q, r)
        p_qr = quaternion_multiply(p, qr)

        assert torch.allclose(pq_r, p_qr, atol=1e-5)

    def test_non_commutativity(self):
        """Test that q * p ≠ p * q (in general)"""
        p = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        q = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

        pq = quaternion_multiply(p, q)
        qp = quaternion_multiply(q, p)

        # Should NOT be equal (quaternions are non-commutative)
        assert not torch.allclose(pq, qp, atol=1e-6)

    def test_norm_multiplicativity(self):
        """Test ||p * q|| = ||p|| * ||q||"""
        p = torch.randn(10, 4)
        q = torch.randn(10, 4)

        pq = quaternion_multiply(p, q)

        norm_p = quaternion_norm(p)
        norm_q = quaternion_norm(q)
        norm_pq = quaternion_norm(pq)

        expected = norm_p * norm_q

        assert torch.allclose(norm_pq, expected, rtol=1e-5)


class TestQuaternionConjugate:
    """Test quaternion conjugation."""

    def test_conjugate_twice(self):
        """Test (q̄)̄ = q"""
        q = torch.randn(10, 4)
        q_conj = quaternion_conjugate(q)
        q_conj_conj = quaternion_conjugate(q_conj)

        assert torch.allclose(q_conj_conj, q, atol=1e-6)

    def test_conjugate_real_part(self):
        """Test that real part is unchanged"""
        q = torch.randn(10, 4)
        q_conj = quaternion_conjugate(q)

        assert torch.allclose(q[:, 0], q_conj[:, 0], atol=1e-6)

    def test_conjugate_imaginary_sign(self):
        """Test that imaginary parts flip sign"""
        q = torch.randn(10, 4)
        q_conj = quaternion_conjugate(q)

        assert torch.allclose(q[:, 1:], -q_conj[:, 1:], atol=1e-6)


class TestQuaternionInverse:
    """Test quaternion inverse."""

    def test_inverse_identity(self):
        """Test 1^{-1} = 1"""
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        inv = quaternion_inverse(identity)

        assert torch.allclose(inv, identity, atol=1e-6)

    def test_inverse_twice(self):
        """Test (q^{-1})^{-1} = q"""
        q = torch.randn(10, 4) + 1.0  # Avoid near-zero
        q_inv = quaternion_inverse(q)
        q_inv_inv = quaternion_inverse(q_inv)

        assert torch.allclose(q_inv_inv, q, atol=1e-5)

    def test_inverse_norm(self):
        """Test ||q^{-1}|| = 1 / ||q||"""
        q = torch.randn(10, 4) + 1.0
        q_inv = quaternion_inverse(q)

        norm_q = quaternion_norm(q)
        norm_q_inv = quaternion_norm(q_inv)

        expected = 1.0 / norm_q

        assert torch.allclose(norm_q_inv, expected, rtol=1e-5)


class TestQuaternionNorm:
    """Test quaternion norm."""

    def test_norm_positive(self):
        """Test ||q|| >= 0"""
        q = torch.randn(100, 4)
        norm = quaternion_norm(q)

        assert (norm >= 0).all()

    def test_norm_identity(self):
        """Test ||1|| = 1"""
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        norm = quaternion_norm(identity)

        assert torch.allclose(norm, torch.tensor([1.0]), atol=1e-6)

    def test_norm_zero(self):
        """Test ||0|| = 0"""
        zero = torch.zeros(1, 4)
        norm = quaternion_norm(zero)

        assert torch.allclose(norm, torch.tensor([0.0]), atol=1e-6)


class TestCayleyDiscretization:
    """Test Cayley transform."""

    def test_cayley_identity(self):
        """Test Cayley(0) = identity"""
        z = torch.zeros(2, 4, 8, 16, 4)
        q = cayley_discretization_fused(z)

        # Expected: identity quaternion (1, 0, 0, 0)
        identity = torch.zeros_like(q)
        identity[..., 0] = 1.0

        assert torch.allclose(q, identity, atol=1e-5)

    def test_cayley_stability(self):
        """Test |q| < 1 when Re(z) < 0"""
        # Create z with negative real part
        z = torch.randn(2, 4, 8, 16, 4)
        z[..., 0] = -torch.abs(z[..., 0]) - 0.5  # Ensure Re(z) < 0

        q = cayley_discretization_fused(z)

        # Compute norms
        norms = torch.sqrt((q ** 2).sum(dim=-1))

        # All norms should be < 1 (stability)
        assert (norms < 1.0 + 1e-4).all(), f"Max norm: {norms.max()}"

    def test_cayley_vs_reference(self):
        """Test fused kernel vs reference implementation"""
        z = torch.randn(2, 4, 8, 16, 4)

        q_fused = cayley_discretization_fused(z)
        q_ref = cayley_discretization_reference(z)

        assert torch.allclose(q_fused, q_ref, atol=1e-5)


class TestParallelScan:
    """Test parallel scan algorithm."""

    def test_scan_identity_sequence(self):
        """Test scan of identity quaternions"""
        identity = torch.zeros(2, 10, 8, 16, 4)
        identity[..., 0] = 1.0

        result = parallel_scan_quaternion(identity)

        # Cumulative product of identities is still identity
        assert torch.allclose(result, identity, atol=1e-5)

    def test_scan_vs_reference(self):
        """Test parallel scan vs sequential reference"""
        q_seq = torch.randn(2, 16, 8, 16, 4)

        # Parallel scan
        result_parallel = parallel_scan_quaternion(q_seq)

        # Sequential scan
        result_sequential = parallel_scan_quaternion_reference(q_seq)

        assert torch.allclose(result_parallel, result_sequential, atol=1e-4)

    def test_scan_associativity(self):
        """Test that scan is independent of grouping (associativity)"""
        q_seq = torch.randn(1, 8, 4, 8, 4)

        # Full scan
        result = parallel_scan_quaternion(q_seq)

        # Manual check: last element should be product of all
        manual_product = q_seq[:, 0]
        for t in range(1, 8):
            manual_product = quaternion_multiply(q_seq[:, t], manual_product)

        assert torch.allclose(result[:, -1], manual_product, atol=1e-4)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

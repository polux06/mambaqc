"""
Tests de Validation pour Quaternion Mamba-2 Optimisé
====================================================

Ce script teste:
1. Les propriétés mathématiques des quaternions
2. La correction des kernels Triton vs PyTorch
3. La stabilité numérique
4. Les gradients (gradcheck)
5. Les performances (benchmarks)
"""

import torch
import numpy as np
import time
from typing import Tuple

# Import du modèle optimisé
from quaternion_mamba_optimized import (
    QuaternionMambaConfig,
    OptimizedQuaternionMambaLM,
    QuaternionOps,
)

# Import des kernels
try:
    from kernels import (
        quat_mul_triton,
        quat_inv_triton,
        quat_norm_triton,
        cayley_discretization_triton,
    )
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("⚠️  Triton non disponible, certains tests seront sautés")


# =============================================================================
# TESTS DE PROPRIÉTÉS MATHÉMATIQUES
# =============================================================================

def test_quaternion_properties():
    """
    Teste les propriétés fondamentales des quaternions:
    - Associativité: (a⊗b)⊗c = a⊗(b⊗c)
    - Norme multiplicative: ||a⊗b|| = ||a|| × ||b||
    - Inverse: q⊗q^{-1} = 1
    """
    print("\n" + "="*70)
    print("TEST 1: Propriétés Mathématiques des Quaternions")
    print("="*70)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Générer des quaternions aléatoires
    a = torch.randn(100, 4, device=device)
    b = torch.randn(100, 4, device=device)
    c = torch.randn(100, 4, device=device)

    # Test 1: Associativité
    print("\n[1.1] Associativité: (a⊗b)⊗c = a⊗(b⊗c)")
    ab = QuaternionOps.mul(a, b, use_triton=False)
    ab_c = QuaternionOps.mul(ab, c, use_triton=False)

    bc = QuaternionOps.mul(b, c, use_triton=False)
    a_bc = QuaternionOps.mul(a, bc, use_triton=False)

    error_assoc = (ab_c - a_bc).abs().max().item()
    print(f"   Erreur max: {error_assoc:.2e}")
    assert error_assoc < 1e-5, f"Associativité échouée: {error_assoc}"
    print("   ✓ Associativité vérifiée")

    # Test 2: Norme multiplicative
    print("\n[1.2] Norme multiplicative: ||a⊗b|| = ||a|| × ||b||")
    norm_a = QuaternionOps.norm(a, use_triton=False)
    norm_b = QuaternionOps.norm(b, use_triton=False)
    norm_ab = QuaternionOps.norm(ab, use_triton=False)
    norm_product = norm_a * norm_b

    error_norm = (norm_ab - norm_product).abs().max().item()
    print(f"   Erreur max: {error_norm:.2e}")
    assert error_norm < 1e-4, f"Norme multiplicative échouée: {error_norm}"
    print("   ✓ Norme multiplicative vérifiée")

    # Test 3: Inverse
    print("\n[1.3] Inverse: q⊗q^{-1} = (1,0,0,0)")
    q = torch.randn(100, 4, device=device)
    q_inv = QuaternionOps.inv(q, use_triton=False)
    identity = QuaternionOps.mul(q, q_inv, use_triton=False)

    expected_identity = torch.zeros_like(identity)
    expected_identity[:, 0] = 1.0

    error_inv = (identity - expected_identity).abs().max().item()
    print(f"   Erreur max: {error_inv:.2e}")
    assert error_inv < 1e-4, f"Inverse échoué: {error_inv}"
    print("   ✓ Inverse vérifié")

    print("\n✅ Tous les tests de propriétés mathématiques passés!")
    return True


# =============================================================================
# TESTS DE CORRECTION TRITON vs PYTORCH
# =============================================================================

def test_triton_correctness():
    """
    Compare les kernels Triton avec l'implémentation PyTorch de référence.
    """
    if not TRITON_AVAILABLE:
        print("\n⚠️  Tests Triton sautés (non disponible)")
        return True

    if not torch.cuda.is_available():
        print("\n⚠️  Tests Triton sautés (CUDA non disponible)")
        return True

    print("\n" + "="*70)
    print("TEST 2: Correction Triton vs PyTorch")
    print("="*70)

    torch.manual_seed(42)
    device = torch.device("cuda")

    # Test multiplication
    print("\n[2.1] Multiplication quaternionique")
    a = torch.randn(1000, 4, device=device)
    b = torch.randn(1000, 4, device=device)

    result_pytorch = QuaternionOps.mul(a, b, use_triton=False)

    try:
        result_triton = quat_mul_triton(a, b)
        error = (result_pytorch - result_triton).abs().max().item()
        print(f"   Erreur max: {error:.2e}")
        assert error < 1e-5, f"Différence trop grande: {error}"
        print("   ✓ quat_mul_triton correct")
    except Exception as e:
        print(f"   ✗ quat_mul_triton échoué: {e}")
        return False

    # Test inverse
    print("\n[2.2] Inverse quaternionique")
    q = torch.randn(1000, 4, device=device)

    result_pytorch = QuaternionOps.inv(q, use_triton=False)

    try:
        result_triton = quat_inv_triton(q)
        error = (result_pytorch - result_triton).abs().max().item()
        print(f"   Erreur max: {error:.2e}")
        assert error < 1e-5, f"Différence trop grande: {error}"
        print("   ✓ quat_inv_triton correct")
    except Exception as e:
        print(f"   ✗ quat_inv_triton échoué: {e}")
        return False

    # Test norme
    print("\n[2.3] Norme quaternionique")
    q = torch.randn(1000, 4, device=device)

    result_pytorch = QuaternionOps.norm(q, use_triton=False)

    try:
        result_triton = quat_norm_triton(q)
        error = (result_pytorch - result_triton).abs().max().item()
        print(f"   Erreur max: {error:.2e}")
        assert error < 1e-5, f"Différence trop grande: {error}"
        print("   ✓ quat_norm_triton correct")
    except Exception as e:
        print(f"   ✗ quat_norm_triton échoué: {e}")
        return False

    # Test Cayley
    print("\n[2.4] Discrétisation de Cayley")
    z = torch.randn(1000, 4, device=device) * 0.1

    # Référence PyTorch
    ones = torch.zeros_like(z)
    ones[:, 0] = 1.0
    num = ones + 0.5 * z
    den = ones - 0.5 * z
    den_inv = QuaternionOps.inv(den, use_triton=False)
    result_pytorch = QuaternionOps.mul(den_inv, num, use_triton=False)

    try:
        result_triton = cayley_discretization_triton(z)
        error = (result_pytorch - result_triton).abs().max().item()
        print(f"   Erreur max: {error:.2e}")
        assert error < 1e-4, f"Différence trop grande: {error}"
        print("   ✓ cayley_discretization_triton correct")
    except Exception as e:
        print(f"   ✗ cayley_discretization_triton échoué: {e}")
        return False

    print("\n✅ Tous les tests de correction Triton passés!")
    return True


# =============================================================================
# TESTS DE STABILITÉ NUMÉRIQUE
# =============================================================================

def test_numerical_stability():
    """
    Teste la stabilité numérique:
    - Pas de NaN/Inf
    - Gradients bien formés
    - Stabilité de Cayley (||q|| < 1 si Re(z) < 0)
    """
    print("\n" + "="*70)
    print("TEST 3: Stabilité Numérique")
    print("="*70)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test 1: Forward sans NaN
    print("\n[3.1] Forward pass sans NaN/Inf")
    config = QuaternionMambaConfig(
        d_model=64,
        n_layers=2,
        vocab_size=100,
        d_state=16,
    )
    model = OptimizedQuaternionMambaLM(config).to(device)

    x = torch.randint(0, 100, (2, 32), device=device)
    logits, loss = model(x, targets=x)

    assert not torch.isnan(logits).any(), "NaN détecté dans logits"
    assert not torch.isinf(logits).any(), "Inf détecté dans logits"
    assert not torch.isnan(loss), "NaN détecté dans loss"
    print(f"   ✓ Forward stable (loss={loss.item():.4f})")

    # Test 2: Backward sans NaN
    print("\n[3.2] Backward pass sans NaN/Inf")
    loss.backward()

    has_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"   ✗ NaN/Inf dans gradient de {name}")
                has_nan = True

    assert not has_nan, "Gradients contiennent NaN/Inf"
    print("   ✓ Backward stable")

    # Test 3: Stabilité de Cayley
    print("\n[3.3] Stabilité de Cayley: ||q|| < 1 si Re(z) < 0")
    z = torch.randn(1000, 4, device=device) * 0.5
    z[:, 0] = -torch.abs(z[:, 0])  # Assurer Re(z) < 0

    # PyTorch
    ones = torch.zeros_like(z)
    ones[:, 0] = 1.0
    num = ones + 0.5 * z
    den = ones - 0.5 * z
    den_inv = QuaternionOps.inv(den, use_triton=False)
    q = QuaternionOps.mul(den_inv, num, use_triton=False)

    norms = QuaternionOps.norm(q, use_triton=False)
    max_norm = norms.max().item()
    print(f"   Norme max: {max_norm:.4f}")
    assert max_norm < 1.0, f"Stabilité Cayley échouée: max ||q|| = {max_norm}"
    print("   ✓ Cayley stable (toutes les normes < 1)")

    print("\n✅ Tous les tests de stabilité numérique passés!")
    return True


# =============================================================================
# TESTS DE GRADIENTS (GRADCHECK)
# =============================================================================

def test_gradients():
    """
    Vérifie que les gradients sont correctement calculés via gradcheck.
    """
    print("\n" + "="*70)
    print("TEST 4: Vérification des Gradients (gradcheck)")
    print("="*70)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test sur une petite configuration
    print("\n[4.1] Gradcheck sur multiplication quaternionique")

    def quat_mul_fn(a, b):
        return QuaternionOps.mul(a, b, use_triton=False)

    a = torch.randn(10, 4, device=device, dtype=torch.float64, requires_grad=True)
    b = torch.randn(10, 4, device=device, dtype=torch.float64, requires_grad=True)

    try:
        torch.autograd.gradcheck(quat_mul_fn, (a, b), eps=1e-6, atol=1e-4)
        print("   ✓ Gradcheck multiplication réussi")
    except Exception as e:
        print(f"   ✗ Gradcheck échoué: {e}")
        return False

    # Test sur inverse
    print("\n[4.2] Gradcheck sur inverse quaternionique")

    def quat_inv_fn(q):
        return QuaternionOps.inv(q, use_triton=False)

    q = torch.randn(10, 4, device=device, dtype=torch.float64, requires_grad=True)

    try:
        torch.autograd.gradcheck(quat_inv_fn, (q,), eps=1e-6, atol=1e-4)
        print("   ✓ Gradcheck inverse réussi")
    except Exception as e:
        print(f"   ✗ Gradcheck échoué: {e}")
        return False

    print("\n✅ Tous les tests de gradients passés!")
    return True


# =============================================================================
# BENCHMARKS DE PERFORMANCE
# =============================================================================

def benchmark_performance():
    """
    Benchmark des performances Triton vs PyTorch.
    """
    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        print("\n⚠️  Benchmarks sautés (Triton/CUDA non disponible)")
        return True

    print("\n" + "="*70)
    print("TEST 5: Benchmarks de Performance")
    print("="*70)

    device = torch.device("cuda")
    sizes = [100, 1000, 10000]

    # Benchmark multiplication
    print("\n[5.1] Benchmark multiplication quaternionique")
    print(f"{'Size':<10} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for size in sizes:
        a = torch.randn(size, 4, device=device)
        b = torch.randn(size, 4, device=device)

        # Warmup
        for _ in range(10):
            _ = QuaternionOps.mul(a, b, use_triton=False)
            if TRITON_AVAILABLE:
                _ = quat_mul_triton(a, b)
        torch.cuda.synchronize()

        # PyTorch
        start = time.time()
        for _ in range(100):
            _ = QuaternionOps.mul(a, b, use_triton=False)
        torch.cuda.synchronize()
        time_pytorch = (time.time() - start) * 10  # ms

        # Triton
        if TRITON_AVAILABLE:
            start = time.time()
            for _ in range(100):
                _ = quat_mul_triton(a, b)
            torch.cuda.synchronize()
            time_triton = (time.time() - start) * 10  # ms

            speedup = time_pytorch / time_triton
            print(f"{size:<10} {time_pytorch:<15.3f} {time_triton:<15.3f} {speedup:<10.2f}x")
        else:
            print(f"{size:<10} {time_pytorch:<15.3f} {'N/A':<15} {'N/A':<10}")

    print("\n✅ Benchmarks terminés!")
    return True


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    """
    Lance tous les tests dans l'ordre.
    """
    print("\n" + "="*70)
    print(" SUITE DE TESTS COMPLÈTE - QUATERNION MAMBA-2 OPTIMISÉ")
    print("="*70)

    results = []

    # Test 1: Propriétés mathématiques
    try:
        results.append(("Propriétés mathématiques", test_quaternion_properties()))
    except Exception as e:
        print(f"\n❌ Test échoué: {e}")
        results.append(("Propriétés mathématiques", False))

    # Test 2: Correction Triton
    try:
        results.append(("Correction Triton", test_triton_correctness()))
    except Exception as e:
        print(f"\n❌ Test échoué: {e}")
        results.append(("Correction Triton", False))

    # Test 3: Stabilité numérique
    try:
        results.append(("Stabilité numérique", test_numerical_stability()))
    except Exception as e:
        print(f"\n❌ Test échoué: {e}")
        results.append(("Stabilité numérique", False))

    # Test 4: Gradients
    try:
        results.append(("Gradients", test_gradients()))
    except Exception as e:
        print(f"\n❌ Test échoué: {e}")
        results.append(("Gradients", False))

    # Test 5: Benchmarks
    try:
        results.append(("Benchmarks", benchmark_performance()))
    except Exception as e:
        print(f"\n❌ Test échoué: {e}")
        results.append(("Benchmarks", False))

    # Résumé
    print("\n" + "="*70)
    print(" RÉSUMÉ DES TESTS")
    print("="*70)

    for name, passed in results:
        status = "✅ PASSÉ" if passed else "❌ ÉCHOUÉ"
        print(f"{name:<30} {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + "="*70)
    if all_passed:
        print("✅ TOUS LES TESTS ONT RÉUSSI!")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
    print("="*70 + "\n")

    return all_passed


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)

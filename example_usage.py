"""
Exemple d'Utilisation - Quaternion Mamba-2 Optimisé
===================================================

Ce script montre comment utiliser le modèle Quaternion Mamba-2 optimisé
pour différentes tâches.
"""

import torch
import torch.nn.functional as F
from quaternion_mamba_optimized import (
    QuaternionMambaConfig,
    OptimizedQuaternionMambaLM,
    QuaternionOps,
)


# =============================================================================
# EXEMPLE 1: MODÈLE DE LANGAGE SIMPLE
# =============================================================================

def example_language_model():
    """
    Créer et utiliser un modèle de langage quaternionique.
    """
    print("\n" + "="*70)
    print("EXEMPLE 1: Modèle de Langage")
    print("="*70)

    # Configuration
    config = QuaternionMambaConfig(
        d_model=256,          # Petit modèle pour l'exemple
        n_layers=6,
        vocab_size=1000,
        d_state=32,
        d_conv=4,
        expand=2,
        use_triton=True
    )

    print(f"\nConfiguration:")
    print(f"  - d_model: {config.d_model}")
    print(f"  - n_layers: {config.n_layers}")
    print(f"  - vocab_size: {config.vocab_size}")
    print(f"  - d_state: {config.d_state}")
    print(f"  - Triton: {config.use_triton}")

    # Créer le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptimizedQuaternionMambaLM(config).to(device)

    # Compter les paramètres
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nNombre de paramètres: {n_params:,}")

    # Forward pass
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    print(f"\nInput shape: {input_ids.shape}")

    with torch.no_grad():
        logits, _ = model(input_ids)

    print(f"Output shape: {logits.shape}")
    print(f"Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

    # Génération autoregressive (simple)
    print("\n[Génération de tokens]")
    context = torch.randint(0, config.vocab_size, (1, 10), device=device)
    generated = context.clone()

    for _ in range(20):
        with torch.no_grad():
            logits, _ = model(generated)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

    print(f"Context length: {context.shape[1]}")
    print(f"Generated length: {generated.shape[1]}")
    print(f"Tokens: {generated[0, :15].tolist()}")

    print("\n✅ Exemple 1 terminé!")


# =============================================================================
# EXEMPLE 2: ENTRAÎNEMENT SIMPLE
# =============================================================================

def example_training():
    """
    Montrer comment entraîner le modèle sur des données synthétiques.
    """
    print("\n" + "="*70)
    print("EXEMPLE 2: Boucle d'Entraînement")
    print("="*70)

    # Config plus petite pour l'entraînement rapide
    config = QuaternionMambaConfig(
        d_model=128,
        n_layers=4,
        vocab_size=500,
        d_state=16,
        use_triton=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptimizedQuaternionMambaLM(config).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Données synthétiques
    batch_size = 8
    seq_len = 64
    n_steps = 10

    print(f"\nConfiguration d'entraînement:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Seq length: {seq_len}")
    print(f"  - Steps: {n_steps}")

    # Boucle d'entraînement
    model.train()
    losses = []

    for step in range(n_steps):
        # Générer batch synthétique
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        # Forward
        logits, loss = model(input_ids, targets=targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update
        optimizer.step()

        losses.append(loss.item())

        if step % 2 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.4f}")

    print(f"\nLoss finale: {losses[-1]:.4f}")
    print(f"Réduction de loss: {losses[0] - losses[-1]:.4f}")

    print("\n✅ Exemple 2 terminé!")


# =============================================================================
# EXEMPLE 3: OPÉRATIONS QUATERNIONIQUES DIRECTES
# =============================================================================

def example_quaternion_ops():
    """
    Montrer l'utilisation directe des opérations quaternioniques.
    """
    print("\n" + "="*70)
    print("EXEMPLE 3: Opérations Quaternioniques")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Créer des quaternions
    print("\n[Création de quaternions]")
    a = torch.randn(5, 4, device=device)
    b = torch.randn(5, 4, device=device)

    print(f"a shape: {a.shape}")
    print(f"b shape: {b.shape}")
    print(f"\na[0] = [{a[0, 0]:.2f}, {a[0, 1]:.2f}i, {a[0, 2]:.2f}j, {a[0, 3]:.2f}k]")
    print(f"b[0] = [{b[0, 0]:.2f}, {b[0, 1]:.2f}i, {b[0, 2]:.2f}j, {b[0, 3]:.2f}k]")

    # Multiplication
    print("\n[Multiplication: a ⊗ b]")
    c = QuaternionOps.mul(a, b, use_triton=True)
    print(f"c[0] = [{c[0, 0]:.2f}, {c[0, 1]:.2f}i, {c[0, 2]:.2f}j, {c[0, 3]:.2f}k]")

    # Vérifier la norme multiplicative
    norm_a = QuaternionOps.norm(a, use_triton=True)
    norm_b = QuaternionOps.norm(b, use_triton=True)
    norm_c = QuaternionOps.norm(c, use_triton=True)

    print(f"\n||a|| = {norm_a[0]:.4f}")
    print(f"||b|| = {norm_b[0]:.4f}")
    print(f"||a⊗b|| = {norm_c[0]:.4f}")
    print(f"||a|| × ||b|| = {(norm_a[0] * norm_b[0]):.4f}")
    print(f"Différence: {abs(norm_c[0] - norm_a[0] * norm_b[0]):.2e}")

    # Inverse
    print("\n[Inverse: a⊗a⁻¹ = 1]")
    a_inv = QuaternionOps.inv(a, use_triton=True)
    identity = QuaternionOps.mul(a, a_inv, use_triton=True)

    print(f"a⊗a⁻¹[0] = [{identity[0, 0]:.2f}, {identity[0, 1]:.2e}i, "
          f"{identity[0, 2]:.2e}j, {identity[0, 3]:.2e}k]")
    print(f"Erreur vs (1,0,0,0): {(identity[0] - torch.tensor([1,0,0,0], device=device)).abs().max():.2e}")

    # Associativité
    print("\n[Associativité: (a⊗b)⊗c = a⊗(b⊗c)]")
    c_tensor = torch.randn(5, 4, device=device)

    ab = QuaternionOps.mul(a, b, use_triton=True)
    ab_c = QuaternionOps.mul(ab, c_tensor, use_triton=True)

    bc = QuaternionOps.mul(b, c_tensor, use_triton=True)
    a_bc = QuaternionOps.mul(a, bc, use_triton=True)

    error = (ab_c - a_bc).abs().max()
    print(f"Erreur max: {error:.2e}")

    print("\n✅ Exemple 3 terminé!")


# =============================================================================
# EXEMPLE 4: BENCHMARKING
# =============================================================================

def example_benchmark():
    """
    Comparer les performances Triton vs PyTorch.
    """
    print("\n" + "="*70)
    print("EXEMPLE 4: Benchmark de Performance")
    print("="*70)

    if not torch.cuda.is_available():
        print("\n⚠️  CUDA non disponible, benchmark sauté")
        return

    import time

    device = torch.device("cuda")
    size = 10000

    # Préparation
    a = torch.randn(size, 4, device=device)
    b = torch.randn(size, 4, device=device)

    # Warmup
    for _ in range(20):
        _ = QuaternionOps.mul(a, b, use_triton=False)
        _ = QuaternionOps.mul(a, b, use_triton=True)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    n_iter = 100
    start = time.time()
    for _ in range(n_iter):
        _ = QuaternionOps.mul(a, b, use_triton=False)
    torch.cuda.synchronize()
    time_pytorch = (time.time() - start) / n_iter * 1000  # ms

    # Benchmark Triton
    start = time.time()
    for _ in range(n_iter):
        _ = QuaternionOps.mul(a, b, use_triton=True)
    torch.cuda.synchronize()
    time_triton = (time.time() - start) / n_iter * 1000  # ms

    print(f"\nBenchmark multiplication quaternionique ({size} éléments):")
    print(f"  - PyTorch: {time_pytorch:.3f} ms")
    print(f"  - Triton:  {time_triton:.3f} ms")
    print(f"  - Speedup: {time_pytorch / time_triton:.2f}×")

    print("\n✅ Exemple 4 terminé!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Lance tous les exemples.
    """
    print("\n" + "="*70)
    print(" EXEMPLES D'UTILISATION - QUATERNION MAMBA-2 OPTIMISÉ")
    print("="*70)

    # Exemple 1: Modèle de langage
    example_language_model()

    # Exemple 2: Entraînement
    example_training()

    # Exemple 3: Opérations quaternioniques
    example_quaternion_ops()

    # Exemple 4: Benchmark
    example_benchmark()

    print("\n" + "="*70)
    print(" ✅ TOUS LES EXEMPLES TERMINÉS!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

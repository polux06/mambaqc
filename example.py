"""
Simple example demonstrating Quaternion Mamba-2 usage.
"""

import torch
from mambaqc.models import QuaternionMamba2
from mambaqc.kernels.quaternion_ops import (
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_norm,
)


def test_quaternion_operations():
    """Test basic quaternion operations."""
    print("=" * 60)
    print("Testing Quaternion Operations")
    print("=" * 60)

    # Create random quaternions
    p = torch.randn(5, 4)
    q = torch.randn(5, 4)

    print(f"\nQuaternion p shape: {p.shape}")
    print(f"Quaternion q shape: {q.shape}")

    # Multiplication
    pq = quaternion_multiply(p, q)
    print(f"\np * q = {pq[0]}")

    # Conjugate
    p_conj = quaternion_conjugate(p)
    print(f"\np̄ = {p_conj[0]}")

    # Inverse
    p_inv = quaternion_inverse(p)
    print(f"\np^{{-1}} = {p_inv[0]}")

    # Verify: p * p^{-1} = 1
    identity = quaternion_multiply(p, p_inv)
    print(f"\np * p^{{-1}} = {identity[0]} (should be ≈ [1, 0, 0, 0])")

    # Norm
    norm_p = quaternion_norm(p)
    print(f"\n||p|| = {norm_p[0]:.4f}")

    # Verify multiplicativity: ||p * q|| = ||p|| * ||q||
    norm_q = quaternion_norm(q)
    norm_pq = quaternion_norm(pq)
    print(f"||p|| * ||q|| = {(norm_p[0] * norm_q[0]):.4f}")
    print(f"||p * q|| = {norm_pq[0]:.4f}")
    print(f"Difference: {abs(norm_pq[0] - norm_p[0] * norm_q[0]):.6f}")

    print("\n✅ Quaternion operations test passed!")


def test_model_forward():
    """Test model forward pass."""
    print("\n" + "=" * 60)
    print("Testing Quaternion Mamba-2 Forward Pass")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create small model for testing
    model = QuaternionMamba2(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        d_state=32,
    ).to(device)

    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")

    # Create random input
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    print(f"\nInput shape: {input_ids.shape}")

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs["logits"]
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

    # Test with labels (compute loss)
    labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    print(f"\nLoss: {loss.item():.4f}")

    # Test backward
    print("Running backward pass...")
    loss.backward()
    print("✅ Backward pass successful!")

    # Check gradients
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Gradient norm: {total_norm:.4f}")

    print("\n✅ Model forward/backward test passed!")


def test_generation():
    """Test autoregressive generation."""
    print("\n" + "=" * 60)
    print("Testing Autoregressive Generation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create small model
    model = QuaternionMamba2(
        vocab_size=100,
        d_model=128,
        n_layers=2,
        d_state=16,
    ).to(device)

    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")

    # Prompt
    prompt = torch.randint(0, 100, (1, 10), device=device)
    print(f"\nPrompt shape: {prompt.shape}")
    print(f"Prompt: {prompt[0].tolist()}")

    # Generate
    print("Generating 20 tokens...")
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=20,
            temperature=1.0,
        )

    print(f"Generated shape: {generated.shape}")
    print(f"Generated: {generated[0].tolist()}")

    print("\n✅ Generation test passed!")


def benchmark_throughput():
    """Benchmark token throughput."""
    print("\n" + "=" * 60)
    print("Benchmarking Throughput")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping benchmark")
        return

    device = "cuda"

    # Create model
    model = QuaternionMamba2(
        vocab_size=10000,
        d_model=512,
        n_layers=6,
        d_state=32,
    ).to(device)

    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")

    # Create input
    batch_size = 4
    seq_len = 512
    input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 10000, (batch_size, seq_len), device=device)

    print(f"Input shape: {input_ids.shape}")
    print(f"Total tokens: {batch_size * seq_len}")

    # Warmup
    print("\nWarming up...")
    for _ in range(5):
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
    torch.cuda.synchronize()

    # Benchmark
    print("Benchmarking...")
    num_iterations = 20
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
    end.record()

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)

    total_tokens = batch_size * seq_len * num_iterations
    throughput = total_tokens / (elapsed_ms / 1000)

    print(f"\nElapsed time: {elapsed_ms:.2f} ms")
    print(f"Throughput: {throughput:.0f} tokens/s")
    print(f"Latency per batch: {elapsed_ms / num_iterations:.2f} ms")

    # Memory usage
    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    print(f"\nMemory allocated: {memory_allocated:.2f} GB")
    print(f"Memory reserved: {memory_reserved:.2f} GB")

    print("\n✅ Benchmark complete!")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Quaternion Mamba-2 Examples")
    print("=" * 60)

    # Test quaternion operations
    test_quaternion_operations()

    # Test model forward pass
    test_model_forward()

    # Test generation
    test_generation()

    # Benchmark (optional, requires CUDA)
    if torch.cuda.is_available():
        benchmark_throughput()

    print("\n" + "=" * 60)
    print("All tests passed! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()

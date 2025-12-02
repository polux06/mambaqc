# Contributing to Quaternion Mamba-2

Thank you for your interest in contributing to Quaternion Mamba-2! This document provides guidelines for contributions.

## 🚀 Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/polux06/mambaqc.git
   cd mambaqc
   ```

2. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 Code Style

We follow PEP 8 style guidelines with some modifications:

- Line length: 100 characters
- Use type hints where possible
- Use descriptive variable names

### Formatting

We use `black` for code formatting:

```bash
black mambaqc/
```

And `isort` for import sorting:

```bash
isort mambaqc/
```

### Linting

Run `flake8` to check for issues:

```bash
flake8 mambaqc/ --max-line-length=100
```

## 🧪 Testing

All new features should include tests.

### Running tests

```bash
# Run all tests
pytest mambaqc/tests/ -v

# Run specific test file
pytest mambaqc/tests/test_quaternion_ops.py -v

# Run with coverage
pytest mambaqc/tests/ --cov=mambaqc --cov-report=html
```

### Writing tests

- Place tests in `mambaqc/tests/`
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Include docstrings explaining what is tested

Example:

```python
def test_quaternion_multiplication_identity():
    """Test that q * 1 = q for all quaternions."""
    q = torch.randn(10, 4)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(q)

    result = quaternion_multiply(q, identity)

    assert torch.allclose(result, q, atol=1e-6)
```

## 🔧 Development Guidelines

### Kernel Development

When writing Triton kernels:

1. **Memory efficiency**: Minimize HBM traffic
2. **Tiling**: Use appropriate tile sizes for shared memory
3. **Coalescing**: Ensure memory accesses are coalesced
4. **Documentation**: Document memory access patterns
5. **Testing**: Compare against reference implementation

Example:

```python
@triton.jit
def my_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Brief description.

    Memory access pattern:
    - Input: Sequential coalesced reads
    - Output: Sequential coalesced writes

    Args:
        input_ptr: [n_elements] input tensor
        output_ptr: [n_elements] output tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Block size (constexpr)
    """
    # Implementation
    pass
```

### Model Development

When adding model components:

1. **Documentation**: Clear docstrings with shapes
2. **Initialization**: Proper weight initialization
3. **Shape checking**: Assert expected shapes
4. **Testing**: Unit tests for each component

### Mathematical Correctness

For quaternion operations:

1. **Verify properties**: Test associativity, multiplicativity, etc.
2. **Reference implementation**: Provide naive version for comparison
3. **Numerical stability**: Test edge cases (near-zero quaternions, etc.)
4. **Precision**: Test with different dtypes (fp16, fp32, bf16)

## 📚 Documentation

- Add docstrings to all public functions and classes
- Use Google style docstrings
- Include shape annotations for tensors
- Provide usage examples

Example:

```python
def quaternion_multiply(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Hamilton quaternion multiplication: p * q

    Properties:
        - Associative: (pq)r = p(qr)
        - Non-commutative: pq ≠ qp
        - Multiplicative norm: ||pq|| = ||p|| * ||q||

    Args:
        p: [..., 4] quaternion tensor (a, b, c, d)
        q: [..., 4] quaternion tensor (same shape as p)

    Returns:
        out: [..., 4] quaternion product

    Example:
        >>> p = torch.randn(10, 4)
        >>> q = torch.randn(10, 4)
        >>> result = quaternion_multiply(p, q)
        >>> result.shape
        torch.Size([10, 4])
    """
    # Implementation
    pass
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Reproduction**: Minimal code to reproduce
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: Python version, PyTorch version, GPU model, etc.

## 💡 Feature Requests

When requesting features:

1. **Use case**: Describe your use case
2. **Proposal**: Outline your proposed solution
3. **Alternatives**: Mention alternatives you've considered
4. **Willingness**: Indicate if you're willing to implement it

## 📦 Pull Requests

1. **Description**: Clearly describe your changes
2. **Testing**: Include tests for new features
3. **Documentation**: Update docs as needed
4. **Changelog**: Add entry to CHANGELOG.md
5. **Small PRs**: Keep PRs focused and small

### PR Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code formatted (`black`, `isort`)
- [ ] Linting passes (`flake8`)
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Type hints added
- [ ] Docstrings added

## 🔍 Code Review

All PRs require review before merging. Reviews focus on:

- Correctness
- Performance
- Code quality
- Documentation
- Tests

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 💬 Communication

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

## 🙏 Recognition

Contributors will be acknowledged in:
- README.md
- CHANGELOG.md
- Git history

Thank you for contributing to Quaternion Mamba-2! 🎉

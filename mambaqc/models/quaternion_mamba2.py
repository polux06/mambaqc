"""
Full Quaternion Mamba-2 model.

Stacks multiple Quaternion Mamba-2 blocks with residual connections.
"""

import torch
import torch.nn as nn

from .quaternion_mamba2_block import QuaternionMamba2Block


class QuaternionMamba2(nn.Module):
    """
    Complete Quaternion Mamba-2 language model.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_layers: Number of Mamba-2 blocks
        d_state: State dimension per mode (default: 64)
        d_conv: Convolution kernel size (default: 4)
        expand_factor: Expansion factor for inner dimension (default: 2)
        pad_vocab_size_multiple: Pad vocab size to multiple (default: 8 for Tensor Cores)
        bias: Whether to use bias in projections
        dropout: Dropout probability (default: 0.0)

    Shape:
        - Input: [batch, seq_len] (token indices)
        - Output: [batch, seq_len, vocab_size] (logits)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        d_state: int = 64,
        d_conv: int = 4,
        expand_factor: int = 2,
        pad_vocab_size_multiple: int = 8,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # Pad vocab size for efficiency (Tensor Cores prefer multiples of 8)
        if pad_vocab_size_multiple > 1:
            self.vocab_size_padded = (
                (vocab_size + pad_vocab_size_multiple - 1) //
                pad_vocab_size_multiple * pad_vocab_size_multiple
            )
        else:
            self.vocab_size_padded = vocab_size

        # === Embedding layer ===
        self.embedding = nn.Embedding(self.vocab_size_padded, d_model)

        # === Dropout ===
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # === Stack of Quaternion Mamba-2 blocks ===
        self.layers = nn.ModuleList([
            QuaternionMamba2Block(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                bias=bias,
            )
            for _ in range(n_layers)
        ])

        # === Final layer norm ===
        self.norm_f = nn.LayerNorm(d_model)

        # === LM head ===
        self.lm_head = nn.Linear(d_model, self.vocab_size_padded, bias=False)

        # Tie weights (optional but common)
        # self.lm_head.weight = self.embedding.weight

        # === Initialize weights ===
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len] - token indices
            labels: [batch, seq_len] - target token indices (optional, for training)

        Returns:
            dict with keys:
                - logits: [batch, seq_len, vocab_size]
                - loss: scalar (if labels provided)
        """
        batch_size, seq_len = input_ids.shape

        # === Embedding ===
        x = self.embedding(input_ids)  # [B, T, d_model]
        x = self.dropout(x)

        # === Pass through Mamba-2 blocks ===
        for layer in self.layers:
            x = layer(x)  # [B, T, d_model]

        # === Final norm ===
        x = self.norm_f(x)  # [B, T, d_model]

        # === LM head ===
        logits = self.lm_head(x)  # [B, T, vocab_size_padded]

        # Truncate to actual vocab size
        logits = logits[..., :self.vocab_size]

        # === Compute loss if labels provided ===
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross-entropy
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # Standard ignore index
            )

        return {
            "logits": logits,
            "loss": loss,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            input_ids: [batch, seq_len] - prompt tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (default: 1.0)
            top_k: Top-k sampling (optional)
            top_p: Nucleus sampling (optional)

        Returns:
            generated: [batch, seq_len + max_new_tokens] - generated tokens
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(input_ids)
            logits = outputs["logits"]  # [B, T, vocab_size]

            # Get logits for last token
            logits_next = logits[:, -1, :] / temperature  # [B, vocab_size]

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits_next < torch.topk(logits_next, top_k)[0][..., -1, None]
                logits_next[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits_next, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits_next[indices_to_remove] = float('-inf')

            # Sample next token
            probs = torch.softmax(logits_next, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return number of parameters in the model.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            n_params: Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            n_params -= self.embedding.weight.numel()
            # Subtract lm_head if not tied
            if self.lm_head.weight is not self.embedding.weight:
                n_params -= self.lm_head.weight.numel()

        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimate model flops utilization (MFU).

        Args:
            fwdbwd_per_iter: Number of forward-backward passes per iteration
            dt: Time in seconds for iteration

        Returns:
            mfu: Model FLOPs utilization (0-1)
        """
        # Rough estimate of FLOPs per token
        # For Mamba-2 quaternion: ~4x more FLOPs than standard Mamba-2
        N = self.get_num_params()
        L, H, Q, T = self.n_layers, self.d_model, self.d_model, 2048  # Assume seq_len=2048

        # FLOPs per token (very rough)
        flops_per_token = 6 * N  # Forward + backward
        flops_per_token *= 4  # Quaternion overhead

        # Total FLOPs
        flops_per_iter = flops_per_token * T * fwdbwd_per_iter

        # Achieved FLOPs/s
        flops_achieved = flops_per_iter / dt

        # Theoretical peak (A100: 312 TFLOPS, RTX 4090: ~82 TFLOPS)
        # Assume FP16 Tensor Cores on A100
        flops_promised = 312e12

        mfu = flops_achieved / flops_promised

        return mfu


# Preset configurations

def quaternion_mamba2_small(vocab_size: int) -> QuaternionMamba2:
    """Small model: ~50M parameters."""
    return QuaternionMamba2(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=8,
        d_state=32,
    )


def quaternion_mamba2_base(vocab_size: int) -> QuaternionMamba2:
    """Base model: ~150M parameters."""
    return QuaternionMamba2(
        vocab_size=vocab_size,
        d_model=768,
        n_layers=12,
        d_state=64,
    )


def quaternion_mamba2_large(vocab_size: int) -> QuaternionMamba2:
    """Large model: ~350M parameters."""
    return QuaternionMamba2(
        vocab_size=vocab_size,
        d_model=1024,
        n_layers=24,
        d_state=64,
    )

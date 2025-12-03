"""Quaternion Mamba-2 Lite language model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .quaternion_mamba2_lite_block import QuaternionMamba2LiteBlock


class QuaternionMamba2Lite(nn.Module):
    """Stack of Quaternion Mamba-2 Lite blocks for language modeling."""

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
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        if pad_vocab_size_multiple > 1:
            self.vocab_size_padded = (
                (vocab_size + pad_vocab_size_multiple - 1)
                // pad_vocab_size_multiple
                * pad_vocab_size_multiple
            )
        else:
            self.vocab_size_padded = vocab_size

        self.embedding = nn.Embedding(self.vocab_size_padded, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.layers = nn.ModuleList(
            [
                QuaternionMamba2LiteBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand_factor=expand_factor,
                    bias=bias,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size_padded, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
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
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        batch_size, _ = input_ids.shape

        assert input_ids.min() >= 0, f"Negative token index: {input_ids.min().item()}"
        assert input_ids.max() < self.vocab_size, (
            f"Token index {input_ids.max().item()} >= vocab_size {self.vocab_size}"
        )

        x = self.embedding(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)
        logits = logits[..., : self.vocab_size]

        loss = None
        if labels is not None:
            valid_labels = labels[labels != -100]
            if len(valid_labels) > 0:
                assert valid_labels.min() >= 0, f"Negative label: {valid_labels.min().item()}"
                assert valid_labels.max() < self.vocab_size, (
                    f"Label {valid_labels.max().item()} >= vocab_size {self.vocab_size}"
                )

            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100
            )

        return {"logits": logits, "loss": loss}

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids)
            logits = outputs["logits"]

            logits_next = logits[:, -1, :] / temperature

            if top_k is not None:
                indices_to_remove = logits_next < torch.topk(logits_next, top_k)[0][..., -1, None]
                logits_next[indices_to_remove] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits_next, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits_next[indices_to_remove] = float("-inf")

            probs = torch.softmax(logits_next, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            n_params -= self.embedding.weight.numel()
            if self.lm_head.weight is not self.embedding.weight:
                n_params -= self.lm_head.weight.numel()

        return n_params


# Preset configurations


def quaternion_mamba2_lite_small(vocab_size: int) -> QuaternionMamba2Lite:
    return QuaternionMamba2Lite(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=8,
        d_state=32,
    )


def quaternion_mamba2_lite_base(vocab_size: int) -> QuaternionMamba2Lite:
    return QuaternionMamba2Lite(
        vocab_size=vocab_size,
        d_model=768,
        n_layers=12,
        d_state=64,
    )


def quaternion_mamba2_lite_large(vocab_size: int) -> QuaternionMamba2Lite:
    return QuaternionMamba2Lite(
        vocab_size=vocab_size,
        d_model=1024,
        n_layers=24,
        d_state=64,
    )

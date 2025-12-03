"""
Training script for Quaternion Mamba-2.

Features:
- Mixed precision training (FP16/BF16)
- Gradient checkpointing for memory efficiency
- Learning rate scheduling with warmup
- Gradient accumulation
- Logging and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import os
import time
import math
from pathlib import Path
from typing import Optional

from mambaqc.models import QuaternionMamba2


class TinyStoriesDataset(Dataset):
    """
    Simple dataset for TinyStories or similar text data.

    TODO: Replace with actual dataset loading.
    """

    def __init__(self, data_path: str, vocab_size: int, seq_len: int = 2048, n_samples: int = 1000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Placeholder: generate random valid tokens
        # IMPORTANT: tokens must be in range [0, vocab_size-1]
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len + 1))

        # Verify all tokens are valid
        assert self.data.min() >= 0, f"Invalid negative token: {self.data.min()}"
        assert self.data.max() < vocab_size, f"Token {self.data.max()} >= vocab_size {vocab_size}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Trainer:
    """
    Trainer for Quaternion Mamba-2.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 4,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_steps: int = 100000,
        warmup_steps: int = 2000,
        grad_accumulation_steps: int = 8,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        save_dir: str = "./checkpoints",
        device: str = "cuda",
        mixed_precision: str = "fp16",  # "fp16", "bf16", or "no"
        gradient_checkpointing: bool = True,
        max_grad_norm: float = 1.0,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.grad_accumulation_steps = grad_accumulation_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.save_dir = Path(save_dir)
        self.device = device
        self.mixed_precision = mixed_precision
        self.max_grad_norm = max_grad_norm

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

        # Optimizer (AdamW)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler (cosine with warmup)
        self.scheduler = self._get_cosine_schedule_with_warmup()

        # Mixed precision scaler
        if mixed_precision == "fp16":
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None

        # Gradient checkpointing
        if gradient_checkpointing:
            # Enable gradient checkpointing for memory efficiency
            # Note: This would need to be implemented in the model
            pass

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

    def _get_cosine_schedule_with_warmup(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Cosine decay
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step."""
        input_ids = batch[:, :-1].to(self.device)
        labels = batch[:, 1:].to(self.device)

        # Forward pass with mixed precision
        if self.mixed_precision == "fp16":
            with autocast('cuda', dtype=torch.float16):
                outputs = self.model(input_ids, labels=labels)
                loss = outputs["loss"] / self.grad_accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

        elif self.mixed_precision == "bf16":
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(input_ids, labels=labels)
                loss = outputs["loss"] / self.grad_accumulation_steps

            loss.backward()

        else:  # No mixed precision
            outputs = self.model(input_ids, labels=labels)
            loss = outputs["loss"] / self.grad_accumulation_steps
            loss.backward()

        return loss.item() * self.grad_accumulation_steps

    def optimizer_step(self):
        """Optimizer step with gradient clipping."""
        if self.mixed_precision == "fp16":
            # Unscale gradients
            self.scaler.unscale_(self.optimizer)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Optimizer step with gradient scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Optimizer step
            self.optimizer.step()

        # Update learning rate
        self.scheduler.step()

        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        if self.val_dataset is None:
            return float('nan')

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch[:, :-1].to(self.device)
            labels = batch[:, 1:].to(self.device)

            if self.mixed_precision == "fp16":
                with autocast('cuda', dtype=torch.float16):
                    outputs = self.model(input_ids, labels=labels)
            elif self.mixed_precision == "bf16":
                with autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(input_ids, labels=labels)
            else:
                outputs = self.model(input_ids, labels=labels)

            total_loss += outputs["loss"].item()
            num_batches += 1

        self.model.train()
        return total_loss / num_batches

    def save_checkpoint(self, step: int, val_loss: Optional[float] = None):
        """Save model checkpoint."""
        checkpoint = {
            "step": step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save checkpoint
        path = self.save_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        # Save best model
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved with val_loss={val_loss:.4f}")

    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.max_steps} steps")
        print(f"Model parameters: {self.model.get_num_params() / 1e6:.2f}M")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Gradient accumulation steps: {self.grad_accumulation_steps}")

        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()

        step_in_epoch = 0
        train_iter = iter(self.train_loader)

        while self.step < self.max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                self.epoch += 1
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
                step_in_epoch = 0

            # Training step
            loss = self.train_step(batch)
            total_loss += loss
            total_tokens += batch.numel()

            # Optimizer step (after gradient accumulation)
            if (step_in_epoch + 1) % self.grad_accumulation_steps == 0:
                self.optimizer_step()
                self.step += 1

                # Logging
                if self.step % 100 == 0:
                    t1 = time.time()
                    dt = t1 - t0
                    tokens_per_sec = total_tokens / dt
                    avg_loss = total_loss / 100

                    lr = self.scheduler.get_last_lr()[0]

                    print(f"Step {self.step}/{self.max_steps} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Tokens/s: {tokens_per_sec:.0f} | "
                          f"Time: {dt:.2f}s")

                    total_loss = 0.0
                    total_tokens = 0
                    t0 = time.time()

                # Evaluation
                if self.step % self.eval_interval == 0:
                    val_loss = self.evaluate()
                    print(f"Validation loss: {val_loss:.4f}")
                    self.save_checkpoint(self.step, val_loss)

                # Save checkpoint
                elif self.step % self.save_interval == 0:
                    self.save_checkpoint(self.step)

            step_in_epoch += 1

        print("Training complete!")
        self.save_checkpoint(self.step)


def main():
    """Main training function."""
    # Configuration
    config = {
        "vocab_size": 10000,
        "d_model": 256,
        "n_layers": 6,
        "d_state": 64,
        "batch_size": 4,
        "learning_rate": 3e-4,
        "max_steps": 100000,
        "warmup_steps": 2000,
        "grad_accumulation_steps": 8,
    }

    # Create model
    print("Creating Quaternion Mamba-2 model...")
    model = QuaternionMamba2(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        d_state=config["d_state"],
    )

    print(f"Model size: {model.get_num_params() / 1e6:.2f}M parameters")

    # Create datasets (placeholder)
    print("Loading datasets...")
    train_dataset = TinyStoriesDataset(
        "train.txt",
        vocab_size=config["vocab_size"],
        seq_len=2048,
        n_samples=1000
    )
    val_dataset = TinyStoriesDataset(
        "val.txt",
        vocab_size=config["vocab_size"],
        seq_len=2048,
        n_samples=100
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        warmup_steps=config["warmup_steps"],
        grad_accumulation_steps=config["grad_accumulation_steps"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision="fp16",
        gradient_checkpointing=True,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()

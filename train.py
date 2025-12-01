# -*- coding: utf-8 -*-
import os
import time
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from dataclasses import dataclass
from tqdm import tqdm

# Import du modèle
from quaternion_mamba import QuaternionMambaConfig, QuaternionMambaLMHeadModel

# =============================================================================
# 0. OPTIMISATIONS MATÉRIELLES (CRITIQUE POUR RTX 30/40/50)
# =============================================================================
# Active les calculs TF32 sur les Tensor Cores (énorme gain de vitesse)
torch.set_float32_matmul_precision('high')

@dataclass
class TrainingConfig:
    batch_size: int = 16          # Taille physique en mémoire
    grad_accum_steps: int = 4     # Accumulation -> Batch effectif = 16 * 4 = 64
    learning_rate: float = 3e-4
    max_steps: int = 20000
    warmup_steps: int = 500
    eval_interval: int = 500
    save_interval: int = 1000
    grad_clip: float = 1.0
    max_seq_len: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir: str = "data"

# =============================================================================
# ... (Dataset et get_lr restent identiques) ...
# =============================================================================
class MemoryMappedDataset(Dataset):
    def __init__(self, bin_path, block_size, dtype=np.uint16):
        self.data = np.memmap(bin_path, dtype=dtype, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        chunk = torch.from_numpy(self.data[idx : idx + self.block_size + 1].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_lr(it, train_cfg):
    if it < train_cfg.warmup_steps:
        return train_cfg.learning_rate * (it + 1) / train_cfg.warmup_steps
    if it > train_cfg.max_steps:
        return train_cfg.learning_rate * 0.1
    decay_ratio = (it - train_cfg.warmup_steps) / (train_cfg.max_steps - train_cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_cfg.learning_rate * 0.1 + 0.9 * coeff * train_cfg.learning_rate

# =============================================================================
# 3. BOUCLE D'ENTRAÎNEMENT OPTIMISÉE
# =============================================================================

def train():
    train_cfg = TrainingConfig()
    
    model_cfg = QuaternionMambaConfig(
        d_model=384,
        n_layers=24,
        vocab_size=2048,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto"
    )

    print(f"🚀 QUATERNION MAMBA TRAINING START")
    print(f"   Device: {train_cfg.device} (TF32 Enabled)")
    
    # --- Data ---
    train_path = os.path.join(train_cfg.data_dir, "train.bin")
    val_path = os.path.join(train_cfg.data_dir, "val.bin")
    
    # Augmentation des workers pour nourrir le GPU plus vite
    train_dataset = MemoryMappedDataset(train_path, train_cfg.max_seq_len)
    val_dataset = MemoryMappedDataset(val_path, train_cfg.max_seq_len)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_cfg.batch_size, 
        shuffle=True, 
        num_workers=8,      # Augmenté pour éviter l'attente I/O
        pin_memory=True,
        prefetch_factor=2   # Précharge les batchs
    )
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=4)
    
    # --- Modèle ---
    model = QuaternionMambaLMHeadModel(model_cfg).to(train_cfg.device)
    
    # !!! L'ARME SECRÈTE : TORCH.COMPILE !!!
    # Cela va essayer de fusionner votre boucle SSM en un kernel Triton
    print("   Compilation du modèle avec torch.compile... (cela peut prendre 1-2 min au début)")
    model = torch.compile(model, mode="reduce-overhead") 

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = GradScaler() 

    model.train()
    iter_loader = iter(train_loader)
    
    pbar = tqdm(range(train_cfg.max_steps), desc="Training", unit="step")
    
    # Variable pour l'accumulation
    accum_loss = 0.0
    
    for step in pbar:
        t0 = time.time()
        
        # 1. Gestion LR
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # 2. Gradient Accumulation Loop
        # On ne fait optimizer.step() que tous les N mini-batchs
        optimizer.zero_grad(set_to_none=True)
        
        for micro_step in range(train_cfg.grad_accum_steps):
            try:
                X, Y = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                X, Y = next(iter_loader)
            
            X, Y = X.to(train_cfg.device, non_blocking=True), Y.to(train_cfg.device, non_blocking=True)
            
            # Autocast est crucial pour la vitesse
            with autocast(device_type='cuda', dtype=torch.float16):
                logits, loss = model(X, targets=Y)
                # On divise la loss par le nombre d'accumulation pour garder la bonne échelle de gradient
                loss = loss / train_cfg.grad_accum_steps
            
            scaler.scale(loss).backward()
            accum_loss += loss.item()
        
        # 3. Update (Une seule fois après l'accumulation)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        dt = time.time() - t0
        tokens_per_sec = (train_cfg.batch_size * train_cfg.grad_accum_steps * train_cfg.max_seq_len) / dt

        # 4. Logging
        if step % 10 == 0:
            pbar.set_postfix({
                "loss": f"{accum_loss:.4f}", 
                "lr": f"{lr:.2e}", 
                "tok/s": f"{int(tokens_per_sec)}"
            })
        accum_loss = 0.0 # Reset pour le prochain step
            
        # 5. Eval & Save
        if step > 0 and step % train_cfg.eval_interval == 0:
            val_loss = evaluate(model, val_loader, train_cfg)
            model.train()
            
        if step > 0 and step % train_cfg.save_interval == 0:
            ckpt_path = f"checkpoints/qmamba_step_{step}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            # Attention: avec torch.compile, il faut parfois utiliser model._orig_mod.state_dict()
            # ou simplement model.state_dict() gère ça automatiquement dans les versions récentes
            torch.save(model.state_dict(), ckpt_path)

@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    losses = []
    eval_steps = 20 # Réduit pour aller vite
    for i, (X, Y) in enumerate(loader):
        if i >= eval_steps: break 
        X, Y = X.to(cfg.device), Y.to(cfg.device)
        with autocast(device_type='cuda', dtype=torch.float16):
            _, loss = model(X, targets=Y)
        losses.append(loss.item())
    
    mean_loss = sum(losses) / len(losses)
    print(f"\n📉 Validation Loss: {mean_loss:.4f}")
    return mean_loss

if __name__ == "__main__":
    train()
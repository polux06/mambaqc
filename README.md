# Quaternion Mamba-2

**Une extension non-commutative du modèle Mamba-2 reposant sur le formalisme SSD (State Space Dual) avec algèbre quaternionique.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/triton-2.1+-green.svg)](https://triton-lang.org/)

## 🔥 Highlights

- **Kernels GPU optimisés** : Tous les calculs fusionnés avec Triton pour maximiser l'utilisation des caches SM
- **Support Tensor Cores** : Opérations optimisées pour les Tensor Cores NVIDIA
- **Dynamiques quaternioniques** : Modélisation de rotations, oscillations et structures géométriques riches
- **Parallel scan O(log T)** : Grâce à l'associativité du produit quaternionique
- **Stabilité numérique** : Discrétisation de Cayley et normalisation géométrique

## 📐 Architecture

```
Input x_t ∈ ℝ^{d_model}
    ↓
[Projection + Split] → (x'_t, z_t)
    ↓
[Conv1D Causale (kernel=4)] → x''_t
    ↓
[Activation SiLU] → x'''_t
    ↓
[Conversion Quaternionique] → x_quat ∈ ℍ^{d_model}
    ↓
[S6 Gate + SSM Quaternionique] → y_quat ∈ ℍ^{d_model}
    ↓
[Normalisation Géométrique]
    ↓
[Projection vers ℝ] → y'_t ∈ ℝ^{d_model}
    ↓
[Gating : z_t ⊙ y'_t]
    ↓
[Projection de sortie + Residual]
```

## 🚀 Installation

### Prérequis

- Python >= 3.9
- CUDA >= 11.8 (pour Triton et Tensor Cores)
- PyTorch >= 2.1
- Triton >= 2.1

### Installation depuis les sources

```bash
# Cloner le dépôt
git clone https://github.com/polux06/mambaqc.git
cd mambaqc

# Installer les dépendances
pip install -r requirements.txt

# Installer le package
pip install -e .
```

## 📖 Usage rapide

### Créer un modèle

```python
from mambaqc.models import QuaternionMamba2

# Configuration de base (~150M paramètres)
model = QuaternionMamba2(
    vocab_size=10000,
    d_model=768,
    n_layers=12,
    d_state=64,
)

# Ou utiliser une config prédéfinie
from mambaqc.models.quaternion_mamba2 import quaternion_mamba2_base
model = quaternion_mamba2_base(vocab_size=10000)
```

### Entraînement

```python
import torch
from mambaqc.models import QuaternionMamba2

# Créer le modèle
model = QuaternionMamba2(vocab_size=10000).cuda()

# Forward pass
input_ids = torch.randint(0, 10000, (4, 2048)).cuda()
outputs = model(input_ids)
logits = outputs["logits"]  # [4, 2048, 10000]

# Avec labels pour calcul de loss
labels = torch.randint(0, 10000, (4, 2048)).cuda()
outputs = model(input_ids, labels=labels)
loss = outputs["loss"]
loss.backward()
```

### Script d'entraînement complet

```bash
python train.py
```

Configuration par défaut :
- Mixed precision (FP16)
- Gradient checkpointing
- Gradient accumulation (8 steps)
- Cosine LR schedule avec warmup

## 🧪 Tests

```bash
# Lancer tous les tests
pytest mambaqc/tests/ -v

# Tests spécifiques
pytest mambaqc/tests/test_quaternion_ops.py -v

# Avec couverture
pytest mambaqc/tests/ --cov=mambaqc --cov-report=html
```

## 🔬 Composants clés

### 1. Kernels Triton optimisés

#### Multiplication quaternionique
```python
from mambaqc.kernels.quaternion_ops import quaternion_multiply

p = torch.randn(100, 4).cuda()  # Quaternions
q = torch.randn(100, 4).cuda()

result = quaternion_multiply(p, q)  # Hamilton product
```

#### Transformée de Cayley
```python
from mambaqc.kernels.cayley_transform import cayley_discretization_fused

# z = Δ * Λ (dynamics)
z = torch.randn(2, 16, 768, 64, 4).cuda()

# q = (1 - 0.5*z)^{-1} * (1 + 0.5*z)
q = cayley_discretization_fused(z)

# Propriété: Si Re(z) < 0, alors |q| < 1 (stabilité)
```

#### Parallel scan
```python
from mambaqc.kernels.parallel_scan import parallel_scan_quaternion

# Séquence de quaternions
q_seq = torch.randn(2, 256, 768, 64, 4).cuda()

# Produit cumulatif parallèle (O(log T) depth)
cumulative = parallel_scan_quaternion(q_seq)
```

### 2. Normalisation géométrique

```python
from mambaqc.layers import QuaternionLayerNorm

norm = QuaternionLayerNorm(d_model=768).cuda()

# Normalise les NORMES tout en préservant les DIRECTIONS
q = torch.randn(4, 256, 768, 4).cuda()
q_normalized = norm(q)
```

### 3. Bloc Quaternion Mamba-2

```python
from mambaqc.models import QuaternionMamba2Block

block = QuaternionMamba2Block(
    d_model=768,
    d_state=64,
    d_conv=4,
    expand_factor=2,
).cuda()

x = torch.randn(4, 256, 768).cuda()
output = block(x)  # [4, 256, 768]
```

## 📊 Performance

### Complexité

| Opération | Temps | Mémoire |
|-----------|-------|---------|
| Multiplication quaternionique | ~4× réel | 4× réel |
| SSM recurrence (séquentiel) | O(T) | O(d·s) |
| SSM recurrence (parallel scan) | O(log T) depth | O(T·d·s) |
| Bloc complet | ~4× Mamba-2 | ~4× Mamba-2 |

### Benchmarks

Configuration : RTX 4090, batch=4, seq_len=2048, d_model=768, d_state=64

| Métrique | Quaternion Mamba-2 | Mamba-2 (baseline) |
|----------|-------------------|-------------------|
| Throughput (tokens/s) | ~12K | ~48K |
| VRAM (training) | 11.8 GB | 3.2 GB |
| Convergence | Stable | Stable |

## 🧮 Propriétés mathématiques

### Multiplication quaternionique

Pour $p = p_0 + p_1 i + p_2 j + p_3 k$ et $q = q_0 + q_1 i + q_2 j + q_3 k$ :

$$pq = (p_0 q_0 - p_1 q_1 - p_2 q_2 - p_3 q_3) + \ldots$$

**Propriétés** :
- ✅ Associative : $(pq)r = p(qr)$
- ❌ Non-commutative : $pq \neq qp$ en général
- ✅ Norme multiplicative : $|pq| = |p| \cdot |q|$

### Discrétisation de Cayley

$$q = \left(1 - \frac{1}{2}z\right)^{-1} \left(1 + \frac{1}{2}z\right)$$

**Garanties** :
- Si $\text{Re}(z) < 0$ alors $|q| < 1$ (stabilité inconditionnelle)
- Précision d'ordre supérieur à ZOH
- Préserve les propriétés unitaires

## 🎯 Applications

Quaternion Mamba-2 est particulièrement adapté pour :

- 🤖 **Robotique** : Trajectoires 3D, poses, contrôle
- 📡 **Signaux physiques** : Champs électromagnétiques, acoustique spatiale
- 🌍 **Géophysique** : Dynamiques rotationnelles, magnétisme terrestre
- 🎮 **Vision 3D** : Nuages de points, estimation de poses
- 🔬 **Physique quantique** : États de spin, dynamiques de qubits

## 📚 Citation

Si vous utilisez ce code dans votre recherche, veuillez citer :

```bibtex
@article{laurent2024quaternion,
  title={Quaternion Mamba-2: Un Modèle SSD Sélectif Multi-États Quaternionique avec Dynamique de Cayley},
  author={Laurent},
  year={2024}
}
```

## 🤝 Contribution

Les contributions sont bienvenues ! Voir [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 Licence

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE).

## 🙏 Remerciements

- **Mamba-2** : Dao & Gu (2024)
- **Structured State Spaces** : Gu et al. (2022)
- **Quaternion Neural Networks** : Parcollet et al. (2019)
- **Triton** : OpenAI

## 📧 Contact

Pour toute question : [laurent@example.com](mailto:laurent@example.com)

---

**Note** : Ce projet est une implémentation de recherche. Pour un usage en production, des optimisations supplémentaires sont recommandées.

# UIBFuse: Uncertainty-aware Information Bottleneck Fusion for Cross-Modal Visual-Temporal Learning



<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch" alt="PyTorch"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://icme2026.org/"><img src="https://img.shields.io/badge/ICME-2026-blue" alt="ICME 2026"></a>
</p>

<p align="center">
  <b>Anonymous ICME 2026 Submission</b>
</p>

---

## 📋 Abstract

Cross-modal learning for digital asset markets demands effective integration of visual aesthetics and temporal transaction dynamics—two modalities exhibiting fundamentally different statistical properties. Current fusion approaches either disregard modality-specific uncertainties or lack theoretical grounding for architectural decisions. We introduce **UIBFuse**, a principled framework combining Information Bottleneck theory with Bayesian precision-weighted fusion. Our approach derives architectural hyperparameters from information-theoretic bounds rather than empirical tuning: latent dimension from mutual information capacity, multi-scale pyramid from spectral frequency analysis, and attention heads from pattern clustering. A volatility-aware gating mechanism dynamically adjusts modality contributions based on market uncertainty.

**Key Results on CryptoPunks Dataset (167,492 transactions):**

| Metric | UIBFuse | Best Baseline | Improvement |
|--------|---------|---------------|-------------|
| R² | **0.847** | 0.781 | +8.5% |
| MAE (ETH) | **7.56** | 9.45 | −20.0% |
| MES | **0.851** | 0.802 | +6.1% |
| ECE | **0.041** | 0.064 | −35.9% |

---

## 🏗️ Architecture


UIBFuse addresses three fundamental limitations:

1. **Lack of theoretical foundations** → Information-theoretic hyperparameter derivation
2. **Blindness to modality uncertainties** → Bayesian precision-weighted fusion
3. **Static fusion weights** → Volatility-aware adaptive gating

### Core Components

| Component | Description | Key Parameters |
|-----------|-------------|----------------|
| **Multi-Scale Visual Pyramid** | Spectral-derived scales with SE attention | {224, 112, 56, 28}, C={64, 128, 256, 512} |
| **Dilated Temporal Encoder** | Octave-based dilation rates | r={1, 2, 4, 8, 16}, RF=187 steps |
| **Bayesian Fusion** | Precision-weighted combination | τ_v = σ_v⁻², τ_t = σ_t⁻² |
| **Volatility Gating** | Market-adaptive modality weighting | g_v = σ(W_g[μ_v‖μ_t‖v̂]) |

---

## 📁 Repository Structure

```
UIBFuse/
├── Figures/                 # Paper figures and visualizations
│   ├── fig_motivation.png
│   ├── fig_architecture.png
│   └── fig_results.png
├── config.py               # Configuration and hyperparameters
├── models.py               # UIBFuse model architecture
├── train.py                # Training pipeline
├── evaluate.py             # Evaluation and metrics
├── utils.py                # Utility functions
└── README.md               # This file
```

---

## ⚙️ Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1.0
- CUDA ≥ 11.8 (for GPU acceleration)

### Setup

```bash
# Clone repository


# Create conda environment
conda create -n uibfuse python=3.10
conda activate uibfuse

# Install dependencies
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm tensorboard
pip install timm einops transformers
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
```

---

## 📊 Dataset

We evaluate on the **CryptoPunks** dataset comprising 167,492 transactions across 10,000 unique digital assets (2017–2021).

### Dataset Statistics

| Attribute | Value |
|-----------|-------|
| Total transactions | 167,492 |
| Sales transactions | 18,979 |
| Unique assets | 10,000 |
| Time span | 2017–2021 |
| Visual attributes | 87 |
| Temporal features | 64 |
| Price range (ETH) | 0.01–4,200 |

### Asset Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| Male | 6,039 | 60.4% |
| Female | 3,840 | 38.4% |
| Zombie | 88 | 0.88% |
| Ape | 24 | 0.24% |
| Alien | 9 | 0.09% |

### Data Preparation

```bash
# Download and prepare dataset (instructions after acceptance)
python utils.py --prepare_data --data_dir ./data
```

---

## 🚀 Usage

### Training

```bash
# Train UIBFuse with default configuration
python train.py --config config.py

# Train with custom parameters
python train.py \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --beta_max 0.1 \
    --warmup_epochs 10 \
    --latent_dim 256 \
    --num_heads 16
```

### Configuration Options

Key hyperparameters in `config.py`:

```python
# Information-theoretic derived parameters
LATENT_DIM = 256          # From MI capacity bound
PYRAMID_SCALES = [224, 112, 56, 28]  # From spectral analysis
NUM_HEADS = 16            # From pattern clustering

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 100
WARMUP_EPOCHS = 10
BETA_MAX = 0.1            # IB regularization weight

# Architecture
VISUAL_CHANNELS = [64, 128, 256, 512]
DILATION_RATES = [1, 2, 4, 8, 16]
TEMPORAL_DIM = 64
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --checkpoint ./checkpoints/best_model.pth

# Evaluate with specific metrics
python evaluate.py \
    --checkpoint ./checkpoints/best_model.pth \
    --metrics r2 mae rmse mes da ece \
    --save_predictions
```

### Inference

```python
from models import UIBFuse
import torch

# Load model
model = UIBFuse(
    latent_dim=256,
    visual_channels=[64, 128, 256, 512],
    num_heads=16,
    dilation_rates=[1, 2, 4, 8, 16]
)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# Inference
with torch.no_grad():
    visual_input = torch.randn(1, 3, 224, 224)    # RGB image
    temporal_input = torch.randn(1, 187, 64)       # Transaction sequence
    volatility = torch.tensor([[0.5]])             # Market volatility
    
    mu_pred, sigma2_pred = model(visual_input, temporal_input, volatility)
    print(f"Predicted price: {mu_pred.item():.2f} ETH")
    print(f"Uncertainty: {sigma2_pred.sqrt().item():.2f} ETH")
```

---

## 📈 Results

### Main Results

| Method | R²↑ | MAE↓ | RMSE↓ | MES↑ | DA↑ | ECE↓ |
|--------|-----|------|-------|------|-----|------|
| Visual (ViT-B) | 0.683 | 12.41 | 18.73 | 0.712 | 0.634 | 0.089 |
| Temporal (Trans.) | 0.658 | 13.42 | 20.12 | 0.691 | 0.612 | 0.098 |
| GMU | 0.756 | 10.12 | 15.67 | 0.778 | 0.689 | 0.071 |
| CLIP-Adapted | 0.768 | 9.87 | 15.21 | 0.789 | 0.701 | 0.068 |
| Static Attention | 0.781 | 9.45 | 14.56 | 0.802 | 0.712 | 0.064 |
| **UIBFuse (Ours)** | **0.847** | **7.56** | **11.89** | **0.851** | **0.762** | **0.041** |

*All improvements statistically significant at p < 0.001 (paired t-test)*

### Ablation Study

| Configuration | R² | ΔR² |
|--------------|-----|-----|
| UIBFuse (full) | 0.847 | — |
| w/o Uncertainty | 0.798 | −0.049 |
| w/o Volatility Gating | 0.812 | −0.035 |
| w/o IB Regularization | 0.821 | −0.026 |

### Hyperparameter Validation

| Parameter | Theoretical | Optimal | Match |
|-----------|-------------|---------|-------|
| Latent dim (d_z) | 256 | 256 | ✓ |
| Pyramid levels | 4 | 4 | ✓ |
| Attention heads (H) | 16 | 16 | ✓ |

---

## 🔬 Reproducing Results

### Full Reproduction Pipeline

```bash
# 1. Prepare environment
conda activate uibfuse

# 2. Prepare data
python utils.py --prepare_data

# 3. Train model
python train.py --config config.py --seed 42

# 4. Evaluate
python evaluate.py --checkpoint ./checkpoints/best_model.pth

# 5. Generate figures
python utils.py --generate_figures --output_dir ./Figures
```

### Expected Training Output

```
Epoch [1/100] - Loss: 2.341 - R²: 0.412 - MAE: 15.67
Epoch [10/100] - Loss: 1.234 - R²: 0.623 - MAE: 11.23
Epoch [50/100] - Loss: 0.567 - R²: 0.789 - MAE: 8.45
Epoch [100/100] - Loss: 0.423 - R²: 0.847 - MAE: 7.56
Training complete. Best R²: 0.847
```

### Hardware Requirements

| Configuration | Training Time | GPU Memory |
|--------------|---------------|------------|
| 4× RTX 3090 (24GB) | ~8 hours | ~18 GB/GPU |
| 1× RTX 3090 (24GB) | ~28 hours | ~22 GB |
| 1× RTX 4090 (24GB) | ~20 hours | ~20 GB |



---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

<p align="center">
  <i>Anonymous ICME 2026 Submission — Code and samples for review purposes only</i>
</p>

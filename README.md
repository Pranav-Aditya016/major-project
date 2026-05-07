# Mask-Guided Medical Image Generation with Diffusion Models

**Research-oriented pipeline for synthetic medical image generation using conditional diffusion models (MONAI DDPM).**

## Overview

This project implements mask-guided image generation using NVIDIA MONAI's diffusion framework. The key innovation is the **Block Consistency Loss** which ensures generated lesions appear only within specified binary mask regions.

### Architecture
```
Input: Binary Lesion Mask (256×256)
  ↓
Diffusion Model (UNet-based DDPM)
  - Concatenate mask with noisy image
  - Iterative denoising (1000 → 0 timesteps)
  ↓
Output: Synthetic Medical Image with Lesion in Mask Region
```

### Research Contribution: Block Consistency Loss

**Standard Diffusion Loss:**
```
L_diffusion = MSE(predicted_noise, ground_truth_noise)
```

**Novel Block Consistency Loss:**
```
L_consistency = w_in × MSE(lesion_detected, mask) + w_out × FalsePositives
```

This enforces:
1. **Inside Mask**: Lesion must appear where mask indicates (minimize mismatch)
2. **Outside Mask**: No false lesions outside masked region (suppress artifacts)

## Project Structure

```
├── data/
│   ├── raw/                    # Original tar files (Heart, Prostate)
│   └── processed/              # Extracted and organized data
│       ├── lung/
│       │   ├── images/
│       │   └── masks/
│       ├── heart/
│       └── prostate/
├── models/
│   └── checkpoints/            # Saved model weights
├── src/
│   ├── __init__.py
│   ├── dataset.py              # Dataset loaders
│   ├── diffusion_model.py      # DDPM + Block Consistency Loss
│   ├── config.py               # Training configuration
│   ├── trainer.py              # Training loop
│   └── data_prep.py            # Dataset preprocessing utilities
├── notebooks/
│   └── 01_train_baseline.ipynb # Training script (START HERE)
├── configs/
│   └── default_config.json     # Hyperparameter config
└── setup.ipynb                 # Environment setup
```

## Installation

### 1. Environment Setup
All dependencies are in `setup.ipynb`. Run:
```python
!pip install torch torchvision
!pip install monai monai-generative
!pip install nibabel pillow tqdm numpy scikit-image
```

### 2. Verify Installation
```python
import torch
import monai
print(f"PyTorch: {torch.__version__}")
print(f"MONAI: {monai.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## Quick Start

### Option 1: Synthetic Data (Testing)
Perfect for checking pipeline correctness:

```python
from src.dataset import SimpleChestXrayDataset
from src.diffusion_model import MaskConditionedDDPM
from src.trainer import DiffusionTrainer
from src.config import get_default_config

# Create synthetic dataset
dataset = SimpleChestXrayDataset(num_samples=100, image_size=256)

# Initialize model
model = MaskConditionedDDPM(image_size=256, in_channels=2)

# Train
trainer = DiffusionTrainer(model, config=get_default_config(), device='cuda')
trainer.fit(train_loader, val_loader)
```

See **notebooks/01_train_baseline.ipynb** for full working example.

### Option 2: Real Medical Data (Heart/Prostate)

```python
from src.data_prep import DatasetManager
from src.dataset import MedicalImageMaskDataset

# Extract and process dataset
manager = DatasetManager(data_root=Path('data'))
manager.extract_tar_dataset(Path('data/raw/Task02_Heart.tar'), organ='heart')
manager.convert_nifti_to_png(organ='heart', slice_selection='middle')

# Load for training
dataset = MedicalImageMaskDataset(
    image_dir=Path('data/processed/heart/images'),
    mask_dir=Path('data/processed/heart/masks'),
    image_size=256
)
```

## Key Components Explained

### 1. Dataset (`src/dataset.py`)

**MedicalImageMaskDataset**: Loads paired images and binary masks
- Supports NIfTI (medical) and PNG formats
- Handles 3D → 2D slice extraction
- Automatic normalization ([0,1] or [-1,1])
- Returns: `{'image': (1,H,W), 'mask': (1,H,W), 'filename': str}`

**SimpleChestXrayDataset**: Synthetic data for testing
- Generates realistic chest X-ray patterns
- Random lesion placement within lung regions
- No file I/O needed

### 2. Diffusion Model (`src/diffusion_model.py`)

**MaskConditionedDDPM**:
- MONAI DiffusionModelUNet with 2D images
- Mask concatenation for conditioning (2 input channels)
- DDPM scheduler (1000 timesteps)
- Methods:
  - `forward()`: Single denoising step
  - `add_noise()`: Forward diffusion (x_0 → x_t)
  - `sample()`: Reverse diffusion (x_T → x_0)

**BlockConsistencyLoss**:
- Detects bright regions in generated image
- Compares with input mask region
- Penalizes misalignment
- Configurable weights (inside/outside)

### 3. Training (`src/trainer.py`)

**DiffusionTrainer**:
- Handles training loop with mixed losses
- Learning rate scheduling with warm-up
- Checkpoint saving/loading
- Validation support
- Gradient clipping for stability

Key hyperparameters:
```python
learning_rate: 1e-4          # Adam optimizer
warmup_steps: 500            # Linear warm-up
diffusion_loss_weight: 1.0   # Standard DDPM loss
block_consistency_weight: 0.1 # Novel loss (increase for stricter mask fidelity)
```

## Training Workflow

### Step 1: Configure
```python
from src.config import get_default_config

config = get_default_config()
config.num_epochs = 100
config.batch_size = 8
config.block_consistency_weight = 0.1  # Adjust this for mask enforcement
```

### Step 2: Load Data
```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
```

### Step 3: Initialize Model & Train
```python
model = MaskConditionedDDPM(image_size=256, in_channels=2, device='cuda')
trainer = DiffusionTrainer(model, config, device='cuda')
trainer.fit(train_loader, val_loader)
```

### Step 4: Generate Samples
```python
with torch.no_grad():
    generated = model.sample(
        mask=test_mask,
        batch_size=5,
        num_steps=50  # Fewer steps = faster sampling
    )
```

## Debugging Common Issues

### Issue: CUDA out of memory
**Solution**: Reduce batch size or model_channels
```python
config.batch_size = 4
config.model_channels = 64  # Default 128
```

### Issue: Lesions appear outside mask
**Solution**: Increase block_consistency_weight
```python
config.block_consistency_weight = 0.5  # Increase from 0.1
```

### Issue: Training loss plateaus
**Solution**: Adjust learning rate or check data normalization
```python
config.learning_rate = 5e-5  # Lower it
# Ensure data is normalized to [0,1] or [-1,1], not [0,255]
```

### Issue: Generated images are blurry
**Solution**: Increase num_sampling_steps
```python
generated = model.sample(mask=test_mask, num_steps=100)  # Was 50
```

## Evaluation Metrics

For research paper evaluation:

```python
# 1. Lesion Localization Accuracy
overlap = (generated_mask & true_mask).sum() / (generated_mask | true_mask).sum()

# 2. Lesion Boundary Precision
boundary_dist = hausdorff_distance(generated_mask, true_mask)

# 3. Image Quality (FID, LPIPS)
# [Implement with torchmetrics]

# 4. Block Consistency Score
consistency_loss = model.block_consistency_loss(generated, mask)
```

## Configuration Reference

Full config options in `src/config.py`:

```python
# Model Architecture
image_size: 256                           # Input resolution
model_channels: 128                       # Base channel count
num_res_blocks: 2                         # Residual blocks per level
num_timesteps: 1000                       # Diffusion steps

# Training
num_epochs: 100
batch_size: 8
learning_rate: 1e-4
warmup_steps: 500

# Loss Weights
diffusion_loss_weight: 1.0
block_consistency_weight: 0.1             # KEY PARAMETER

# Sampling
num_sampling_steps: 50                    # Steps for inference
guidance_scale: 1.0                       # For future classifier-free guidance
```

## Next Steps (Research Pipeline)

- [ ] **Real Data**: Extract Heart/Prostate, train on actual lesion masks
- [ ] **Hyperparameter Sweep**: Optimize block_consistency_weight
- [ ] **Ablation Study**: Compare with/without novel loss
- [ ] **DDIM Sampling**: Faster inference (1000 → 50 steps)
- [ ] **Visualization**: Generate comparison plots
- [ ] **Metrics**: FID, LPIPS, clinical evaluation
- [ ] **Advanced Conditioning**: Classifier-free guidance
- [ ] **Multi-organ**: Scale to Brain (MRI), Breast (Mammogram)

## References

- MONAI Diffusion: https://github.com/Project-MONAI/generative
- DDPM Paper: Ho et al., 2020 (Denoising Diffusion Probabilistic Models)
- Block-wise Loss Inspiration: Mask-guided GANs literature

## Notes for Reproducibility

- All code is deterministic (seed-controlled)
- Checkpoints include full config for reproducibility
- Synthetic data generation uses fixed seeds
- Training metrics logged to `logs/` directory
- Paper-friendly hyperparameter configs in `configs/`

---

**Status**: Research Prototype v0.1  
**License**: Research Use Only  
**Contact**: pranav.aditya@...

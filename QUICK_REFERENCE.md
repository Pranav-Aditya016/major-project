# Quick Reference Guide

## File Locations

```
c:\Pranav Aditya\Major Project\
├── src/                              # Core Python modules
│   ├── dataset.py                   # MedicalImageMaskDataset, SimpleChestXrayDataset
│   ├── diffusion_model.py           # MaskConditionedDDPM, BlockConsistencyLoss
│   ├── config.py                    # TrainingConfig
│   ├── trainer.py                   # DiffusionTrainer
│   ├── data_prep.py                 # DatasetManager (extract tar files)
│   └── __init__.py                  # Package init
│
├── notebooks/                        # Jupyter notebooks (START HERE)
│   ├── 00_data_prep.ipynb           # Extract Task02_Heart.tar, Task05_Prostate.tar
│   └── 01_train_baseline.ipynb      # Training script (synthetic + real data)
│
├── data/
│   ├── raw/                         # Original tar files
│   │   ├── Task02_Heart.tar
│   │   └── Task05_Prostate.tar
│   └── processed/                   # Extracted data (created by 00_data_prep.ipynb)
│       ├── heart/
│       │   ├── images/              # NIfTI or PNG images
│       │   └── masks/               # Binary masks
│       └── prostate/
│
├── models/
│   └── checkpoints/                 # Saved model weights
│       ├── ckpt_epoch_0010.pt
│       ├── best_model.pt
│       └── final_model.pt
│
├── configs/
│   └── default_config.json          # Saved training configuration
│
├── setup.ipynb                      # Environment setup (already ran)
├── README.md                        # Full documentation
└── IMPLEMENTATION_SUMMARY.md        # This guide
```

---

## Quick Commands

### Python Module Usage
```python
# Import all components
from src import (
    MedicalImageMaskDataset,
    SimpleChestXrayDataset,
    MaskConditionedDDPM,
    BlockConsistencyLoss,
    TrainingConfig,
    get_default_config,
    DiffusionTrainer,
    DatasetManager
)

# Create synthetic dataset for testing
dataset = SimpleChestXrayDataset(num_samples=100, image_size=256)

# Create real dataset (after running 00_data_prep.ipynb)
dataset = MedicalImageMaskDataset(
    image_dir=Path('data/processed/heart/images'),
    mask_dir=Path('data/processed/heart/masks'),
    image_size=256
)

# Initialize model
model = MaskConditionedDDPM(image_size=256, device='cuda')

# Train
trainer = DiffusionTrainer(model, config=get_default_config(), device='cuda')
trainer.fit(train_loader, val_loader)

# Sample
with torch.no_grad():
    generated = model.sample(mask=test_mask, batch_size=5, num_steps=50)
```

---

## Configuration Customization

### Edit Hyperparameters
```python
from src.config import get_default_config

config = get_default_config()
config.num_epochs = 100
config.batch_size = 8
config.learning_rate = 1e-4
config.block_consistency_weight = 0.1  # Adjust for mask enforcement
config.save()  # Saves to configs/default_config.json
```

### Common Adjustments

| Parameter | Small GPU (4GB) | Medium GPU (8GB) | Large GPU (16GB) |
|-----------|-----------------|------------------|------------------|
| `batch_size` | 2 | 4 | 8 |
| `model_channels` | 64 | 128 | 256 |
| `image_size` | 128 | 256 | 512 |

---

## Dataset Preparation Workflow

### Step 1: Extract Tar Files
```python
# In 00_data_prep.ipynb
from src.data_prep import DatasetManager
from pathlib import Path

manager = DatasetManager(data_root=Path('data'))
manager.extract_tar_dataset(Path('data/raw/Task02_Heart.tar'), organ='heart')
manager.extract_tar_dataset(Path('data/raw/Task05_Prostate.tar'), organ='prostate')
```

### Step 2: Convert to PNG (Optional)
```python
manager.convert_nifti_to_png(organ='heart', slice_selection='middle')
manager.convert_nifti_to_png(organ='prostate', slice_selection='middle')
```

### Step 3: Verify & Load
```python
from src.dataset import MedicalImageMaskDataset
dataset = MedicalImageMaskDataset(
    image_dir=Path('data/processed/heart/images'),
    mask_dir=Path('data/processed/heart/masks'),
    image_size=256
)
sample = dataset[0]  # Test loading
```

---

## Training Workflow

### Step 1: Create Config
```python
from src.config import get_default_config
config = get_default_config()
# Adjust hyperparameters as needed
```

### Step 2: Create DataLoaders
```python
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
```

### Step 3: Initialize Model
```python
from src.diffusion_model import MaskConditionedDDPM
model = MaskConditionedDDPM(
    image_size=256,
    in_channels=2,
    model_channels=config.model_channels,
    device='cuda'
)
```

### Step 4: Train
```python
from src.trainer import DiffusionTrainer
trainer = DiffusionTrainer(model, config, device='cuda')
trainer.fit(train_loader, val_loader)
```

### Step 5: Evaluate
```python
# Generate samples
test_mask = torch.zeros(1, 1, 256, 256, device='cuda')
test_mask[0, 0, 100:150, 100:150] = 1.0

with torch.no_grad():
    generated = model.sample(mask=test_mask, batch_size=1, num_steps=50)

# Visualize or save
import matplotlib.pyplot as plt
plt.imshow(generated[0, 0].cpu(), cmap='gray')
plt.savefig('generated_sample.png')
```

---

## Loss Components Explained

### Diffusion Loss (Standard DDPM)
```
L_diffusion = MSE(predicted_noise, ground_truth_noise)
```
- Penalizes incorrect noise prediction
- Standard for all diffusion models

### Block Consistency Loss (Novel)
```
L_consistency = w_in × MSE(detected_lesion, mask)
              + w_out × sum(lesion_outside_mask)
```
- `w_in`: Weight for enforcing lesion appearance in mask region
- `w_out`: Weight for suppressing false positives outside mask
- Default: w_in=1.0, w_out=0.5

### Total Loss
```
L_total = w_d × L_diffusion + w_c × L_consistency

where:
  w_d = diffusion_loss_weight (default: 1.0)
  w_c = block_consistency_weight (default: 0.1)
```

**Tuning Guidance**:
- If lesions appear outside mask → increase `block_consistency_weight` to 0.3-0.5
- If lesions not appearing in mask → increase `diffusion_loss_weight`
- For strong enforcement: set `block_consistency_weight = 0.5+`

---

## Checkpoint Management

### Save Checkpoint
```python
trainer.save_checkpoint(name="my_checkpoint.pt", is_best=True)
```

### Load Checkpoint
```python
trainer.load_checkpoint("models/checkpoints/best_model.pt")
```

### Checkpoint Contents
```python
{
    'epoch': int,
    'global_step': int,
    'model_state': dict,
    'optimizer_state': dict,
    'scheduler_state': dict,
    'train_history': {'loss': [...], 'diffusion_loss': [...], ...},
    'val_history': {'loss': [...]},
    'config': {all hyperparameters}
}
```

---

## Debugging Tips

### Issue: CUDA Out of Memory
```python
config.batch_size = 2          # Reduce from 4
config.model_channels = 64     # Reduce from 128
# Or use CPU: device='cpu'
```

### Issue: Training Loss Increasing
```python
config.learning_rate = 5e-5    # Reduce from 1e-4
config.warmup_steps = 1000     # Increase warm-up
```

### Issue: Generated Lesions Outside Mask
```python
config.block_consistency_weight = 0.5  # Increase from 0.1
```

### Issue: Training Too Slow
```python
config.log_every_n_steps = 200     # Log less frequently
config.val_every_n_epochs = 10     # Validate less often
# For sampling: num_steps=25 instead of 50
```

### Issue: Blurry Generated Images
```python
# Use more sampling steps
model.sample(mask=test_mask, num_steps=100)  # Was 50

# Or use DDIM (if implemented)
# Or reduce diffusion noise scale
```

---

## Performance Metrics to Track

During training, monitor:
1. **Total Loss**: Should decrease over time
2. **Diffusion Loss**: Specific component
3. **Block Consistency Loss**: Should decrease if weight > 0
4. **Learning Rate**: Check scheduler is working
5. **Validation Loss**: Overfitting indicator

After training, evaluate:
1. **Lesion Localization Accuracy**: How well mask is followed
2. **Image Quality**: FID, LPIPS (implement if needed)
3. **Boundary Precision**: Edge accuracy of generated lesion
4. **Diversity**: Generate multiple samples from same mask

---

## File Naming Conventions

### Checkpoints
- `ckpt_epoch_0010.pt`: Regular checkpoint at epoch 10
- `best_model.pt`: Best validation loss model
- `final_model.pt`: Final model after training

### Datasets (Organized by 00_data_prep.ipynb)
- `heart_img_0001.nii.gz`: Heart image
- `heart_mask_0001.nii.gz`: Corresponding heart mask
- `heart_img_0001.png`: PNG version (if converted)

### Logs
- `logs/training_log.txt`: Text logs
- `configs/default_config.json`: Saved hyperparameters

---

## Key Classes & Methods

### MaskConditionedDDPM
- `forward(x, mask, timesteps)`: Single denoising step
- `add_noise(x0, timesteps)`: Forward diffusion (training)
- `sample(mask, batch_size, num_steps)`: Generate images (inference)

### DiffusionTrainer
- `fit(train_loader, val_loader)`: Full training loop
- `train_epoch(loader)`: Single epoch
- `validate(loader)`: Validation
- `save_checkpoint(name, is_best)`: Save weights
- `load_checkpoint(path)`: Load weights

### MedicalImageMaskDataset
- `__getitem__(idx)`: Returns {'image', 'mask', 'filename'}
- Supports NIfTI and PNG formats
- Automatic normalization and resizing

---

## For Research Paper

### Key Hyperparameters to Report
- Learning rate, batch size, number of epochs
- Model architecture (channels, residual blocks)
- Loss weights (diffusion vs block consistency)
- Warmup steps and scheduler type
- Number of diffusion timesteps

### Reproducibility
All saved in checkpoint:
```python
checkpoint['config']  # Contains all hyperparameters
```

### Ablation Study Template
```
Experiment 1: block_consistency_weight = 0.0 (baseline)
Experiment 2: block_consistency_weight = 0.1
Experiment 3: block_consistency_weight = 0.5
```

Compare lesion localization accuracy across experiments.

---

## Next Steps

1. ✅ **Setup**: Run setup.ipynb (done)
2. ✅ **Structure**: Project scaffolded (done)
3. 📌 **Test**: Run 01_train_baseline.ipynb with synthetic data
4. 📌 **Extract**: Run 00_data_prep.ipynb for real datasets
5. 📌 **Train**: Fine-tune on Heart/Prostate data
6. 📌 **Evaluate**: Measure metrics and visualize results
7. 📌 **Ablate**: Compare with/without Block Consistency Loss
8. 📌 **Paper**: Write results and methodology

---

Good luck with your research! 🚀

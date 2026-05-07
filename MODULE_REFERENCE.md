# Module Documentation & API Reference

## src/dataset.py

### MedicalImageMaskDataset
Loads paired medical images and binary lesion masks.

**Constructor:**
```python
dataset = MedicalImageMaskDataset(
    image_dir: Path,           # Directory with medical images
    mask_dir: Path,            # Directory with binary masks
    image_size: int = 256,     # Resize to (H, W)
    normalize: bool = True,    # Normalize intensities
    normalization_type: str = 'zero_one'  # 'zero_one' or 'neg_one_one'
)
```

**Supported Formats:**
- NIfTI: `.nii.gz`, `.nii` (3D → middle slice)
- Images: `.png`, `.jpg` (2D)

**Returns:**
```python
{
    'image': torch.Tensor,      # Shape (1, 256, 256), normalized
    'mask': torch.Tensor,       # Shape (1, 256, 256), binary {0, 1}
    'filename': str             # Original filename
}
```

**Example:**
```python
from pathlib import Path
from src.dataset import MedicalImageMaskDataset
from torch.utils.data import DataLoader

dataset = MedicalImageMaskDataset(
    image_dir=Path('data/processed/heart/images'),
    mask_dir=Path('data/processed/heart/masks'),
    image_size=256
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))
print(batch['image'].shape)  # (4, 1, 256, 256)
```

---

### SimpleChestXrayDataset
Generates synthetic chest X-rays with lesions for testing.

**Constructor:**
```python
dataset = SimpleChestXrayDataset(
    num_samples: int = 100,
    image_size: int = 256,
    normalization_type: str = 'zero_one'
)
```

**Features:**
- Realistic chest X-ray patterns
- Random lesion placement in lung regions
- Reproducible with seed control

**Example:**
```python
from src.dataset import SimpleChestXrayDataset

dataset = SimpleChestXrayDataset(num_samples=1000)
sample = dataset[0]

print(sample['image'].shape)   # (1, 256, 256)
print(sample['mask'].shape)    # (1, 256, 256)
print(sample['filename'])      # 'synthetic_000000.png'
```

---

## src/diffusion_model.py

### MaskConditionedDDPM
Core diffusion model with mask conditioning.

**Constructor:**
```python
model = MaskConditionedDDPM(
    image_size: int = 256,
    in_channels: int = 2,              # Image + Mask
    model_channels: int = 128,
    num_res_blocks: int = 2,
    attention_resolutions: Tuple = (16, 8),
    num_timesteps: int = 1000,
    device: str = 'cuda'
)
```

**Methods:**

#### forward(x, mask, timesteps) → noise_pred
Single denoising step (used during training).
```python
x = torch.randn(4, 1, 256, 256)
mask = torch.randint(0, 2, (4, 1, 256, 256)).float()
timesteps = torch.randint(0, 1000, (4,))

noise_pred = model(x, mask, timesteps)  # Shape: (4, 1, 256, 256)
```

#### add_noise(x0, timesteps) → (x_t, noise)
Forward diffusion: Add noise to clean image.
```python
x0 = torch.randn(4, 1, 256, 256)
timesteps = torch.randint(0, 1000, (4,))

x_t, noise = model.add_noise(x0, timesteps)
# x_t: noisy image at timestep t
# noise: ground-truth noise (for loss)
```

#### sample(mask, batch_size, num_steps, guidance_scale) → generated_image
Reverse diffusion: Generate image from mask.
```python
test_mask = torch.zeros(1, 1, 256, 256, device='cuda')
test_mask[0, 0, 100:150, 100:150] = 1.0

with torch.no_grad():
    generated = model.sample(
        mask=test_mask,
        batch_size=5,
        num_steps=50,  # Fewer steps = faster
        guidance_scale=1.0
    )
# generated shape: (5, 1, 256, 256)
```

**Architecture:**
- UNet with 3 resolution levels
- Attention at resolutions 16 and 8
- Concatenated conditioning (image + mask)

---

### BlockConsistencyLoss
Novel loss enforcing lesion consistency with mask.

**Constructor:**
```python
loss_fn = BlockConsistencyLoss(
    threshold: float = 0.5,         # Brightness threshold for lesion detection
    weight_inside: float = 1.0,     # Penalty for lesions not in mask
    weight_outside: float = 0.5     # Penalty for false positives
)
```

**Forward:**
```python
generated = torch.randn(4, 1, 256, 256)  # Generated images
mask = torch.randint(0, 2, (4, 1, 256, 256)).float()  # Input masks

loss = loss_fn(generated, mask)  # Scalar loss value
```

**Loss Breakdown:**
```
L_inside = MSE(detected_lesion, mask)
L_outside = sum(detected_lesion * (1 - mask))
L_total = w_in * L_inside + w_out * L_outside
```

**Tuning:**
- Increase w_in (1.0 → 2.0) for stricter mask adherence
- Increase w_out (0.5 → 1.0) to suppress artifacts outside mask

---

## src/config.py

### TrainingConfig
Dataclass containing all hyperparameters.

**Key Attributes:**
```python
# Model
image_size: int = 256
model_channels: int = 128
num_res_blocks: int = 2
num_timesteps: int = 1000

# Training
num_epochs: int = 100
batch_size: int = 8
learning_rate: float = 1e-4
warmup_steps: int = 500

# Losses
diffusion_loss_weight: float = 1.0
block_consistency_weight: float = 0.1  # KEY FOR RESEARCH

# Data
data_root: Path = Path("data/processed")
image_dir: str = "images"
mask_dir: str = "masks"

# Sampling
num_sampling_steps: int = 50
guidance_scale: float = 1.0
```

**Methods:**

#### to_dict() → dict
```python
config = get_default_config()
d = config.to_dict()
print(d['learning_rate'])  # 1e-4
```

#### save(path)
```python
config.save(Path("configs/my_config.json"))
```

#### from_json(path) → TrainingConfig
```python
config = TrainingConfig.from_json(Path("configs/my_config.json"))
```

**Example:**
```python
from src.config import get_default_config

config = get_default_config()
config.num_epochs = 50
config.batch_size = 16
config.block_consistency_weight = 0.2
config.save(Path("configs/experiment_1.json"))
```

---

## src/trainer.py

### DiffusionTrainer
Full training pipeline with validation and checkpointing.

**Constructor:**
```python
trainer = DiffusionTrainer(
    model: MaskConditionedDDPM,
    config: TrainingConfig,
    device: str = 'cuda'
)
```

**Methods:**

#### fit(train_loader, val_loader)
Full training loop.
```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

trainer.fit(train_loader, val_loader)
# Outputs: Checkpoints saved to config.checkpoint_dir
```

#### train_epoch(loader) → metrics_dict
Single epoch training.
```python
metrics = trainer.train_epoch(train_loader)
# Returns: {'total': float, 'diffusion': float, 'consistency': float}
```

#### validate(loader) → metrics_dict
Validation pass.
```python
val_metrics = trainer.validate(val_loader)
# Returns: {'loss': float}
```

#### save_checkpoint(name, is_best)
```python
trainer.save_checkpoint(name="checkpoint_epoch_50.pt", is_best=True)
```

#### load_checkpoint(path)
```python
trainer.load_checkpoint("models/checkpoints/best_model.pt")
# Restores: model state, optimizer state, epoch counter
```

**Training History:**
```python
print(trainer.train_history['loss'])          # List of epoch losses
print(trainer.train_history['diffusion_loss']) # Diffusion loss per epoch
print(trainer.train_history['consistency_loss']) # Block consistency per epoch
print(trainer.val_history['loss'])            # Validation loss per epoch
```

---

## src/data_prep.py

### DatasetManager
Extract and organize medical image datasets.

**Constructor:**
```python
from pathlib import Path
from src.data_prep import DatasetManager

manager = DatasetManager(data_root=Path('data'))
```

**Methods:**

#### extract_tar_dataset(tar_path, organ)
Extract and organize tar file.
```python
manager.extract_tar_dataset(
    tar_path=Path('data/raw/Task02_Heart.tar'),
    organ='heart'
)
# Creates: data/processed/heart/images/, data/processed/heart/masks/
```

#### convert_nifti_to_png(organ, slice_selection)
Convert 3D NIfTI to 2D PNG slices.
```python
manager.convert_nifti_to_png(organ='heart', slice_selection='middle')
# Creates: data/processed/heart/images_png/, data/processed/heart/masks_png/
```

Options for slice_selection:
- `'middle'`: Take middle slice of each 3D volume (recommended)
- `'all'`: Create dataset from all slices (generates more data)

#### get_dataset_info(organ) → dict
```python
info = manager.get_dataset_info('heart')
print(info)
# {'organ': 'heart', 'num_images': 30, 'num_masks': 30, ...}
```

**Usage Example:**
```python
manager = DatasetManager(Path('data'))
manager.extract_tar_dataset(Path('data/raw/Task02_Heart.tar'), 'heart')
manager.convert_nifti_to_png('heart', 'middle')
info = manager.get_dataset_info('heart')
print(f"Heart: {info['num_images']} images extracted")
```

---

## Summary Table

| Class | Module | Purpose |
|-------|--------|---------|
| MedicalImageMaskDataset | dataset.py | Load real medical images + masks |
| SimpleChestXrayDataset | dataset.py | Generate synthetic data for testing |
| MaskConditionedDDPM | diffusion_model.py | Core diffusion model |
| BlockConsistencyLoss | diffusion_model.py | Novel loss for mask fidelity |
| TrainingConfig | config.py | Hyperparameter management |
| DiffusionTrainer | trainer.py | Training loop & checkpointing |
| DatasetManager | data_prep.py | Dataset extraction & preprocessing |

---

## Common Patterns

### Pattern 1: Quick Test
```python
from src.dataset import SimpleChestXrayDataset
from src.diffusion_model import MaskConditionedDDPM
from src.config import get_default_config
from src.trainer import DiffusionTrainer
from torch.utils.data import DataLoader

# Create dataset
dataset = SimpleChestXrayDataset(100)
loader = DataLoader(dataset, batch_size=4)

# Initialize
model = MaskConditionedDDPM(device='cuda')
config = get_default_config()
config.num_epochs = 5

# Train
trainer = DiffusionTrainer(model, config)
trainer.fit(loader)
```

### Pattern 2: Load Real Data
```python
from src.data_prep import DatasetManager
from src.dataset import MedicalImageMaskDataset
from pathlib import Path

# Extract
manager = DatasetManager(Path('data'))
manager.extract_tar_dataset(Path('data/raw/Task02_Heart.tar'), 'heart')

# Load
dataset = MedicalImageMaskDataset(
    image_dir=Path('data/processed/heart/images'),
    mask_dir=Path('data/processed/heart/masks')
)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
```

### Pattern 3: Custom Config
```python
from src.config import get_default_config

config = get_default_config()
config.learning_rate = 5e-5
config.block_consistency_weight = 0.3
config.num_epochs = 200
config.batch_size = 16

trainer = DiffusionTrainer(model, config)
trainer.fit(train_loader, val_loader)
```

### Pattern 4: Resume Training
```python
trainer = DiffusionTrainer(model, config)
trainer.load_checkpoint("models/checkpoints/best_model.pt")
trainer.fit(train_loader, val_loader)  # Continues from epoch N
```

---

**All modules are fully documented and ready to use!**

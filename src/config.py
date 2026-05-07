"""
Training configuration and utilities for Mask-Guided Diffusion
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json


@dataclass
class TrainingConfig:
    """Hyperparameters for training the diffusion model"""
    
    # Model architecture
    image_size: int = 256
    in_channels: int = 2
    model_channels: int = 128
    num_res_blocks: int = 2
    num_timesteps: int = 1000
    
    # Training
    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    
    # Data
    data_root: Path = Path("data/processed")
    image_dir: str = "images"
    mask_dir: str = "masks"
    num_workers: int = 4
    
    # Loss weights
    diffusion_loss_weight: float = 1.0
    block_consistency_weight: float = 0.1  # Novel loss weight
    
    # Checkpointing
    checkpoint_dir: Path = Path("models/checkpoints")
    save_every_n_epochs: int = 10
    val_every_n_epochs: int = 5
    
    # Logging
    log_every_n_steps: int = 100
    log_dir: Path = Path("logs")
    use_wandb: bool = False  # W&B integration (optional)
    
    # Device
    device: str = "cuda"
    mixed_precision: bool = False  # AMP (Automatic Mixed Precision)
    
    # Sampling
    num_sampling_steps: int = 50  # For inference (can be < num_timesteps)
    guidance_scale: float = 1.0
    
    def to_dict(self) -> dict:
        """Convert config to dictionary (for logging)"""
        d = self.__dict__.copy()
        d['data_root'] = str(d['data_root'])
        d['checkpoint_dir'] = str(d['checkpoint_dir'])
        d['log_dir'] = str(d['log_dir'])
        return d
    
    def save(self, path: Path):
        """Save config to JSON"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to {path}")
    
    @classmethod
    def from_json(cls, path: Path) -> 'TrainingConfig':
        """Load config from JSON"""
        with open(path, 'r') as f:
            d = json.load(f)
        d['data_root'] = Path(d['data_root'])
        d['checkpoint_dir'] = Path(d['checkpoint_dir'])
        d['log_dir'] = Path(d['log_dir'])
        return cls(**d)


def get_default_config() -> TrainingConfig:
    """Get default training configuration"""
    return TrainingConfig()


if __name__ == '__main__':
    config = get_default_config()
    print("Default Training Config:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Save example config
    config.save(Path("configs/default_config.json"))

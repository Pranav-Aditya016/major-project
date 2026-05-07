"""
Medical Image Generation with Mask-Guided Diffusion
Research package for MONAI-based conditional image synthesis
"""

from .dataset import MedicalImageMaskDataset, SimpleChestXrayDataset
from .diffusion_model import MaskConditionedDDPM, BlockConsistencyLoss
from .config import TrainingConfig, get_default_config
from .trainer import DiffusionTrainer
from .data_prep import DatasetManager

__version__ = "0.1.0"
__author__ = "Pranav Aditya"

__all__ = [
    "MedicalImageMaskDataset",
    "SimpleChestXrayDataset",
    "MaskConditionedDDPM",
    "BlockConsistencyLoss",
    "TrainingConfig",
    "get_default_config",
    "DiffusionTrainer",
    "DatasetManager"
]

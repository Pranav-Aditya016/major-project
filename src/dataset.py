"""
Medical Image Dataset Loader for Mask-Guided Diffusion
Loads chest X-ray images + binary lesion masks
Handles preprocessing, normalization, and augmentation
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional
import nibabel as nib
from PIL import Image


class MedicalImageMaskDataset(Dataset):
    """
    Dataset for medical images with binary lesion masks.
    
    Expected structure:
    data/
    ├── images/
    │   ├── img_001.nii.gz or .png
    │   └── ...
    └── masks/
        ├── mask_001.nii.gz or .png
        └── ...
    
    Args:
        image_dir (Path): Directory containing medical images
        mask_dir (Path): Directory containing binary lesion masks
        image_size (int): Resize images to (image_size, image_size)
        normalize (bool): Normalize images to [0, 1] or [-1, 1]
        normalization_type (str): 'zero_one' or 'neg_one_one'
    """
    
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_size: int = 256,
        normalize: bool = True,
        normalization_type: str = 'zero_one'
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.normalize = normalize
        self.normalization_type = normalization_type
        
        # Find all image files
        self.image_paths = sorted(
            list(self.image_dir.glob('*.nii.gz')) + 
            list(self.image_dir.glob('*.nii')) +
            list(self.image_dir.glob('*.png')) +
            list(self.image_dir.glob('*.jpg'))
        )
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        print(f"[Dataset] Found {len(self.image_paths)} images")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                'image': (1, H, W) normalized tensor
                'mask': (1, H, W) binary tensor {0, 1}
                'filename': str
        """
        image_path = self.image_paths[idx]
        mask_path = self.mask_dir / image_path.name
        
        # Load image
        image = self._load_image(image_path)
        
        # Load mask
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = self._load_mask(mask_path)
        
        # Resize to target size
        image = self._resize(image, self.image_size)
        mask = self._resize(mask, self.image_size)
        
        # Normalize image
        if self.normalize:
            image = self._normalize(image)
        
        # Ensure mask is binary {0, 1}
        mask = (mask > 0.5).astype(np.float32)
        
        # Convert to tensors (add channel dimension)
        image_tensor = torch.from_numpy(image[np.newaxis, ...]).float()  # (1, H, W)
        mask_tensor = torch.from_numpy(mask[np.newaxis, ...]).float()    # (1, H, W)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'filename': image_path.name
        }
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from NIfTI or standard image format"""
        if path.suffix in ['.nii', '.gz']:
            # NIfTI format (medical imaging standard)
            img = nib.load(path).get_fdata()
            # Handle 3D (take middle slice) or 2D
            if len(img.shape) == 3:
                img = img[:, :, img.shape[2] // 2]
        else:
            # Standard image format (PNG, JPG)
            img = np.array(Image.open(path).convert('L'))
        
        return img.astype(np.float32)
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load binary mask"""
        if path.suffix in ['.nii', '.gz']:
            mask = nib.load(path).get_fdata()
            if len(mask.shape) == 3:
                mask = mask[:, :, mask.shape[2] // 2]
        else:
            mask = np.array(Image.open(path).convert('L'))
        
        return mask.astype(np.float32)
    
    def _resize(self, img: np.ndarray, size: int) -> np.ndarray:
        """Resize image to target size"""
        if img.shape != (size, size):
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize((size, size), Image.BILINEAR)
            img = np.array(img_pil)
        return img
    
    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image intensities"""
        # Standard normalization: [0, 255] or [0, 4095] → [0, 1]
        img_min = img.min()
        img_max = img.max()
        
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        
        if self.normalization_type == 'neg_one_one':
            # Convert [0, 1] → [-1, 1]
            img = 2 * img - 1
        
        return img


class SimpleChestXrayDataset(Dataset):
    """
    Simplified dataset for testing. 
    Generates synthetic chest X-rays + masks if no real data available.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 256,
        normalization_type: str = 'zero_one'
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.normalization_type = normalization_type
        print(f"[Dataset] Using synthetic data: {num_samples} samples")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        """Generate synthetic chest X-ray + mask"""
        np.random.seed(idx)  # Reproducible
        
        # Synthetic chest X-ray (roughly realistic intensities)
        image = np.random.normal(loc=0.5, scale=0.15, size=(self.image_size, self.image_size))
        image = np.clip(image, 0, 1)
        
        # Add synthetic lung regions (darker in X-ray)
        y, x = np.ogrid[:self.image_size, :self.image_size]
        lung_mask = ((x - self.image_size/2)**2 + (y - self.image_size/2)**2) < (self.image_size/3)**2
        image[lung_mask] *= 0.6
        
        # Add synthetic lesion (bright spot in lung)
        # Scale lesion position and size to image size
        margin = max(5, self.image_size // 6)  # Adaptive margin
        lesion_y = np.random.randint(margin, self.image_size - margin)
        lesion_x = np.random.randint(margin, self.image_size - margin)
        lesion_radius = np.random.randint(
            max(3, self.image_size // 12),  # min radius
            max(5, self.image_size // 6)     # max radius
        )
        
        lesion_mask = ((x - lesion_x)**2 + (y - lesion_y)**2) < lesion_radius**2
        lesion_mask = lesion_mask & lung_mask
        image[lesion_mask] = np.clip(image[lesion_mask] + 0.3, 0, 1)
        
        # Binary mask (1 where lesion, 0 elsewhere)
        mask = lesion_mask.astype(np.float32)
        
        if self.normalization_type == 'neg_one_one':
            image = 2 * image - 1
        
        image_tensor = torch.from_numpy(image[np.newaxis, ...]).float()
        mask_tensor = torch.from_numpy(mask[np.newaxis, ...]).float()
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'filename': f'synthetic_{idx:06d}.png'
        }


class MedicalDecathlonDataset(Dataset):
    """
    Dataset loader for Medical Segmentation Decathlon (MSD) format.
    
    Supports:
    - Task02_Heart (Left Atrium MRI)
    - Task05_Prostate (Prostate MRI)
    - Any MSD-format dataset with 3D NIfTI images
    
    Extracts 2D slices from 3D volumes for diffusion training.
    
    Args:
        data_dir: Path to extracted MSD task folder (e.g., data/processed/Task02_Heart)
        image_size: Target image size (will resize slices)
        split: 'train' or 'test'
        slice_axis: Axis to slice along (0=sagittal, 1=coronal, 2=axial)
        min_mask_ratio: Minimum mask coverage to include a slice (filters empty slices)
        normalization_type: 'zero_one' or 'neg_one_one'
    """
    
    def __init__(
        self,
        data_dir: Path,
        image_size: int = 128,
        split: str = 'train',
        slice_axis: int = 2,  # Axial slices by default
        min_mask_ratio: float = 0.001,  # At least 0.1% mask coverage
        normalization_type: str = 'zero_one',
        max_slices_per_volume: int = 20,  # Limit slices per volume
    ):
        import json
        
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.split = split
        self.slice_axis = slice_axis
        self.min_mask_ratio = min_mask_ratio
        self.normalization_type = normalization_type
        self.max_slices_per_volume = max_slices_per_volume
        
        # Load dataset metadata
        dataset_json = self.data_dir / 'dataset.json'
        if not dataset_json.exists():
            raise FileNotFoundError(f"dataset.json not found in {self.data_dir}")
        
        with open(dataset_json, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"[MSD Dataset] Loading: {self.metadata['name']}")
        print(f"  Description: {self.metadata['description']}")
        print(f"  Modality: {self.metadata['modality']}")
        print(f"  Labels: {self.metadata['labels']}")
        
        # Build slice index
        self.slices = []
        self._build_slice_index()
        
        print(f"  Total slices ({split}): {len(self.slices)}")
    
    def _build_slice_index(self):
        """Pre-compute list of (volume_path, label_path, slice_idx) tuples"""
        
        if self.split == 'train':
            pairs = self.metadata.get('training', [])
        else:
            # Test set has no labels - use training with split
            pairs = self.metadata.get('training', [])
            # Use last 20% as validation
            split_idx = int(len(pairs) * 0.8)
            pairs = pairs[split_idx:] if self.split == 'val' else pairs[:split_idx]
        
        for pair in pairs:
            image_path = self.data_dir / pair['image'].replace('./', '')
            label_path = self.data_dir / pair['label'].replace('./', '')
            
            if not image_path.exists() or not label_path.exists():
                print(f"  Warning: Missing {image_path.name}")
                continue
            
            # Load volume to get slice count
            try:
                label_vol = nib.load(label_path).get_fdata()
                
                # Handle 4D data (take first channel for multimodal)
                if len(label_vol.shape) == 4:
                    label_vol = label_vol[..., 0]
                
                n_slices = label_vol.shape[self.slice_axis]
                
                # Find slices with sufficient mask coverage
                slice_count = 0
                for s in range(n_slices):
                    if self.slice_axis == 0:
                        mask_slice = label_vol[s, :, :]
                    elif self.slice_axis == 1:
                        mask_slice = label_vol[:, s, :]
                    else:
                        mask_slice = label_vol[:, :, s]
                    
                    # Check mask coverage
                    mask_ratio = (mask_slice > 0).sum() / mask_slice.size
                    
                    if mask_ratio >= self.min_mask_ratio:
                        self.slices.append({
                            'image_path': image_path,
                            'label_path': label_path,
                            'slice_idx': s
                        })
                        slice_count += 1
                        
                        if slice_count >= self.max_slices_per_volume:
                            break
                            
            except Exception as e:
                print(f"  Error loading {image_path.name}: {e}")
                continue
    
    def __len__(self) -> int:
        return len(self.slices)
    
    def __getitem__(self, idx: int) -> dict:
        slice_info = self.slices[idx]
        
        # Load volumes
        image_vol = nib.load(slice_info['image_path']).get_fdata()
        label_vol = nib.load(slice_info['label_path']).get_fdata()
        s = slice_info['slice_idx']
        
        # Handle 4D data (multimodal - take first modality)
        if len(image_vol.shape) == 4:
            image_vol = image_vol[..., 0]
        if len(label_vol.shape) == 4:
            label_vol = label_vol[..., 0]
        
        # Extract 2D slice
        if self.slice_axis == 0:
            image = image_vol[s, :, :]
            mask = label_vol[s, :, :]
        elif self.slice_axis == 1:
            image = image_vol[:, s, :]
            mask = label_vol[:, s, :]
        else:
            image = image_vol[:, :, s]
            mask = label_vol[:, :, s]
        
        # Convert to float32
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        
        # Resize to target size
        image = self._resize(image, self.image_size)
        mask = self._resize(mask, self.image_size, is_mask=True)
        
        # Normalize image to [0, 1]
        image = self._normalize(image)
        
        # Binarize mask (any label > 0 becomes 1)
        mask = (mask > 0).astype(np.float32)
        
        if self.normalization_type == 'neg_one_one':
            image = 2 * image - 1
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image[np.newaxis, ...]).float()
        mask_tensor = torch.from_numpy(mask[np.newaxis, ...]).float()
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'filename': f"{slice_info['image_path'].stem}_s{s}"
        }
    
    def _resize(self, img: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
        """Resize image to target size"""
        if img.shape[0] != size or img.shape[1] != size:
            mode = Image.NEAREST if is_mask else Image.BILINEAR
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize((size, size), mode)
            img = np.array(img_pil, dtype=np.float32)
        return img
    
    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1] using percentile clipping"""
        # Use percentile clipping for MRI (handles outliers)
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        
        if p99 > p1:
            img = (img - p1) / (p99 - p1)
        else:
            img = np.zeros_like(img)
        
        return img


def get_msd_dataset(task_name: str, data_root: str = 'data/processed', **kwargs):
    """
    Convenience function to load a Medical Segmentation Decathlon dataset.
    
    Args:
        task_name: 'heart', 'prostate', 'Task02_Heart', etc.
        data_root: Root directory containing extracted task folders
        **kwargs: Additional arguments for MedicalDecathlonDataset
        
    Returns:
        MedicalDecathlonDataset instance
    """
    task_mapping = {
        'heart': 'Task02_Heart',
        'prostate': 'Task05_Prostate',
        'lung': 'Task06_Lung',
        'liver': 'Task03_Liver',
        'spleen': 'Task09_Spleen',
    }
    
    # Normalize task name
    if task_name.lower() in task_mapping:
        task_folder = task_mapping[task_name.lower()]
    else:
        task_folder = task_name
    
    data_dir = Path(data_root) / task_folder
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_dir}\n"
            f"Available: {list(Path(data_root).glob('Task*'))}"
        )
    
    return MedicalDecathlonDataset(data_dir, **kwargs)


if __name__ == '__main__':
    # Test synthetic dataset
    print("Testing SimpleChestXrayDataset...")
    dataset = SimpleChestXrayDataset(num_samples=10, image_size=256)
    
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"Mask values: {torch.unique(sample['mask'])}")
    print("✓ Dataset test passed!")


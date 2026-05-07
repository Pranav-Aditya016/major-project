"""
Data Preparation Utilities for Medical Image Datasets
Handles extraction and preprocessing of Heart, Prostate, Lung datasets
"""

import tarfile
import numpy as np
from pathlib import Path
from typing import Tuple
import nibabel as nib
from PIL import Image
import shutil


class DatasetManager:
    """Manage dataset extraction and preprocessing"""
    
    def __init__(self, data_root: Path = Path("data")):
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_tar_dataset(self, tar_path: Path, organ: str):
        """
        Extract tar dataset and organize into images/masks structure
        
        Args:
            tar_path: Path to .tar file
            organ: 'heart', 'prostate', 'lung', 'brain', 'breast'
        """
        print(f"Extracting {organ} dataset from {tar_path}...")
        
        extract_dir = self.raw_dir / organ
        extract_dir.mkdir(exist_ok=True)
        
        # Extract tar
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_dir)
        
        print(f"✓ Extracted to {extract_dir}")
        
        # Organize extracted files
        self._organize_dataset(extract_dir, organ)
    
    def _organize_dataset(self, extract_dir: Path, organ: str):
        """
        Organize raw extracted data into:
        processed/{organ}/
        ├── images/
        │   ├── img_001.nii.gz
        │   └── ...
        └── masks/
            ├── mask_001.nii.gz
            └── ...
        """
        organ_processed = self.processed_dir / organ
        images_dir = organ_processed / "images"
        masks_dir = organ_processed / "masks"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files (common patterns)
        image_patterns = ['*.nii.gz', '*.nii', '*_0000.nii.gz']
        mask_patterns = ['*_seg.nii.gz', '*_label.nii.gz', '*mask.nii.gz']
        
        image_count = 0
        mask_count = 0
        
        # Recursively search for images and masks
        for pattern in image_patterns:
            for img_path in extract_dir.rglob(pattern):
                # Skip if it's a mask/label file
                if any(x in img_path.name for x in ['seg', 'label', 'mask']):
                    continue
                
                # Create indexed filename
                dest_name = f"{organ}_img_{image_count:04d}.nii.gz"
                dest_path = images_dir / dest_name
                
                if not dest_path.exists():
                    shutil.copy2(img_path, dest_path)
                    image_count += 1
        
        for pattern in mask_patterns:
            for mask_path in extract_dir.rglob(pattern):
                dest_name = f"{organ}_mask_{mask_count:04d}.nii.gz"
                dest_path = masks_dir / dest_name
                
                if not dest_path.exists():
                    shutil.copy2(mask_path, dest_path)
                    mask_count += 1
        
        print(f"✓ Organized {image_count} images and {mask_count} masks")
    
    def convert_nifti_to_png(
        self,
        organ: str,
        slice_selection: str = 'middle'
    ):
        """
        Convert 3D NIfTI volumes to 2D PNG slices for easier handling
        
        Args:
            organ: 'heart', 'prostate', etc.
            slice_selection: 'middle' or 'all' (all creates dataset from all slices)
        """
        images_dir = self.processed_dir / organ / "images"
        masks_dir = self.processed_dir / organ / "masks"
        
        # Create PNG directories
        png_images_dir = self.processed_dir / organ / "images_png"
        png_masks_dir = self.processed_dir / organ / "masks_png"
        
        png_images_dir.mkdir(exist_ok=True)
        png_masks_dir.mkdir(exist_ok=True)
        
        nifti_files = sorted(images_dir.glob("*.nii.gz"))
        
        for idx, nifti_path in enumerate(nifti_files):
            print(f"Processing {idx+1}/{len(nifti_files)}: {nifti_path.name}")
            
            # Load image
            img_data = nib.load(nifti_path).get_fdata()
            
            # Find corresponding mask
            mask_name = nifti_path.name.replace('_img_', '_mask_')
            mask_path = masks_dir / mask_name
            
            if not mask_path.exists():
                print(f"  ⚠ Mask not found: {mask_path}")
                continue
            
            mask_data = nib.load(mask_path).get_fdata()
            
            # Handle 3D volumes
            if len(img_data.shape) == 3:
                if slice_selection == 'middle':
                    slices_to_process = [img_data.shape[2] // 2]
                else:  # 'all'
                    slices_to_process = range(img_data.shape[2])
            else:
                slices_to_process = [0]
            
            # Save selected slices
            for slice_idx in slices_to_process:
                if len(img_data.shape) == 3:
                    img_slice = img_data[:, :, slice_idx]
                    mask_slice = mask_data[:, :, slice_idx]
                else:
                    img_slice = img_data
                    mask_slice = mask_data
                
                # Normalize to 0-255
                img_slice_norm = ((img_slice - img_slice.min()) /
                                  (img_slice.max() - img_slice.min() + 1e-6) * 255).astype(np.uint8)
                mask_slice_norm = (mask_slice > 0).astype(np.uint8) * 255
                
                # Save as PNG
                file_id = f"{idx:04d}_slice_{slice_idx:03d}"
                
                Image.fromarray(img_slice_norm).save(
                    png_images_dir / f"{organ}_img_{file_id}.png"
                )
                Image.fromarray(mask_slice_norm).save(
                    png_masks_dir / f"{organ}_mask_{file_id}.png"
                )
        
        print(f"✓ Converted to PNG: {png_images_dir}")
    
    def get_dataset_info(self, organ: str) -> dict:
        """Get info about a processed dataset"""
        images_dir = self.processed_dir / organ / "images"
        masks_dir = self.processed_dir / organ / "masks"
        
        num_images = len(list(images_dir.glob("*.nii.gz")))
        num_masks = len(list(masks_dir.glob("*.nii.gz")))
        
        return {
            'organ': organ,
            'num_images': num_images,
            'num_masks': num_masks,
            'images_dir': images_dir,
            'masks_dir': masks_dir
        }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['extract', 'convert'], default='extract')
    parser.add_argument('--tar-path', type=Path, help='Path to tar file')
    parser.add_argument('--organ', type=str, default='heart', 
                       choices=['heart', 'prostate', 'lung', 'brain', 'breast'])
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    
    args = parser.parse_args()
    
    manager = DatasetManager(args.data_root)
    
    if args.action == 'extract':
        if args.tar_path is None:
            raise ValueError("--tar-path required for extract action")
        manager.extract_tar_dataset(args.tar_path, args.organ)
        # Print dataset info
        info = manager.get_dataset_info(args.organ)
        print(f"\nDataset Info: {info}")
    
    elif args.action == 'convert':
        print(f"Converting {args.organ} dataset to PNG...")
        manager.convert_nifti_to_png(args.organ, slice_selection='middle')

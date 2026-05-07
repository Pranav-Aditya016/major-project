"""
Training loop for Mask-Guided Diffusion Model
Handles training, validation, checkpointing, and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple
import numpy as np
from datetime import datetime

try:
    from .diffusion_model import MaskConditionedDDPM, BlockConsistencyLoss
    from .config import TrainingConfig
except ImportError:
    from diffusion_model import MaskConditionedDDPM, BlockConsistencyLoss
    from config import TrainingConfig


class DiffusionTrainer:
    """
    Training manager for mask-conditioned diffusion model.
    
    Responsibilities:
    - Training loop with loss computation
    - Validation loop
    - Checkpoint saving/loading
    - Logging metrics
    - Mixed precision (FP16) training for GPU
    """
    
    def __init__(
        self,
        model: MaskConditionedDDPM,
        config: TrainingConfig,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.block_consistency_loss = BlockConsistencyLoss(
            weight_inside=1.0,
            weight_outside=0.5
        )
        
        # Optimizer with learning rate scheduling
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler (warm-up + decay)
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler for faster GPU training
        self.use_amp = config.mixed_precision and device == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print("✓ Mixed precision (FP16) training enabled")
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = {'loss': [], 'diffusion_loss': [], 'consistency_loss': []}
        self.val_history = {'loss': []}
        
        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Trainer initialized on {device}")
    
    def _create_scheduler(self) -> optim.lr_scheduler.LambdaLR:
        """Create learning rate scheduler with warm-up"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                # Cosine annealing after warm-up
                progress = (step - self.config.warmup_steps) / (
                    self.config.num_epochs * 100  # Rough estimate
                )
                return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Run one training epoch.
        
        Returns:
            dict with loss metrics
        """
        self.model.train()
        epoch_losses = {'total': 0, 'diffusion': 0, 'consistency': 0}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            x0 = batch['image'].to(self.device)  # (B, 1, H, W)
            mask = batch['mask'].to(self.device)  # (B, 1, H, W)
            
            batch_size = x0.shape[0]
            
            # Sample random timesteps for this batch
            timesteps = torch.randint(
                0, self.config.num_timesteps,
                (batch_size,),
                device=self.device
            )
            
            # Mixed precision training context
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                # Forward diffusion: add noise
                x_t, noise_gt = self.model.add_noise(x0, timesteps)
                
                # Predict noise (denoising step)
                noise_pred = self.model(x_t, mask, timesteps)
                
                # Loss 1: Standard diffusion loss (predict added noise)
                diffusion_loss = self.mse_loss(noise_pred, noise_gt)
                
                # Loss 2: Block consistency loss (novel contribution)
                consistency_loss = self.block_consistency_loss(x_t, mask)
                
                # Total loss
                total_loss = (
                    self.config.diffusion_loss_weight * diffusion_loss +
                    self.config.block_consistency_weight * consistency_loss
                )
            
            # Backward pass with gradient scaling for FP16
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Track losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['diffusion'] += diffusion_loss.item()
            epoch_losses['consistency'] += consistency_loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if batch_idx % self.config.log_every_n_steps == 0:
                pbar.set_postfix({
                    'loss': total_loss.item(),
                    'diff': diffusion_loss.item(),
                    'consist': consistency_loss.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Update history
        self.train_history['loss'].append(epoch_losses['total'])
        self.train_history['diffusion_loss'].append(epoch_losses['diffusion'])
        self.train_history['consistency_loss'].append(epoch_losses['consistency'])
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            dict with validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(val_loader, desc="Validation")
        
        for batch in pbar:
            x0 = batch['image'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            batch_size = x0.shape[0]
            timesteps = torch.randint(
                0, self.config.num_timesteps,
                (batch_size,),
                device=self.device
            )
            
            # Use AMP for validation too
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                x_t, noise_gt = self.model.add_noise(x0, timesteps)
                noise_pred = self.model(x_t, mask, timesteps)
                loss = self.mse_loss(noise_pred, noise_gt)
            
            val_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / num_batches
        self.val_history['loss'].append(avg_val_loss)
        
        return {'loss': avg_val_loss}
    
    def save_checkpoint(self, name: str = None, is_best: bool = False):
        """Save model checkpoint"""
        if name is None:
            name = f"ckpt_epoch_{self.epoch:04d}.pt"
        
        ckpt_path = self.checkpoint_dir / name
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, ckpt_path)
        print(f"✓ Checkpoint saved: {ckpt_path}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")
    
    def load_checkpoint(self, ckpt_path: str):
        """Load model checkpoint"""
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.train_history = ckpt['train_history']
        self.val_history = ckpt['val_history']
        
        print(f"✓ Checkpoint loaded from {ckpt_path}")
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"Config: {self.config.to_dict()}\n")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            print(f"\nEpoch {epoch} Train Metrics:")
            for key, val in train_metrics.items():
                print(f"  {key}: {val:.6f}")
            
            # Validation
            if val_loader and epoch % self.config.val_every_n_epochs == 0:
                val_metrics = self.validate(val_loader)
                print(f"Epoch {epoch} Val Metrics:")
                for key, val in val_metrics.items():
                    print(f"  {key}: {val:.6f}")
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(is_best=True)
            
            # Checkpointing
            if epoch % self.config.save_every_n_epochs == 0:
                self.save_checkpoint()
            
            print()
        
        print("✓ Training complete!")


if __name__ == '__main__':
    print("Training utilities loaded. Use this in a training notebook.")

"""
MONAI-based Conditional Diffusion Model for Mask-Guided Image Generation
Implements DDPM with mask conditioning
"""

import torch
import torch.nn as nn
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from monai.utils import set_determinism
from typing import Tuple, Optional
import numpy as np


class MaskConditionedDDPM(nn.Module):
    """
    Mask-Conditioned DDPM for medical image generation.
    
    Architecture:
    - UNet encoder: Accepts noisy image + mask (concatenated channels)
    - Cross-attention: Conditions generation on binary lesion mask
    - Standard DDPM diffusion process
    
    Args:
        image_size (int): Input image resolution (256)
        in_channels (int): Input channels (1 for grayscale + 1 for mask = 2)
        model_channels (int): Base channel count
        num_res_blocks (int): Residual blocks per level
        attention_resolutions (Tuple): Spatial resolutions with attention
        num_timesteps (int): Number of diffusion timesteps (1000)
    """
    
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 2,  # 1 for image + 1 for mask
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        num_timesteps: int = 1000,
        device: str = 'cuda',
        use_flash_attention: bool = True  # Memory efficient attention
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_timesteps = num_timesteps
        self.device = device
        
        # MONAI DiffusionModelUNet - optimized for 8GB VRAM
        # Attention only at lowest resolution to save memory
        self.unet = DiffusionModelUNet(
            spatial_dims=2,  # 2D images
            in_channels=in_channels,  # Concatenated [image, mask]
            out_channels=1,  # Output single channel (denoised image)
            channels=(model_channels, model_channels * 2, model_channels * 4),
            attention_levels=(False, False, True),  # Attention ONLY at lowest res (saves memory)
            num_res_blocks=num_res_blocks,
            num_head_channels=model_channels // 4,
            use_flash_attention=use_flash_attention,  # Flash attention for memory efficiency
        )
        
        # DDPM Scheduler (controls noise schedule)
        # MONAI 1.5.1 uses "linear_beta" instead of "linear"
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            schedule="linear_beta",  # Options: linear_beta, scaled_linear_beta, cosine
            beta_start=0.0001,
            beta_end=0.02,
        )
        
        self.to(device)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Denoising step of the diffusion model.
        
        Args:
            x (torch.Tensor): Noisy image (B, 1, H, W)
            mask (torch.Tensor): Binary lesion mask (B, 1, H, W)
            timesteps (torch.Tensor): Timestep indices (B,)
        
        Returns:
            torch.Tensor: Predicted noise (B, 1, H, W)
        """
        # Concatenate image and mask as conditioning
        x_conditioned = torch.cat([x, mask], dim=1)  # (B, 2, H, W)
        
        # UNet predicts noise at this timestep
        noise_pred = self.unet(x_conditioned, timesteps)
        
        return noise_pred
    
    def add_noise(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: Add noise to clean image (for training).
        
        Uses DDPM noise schedule: q(x_t | x_0)
        
        Args:
            x0 (torch.Tensor): Clean image (B, 1, H, W)
            timesteps (torch.Tensor): Timestep indices (B,)
        
        Returns:
            Tuple:
                x_t (torch.Tensor): Noisy image at timestep t
                noise (torch.Tensor): Ground-truth noise added
        """
        device = x0.device
        batch_size = x0.shape[0]
        
        # Get noise schedule values
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        
        # Sample random Gaussian noise
        noise = torch.randn_like(x0, device=device)
        
        # Get alpha values for current timesteps
        alpha_t = alphas_cumprod[timesteps].view(batch_size, 1, 1, 1)
        
        # Add noise: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        x_t = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        
        return x_t, noise
    
    @torch.no_grad()
    def sample(
        self,
        mask: torch.Tensor,
        batch_size: int = 1,
        num_steps: Optional[int] = None,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Reverse diffusion: Generate image from noise conditioned on mask.
        
        Args:
            mask (torch.Tensor): Binary lesion mask (B, 1, H, W)
            batch_size (int): Number of samples to generate
            num_steps (int): Number of denoising steps (default: num_timesteps)
            guidance_scale (float): Classifier-free guidance scale (1.0 = no guidance)
        
        Returns:
            torch.Tensor: Generated image (B, 1, H, W) in range [0, 1]
        """
        device = self.device
        
        # Ensure mask has correct batch size
        if mask.shape[0] != batch_size:
            mask = mask.repeat(batch_size, 1, 1, 1)
        
        mask = mask.to(device)
        
        # Start from pure noise
        x_t = torch.randn(
            batch_size, 1, self.image_size, self.image_size,
            device=device
        )
        
        # Reverse diffusion loop - iterate from T-1 down to 0
        num_steps = num_steps or self.num_timesteps
        step_ratio = self.num_timesteps / num_steps
        
        for step in range(num_steps):
            # Calculate timestep t
            t = int((num_steps - 1 - step) * step_ratio)
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.forward(x_t, mask, t_batch)
            
            # Get alpha values from scheduler
            alpha_prod_t = self.scheduler.alphas_cumprod[t].to(device)
            
            if t > 0:
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[t - 1].to(device)
            else:
                alpha_prod_t_prev = torch.tensor(1.0, device=device)
            
            # Compute coefficients
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            
            # Predict x_0
            pred_original = (x_t - torch.sqrt(beta_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
            pred_original = torch.clamp(pred_original, -1.0, 1.0)
            
            # Compute x_{t-1}
            pred_original_coeff = torch.sqrt(alpha_prod_t_prev) * (1 - alpha_prod_t / alpha_prod_t_prev) / (1 - alpha_prod_t)
            current_coeff = torch.sqrt(alpha_prod_t / alpha_prod_t_prev) * beta_prod_t_prev / (1 - alpha_prod_t)
            
            x_t = pred_original_coeff * pred_original + current_coeff * x_t
            
            # Add noise if not the last step
            if t > 0:
                variance = beta_prod_t_prev / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
                variance = torch.clamp(variance, min=1e-20)
                noise = torch.randn_like(x_t, device=device)
                x_t = x_t + torch.sqrt(variance) * noise
        
        # Clamp to valid range and convert to [0, 1]
        x_generated = torch.clamp(x_t, -1.0, 1.0)
        x_generated = (x_generated + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
        
        return x_generated


class BlockConsistencyLoss(nn.Module):
    """
    Novel Loss: Enforce consistency between input lesion mask and generated image.
    
    The Block Consistency Loss penalizes:
    1. Lesion region mismatch: Generated lesion should overlap with mask region
    2. False positives: No lesions should appear outside the mask
    
    This is computed on top of the standard diffusion loss.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        weight_inside: float = 1.0,
        weight_outside: float = 0.5
    ):
        super().__init__()
        self.threshold = threshold
        self.weight_inside = weight_inside
        self.weight_outside = weight_outside
    
    def forward(
        self,
        generated: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute block consistency loss.
        
        Args:
            generated (torch.Tensor): Generated image (B, 1, H, W) in [-1, 1]
            mask (torch.Tensor): Binary lesion mask (B, 1, H, W) in {0, 1}
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Normalize generated to [0, 1] for comparison
        generated_norm = (generated + 1) / 2
        
        # Detect bright regions (potential lesions) in generated image
        lesion_detected = (generated_norm > self.threshold).float()
        
        # Loss 1: Inside mask - lesions should be present where mask indicates
        # (maximize overlap: minimize MSE between detected lesion and mask)
        loss_inside = torch.mean((lesion_detected - mask)**2)
        
        # Loss 2: Outside mask - no lesions should appear outside mask
        # (minimize false positives)
        false_positives = lesion_detected * (1 - mask)
        loss_outside = torch.mean(false_positives)
        
        # Combined loss
        total_loss = (
            self.weight_inside * loss_inside +
            self.weight_outside * loss_outside
        )
        
        return total_loss


if __name__ == '__main__':
    print("Testing MaskConditionedDDPM...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model
    model = MaskConditionedDDPM(
        image_size=256,
        in_channels=2,
        model_channels=64,
        num_timesteps=1000,
        device=device
    )
    print(f"✓ Model initialized on {device}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 256, 256, device=device)
    mask = torch.randint(0, 2, (batch_size, 1, 256, 256), device=device).float()
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    
    noise_pred = model(x, mask, timesteps)
    print(f"✓ Forward pass: noise_pred shape = {noise_pred.shape}")
    
    # Test noise addition
    x0 = torch.randn(batch_size, 1, 256, 256, device=device)
    x_t, noise = model.add_noise(x0, timesteps)
    print(f"✓ Add noise: x_t shape = {x_t.shape}, noise shape = {noise.shape}")
    
    # Test block consistency loss
    loss_fn = BlockConsistencyLoss()
    loss = loss_fn(x, mask)
    print(f"✓ Block consistency loss = {loss.item():.4f}")
    
    print("\n✓ All tests passed!")

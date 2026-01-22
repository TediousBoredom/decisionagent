"""
Diffusion Model for Trading Strategy Generation

This module implements a conditional diffusion model that learns to generate
trading strategies by modeling the distribution of high-return trading actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time and condition embeddings."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, 
                 cond_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, out_channels),
            nn.Dropout(dropout)
        )
        
        self.block2 = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.Dropout(dropout)
        )
        
        self.residual_conv = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, 
                cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, None, :]
        
        # Add condition embedding
        cond_emb = self.cond_mlp(cond_emb)
        h = h + cond_emb[:, None, :]
        
        h = self.block2(h)
        
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention block."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h, _ = self.attention(h, h, h)
        return x + self.dropout(h)


class DiffusionPolicyNetwork(nn.Module):
    """
    Neural network for denoising in the diffusion process.
    Takes noisy actions, timestep, and market conditions as input.
    """
    
    def __init__(
        self,
        action_dim: int,
        seq_length: int,
        hidden_dim: int = 256,
        time_emb_dim: int = 128,
        cond_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.seq_length = seq_length
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(action_dim, hidden_dim)
        
        # Residual blocks with attention
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                ResidualBlock(hidden_dim, hidden_dim, time_emb_dim, cond_dim, dropout),
                AttentionBlock(hidden_dim, num_heads, dropout)
            ])
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy actions [batch, seq_length, action_dim]
            t: Timestep [batch]
            cond: Market condition embedding [batch, cond_dim]
        
        Returns:
            Predicted noise [batch, seq_length, action_dim]
        """
        # Embed timestep
        time_emb = self.time_embed(t)
        
        # Project input
        h = self.input_proj(x)
        
        # Apply residual blocks with attention
        for res_block, attn_block in self.blocks:
            h = res_block(h, time_emb, cond)
            h = attn_block(h)
        
        # Project to output
        return self.output_proj(h)


class TradingDiffusionModel(nn.Module):
    """
    Complete diffusion model for trading strategy generation.
    Implements DDPM (Denoising Diffusion Probabilistic Models) for trading actions.
    """
    
    def __init__(
        self,
        action_dim: int,
        seq_length: int,
        hidden_dim: int = 256,
        time_emb_dim: int = 128,
        cond_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.seq_length = seq_length
        self.num_timesteps = num_timesteps
        
        # Denoising network
        self.denoise_net = DiffusionPolicyNetwork(
            action_dim=action_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            cond_dim=cond_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Diffusion schedule
        self.register_buffer('betas', self._cosine_beta_schedule(num_timesteps, beta_start, beta_end))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', 
                           F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1.0 - self.alphas_cumprod))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_variance',
                           self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008, 
                             beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
        """Cosine schedule for beta values."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to clean actions.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, 
                 cond: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Training loss: predict noise added to actions.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Add noise to actions
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self.denoise_net(x_noisy, t, cond)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, cond: torch.Tensor) -> torch.Tensor:
        """
        Reverse diffusion process: p(x_{t-1} | x_t)
        Single denoising step.
        """
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x.device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = self.denoise_net(x, t_tensor, cond)
        
        # Calculate mean
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        mean = (x - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_t)
        
        if t > 0:
            noise = torch.randn_like(x)
            variance = self.posterior_variance[t]
            return mean + torch.sqrt(variance) * noise
        else:
            return mean
    
    @torch.no_grad()
    def sample(self, batch_size: int, cond: torch.Tensor, 
               device: str = 'cuda') -> torch.Tensor:
        """
        Generate trading actions from noise.
        
        Args:
            batch_size: Number of samples to generate
            cond: Market condition embedding [batch_size, cond_dim]
            device: Device to generate on
        
        Returns:
            Generated actions [batch_size, seq_length, action_dim]
        """
        # Start from pure noise
        x = torch.randn(batch_size, self.seq_length, self.action_dim, device=device)
        
        # Iteratively denoise
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t, cond)
        
        return x
    
    @torch.no_grad()
    def ddim_sample(self, batch_size: int, cond: torch.Tensor, 
                    num_steps: int = 50, eta: float = 0.0, 
                    device: str = 'cuda') -> torch.Tensor:
        """
        Fast sampling using DDIM (Denoising Diffusion Implicit Models).
        
        Args:
            batch_size: Number of samples
            cond: Market condition embedding
            num_steps: Number of denoising steps (fewer than training steps)
            eta: Stochasticity parameter (0 = deterministic)
            device: Device to generate on
        
        Returns:
            Generated actions
        """
        # Create subset of timesteps
        step_size = self.num_timesteps // num_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=device)
        timesteps = torch.flip(timesteps, [0])
        
        # Start from noise
        x = torch.randn(batch_size, self.seq_length, self.action_dim, device=device)
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.denoise_net(x, t_tensor, cond)
            
            # Get alpha values
            alpha_cumprod_t = self.alphas_cumprod[t]
            
            if i < len(timesteps) - 1:
                alpha_cumprod_t_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=device)
            
            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - eta ** 2 * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)) * predicted_noise
            
            # Random noise
            if eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x)
                sigma_t = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            else:
                noise = 0
                sigma_t = 0
            
            # Update x
            x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + sigma_t * noise
        
        return x
    
    def forward(self, x_start: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.
        
        Args:
            x_start: Clean actions [batch, seq_length, action_dim]
            cond: Market conditions [batch, cond_dim]
        
        Returns:
            Loss value
        """
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_start.device).long()
        
        # Calculate loss
        loss = self.p_losses(x_start, t, cond)
        
        return loss


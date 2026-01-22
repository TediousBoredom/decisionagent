"""
Diffusion Policy Model for Trading Strategy Learning

This module implements a diffusion-based policy that learns from human trading strategies
and generates trading actions conditioned on market states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timestep encoding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time and condition embedding."""
    
    def __init__(self, dim: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim),
            nn.Mish()
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb
        h = self.mlp(h)
        h = self.dropout(h)
        return x + self.norm2(h)


class AttentionBlock(nn.Module):
    """Multi-head self-attention block."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h, _ = self.attention(h, h, h)
        h = self.dropout(h)
        return x + h


class DiffusionPolicyNetwork(nn.Module):
    """
    Diffusion-based policy network for trading strategy learning.
    
    The network learns to denoise trading actions conditioned on market states,
    effectively learning the distribution of successful trading strategies.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        time_dim: int = 128
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbedding(time_dim)
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Residual blocks with attention
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                ResidualBlock(hidden_dim, time_dim, dropout),
                AttentionBlock(hidden_dim, num_heads, dropout)
            ])
            for _ in range(num_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the diffusion policy network.
        
        Args:
            noisy_action: Noisy action at timestep t, shape (batch, action_dim)
            timestep: Diffusion timestep, shape (batch,)
            state: Market state condition, shape (batch, state_dim)
        
        Returns:
            Predicted noise or denoised action, shape (batch, action_dim)
        """
        # Encode timestep
        time_emb = self.time_embedding(timestep)
        
        # Encode state and action
        state_emb = self.state_encoder(state)
        action_emb = self.action_encoder(noisy_action)
        
        # Fuse state and action
        x = self.fusion(torch.cat([state_emb, action_emb], dim=-1))
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Process through residual and attention blocks
        for res_block, attn_block in self.blocks:
            x = res_block(x, time_emb)
            x = attn_block(x)
        
        x = x.squeeze(1)  # Remove sequence dimension
        
        # Output prediction
        output = self.output_head(x)
        
        return output


class DiffusionPolicy:
    """
    Complete diffusion policy for trading strategy learning and inference.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_diffusion_steps: int = 100,
        beta_schedule: str = "cosine",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.num_diffusion_steps = num_diffusion_steps
        self.action_dim = action_dim
        
        # Initialize network
        self.network = DiffusionPolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        ).to(device)
        
        # Setup noise schedule
        self.betas = self._get_beta_schedule(beta_schedule, num_diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values for diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def _get_beta_schedule(self, schedule: str, num_steps: int) -> torch.Tensor:
        """Get noise schedule."""
        if schedule == "linear":
            return torch.linspace(1e-4, 0.02, num_steps)
        elif schedule == "cosine":
            steps = torch.arange(num_steps + 1, dtype=torch.float32) / num_steps
            alphas_cumprod = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        elif schedule == "quadratic":
            return torch.linspace(1e-4 ** 0.5, 0.02 ** 0.5, num_steps) ** 2
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    def add_noise(
        self,
        action: torch.Tensor,
        timestep: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to action at given timestep."""
        if noise is None:
            noise = torch.randn_like(action)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timestep].to(action.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timestep].to(action.device)
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(action.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_action = sqrt_alpha_prod * action + sqrt_one_minus_alpha_prod * noise
        return noisy_action, noise
    
    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,
        num_samples: int = 1,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Sample trading actions from the learned policy.
        
        Args:
            state: Market state condition, shape (batch, state_dim)
            num_samples: Number of action samples to generate
            guidance_scale: Classifier-free guidance scale
        
        Returns:
            Sampled actions, shape (batch, num_samples, action_dim)
        """
        batch_size = state.shape[0]
        
        # Start from random noise
        action = torch.randn(batch_size, num_samples, self.action_dim).to(self.device)
        
        # Expand state for multiple samples
        state_expanded = state.unsqueeze(1).expand(-1, num_samples, -1)
        state_expanded = state_expanded.reshape(-1, state.shape[-1])
        
        # Denoise iteratively
        for t in reversed(range(self.num_diffusion_steps)):
            timestep = torch.full((batch_size * num_samples,), t, device=self.device, dtype=torch.long)
            action_flat = action.reshape(-1, self.action_dim)
            
            # Predict noise
            predicted_noise = self.network(action_flat, timestep, state_expanded)
            
            # Compute denoised action
            alpha = self.alphas[t].to(self.device)
            alpha_cumprod = self.alphas_cumprod[t].to(self.device)
            beta = self.betas[t].to(self.device)
            
            # Denoising step
            action_flat = (action_flat - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
            
            # Add noise (except for last step)
            if t > 0:
                noise = torch.randn_like(action_flat)
                variance = self.posterior_variance[t].to(self.device)
                action_flat = action_flat + torch.sqrt(variance) * noise
            
            action = action_flat.reshape(batch_size, num_samples, self.action_dim)
        
        return action
    
    def compute_loss(
        self,
        action: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diffusion training loss.
        
        Args:
            action: Ground truth actions from human traders, shape (batch, action_dim)
            state: Market states, shape (batch, state_dim)
        
        Returns:
            Loss value
        """
        batch_size = action.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)
        
        # Add noise to actions
        noise = torch.randn_like(action)
        noisy_action, _ = self.add_noise(action, timesteps, noise)
        
        # Predict noise
        predicted_noise = self.network(noisy_action, timesteps, state)
        
        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'betas': self.betas,
            'alphas_cumprod': self.alphas_cumprod,
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.betas = checkpoint['betas']
        self.alphas_cumprod = checkpoint['alphas_cumprod']


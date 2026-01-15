"""
Diffusion-based World Model for Trajectory Prediction

This module implements a diffusion model that captures distributions over possible
future trajectories conditioned on current world state, enabling risk-aware planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WorldState:
    """Represents the current state of the world"""
    features: torch.Tensor  # [batch, state_dim]
    context: Optional[Dict[str, torch.Tensor]] = None
    timestamp: float = 0.0


@dataclass
class TrajectoryDistribution:
    """Distribution over possible future trajectories"""
    samples: torch.Tensor  # [batch, num_samples, horizon, state_dim]
    log_probs: torch.Tensor  # [batch, num_samples]
    uncertainty: torch.Tensor  # [batch, horizon, state_dim]


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class TemporalAttention(nn.Module):
    """Multi-head attention over temporal dimension"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x


class DiffusionWorldModel(nn.Module):
    """
    Diffusion-based world model that predicts distributions over future trajectories.
    
    Uses a denoising diffusion process to model p(trajectory | world_state, action).
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        horizon: int = 16,
        num_diffusion_steps: int = 100,
        beta_schedule: str = "cosine"
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.num_diffusion_steps = num_diffusion_steps
        
        # Diffusion schedule
        self.betas = self._get_beta_schedule(beta_schedule, num_diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Posterior variance for sampling
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Embeddings
        self.time_embed = SinusoidalPositionEmbedding(hidden_dim)
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.trajectory_embed = nn.Linear(state_dim, hidden_dim)
        
        # Temporal transformer blocks
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': TemporalAttention(hidden_dim, num_heads),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim)
            })
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, state_dim)
        
    def _get_beta_schedule(self, schedule: str, num_steps: int) -> torch.Tensor:
        """Get noise schedule for diffusion process"""
        if schedule == "linear":
            return torch.linspace(1e-4, 0.02, num_steps)
        elif schedule == "cosine":
            steps = torch.arange(num_steps + 1, dtype=torch.float32) / num_steps
            alphas_cumprod = torch.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        timestep: torch.Tensor,
        world_state: WorldState,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise in the trajectory at given diffusion timestep.
        
        Args:
            noisy_trajectory: [batch, horizon, state_dim]
            timestep: [batch]
            world_state: Current world state
            action: [batch, action_dim] or [batch, horizon, action_dim]
            
        Returns:
            Predicted noise: [batch, horizon, state_dim]
        """
        batch_size = noisy_trajectory.shape[0]
        
        # Embed timestep
        t_emb = self.time_embed(timestep)  # [batch, hidden_dim]
        
        # Embed world state
        state_emb = self.state_embed(world_state.features)  # [batch, hidden_dim]
        
        # Embed action
        if action.dim() == 2:
            action = action.unsqueeze(1).expand(-1, self.horizon, -1)
        action_emb = self.action_embed(action)  # [batch, horizon, hidden_dim]
        
        # Embed noisy trajectory
        traj_emb = self.trajectory_embed(noisy_trajectory)  # [batch, horizon, hidden_dim]
        
        # Combine embeddings
        x = traj_emb + action_emb
        x = x + t_emb.unsqueeze(1) + state_emb.unsqueeze(1)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = x + block['attn'](block['norm1'](x))
            x = x + block['mlp'](block['norm2'](x))
        
        # Project to state space
        noise_pred = self.output_proj(x)
        
        return noise_pred
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion process: add noise to trajectory"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.alphas_cumprod.sqrt(), t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            (1.0 - self.alphas_cumprod).sqrt(), t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract coefficients at specified timesteps"""
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        world_state: WorldState,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Single reverse diffusion step"""
        # Predict noise
        noise_pred = self.forward(x, t, world_state, action)
        
        # Compute coefficients
        sqrt_recip_alphas_t = self._extract(1.0 / self.alphas.sqrt(), t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            (1.0 - self.alphas_cumprod).sqrt(), t, x.shape
        )
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (
            x - self.betas.to(x.device).gather(0, t).reshape(-1, 1, 1) * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + posterior_variance_t.sqrt() * noise
    
    @torch.no_grad()
    def sample_trajectories(
        self,
        world_state: WorldState,
        action: torch.Tensor,
        num_samples: int = 32,
        guidance_scale: float = 1.0
    ) -> TrajectoryDistribution:
        """
        Sample multiple plausible future trajectories from the learned distribution.
        
        Args:
            world_state: Current world state
            action: Planned action sequence
            num_samples: Number of trajectory samples to generate
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Distribution over trajectories with uncertainty estimates
        """
        batch_size = world_state.features.shape[0]
        device = world_state.features.device
        
        # Expand world state and action for multiple samples
        expanded_state = WorldState(
            features=world_state.features.repeat_interleave(num_samples, dim=0),
            context=world_state.context,
            timestamp=world_state.timestamp
        )
        expanded_action = action.repeat_interleave(num_samples, dim=0)
        
        # Start from pure noise
        x = torch.randn(
            batch_size * num_samples,
            self.horizon,
            self.state_dim,
            device=device
        )
        
        # Reverse diffusion process
        for t in reversed(range(self.num_diffusion_steps)):
            t_batch = torch.full((batch_size * num_samples,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, expanded_state, expanded_action)
        
        # Reshape to [batch, num_samples, horizon, state_dim]
        trajectories = x.reshape(batch_size, num_samples, self.horizon, self.state_dim)
        
        # Compute uncertainty as variance across samples
        uncertainty = trajectories.var(dim=1)
        
        # Compute log probabilities (simplified)
        log_probs = torch.zeros(batch_size, num_samples, device=device)
        
        return TrajectoryDistribution(
            samples=trajectories,
            log_probs=log_probs,
            uncertainty=uncertainty
        )
    
    def compute_loss(
        self,
        true_trajectory: torch.Tensor,
        world_state: WorldState,
        action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion training loss.
        
        Args:
            true_trajectory: Ground truth trajectory [batch, horizon, state_dim]
            world_state: Current world state
            action: Action sequence
            
        Returns:
            Dictionary of losses
        """
        batch_size = true_trajectory.shape[0]
        device = true_trajectory.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(true_trajectory)
        
        # Add noise to trajectory
        noisy_trajectory = self.q_sample(true_trajectory, t, noise)
        
        # Predict noise
        noise_pred = self.forward(noisy_trajectory, t, world_state, action)
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        return {
            'total_loss': loss,
            'diffusion_loss': loss
        }


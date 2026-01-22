"""
Policy Network for Trading Actions

Defines the action space and policy network for trading decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class TradingAction:
    """
    Defines the trading action space.
    
    Action components:
    - position: Target position size [-1, 1] (short to long)
    - urgency: Execution urgency [0, 1] (passive to aggressive)
    - stop_loss: Stop loss distance [0, 1] (normalized)
    - take_profit: Take profit distance [0, 1] (normalized)
    """
    
    def __init__(self):
        self.action_dim = 4
        self.action_names = ['position', 'urgency', 'stop_loss', 'take_profit']
        
        # Action bounds
        self.action_low = np.array([-1.0, 0.0, 0.0, 0.0])
        self.action_high = np.array([1.0, 1.0, 1.0, 1.0])
    
    def normalize(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [-1, 1] range."""
        return 2 * (action - self.action_low) / (self.action_high - self.action_low + 1e-8) - 1
    
    def denormalize(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action from [-1, 1] to original range."""
        return (action + 1) / 2 * (self.action_high - self.action_low) + self.action_low
    
    def clip(self, action: np.ndarray) -> np.ndarray:
        """Clip action to valid range."""
        return np.clip(action, self.action_low, self.action_high)


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs trading actions given market state.
    Can be used standalone or as part of the diffusion model.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build MLP layers
        layers = []
        in_dim = state_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Action heads
        self.position_head = nn.Linear(hidden_dim, 1)
        self.urgency_head = nn.Linear(hidden_dim, 1)
        self.stop_loss_head = nn.Linear(hidden_dim, 1)
        self.take_profit_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Market state embedding [batch, state_dim]
        
        Returns:
            Actions [batch, action_dim]
        """
        features = self.backbone(state)
        
        # Output each action component
        position = torch.tanh(self.position_head(features))  # [-1, 1]
        urgency = torch.sigmoid(self.urgency_head(features))  # [0, 1]
        stop_loss = torch.sigmoid(self.stop_loss_head(features))  # [0, 1]
        take_profit = torch.sigmoid(self.take_profit_head(features))  # [0, 1]
        
        actions = torch.cat([position, urgency, stop_loss, take_profit], dim=-1)
        
        return actions


class StochasticPolicyNetwork(nn.Module):
    """
    Stochastic policy network that outputs action distribution.
    Useful for exploration and uncertainty estimation.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        min_std: float = 0.01,
        max_std: float = 1.0
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std
        
        # Shared backbone
        layers = []
        in_dim = state_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Mean and std heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: Market state embedding [batch, state_dim]
        
        Returns:
            mean: Action mean [batch, action_dim]
            std: Action std [batch, action_dim]
        """
        features = self.backbone(state)
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp std to reasonable range
        log_std = torch.clamp(log_std, np.log(self.min_std), np.log(self.max_std))
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            state: Market state embedding
            deterministic: If True, return mean action
        
        Returns:
            actions: Sampled actions
            log_probs: Log probabilities of actions
        """
        mean, std = self.forward(state)
        
        if deterministic:
            actions = mean
            log_probs = None
        else:
            # Sample from Gaussian
            dist = torch.distributions.Normal(mean, std)
            actions = dist.rsample()
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # Apply tanh squashing for bounded actions
        actions_squashed = torch.tanh(actions)
        
        if log_probs is not None:
            # Adjust log probs for tanh squashing
            log_probs = log_probs - torch.log(1 - actions_squashed.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        return actions_squashed, log_probs


class ValueNetwork(nn.Module):
    """
    Value network for estimating expected returns.
    Used in actor-critic methods and for strategy evaluation.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        in_dim = state_dim + action_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Market state [batch, state_dim]
            action: Trading action [batch, action_dim]
        
        Returns:
            Value estimate [batch, 1]
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class EnsemblePolicyNetwork(nn.Module):
    """
    Ensemble of policy networks for uncertainty estimation and robustness.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_ensemble: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_ensemble = num_ensemble
        
        # Create ensemble of policies
        self.policies = nn.ModuleList([
            PolicyNetwork(state_dim, action_dim, hidden_dim, num_layers, dropout)
            for _ in range(num_ensemble)
        ])
        
    def forward(self, state: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """
        Args:
            state: Market state [batch, state_dim]
            return_all: If True, return all ensemble predictions
        
        Returns:
            If return_all=False: Mean action [batch, action_dim]
            If return_all=True: All actions [batch, num_ensemble, action_dim]
        """
        actions = torch.stack([policy(state) for policy in self.policies], dim=1)
        
        if return_all:
            return actions
        else:
            return actions.mean(dim=1)
    
    def get_uncertainty(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate uncertainty as ensemble disagreement.
        
        Args:
            state: Market state [batch, state_dim]
        
        Returns:
            Uncertainty (std) [batch, action_dim]
        """
        actions = self.forward(state, return_all=True)
        return actions.std(dim=1)


class SequentialPolicyNetwork(nn.Module):
    """
    Sequential policy network that generates action sequences.
    Uses LSTM/GRU to maintain temporal consistency.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_lstm: bool = True
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_lstm = use_lstm
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Recurrent layer
        if use_lstm:
            self.rnn = nn.LSTM(
                hidden_dim + action_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            self.rnn = nn.GRU(
                hidden_dim + action_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor, seq_length: int = 10,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Generate action sequence autoregressively.
        
        Args:
            state: Market state [batch, state_dim]
            seq_length: Length of action sequence to generate
            hidden: Initial hidden state
        
        Returns:
            actions: Action sequence [batch, seq_length, action_dim]
            hidden: Final hidden state
        """
        batch_size = state.shape[0]
        
        # Encode state
        state_emb = self.state_encoder(state)
        
        # Initialize first action as zeros
        action = torch.zeros(batch_size, self.action_dim, device=state.device)
        
        actions = []
        
        for t in range(seq_length):
            # Concatenate state and previous action
            rnn_input = torch.cat([state_emb, action], dim=-1).unsqueeze(1)
            
            # RNN step
            if hidden is None:
                rnn_out, hidden = self.rnn(rnn_input)
            else:
                rnn_out, hidden = self.rnn(rnn_input, hidden)
            
            # Decode action
            action = self.action_decoder(rnn_out.squeeze(1))
            actions.append(action)
        
        actions = torch.stack(actions, dim=1)
        
        return actions, hidden
    
    def init_hidden(self, batch_size: int, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        
        if self.use_lstm:
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h, c)
        else:
            return h


class HierarchicalPolicyNetwork(nn.Module):
    """
    Hierarchical policy with high-level strategy and low-level execution.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 4,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # High-level policy (strategy selector)
        self.high_level_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Low-level policy (action executor)
        self.low_level_policy = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: Market state [batch, state_dim]
        
        Returns:
            actions: Trading actions [batch, action_dim]
            latent: High-level strategy [batch, latent_dim]
        """
        # High-level strategy
        latent = self.high_level_policy(state)
        
        # Low-level execution
        combined = torch.cat([state, latent], dim=-1)
        actions = self.low_level_policy(combined)
        
        return actions, latent


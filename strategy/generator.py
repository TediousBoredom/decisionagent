"""
Strategy Generator using Diffusion Model

Generates trading strategies conditioned on market state and risk constraints.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import numpy as np

from models.diffusion import TradingDiffusionModel
from models.encoder import MarketStateEncoder
from models.policy_net import PolicyNetwork
from risk.constraints import RiskConstraints, RiskAwareActionFilter


class DiffusionStrategyGenerator(nn.Module):
    """
    Complete strategy generation system using diffusion models.
    """
    
    def __init__(
        self,
        # Market encoder params
        price_dim: int = 5,
        indicator_dim: int = 20,
        orderbook_dim: int = 10,
        regime_dim: int = 5,
        seq_length: int = 100,
        
        # Diffusion model params
        action_dim: int = 4,
        action_seq_length: int = 10,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        num_timesteps: int = 1000,
        
        # Risk params
        constraints: Optional[RiskConstraints] = None
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_seq_length = action_seq_length
        self.cond_dim = cond_dim
        
        # Market state encoder
        self.state_encoder = MarketStateEncoder(
            price_dim=price_dim,
            indicator_dim=indicator_dim,
            orderbook_dim=orderbook_dim,
            regime_dim=regime_dim,
            seq_length=seq_length,
            output_dim=cond_dim
        )
        
        # Diffusion model for strategy generation
        self.diffusion_model = TradingDiffusionModel(
            action_dim=action_dim,
            seq_length=action_seq_length,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            num_timesteps=num_timesteps
        )
        
        # Risk-aware action filter
        self.constraints = constraints or RiskConstraints()
        self.action_filter = RiskAwareActionFilter(
            constraints=self.constraints,
            state_dim=cond_dim,
            action_dim=action_dim
        )
        
        # Auxiliary networks
        self.return_predictor = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_seq_length)
        )
        
        self.volatility_predictor = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_seq_length)
        )
        
    def encode_market_state(
        self,
        price: torch.Tensor,
        indicators: torch.Tensor,
        orderbook: torch.Tensor,
        regime: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode market state into conditioning vector.
        
        Args:
            price: [batch, seq_length, price_dim]
            indicators: [batch, seq_length, indicator_dim]
            orderbook: [batch, orderbook_dim]
            regime: [batch, regime_dim]
        
        Returns:
            state_embedding: [batch, cond_dim]
        """
        return self.state_encoder(price, indicators, orderbook, regime)
    
    def generate_strategy(
        self,
        market_state: torch.Tensor,
        num_samples: int = 1,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate trading strategy using diffusion model.
        
        Args:
            market_state: Market state embedding [batch, cond_dim]
            num_samples: Number of strategy samples to generate
            use_ddim: Use DDIM for faster sampling
            ddim_steps: Number of DDIM steps
            temperature: Sampling temperature
        
        Returns:
            actions: Generated action sequences [batch * num_samples, action_seq_length, action_dim]
        """
        batch_size = market_state.shape[0]
        device = market_state.device
        
        # Repeat market state for multiple samples
        if num_samples > 1:
            market_state = market_state.repeat_interleave(num_samples, dim=0)
        
        # Generate actions using diffusion model
        if use_ddim:
            actions = self.diffusion_model.ddim_sample(
                batch_size=batch_size * num_samples,
                cond=market_state,
                num_steps=ddim_steps,
                device=device
            )
        else:
            actions = self.diffusion_model.sample(
                batch_size=batch_size * num_samples,
                cond=market_state,
                device=device
            )
        
        # Apply temperature scaling
        if temperature != 1.0:
            actions = actions * temperature
        
        # Clip actions to valid range
        actions = torch.tanh(actions)  # Ensure bounded
        
        return actions
    
    def filter_actions(
        self,
        market_state: torch.Tensor,
        actions: torch.Tensor,
        current_position: torch.Tensor,
        portfolio_value: torch.Tensor,
        current_drawdown: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply risk filters to generated actions.
        
        Args:
            market_state: [batch, cond_dim]
            actions: [batch, action_seq_length, action_dim]
            current_position: [batch, 1]
            portfolio_value: [batch, 1]
            current_drawdown: [batch, 1]
        
        Returns:
            filtered_actions: [batch, action_seq_length, action_dim]
            risk_info: Dictionary of risk information
        """
        batch_size, seq_length, action_dim = actions.shape
        
        # Filter each action in sequence
        filtered_actions = []
        risk_infos = []
        
        for t in range(seq_length):
            action_t = actions[:, t, :]
            
            filtered_action_t, risk_info_t = self.action_filter(
                market_state,
                action_t,
                current_position,
                portfolio_value,
                current_drawdown
            )
            
            filtered_actions.append(filtered_action_t)
            risk_infos.append(risk_info_t)
        
        filtered_actions = torch.stack(filtered_actions, dim=1)
        
        # Aggregate risk info
        aggregated_risk_info = {
            'mean_risk_score': np.mean([r['risk_score'] for r in risk_infos]),
            'mean_adjustment': np.mean([r['adjustment_magnitude'] for r in risk_infos])
        }
        
        return filtered_actions, aggregated_risk_info
    
    def predict_returns_and_volatility(
        self,
        market_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict expected returns and volatility.
        
        Args:
            market_state: [batch, cond_dim]
        
        Returns:
            predicted_returns: [batch, action_seq_length]
            predicted_volatility: [batch, action_seq_length]
        """
        returns = self.return_predictor(market_state)
        volatility = torch.nn.functional.softplus(self.volatility_predictor(market_state))
        
        return returns, volatility
    
    def forward(
        self,
        price: torch.Tensor,
        indicators: torch.Tensor,
        orderbook: torch.Tensor,
        regime: torch.Tensor,
        current_position: torch.Tensor,
        portfolio_value: torch.Tensor,
        current_drawdown: torch.Tensor,
        num_samples: int = 1,
        return_best: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass: encode state, generate strategies, filter actions.
        
        Args:
            price: [batch, seq_length, price_dim]
            indicators: [batch, seq_length, indicator_dim]
            orderbook: [batch, orderbook_dim]
            regime: [batch, regime_dim]
            current_position: [batch, 1]
            portfolio_value: [batch, 1]
            current_drawdown: [batch, 1]
            num_samples: Number of strategy samples
            return_best: If True, return best strategy based on predicted return
        
        Returns:
            Dictionary containing actions and predictions
        """
        # Encode market state
        market_state = self.encode_market_state(price, indicators, orderbook, regime)
        
        # Predict returns and volatility
        predicted_returns, predicted_volatility = self.predict_returns_and_volatility(market_state)
        
        # Generate strategies
        actions = self.generate_strategy(
            market_state,
            num_samples=num_samples,
            use_ddim=True,
            ddim_steps=50
        )
        
        # Reshape for filtering
        batch_size = market_state.shape[0]
        if num_samples > 1:
            actions = actions.view(batch_size, num_samples, self.action_seq_length, self.action_dim)
            
            if return_best:
                # Select best strategy based on predicted return
                # Expand market state for filtering
                market_state_expanded = market_state.unsqueeze(1).expand(-1, num_samples, -1)
                market_state_expanded = market_state_expanded.reshape(batch_size * num_samples, -1)
                
                actions_flat = actions.reshape(batch_size * num_samples, self.action_seq_length, self.action_dim)
                
                # Filter all samples
                current_position_expanded = current_position.repeat_interleave(num_samples, dim=0)
                portfolio_value_expanded = portfolio_value.repeat_interleave(num_samples, dim=0)
                current_drawdown_expanded = current_drawdown.repeat_interleave(num_samples, dim=0)
                
                filtered_actions, risk_info = self.filter_actions(
                    market_state_expanded,
                    actions_flat,
                    current_position_expanded,
                    portfolio_value_expanded,
                    current_drawdown_expanded
                )
                
                # Reshape back
                filtered_actions = filtered_actions.view(batch_size, num_samples, self.action_seq_length, self.action_dim)
                
                # Score each sample (use first action's position as proxy)
                scores = filtered_actions[:, :, 0, 0] * predicted_returns[:, 0:1]
                best_idx = scores.argmax(dim=1)
                
                # Select best
                actions = filtered_actions[torch.arange(batch_size), best_idx]
            else:
                # Average all samples
                actions = actions.mean(dim=1)
                
                # Filter averaged actions
                actions, risk_info = self.filter_actions(
                    market_state,
                    actions,
                    current_position,
                    portfolio_value,
                    current_drawdown
                )
        else:
            # Filter single sample
            actions, risk_info = self.filter_actions(
                market_state,
                actions,
                current_position,
                portfolio_value,
                current_drawdown
            )
        
        return {
            'actions': actions,
            'market_state': market_state,
            'predicted_returns': predicted_returns,
            'predicted_volatility': predicted_volatility,
            'risk_info': risk_info
        }
    
    def train_step(
        self,
        price: torch.Tensor,
        indicators: torch.Tensor,
        orderbook: torch.Tensor,
        regime: torch.Tensor,
        expert_actions: torch.Tensor,
        returns: torch.Tensor,
        volatility: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for the model.
        
        Args:
            price: [batch, seq_length, price_dim]
            indicators: [batch, seq_length, indicator_dim]
            orderbook: [batch, orderbook_dim]
            regime: [batch, regime_dim]
            expert_actions: [batch, action_seq_length, action_dim] - High-return strategy actions
            returns: [batch, action_seq_length] - Actual returns
            volatility: [batch, action_seq_length] - Actual volatility
        
        Returns:
            Dictionary of losses
        """
        # Encode market state
        market_state = self.encode_market_state(price, indicators, orderbook, regime)
        
        # Diffusion loss (learn to generate expert actions)
        diffusion_loss = self.diffusion_model(expert_actions, market_state)
        
        # Return prediction loss
        predicted_returns = self.return_predictor(market_state)
        return_loss = torch.nn.functional.mse_loss(predicted_returns, returns)
        
        # Volatility prediction loss
        predicted_volatility = torch.nn.functional.softplus(self.volatility_predictor(market_state))
        volatility_loss = torch.nn.functional.mse_loss(predicted_volatility, volatility)
        
        # Total loss
        total_loss = diffusion_loss + 0.1 * return_loss + 0.1 * volatility_loss
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'return_loss': return_loss,
            'volatility_loss': volatility_loss
        }


class StrategyEnsemble(nn.Module):
    """
    Ensemble of strategy generators for robustness.
    """
    
    def __init__(
        self,
        num_models: int = 3,
        **generator_kwargs
    ):
        super().__init__()
        
        self.num_models = num_models
        
        # Create ensemble of generators
        self.generators = nn.ModuleList([
            DiffusionStrategyGenerator(**generator_kwargs)
            for _ in range(num_models)
        ])
        
    def forward(
        self,
        price: torch.Tensor,
        indicators: torch.Tensor,
        orderbook: torch.Tensor,
        regime: torch.Tensor,
        current_position: torch.Tensor,
        portfolio_value: torch.Tensor,
        current_drawdown: torch.Tensor,
        aggregation: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Generate strategies using ensemble.
        
        Args:
            aggregation: 'mean', 'median', or 'vote'
        
        Returns:
            Aggregated strategy
        """
        outputs = []
        
        for generator in self.generators:
            output = generator(
                price, indicators, orderbook, regime,
                current_position, portfolio_value, current_drawdown,
                num_samples=1, return_best=True
            )
            outputs.append(output)
        
        # Aggregate actions
        actions = torch.stack([o['actions'] for o in outputs], dim=0)
        
        if aggregation == 'mean':
            aggregated_actions = actions.mean(dim=0)
        elif aggregation == 'median':
            aggregated_actions = actions.median(dim=0)[0]
        elif aggregation == 'vote':
            # Majority vote on position direction
            position_signs = torch.sign(actions[:, :, :, 0])
            aggregated_position = torch.mode(position_signs, dim=0)[0]
            aggregated_actions = actions.mean(dim=0)
            aggregated_actions[:, :, 0] = aggregated_position
        else:
            aggregated_actions = actions.mean(dim=0)
        
        # Aggregate predictions
        predicted_returns = torch.stack([o['predicted_returns'] for o in outputs], dim=0).mean(dim=0)
        predicted_volatility = torch.stack([o['predicted_volatility'] for o in outputs], dim=0).mean(dim=0)
        
        return {
            'actions': aggregated_actions,
            'predicted_returns': predicted_returns,
            'predicted_volatility': predicted_volatility,
            'ensemble_std': actions.std(dim=0)  # Uncertainty estimate
        }


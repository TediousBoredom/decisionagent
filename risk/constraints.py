"""
Risk Management and Constraints

Implements risk control mechanisms and strategy constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class RiskConstraints:
    """
    Defines risk constraints for trading strategies.
    """
    
    def __init__(
        self,
        max_position_size: float = 1.0,
        max_leverage: float = 3.0,
        max_drawdown: float = 0.2,
        max_daily_loss: float = 0.05,
        max_concentration: float = 0.3,
        min_sharpe_ratio: float = 1.0,
        max_var_95: float = 0.03,
        max_turnover: float = 5.0
    ):
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.max_concentration = max_concentration
        self.min_sharpe_ratio = min_sharpe_ratio
        self.max_var_95 = max_var_95
        self.max_turnover = max_turnover
    
    def to_dict(self) -> Dict:
        return {
            'max_position_size': self.max_position_size,
            'max_leverage': self.max_leverage,
            'max_drawdown': self.max_drawdown,
            'max_daily_loss': self.max_daily_loss,
            'max_concentration': self.max_concentration,
            'min_sharpe_ratio': self.min_sharpe_ratio,
            'max_var_95': self.max_var_95,
            'max_turnover': self.max_turnover
        }


class PositionSizer(nn.Module):
    """
    Dynamic position sizing based on market conditions and risk.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        hidden_dim: int = 128,
        base_position_size: float = 1.0,
        use_kelly: bool = True
    ):
        super().__init__()
        
        self.base_position_size = base_position_size
        self.use_kelly = use_kelly
        
        # Neural network for position sizing
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(self, state: torch.Tensor, predicted_return: torch.Tensor,
                predicted_volatility: torch.Tensor, win_rate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate optimal position size.
        
        Args:
            state: Market state [batch, state_dim]
            predicted_return: Expected return [batch, 1]
            predicted_volatility: Expected volatility [batch, 1]
            win_rate: Historical win rate [batch, 1]
        
        Returns:
            Position size multiplier [batch, 1]
        """
        # Neural network component
        nn_multiplier = self.network(state)
        
        if self.use_kelly and win_rate is not None:
            # Kelly criterion: f = (p * b - q) / b
            # where p = win rate, q = 1 - p, b = win/loss ratio
            win_loss_ratio = torch.abs(predicted_return) / (predicted_volatility + 1e-8)
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / (win_loss_ratio + 1e-8)
            kelly_fraction = torch.clamp(kelly_fraction, 0, 1)
            
            # Combine with neural network
            position_size = 0.5 * nn_multiplier + 0.5 * kelly_fraction
        else:
            # Inverse volatility scaling
            vol_scaling = 1.0 / (1.0 + predicted_volatility)
            position_size = nn_multiplier * vol_scaling
        
        return position_size * self.base_position_size


class DrawdownController(nn.Module):
    """
    Controls trading based on current drawdown level.
    """
    
    def __init__(self, max_drawdown: float = 0.2, recovery_threshold: float = 0.1):
        super().__init__()
        
        self.max_drawdown = max_drawdown
        self.recovery_threshold = recovery_threshold
        
    def forward(self, current_drawdown: torch.Tensor) -> torch.Tensor:
        """
        Calculate position scaling factor based on drawdown.
        
        Args:
            current_drawdown: Current drawdown [batch, 1]
        
        Returns:
            Scaling factor [batch, 1] in [0, 1]
        """
        # Linear reduction as drawdown approaches max
        scaling = 1.0 - (current_drawdown / self.max_drawdown)
        scaling = torch.clamp(scaling, 0, 1)
        
        # Smooth transition
        scaling = torch.sigmoid(10 * (scaling - 0.5))
        
        return scaling


class VolatilityScaler(nn.Module):
    """
    Scales positions based on market volatility.
    """
    
    def __init__(self, target_volatility: float = 0.15, lookback: int = 20):
        super().__init__()
        
        self.target_volatility = target_volatility
        self.lookback = lookback
        
    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Calculate volatility scaling factor.
        
        Args:
            returns: Historical returns [batch, lookback]
        
        Returns:
            Scaling factor [batch, 1]
        """
        # Calculate realized volatility
        realized_vol = returns.std(dim=1, keepdim=True)
        
        # Scale to target volatility
        scaling = self.target_volatility / (realized_vol + 1e-8)
        
        # Clamp to reasonable range
        scaling = torch.clamp(scaling, 0.1, 3.0)
        
        return scaling


class RiskMetrics(nn.Module):
    """
    Calculates various risk metrics for strategy evaluation.
    """
    
    def __init__(self):
        super().__init__()
        
    def sharpe_ratio(self, returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate
        return excess_returns.mean(dim=1) / (returns.std(dim=1) + 1e-8)
    
    def sortino_ratio(self, returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
        """Calculate Sortino ratio (downside deviation)."""
        excess_returns = returns - risk_free_rate
        downside_returns = torch.where(returns < 0, returns, torch.zeros_like(returns))
        downside_std = downside_returns.std(dim=1)
        return excess_returns.mean(dim=1) / (downside_std + 1e-8)
    
    def max_drawdown(self, returns: torch.Tensor) -> torch.Tensor:
        """Calculate maximum drawdown."""
        cumulative = torch.cumsum(returns, dim=1)
        running_max = torch.cummax(cumulative, dim=1)[0]
        drawdown = cumulative - running_max
        return drawdown.min(dim=1)[0]
    
    def calmar_ratio(self, returns: torch.Tensor) -> torch.Tensor:
        """Calculate Calmar ratio (return / max drawdown)."""
        total_return = returns.sum(dim=1)
        max_dd = torch.abs(self.max_drawdown(returns))
        return total_return / (max_dd + 1e-8)
    
    def value_at_risk(self, returns: torch.Tensor, confidence: float = 0.95) -> torch.Tensor:
        """Calculate Value at Risk (VaR)."""
        sorted_returns, _ = torch.sort(returns, dim=1)
        index = int((1 - confidence) * returns.shape[1])
        return sorted_returns[:, index]
    
    def conditional_var(self, returns: torch.Tensor, confidence: float = 0.95) -> torch.Tensor:
        """Calculate Conditional Value at Risk (CVaR / Expected Shortfall)."""
        var = self.value_at_risk(returns, confidence)
        # Average of returns below VaR
        mask = returns <= var.unsqueeze(1)
        cvar = (returns * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return cvar
    
    def omega_ratio(self, returns: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """Calculate Omega ratio."""
        gains = torch.where(returns > threshold, returns - threshold, torch.zeros_like(returns))
        losses = torch.where(returns < threshold, threshold - returns, torch.zeros_like(returns))
        return gains.sum(dim=1) / (losses.sum(dim=1) + 1e-8)
    
    def information_ratio(self, returns: torch.Tensor, benchmark_returns: torch.Tensor) -> torch.Tensor:
        """Calculate Information ratio."""
        active_returns = returns - benchmark_returns
        return active_returns.mean(dim=1) / (active_returns.std(dim=1) + 1e-8)


class RiskAwareActionFilter(nn.Module):
    """
    Filters and adjusts actions based on risk constraints.
    """
    
    def __init__(
        self,
        constraints: RiskConstraints,
        state_dim: int = 128,
        action_dim: int = 4,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.constraints = constraints
        
        # Risk assessment network
        self.risk_assessor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Risk score [0, 1]
        )
        
        # Action adjustment network
        self.action_adjuster = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim),  # +1 for risk score
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor,
                current_position: torch.Tensor, portfolio_value: torch.Tensor,
                current_drawdown: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Filter and adjust actions based on risk constraints.
        
        Args:
            state: Market state [batch, state_dim]
            action: Proposed action [batch, action_dim]
            current_position: Current position [batch, 1]
            portfolio_value: Current portfolio value [batch, 1]
            current_drawdown: Current drawdown [batch, 1]
        
        Returns:
            adjusted_action: Risk-adjusted action [batch, action_dim]
            risk_info: Dictionary of risk information
        """
        batch_size = state.shape[0]
        
        # Assess risk of proposed action
        risk_score = self.risk_assessor(torch.cat([state, action], dim=-1))
        
        # Check hard constraints
        violations = {}
        
        # Position size constraint
        target_position = action[:, 0:1]  # First component is position
        position_violation = torch.abs(target_position) > self.constraints.max_position_size
        violations['position_size'] = position_violation.float().mean()
        
        # Drawdown constraint
        drawdown_violation = current_drawdown > self.constraints.max_drawdown
        violations['drawdown'] = drawdown_violation.float().mean()
        
        # Adjust action if constraints violated
        if position_violation.any() or drawdown_violation.any():
            # Use neural network to adjust
            adjustment_input = torch.cat([state, action, risk_score], dim=-1)
            adjustment = self.action_adjuster(adjustment_input)
            adjusted_action = action + 0.1 * adjustment  # Small adjustment
            
            # Hard clip position size
            adjusted_action[:, 0] = torch.clamp(
                adjusted_action[:, 0],
                -self.constraints.max_position_size,
                self.constraints.max_position_size
            )
            
            # Reduce position if in drawdown
            if drawdown_violation.any():
                drawdown_scale = 1.0 - (current_drawdown / self.constraints.max_drawdown)
                drawdown_scale = torch.clamp(drawdown_scale, 0, 1)
                adjusted_action[:, 0] = adjusted_action[:, 0] * drawdown_scale
        else:
            adjusted_action = action
        
        risk_info = {
            'risk_score': risk_score.mean().item(),
            'violations': violations,
            'adjustment_magnitude': (adjusted_action - action).abs().mean().item()
        }
        
        return adjusted_action, risk_info


class AdaptiveRiskManager(nn.Module):
    """
    Adaptive risk management that learns from market conditions.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        hidden_dim: int = 128,
        num_risk_regimes: int = 3
    ):
        super().__init__()
        
        self.num_risk_regimes = num_risk_regimes
        
        # Regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_risk_regimes)
        )
        
        # Risk parameters for each regime
        self.regime_risk_params = nn.Parameter(
            torch.randn(num_risk_regimes, 4)  # [max_position, max_leverage, stop_loss, take_profit]
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determine risk regime and parameters.
        
        Args:
            state: Market state [batch, state_dim]
        
        Returns:
            regime_probs: Regime probabilities [batch, num_risk_regimes]
            risk_params: Risk parameters [batch, 4]
        """
        # Classify regime
        regime_logits = self.regime_classifier(state)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # Weighted combination of regime parameters
        risk_params = torch.matmul(regime_probs, self.regime_risk_params)
        
        # Apply activation functions
        risk_params = torch.sigmoid(risk_params)
        
        return regime_probs, risk_params


class PortfolioRiskMonitor:
    """
    Monitors portfolio-level risk metrics in real-time.
    """
    
    def __init__(
        self,
        constraints: RiskConstraints,
        lookback_window: int = 100
    ):
        self.constraints = constraints
        self.lookback_window = lookback_window
        
        # Historical data
        self.returns_history = []
        self.positions_history = []
        self.values_history = []
        
    def update(self, returns: float, position: float, portfolio_value: float):
        """Update with new data."""
        self.returns_history.append(returns)
        self.positions_history.append(position)
        self.values_history.append(portfolio_value)
        
        # Keep only recent history
        if len(self.returns_history) > self.lookback_window:
            self.returns_history.pop(0)
            self.positions_history.pop(0)
            self.values_history.pop(0)
    
    def get_current_metrics(self) -> Dict:
        """Calculate current risk metrics."""
        if len(self.returns_history) < 2:
            return {}
        
        returns = np.array(self.returns_history)
        positions = np.array(self.positions_history)
        values = np.array(self.values_history)
        
        # Calculate metrics
        metrics = {}
        
        # Sharpe ratio
        if len(returns) > 0:
            metrics['sharpe_ratio'] = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        
        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        metrics['max_drawdown'] = drawdown.min()
        metrics['current_drawdown'] = drawdown[-1]
        
        # Volatility
        metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # VaR
        metrics['var_95'] = np.percentile(returns, 5)
        
        # Current position
        metrics['current_position'] = positions[-1] if len(positions) > 0 else 0
        
        # Turnover
        if len(positions) > 1:
            position_changes = np.abs(np.diff(positions))
            metrics['turnover'] = position_changes.sum()
        
        return metrics
    
    def check_constraints(self) -> Tuple[bool, List[str]]:
        """Check if constraints are violated."""
        metrics = self.get_current_metrics()
        violations = []
        
        if 'max_drawdown' in metrics:
            if abs(metrics['max_drawdown']) > self.constraints.max_drawdown:
                violations.append(f"Max drawdown exceeded: {metrics['max_drawdown']:.2%}")
        
        if 'sharpe_ratio' in metrics:
            if metrics['sharpe_ratio'] < self.constraints.min_sharpe_ratio:
                violations.append(f"Sharpe ratio below minimum: {metrics['sharpe_ratio']:.2f}")
        
        if 'var_95' in metrics:
            if abs(metrics['var_95']) > self.constraints.max_var_95:
                violations.append(f"VaR exceeded: {metrics['var_95']:.2%}")
        
        if 'current_position' in metrics:
            if abs(metrics['current_position']) > self.constraints.max_position_size:
                violations.append(f"Position size exceeded: {metrics['current_position']:.2f}")
        
        if 'turnover' in metrics:
            if metrics['turnover'] > self.constraints.max_turnover:
                violations.append(f"Turnover exceeded: {metrics['turnover']:.2f}")
        
        return len(violations) == 0, violations
    
    def should_halt_trading(self) -> bool:
        """Determine if trading should be halted."""
        metrics = self.get_current_metrics()
        
        # Halt if severe drawdown
        if 'current_drawdown' in metrics:
            if abs(metrics['current_drawdown']) > self.constraints.max_drawdown * 0.9:
                return True
        
        # Halt if daily loss limit exceeded
        if len(self.returns_history) > 0:
            daily_return = sum(self.returns_history[-20:]) if len(self.returns_history) >= 20 else sum(self.returns_history)
            if daily_return < -self.constraints.max_daily_loss:
                return True
        
        return False


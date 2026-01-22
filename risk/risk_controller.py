"""
Risk Management Module

Implements position sizing, stop-loss, and portfolio risk controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    entry_price: float
    current_price: float
    position_size: float  # Positive for long, negative for short
    stop_loss: float
    take_profit: Optional[float] = None
    entry_time: datetime = None
    unrealized_pnl: float = 0.0
    
    def update_price(self, current_price: float):
        """Update current price and unrealized PnL."""
        self.current_price = current_price
        
        if self.position_size > 0:  # Long position
            self.unrealized_pnl = (current_price - self.entry_price) * abs(self.position_size)
        else:  # Short position
            self.unrealized_pnl = (self.entry_price - current_price) * abs(self.position_size)
    
    def should_stop_loss(self) -> bool:
        """Check if stop loss should be triggered."""
        if self.position_size > 0:  # Long
            return self.current_price <= self.stop_loss
        else:  # Short
            return self.current_price >= self.stop_loss
    
    def should_take_profit(self) -> bool:
        """Check if take profit should be triggered."""
        if self.take_profit is None:
            return False
        
        if self.position_size > 0:  # Long
            return self.current_price >= self.take_profit
        else:  # Short
            return self.current_price <= self.take_profit


class PositionManager:
    """Manages trading positions and sizing."""
    
    def __init__(
        self,
        initial_capital: float,
        max_position_size: float = 0.1,
        min_position_size: float = 0.01,
        max_positions: int = 5
    ):
        """
        Initialize position manager.
        
        Args:
            initial_capital: Initial trading capital
            max_position_size: Maximum position size as fraction of capital
            min_position_size: Minimum position size as fraction of capital
            max_positions: Maximum number of concurrent positions
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.max_positions = max_positions
        
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        logger.info(f"Initialized PositionManager with capital: ${initial_capital:,.2f}")
    
    def calculate_position_size(
        self,
        symbol: str,
        action_size: float,
        current_price: float,
        volatility: float,
        confidence: float = 1.0
    ) -> float:
        """
        Calculate optimal position size.
        
        Args:
            symbol: Trading symbol
            action_size: Desired position size from model [-1, 1]
            current_price: Current market price
            volatility: Market volatility
            confidence: Model confidence [0, 1]
        
        Returns:
            Position size in base currency
        """
        # Base position size from model
        base_size = abs(action_size) * self.max_position_size
        
        # Adjust for confidence
        adjusted_size = base_size * confidence
        
        # Adjust for volatility (reduce size in high volatility)
        volatility_factor = 1.0 / (1.0 + volatility * 10)
        adjusted_size *= volatility_factor
        
        # Ensure within bounds
        adjusted_size = np.clip(adjusted_size, self.min_position_size, self.max_position_size)
        
        # Calculate in base currency
        position_value = self.current_capital * adjusted_size
        position_size = position_value / current_price
        
        # Preserve sign (long/short)
        if action_size < 0:
            position_size = -position_size
        
        logger.debug(f"Calculated position size for {symbol}: {position_size:.4f} (${position_value:,.2f})")
        return position_size
    
    def can_open_position(self, symbol: str) -> bool:
        """Check if a new position can be opened."""
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return False
        
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Maximum positions reached: {self.max_positions}")
            return False
        
        return True
    
    def open_position(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
        stop_loss: float,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_size: Position size (positive for long, negative for short)
            stop_loss: Stop loss price
            take_profit: Take profit price (optional)
        
        Returns:
            True if position opened successfully
        """
        if not self.can_open_position(symbol):
            return False
        
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            current_price=entry_price,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now()
        )
        
        self.positions[symbol] = position
        
        direction = "LONG" if position_size > 0 else "SHORT"
        logger.info(f"Opened {direction} position: {symbol} @ ${entry_price:.2f}, size: {abs(position_size):.4f}, SL: ${stop_loss:.2f}")
        
        return True
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[float]:
        """
        Close a position.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
        
        Returns:
            Realized PnL
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return None
        
        position = self.positions[symbol]
        
        # Calculate realized PnL
        if position.position_size > 0:  # Long
            pnl = (exit_price - position.entry_price) * abs(position.position_size)
        else:  # Short
            pnl = (position.entry_price - exit_price) * abs(position.position_size)
        
        # Update capital
        self.current_capital += pnl
        
        # Move to closed positions
        position.update_price(exit_price)
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        logger.info(f"Closed position: {symbol} @ ${exit_price:.2f}, PnL: ${pnl:,.2f}")
        
        return pnl
    
    def update_positions(self, prices: Dict[str, float]) -> List[str]:
        """
        Update all positions with current prices.
        
        Args:
            prices: Dictionary of symbol -> current price
        
        Returns:
            List of symbols that should be closed (stop loss/take profit triggered)
        """
        to_close = []
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])
                
                if position.should_stop_loss():
                    logger.warning(f"Stop loss triggered for {symbol}")
                    to_close.append(symbol)
                elif position.should_take_profit():
                    logger.info(f"Take profit triggered for {symbol}")
                    to_close.append(symbol)
        
        return to_close
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total_value = self.current_capital
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])
                total_value += position.unrealized_pnl
        
        return total_value
    
    def get_position_summary(self) -> Dict:
        """Get summary of current positions."""
        return {
            'num_positions': len(self.positions),
            'capital': self.current_capital,
            'total_pnl': sum(p.unrealized_pnl for p in self.positions.values()),
            'positions': {
                symbol: {
                    'size': pos.position_size,
                    'entry': pos.entry_price,
                    'current': pos.current_price,
                    'pnl': pos.unrealized_pnl
                }
                for symbol, pos in self.positions.items()
            }
        }


class RiskController:
    """Controls portfolio-level risk."""
    
    def __init__(
        self,
        max_loss_per_trade: float = 0.02,
        daily_risk_budget: float = 0.05,
        weekly_risk_budget: float = 0.15,
        monthly_risk_budget: float = 0.30,
        max_correlation: float = 0.7
    ):
        """
        Initialize risk controller.
        
        Args:
            max_loss_per_trade: Maximum loss per trade as fraction of capital
            daily_risk_budget: Daily risk budget
            weekly_risk_budget: Weekly risk budget
            monthly_risk_budget: Monthly risk budget
            max_correlation: Maximum correlation between positions
        """
        self.max_loss_per_trade = max_loss_per_trade
        self.daily_risk_budget = daily_risk_budget
        self.weekly_risk_budget = weekly_risk_budget
        self.monthly_risk_budget = monthly_risk_budget
        self.max_correlation = max_correlation
        
        self.daily_losses = []
        self.weekly_losses = []
        self.monthly_losses = []
        
        logger.info("Initialized RiskController")
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        position_size: float,
        capital: float,
        atr: float
    ) -> float:
        """
        Calculate stop loss price.
        
        Args:
            entry_price: Entry price
            position_size: Position size (positive for long, negative for short)
            capital: Current capital
            atr: Average True Range (volatility measure)
        
        Returns:
            Stop loss price
        """
        # Maximum loss in currency
        max_loss = capital * self.max_loss_per_trade
        
        # Calculate stop loss distance
        stop_distance = max_loss / abs(position_size)
        
        # Use ATR-based stop loss (2x ATR)
        atr_stop_distance = 2 * atr
        
        # Use the tighter of the two
        stop_distance = min(stop_distance, atr_stop_distance)
        
        # Calculate stop loss price
        if position_size > 0:  # Long
            stop_loss = entry_price - stop_distance
        else:  # Short
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def check_risk_budget(self, current_loss: float, capital: float) -> Tuple[bool, str]:
        """
        Check if risk budget is exceeded.
        
        Args:
            current_loss: Current loss amount
            capital: Current capital
        
        Returns:
            Tuple of (is_within_budget, reason)
        """
        loss_pct = abs(current_loss) / capital
        
        # Check daily budget
        daily_loss = sum(self.daily_losses) + current_loss
        if abs(daily_loss) / capital > self.daily_risk_budget:
            return False, f"Daily risk budget exceeded: {abs(daily_loss)/capital:.2%}"
        
        # Check weekly budget
        weekly_loss = sum(self.weekly_losses) + current_loss
        if abs(weekly_loss) / capital > self.weekly_risk_budget:
            return False, f"Weekly risk budget exceeded: {abs(weekly_loss)/capital:.2%}"
        
        # Check monthly budget
        monthly_loss = sum(self.monthly_losses) + current_loss
        if abs(monthly_loss) / capital > self.monthly_risk_budget:
            return False, f"Monthly risk budget exceeded: {abs(monthly_loss)/capital:.2%}"
        
        return True, "Within risk budget"
    
    def record_trade(self, pnl: float, timestamp: datetime):
        """Record a completed trade."""
        self.daily_losses.append(pnl)
        self.weekly_losses.append(pnl)
        self.monthly_losses.append(pnl)
        
        # Clean up old records
        self._cleanup_old_records(timestamp)
    
    def _cleanup_old_records(self, current_time: datetime):
        """Remove old loss records outside the time windows."""
        # This is simplified - in production, you'd track timestamps
        if len(self.daily_losses) > 100:
            self.daily_losses = self.daily_losses[-100:]
        if len(self.weekly_losses) > 500:
            self.weekly_losses = self.weekly_losses[-500:]
        if len(self.monthly_losses) > 2000:
            self.monthly_losses = self.monthly_losses[-2000:]
    
    def check_correlation(
        self,
        new_symbol: str,
        existing_symbols: List[str],
        returns_data: pd.DataFrame
    ) -> bool:
        """
        Check if adding a new position would violate correlation limits.
        
        Args:
            new_symbol: Symbol to add
            existing_symbols: Currently held symbols
            returns_data: DataFrame with returns for all symbols
        
        Returns:
            True if correlation is acceptable
        """
        if not existing_symbols:
            return True
        
        for symbol in existing_symbols:
            if symbol in returns_data.columns and new_symbol in returns_data.columns:
                correlation = returns_data[symbol].corr(returns_data[new_symbol])
                
                if abs(correlation) > self.max_correlation:
                    logger.warning(f"High correlation between {new_symbol} and {symbol}: {correlation:.2f}")
                    return False
        
        return True


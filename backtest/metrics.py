"""
Performance Metrics Calculation

Calculates various trading performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats


class PerformanceMetrics:
    """Calculate trading performance metrics."""
    
    def __init__(
        self,
        equity_curve: np.ndarray,
        trades: pd.DataFrame,
        initial_capital: float,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize performance metrics calculator.
        
        Args:
            equity_curve: Array of portfolio values over time
            trades: DataFrame with trade history
            initial_capital: Initial capital
            risk_free_rate: Annual risk-free rate
        """
        self.equity_curve = equity_curve
        self.trades = trades
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        
        # Calculate returns
        self.returns = np.diff(equity_curve) / equity_curve[:-1]
        self.returns = self.returns[~np.isnan(self.returns)]
    
    def total_return(self) -> float:
        """Calculate total return."""
        if len(self.equity_curve) == 0:
            return 0.0
        return (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
    
    def annualized_return(self, periods_per_year: int = 252) -> float:
        """Calculate annualized return."""
        total_ret = self.total_return()
        n_periods = len(self.equity_curve)
        years = n_periods / periods_per_year
        
        if years > 0:
            return (1 + total_ret) ** (1 / years) - 1
        return 0.0
    
    def sharpe_ratio(self, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            periods_per_year: Number of periods per year (252 for daily, 8760 for hourly)
        
        Returns:
            Sharpe ratio
        """
        if len(self.returns) == 0 or self.returns.std() == 0:
            return 0.0
        
        excess_returns = self.returns - self.risk_free_rate / periods_per_year
        return np.sqrt(periods_per_year) * excess_returns.mean() / self.returns.std()
    
    def sortino_ratio(self, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).
        
        Args:
            periods_per_year: Number of periods per year
        
        Returns:
            Sortino ratio
        """
        if len(self.returns) == 0:
            return 0.0
        
        excess_returns = self.returns - self.risk_free_rate / periods_per_year
        downside_returns = self.returns[self.returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        downside_std = downside_returns.std()
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
    
    def max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.
        
        Returns:
            Maximum drawdown as a fraction
        """
        if len(self.equity_curve) == 0:
            return 0.0
        
        cummax = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - cummax) / cummax
        return abs(drawdown.min())
    
    def calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).
        
        Returns:
            Calmar ratio
        """
        max_dd = self.max_drawdown()
        if max_dd == 0:
            return 0.0
        
        ann_return = self.annualized_return()
        return ann_return / max_dd
    
    def win_rate(self) -> float:
        """
        Calculate win rate.
        
        Returns:
            Win rate as a fraction
        """
        if len(self.trades) == 0:
            return 0.0
        
        closed_trades = self.trades[self.trades['action'] == 'close']
        if len(closed_trades) == 0:
            return 0.0
        
        winning_trades = closed_trades[closed_trades['pnl'] > 0]
        return len(winning_trades) / len(closed_trades)
    
    def profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Returns:
            Profit factor
        """
        if len(self.trades) == 0:
            return 0.0
        
        closed_trades = self.trades[self.trades['action'] == 'close']
        if len(closed_trades) == 0:
            return 0.0
        
        gross_profit = closed_trades[closed_trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(closed_trades[closed_trades['pnl'] < 0]['pnl'].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def average_win(self) -> float:
        """Calculate average winning trade."""
        if len(self.trades) == 0:
            return 0.0
        
        closed_trades = self.trades[self.trades['action'] == 'close']
        winning_trades = closed_trades[closed_trades['pnl'] > 0]
        
        if len(winning_trades) == 0:
            return 0.0
        
        return winning_trades['pnl'].mean()
    
    def average_loss(self) -> float:
        """Calculate average losing trade."""
        if len(self.trades) == 0:
            return 0.0
        
        closed_trades = self.trades[self.trades['action'] == 'close']
        losing_trades = closed_trades[closed_trades['pnl'] < 0]
        
        if len(losing_trades) == 0:
            return 0.0
        
        return losing_trades['pnl'].mean()
    
    def expectancy(self) -> float:
        """
        Calculate expectancy (average profit per trade).
        
        Returns:
            Expectancy
        """
        win_rate = self.win_rate()
        avg_win = self.average_win()
        avg_loss = self.average_loss()
        
        return win_rate * avg_win + (1 - win_rate) * avg_loss
    
    def var(self, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            confidence: Confidence level
        
        Returns:
            VaR value
        """
        if len(self.returns) == 0:
            return 0.0
        
        return np.percentile(self.returns, (1 - confidence) * 100)
    
    def cvar(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            confidence: Confidence level
        
        Returns:
            CVaR value
        """
        if len(self.returns) == 0:
            return 0.0
        
        var = self.var(confidence)
        return self.returns[self.returns <= var].mean()
    
    def information_ratio(self, benchmark_returns: np.ndarray) -> float:
        """
        Calculate Information Ratio.
        
        Args:
            benchmark_returns: Benchmark returns
        
        Returns:
            Information ratio
        """
        if len(self.returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        active_returns = self.returns - benchmark_returns[:len(self.returns)]
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        return active_returns.mean() / tracking_error
    
    def get_all_metrics(self) -> dict:
        """Get all metrics as a dictionary."""
        return {
            'Total Return': f"{self.total_return():.2%}",
            'Annualized Return': f"{self.annualized_return():.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio():.2f}",
            'Sortino Ratio': f"{self.sortino_ratio():.2f}",
            'Max Drawdown': f"{self.max_drawdown():.2%}",
            'Calmar Ratio': f"{self.calmar_ratio():.2f}",
            'Win Rate': f"{self.win_rate():.2%}",
            'Profit Factor': f"{self.profit_factor():.2f}",
            'Average Win': f"${self.average_win():,.2f}",
            'Average Loss': f"${self.average_loss():,.2f}",
            'Expectancy': f"${self.expectancy():,.2f}",
            'VaR (95%)': f"{self.var():.2%}",
            'CVaR (95%)': f"{self.cvar():.2%}",
        }
    
    def print_summary(self):
        """Print performance summary."""
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        
        metrics = self.get_all_metrics()
        for name, value in metrics.items():
            print(f"{name:.<30} {value:>15}")
        
        print("="*50 + "\n")


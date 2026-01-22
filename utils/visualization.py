"""
Visualization utilities for trading results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path


def plot_equity_curve(equity_curve: pd.Series, save_path: str = None):
    """Plot equity curve."""
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve.values, linewidth=2)
    plt.title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_drawdown(equity_curve: pd.Series, save_path: str = None):
    """Plot drawdown chart."""
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    
    plt.figure(figsize=(12, 6))
    plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    plt.plot(drawdown.index, drawdown, color='red', linewidth=2)
    plt.title('Drawdown', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_returns_distribution(returns: pd.Series, save_path: str = None):
    """Plot returns distribution."""
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.4f}')
    plt.axvline(returns.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {returns.median():.4f}')
    plt.title('Returns Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_trade_pnl(trades_df: pd.DataFrame, save_path: str = None):
    """Plot individual trade PnL."""
    closed_trades = trades_df[trades_df['action'] == 'close']
    
    if len(closed_trades) == 0:
        return
    
    plt.figure(figsize=(12, 6))
    colors = ['green' if x > 0 else 'red' for x in closed_trades['pnl']]
    plt.bar(range(len(closed_trades)), closed_trades['pnl'], color=colors)
    plt.title('Individual Trade PnL', fontsize=16, fontweight='bold')
    plt.xlabel('Trade Number')
    plt.ylabel('PnL ($)')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_dashboard(results: Dict, output_dir: Path):
    """Create a comprehensive trading dashboard."""
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("darkgrid")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    equity_curve = results['equity_curve']['equity']
    trades = results['trades']
    
    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity_curve.index, equity_curve.values, linewidth=2, color='#2E86AB')
    ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='#A23B72')
    ax2.plot(drawdown.index, drawdown, color='#A23B72', linewidth=2)
    ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Returns Distribution
    ax3 = fig.add_subplot(gs[2, 0])
    returns = equity_curve.pct_change().dropna()
    ax3.hist(returns, bins=30, alpha=0.7, edgecolor='black', color='#F18F01')
    ax3.axvline(returns.mean(), color='red', linestyle='--', linewidth=2)
    ax3.set_title('Returns Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Returns')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # 4. Trade PnL
    ax4 = fig.add_subplot(gs[2, 1])
    closed_trades = trades[trades['action'] == 'close']
    if len(closed_trades) > 0:
        colors = ['#06A77D' if x > 0 else '#D62246' for x in closed_trades['pnl']]
        ax4.bar(range(len(closed_trades)), closed_trades['pnl'], color=colors)
        ax4.set_title('Trade PnL', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('PnL ($)')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.grid(True, alpha=0.3)
    
    # 5. Metrics Summary
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    metrics_text = f"""
    PERFORMANCE METRICS
    
    Total Return: {results['total_return']:.2%}
    Sharpe Ratio: {results['sharpe_ratio']:.2f}
    Sortino Ratio: {results['sortino_ratio']:.2f}
    Max Drawdown: {results['max_drawdown']:.2%}
    
    Win Rate: {results['win_rate']:.2%}
    Profit Factor: {results['profit_factor']:.2f}
    
    Avg Win: ${results['avg_win']:,.2f}
    Avg Loss: ${results['avg_loss']:,.2f}
    
    Total Trades: {results['num_trades']}
    """
    
    ax5.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(output_dir / 'dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dashboard saved to {output_dir / 'dashboard.png'}")


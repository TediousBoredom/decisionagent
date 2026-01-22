"""
Backtest Script

Runs backtesting on historical data with trained model.
"""

import torch
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
import sys

from backtest.engine import load_and_backtest


def setup_logger(log_file: str):
    """Setup logger configuration."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, rotation="1 day", retention="30 days", level="DEBUG")


def plot_results(results: dict, output_dir: Path):
    """Plot backtest results."""
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("darkgrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. Equity Curve
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    equity_curve = results['equity_curve']
    
    # Equity curve
    axes[0, 0].plot(equity_curve.index, equity_curve['equity'], linewidth=2)
    axes[0, 0].set_title('Equity Curve', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Drawdown
    cummax = equity_curve['equity'].cummax()
    drawdown = (equity_curve['equity'] - cummax) / cummax
    axes[0, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    axes[0, 1].plot(drawdown.index, drawdown, color='red', linewidth=2)
    axes[0, 1].set_title('Drawdown', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Returns distribution
    returns = equity_curve['equity'].pct_change().dropna()
    axes[1, 0].hist(returns, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 0].set_title('Returns Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Returns')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Trade PnL
    trades = results['trades']
    closed_trades = trades[trades['action'] == 'close']
    if len(closed_trades) > 0:
        axes[1, 1].bar(range(len(closed_trades)), closed_trades['pnl'], 
                      color=['green' if x > 0 else 'red' for x in closed_trades['pnl']])
        axes[1, 1].set_title('Trade PnL', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('PnL ($)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'backtest_results.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_dir / 'backtest_results.png'}")
    
    # 2. Monthly Returns Heatmap
    if len(equity_curve) > 30:
        monthly_returns = equity_curve['equity'].resample('M').last().pct_change()
        monthly_returns = monthly_returns.dropna()
        
        if len(monthly_returns) > 0:
            # Create pivot table for heatmap
            monthly_returns_df = pd.DataFrame({
                'year': monthly_returns.index.year,
                'month': monthly_returns.index.month,
                'return': monthly_returns.values
            })
            
            pivot = monthly_returns_df.pivot(index='month', columns='year', values='return')
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn', center=0, 
                       cbar_kws={'label': 'Monthly Return'})
            plt.title('Monthly Returns Heatmap', fontsize=16, fontweight='bold')
            plt.ylabel('Month')
            plt.xlabel('Year')
            plt.tight_layout()
            plt.savefig(output_dir / 'monthly_returns.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved monthly returns heatmap")


def save_results(results: dict, output_dir: Path):
    """Save backtest results to files."""
    output_dir.mkdir(exist_ok=True)
    
    # Save equity curve
    results['equity_curve'].to_csv(output_dir / 'equity_curve.csv')
    
    # Save trades
    results['trades'].to_csv(output_dir / 'trades.csv', index=False)
    
    # Save metrics
    metrics = {
        'Initial Capital': results['initial_capital'],
        'Final Capital': results['final_capital'],
        'Total Return': results['total_return'],
        'Number of Trades': results['num_trades'],
        'Sharpe Ratio': results['sharpe_ratio'],
        'Sortino Ratio': results['sortino_ratio'],
        'Max Drawdown': results['max_drawdown'],
        'Win Rate': results['win_rate'],
        'Profit Factor': results['profit_factor'],
        'Average Win': results['avg_win'],
        'Average Loss': results['avg_loss']
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / 'metrics.csv', index=False)
    
    logger.info(f"Saved results to {output_dir}")


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    setup_logger(log_dir / "backtest.log")
    
    logger.info("Starting backtest...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Period: {args.start_date} to {args.end_date}")
    
    # Run backtest
    results = load_and_backtest(
        model_path=args.model_path,
        market_data_path=args.data_path,
        config=config,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Initial Capital:    ${results['initial_capital']:>15,.2f}")
    print(f"Final Capital:      ${results['final_capital']:>15,.2f}")
    print(f"Total Return:       {results['total_return']:>15.2%}")
    print(f"Number of Trades:   {results['num_trades']:>15}")
    print("-"*60)
    print(f"Sharpe Ratio:       {results['sharpe_ratio']:>15.2f}")
    print(f"Sortino Ratio:      {results['sortino_ratio']:>15.2f}")
    print(f"Max Drawdown:       {results['max_drawdown']:>15.2%}")
    print(f"Win Rate:           {results['win_rate']:>15.2%}")
    print(f"Profit Factor:      {results['profit_factor']:>15.2f}")
    print(f"Average Win:        ${results['avg_win']:>15,.2f}")
    print(f"Average Loss:       ${results['avg_loss']:>15,.2f}")
    print("="*60 + "\n")
    
    # Save results
    output_dir = Path(args.output_dir)
    save_results(results, output_dir)
    
    # Plot results
    if not args.no_plot:
        plot_results(results, output_dir)
    
    logger.info("Backtest completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Diffusion Policy")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to market data")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output_dir", type=str, default="./backtest_results", help="Output directory")
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting")
    
    args = parser.parse_args()
    main(args)


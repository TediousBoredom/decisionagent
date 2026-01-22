"""
Quick Start Demo

A simple demonstration of the AlphaPolicy system.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from strategy.generator import DiffusionStrategyGenerator
from risk.constraints import RiskConstraints
from data.dataset import DataProcessor


def demo_strategy_generation():
    """Demonstrate strategy generation."""
    print("="*60)
    print("AlphaPolicy Demo: Strategy Generation")
    print("="*60)
    
    # Create risk constraints
    constraints = RiskConstraints(
        max_position_size=1.0,
        max_leverage=2.0,
        max_drawdown=0.15,
        max_daily_loss=0.03
    )
    
    # Create model
    print("\n1. Creating diffusion model...")
    model = DiffusionStrategyGenerator(
        price_dim=5,
        indicator_dim=20,
        orderbook_dim=10,
        regime_dim=5,
        seq_length=100,
        action_dim=4,
        action_seq_length=10,
        hidden_dim=128,
        cond_dim=64,
        num_timesteps=100,  # Reduced for demo
        constraints=constraints
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {num_params:,} parameters")
    
    # Create dummy market data
    print("\n2. Creating sample market data...")
    batch_size = 2
    seq_length = 100
    
    price = torch.randn(batch_size, seq_length, 5)
    indicators = torch.randn(batch_size, seq_length, 20)
    orderbook = torch.randn(batch_size, 10)
    regime = torch.randn(batch_size, 5)
    
    current_position = torch.zeros(batch_size, 1)
    portfolio_value = torch.ones(batch_size, 1) * 100000
    current_drawdown = torch.zeros(batch_size, 1)
    
    # Generate strategy
    print("\n3. Generating trading strategy...")
    model.eval()
    with torch.no_grad():
        output = model(
            price, indicators, orderbook, regime,
            current_position, portfolio_value, current_drawdown,
            num_samples=3,
            return_best=True
        )
    
    actions = output['actions']
    predicted_returns = output['predicted_returns']
    predicted_volatility = output['predicted_volatility']
    
    print(f"\n4. Strategy generated!")
    print(f"   Actions shape: {actions.shape}")
    print(f"   Sample action (first timestep):")
    print(f"     Position: {actions[0, 0, 0].item():.3f} (range: -1 to 1)")
    print(f"     Urgency: {actions[0, 0, 1].item():.3f} (range: 0 to 1)")
    print(f"     Stop Loss: {actions[0, 0, 2].item():.3f}")
    print(f"     Take Profit: {actions[0, 0, 3].item():.3f}")
    print(f"\n   Predicted return: {predicted_returns[0, 0].item():.4f}")
    print(f"   Predicted volatility: {predicted_volatility[0, 0].item():.4f}")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


def demo_data_processing():
    """Demonstrate data processing."""
    print("\n" + "="*60)
    print("AlphaPolicy Demo: Data Processing")
    print("="*60)
    
    # Create sample OHLCV data
    print("\n1. Creating sample OHLCV data...")
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Simulate price data
    np.random.seed(42)
    close = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close * (1 + np.random.randn(100) * 0.01),
        'high': close * (1 + np.abs(np.random.randn(100)) * 0.02),
        'low': close * (1 - np.abs(np.random.randn(100)) * 0.02),
        'close': close,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    
    print(f"   Created {len(df)} bars of data")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Process data
    print("\n2. Processing data and calculating indicators...")
    processor = DataProcessor()
    df_processed = processor.process_ohlcv(df)
    
    print(f"   Processed data shape: {df_processed.shape}")
    print(f"   Added indicators: {len(df_processed.columns) - 6}")
    
    # Show some indicators
    print("\n3. Sample indicators:")
    indicator_cols = ['sma_20', 'rsi_14', 'macd', 'bb_width', 'atr_14']
    for col in indicator_cols:
        if col in df_processed.columns:
            print(f"   {col}: {df_processed[col].iloc[-1]:.4f}")
    
    print("\n" + "="*60)


def demo_risk_management():
    """Demonstrate risk management."""
    print("\n" + "="*60)
    print("AlphaPolicy Demo: Risk Management")
    print("="*60)
    
    from risk.constraints import PortfolioRiskMonitor, RiskMetrics
    
    # Create risk constraints
    constraints = RiskConstraints(
        max_position_size=1.0,
        max_drawdown=0.2,
        min_sharpe_ratio=1.0
    )
    
    # Create risk monitor
    print("\n1. Creating risk monitor...")
    monitor = PortfolioRiskMonitor(constraints, lookback_window=50)
    
    # Simulate trading
    print("\n2. Simulating trading sequence...")
    np.random.seed(42)
    
    for i in range(50):
        returns = np.random.randn() * 0.02
        position = np.random.uniform(-0.5, 0.5)
        portfolio_value = 100000 * (1 + returns * i * 0.01)
        
        monitor.update(returns, position, portfolio_value)
    
    # Get metrics
    print("\n3. Risk metrics:")
    metrics = monitor.get_current_metrics()
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
    
    # Check constraints
    print("\n4. Checking risk constraints...")
    is_safe, violations = monitor.check_constraints()
    
    if is_safe:
        print("   ✓ All risk constraints satisfied")
    else:
        print("   ✗ Risk violations detected:")
        for violation in violations:
            print(f"     - {violation}")
    
    print("\n" + "="*60)


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("Welcome to AlphaPolicy Demo")
    print("Diffusion-Based Autonomous Trading System")
    print("="*60)
    
    try:
        # Demo 1: Strategy Generation
        demo_strategy_generation()
        
        # Demo 2: Data Processing
        demo_data_processing()
        
        # Demo 3: Risk Management
        demo_risk_management()
        
        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("\nNext steps:")
        print("1. Prepare your training data")
        print("2. Train the model: python examples/train.py")
        print("3. Run backtest: python examples/trade.py --model checkpoints/best_model.pt --mode backtest")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()


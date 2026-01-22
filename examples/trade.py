"""
Live Trading Example Script

Demonstrates how to use the trained model for live trading.
"""

import torch
import argparse
import yaml
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from strategy.generator import DiffusionStrategyGenerator
from execution.engine import ExecutionEngine, SimulatedBroker
from risk.constraints import RiskConstraints, PortfolioRiskMonitor
from data.dataset import DataProcessor


def parse_args():
    parser = argparse.ArgumentParser(description='Live Trading with Diffusion Model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='../configs/trade_config.yaml',
                       help='Path to trading config file')
    parser.add_argument('--mode', type=str, default='backtest',
                       choices=['backtest', 'paper', 'live'],
                       help='Trading mode')
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Trading symbol')
    parser.add_argument('--initial_capital', type=float, default=100000.0,
                       help='Initial capital')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run model on')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, config: dict, device: str) -> DiffusionStrategyGenerator:
    """Load trained model from checkpoint."""
    # Create risk constraints
    risk_constraints = RiskConstraints(
        max_position_size=config.get('max_position_size', 1.0),
        max_leverage=config.get('max_leverage', 3.0),
        max_drawdown=config.get('max_drawdown', 0.2),
        max_daily_loss=config.get('max_daily_loss', 0.05)
    )
    
    # Create model
    model = DiffusionStrategyGenerator(
        price_dim=5,
        indicator_dim=config.get('indicator_dim', 20),
        orderbook_dim=10,
        regime_dim=5,
        seq_length=config.get('seq_length', 100),
        action_dim=4,
        action_seq_length=config.get('action_seq_length', 10),
        hidden_dim=config.get('hidden_dim', 256),
        cond_dim=config.get('cond_dim', 128),
        num_timesteps=config.get('num_timesteps', 1000),
        constraints=risk_constraints
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    
    return model


def prepare_market_data(df, seq_length: int = 100):
    """Prepare market data for model input."""
    # Extract price data (last seq_length bars)
    price_data = df[['open', 'high', 'low', 'close', 'volume']].tail(seq_length).values
    
    # Normalize
    price_norm = np.zeros_like(price_data)
    for i in range(4):  # OHLC
        price_norm[:, i] = np.log(price_data[:, i] / price_data[0, i])
    price_norm[:, 4] = np.log(price_data[:, 4] + 1)  # Volume
    
    # Extract indicators
    indicator_cols = [col for col in df.columns 
                     if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
    
    if indicator_cols:
        indicators = df[indicator_cols].tail(seq_length).values
    else:
        indicators = np.zeros((seq_length, 20))
    
    # Simulate order book (in practice, use real order book data)
    orderbook = np.random.randn(10) * 0.1
    
    # Calculate regime features
    close = df['close'].tail(50).values
    returns = np.diff(close) / close[:-1]
    
    trend = (close[-1] - close[0]) / close[0]
    volatility = returns.std()
    volume_trend = (df['volume'].iloc[-1] - df['volume'].iloc[-20]) / df['volume'].iloc[-20]
    momentum = (returns > 0).sum() / len(returns)
    market_state = 1.0 if trend > 0 else -1.0
    
    regime = np.array([trend, volatility, volume_trend, momentum, market_state])
    
    return price_norm, indicators, orderbook, regime


def backtest(model, symbol: str, config: dict, initial_capital: float, device: str):
    """Run backtest."""
    print(f"\n{'='*60}")
    print(f"Starting Backtest for {symbol}")
    print(f"{'='*60}\n")
    
    # Download historical data
    processor = DataProcessor()
    df = processor.download_data(
        symbol=symbol,
        start_date=config.get('backtest_start', '2023-01-01'),
        end_date=config.get('backtest_end', '2024-01-01'),
        source='yahoo'
    )
    
    # Process data
    df = processor.process_ohlcv(df)
    
    print(f"Loaded {len(df)} bars of data")
    
    # Create execution engine with simulated broker
    broker = SimulatedBroker(slippage=0.001)
    engine = ExecutionEngine(
        broker=broker,
        initial_capital=initial_capital,
        max_slippage=0.001,
        commission_rate=0.001
    )
    
    # Create risk monitor
    risk_monitor = PortfolioRiskMonitor(
        constraints=model.constraints,
        lookback_window=100
    )
    
    seq_length = config.get('seq_length', 100)
    
    # Run backtest
    for i in range(seq_length, len(df)):
        current_bar = df.iloc[:i+1]
        current_price = current_bar['close'].iloc[-1]
        
        # Update broker prices
        broker.update_prices({symbol: current_price})
        
        # Prepare market data
        price, indicators, orderbook, regime = prepare_market_data(current_bar, seq_length)
        
        # Convert to tensors
        price_tensor = torch.FloatTensor(price).unsqueeze(0).to(device)
        indicators_tensor = torch.FloatTensor(indicators).unsqueeze(0).to(device)
        orderbook_tensor = torch.FloatTensor(orderbook).unsqueeze(0).to(device)
        regime_tensor = torch.FloatTensor(regime).unsqueeze(0).to(device)
        
        # Get portfolio state
        portfolio_state = engine.get_portfolio_state()
        current_position = torch.FloatTensor([[portfolio_state.get('num_positions', 0)]]).to(device)
        portfolio_value = torch.FloatTensor([[portfolio_state['total_equity']]]).to(device)
        current_drawdown = torch.FloatTensor([[portfolio_state['current_drawdown']]]).to(device)
        
        # Generate trading action
        with torch.no_grad():
            output = model(
                price_tensor,
                indicators_tensor,
                orderbook_tensor,
                regime_tensor,
                current_position,
                portfolio_value,
                current_drawdown,
                num_samples=3,
                return_best=True
            )
        
        # Extract first action
        action = output['actions'][0, 0].cpu().numpy()
        
        # Check risk constraints
        risk_monitor.update(
            returns=portfolio_state.get('total_return', 0),
            position=action[0],
            portfolio_value=portfolio_state['total_equity']
        )
        
        if risk_monitor.should_halt_trading():
            print(f"\n[{i}] Trading halted due to risk constraints")
            break
        
        # Execute action
        orders = engine.execute_action(symbol, action, current_price)
        
        # Update positions
        engine.update_positions({symbol: current_price})
        
        # Print progress
        if i % 20 == 0:
            metrics = engine.get_performance_metrics()
            print(f"[{i}/{len(df)}] Price: ${current_price:.2f} | "
                  f"Equity: ${portfolio_state['total_equity']:.2f} | "
                  f"Return: {portfolio_state['total_return']:.2%} | "
                  f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
    
    # Final results
    print(f"\n{'='*60}")
    print("Backtest Results")
    print(f"{'='*60}\n")
    
    final_metrics = engine.get_performance_metrics()
    portfolio_state = engine.get_portfolio_state()
    
    print(f"Initial Capital:    ${initial_capital:,.2f}")
    print(f"Final Equity:       ${portfolio_state['total_equity']:,.2f}")
    print(f"Total Return:       {final_metrics['total_return']:.2%}")
    print(f"Sharpe Ratio:       {final_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Sortino Ratio:      {final_metrics.get('sortino_ratio', 0):.2f}")
    print(f"Max Drawdown:       {final_metrics['max_drawdown']:.2%}")
    print(f"Win Rate:           {final_metrics['win_rate']:.2%}")
    print(f"Number of Trades:   {final_metrics['num_trades']}")
    print(f"Volatility:         {final_metrics['volatility']:.2%}")
    
    if 'calmar_ratio' in final_metrics:
        print(f"Calmar Ratio:       {final_metrics['calmar_ratio']:.2f}")
    
    print(f"\n{'='*60}\n")
    
    return engine, final_metrics


def paper_trade(model, symbol: str, config: dict, initial_capital: float, device: str):
    """Run paper trading (simulated real-time)."""
    print(f"\n{'='*60}")
    print(f"Starting Paper Trading for {symbol}")
    print(f"{'='*60}\n")
    
    # Similar to backtest but with real-time data fetching
    print("Paper trading mode - fetching real-time data...")
    
    # Implementation would fetch real-time data and execute trades
    # This is a placeholder
    print("Paper trading not fully implemented in this example")


def live_trade(model, symbol: str, config: dict, initial_capital: float, device: str):
    """Run live trading."""
    print(f"\n{'='*60}")
    print(f"⚠️  WARNING: LIVE TRADING MODE")
    print(f"{'='*60}\n")
    
    response = input("Are you sure you want to start live trading? (yes/no): ")
    if response.lower() != 'yes':
        print("Live trading cancelled.")
        return
    
    print("Live trading mode - connecting to broker...")
    
    # Implementation would connect to real broker
    # This is a placeholder
    print("Live trading not fully implemented in this example")
    print("Please implement broker connection and use with caution!")


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load model
    print("Loading model...")
    model = load_model(args.model, config, args.device)
    
    # Run trading based on mode
    if args.mode == 'backtest':
        backtest(model, args.symbol, config, args.initial_capital, args.device)
    elif args.mode == 'paper':
        paper_trade(model, args.symbol, config, args.initial_capital, args.device)
    elif args.mode == 'live':
        live_trade(model, args.symbol, config, args.initial_capital, args.device)


if __name__ == '__main__':
    main()


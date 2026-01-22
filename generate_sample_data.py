"""
Example script to generate sample human trading data for testing.

This creates synthetic trading data that mimics human trading patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_trades(
    num_trades: int = 500,
    start_date: str = "2023-01-01",
    symbols: list = None
) -> pd.DataFrame:
    """
    Generate sample human trading data.
    
    Args:
        num_trades: Number of trades to generate
        start_date: Start date for trades
        symbols: List of trading symbols
    
    Returns:
        DataFrame with sample trades
    """
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    np.random.seed(42)
    
    # Generate timestamps
    start = pd.to_datetime(start_date)
    timestamps = [start + timedelta(hours=np.random.randint(1, 24*30)) for _ in range(num_trades)]
    timestamps.sort()
    
    trades = []
    
    for i, timestamp in enumerate(timestamps):
        symbol = np.random.choice(symbols)
        
        # Simulate price based on symbol
        if "BTC" in symbol:
            base_price = 30000 + np.random.randn() * 5000
        elif "ETH" in symbol:
            base_price = 2000 + np.random.randn() * 300
        else:
            base_price = 50 + np.random.randn() * 10
        
        base_price = max(base_price, 1)
        
        # Generate trade parameters
        # Bias towards profitable trades (simulating a good trader)
        is_profitable = np.random.rand() > 0.35  # 65% win rate
        
        position_size = np.random.uniform(0.02, 0.15)  # 2-15% of capital
        
        # Long or short
        is_long = np.random.rand() > 0.5
        if not is_long:
            position_size = -position_size
        
        entry_price = base_price
        
        # Stop loss (1-5% from entry)
        stop_loss_pct = np.random.uniform(0.01, 0.05)
        if is_long:
            stop_loss = entry_price * (1 - stop_loss_pct)
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)
        
        # Exit price and PnL
        if is_profitable:
            # Profitable trade
            profit_pct = np.random.uniform(0.02, 0.10)  # 2-10% profit
            if is_long:
                exit_price = entry_price * (1 + profit_pct)
            else:
                exit_price = entry_price * (1 - profit_pct)
        else:
            # Losing trade (usually hits stop loss)
            if np.random.rand() > 0.3:
                exit_price = stop_loss
            else:
                loss_pct = np.random.uniform(0.005, stop_loss_pct)
                if is_long:
                    exit_price = entry_price * (1 - loss_pct)
                else:
                    exit_price = entry_price * (1 + loss_pct)
        
        # Calculate PnL
        if is_long:
            pnl = (exit_price - entry_price) * abs(position_size) * 10000  # Assuming $10k capital
        else:
            pnl = (entry_price - exit_price) * abs(position_size) * 10000
        
        trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'position_size': position_size,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'exit_price': exit_price,
            'pnl': pnl
        })
    
    df = pd.DataFrame(trades)
    
    # Calculate some statistics
    total_pnl = df['pnl'].sum()
    win_rate = (df['pnl'] > 0).sum() / len(df)
    sharpe = df['pnl'].mean() / df['pnl'].std() * np.sqrt(252) if df['pnl'].std() > 0 else 0
    
    print(f"Generated {num_trades} trades")
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    return df


if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_trades(num_trades=500)
    
    # Save to CSV
    output_path = "./data/human_trades.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    # Display first few rows
    print("\nFirst 5 trades:")
    print(df.head())


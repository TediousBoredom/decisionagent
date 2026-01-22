"""
Quick Start Guide and Examples
"""

# Example 1: Generate Sample Data
print("="*60)
print("EXAMPLE 1: Generate Sample Trading Data")
print("="*60)
print("""
python generate_sample_data.py

This will create a sample CSV file with human trading data at:
./data/human_trades.csv
""")

# Example 2: Train Model
print("\n" + "="*60)
print("EXAMPLE 2: Train Diffusion Policy Model")
print("="*60)
print("""
python train.py --config config.yaml --data_path ./data/human_trades.csv

This will:
1. Load human trading data
2. Fetch historical market data
3. Train the diffusion policy model
4. Save checkpoints to ./checkpoints/
5. Log training progress to ./runs/ (TensorBoard)

Monitor training with TensorBoard:
tensorboard --logdir=./runs
""")

# Example 3: Backtest
print("\n" + "="*60)
print("EXAMPLE 3: Backtest Strategy")
print("="*60)
print("""
python backtest.py \\
    --model_path ./checkpoints/best_model.pt \\
    --start_date 2023-01-01 \\
    --end_date 2024-01-01

This will:
1. Load the trained model
2. Run backtest on historical data
3. Calculate performance metrics
4. Generate plots and save results to ./backtest_results/
""")

# Example 4: Paper Trading
print("\n" + "="*60)
print("EXAMPLE 4: Paper Trading (Simulation)")
print("="*60)
print("""
python live_trading.py \\
    --model_path ./checkpoints/best_model.pt \\
    --mode paper

This will:
1. Connect to exchange testnet
2. Run the strategy in simulation mode
3. Execute trades with virtual money
4. Log all activities to ./logs/
""")

# Example 5: Live Trading
print("\n" + "="*60)
print("EXAMPLE 5: Live Trading (REAL MONEY)")
print("="*60)
print("""
⚠️  WARNING: This uses real money! Only use after thorough testing!

1. Set up API keys in config.yaml or environment variables:
   export BINANCE_API_KEY="your_api_key"
   export BINANCE_API_SECRET="your_api_secret"

2. Run live trading:
   python live_trading.py \\
       --model_path ./checkpoints/best_model.pt \\
       --mode live

The system will ask for confirmation before starting.
""")

# Configuration Tips
print("\n" + "="*60)
print("CONFIGURATION TIPS")
print("="*60)
print("""
Edit config.yaml to customize:

1. Trading Parameters:
   - symbols: Which assets to trade
   - timeframe: Trading timeframe (1h, 4h, 1d)
   - max_positions: Maximum concurrent positions

2. Risk Management:
   - max_position_size: Max % of capital per trade
   - max_loss_per_trade: Max loss per trade
   - daily_risk_budget: Daily risk limit

3. Model Parameters:
   - hidden_dim: Model capacity
   - num_diffusion_steps: Diffusion steps (more = better quality, slower)
   - beta_schedule: Noise schedule (cosine recommended)

4. Training:
   - batch_size: Batch size for training
   - learning_rate: Learning rate
   - num_epochs: Number of training epochs
""")

# Best Practices
print("\n" + "="*60)
print("BEST PRACTICES")
print("="*60)
print("""
1. Data Quality:
   - Use high-quality human trading data with Sharpe > 1.5
   - Ensure sufficient number of trades (>100)
   - Filter out outliers and bad trades

2. Training:
   - Start with paper trading to validate
   - Monitor validation loss to avoid overfitting
   - Use early stopping if validation loss increases

3. Risk Management:
   - Always use stop losses
   - Start with small position sizes
   - Monitor daily/weekly risk budgets
   - Diversify across multiple assets

4. Live Trading:
   - Test thoroughly in paper trading first
   - Start with small capital
   - Monitor performance closely
   - Have a kill switch ready

5. Monitoring:
   - Check logs regularly
   - Monitor Sharpe ratio and drawdown
   - Review trades for anomalies
   - Adjust parameters based on performance
""")

print("\n" + "="*60)
print("For more information, see README.md")
print("="*60 + "\n")


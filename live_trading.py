"""
Live Trading Script

Runs the trained diffusion policy in live trading mode.
"""

import torch
import yaml
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger
import sys

from models.diffusion_policy import DiffusionPolicy
from data.market_data import MarketDataCollector, MarketDataStream
from data.preprocessor import MarketFeatureEngineer, ActionNormalizer
from trading.executor import TradingExecutor, ExchangeAdapter, OrderType
from risk.risk_controller import PositionManager, RiskController


def setup_logger(log_file: str):
    """Setup logger configuration."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, rotation="1 day", retention="30 days", level="DEBUG")


class LiveTradingBot:
    """Live trading bot using diffusion policy."""
    
    def __init__(
        self,
        policy: DiffusionPolicy,
        config: dict,
        feature_engineer: MarketFeatureEngineer,
        action_normalizer: ActionNormalizer,
        mode: str = "paper"
    ):
        """
        Initialize live trading bot.
        
        Args:
            policy: Trained diffusion policy
            config: Configuration dictionary
            feature_engineer: Feature engineer
            action_normalizer: Action normalizer
            mode: Trading mode ('paper' or 'live')
        """
        self.policy = policy
        self.config = config
        self.feature_engineer = feature_engineer
        self.action_normalizer = action_normalizer
        self.mode = mode
        
        # Initialize components
        self.market_collector = MarketDataCollector(
            exchange_name=config['trading']['exchanges'][0]['name'],
            api_key=config['trading']['exchanges'][0].get('api_key'),
            api_secret=config['trading']['exchanges'][0].get('api_secret'),
            testnet=config['trading']['exchanges'][0]['testnet'] or mode == 'paper'
        )
        
        exchange_adapter = ExchangeAdapter(
            exchange_name=config['trading']['exchanges'][0]['name'],
            api_key=config['trading']['exchanges'][0].get('api_key'),
            api_secret=config['trading']['exchanges'][0].get('api_secret'),
            testnet=config['trading']['exchanges'][0]['testnet'] or mode == 'paper'
        )
        
        self.executor = TradingExecutor(exchange_adapter)
        
        # Get initial balance
        initial_balance = asyncio.run(exchange_adapter.get_balance())
        initial_capital = initial_balance.get('USDT', 100000)
        
        self.position_manager = PositionManager(
            initial_capital=initial_capital,
            max_position_size=config['risk']['max_position_size'],
            min_position_size=config['risk']['min_position_size'],
            max_positions=config['trading']['max_positions']
        )
        
        self.risk_controller = RiskController(
            max_loss_per_trade=config['risk']['max_loss_per_trade'],
            daily_risk_budget=config['risk']['daily_risk_budget'],
            weekly_risk_budget=config['risk']['weekly_risk_budget'],
            monthly_risk_budget=config['risk']['monthly_risk_budget']
        )
        
        # Market data cache
        self.market_data_cache = {}
        
        logger.info(f"Initialized LiveTradingBot in {mode} mode")
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
    
    async def update_market_data(self, symbol: str):
        """Update market data for a symbol."""
        try:
            # Fetch latest OHLCV data
            df = self.market_collector.fetch_ohlcv(
                symbol=symbol,
                timeframe=self.config['trading']['timeframe'],
                limit=self.config['strategy']['lookback_window'] + 200  # Extra for indicators
            )
            
            # Process features
            df = self.feature_engineer.process_dataframe(df, fit_scaler=False)
            
            self.market_data_cache[symbol] = df
            
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {e}")
    
    async def generate_action(self, symbol: str) -> dict:
        """Generate trading action for a symbol."""
        try:
            if symbol not in self.market_data_cache:
                await self.update_market_data(symbol)
            
            df = self.market_data_cache[symbol]
            
            # Create market state
            state = self.feature_engineer.create_market_state(df, len(df) - 1)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.policy.device)
            
            # Sample action from policy
            with torch.no_grad():
                action_tensor = self.policy.sample(state_tensor, num_samples=1)
                action_array = action_tensor[0, 0].cpu().numpy()
            
            # Denormalize action
            action = self.action_normalizer.denormalize_action(action_array)
            
            return action
            
        except Exception as e:
            logger.error(f"Error generating action for {symbol}: {e}")
            return None
    
    async def execute_strategy(self):
        """Main strategy execution loop."""
        logger.info("Starting strategy execution...")
        
        symbols = self.config['trading']['symbols']
        
        while True:
            try:
                # Update market data for all symbols
                for symbol in symbols:
                    await self.update_market_data(symbol)
                
                # Get current prices
                current_prices = {}
                for symbol in symbols:
                    ticker = self.market_collector.fetch_ticker(symbol)
                    current_prices[symbol] = ticker['last']
                
                # Update existing positions
                to_close = self.position_manager.update_positions(current_prices)
                
                # Close positions that hit stop loss / take profit
                for symbol in to_close:
                    position = self.position_manager.positions[symbol]
                    await self.executor.close_position(symbol, position.position_size)
                    
                    pnl = self.position_manager.close_position(symbol, current_prices[symbol])
                    self.risk_controller.record_trade(pnl, datetime.now())
                    
                    logger.info(f"Closed position {symbol}: PnL = ${pnl:,.2f}")
                
                # Generate actions for symbols without positions
                for symbol in symbols:
                    # Skip if already have position
                    if symbol in self.position_manager.positions:
                        continue
                    
                    # Skip if max positions reached
                    if len(self.position_manager.positions) >= self.config['trading']['max_positions']:
                        continue
                    
                    # Generate action
                    action = await self.generate_action(symbol)
                    
                    if action is None:
                        continue
                    
                    # Check if action is significant
                    if abs(action['position_size']) < 0.1:
                        continue
                    
                    # Calculate position size
                    df = self.market_data_cache[symbol]
                    volatility = df.iloc[-1].get('volatility_20', 0.02)
                    
                    position_size = self.position_manager.calculate_position_size(
                        symbol=symbol,
                        action_size=action['position_size'],
                        current_price=current_prices[symbol],
                        volatility=volatility,
                        confidence=1.0
                    )
                    
                    # Calculate entry price and stop loss
                    entry_price = current_prices[symbol] * (1 + action['entry_price_offset'])
                    atr = df.iloc[-1].get('atr_14', current_prices[symbol] * 0.02)
                    
                    stop_loss = self.risk_controller.calculate_stop_loss(
                        entry_price=entry_price,
                        position_size=position_size,
                        capital=self.position_manager.current_capital,
                        atr=atr
                    )
                    
                    # Check risk budget
                    max_loss = abs(entry_price - stop_loss) * abs(position_size)
                    within_budget, reason = self.risk_controller.check_risk_budget(
                        -max_loss,
                        self.position_manager.current_capital
                    )
                    
                    if not within_budget:
                        logger.warning(f"Skipping {symbol}: {reason}")
                        continue
                    
                    # Execute trade
                    logger.info(f"Opening position: {symbol}, size: {position_size:.4f}, "
                               f"entry: ${entry_price:.2f}, stop: ${stop_loss:.2f}")
                    
                    order = await self.executor.execute_trade(
                        symbol=symbol,
                        position_size=position_size,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        order_type=OrderType.MARKET
                    )
                    
                    if order and order.status in ['closed', 'filled']:
                        self.position_manager.open_position(
                            symbol=symbol,
                            entry_price=order.average_price,
                            position_size=position_size,
                            stop_loss=stop_loss
                        )
                
                # Log portfolio status
                portfolio_value = self.position_manager.get_portfolio_value(current_prices)
                summary = self.position_manager.get_position_summary()
                
                logger.info(f"Portfolio Value: ${portfolio_value:,.2f}")
                logger.info(f"Open Positions: {summary['num_positions']}")
                logger.info(f"Total PnL: ${summary['total_pnl']:,.2f}")
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in strategy execution: {e}")
                await asyncio.sleep(60)


async def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    setup_logger(log_dir / f"live_trading_{args.mode}.log")
    
    logger.info(f"Starting live trading in {args.mode} mode...")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    
    # Calculate state dimension (need to load sample data)
    feature_engineer = MarketFeatureEngineer(
        lookback_window=config['strategy']['lookback_window'],
        technical_indicators=config['strategy']['technical_indicators']
    )
    
    # Initialize policy
    policy = DiffusionPolicy(
        state_dim=config['model']['state_dim'],
        action_dim=config['model']['action_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        num_diffusion_steps=config['model']['num_diffusion_steps'],
        beta_schedule=config['model']['beta_schedule'],
        device=device
    )
    
    policy.load(args.model_path)
    policy.network.eval()
    
    logger.info("Model loaded successfully")
    
    # Initialize action normalizer
    action_normalizer = ActionNormalizer()
    
    # Create trading bot
    bot = LiveTradingBot(
        policy=policy,
        config=config,
        feature_engineer=feature_engineer,
        action_normalizer=action_normalizer,
        mode=args.mode
    )
    
    # Run strategy
    await bot.execute_strategy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Trading with Diffusion Policy")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--mode", type=str, default="paper", choices=["paper", "live"],
                       help="Trading mode: paper (simulation) or live (real money)")
    
    args = parser.parse_args()
    
    # Confirm if running in live mode
    if args.mode == "live":
        response = input("WARNING: You are about to run LIVE trading with real money. Are you sure? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            sys.exit(0)
    
    asyncio.run(main(args))


"""
Backtesting Engine

Simulates trading strategy on historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger

from risk.risk_controller import PositionManager, RiskController, Position
from backtest.metrics import PerformanceMetrics


class BacktestEngine:
    """Backtesting engine for strategy evaluation."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 0.1,
        max_positions: int = 5
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade
            slippage: Slippage rate
            max_position_size: Maximum position size
            max_positions: Maximum concurrent positions
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.position_manager = PositionManager(
            initial_capital=initial_capital,
            max_position_size=max_position_size,
            max_positions=max_positions
        )
        
        self.risk_controller = RiskController()
        
        self.trades: List[Dict] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        logger.info(f"Initialized BacktestEngine with ${initial_capital:,.2f}")
    
    def apply_costs(self, price: float, amount: float, side: str) -> float:
        """
        Apply commission and slippage to a trade.
        
        Args:
            price: Trade price
            amount: Trade amount
            side: 'buy' or 'sell'
        
        Returns:
            Adjusted price after costs
        """
        # Apply slippage
        if side == 'buy':
            price *= (1 + self.slippage)
        else:
            price *= (1 - self.slippage)
        
        # Commission is applied separately to capital
        return price
    
    def execute_action(
        self,
        timestamp: datetime,
        symbol: str,
        action: Dict[str, float],
        current_price: float,
        atr: float
    ) -> bool:
        """
        Execute a trading action in backtest.
        
        Args:
            timestamp: Current timestamp
            symbol: Trading symbol
            action: Action dict with position_size, entry_price_offset, stop_loss_pct
            current_price: Current market price
            atr: Average True Range
        
        Returns:
            True if action executed successfully
        """
        position_size = action['position_size']
        entry_price_offset = action['entry_price_offset']
        stop_loss_pct = action['stop_loss_pct']
        
        # Skip if position size too small
        if abs(position_size) < 0.01:
            return False
        
        # Calculate entry price
        entry_price = current_price * (1 + entry_price_offset)
        entry_price = self.apply_costs(entry_price, abs(position_size), 
                                       'buy' if position_size > 0 else 'sell')
        
        # Calculate stop loss
        if position_size > 0:  # Long
            stop_loss = entry_price * (1 - stop_loss_pct)
        else:  # Short
            stop_loss = entry_price * (1 + stop_loss_pct)
        
        # Check if we can open position
        if not self.position_manager.can_open_position(symbol):
            return False
        
        # Calculate position size in base currency
        position_value = self.position_manager.current_capital * abs(position_size)
        actual_size = position_value / entry_price
        
        if position_size < 0:
            actual_size = -actual_size
        
        # Apply commission
        commission_cost = position_value * self.commission
        self.position_manager.current_capital -= commission_cost
        
        # Open position
        success = self.position_manager.open_position(
            symbol=symbol,
            entry_price=entry_price,
            position_size=actual_size,
            stop_loss=stop_loss
        )
        
        if success:
            self.trades.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'open',
                'side': 'long' if position_size > 0 else 'short',
                'price': entry_price,
                'size': actual_size,
                'stop_loss': stop_loss,
                'commission': commission_cost
            })
        
        return success
    
    def update_positions(
        self,
        timestamp: datetime,
        prices: Dict[str, float]
    ) -> List[str]:
        """
        Update positions and check for exits.
        
        Args:
            timestamp: Current timestamp
            prices: Current prices for all symbols
        
        Returns:
            List of symbols that were closed
        """
        closed_symbols = []
        
        # Check for stop loss / take profit
        to_close = self.position_manager.update_positions(prices)
        
        for symbol in to_close:
            exit_price = prices[symbol]
            exit_price = self.apply_costs(
                exit_price,
                abs(self.position_manager.positions[symbol].position_size),
                'sell' if self.position_manager.positions[symbol].position_size > 0 else 'buy'
            )
            
            # Apply commission
            position_value = abs(self.position_manager.positions[symbol].position_size) * exit_price
            commission_cost = position_value * self.commission
            self.position_manager.current_capital -= commission_cost
            
            # Close position
            pnl = self.position_manager.close_position(symbol, exit_price)
            
            if pnl is not None:
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'close',
                    'price': exit_price,
                    'pnl': pnl,
                    'commission': commission_cost
                })
                
                closed_symbols.append(symbol)
                
                # Record trade for risk management
                self.risk_controller.record_trade(pnl, timestamp)
        
        # Record equity
        portfolio_value = self.position_manager.get_portfolio_value(prices)
        self.equity_curve.append((timestamp, portfolio_value))
        
        return closed_symbols
    
    def run_backtest(
        self,
        market_data: Dict[str, pd.DataFrame],
        policy,
        feature_engineer,
        action_normalizer
    ) -> Dict:
        """
        Run complete backtest.
        
        Args:
            market_data: Dictionary of symbol -> DataFrame with OHLCV and features
            policy: Trained diffusion policy
            feature_engineer: Feature engineer instance
            action_normalizer: Action normalizer instance
        
        Returns:
            Backtest results dictionary
        """
        logger.info("Starting backtest...")
        
        # Get all timestamps (assuming all symbols have same timestamps)
        first_symbol = list(market_data.keys())[0]
        timestamps = market_data[first_symbol].index
        
        for i, timestamp in enumerate(timestamps):
            # Get current prices
            current_prices = {
                symbol: df.iloc[i]['close']
                for symbol, df in market_data.items()
            }
            
            # Update existing positions
            self.update_positions(timestamp, current_prices)
            
            # Generate actions for each symbol
            for symbol, df in market_data.items():
                # Skip if already have position
                if symbol in self.position_manager.positions:
                    continue
                
                # Skip if max positions reached
                if len(self.position_manager.positions) >= self.position_manager.max_positions:
                    continue
                
                try:
                    # Create market state
                    state = feature_engineer.create_market_state(df, i)
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(policy.device)
                    
                    # Sample action from policy
                    with torch.no_grad():
                        action_tensor = policy.sample(state_tensor, num_samples=1)
                        action_array = action_tensor[0, 0].cpu().numpy()
                    
                    # Denormalize action
                    action = action_normalizer.denormalize_action(action_array)
                    
                    # Get ATR for stop loss calculation
                    atr = df.iloc[i].get('atr_14', df.iloc[i]['close'] * 0.02)
                    
                    # Execute action
                    self.execute_action(
                        timestamp=timestamp,
                        symbol=symbol,
                        action=action,
                        current_price=current_prices[symbol],
                        atr=atr
                    )
                    
                except Exception as e:
                    logger.warning(f"Error processing {symbol} at {timestamp}: {e}")
                    continue
            
            # Log progress
            if i % 100 == 0:
                portfolio_value = self.position_manager.get_portfolio_value(current_prices)
                logger.info(f"Progress: {i}/{len(timestamps)}, Portfolio: ${portfolio_value:,.2f}")
        
        # Close all remaining positions at end
        final_prices = {
            symbol: df.iloc[-1]['close']
            for symbol, df in market_data.items()
        }
        
        for symbol in list(self.position_manager.positions.keys()):
            exit_price = final_prices[symbol]
            pnl = self.position_manager.close_position(symbol, exit_price)
            
            if pnl is not None:
                self.trades.append({
                    'timestamp': timestamps[-1],
                    'symbol': symbol,
                    'action': 'close',
                    'price': exit_price,
                    'pnl': pnl,
                    'commission': 0
                })
        
        # Calculate metrics
        results = self.calculate_results()
        
        logger.info("Backtest completed!")
        logger.info(f"Final Capital: ${results['final_capital']:,.2f}")
        logger.info(f"Total Return: {results['total_return']:.2%}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
        
        return results
    
    def calculate_results(self) -> Dict:
        """Calculate backtest results and metrics."""
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate metrics
        metrics = PerformanceMetrics(
            equity_curve=equity_df['equity'].values,
            trades=trades_df,
            initial_capital=self.initial_capital
        )
        
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.position_manager.current_capital,
            'total_return': (self.position_manager.current_capital - self.initial_capital) / self.initial_capital,
            'num_trades': len([t for t in self.trades if t['action'] == 'close']),
            'sharpe_ratio': metrics.sharpe_ratio(),
            'sortino_ratio': metrics.sortino_ratio(),
            'max_drawdown': metrics.max_drawdown(),
            'win_rate': metrics.win_rate(),
            'profit_factor': metrics.profit_factor(),
            'avg_win': metrics.average_win(),
            'avg_loss': metrics.average_loss(),
            'equity_curve': equity_df,
            'trades': trades_df
        }
        
        return results


import torch


def load_and_backtest(
    model_path: str,
    market_data_path: str,
    config: Dict,
    start_date: str,
    end_date: str
) -> Dict:
    """
    Load model and run backtest.
    
    Args:
        model_path: Path to trained model
        market_data_path: Path to market data
        config: Configuration dictionary
        start_date: Start date for backtest
        end_date: End date for backtest
    
    Returns:
        Backtest results
    """
    from models.diffusion_policy import DiffusionPolicy
    from data.market_data import MarketDataCollector
    from data.preprocessor import MarketFeatureEngineer, ActionNormalizer
    
    # Load model
    policy = DiffusionPolicy(
        state_dim=config['model']['state_dim'],
        action_dim=config['model']['action_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        num_diffusion_steps=config['model']['num_diffusion_steps'],
        beta_schedule=config['model']['beta_schedule']
    )
    policy.load(model_path)
    policy.network.eval()
    
    # Initialize feature engineer and normalizer
    feature_engineer = MarketFeatureEngineer(
        lookback_window=config['strategy']['lookback_window'],
        technical_indicators=config['strategy']['technical_indicators']
    )
    action_normalizer = ActionNormalizer()
    
    # Load market data
    collector = MarketDataCollector()
    market_data = {}
    
    for symbol in config['trading']['symbols']:
        df = collector.fetch_historical_data(
            symbol=symbol,
            timeframe=config['trading']['timeframe'],
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date)
        )
        
        # Process features
        df = feature_engineer.process_dataframe(df, fit_scaler=False)
        market_data[symbol] = df
    
    # Run backtest
    engine = BacktestEngine(
        initial_capital=config['backtest']['initial_capital'],
        commission=config['backtest']['commission'],
        slippage=config['backtest']['slippage']
    )
    
    results = engine.run_backtest(
        market_data=market_data,
        policy=policy,
        feature_engineer=feature_engineer,
        action_normalizer=action_normalizer
    )
    
    return results


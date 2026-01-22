"""
Strategy Dataset for Training Diffusion Policy

Loads and processes human trading data for model training.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from loguru import logger

from data.preprocessor import MarketFeatureEngineer, ActionNormalizer


class HumanTradingDataset(Dataset):
    """Dataset of human trading strategies."""
    
    def __init__(
        self,
        trades_df: pd.DataFrame,
        market_df: pd.DataFrame,
        feature_engineer: MarketFeatureEngineer,
        action_normalizer: ActionNormalizer,
        min_sharpe_ratio: float = 1.5,
        min_trades: int = 100
    ):
        """
        Initialize trading dataset.
        
        Args:
            trades_df: DataFrame with human trades
                Required columns: timestamp, symbol, position_size, entry_price, stop_loss, pnl
            market_df: DataFrame with market data (OHLCV + features)
            feature_engineer: Feature engineering instance
            action_normalizer: Action normalization instance
            min_sharpe_ratio: Minimum Sharpe ratio to include trades
            min_trades: Minimum number of trades required
        """
        self.feature_engineer = feature_engineer
        self.action_normalizer = action_normalizer
        
        # Filter high-quality trades
        self.trades_df = self._filter_quality_trades(
            trades_df,
            min_sharpe_ratio,
            min_trades
        )
        
        self.market_df = market_df
        
        # Align trades with market data
        self.samples = self._create_samples()
        
        logger.info(f"Created dataset with {len(self.samples)} samples")
    
    def _filter_quality_trades(
        self,
        trades_df: pd.DataFrame,
        min_sharpe_ratio: float,
        min_trades: int
    ) -> pd.DataFrame:
        """Filter trades based on quality metrics."""
        
        # Calculate returns
        returns = trades_df['pnl'].values
        
        # Calculate Sharpe ratio
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        logger.info(f"Strategy Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Filter based on criteria
        if len(trades_df) < min_trades:
            logger.warning(f"Insufficient trades: {len(trades_df)} < {min_trades}")
            return pd.DataFrame()
        
        if sharpe_ratio < min_sharpe_ratio:
            logger.warning(f"Low Sharpe ratio: {sharpe_ratio:.2f} < {min_sharpe_ratio}")
            return pd.DataFrame()
        
        # Filter out outlier trades (beyond 3 std)
        mean_pnl = returns.mean()
        std_pnl = returns.std()
        trades_df = trades_df[
            (trades_df['pnl'] >= mean_pnl - 3 * std_pnl) &
            (trades_df['pnl'] <= mean_pnl + 3 * std_pnl)
        ]
        
        logger.info(f"Filtered to {len(trades_df)} high-quality trades")
        return trades_df
    
    def _create_samples(self) -> List[Dict]:
        """Create training samples by aligning trades with market states."""
        samples = []
        
        for idx, trade in self.trades_df.iterrows():
            try:
                # Find corresponding market state
                timestamp = trade['timestamp']
                symbol = trade['symbol']
                
                # Get market data at trade time
                market_idx = self.market_df.index.get_loc(timestamp, method='nearest')
                
                # Create market state
                state = self.feature_engineer.create_market_state(
                    self.market_df,
                    market_idx
                )
                
                # Get current price
                current_price = self.market_df.iloc[market_idx]['close']
                
                # Create action
                action = {
                    'position_size': trade['position_size'],
                    'entry_price_offset': (trade['entry_price'] - current_price) / current_price,
                    'stop_loss_pct': abs(trade['stop_loss'] - trade['entry_price']) / trade['entry_price']
                }
                
                # Normalize action
                normalized_action = self.action_normalizer.normalize_action(action)
                
                samples.append({
                    'state': state,
                    'action': normalized_action,
                    'pnl': trade['pnl'],
                    'timestamp': timestamp,
                    'symbol': symbol
                })
                
            except Exception as e:
                logger.warning(f"Error creating sample for trade {idx}: {e}")
                continue
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample."""
        sample = self.samples[idx]
        
        state = torch.tensor(sample['state'], dtype=torch.float32)
        action = torch.tensor(sample['action'], dtype=torch.float32)
        
        return state, action


def load_human_trades(file_path: str) -> pd.DataFrame:
    """
    Load human trading data from CSV.
    
    Expected CSV format:
    timestamp, symbol, position_size, entry_price, stop_loss, exit_price, pnl
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        DataFrame with trades
    """
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        logger.info(f"Loaded {len(df)} trades from {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading trades from {file_path}: {e}")
        raise


def create_dataloaders(
    trades_df: pd.DataFrame,
    market_df: pd.DataFrame,
    feature_engineer: MarketFeatureEngineer,
    action_normalizer: ActionNormalizer,
    batch_size: int = 64,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        trades_df: Trading data
        market_df: Market data
        feature_engineer: Feature engineer
        action_normalizer: Action normalizer
        batch_size: Batch size
        train_split: Training split ratio
        val_split: Validation split ratio
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    full_dataset = HumanTradingDataset(
        trades_df,
        market_df,
        feature_engineer,
        action_normalizer
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders: train={train_size}, val={val_size}, test={test_size}")
    
    return train_loader, val_loader, test_loader


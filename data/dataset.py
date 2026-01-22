"""
Market Data Processing and Feature Engineering
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class MarketDataset(Dataset):
    """
    Dataset for training the diffusion trading model.
    """
    
    def __init__(
        self,
        data_path: str,
        seq_length: int = 100,
        action_seq_length: int = 10,
        price_cols: List[str] = ['open', 'high', 'low', 'close', 'volume'],
        indicator_cols: Optional[List[str]] = None,
        transform: Optional[callable] = None
    ):
        self.seq_length = seq_length
        self.action_seq_length = action_seq_length
        self.price_cols = price_cols
        self.transform = transform
        
        # Load data
        self.data = pd.read_csv(data_path)
        
        # Auto-detect indicator columns if not provided
        if indicator_cols is None:
            self.indicator_cols = [col for col in self.data.columns 
                                  if col not in price_cols + ['timestamp', 'symbol']]
        else:
            self.indicator_cols = indicator_cols
        
        # Normalize data
        self._normalize_data()
        
        # Calculate valid indices
        self.valid_indices = list(range(
            seq_length,
            len(self.data) - action_seq_length
        ))
        
    def _normalize_data(self):
        """Normalize features."""
        # Price normalization (log returns)
        for col in ['open', 'high', 'low', 'close']:
            if col in self.data.columns:
                self.data[f'{col}_norm'] = np.log(self.data[col] / self.data[col].shift(1))
                self.data[f'{col}_norm'].fillna(0, inplace=True)
        
        # Volume normalization
        if 'volume' in self.data.columns:
            self.data['volume_norm'] = np.log(self.data['volume'] + 1)
            self.data['volume_norm'] = (self.data['volume_norm'] - self.data['volume_norm'].mean()) / \
                                       (self.data['volume_norm'].std() + 1e-8)
        
        # Indicator normalization (z-score)
        for col in self.indicator_cols:
            if col in self.data.columns:
                mean = self.data[col].mean()
                std = self.data[col].std()
                self.data[f'{col}_norm'] = (self.data[col] - mean) / (std + 1e-8)
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary containing:
            - price: [seq_length, 5] - OHLCV data
            - indicators: [seq_length, num_indicators]
            - orderbook: [10] - Order book features (simulated)
            - regime: [5] - Market regime features
            - actions: [action_seq_length, 4] - Expert actions
            - returns: [action_seq_length] - Realized returns
            - volatility: [action_seq_length] - Realized volatility
        """
        actual_idx = self.valid_indices[idx]
        
        # Extract price sequence
        price_data = []
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if f'{col}_norm' in self.data.columns:
                price_data.append(
                    self.data[f'{col}_norm'].iloc[actual_idx - self.seq_length:actual_idx].values
                )
        price = np.stack(price_data, axis=-1)  # [seq_length, 5]
        
        # Extract indicators
        indicator_data = []
        for col in self.indicator_cols:
            if f'{col}_norm' in self.data.columns:
                indicator_data.append(
                    self.data[f'{col}_norm'].iloc[actual_idx - self.seq_length:actual_idx].values
                )
        
        if indicator_data:
            indicators = np.stack(indicator_data, axis=-1)  # [seq_length, num_indicators]
        else:
            indicators = np.zeros((self.seq_length, 20))  # Default size
        
        # Simulate order book features (in practice, use real order book data)
        orderbook = np.random.randn(10) * 0.1
        
        # Extract regime features
        regime = self._extract_regime_features(actual_idx)
        
        # Extract expert actions (in practice, these come from high-return strategies)
        actions = self._extract_expert_actions(actual_idx)
        
        # Calculate returns and volatility for the action period
        close_prices = self.data['close'].iloc[actual_idx:actual_idx + self.action_seq_length].values
        returns = np.diff(close_prices) / close_prices[:-1]
        returns = np.concatenate([[0], returns])  # Pad to match length
        
        # Rolling volatility
        volatility = pd.Series(returns).rolling(window=5, min_periods=1).std().values
        
        sample = {
            'price': torch.FloatTensor(price),
            'indicators': torch.FloatTensor(indicators),
            'orderbook': torch.FloatTensor(orderbook),
            'regime': torch.FloatTensor(regime),
            'actions': torch.FloatTensor(actions),
            'returns': torch.FloatTensor(returns),
            'volatility': torch.FloatTensor(volatility)
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _extract_regime_features(self, idx: int) -> np.ndarray:
        """Extract market regime features."""
        # Calculate trend strength
        close = self.data['close'].iloc[idx - 50:idx].values
        trend = (close[-1] - close[0]) / close[0]
        
        # Calculate volatility
        returns = np.diff(close) / close[:-1]
        volatility = returns.std()
        
        # Calculate volume trend
        volume = self.data['volume'].iloc[idx - 20:idx].values
        volume_trend = (volume[-1] - volume[0]) / (volume[0] + 1e-8)
        
        # RSI-like momentum
        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        momentum = gains / (gains + losses + 1e-8)
        
        # Market state (simplified)
        market_state = 1.0 if trend > 0 else -1.0
        
        regime = np.array([trend, volatility, volume_trend, momentum, market_state])
        
        return regime
    
    def _extract_expert_actions(self, idx: int) -> np.ndarray:
        """
        Extract expert actions (high-return strategy).
        In practice, this would come from backtested strategies.
        """
        actions = []
        
        for t in range(self.action_seq_length):
            current_idx = idx + t
            
            # Simple momentum strategy as example
            if current_idx < len(self.data) - 1:
                # Calculate short-term momentum
                close = self.data['close'].iloc[current_idx - 10:current_idx].values
                momentum = (close[-1] - close[0]) / close[0]
                
                # Position based on momentum
                position = np.tanh(momentum * 10)  # Scale and bound to [-1, 1]
                
                # Urgency based on volatility
                returns = np.diff(close) / close[:-1]
                volatility = returns.std()
                urgency = min(volatility * 5, 1.0)
                
                # Stop loss and take profit
                stop_loss = 0.02 + volatility * 0.5
                take_profit = 0.05 + abs(momentum) * 0.5
                
                action = np.array([position, urgency, stop_loss, take_profit])
            else:
                action = np.zeros(4)
            
            actions.append(action)
        
        return np.array(actions)


class DataProcessor:
    """
    Processes raw market data and calculates technical indicators.
    """
    
    def __init__(self):
        pass
    
    def process_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process OHLCV data and add technical indicators.
        
        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
        
        Returns:
            Processed DataFrame with additional indicator columns
        """
        df = df.copy()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], period=14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR
        df['atr_14'] = self._calculate_atr(df, period=14)
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Momentum
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Volatility
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        
        # Drop NaN rows
        df.dropna(inplace=True)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, 
                                   period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def download_data(self, symbol: str, start_date: str, end_date: str, 
                     source: str = 'yahoo') -> pd.DataFrame:
        """
        Download market data from various sources.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: Data source ('yahoo', 'binance', 'alpaca')
        
        Returns:
            DataFrame with OHLCV data
        """
        if source == 'yahoo':
            import yfinance as yf
            df = yf.download(symbol, start=start_date, end=end_date)
            df.columns = [col.lower() for col in df.columns]
            df.reset_index(inplace=True)
            df.rename(columns={'date': 'timestamp'}, inplace=True)
            
        elif source == 'binance':
            import ccxt
            exchange = ccxt.binance()
            
            # Convert dates to timestamps
            since = exchange.parse8601(f'{start_date}T00:00:00Z')
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
        else:
            raise ValueError(f"Unsupported data source: {source}")
        
        return df


def create_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_path: Path to training data CSV
        val_path: Path to validation data CSV
        batch_size: Batch size
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for MarketDataset
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = MarketDataset(train_path, **dataset_kwargs)
    val_dataset = MarketDataset(val_path, **dataset_kwargs)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


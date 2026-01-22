"""
Data Preprocessing and Feature Engineering Module

Transforms raw market data into features suitable for the diffusion policy model.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import talib
from loguru import logger


class MarketFeatureEngineer:
    """Engineer features from raw market data."""
    
    def __init__(
        self,
        lookback_window: int = 168,
        technical_indicators: Optional[List[str]] = None
    ):
        """
        Initialize feature engineer.
        
        Args:
            lookback_window: Number of historical periods to include
            technical_indicators: List of technical indicators to compute
        """
        self.lookback_window = lookback_window
        self.technical_indicators = technical_indicators or [
            "RSI", "MACD", "Bollinger_Bands", "ATR", "Volume_Profile",
            "EMA_20", "EMA_50", "EMA_200"
        ]
        self.scaler = StandardScaler()
        self.fitted = False
    
    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Price-based indicators
        if "RSI" in self.technical_indicators:
            df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
            df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
        
        if "MACD" in self.technical_indicators:
            macd, signal, hist = talib.MACD(df['close'])
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
        
        if "Bollinger_Bands" in self.technical_indicators:
            upper, middle, lower = talib.BBANDS(df['close'])
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle
            df['bb_position'] = (df['close'] - lower) / (upper - lower)
        
        if "ATR" in self.technical_indicators:
            df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            df['atr_7'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=7)
        
        # Moving averages
        if "EMA_20" in self.technical_indicators:
            df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        
        if "EMA_50" in self.technical_indicators:
            df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        
        if "EMA_200" in self.technical_indicators:
            df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        
        # Volume indicators
        if "Volume_Profile" in self.technical_indicators:
            df['volume_sma_20'] = talib.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Momentum indicators
        df['momentum_10'] = talib.MOM(df['close'], timeperiod=10)
        df['roc_10'] = talib.ROC(df['close'], timeperiod=10)
        
        # Volatility
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
        
        # Price patterns
        df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        
        # Returns
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_20'] = df['close'].pct_change(20)
        
        # High-low range
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        
        logger.info(f"Computed {len(df.columns) - 6} technical indicators")
        return df
    
    def create_market_state(
        self,
        df: pd.DataFrame,
        current_idx: int
    ) -> np.ndarray:
        """
        Create market state vector for a given timestamp.
        
        Args:
            df: DataFrame with features
            current_idx: Current index in the DataFrame
        
        Returns:
            Market state vector
        """
        start_idx = max(0, current_idx - self.lookback_window)
        
        # Get historical window
        window = df.iloc[start_idx:current_idx + 1]
        
        if len(window) < self.lookback_window:
            # Pad if not enough history
            padding = self.lookback_window - len(window)
            window = pd.concat([
                pd.DataFrame(np.zeros((padding, len(window.columns))), columns=window.columns),
                window
            ])
        
        # Select feature columns (exclude OHLCV)
        feature_cols = [col for col in window.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Extract features
        features = window[feature_cols].values
        
        # Flatten to 1D vector
        state = features.flatten()
        
        # Handle NaN values
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        return state
    
    def fit_scaler(self, df: pd.DataFrame):
        """Fit the scaler on training data."""
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        self.scaler.fit(df[feature_cols].fillna(0))
        self.fitted = True
        logger.info("Fitted feature scaler")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        df = df.copy()
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        df[feature_cols] = self.scaler.transform(df[feature_cols].fillna(0))
        return df
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = False
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw OHLCV DataFrame
            fit_scaler: Whether to fit the scaler
        
        Returns:
            Processed DataFrame with features
        """
        # Compute technical indicators
        df = self.compute_technical_indicators(df)
        
        # Drop NaN rows (from indicator computation)
        df = df.dropna()
        
        # Fit or transform scaler
        if fit_scaler:
            self.fit_scaler(df)
        
        if self.fitted:
            df = self.transform(df)
        
        logger.info(f"Processed {len(df)} rows with {len(df.columns)} features")
        return df


class ActionNormalizer:
    """Normalize and denormalize trading actions."""
    
    def __init__(self):
        """
        Action space:
        - position_size: [-1, 1] where -1 is max short, 1 is max long, 0 is no position
        - entry_price_offset: [-0.05, 0.05] relative to current price
        - stop_loss_pct: [0.01, 0.10] percentage stop loss
        """
        self.action_bounds = {
            'position_size': (-1.0, 1.0),
            'entry_price_offset': (-0.05, 0.05),
            'stop_loss_pct': (0.01, 0.10)
        }
    
    def normalize_action(self, action: Dict[str, float]) -> np.ndarray:
        """
        Normalize action to [-1, 1] range.
        
        Args:
            action: Dictionary with action components
        
        Returns:
            Normalized action array
        """
        normalized = []
        
        for key in ['position_size', 'entry_price_offset', 'stop_loss_pct']:
            value = action[key]
            min_val, max_val = self.action_bounds[key]
            
            # Normalize to [-1, 1]
            norm_value = 2 * (value - min_val) / (max_val - min_val) - 1
            normalized.append(norm_value)
        
        return np.array(normalized, dtype=np.float32)
    
    def denormalize_action(self, normalized_action: np.ndarray) -> Dict[str, float]:
        """
        Denormalize action from [-1, 1] range.
        
        Args:
            normalized_action: Normalized action array
        
        Returns:
            Dictionary with denormalized action components
        """
        action = {}
        keys = ['position_size', 'entry_price_offset', 'stop_loss_pct']
        
        for i, key in enumerate(keys):
            norm_value = np.clip(normalized_action[i], -1, 1)
            min_val, max_val = self.action_bounds[key]
            
            # Denormalize from [-1, 1]
            value = (norm_value + 1) / 2 * (max_val - min_val) + min_val
            action[key] = float(value)
        
        return action
    
    def clip_action(self, action: Dict[str, float]) -> Dict[str, float]:
        """Clip action to valid bounds."""
        clipped = {}
        
        for key, value in action.items():
            if key in self.action_bounds:
                min_val, max_val = self.action_bounds[key]
                clipped[key] = np.clip(value, min_val, max_val)
            else:
                clipped[key] = value
        
        return clipped


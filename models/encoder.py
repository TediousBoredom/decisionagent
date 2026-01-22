"""
Market State Encoder

Encodes market data into latent representations for conditioning the diffusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class TemporalConvBlock(nn.Module):
    """Temporal convolutional block for time series processing."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, time]
        Returns:
            [batch, channels, time]
        """
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        return out + residual


class MarketStateEncoder(nn.Module):
    """
    Encodes market state from multiple data sources:
    - Price/volume time series
    - Technical indicators
    - Order book features
    - Market regime indicators
    """
    
    def __init__(
        self,
        price_dim: int = 5,  # OHLCV
        indicator_dim: int = 20,  # Technical indicators
        orderbook_dim: int = 10,  # Order book features
        regime_dim: int = 5,  # Market regime features
        seq_length: int = 100,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.price_dim = price_dim
        self.indicator_dim = indicator_dim
        self.orderbook_dim = orderbook_dim
        self.regime_dim = regime_dim
        self.seq_length = seq_length
        
        # Price encoder (temporal convolutions)
        self.price_encoder = nn.Sequential(
            nn.Conv1d(price_dim, hidden_dim, 1),
            *[TemporalConvBlock(hidden_dim, hidden_dim, kernel_size=3, 
                               dilation=2**i, dropout=dropout)
              for i in range(num_layers)]
        )
        
        # Indicator encoder
        self.indicator_encoder = nn.Sequential(
            nn.Linear(indicator_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Order book encoder
        self.orderbook_encoder = nn.Sequential(
            nn.Linear(orderbook_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Regime encoder
        self.regime_encoder = nn.Sequential(
            nn.Linear(regime_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Attention pooling for temporal features
        self.attention_pool = nn.MultiheadAttention(hidden_dim, num_heads=8, 
                                                    dropout=dropout, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Fusion layer
        total_dim = hidden_dim + hidden_dim + hidden_dim // 2 + hidden_dim // 2
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, price: torch.Tensor, indicators: torch.Tensor,
                orderbook: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        """
        Args:
            price: [batch, seq_length, price_dim] - OHLCV data
            indicators: [batch, seq_length, indicator_dim] - Technical indicators
            orderbook: [batch, orderbook_dim] - Current order book state
            regime: [batch, regime_dim] - Market regime features
        
        Returns:
            Market state embedding [batch, output_dim]
        """
        batch_size = price.shape[0]
        
        # Encode price time series
        price_features = price.transpose(1, 2)  # [batch, price_dim, seq_length]
        price_features = self.price_encoder(price_features)  # [batch, hidden_dim, seq_length]
        price_features = price_features.transpose(1, 2)  # [batch, seq_length, hidden_dim]
        
        # Attention pooling over time
        query = self.query.expand(batch_size, -1, -1)
        price_pooled, _ = self.attention_pool(query, price_features, price_features)
        price_pooled = price_pooled.squeeze(1)  # [batch, hidden_dim]
        
        # Encode indicators (use last timestep)
        indicator_features = self.indicator_encoder(indicators[:, -1, :])  # [batch, hidden_dim]
        
        # Encode order book
        orderbook_features = self.orderbook_encoder(orderbook)  # [batch, hidden_dim // 2]
        
        # Encode regime
        regime_features = self.regime_encoder(regime)  # [batch, hidden_dim // 2]
        
        # Concatenate all features
        combined = torch.cat([
            price_pooled,
            indicator_features,
            orderbook_features,
            regime_features
        ], dim=-1)
        
        # Fusion
        state_embedding = self.fusion(combined)
        
        return state_embedding


class MarketRegimeClassifier(nn.Module):
    """
    Classifies market regime (trend, range-bound, high volatility, etc.)
    Used as auxiliary task and for conditioning.
    """
    
    def __init__(self, input_dim: int = 128, num_regimes: int = 5):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_regimes)
        )
        
    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_embedding: [batch, input_dim]
        
        Returns:
            Regime logits [batch, num_regimes]
        """
        return self.classifier(state_embedding)


class VolatilityPredictor(nn.Module):
    """
    Predicts future volatility for risk assessment.
    """
    
    def __init__(self, input_dim: int = 128, horizon: int = 20):
        super().__init__()
        
        self.horizon = horizon
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, horizon)
        )
        
    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_embedding: [batch, input_dim]
        
        Returns:
            Predicted volatility [batch, horizon]
        """
        return F.softplus(self.predictor(state_embedding))


class MultiScaleEncoder(nn.Module):
    """
    Encodes market data at multiple time scales (e.g., 1min, 5min, 1hour).
    """
    
    def __init__(
        self,
        feature_dim: int,
        scales: List[int] = [1, 5, 15, 60],  # Time scales in minutes
        hidden_dim: int = 128,
        output_dim: int = 128
    ):
        super().__init__()
        
        self.scales = scales
        
        # Encoder for each scale
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in scales
        ])
        
        # Cross-scale attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * len(scales), hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, features: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: Dict mapping scale to features [batch, feature_dim]
        
        Returns:
            Multi-scale embedding [batch, output_dim]
        """
        scale_embeddings = []
        
        for scale, encoder in zip(self.scales, self.scale_encoders):
            if scale in features:
                emb = encoder(features[scale])
                scale_embeddings.append(emb)
        
        # Stack embeddings
        stacked = torch.stack(scale_embeddings, dim=1)  # [batch, num_scales, hidden_dim]
        
        # Apply cross-scale attention
        attended, _ = self.cross_attention(stacked, stacked, stacked)
        
        # Flatten and project
        flattened = attended.flatten(1)
        output = self.output_proj(flattened)
        
        return output


class FeatureExtractor(nn.Module):
    """
    Extracts and normalizes features from raw market data.
    """
    
    def __init__(self):
        super().__init__()
        
    def extract_price_features(self, ohlcv: torch.Tensor) -> torch.Tensor:
        """
        Extract features from OHLCV data.
        
        Args:
            ohlcv: [batch, seq_length, 5] - Open, High, Low, Close, Volume
        
        Returns:
            Features [batch, seq_length, feature_dim]
        """
        open_price = ohlcv[..., 0:1]
        high = ohlcv[..., 1:2]
        low = ohlcv[..., 2:3]
        close = ohlcv[..., 3:4]
        volume = ohlcv[..., 4:5]
        
        # Returns
        returns = (close - open_price) / (open_price + 1e-8)
        
        # High-Low range
        hl_range = (high - low) / (close + 1e-8)
        
        # Volume change
        volume_change = torch.diff(volume, dim=1, prepend=volume[:, :1])
        volume_change = volume_change / (volume + 1e-8)
        
        # Log volume
        log_volume = torch.log(volume + 1)
        
        # Combine features
        features = torch.cat([
            returns,
            hl_range,
            volume_change,
            log_volume,
            close / (close.mean(dim=1, keepdim=True) + 1e-8)  # Normalized price
        ], dim=-1)
        
        return features
    
    def extract_technical_indicators(self, close: torch.Tensor, 
                                     volume: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate technical indicators.
        
        Args:
            close: [batch, seq_length] - Close prices
            volume: [batch, seq_length] - Volume
        
        Returns:
            Dictionary of indicators
        """
        indicators = {}
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            ma = self._moving_average(close, window)
            indicators[f'ma_{window}'] = ma
            indicators[f'price_to_ma_{window}'] = close / (ma + 1e-8)
        
        # RSI
        indicators['rsi'] = self._rsi(close, period=14)
        
        # MACD
        macd, signal = self._macd(close)
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_hist'] = macd - signal
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(close, period=20)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        indicators['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        # ATR (Average True Range)
        indicators['atr'] = self._atr(close, period=14)
        
        return indicators
    
    def _moving_average(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """Calculate moving average."""
        kernel = torch.ones(1, 1, window, device=x.device) / window
        x_padded = F.pad(x.unsqueeze(1), (window - 1, 0), mode='replicate')
        ma = F.conv1d(x_padded, kernel).squeeze(1)
        return ma
    
    def _rsi(self, close: torch.Tensor, period: int = 14) -> torch.Tensor:
        """Calculate RSI."""
        delta = torch.diff(close, dim=1, prepend=close[:, :1])
        gain = torch.where(delta > 0, delta, torch.zeros_like(delta))
        loss = torch.where(delta < 0, -delta, torch.zeros_like(delta))
        
        avg_gain = self._moving_average(gain, period)
        avg_loss = self._moving_average(loss, period)
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _macd(self, close: torch.Tensor, fast: int = 12, 
              slow: int = 26, signal: int = 9) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate MACD."""
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        macd = ema_fast - ema_slow
        signal_line = self._ema(macd, signal)
        return macd, signal_line
    
    def _ema(self, x: torch.Tensor, period: int) -> torch.Tensor:
        """Calculate exponential moving average."""
        alpha = 2 / (period + 1)
        ema = torch.zeros_like(x)
        ema[:, 0] = x[:, 0]
        
        for t in range(1, x.shape[1]):
            ema[:, t] = alpha * x[:, t] + (1 - alpha) * ema[:, t - 1]
        
        return ema
    
    def _bollinger_bands(self, close: torch.Tensor, period: int = 20, 
                        num_std: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate Bollinger Bands."""
        ma = self._moving_average(close, period)
        
        # Calculate rolling std
        close_expanded = close.unsqueeze(1)
        kernel = torch.ones(1, 1, period, device=close.device) / period
        close_padded = F.pad(close_expanded, (period - 1, 0), mode='replicate')
        
        mean = F.conv1d(close_padded, kernel).squeeze(1)
        sq_diff = (close - mean) ** 2
        sq_diff_padded = F.pad(sq_diff.unsqueeze(1), (period - 1, 0), mode='replicate')
        variance = F.conv1d(sq_diff_padded, kernel).squeeze(1)
        std = torch.sqrt(variance + 1e-8)
        
        upper = ma + num_std * std
        lower = ma - num_std * std
        
        return upper, ma, lower
    
    def _atr(self, close: torch.Tensor, period: int = 14) -> torch.Tensor:
        """Calculate Average True Range (simplified version)."""
        high_low = torch.abs(torch.diff(close, dim=1, prepend=close[:, :1]))
        atr = self._moving_average(high_low, period)
        return atr


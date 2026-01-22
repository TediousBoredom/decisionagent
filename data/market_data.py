"""
Market Data Collection and Processing Module

Handles real-time and historical market data collection from various exchanges.
"""

import ccxt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from loguru import logger


class MarketDataCollector:
    """Collects and manages market data from exchanges."""
    
    def __init__(
        self,
        exchange_name: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True
    ):
        """
        Initialize market data collector.
        
        Args:
            exchange_name: Name of the exchange (binance, okx, etc.)
            api_key: API key for authenticated requests
            api_secret: API secret
            testnet: Whether to use testnet
        """
        self.exchange_name = exchange_name
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_name)
        config = {
            'enableRateLimit': True,
        }
        
        if api_key and api_secret:
            config['apiKey'] = api_key
            config['secret'] = api_secret
        
        if testnet and hasattr(exchange_class, 'set_sandbox_mode'):
            config['sandbox'] = True
        
        self.exchange = exchange_class(config)
        logger.info(f"Initialized {exchange_name} exchange (testnet={testnet})")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            since: Start datetime
            limit: Number of candles to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if since:
                since_ms = int(since.timestamp() * 1000)
            else:
                since_ms = None
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            raise
    
    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical data for a date range.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with historical OHLCV data
        """
        all_data = []
        current_date = start_date
        
        # Calculate timeframe in milliseconds
        timeframe_ms = self._timeframe_to_ms(timeframe)
        limit = 1000
        
        while current_date < end_date:
            try:
                df = self.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_date,
                    limit=limit
                )
                
                if df.empty:
                    break
                
                all_data.append(df)
                
                # Move to next batch
                current_date = df.index[-1].to_pydatetime() + timedelta(milliseconds=timeframe_ms)
                
                # Rate limiting
                asyncio.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                logger.error(f"Error fetching historical data: {e}")
                break
        
        if all_data:
            result = pd.concat(all_data)
            result = result[~result.index.duplicated(keep='first')]
            result = result.sort_index()
            result = result[result.index <= end_date]
            logger.info(f"Fetched {len(result)} historical candles for {symbol}")
            return result
        else:
            return pd.DataFrame()
    
    def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch current ticker data."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise
    
    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Fetch order book data."""
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=limit)
            return orderbook
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            raise
    
    def fetch_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Fetch recent trades."""
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            return trades
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            raise
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        unit = timeframe[-1]
        amount = int(timeframe[:-1])
        
        units = {
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
            'w': 7 * 24 * 60 * 60 * 1000,
        }
        
        return amount * units[unit]
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols."""
        try:
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            logger.error(f"Error fetching available symbols: {e}")
            raise


class MarketDataStream:
    """Real-time market data streaming."""
    
    def __init__(self, collector: MarketDataCollector):
        self.collector = collector
        self.subscribers = {}
    
    async def subscribe(self, symbol: str, callback):
        """Subscribe to real-time data for a symbol."""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        self.subscribers[symbol].append(callback)
        logger.info(f"Subscribed to {symbol}")
    
    async def start_streaming(self, symbols: List[str], timeframe: str = "1m"):
        """Start streaming market data."""
        logger.info(f"Starting market data stream for {symbols}")
        
        while True:
            for symbol in symbols:
                try:
                    # Fetch latest data
                    df = self.collector.fetch_ohlcv(symbol, timeframe, limit=1)
                    
                    if not df.empty and symbol in self.subscribers:
                        # Notify subscribers
                        for callback in self.subscribers[symbol]:
                            await callback(symbol, df)
                    
                except Exception as e:
                    logger.error(f"Error streaming {symbol}: {e}")
            
            # Wait before next update
            await asyncio.sleep(60)  # Update every minute


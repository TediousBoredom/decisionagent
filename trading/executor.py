"""
Trading Execution Engine

Handles order execution and interaction with exchanges.
"""

import ccxt
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from loguru import logger


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    status: str = "pending"
    filled_amount: float = 0.0
    average_price: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ExchangeAdapter:
    """Adapter for exchange interactions."""
    
    def __init__(
        self,
        exchange_name: str,
        api_key: str,
        api_secret: str,
        testnet: bool = True
    ):
        """
        Initialize exchange adapter.
        
        Args:
            exchange_name: Name of the exchange
            api_key: API key
            api_secret: API secret
            testnet: Whether to use testnet
        """
        self.exchange_name = exchange_name
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_name)
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Use futures for leverage
            }
        }
        
        if testnet:
            config['sandbox'] = True
        
        self.exchange = exchange_class(config)
        
        logger.info(f"Initialized {exchange_name} adapter (testnet={testnet})")
    
    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float
    ) -> Optional[Order]:
        """
        Create a market order.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            amount: Order amount
        
        Returns:
            Order object if successful
        """
        try:
            order_result = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side.value,
                amount=abs(amount)
            )
            
            order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                amount=amount,
                order_id=order_result['id'],
                status=order_result['status'],
                filled_amount=order_result.get('filled', 0),
                average_price=order_result.get('average', 0)
            )
            
            logger.info(f"Created market order: {side.value} {amount} {symbol}")
            return order
            
        except Exception as e:
            logger.error(f"Error creating market order: {e}")
            return None
    
    async def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float
    ) -> Optional[Order]:
        """Create a limit order."""
        try:
            order_result = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side.value,
                amount=abs(amount),
                price=price
            )
            
            order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                amount=amount,
                price=price,
                order_id=order_result['id'],
                status=order_result['status']
            )
            
            logger.info(f"Created limit order: {side.value} {amount} {symbol} @ ${price}")
            return order
            
        except Exception as e:
            logger.error(f"Error creating limit order: {e}")
            return None
    
    async def create_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        stop_price: float
    ) -> Optional[Order]:
        """Create a stop loss order."""
        try:
            order_result = self.exchange.create_order(
                symbol=symbol,
                type='stop_market',
                side=side.value,
                amount=abs(amount),
                params={'stopPrice': stop_price}
            )
            
            order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.STOP_LOSS,
                amount=amount,
                stop_price=stop_price,
                order_id=order_result['id'],
                status=order_result['status']
            )
            
            logger.info(f"Created stop loss order: {side.value} {amount} {symbol} @ ${stop_price}")
            return order
            
        except Exception as e:
            logger.error(f"Error creating stop loss order: {e}")
            return None
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Cancelled order {order_id} for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Dict]:
        """Get order status."""
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            logger.error(f"Error fetching order status: {e}")
            return None
    
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        try:
            balance = self.exchange.fetch_balance()
            return balance['total']
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {}
    
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol."""
        try:
            positions = self.exchange.fetch_positions([symbol])
            if positions:
                return positions[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching position: {e}")
            return None


class TradingExecutor:
    """Main trading execution engine."""
    
    def __init__(
        self,
        exchange_adapter: ExchangeAdapter,
        max_slippage: float = 0.001
    ):
        """
        Initialize trading executor.
        
        Args:
            exchange_adapter: Exchange adapter instance
            max_slippage: Maximum acceptable slippage
        """
        self.exchange = exchange_adapter
        self.max_slippage = max_slippage
        self.pending_orders: Dict[str, Order] = {}
        self.executed_orders: List[Order] = []
        
        logger.info("Initialized TradingExecutor")
    
    async def execute_trade(
        self,
        symbol: str,
        position_size: float,
        entry_price: float,
        stop_loss: float,
        order_type: OrderType = OrderType.MARKET
    ) -> Optional[Order]:
        """
        Execute a trade.
        
        Args:
            symbol: Trading symbol
            position_size: Position size (positive for long, negative for short)
            entry_price: Desired entry price
            stop_loss: Stop loss price
            order_type: Type of order to use
        
        Returns:
            Order object if successful
        """
        # Determine side
        side = OrderSide.BUY if position_size > 0 else OrderSide.SELL
        amount = abs(position_size)
        
        # Create entry order
        if order_type == OrderType.MARKET:
            order = await self.exchange.create_market_order(symbol, side, amount)
        elif order_type == OrderType.LIMIT:
            order = await self.exchange.create_limit_order(symbol, side, amount, entry_price)
        else:
            logger.error(f"Unsupported order type: {order_type}")
            return None
        
        if order is None:
            return None
        
        # Wait for order to fill (for market orders)
        if order_type == OrderType.MARKET:
            await asyncio.sleep(1)  # Give time for order to fill
            order_status = await self.exchange.get_order_status(order.order_id, symbol)
            
            if order_status:
                order.status = order_status['status']
                order.filled_amount = order_status.get('filled', 0)
                order.average_price = order_status.get('average', 0)
        
        # Create stop loss order
        if order.status == 'closed' or order.status == 'filled':
            stop_side = OrderSide.SELL if position_size > 0 else OrderSide.BUY
            stop_order = await self.exchange.create_stop_loss_order(
                symbol, stop_side, amount, stop_loss
            )
            
            if stop_order:
                self.pending_orders[f"{symbol}_stop"] = stop_order
        
        # Record order
        self.executed_orders.append(order)
        
        return order
    
    async def close_position(
        self,
        symbol: str,
        position_size: float,
        order_type: OrderType = OrderType.MARKET
    ) -> Optional[Order]:
        """
        Close a position.
        
        Args:
            symbol: Trading symbol
            position_size: Current position size
            order_type: Order type to use
        
        Returns:
            Order object if successful
        """
        # Cancel any pending stop loss orders
        stop_key = f"{symbol}_stop"
        if stop_key in self.pending_orders:
            stop_order = self.pending_orders[stop_key]
            await self.exchange.cancel_order(stop_order.order_id, symbol)
            del self.pending_orders[stop_key]
        
        # Create closing order (opposite side)
        side = OrderSide.SELL if position_size > 0 else OrderSide.BUY
        amount = abs(position_size)
        
        order = await self.exchange.create_market_order(symbol, side, amount)
        
        if order:
            self.executed_orders.append(order)
            logger.info(f"Closed position: {symbol}")
        
        return order
    
    async def update_stop_loss(
        self,
        symbol: str,
        position_size: float,
        new_stop_loss: float
    ) -> bool:
        """
        Update stop loss for a position (trailing stop).
        
        Args:
            symbol: Trading symbol
            position_size: Current position size
            new_stop_loss: New stop loss price
        
        Returns:
            True if successful
        """
        # Cancel old stop loss
        stop_key = f"{symbol}_stop"
        if stop_key in self.pending_orders:
            old_stop = self.pending_orders[stop_key]
            await self.exchange.cancel_order(old_stop.order_id, symbol)
        
        # Create new stop loss
        side = OrderSide.SELL if position_size > 0 else OrderSide.BUY
        amount = abs(position_size)
        
        stop_order = await self.exchange.create_stop_loss_order(
            symbol, side, amount, new_stop_loss
        )
        
        if stop_order:
            self.pending_orders[stop_key] = stop_order
            logger.info(f"Updated stop loss for {symbol}: ${new_stop_loss:.2f}")
            return True
        
        return False
    
    def get_execution_summary(self) -> Dict:
        """Get summary of executed orders."""
        return {
            'total_orders': len(self.executed_orders),
            'pending_orders': len(self.pending_orders),
            'recent_orders': [
                {
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'amount': order.amount,
                    'price': order.average_price,
                    'status': order.status
                }
                for order in self.executed_orders[-10:]
            ]
        }


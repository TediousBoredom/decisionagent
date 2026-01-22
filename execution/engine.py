"""
Trading Execution Engine

Handles order execution, position management, and interaction with exchanges.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_price(self, price: float):
        """Update current price and unrealized PnL."""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity


class ExecutionEngine:
    """
    Main execution engine for trading.
    Manages orders, positions, and broker interactions.
    """
    
    def __init__(
        self,
        broker,
        initial_capital: float = 100000.0,
        max_slippage: float = 0.001,
        commission_rate: float = 0.001,
        max_order_size: float = 10000.0
    ):
        self.broker = broker
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_slippage = max_slippage
        self.commission_rate = commission_rate
        self.max_order_size = max_order_size
        
        # State
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.trade_history: List[Dict] = []
        
        # Performance tracking
        self.equity_curve = [initial_capital]
        self.returns = []
        self.drawdowns = []
        
        # Order ID counter
        self.order_counter = 0
        
    def get_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        return f"order_{self.order_counter}_{int(time.time())}"
    
    def execute_action(
        self,
        symbol: str,
        action: np.ndarray,
        current_price: float,
        market_data: Optional[Dict] = None
    ) -> List[Order]:
        """
        Execute trading action.
        
        Args:
            symbol: Trading symbol
            action: [position, urgency, stop_loss, take_profit]
            current_price: Current market price
            market_data: Additional market data
        
        Returns:
            List of created orders
        """
        target_position = action[0]  # [-1, 1]
        urgency = action[1]  # [0, 1]
        stop_loss_pct = action[2]  # [0, 1]
        take_profit_pct = action[3]  # [0, 1]
        
        # Get current position
        current_position = self.positions.get(symbol)
        current_quantity = current_position.quantity if current_position else 0.0
        
        # Calculate target quantity in dollars
        target_value = target_position * self.capital
        target_quantity = target_value / current_price
        
        # Calculate quantity to trade
        quantity_to_trade = target_quantity - current_quantity
        
        if abs(quantity_to_trade) < 0.01:  # Minimum trade size
            return []
        
        # Determine order type based on urgency
        if urgency > 0.7:
            order_type = OrderType.MARKET
            price = None
        else:
            order_type = OrderType.LIMIT
            # Place limit order with some buffer
            if quantity_to_trade > 0:
                price = current_price * (1 - 0.001 * (1 - urgency))
            else:
                price = current_price * (1 + 0.001 * (1 - urgency))
        
        # Create main order
        side = OrderSide.BUY if quantity_to_trade > 0 else OrderSide.SELL
        
        main_order = Order(
            order_id=self.get_order_id(),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=abs(quantity_to_trade),
            price=price
        )
        
        orders = [main_order]
        
        # Add stop loss order if specified
        if stop_loss_pct > 0.01 and target_quantity != 0:
            if target_quantity > 0:  # Long position
                stop_price = current_price * (1 - stop_loss_pct * 0.1)
                stop_side = OrderSide.SELL
            else:  # Short position
                stop_price = current_price * (1 + stop_loss_pct * 0.1)
                stop_side = OrderSide.BUY
            
            stop_order = Order(
                order_id=self.get_order_id(),
                symbol=symbol,
                side=stop_side,
                order_type=OrderType.STOP_LOSS,
                quantity=abs(target_quantity),
                stop_price=stop_price
            )
            orders.append(stop_order)
        
        # Add take profit order if specified
        if take_profit_pct > 0.01 and target_quantity != 0:
            if target_quantity > 0:  # Long position
                tp_price = current_price * (1 + take_profit_pct * 0.2)
                tp_side = OrderSide.SELL
            else:  # Short position
                tp_price = current_price * (1 - take_profit_pct * 0.2)
                tp_side = OrderSide.BUY
            
            tp_order = Order(
                order_id=self.get_order_id(),
                symbol=symbol,
                side=tp_side,
                order_type=OrderType.TAKE_PROFIT,
                quantity=abs(target_quantity),
                price=tp_price
            )
            orders.append(tp_order)
        
        # Submit orders
        for order in orders:
            self.submit_order(order)
        
        return orders
    
    def submit_order(self, order: Order) -> bool:
        """Submit order to broker."""
        try:
            # Check if we have enough capital
            if order.side == OrderSide.BUY:
                required_capital = order.quantity * (order.price or 0)
                if required_capital > self.capital * 0.95:  # Keep 5% buffer
                    order.status = OrderStatus.REJECTED
                    return False
            
            # Submit to broker
            success = self.broker.submit_order(order)
            
            if success:
                self.pending_orders[order.order_id] = order
                return True
            else:
                order.status = OrderStatus.REJECTED
                return False
                
        except Exception as e:
            print(f"Error submitting order: {e}")
            order.status = OrderStatus.REJECTED
            return False
    
    def update_order_status(self, order_id: str, status: OrderStatus,
                           filled_quantity: float = 0.0,
                           fill_price: float = 0.0):
        """Update order status after execution."""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = status
            order.filled_quantity = filled_quantity
            order.average_fill_price = fill_price
            
            if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                # Move to history
                self.order_history.append(order)
                del self.pending_orders[order_id]
                
                # Update position if filled
                if status == OrderStatus.FILLED:
                    self._update_position(order)
    
    def _update_position(self, order: Order):
        """Update position after order fill."""
        symbol = order.symbol
        
        # Calculate commission
        commission = order.filled_quantity * order.average_fill_price * self.commission_rate
        
        if symbol not in self.positions:
            # New position
            if order.side == OrderSide.BUY:
                quantity = order.filled_quantity
            else:
                quantity = -order.filled_quantity
            
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=order.average_fill_price,
                current_price=order.average_fill_price
            )
        else:
            # Update existing position
            position = self.positions[symbol]
            
            if order.side == OrderSide.BUY:
                new_quantity = position.quantity + order.filled_quantity
            else:
                new_quantity = position.quantity - order.filled_quantity
            
            # Calculate realized PnL if closing/reducing position
            if (position.quantity > 0 and order.side == OrderSide.SELL) or \
               (position.quantity < 0 and order.side == OrderSide.BUY):
                closed_quantity = min(abs(position.quantity), order.filled_quantity)
                pnl = closed_quantity * (order.average_fill_price - position.entry_price)
                if position.quantity < 0:
                    pnl = -pnl
                position.realized_pnl += pnl - commission
                self.capital += pnl - commission
            
            # Update position
            if abs(new_quantity) < 0.01:
                # Position closed
                del self.positions[symbol]
            else:
                # Update entry price if adding to position
                if (position.quantity > 0 and order.side == OrderSide.BUY) or \
                   (position.quantity < 0 and order.side == OrderSide.SELL):
                    total_cost = position.quantity * position.entry_price + \
                                order.filled_quantity * order.average_fill_price
                    position.entry_price = total_cost / new_quantity
                
                position.quantity = new_quantity
        
        # Record trade
        self.trade_history.append({
            'timestamp': order.timestamp,
            'symbol': symbol,
            'side': order.side.value,
            'quantity': order.filled_quantity,
            'price': order.average_fill_price,
            'commission': commission
        })
    
    def update_positions(self, prices: Dict[str, float]):
        """Update all positions with current prices."""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])
        
        # Update equity
        total_equity = self.capital
        for position in self.positions.values():
            total_equity += position.unrealized_pnl
        
        self.equity_curve.append(total_equity)
        
        # Calculate return
        if len(self.equity_curve) > 1:
            ret = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            self.returns.append(ret)
        
        # Calculate drawdown
        peak = max(self.equity_curve)
        drawdown = (total_equity - peak) / peak
        self.drawdowns.append(drawdown)
    
    def get_portfolio_state(self) -> Dict:
        """Get current portfolio state."""
        total_equity = self.capital
        total_unrealized_pnl = 0.0
        total_realized_pnl = 0.0
        
        for position in self.positions.values():
            total_unrealized_pnl += position.unrealized_pnl
            total_realized_pnl += position.realized_pnl
        
        total_equity += total_unrealized_pnl
        
        return {
            'capital': self.capital,
            'total_equity': total_equity,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'total_return': (total_equity - self.initial_capital) / self.initial_capital,
            'num_positions': len(self.positions),
            'num_pending_orders': len(self.pending_orders),
            'current_drawdown': self.drawdowns[-1] if self.drawdowns else 0.0
        }
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if len(self.returns) < 2:
            return {}
        
        returns = np.array(self.returns)
        
        metrics = {
            'total_return': (self.equity_curve[-1] - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': returns.mean() / (returns.std() + 1e-8) * np.sqrt(252),
            'max_drawdown': min(self.drawdowns) if self.drawdowns else 0.0,
            'win_rate': (returns > 0).sum() / len(returns),
            'avg_win': returns[returns > 0].mean() if (returns > 0).any() else 0.0,
            'avg_loss': returns[returns < 0].mean() if (returns < 0).any() else 0.0,
            'num_trades': len(self.trade_history),
            'volatility': returns.std() * np.sqrt(252)
        }
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            metrics['sortino_ratio'] = returns.mean() / (downside_std + 1e-8) * np.sqrt(252)
        
        # Calmar ratio
        if metrics['max_drawdown'] != 0:
            metrics['calmar_ratio'] = metrics['total_return'] / abs(metrics['max_drawdown'])
        
        return metrics
    
    def close_all_positions(self, prices: Dict[str, float]):
        """Close all open positions."""
        for symbol, position in list(self.positions.items()):
            if symbol in prices:
                quantity = abs(position.quantity)
                side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                
                order = Order(
                    order_id=self.get_order_id(),
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=quantity
                )
                
                self.submit_order(order)
    
    def cancel_all_orders(self):
        """Cancel all pending orders."""
        for order_id in list(self.pending_orders.keys()):
            self.update_order_status(order_id, OrderStatus.CANCELLED)


class SimulatedBroker:
    """
    Simulated broker for backtesting.
    """
    
    def __init__(self, slippage: float = 0.001, latency_ms: float = 10.0):
        self.slippage = slippage
        self.latency_ms = latency_ms
        self.current_prices: Dict[str, float] = {}
        
    def update_prices(self, prices: Dict[str, float]):
        """Update current market prices."""
        self.current_prices = prices
    
    def submit_order(self, order: Order) -> bool:
        """Simulate order submission."""
        if order.symbol not in self.current_prices:
            return False
        
        current_price = self.current_prices[order.symbol]
        
        # Simulate order execution
        if order.order_type == OrderType.MARKET:
            # Market order fills immediately with slippage
            if order.side == OrderSide.BUY:
                fill_price = current_price * (1 + self.slippage)
            else:
                fill_price = current_price * (1 - self.slippage)
            
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = fill_price
            return True
            
        elif order.order_type == OrderType.LIMIT:
            # Limit order fills if price is favorable
            if order.side == OrderSide.BUY and current_price <= order.price:
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.average_fill_price = order.price
                return True
            elif order.side == OrderSide.SELL and current_price >= order.price:
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.average_fill_price = order.price
                return True
            else:
                # Order remains pending
                return True
        
        return False


class RealBrokerInterface:
    """
    Interface for real broker APIs (CCXT, Alpaca, etc.)
    """
    
    def __init__(self, broker_name: str, api_key: str, api_secret: str, testnet: bool = True):
        self.broker_name = broker_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Initialize broker connection
        self._init_broker()
    
    def _init_broker(self):
        """Initialize broker connection."""
        # This would use ccxt or other broker APIs
        # Example:
        # import ccxt
        # self.exchange = ccxt.binance({
        #     'apiKey': self.api_key,
        #     'secret': self.api_secret,
        #     'enableRateLimit': True
        # })
        pass
    
    def submit_order(self, order: Order) -> bool:
        """Submit order to real broker."""
        # Implementation depends on broker API
        pass
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions from broker."""
        pass
    
    def get_balance(self) -> float:
        """Get account balance."""
        pass


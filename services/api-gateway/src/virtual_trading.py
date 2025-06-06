"""
Virtual Trading Engine for executing and managing virtual trades
"""

import asyncio
import aioredis
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import uuid
import time
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class VirtualTrade:
    trade_id: str
    user_id: str
    portfolio_id: str
    symbol: str
    action: str  # buy, sell, close
    quantity: int
    order_type: str  # market, limit, stop
    limit_price: Optional[float]
    stop_price: Optional[float]
    executed_price: float
    total_value: float
    commission: float
    timestamp: datetime
    status: str  # executed, pending, cancelled

@dataclass
class VirtualPosition:
    position_id: str
    portfolio_id: str
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    opened_at: datetime
    last_updated: datetime

class VirtualTradingEngine:
    def __init__(self):
        self.redis = None
        
        # Trading configuration
        self.config = {
            "commission_rate": 0.001,  # 0.1% commission
            "min_commission": 1.0,     # Minimum $1 commission
            "max_commission": 50.0,    # Maximum $50 commission
            "slippage_rate": 0.0005,   # 0.05% slippage
            "max_order_size": 1000000, # Maximum order size
            "trading_hours": {
                "start": "09:30",
                "end": "16:00",
                "timezone": "US/Eastern"
            }
        }
        
        # Order book for pending orders
        self.pending_orders = {}
        
        # Trade execution statistics
        self.execution_stats = {
            "total_trades": 0,
            "total_volume": 0.0,
            "avg_execution_time_ms": 0.0,
            "success_rate": 1.0
        }
        
    async def initialize(self):
        """Initialize the virtual trading engine"""
        try:
            # Initialize Redis connection
            self.redis = aioredis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            
            # Load pending orders
            await self._load_pending_orders()
            
            # Start background tasks
            asyncio.create_task(self._process_pending_orders())
            asyncio.create_task(self._update_execution_stats())
            
            logger.info("Virtual trading engine initialized")
            
        except Exception as e:
            logger.error(f"Virtual trading engine initialization failed: {e}")
            raise
    
    async def close(self):
        """Close the virtual trading engine"""
        if self.redis:
            await self.redis.close()
    
    async def health_check(self) -> str:
        """Check health of virtual trading engine"""
        try:
            if not self.redis:
                return "unhealthy - no redis connection"
            
            # Test Redis connection
            await self.redis.ping()
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Virtual trading health check failed: {e}")
            return "unhealthy"
    
    async def execute_trade(
        self,
        user_id: str,
        portfolio_id: str,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        current_market_price: float = 100.0
    ) -> Dict[str, Any]:
        """Execute a virtual trade"""
        
        try:
            trade_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()
            
            # Validate trade parameters
            self._validate_trade_parameters(
                action, quantity, order_type, limit_price, stop_price, current_market_price
            )
            
            # Calculate execution price based on order type
            executed_price = await self._calculate_execution_price(
                symbol, action, order_type, limit_price, stop_price, current_market_price
            )
            
            # Apply slippage for market orders
            if order_type == "market":
                executed_price = self._apply_slippage(executed_price, action)
            
            # Calculate total value and commission
            total_value = abs(quantity) * executed_price
            commission = self._calculate_commission(total_value)
            
            # Create trade record
            trade = VirtualTrade(
                trade_id=trade_id,
                user_id=user_id,
                portfolio_id=portfolio_id,
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                executed_price=executed_price,
                total_value=total_value,
                commission=commission,
                timestamp=timestamp,
                status="executed" if order_type == "market" else "pending"
            )
            
            # Execute immediately for market orders
            if order_type == "market":
                await self._execute_market_order(trade)
            else:
                await self._queue_pending_order(trade)
            
            # Update execution statistics
            await self._update_trade_stats(trade)
            
            return {
                "trade_id": trade_id,
                "executed_price": executed_price,
                "total_value": total_value,
                "commission": commission,
                "timestamp": timestamp.isoformat(),
                "status": trade.status
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            raise
    
    async def get_trading_history(
        self,
        portfolio_id: str,
        limit: int = 100,
        offset: int = 0,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get trading history for a portfolio"""
        
        try:
            # Build Redis key pattern
            key_pattern = f"trades:{portfolio_id}:*"
            
            # Get trade keys
            trade_keys = await self.redis.keys(key_pattern)
            
            # Sort by timestamp (newest first)
            trade_keys.sort(reverse=True)
            
            # Apply pagination
            paginated_keys = trade_keys[offset:offset + limit]
            
            trades = []
            for key in paginated_keys:
                trade_data = await self.redis.get(key)
                if trade_data:
                    trade = json.loads(trade_data)
                    
                    # Filter by symbol if specified
                    if symbol and trade.get("symbol") != symbol:
                        continue
                    
                    trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"Trading history retrieval failed: {e}")
            return []
    
    async def get_position(
        self,
        portfolio_id: str,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Get current position for a symbol in portfolio"""
        
        try:
            position_key = f"positions:{portfolio_id}:{symbol}"
            position_data = await self.redis.get(position_key)
            
            if position_data:
                return json.loads(position_data)
            return None
            
        except Exception as e:
            logger.error(f"Position retrieval failed: {e}")
            return None
    
    async def get_all_positions(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Get all positions for a portfolio"""
        
        try:
            position_keys = await self.redis.keys(f"positions:{portfolio_id}:*")
            positions = []
            
            for key in position_keys:
                position_data = await self.redis.get(key)
                if position_data:
                    positions.append(json.loads(position_data))
            
            return positions
            
        except Exception as e:
            logger.error(f"All positions retrieval failed: {e}")
            return []
    
    async def update_position_prices(
        self,
        portfolio_id: str,
        price_updates: Dict[str, float]
    ):
        """Update position prices with current market data"""
        
        try:
            for symbol, current_price in price_updates.items():
                position = await self.get_position(portfolio_id, symbol)
                
                if position:
                    # Update position with current price
                    position["current_price"] = current_price
                    position["market_value"] = position["quantity"] * current_price
                    position["unrealized_pnl"] = (current_price - position["avg_cost"]) * position["quantity"]
                    position["last_updated"] = datetime.utcnow().isoformat()
                    
                    # Save updated position
                    await self._save_position(portfolio_id, symbol, position)
            
        except Exception as e:
            logger.error(f"Position price update failed: {e}")
    
    async def close_position(
        self,
        user_id: str,
        portfolio_id: str,
        symbol: str,
        current_market_price: float
    ) -> Dict[str, Any]:
        """Close an entire position"""
        
        try:
            position = await self.get_position(portfolio_id, symbol)
            
            if not position:
                raise Exception(f"No position found for {symbol}")
            
            quantity = position["quantity"]
            
            if quantity == 0:
                raise Exception(f"Position for {symbol} is already closed")
            
            # Determine close action (opposite of current position)
            close_action = "sell" if quantity > 0 else "buy"
            close_quantity = abs(quantity)
            
            # Execute closing trade
            return await self.execute_trade(
                user_id=user_id,
                portfolio_id=portfolio_id,
                symbol=symbol,
                action=close_action,
                quantity=close_quantity,
                order_type="market",
                current_market_price=current_market_price
            )
            
        except Exception as e:
            logger.error(f"Position closing failed: {e}")
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get virtual trading system status"""
        
        try:
            return {
                "trading_engine": "active",
                "total_trades": self.execution_stats["total_trades"],
                "total_volume": self.execution_stats["total_volume"],
                "avg_execution_time_ms": self.execution_stats["avg_execution_time_ms"],
                "success_rate": self.execution_stats["success_rate"],
                "pending_orders": len(self.pending_orders),
                "commission_rate": self.config["commission_rate"],
                "slippage_rate": self.config["slippage_rate"],
                "max_order_size": self.config["max_order_size"]
            }
            
        except Exception as e:
            logger.error(f"System status retrieval failed: {e}")
            return {"error": str(e)}
    
    def _validate_trade_parameters(
        self,
        action: str,
        quantity: int,
        order_type: str,
        limit_price: Optional[float],
        stop_price: Optional[float],
        current_price: float
    ):
        """Validate trade parameters"""
        
        if action not in ["buy", "sell", "close"]:
            raise ValueError(f"Invalid action: {action}")
        
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}")
        
        if quantity > self.config["max_order_size"]:
            raise ValueError(f"Order size exceeds maximum: {self.config['max_order_size']}")
        
        if order_type not in ["market", "limit", "stop"]:
            raise ValueError(f"Invalid order type: {order_type}")
        
        if order_type == "limit" and limit_price is None:
            raise ValueError("Limit price required for limit orders")
        
        if order_type == "stop" and stop_price is None:
            raise ValueError("Stop price required for stop orders")
        
        if current_price <= 0:
            raise ValueError(f"Invalid current price: {current_price}")
    
    async def _calculate_execution_price(
        self,
        symbol: str,
        action: str,
        order_type: str,
        limit_price: Optional[float],
        stop_price: Optional[float],
        current_market_price: float
    ) -> float:
        """Calculate execution price based on order type"""
        
        if order_type == "market":
            return current_market_price
        
        elif order_type == "limit":
            # For buy orders, execute if market price <= limit price
            # For sell orders, execute if market price >= limit price
            if action == "buy" and current_market_price <= limit_price:
                return min(current_market_price, limit_price)
            elif action == "sell" and current_market_price >= limit_price:
                return max(current_market_price, limit_price)
            else:
                # Order not executable at current price
                return limit_price
        
        elif order_type == "stop":
            # Stop orders become market orders when triggered
            if action == "buy" and current_market_price >= stop_price:
                return current_market_price
            elif action == "sell" and current_market_price <= stop_price:
                return current_market_price
            else:
                # Stop not triggered
                return stop_price
        
        return current_market_price
    
    def _apply_slippage(self, price: float, action: str) -> float:
        """Apply slippage to execution price"""
        
        slippage_amount = price * self.config["slippage_rate"]
        
        if action == "buy":
            return price + slippage_amount  # Worse price for buyer
        else:
            return price - slippage_amount  # Worse price for seller
    
    def _calculate_commission(self, total_value: float) -> float:
        """Calculate commission for trade"""
        
        commission = total_value * self.config["commission_rate"]
        
        # Apply min/max commission limits
        commission = max(commission, self.config["min_commission"])
        commission = min(commission, self.config["max_commission"])
        
        return commission
    
    async def _execute_market_order(self, trade: VirtualTrade):
        """Execute a market order immediately"""
        
        try:
            # Update position
            await self._update_position(trade)
            
            # Store trade record
            await self._store_trade(trade)
            
            logger.info(f"Market order executed: {trade.trade_id}")
            
        except Exception as e:
            logger.error(f"Market order execution failed: {e}")
            raise
    
    async def _queue_pending_order(self, trade: VirtualTrade):
        """Queue a pending order for later execution"""
        
        try:
            # Store in pending orders
            self.pending_orders[trade.trade_id] = trade
            
            # Store in Redis for persistence
            await self.redis.setex(
                f"pending_order:{trade.trade_id}",
                86400,  # 24 hours TTL
                json.dumps(asdict(trade), default=str)
            )
            
            logger.info(f"Pending order queued: {trade.trade_id}")
            
        except Exception as e:
            logger.error(f"Pending order queueing failed: {e}")
            raise
    
    async def _update_position(self, trade: VirtualTrade):
        """Update position after trade execution"""
        
        try:
            symbol = trade.symbol
            portfolio_id = trade.portfolio_id
            
            # Get current position
            current_position = await self.get_position(portfolio_id, symbol)
            
            if current_position:
                # Update existing position
                old_quantity = current_position["quantity"]
                old_avg_cost = current_position["avg_cost"]
                
                if trade.action == "buy":
                    new_quantity = old_quantity + trade.quantity
                    if new_quantity != 0:
                        new_avg_cost = ((old_quantity * old_avg_cost) + (trade.quantity * trade.executed_price)) / new_quantity
                    else:
                        new_avg_cost = trade.executed_price
                        
                elif trade.action == "sell":
                    new_quantity = old_quantity - trade.quantity
                    new_avg_cost = old_avg_cost  # Avg cost doesn't change on sale
                    
                    # Calculate realized P&L
                    realized_pnl = (trade.executed_price - old_avg_cost) * trade.quantity
                    current_position["realized_pnl"] += realized_pnl
                
                current_position["quantity"] = new_quantity
                current_position["avg_cost"] = new_avg_cost
                current_position["current_price"] = trade.executed_price
                current_position["market_value"] = new_quantity * trade.executed_price
                current_position["unrealized_pnl"] = (trade.executed_price - new_avg_cost) * new_quantity
                current_position["last_updated"] = trade.timestamp.isoformat()
                
            else:
                # Create new position
                current_position = {
                    "position_id": str(uuid.uuid4()),
                    "portfolio_id": portfolio_id,
                    "symbol": symbol,
                    "quantity": trade.quantity if trade.action == "buy" else -trade.quantity,
                    "avg_cost": trade.executed_price,
                    "current_price": trade.executed_price,
                    "market_value": trade.quantity * trade.executed_price,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "opened_at": trade.timestamp.isoformat(),
                    "last_updated": trade.timestamp.isoformat()
                }
            
            # Save updated position
            await self._save_position(portfolio_id, symbol, current_position)
            
        except Exception as e:
            logger.error(f"Position update failed: {e}")
            raise
    
    async def _save_position(
        self,
        portfolio_id: str,
        symbol: str,
        position: Dict[str, Any]
    ):
        """Save position to Redis"""
        
        try:
            position_key = f"positions:{portfolio_id}:{symbol}"
            
            # If quantity is 0, remove the position
            if position["quantity"] == 0:
                await self.redis.delete(position_key)
            else:
                await self.redis.set(
                    position_key,
                    json.dumps(position, default=str)
                )
            
        except Exception as e:
            logger.error(f"Position save failed: {e}")
            raise
    
    async def _store_trade(self, trade: VirtualTrade):
        """Store trade record in Redis"""
        
        try:
            trade_key = f"trades:{trade.portfolio_id}:{trade.timestamp.timestamp()}:{trade.trade_id}"
            
            await self.redis.set(
                trade_key,
                json.dumps(asdict(trade), default=str)
            )
            
            # Also store in user's trade history
            user_trade_key = f"user_trades:{trade.user_id}:{trade.timestamp.timestamp()}:{trade.trade_id}"
            await self.redis.set(
                user_trade_key,
                json.dumps(asdict(trade), default=str)
            )
            
        except Exception as e:
            logger.error(f"Trade storage failed: {e}")
            raise
    
    async def _load_pending_orders(self):
        """Load pending orders from Redis"""
        
        try:
            pending_keys = await self.redis.keys("pending_order:*")
            
            for key in pending_keys:
                order_data = await self.redis.get(key)
                if order_data:
                    order_dict = json.loads(order_data)
                    
                    # Convert back to VirtualTrade object
                    trade = VirtualTrade(
                        trade_id=order_dict["trade_id"],
                        user_id=order_dict["user_id"],
                        portfolio_id=order_dict["portfolio_id"],
                        symbol=order_dict["symbol"],
                        action=order_dict["action"],
                        quantity=order_dict["quantity"],
                        order_type=order_dict["order_type"],
                        limit_price=order_dict.get("limit_price"),
                        stop_price=order_dict.get("stop_price"),
                        executed_price=order_dict["executed_price"],
                        total_value=order_dict["total_value"],
                        commission=order_dict["commission"],
                        timestamp=datetime.fromisoformat(order_dict["timestamp"]),
                        status=order_dict["status"]
                    )
                    
                    self.pending_orders[trade.trade_id] = trade
            
            logger.info(f"Loaded {len(self.pending_orders)} pending orders")
            
        except Exception as e:
            logger.error(f"Pending orders loading failed: {e}")
    
    async def _process_pending_orders(self):
        """Background task to process pending orders"""
        
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                
                orders_to_remove = []
                
                for trade_id, trade in self.pending_orders.items():
                    try:
                        # Check if order should be executed
                        # This would integrate with real market data
                        # For now, execute all pending orders after 1 minute
                        
                        time_since_order = datetime.utcnow() - trade.timestamp
                        if time_since_order.total_seconds() > 60:  # 1 minute
                            await self._execute_market_order(trade)
                            orders_to_remove.append(trade_id)
                            
                            # Remove from Redis
                            await self.redis.delete(f"pending_order:{trade_id}")
                    
                    except Exception as e:
                        logger.error(f"Pending order processing failed for {trade_id}: {e}")
                        orders_to_remove.append(trade_id)
                
                # Remove processed orders
                for trade_id in orders_to_remove:
                    self.pending_orders.pop(trade_id, None)
                
            except Exception as e:
                logger.error(f"Pending order processing error: {e}")
                await asyncio.sleep(5)
    
    async def _update_trade_stats(self, trade: VirtualTrade):
        """Update trade execution statistics"""
        
        try:
            self.execution_stats["total_trades"] += 1
            self.execution_stats["total_volume"] += trade.total_value
            
            # Update success rate (assume all trades succeed for now)
            self.execution_stats["success_rate"] = 1.0
            
        except Exception as e:
            logger.error(f"Trade stats update failed: {e}")
    
    async def _update_execution_stats(self):
        """Background task to update execution statistics"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                
                # Calculate average execution time and other metrics
                # This would integrate with real performance monitoring
                
            except Exception as e:
                logger.error(f"Execution stats update error: {e}")
                await asyncio.sleep(60)

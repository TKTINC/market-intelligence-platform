"""
WebSocket Handler for real-time communication
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import asdict

logger = logging.getLogger(__name__)

class WebSocketConnection:
    def __init__(self, websocket: WebSocket, connection_id: str, user_id: str = None):
        self.websocket = websocket
        self.connection_id = connection_id
        self.user_id = user_id
        self.subscriptions = set()
        self.connected_at = datetime.utcnow()
        self.last_ping = datetime.utcnow()
        self.is_alive = True

class WebSocketHandler:
    def __init__(self):
        # Connection management
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.portfolio_subscribers: Dict[str, Set[str]] = {}  # portfolio_id -> connection_ids
        self.market_data_subscribers: Dict[str, Set[str]] = {}  # symbol -> connection_ids
        
        # Configuration
        self.config = {
            "ping_interval": 30,        # Send ping every 30 seconds
            "ping_timeout": 10,         # Wait 10 seconds for pong
            "max_connections": 10000,   # Maximum concurrent connections
            "max_subscriptions_per_connection": 100,
            "heartbeat_interval": 5,    # Check connection health every 5 seconds
            "message_rate_limit": 10    # Max 10 messages per second per connection
        }
        
        # Message queues and rate limiting
        self.message_queues: Dict[str, List] = {}
        self.rate_limits: Dict[str, List[float]] = {}
        
        # Performance tracking
        self.stats = {
            "total_connections": 0,
            "current_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "disconnections": 0,
            "errors": 0
        }
        
    async def initialize(self):
        """Initialize the WebSocket handler"""
        try:
            # Start background tasks
            asyncio.create_task(self._heartbeat_task())
            asyncio.create_task(self._cleanup_task())
            asyncio.create_task(self._message_queue_processor())
            
            logger.info("WebSocket handler initialized")
            
        except Exception as e:
            logger.error(f"WebSocket handler initialization failed: {e}")
            raise
    
    async def close_all_connections(self):
        """Close all active WebSocket connections"""
        try:
            disconnection_tasks = []
            
            for connection_id, connection in list(self.active_connections.items()):
                task = self._disconnect_connection(connection_id, "Server shutdown")
                disconnection_tasks.append(task)
            
            if disconnection_tasks:
                await asyncio.gather(*disconnection_tasks, return_exceptions=True)
            
            logger.info("All WebSocket connections closed")
            
        except Exception as e:
            logger.error(f"Error closing WebSocket connections: {e}")
    
    async def health_check(self) -> str:
        """Check WebSocket handler health"""
        try:
            current_connections = len(self.active_connections)
            
            if current_connections > self.config["max_connections"]:
                return "degraded - too many connections"
            
            # Check for stale connections
            stale_count = sum(
                1 for conn in self.active_connections.values()
                if not conn.is_alive
            )
            
            if stale_count > current_connections * 0.1:  # More than 10% stale
                return "degraded - many stale connections"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"WebSocket health check failed: {e}")
            return "unhealthy"
    
    async def handle_portfolio_connection(self, websocket: WebSocket, portfolio_id: str):
        """Handle WebSocket connection for portfolio updates"""
        
        connection_id = f"portfolio_{portfolio_id}_{datetime.utcnow().timestamp()}"
        
        try:
            await websocket.accept()
            
            # Create connection
            connection = WebSocketConnection(websocket, connection_id)
            await self._register_connection(connection)
            
            # Subscribe to portfolio updates
            await self._subscribe_to_portfolio(connection_id, portfolio_id)
            
            # Send initial portfolio data
            await self._send_initial_portfolio_data(connection_id, portfolio_id)
            
            # Handle messages
            await self._handle_connection_messages(connection)
            
        except WebSocketDisconnect:
            logger.info(f"Portfolio WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"Portfolio WebSocket error: {e}")
        finally:
            await self._cleanup_connection(connection_id)
    
    async def handle_market_data_connection(self, websocket: WebSocket, symbols: List[str]):
        """Handle WebSocket connection for market data updates"""
        
        connection_id = f"market_{symbols[0]}_{datetime.utcnow().timestamp()}"
        
        try:
            await websocket.accept()
            
            # Create connection
            connection = WebSocketConnection(websocket, connection_id)
            await self._register_connection(connection)
            
            # Subscribe to market data
            for symbol in symbols:
                await self._subscribe_to_market_data(connection_id, symbol)
            
            # Send initial market data
            await self._send_initial_market_data(connection_id, symbols)
            
            # Handle messages
            await self._handle_connection_messages(connection)
            
        except WebSocketDisconnect:
            logger.info(f"Market data WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"Market data WebSocket error: {e}")
        finally:
            await self._cleanup_connection(connection_id)
    
    async def handle_user_connection(self, websocket: WebSocket, user_id: str):
        """Handle WebSocket connection for user-specific updates"""
        
        connection_id = f"user_{user_id}_{datetime.utcnow().timestamp()}"
        
        try:
            await websocket.accept()
            
            # Create connection
            connection = WebSocketConnection(websocket, connection_id, user_id)
            await self._register_connection(connection)
            
            # Add to user connections
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)
            
            # Send welcome message
            await self._send_message(connection_id, {
                "type": "welcome",
                "user_id": user_id,
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Handle messages
            await self._handle_connection_messages(connection)
            
        except WebSocketDisconnect:
            logger.info(f"User WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"User WebSocket error: {e}")
        finally:
            await self._cleanup_connection(connection_id)
    
    async def broadcast_price_updates(self, price_updates: Dict[str, float]):
        """Broadcast price updates to subscribed connections"""
        
        try:
            for symbol, price in price_updates.items():
                if symbol in self.market_data_subscribers:
                    message = {
                        "type": "price_update",
                        "symbol": symbol,
                        "price": price,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    # Send to all subscribers
                    connection_ids = list(self.market_data_subscribers[symbol])
                    await self._broadcast_to_connections(connection_ids, message)
            
        except Exception as e:
            logger.error(f"Price update broadcast failed: {e}")
    
    async def broadcast_portfolio_update(self, portfolio_id: str, portfolio_data: Dict[str, Any]):
        """Broadcast portfolio update to subscribed connections"""
        
        try:
            if portfolio_id in self.portfolio_subscribers:
                message = {
                    "type": "portfolio_update",
                    "portfolio_id": portfolio_id,
                    "data": portfolio_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                connection_ids = list(self.portfolio_subscribers[portfolio_id])
                await self._broadcast_to_connections(connection_ids, message)
            
        except Exception as e:
            logger.error(f"Portfolio update broadcast failed: {e}")
    
    async def broadcast_trade_execution(self, user_id: str, trade_data: Dict[str, Any]):
        """Broadcast trade execution to user connections"""
        
        try:
            if user_id in self.user_connections:
                message = {
                    "type": "trade_executed",
                    "data": trade_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                connection_ids = list(self.user_connections[user_id])
                await self._broadcast_to_connections(connection_ids, message)
            
        except Exception as e:
            logger.error(f"Trade execution broadcast failed: {e}")
    
    async def broadcast_analysis_result(self, user_id: str, analysis_data: Dict[str, Any]):
        """Broadcast analysis result to user connections"""
        
        try:
            if user_id in self.user_connections:
                message = {
                    "type": "analysis_result",
                    "data": analysis_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                connection_ids = list(self.user_connections[user_id])
                await self._broadcast_to_connections(connection_ids, message)
            
        except Exception as e:
            logger.error(f"Analysis result broadcast failed: {e}")
    
    async def broadcast_risk_alert(self, user_id: str, alert_data: Dict[str, Any]):
        """Broadcast risk alert to user connections"""
        
        try:
            if user_id in self.user_connections:
                message = {
                    "type": "risk_alert",
                    "data": alert_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                connection_ids = list(self.user_connections[user_id])
                await self._broadcast_to_connections(connection_ids, message)
            
        except Exception as e:
            logger.error(f"Risk alert broadcast failed: {e}")
    
    async def broadcast_admin_alert(self, alert_data: Dict[str, Any]):
        """Broadcast admin alert to all admin connections"""
        
        try:
            # Find admin connections (would check user roles)
            admin_connections = [
                conn_id for conn_id, conn in self.active_connections.items()
                if conn.user_id  # Would check if user has admin role
            ]
            
            message = {
                "type": "admin_alert",
                "data": alert_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._broadcast_to_connections(admin_connections, message)
            
        except Exception as e:
            logger.error(f"Admin alert broadcast failed: {e}")
    
    def has_subscribers(self, user_id: str) -> bool:
        """Check if user has active WebSocket connections"""
        return user_id in self.user_connections and len(self.user_connections[user_id]) > 0
    
    def get_connection_count(self) -> int:
        """Get current connection count"""
        return len(self.active_connections)
    
    async def _register_connection(self, connection: WebSocketConnection):
        """Register a new WebSocket connection"""
        
        try:
            # Check connection limits
            if len(self.active_connections) >= self.config["max_connections"]:
                await connection.websocket.close(code=1013, reason="Too many connections")
                return
            
            # Register connection
            self.active_connections[connection.connection_id] = connection
            
            # Initialize rate limiting
            self.rate_limits[connection.connection_id] = []
            
            # Update stats
            self.stats["total_connections"] += 1
            self.stats["current_connections"] = len(self.active_connections)
            
            logger.debug(f"WebSocket connection registered: {connection.connection_id}")
            
        except Exception as e:
            logger.error(f"Connection registration failed: {e}")
            raise
    
    async def _handle_connection_messages(self, connection: WebSocketConnection):
        """Handle incoming WebSocket messages"""
        
        try:
            while connection.is_alive:
                try:
                    # Receive message with timeout
                    message = await asyncio.wait_for(
                        connection.websocket.receive_text(),
                        timeout=30.0
                    )
                    
                    # Check rate limiting
                    if not await self._check_rate_limit(connection.connection_id):
                        await self._send_error(connection.connection_id, "Rate limit exceeded")
                        continue
                    
                    # Process message
                    await self._process_message(connection.connection_id, message)
                    
                    # Update stats
                    self.stats["messages_received"] += 1
                    
                except asyncio.TimeoutError:
                    # Send ping to check if connection is alive
                    await self._send_ping(connection.connection_id)
                    
                except WebSocketDisconnect:
                    break
                    
                except Exception as e:
                    logger.error(f"Message handling error for {connection.connection_id}: {e}")
                    self.stats["errors"] += 1
                    await self._send_error(connection.connection_id, "Message processing error")
            
        except Exception as e:
            logger.error(f"Connection message handling failed: {e}")
        finally:
            connection.is_alive = False
    
    async def _process_message(self, connection_id: str, message: str):
        """Process incoming WebSocket message"""
        
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "ping":
                await self._handle_ping(connection_id, data)
            elif message_type == "pong":
                await self._handle_pong(connection_id, data)
            elif message_type == "subscribe":
                await self._handle_subscription(connection_id, data)
            elif message_type == "unsubscribe":
                await self._handle_unsubscription(connection_id, data)
            elif message_type == "portfolio_request":
                await self._handle_portfolio_request(connection_id, data)
            elif message_type == "market_data_request":
                await self._handle_market_data_request(connection_id, data)
            else:
                await self._send_error(connection_id, f"Unknown message type: {message_type}")
            
        except json.JSONDecodeError:
            await self._send_error(connection_id, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            await self._send_error(connection_id, "Message processing error")
    
    async def _handle_ping(self, connection_id: str, data: Dict[str, Any]):
        """Handle ping message"""
        
        try:
            await self._send_message(connection_id, {
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Ping handling failed: {e}")
    
    async def _handle_pong(self, connection_id: str, data: Dict[str, Any]):
        """Handle pong message"""
        
        try:
            if connection_id in self.active_connections:
                self.active_connections[connection_id].last_ping = datetime.utcnow()
                self.active_connections[connection_id].is_alive = True
            
        except Exception as e:
            logger.error(f"Pong handling failed: {e}")
    
    async def _handle_subscription(self, connection_id: str, data: Dict[str, Any]):
        """Handle subscription request"""
        
        try:
            subscription_type = data.get("subscription_type")
            target = data.get("target")
            
            if not subscription_type or not target:
                await self._send_error(connection_id, "Invalid subscription request")
                return
            
            # Check subscription limits
            connection = self.active_connections.get(connection_id)
            if connection and len(connection.subscriptions) >= self.config["max_subscriptions_per_connection"]:
                await self._send_error(connection_id, "Subscription limit exceeded")
                return
            
            if subscription_type == "portfolio":
                await self._subscribe_to_portfolio(connection_id, target)
            elif subscription_type == "market_data":
                await self._subscribe_to_market_data(connection_id, target)
            else:
                await self._send_error(connection_id, f"Unknown subscription type: {subscription_type}")
                return
            
            await self._send_message(connection_id, {
                "type": "subscription_confirmed",
                "subscription_type": subscription_type,
                "target": target,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Subscription handling failed: {e}")
            await self._send_error(connection_id, "Subscription processing error")
    
    async def _handle_unsubscription(self, connection_id: str, data: Dict[str, Any]):
        """Handle unsubscription request"""
        
        try:
            subscription_type = data.get("subscription_type")
            target = data.get("target")
            
            if subscription_type == "portfolio":
                await self._unsubscribe_from_portfolio(connection_id, target)
            elif subscription_type == "market_data":
                await self._unsubscribe_from_market_data(connection_id, target)
            
            await self._send_message(connection_id, {
                "type": "unsubscription_confirmed",
                "subscription_type": subscription_type,
                "target": target,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Unsubscription handling failed: {e}")
            await self._send_error(connection_id, "Unsubscription processing error")
    
    async def _handle_portfolio_request(self, connection_id: str, data: Dict[str, Any]):
        """Handle portfolio data request"""
        
        try:
            portfolio_id = data.get("portfolio_id")
            if not portfolio_id:
                await self._send_error(connection_id, "Portfolio ID required")
                return
            
            # Would fetch portfolio data here
            # For now, send acknowledgment
            await self._send_message(connection_id, {
                "type": "portfolio_data_requested",
                "portfolio_id": portfolio_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Portfolio request handling failed: {e}")
            await self._send_error(connection_id, "Portfolio request processing error")
    
    async def _handle_market_data_request(self, connection_id: str, data: Dict[str, Any]):
        """Handle market data request"""
        
        try:
            symbols = data.get("symbols", [])
            if not symbols:
                await self._send_error(connection_id, "Symbols required")
                return
            
            # Would fetch market data here
            await self._send_message(connection_id, {
                "type": "market_data_requested",
                "symbols": symbols,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Market data request handling failed: {e}")
            await self._send_error(connection_id, "Market data request processing error")
    
    async def _subscribe_to_portfolio(self, connection_id: str, portfolio_id: str):
        """Subscribe connection to portfolio updates"""
        
        try:
            if portfolio_id not in self.portfolio_subscribers:
                self.portfolio_subscribers[portfolio_id] = set()
            
            self.portfolio_subscribers[portfolio_id].add(connection_id)
            
            # Add to connection subscriptions
            connection = self.active_connections.get(connection_id)
            if connection:
                connection.subscriptions.add(f"portfolio:{portfolio_id}")
            
        except Exception as e:
            logger.error(f"Portfolio subscription failed: {e}")
    
    async def _subscribe_to_market_data(self, connection_id: str, symbol: str):
        """Subscribe connection to market data updates"""
        
        try:
            if symbol not in self.market_data_subscribers:
                self.market_data_subscribers[symbol] = set()
            
            self.market_data_subscribers[symbol].add(connection_id)
            
            # Add to connection subscriptions
            connection = self.active_connections.get(connection_id)
            if connection:
                connection.subscriptions.add(f"market_data:{symbol}")
            
        except Exception as e:
            logger.error(f"Market data subscription failed: {e}")
    
    async def _unsubscribe_from_portfolio(self, connection_id: str, portfolio_id: str):
        """Unsubscribe connection from portfolio updates"""
        
        try:
            if portfolio_id in self.portfolio_subscribers:
                self.portfolio_subscribers[portfolio_id].discard(connection_id)
                
                # Clean up empty subscription sets
                if not self.portfolio_subscribers[portfolio_id]:
                    del self.portfolio_subscribers[portfolio_id]
            
            # Remove from connection subscriptions
            connection = self.active_connections.get(connection_id)
            if connection:
                connection.subscriptions.discard(f"portfolio:{portfolio_id}")
            
        except Exception as e:
            logger.error(f"Portfolio unsubscription failed: {e}")
    
    async def _unsubscribe_from_market_data(self, connection_id: str, symbol: str):
        """Unsubscribe connection from market data updates"""
        
        try:
            if symbol in self.market_data_subscribers:
                self.market_data_subscribers[symbol].discard(connection_id)
                
                # Clean up empty subscription sets
                if not self.market_data_subscribers[symbol]:
                    del self.market_data_subscribers[symbol]
            
            # Remove from connection subscriptions
            connection = self.active_connections.get(connection_id)
            if connection:
                connection.subscriptions.discard(f"market_data:{symbol}")
            
        except Exception as e:
            logger.error(f"Market data unsubscription failed: {e}")
    
    async def _send_initial_portfolio_data(self, connection_id: str, portfolio_id: str):
        """Send initial portfolio data to connection"""
        
        try:
            # Would fetch actual portfolio data here
            message = {
                "type": "initial_portfolio_data",
                "portfolio_id": portfolio_id,
                "message": "Portfolio data would be loaded here",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._send_message(connection_id, message)
            
        except Exception as e:
            logger.error(f"Initial portfolio data send failed: {e}")
    
    async def _send_initial_market_data(self, connection_id: str, symbols: List[str]):
        """Send initial market data to connection"""
        
        try:
            # Would fetch actual market data here
            message = {
                "type": "initial_market_data",
                "symbols": symbols,
                "message": "Market data would be loaded here",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._send_message(connection_id, message)
            
        except Exception as e:
            logger.error(f"Initial market data send failed: {e}")
    
    async def _send_message(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific connection"""
        
        try:
            connection = self.active_connections.get(connection_id)
            if not connection or not connection.is_alive:
                return
            
            # Add to message queue for processing
            if connection_id not in self.message_queues:
                self.message_queues[connection_id] = []
            
            self.message_queues[connection_id].append(message)
            
        except Exception as e:
            logger.error(f"Message send failed for {connection_id}: {e}")
    
    async def _send_error(self, connection_id: str, error_message: str):
        """Send error message to connection"""
        
        try:
            message = {
                "type": "error",
                "error": error_message,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._send_message(connection_id, message)
            
        except Exception as e:
            logger.error(f"Error message send failed: {e}")
    
    async def _send_ping(self, connection_id: str):
        """Send ping to connection"""
        
        try:
            message = {
                "type": "ping",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._send_message(connection_id, message)
            
        except Exception as e:
            logger.error(f"Ping send failed: {e}")
    
    async def _broadcast_to_connections(self, connection_ids: List[str], message: Dict[str, Any]):
        """Broadcast message to multiple connections"""
        
        try:
            send_tasks = []
            
            for connection_id in connection_ids:
                if connection_id in self.active_connections:
                    task = self._send_message(connection_id, message)
                    send_tasks.append(task)
            
            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Broadcast failed: {e}")
    
    async def _check_rate_limit(self, connection_id: str) -> bool:
        """Check if connection is within rate limits"""
        
        try:
            current_time = time.time()
            
            if connection_id not in self.rate_limits:
                self.rate_limits[connection_id] = []
            
            # Clean old timestamps
            cutoff_time = current_time - 1.0  # 1 second window
            self.rate_limits[connection_id] = [
                ts for ts in self.rate_limits[connection_id] if ts > cutoff_time
            ]
            
            # Check rate limit
            if len(self.rate_limits[connection_id]) >= self.config["message_rate_limit"]:
                return False
            
            # Add current timestamp
            self.rate_limits[connection_id].append(current_time)
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow on error
    
    async def _cleanup_connection(self, connection_id: str):
        """Clean up connection and subscriptions"""
        
        try:
            # Remove from active connections
            connection = self.active_connections.pop(connection_id, None)
            
            if connection:
                # Remove from user connections
                if connection.user_id and connection.user_id in self.user_connections:
                    self.user_connections[connection.user_id].discard(connection_id)
                    
                    # Clean up empty user connection sets
                    if not self.user_connections[connection.user_id]:
                        del self.user_connections[connection.user_id]
                
                # Remove from all subscriptions
                for subscription in list(connection.subscriptions):
                    if subscription.startswith("portfolio:"):
                        portfolio_id = subscription.split(":", 1)[1]
                        await self._unsubscribe_from_portfolio(connection_id, portfolio_id)
                    elif subscription.startswith("market_data:"):
                        symbol = subscription.split(":", 1)[1]
                        await self._unsubscribe_from_market_data(connection_id, symbol)
            
            # Clean up rate limits and message queues
            self.rate_limits.pop(connection_id, None)
            self.message_queues.pop(connection_id, None)
            
            # Update stats
            self.stats["current_connections"] = len(self.active_connections)
            self.stats["disconnections"] += 1
            
            logger.debug(f"Connection cleaned up: {connection_id}")
            
        except Exception as e:
            logger.error(f"Connection cleanup failed: {e}")
    
    async def _disconnect_connection(self, connection_id: str, reason: str = "Disconnected"):
        """Disconnect a specific connection"""
        
        try:
            connection = self.active_connections.get(connection_id)
            if connection and connection.is_alive:
                await connection.websocket.close(code=1000, reason=reason)
                connection.is_alive = False
            
        except Exception as e:
            logger.error(f"Connection disconnect failed: {e}")
    
    # Background tasks
    async def _heartbeat_task(self):
        """Background task to check connection health"""
        
        while True:
            try:
                await asyncio.sleep(self.config["heartbeat_interval"])
                
                current_time = datetime.utcnow()
                stale_connections = []
                
                for connection_id, connection in self.active_connections.items():
                    # Check if connection is stale
                    time_since_ping = (current_time - connection.last_ping).total_seconds()
                    
                    if time_since_ping > self.config["ping_timeout"]:
                        stale_connections.append(connection_id)
                    elif time_since_ping > self.config["ping_interval"]:
                        # Send ping
                        await self._send_ping(connection_id)
                
                # Clean up stale connections
                for connection_id in stale_connections:
                    await self._disconnect_connection(connection_id, "Connection timeout")
                    await self._cleanup_connection(connection_id)
                
            except Exception as e:
                logger.error(f"Heartbeat task error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_task(self):
        """Background task for general cleanup"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean up empty subscription sets
                empty_portfolio_subs = [
                    portfolio_id for portfolio_id, subs in self.portfolio_subscribers.items()
                    if not subs
                ]
                
                for portfolio_id in empty_portfolio_subs:
                    del self.portfolio_subscribers[portfolio_id]
                
                empty_market_subs = [
                    symbol for symbol, subs in self.market_data_subscribers.items()
                    if not subs
                ]
                
                for symbol in empty_market_subs:
                    del self.market_data_subscribers[symbol]
                
                logger.debug(f"Cleanup completed: removed {len(empty_portfolio_subs)} portfolio subs, {len(empty_market_subs)} market subs")
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(300)
    
    async def _message_queue_processor(self):
        """Background task to process message queues"""
        
        while True:
            try:
                await asyncio.sleep(0.1)  # Process every 100ms
                
                for connection_id, message_queue in list(self.message_queues.items()):
                    if not message_queue:
                        continue
                    
                    connection = self.active_connections.get(connection_id)
                    if not connection or not connection.is_alive:
                        # Clean up queue for disconnected connection
                        self.message_queues.pop(connection_id, None)
                        continue
                    
                    # Process messages in queue
                    messages_to_send = message_queue[:10]  # Send up to 10 messages at once
                    self.message_queues[connection_id] = message_queue[10:]
                    
                    for message in messages_to_send:
                        try:
                            await connection.websocket.send_text(json.dumps(message))
                            self.stats["messages_sent"] += 1
                        except Exception as e:
                            logger.error(f"Message send failed for {connection_id}: {e}")
                            connection.is_alive = False
                            break
                
            except Exception as e:
                logger.error(f"Message queue processor error: {e}")
                await asyncio.sleep(1)

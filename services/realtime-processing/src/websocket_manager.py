"""
WebSocket Manager for real-time client connections and data streaming
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Set, Any, Optional
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, asdict
import aioredis
import uuid

logger = logging.getLogger(__name__)

@dataclass
class WebSocketConnection:
    connection_id: str
    websocket: WebSocket
    user_id: Optional[str]
    subscriptions: Set[str]
    connected_at: datetime
    last_ping: datetime
    message_count: int = 0
    data_sent_bytes: int = 0

@dataclass
class SubscriptionInfo:
    subscription_id: str
    connection_id: str
    subscription_type: str  # 'intelligence', 'market_data', 'alerts'
    symbols: List[str]
    filters: Dict[str, Any]
    created_at: datetime

class WebSocketManager:
    def __init__(self):
        # Active connections
        self.connections: Dict[str, WebSocketConnection] = {}
        
        # Subscriptions by type
        self.intelligence_subscriptions: Dict[str, List[str]] = {}  # stream_id -> connection_ids
        self.market_data_subscriptions: Dict[str, List[str]] = {}  # symbol -> connection_ids
        self.alert_subscriptions: Dict[str, List[str]] = {}  # user_id -> connection_ids
        
        # Subscription metadata
        self.subscription_info: Dict[str, SubscriptionInfo] = {}
        
        # Redis for pub/sub
        self.redis = None
        self.redis_subscriber = None
        
        # Configuration
        self.config = {
            "max_connections_per_user": 5,
            "max_message_rate": 100,  # messages per minute
            "ping_interval": 30,      # seconds
            "connection_timeout": 300, # 5 minutes
            "max_subscriptions_per_connection": 10,
            "message_buffer_size": 1000
        }
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # Message queue for broadcasting
        self.broadcast_queue = asyncio.Queue()
        
        # Background tasks
        self.background_tasks = []
        
    async def start(self):
        """Start the WebSocket manager"""
        try:
            # Initialize Redis for pub/sub
            self.redis = aioredis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            
            # Start background tasks
            self.background_tasks = [
                asyncio.create_task(self._connection_monitor()),
                asyncio.create_task(self._broadcast_processor()),
                asyncio.create_task(self._ping_connections()),
                asyncio.create_task(self._cleanup_disconnected())
            ]
            
            logger.info("WebSocket manager started successfully")
            
        except Exception as e:
            logger.error(f"WebSocket manager startup failed: {e}")
            raise
    
    async def close_all_connections(self):
        """Close all WebSocket connections"""
        try:
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Close all connections
            for connection in list(self.connections.values()):
                try:
                    await connection.websocket.close()
                except Exception as e:
                    logger.error(f"Error closing connection {connection.connection_id}: {e}")
            
            # Close Redis connections
            if self.redis_subscriber:
                await self.redis_subscriber.close()
            if self.redis:
                await self.redis.close()
            
            logger.info("All WebSocket connections closed")
            
        except Exception as e:
            logger.error(f"WebSocket manager shutdown error: {e}")
    
    async def handle_intelligence_connection(self, websocket: WebSocket, stream_id: str):
        """Handle intelligence streaming WebSocket connection"""
        
        connection_id = str(uuid.uuid4())
        
        try:
            await websocket.accept()
            
            # Create connection record
            connection = WebSocketConnection(
                connection_id=connection_id,
                websocket=websocket,
                user_id=None,  # Extract from auth if needed
                subscriptions={stream_id},
                connected_at=datetime.utcnow(),
                last_ping=datetime.utcnow()
            )
            
            self.connections[connection_id] = connection
            
            # Subscribe to intelligence stream
            if stream_id not in self.intelligence_subscriptions:
                self.intelligence_subscriptions[stream_id] = []
            self.intelligence_subscriptions[stream_id].append(connection_id)
            
            # Send connection confirmation
            await self._send_message(connection_id, {
                "type": "connection_established",
                "stream_id": stream_id,
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(f"Intelligence WebSocket connected: {connection_id} for stream {stream_id}")
            
            # Handle incoming messages
            await self._handle_connection_messages(connection)
            
        except WebSocketDisconnect:
            logger.info(f"Intelligence WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"Intelligence WebSocket error for {connection_id}: {e}")
        finally:
            await self._cleanup_connection(connection_id)
    
    async def handle_market_data_connection(self, websocket: WebSocket, symbols: List[str]):
        """Handle market data streaming WebSocket connection"""
        
        connection_id = str(uuid.uuid4())
        
        try:
            await websocket.accept()
            
            # Create connection record
            connection = WebSocketConnection(
                connection_id=connection_id,
                websocket=websocket,
                user_id=None,
                subscriptions=set(symbols),
                connected_at=datetime.utcnow(),
                last_ping=datetime.utcnow()
            )
            
            self.connections[connection_id] = connection
            
            # Subscribe to market data for symbols
            for symbol in symbols:
                if symbol not in self.market_data_subscriptions:
                    self.market_data_subscriptions[symbol] = []
                self.market_data_subscriptions[symbol].append(connection_id)
            
            # Send connection confirmation
            await self._send_message(connection_id, {
                "type": "connection_established",
                "symbols": symbols,
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(f"Market data WebSocket connected: {connection_id} for symbols {symbols}")
            
            # Handle incoming messages
            await self._handle_connection_messages(connection)
            
        except WebSocketDisconnect:
            logger.info(f"Market data WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"Market data WebSocket error for {connection_id}: {e}")
        finally:
            await self._cleanup_connection(connection_id)
    
    async def _handle_connection_messages(self, connection: WebSocketConnection):
        """Handle incoming messages from WebSocket connection"""
        
        try:
            while True:
                # Wait for message with timeout
                try:
                    data = await asyncio.wait_for(
                        connection.websocket.receive_text(),
                        timeout=self.config["connection_timeout"]
                    )
                    
                    message = json.loads(data)
                    await self._process_client_message(connection, message)
                    
                except asyncio.TimeoutError:
                    # Send ping to check if connection is alive
                    await self._send_ping(connection.connection_id)
                    
                except json.JSONDecodeError:
                    await self._send_error(connection.connection_id, "Invalid JSON format")
                    
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Message handling error for {connection.connection_id}: {e}")
    
    async def _process_client_message(self, connection: WebSocketConnection, message: Dict[str, Any]):
        """Process message from client"""
        
        try:
            message_type = message.get("type", "unknown")
            
            # Rate limiting
            if not await self._check_rate_limit(connection.connection_id):
                await self._send_error(connection.connection_id, "Rate limit exceeded")
                return
            
            if message_type == "ping":
                await self._handle_ping(connection)
                
            elif message_type == "subscribe":
                await self._handle_subscribe(connection, message)
                
            elif message_type == "unsubscribe":
                await self._handle_unsubscribe(connection, message)
                
            elif message_type == "get_status":
                await self._handle_status_request(connection)
                
            else:
                await self._send_error(connection.connection_id, f"Unknown message type: {message_type}")
            
        except Exception as e:
            logger.error(f"Client message processing error: {e}")
            await self._send_error(connection.connection_id, "Message processing failed")
    
    async def _handle_ping(self, connection: WebSocketConnection):
        """Handle ping message"""
        
        connection.last_ping = datetime.utcnow()
        
        await self._send_message(connection.connection_id, {
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _handle_subscribe(self, connection: WebSocketConnection, message: Dict[str, Any]):
        """Handle subscription request"""
        
        try:
            subscription_type = message.get("subscription_type", "market_data")
            symbols = message.get("symbols", [])
            filters = message.get("filters", {})
            
            # Check subscription limits
            if len(connection.subscriptions) >= self.config["max_subscriptions_per_connection"]:
                await self._send_error(connection.connection_id, "Maximum subscriptions reached")
                return
            
            # Create subscription
            subscription_id = str(uuid.uuid4())
            
            subscription_info = SubscriptionInfo(
                subscription_id=subscription_id,
                connection_id=connection.connection_id,
                subscription_type=subscription_type,
                symbols=symbols,
                filters=filters,
                created_at=datetime.utcnow()
            )
            
            self.subscription_info[subscription_id] = subscription_info
            connection.subscriptions.add(subscription_id)
            
            # Add to appropriate subscription registry
            if subscription_type == "market_data":
                for symbol in symbols:
                    if symbol not in self.market_data_subscriptions:
                        self.market_data_subscriptions[symbol] = []
                    self.market_data_subscriptions[symbol].append(connection.connection_id)
            
            elif subscription_type == "alerts":
                user_id = connection.user_id or "anonymous"
                if user_id not in self.alert_subscriptions:
                    self.alert_subscriptions[user_id] = []
                self.alert_subscriptions[user_id].append(connection.connection_id)
            
            # Send confirmation
            await self._send_message(connection.connection_id, {
                "type": "subscription_confirmed",
                "subscription_id": subscription_id,
                "subscription_type": subscription_type,
                "symbols": symbols,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(f"New subscription: {subscription_id} for connection {connection.connection_id}")
            
        except Exception as e:
            logger.error(f"Subscription handling error: {e}")
            await self._send_error(connection.connection_id, "Subscription failed")
    
    async def _handle_unsubscribe(self, connection: WebSocketConnection, message: Dict[str, Any]):
        """Handle unsubscription request"""
        
        try:
            subscription_id = message.get("subscription_id")
            
            if subscription_id in self.subscription_info:
                subscription_info = self.subscription_info[subscription_id]
                
                # Remove from registries
                if subscription_info.subscription_type == "market_data":
                    for symbol in subscription_info.symbols:
                        if symbol in self.market_data_subscriptions:
                            if connection.connection_id in self.market_data_subscriptions[symbol]:
                                self.market_data_subscriptions[symbol].remove(connection.connection_id)
                
                elif subscription_info.subscription_type == "alerts":
                    user_id = connection.user_id or "anonymous"
                    if user_id in self.alert_subscriptions:
                        if connection.connection_id in self.alert_subscriptions[user_id]:
                            self.alert_subscriptions[user_id].remove(connection.connection_id)
                
                # Remove subscription
                del self.subscription_info[subscription_id]
                connection.subscriptions.discard(subscription_id)
                
                # Send confirmation
                await self._send_message(connection.connection_id, {
                    "type": "unsubscription_confirmed",
                    "subscription_id": subscription_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                logger.info(f"Unsubscribed: {subscription_id} for connection {connection.connection_id}")
            
        except Exception as e:
            logger.error(f"Unsubscription handling error: {e}")
            await self._send_error(connection.connection_id, "Unsubscription failed")
    
    async def _handle_status_request(self, connection: WebSocketConnection):
        """Handle status request"""
        
        try:
            status = {
                "connection_id": connection.connection_id,
                "connected_at": connection.connected_at.isoformat(),
                "subscriptions": list(connection.subscriptions),
                "message_count": connection.message_count,
                "data_sent_bytes": connection.data_sent_bytes,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._send_message(connection.connection_id, {
                "type": "status",
                "data": status
            })
            
        except Exception as e:
            logger.error(f"Status request handling error: {e}")
    
    async def broadcast_intelligence_update(self, symbols: List[str], data: Dict[str, Any]):
        """Broadcast intelligence update to relevant subscribers"""
        
        try:
            # Find connections subscribed to these symbols
            target_connections = set()
            
            for stream_id, connection_ids in self.intelligence_subscriptions.items():
                # Check if any symbols match
                # For now, broadcast to all intelligence subscribers
                target_connections.update(connection_ids)
            
            # Prepare broadcast message
            message = {
                "type": "intelligence_update",
                "symbols": symbols,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._broadcast_to_connections(target_connections, message)
            
        except Exception as e:
            logger.error(f"Intelligence broadcast failed: {e}")
    
    async def broadcast_market_data_update(self, symbol: str, data: Dict[str, Any]):
        """Broadcast market data update to subscribers"""
        
        try:
            target_connections = self.market_data_subscriptions.get(symbol, [])
            
            message = {
                "type": "market_data_update",
                "symbol": symbol,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._broadcast_to_connections(target_connections, message)
            
        except Exception as e:
            logger.error(f"Market data broadcast failed for {symbol}: {e}")
    
    async def broadcast_alert(self, user_id: str, alert_data: Dict[str, Any]):
        """Broadcast alert to user's connections"""
        
        try:
            target_connections = self.alert_subscriptions.get(user_id, [])
            
            message = {
                "type": "alert",
                "data": alert_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._broadcast_to_connections(target_connections, message)
            
        except Exception as e:
            logger.error(f"Alert broadcast failed for user {user_id}: {e}")
    
    async def _broadcast_to_connections(self, connection_ids: List[str], message: Dict[str, Any]):
        """Broadcast message to specific connections"""
        
        successful_sends = 0
        failed_sends = 0
        
        for connection_id in connection_ids:
            try:
                await self._send_message(connection_id, message)
                successful_sends += 1
            except Exception as e:
                logger.debug(f"Failed to send to connection {connection_id}: {e}")
                failed_sends += 1
        
        logger.debug(f"Broadcast complete: {successful_sends} successful, {failed_sends} failed")
    
    async def _send_message(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific connection"""
        
        try:
            if connection_id not in self.connections:
                raise Exception(f"Connection {connection_id} not found")
            
            connection = self.connections[connection_id]
            
            # Serialize message
            message_text = json.dumps(message)
            
            # Send message
            await connection.websocket.send_text(message_text)
            
            # Update connection stats
            connection.message_count += 1
            connection.data_sent_bytes += len(message_text.encode())
            
        except Exception as e:
            logger.debug(f"Send message failed for {connection_id}: {e}")
            # Schedule connection cleanup
            await self._cleanup_connection(connection_id)
            raise
    
    async def _send_error(self, connection_id: str, error_message: str):
        """Send error message to connection"""
        
        try:
            await self._send_message(connection_id, {
                "type": "error",
                "message": error_message,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception:
            pass  # Ignore errors when sending error messages
    
    async def _send_ping(self, connection_id: str):
        """Send ping to connection"""
        
        try:
            await self._send_message(connection_id, {
                "type": "ping",
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception:
            pass  # Connection will be cleaned up by monitor
    
    async def _check_rate_limit(self, connection_id: str) -> bool:
        """Check if connection is within rate limits"""
        
        try:
            current_time = time.time()
            
            if connection_id not in self.rate_limits:
                self.rate_limits[connection_id] = {
                    "messages": [],
                    "last_reset": current_time
                }
            
            rate_data = self.rate_limits[connection_id]
            
            # Reset counter if needed
            if current_time - rate_data["last_reset"] > 60:  # 1 minute
                rate_data["messages"] = []
                rate_data["last_reset"] = current_time
            
            # Check rate limit
            rate_data["messages"].append(current_time)
            
            # Remove old messages (older than 1 minute)
            rate_data["messages"] = [
                msg_time for msg_time in rate_data["messages"]
                if current_time - msg_time < 60
            ]
            
            return len(rate_data["messages"]) <= self.config["max_message_rate"]
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow on error
    
    async def _cleanup_connection(self, connection_id: str):
        """Cleanup disconnected connection"""
        
        try:
            if connection_id not in self.connections:
                return
            
            connection = self.connections[connection_id]
            
            # Remove from subscription registries
            for subscription_id in connection.subscriptions:
                if subscription_id in self.subscription_info:
                    subscription_info = self.subscription_info[subscription_id]
                    
                    if subscription_info.subscription_type == "market_data":
                        for symbol in subscription_info.symbols:
                            if symbol in self.market_data_subscriptions:
                                if connection_id in self.market_data_subscriptions[symbol]:
                                    self.market_data_subscriptions[symbol].remove(connection_id)
                    
                    elif subscription_info.subscription_type == "alerts":
                        user_id = connection.user_id or "anonymous"
                        if user_id in self.alert_subscriptions:
                            if connection_id in self.alert_subscriptions[user_id]:
                                self.alert_subscriptions[user_id].remove(connection_id)
                    
                    del self.subscription_info[subscription_id]
            
            # Remove from intelligence subscriptions
            for stream_id, connection_ids in self.intelligence_subscriptions.items():
                if connection_id in connection_ids:
                    connection_ids.remove(connection_id)
            
            # Remove connection
            del self.connections[connection_id]
            
            # Remove rate limit data
            if connection_id in self.rate_limits:
                del self.rate_limits[connection_id]
            
            logger.info(f"Connection cleaned up: {connection_id}")
            
        except Exception as e:
            logger.error(f"Connection cleanup failed for {connection_id}: {e}")
    
    def has_subscribers(self, symbols: List[str]) -> bool:
        """Check if there are subscribers for given symbols"""
        
        for symbol in symbols:
            if symbol in self.market_data_subscriptions:
                if self.market_data_subscriptions[symbol]:
                    return True
        
        # Check intelligence subscriptions
        return len(self.intelligence_subscriptions) > 0
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return len(self.connections)
    
    def get_subscription_stats(self) -> Dict[str, Any]:
        """Get subscription statistics"""
        
        return {
            "total_connections": len(self.connections),
            "intelligence_subscriptions": len(self.intelligence_subscriptions),
            "market_data_subscriptions": len(self.market_data_subscriptions),
            "alert_subscriptions": len(self.alert_subscriptions),
            "total_subscriptions": len(self.subscription_info)
        }
    
    async def _connection_monitor(self):
        """Background task to monitor connection health"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.utcnow()
                stale_connections = []
                
                for connection_id, connection in self.connections.items():
                    # Check for stale connections
                    time_since_ping = (current_time - connection.last_ping).total_seconds()
                    
                    if time_since_ping > self.config["connection_timeout"]:
                        stale_connections.append(connection_id)
                
                # Cleanup stale connections
                for connection_id in stale_connections:
                    logger.info(f"Cleaning up stale connection: {connection_id}")
                    await self._cleanup_connection(connection_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Connection monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _broadcast_processor(self):
        """Background task to process broadcast queue"""
        
        while True:
            try:
                # Process broadcast messages
                # For now, just wait - broadcasting is handled directly
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast processor error: {e}")
                await asyncio.sleep(1)
    
    async def _ping_connections(self):
        """Background task to ping connections"""
        
        while True:
            try:
                await asyncio.sleep(self.config["ping_interval"])
                
                for connection_id in list(self.connections.keys()):
                    try:
                        await self._send_ping(connection_id)
                    except Exception:
                        pass  # Connection will be cleaned up by monitor
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ping connections error: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_disconnected(self):
        """Background task to cleanup disconnected WebSockets"""
        
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                disconnected_connections = []
                
                for connection_id, connection in self.connections.items():
                    # Check if WebSocket is still connected
                    try:
                        if connection.websocket.client_state.name != "CONNECTED":
                            disconnected_connections.append(connection_id)
                    except Exception:
                        disconnected_connections.append(connection_id)
                
                # Cleanup disconnected connections
                for connection_id in disconnected_connections:
                    await self._cleanup_connection(connection_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup disconnected error: {e}")
                await asyncio.sleep(30)

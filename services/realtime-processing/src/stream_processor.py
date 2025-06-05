"""
Stream Processor for real-time data processing and event handling
"""

import asyncio
import aioredis
import json
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)

@dataclass
class StreamSubscription:
    stream_id: str
    user_id: str
    symbols: List[str]
    data_types: List[str]
    update_frequency: int
    last_update: datetime
    active: bool = True

@dataclass
class AlertSubscription:
    subscription_id: str
    user_id: str
    symbols: List[str]
    alert_types: List[str]
    conditions: Dict[str, Any]
    last_triggered: Optional[datetime] = None

@dataclass
class StreamEvent:
    event_type: str
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str

class StreamProcessor:
    def __init__(self):
        self.redis = None
        self.active_streams: Dict[str, StreamSubscription] = {}
        self.alert_subscriptions: Dict[str, AlertSubscription] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Processing queues
        self.event_queue = asyncio.Queue()
        self.alert_queue = asyncio.Queue()
        
        # Background tasks
        self.background_tasks = []
        
        # Stream configurations
        self.max_streams_per_user = 10
        self.max_events_per_second = 1000
        self.event_buffer_size = 10000
        
    async def start(self):
        """Start the stream processor"""
        try:
            # Initialize Redis connection
            self.redis = aioredis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            
            # Start background tasks
            self.background_tasks = [
                asyncio.create_task(self._event_processor()),
                asyncio.create_task(self._alert_processor()),
                asyncio.create_task(self._stream_updater()),
                asyncio.create_task(self._cleanup_inactive_streams())
            ]
            
            logger.info("Stream processor started successfully")
            
        except Exception as e:
            logger.error(f"Stream processor startup failed: {e}")
            raise
    
    async def stop(self):
        """Stop the stream processor"""
        try:
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Close Redis connection
            if self.redis:
                await self.redis.close()
            
            logger.info("Stream processor stopped")
            
        except Exception as e:
            logger.error(f"Stream processor shutdown error: {e}")
    
    async def health_check(self) -> str:
        """Check stream processor health"""
        try:
            if not self.redis:
                return "unhealthy - no redis connection"
            
            # Test Redis connection
            await self.redis.ping()
            
            # Check if background tasks are running
            running_tasks = sum(1 for task in self.background_tasks if not task.done())
            
            if running_tasks == len(self.background_tasks):
                return "healthy"
            else:
                return f"degraded - {running_tasks}/{len(self.background_tasks)} tasks running"
                
        except Exception as e:
            logger.error(f"Stream processor health check failed: {e}")
            return "unhealthy"
    
    async def register_stream(
        self,
        stream_id: str,
        user_id: str,
        symbols: List[str],
        data_types: List[str],
        update_frequency: int
    ):
        """Register a new stream subscription"""
        
        try:
            # Check user stream limits
            user_streams = [s for s in self.active_streams.values() if s.user_id == user_id]
            if len(user_streams) >= self.max_streams_per_user:
                raise Exception(f"User {user_id} has reached maximum stream limit")
            
            # Create subscription
            subscription = StreamSubscription(
                stream_id=stream_id,
                user_id=user_id,
                symbols=symbols,
                data_types=data_types,
                update_frequency=update_frequency,
                last_update=datetime.utcnow()
            )
            
            self.active_streams[stream_id] = subscription
            
            # Store in Redis for persistence
            await self.redis.hset(
                "stream_subscriptions",
                stream_id,
                json.dumps(asdict(subscription), default=str)
            )
            
            logger.info(f"Registered stream {stream_id} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Stream registration failed: {e}")
            raise
    
    async def unregister_stream(self, stream_id: str):
        """Unregister a stream subscription"""
        
        try:
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
            
            # Remove from Redis
            await self.redis.hdel("stream_subscriptions", stream_id)
            
            logger.info(f"Unregistered stream {stream_id}")
            
        except Exception as e:
            logger.error(f"Stream unregistration failed: {e}")
            raise
    
    async def register_alert_subscription(
        self,
        subscription_id: str,
        user_id: str,
        symbols: List[str],
        alert_types: List[str],
        conditions: Optional[Dict[str, Any]] = None
    ):
        """Register alert subscription"""
        
        try:
            subscription = AlertSubscription(
                subscription_id=subscription_id,
                user_id=user_id,
                symbols=symbols,
                alert_types=alert_types,
                conditions=conditions or {}
            )
            
            self.alert_subscriptions[subscription_id] = subscription
            
            # Store in Redis
            await self.redis.hset(
                "alert_subscriptions",
                subscription_id,
                json.dumps(asdict(subscription), default=str)
            )
            
            logger.info(f"Registered alert subscription {subscription_id} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Alert subscription registration failed: {e}")
            raise
    
    async def unregister_alert_subscription(self, subscription_id: str):
        """Unregister alert subscription"""
        
        try:
            if subscription_id in self.alert_subscriptions:
                del self.alert_subscriptions[subscription_id]
            
            await self.redis.hdel("alert_subscriptions", subscription_id)
            
            logger.info(f"Unregistered alert subscription {subscription_id}")
            
        except Exception as e:
            logger.error(f"Alert subscription unregistration failed: {e}")
            raise
    
    async def publish_event(self, event: StreamEvent):
        """Publish an event to the processing queue"""
        
        try:
            await self.event_queue.put(event)
            
        except Exception as e:
            logger.error(f"Event publishing failed: {e}")
    
    async def add_event_handler(self, event_type: str, handler: Callable):
        """Add an event handler for specific event types"""
        
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
    
    async def _event_processor(self):
        """Background task to process events"""
        
        while True:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Process event for relevant streams
                await self._process_event_for_streams(event)
                
                # Process event for alerts
                await self._process_event_for_alerts(event)
                
                # Call registered event handlers
                await self._call_event_handlers(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")
    
    async def _alert_processor(self):
        """Background task to process alerts"""
        
        while True:
            try:
                # Check for alert conditions
                await self._check_alert_conditions()
                
                # Process alert queue
                try:
                    alert = await asyncio.wait_for(self.alert_queue.get(), timeout=5.0)
                    await self._send_alert(alert)
                except asyncio.TimeoutError:
                    continue
                    
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
    
    async def _stream_updater(self):
        """Background task to update streams"""
        
        while True:
            try:
                current_time = datetime.utcnow()
                
                for stream_id, subscription in self.active_streams.items():
                    # Check if stream needs update
                    time_since_update = (current_time - subscription.last_update).total_seconds() * 1000
                    
                    if time_since_update >= subscription.update_frequency:
                        await self._update_stream(subscription)
                        subscription.last_update = current_time
                
                await asyncio.sleep(0.1)  # 100ms update cycle
                
            except Exception as e:
                logger.error(f"Stream updater error: {e}")
    
    async def _cleanup_inactive_streams(self):
        """Background task to cleanup inactive streams"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
                current_time = datetime.utcnow()
                inactive_streams = []
                
                for stream_id, subscription in self.active_streams.items():
                    # Mark streams inactive if no updates for 30 minutes
                    if (current_time - subscription.last_update).total_seconds() > 1800:
                        inactive_streams.append(stream_id)
                
                # Remove inactive streams
                for stream_id in inactive_streams:
                    await self.unregister_stream(stream_id)
                    logger.info(f"Cleaned up inactive stream {stream_id}")
                
            except Exception as e:
                logger.error(f"Stream cleanup error: {e}")
    
    async def _process_event_for_streams(self, event: StreamEvent):
        """Process event for relevant streams"""
        
        try:
            relevant_streams = [
                s for s in self.active_streams.values()
                if event.symbol in s.symbols and event.event_type in s.data_types
            ]
            
            for subscription in relevant_streams:
                # Send event to stream
                await self._send_event_to_stream(subscription, event)
                
        except Exception as e:
            logger.error(f"Stream event processing failed: {e}")
    
    async def _process_event_for_alerts(self, event: StreamEvent):
        """Process event for alert conditions"""
        
        try:
            relevant_alerts = [
                a for a in self.alert_subscriptions.values()
                if event.symbol in a.symbols and event.event_type in a.alert_types
            ]
            
            for alert_sub in relevant_alerts:
                if await self._check_alert_condition(alert_sub, event):
                    alert_data = {
                        "subscription_id": alert_sub.subscription_id,
                        "user_id": alert_sub.user_id,
                        "symbol": event.symbol,
                        "event_type": event.event_type,
                        "event_data": event.data,
                        "timestamp": event.timestamp.isoformat()
                    }
                    
                    await self.alert_queue.put(alert_data)
                    alert_sub.last_triggered = datetime.utcnow()
                    
        except Exception as e:
            logger.error(f"Alert event processing failed: {e}")
    
    async def _call_event_handlers(self, event: StreamEvent):
        """Call registered event handlers"""
        
        try:
            handlers = self.event_handlers.get(event.event_type, [])
            
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
                    
        except Exception as e:
            logger.error(f"Event handler calling failed: {e}")
    
    async def _update_stream(self, subscription: StreamSubscription):
        """Update a specific stream with latest data"""
        
        try:
            # Get latest data for stream symbols
            stream_data = await self._get_stream_data(
                subscription.symbols, 
                subscription.data_types
            )
            
            # Send data to stream subscribers
            await self._send_stream_update(subscription, stream_data)
            
        except Exception as e:
            logger.error(f"Stream update failed: {e}")
    
    async def _get_stream_data(
        self, 
        symbols: List[str], 
        data_types: List[str]
    ) -> Dict[str, Any]:
        """Get latest data for stream"""
        
        # This would integrate with actual data sources
        # For now, return mock data
        return {
            "symbols": symbols,
            "data_types": data_types,
            "data": {
                symbol: {
                    "price": 100.0 + (hash(symbol) % 100),
                    "volume": 1000000,
                    "timestamp": datetime.utcnow().isoformat()
                }
                for symbol in symbols
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _send_event_to_stream(
        self, 
        subscription: StreamSubscription, 
        event: StreamEvent
    ):
        """Send event to stream subscribers"""
        
        try:
            # Publish to Redis channel for WebSocket subscribers
            channel = f"stream:{subscription.stream_id}"
            
            event_data = {
                "event_type": event.event_type,
                "symbol": event.symbol,
                "data": event.data,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source
            }
            
            await self.redis.publish(channel, json.dumps(event_data))
            
        except Exception as e:
            logger.error(f"Event sending failed: {e}")
    
    async def _send_stream_update(
        self, 
        subscription: StreamSubscription, 
        data: Dict[str, Any]
    ):
        """Send stream update to subscribers"""
        
        try:
            channel = f"stream:{subscription.stream_id}"
            
            update_data = {
                "type": "stream_update",
                "stream_id": subscription.stream_id,
                "data": data
            }
            
            await self.redis.publish(channel, json.dumps(update_data))
            
        except Exception as e:
            logger.error(f"Stream update sending failed: {e}")
    
    async def _check_alert_conditions(self):
        """Check alert conditions for all subscriptions"""
        
        try:
            for alert_sub in self.alert_subscriptions.values():
                # Skip if recently triggered
                if (alert_sub.last_triggered and 
                    (datetime.utcnow() - alert_sub.last_triggered).total_seconds() < 60):
                    continue
                
                # Check conditions (simplified)
                for symbol in alert_sub.symbols:
                    # Get current market data
                    market_data = await self._get_current_market_data(symbol)
                    
                    # Check alert conditions
                    if await self._evaluate_alert_conditions(alert_sub, symbol, market_data):
                        alert_data = {
                            "subscription_id": alert_sub.subscription_id,
                            "user_id": alert_sub.user_id,
                            "symbol": symbol,
                            "alert_type": "condition_met",
                            "data": market_data,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        await self.alert_queue.put(alert_data)
                        alert_sub.last_triggered = datetime.utcnow()
                        
        except Exception as e:
            logger.error(f"Alert condition checking failed: {e}")
    
    async def _check_alert_condition(
        self, 
        alert_sub: AlertSubscription, 
        event: StreamEvent
    ) -> bool:
        """Check if event triggers alert condition"""
        
        try:
            # Simple condition checking
            conditions = alert_sub.conditions
            
            if not conditions:
                return True  # No conditions = always trigger
            
            # Check price conditions
            if "price_threshold" in conditions:
                current_price = event.data.get("price", 0)
                threshold = conditions["price_threshold"]
                
                if current_price > threshold:
                    return True
            
            # Check volume conditions
            if "volume_threshold" in conditions:
                current_volume = event.data.get("volume", 0)
                threshold = conditions["volume_threshold"]
                
                if current_volume > threshold:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Alert condition check failed: {e}")
            return False
    
    async def _get_current_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for symbol"""
        
        # Mock implementation
        return {
            "symbol": symbol,
            "price": 100.0 + (hash(symbol) % 100),
            "volume": 1000000,
            "change": 0.5,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _evaluate_alert_conditions(
        self, 
        alert_sub: AlertSubscription, 
        symbol: str, 
        market_data: Dict[str, Any]
    ) -> bool:
        """Evaluate alert conditions for symbol"""
        
        # Simplified condition evaluation
        conditions = alert_sub.conditions
        
        if "price_change_threshold" in conditions:
            change = market_data.get("change", 0)
            threshold = conditions["price_change_threshold"]
            
            if abs(change) > threshold:
                return True
        
        return False
    
    async def _send_alert(self, alert_data: Dict[str, Any]):
        """Send alert to user"""
        
        try:
            # Publish alert to Redis channel
            channel = f"alerts:{alert_data['user_id']}"
            await self.redis.publish(channel, json.dumps(alert_data))
            
            logger.info(f"Alert sent to user {alert_data['user_id']}: {alert_data['symbol']}")
            
        except Exception as e:
            logger.error(f"Alert sending failed: {e}")

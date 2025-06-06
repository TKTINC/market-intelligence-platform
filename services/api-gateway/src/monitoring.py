"""
System Monitoring and Metrics Collection
"""

import asyncio
import aioredis
import psutil
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import statistics
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_in: int
    network_out: int
    active_connections: int
    request_rate: float
    error_rate: float
    response_time_avg: float

@dataclass
class ServiceMetrics:
    service_name: str
    timestamp: datetime
    status: str
    response_time_ms: float
    request_count: int
    error_count: int
    success_rate: float
    cpu_usage: float
    memory_usage: float

@dataclass
class BusinessMetrics:
    timestamp: datetime
    total_users: int
    active_users: int
    total_portfolios: int
    total_trades: int
    total_volume: float
    agent_analysis_requests: int
    api_calls: int
    websocket_connections: int

class GatewayMonitoring:
    def __init__(self):
        self.redis = None
        
        # Monitoring configuration
        self.config = {
            "metrics_retention_days": 30,
            "alert_thresholds": {
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "disk_usage": 90.0,
                "error_rate": 5.0,
                "response_time_ms": 1000.0
            },
            "sampling_interval": 60,  # Sample every minute
            "aggregation_intervals": [60, 300, 3600],  # 1min, 5min, 1hour
            "alert_cooldown_minutes": 15
        }
        
        # Metrics storage
        self.system_metrics_buffer = []
        self.service_metrics_buffer = {}
        self.business_metrics_buffer = []
        
        # Performance counters
        self.request_counter = 0
        self.error_counter = 0
        self.response_times = []
        self.last_metrics_time = time.time()
        
        # Alert state
        self.active_alerts = {}
        self.alert_history = []
        
        # Service health status
        self.service_health = {}
        
    async def initialize(self):
        """Initialize the monitoring system"""
        try:
            # Initialize Redis connection
            self.redis = aioredis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            
            # Load existing metrics
            await self._load_historical_metrics()
            
            # Start background tasks
            asyncio.create_task(self._metrics_collection_task())
            asyncio.create_task(self._metrics_aggregation_task())
            asyncio.create_task(self._alert_monitoring_task())
            asyncio.create_task(self._cleanup_task())
            
            logger.info("Gateway monitoring initialized")
            
        except Exception as e:
            logger.error(f"Monitoring initialization failed: {e}")
            raise
    
    async def close(self):
        """Close the monitoring system"""
        if self.redis:
            await self.redis.close()
    
    async def record_request(self, response_time_ms: float, status_code: int):
        """Record API request metrics"""
        
        try:
            self.request_counter += 1
            self.response_times.append(response_time_ms)
            
            if status_code >= 400:
                self.error_counter += 1
            
            # Keep only recent response times (last 1000 requests)
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]
            
        except Exception as e:
            logger.error(f"Request recording failed: {e}")
    
    async def record_agent_analysis(
        self,
        request_id: str,
        user_id: str,
        symbols: List[str],
        processing_time_ms: int,
        cost_usd: float
    ):
        """Record agent analysis metrics"""
        
        try:
            metrics = {
                "request_id": request_id,
                "user_id": user_id,
                "symbol_count": len(symbols),
                "processing_time_ms": processing_time_ms,
                "cost_usd": cost_usd,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store in Redis
            await self.redis.lpush(
                "agent_analysis_metrics",
                json.dumps(metrics)
            )
            
            # Keep only recent metrics
            await self.redis.ltrim("agent_analysis_metrics", 0, 9999)
            
        except Exception as e:
            logger.error(f"Agent analysis recording failed: {e}")
    
    async def record_trade_execution(
        self,
        trade_id: str,
        user_id: str,
        symbol: str,
        trade_value: float
    ):
        """Record trade execution metrics"""
        
        try:
            metrics = {
                "trade_id": trade_id,
                "user_id": user_id,
                "symbol": symbol,
                "trade_value": trade_value,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store in Redis
            await self.redis.lpush(
                "trade_execution_metrics",
                json.dumps(metrics)
            )
            
            # Keep only recent metrics
            await self.redis.ltrim("trade_execution_metrics", 0, 9999)
            
        except Exception as e:
            logger.error(f"Trade execution recording failed: {e}")
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Calculate request metrics
            current_time = time.time()
            time_diff = current_time - self.last_metrics_time
            
            if time_diff > 0:
                request_rate = self.request_counter / time_diff
                error_rate = (self.error_counter / max(1, self.request_counter)) * 100
            else:
                request_rate = 0.0
                error_rate = 0.0
            
            avg_response_time = statistics.mean(self.response_times) if self.response_times else 0.0
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "system": {
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory.percent,
                    "disk_usage": (disk.used / disk.total) * 100,
                    "network_in_bytes": network.bytes_recv,
                    "network_out_bytes": network.bytes_sent
                },
                "api": {
                    "request_rate_per_second": round(request_rate, 2),
                    "error_rate_percent": round(error_rate, 2),
                    "avg_response_time_ms": round(avg_response_time, 2),
                    "total_requests": self.request_counter,
                    "total_errors": self.error_counter
                },
                "services": self.service_health
            }
            
        except Exception as e:
            logger.error(f"Current metrics retrieval failed: {e}")
            return {"error": str(e)}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics over time"""
        
        try:
            # Get metrics from different time intervals
            metrics_1h = await self._get_aggregated_metrics("1h")
            metrics_24h = await self._get_aggregated_metrics("24h")
            metrics_7d = await self._get_aggregated_metrics("7d")
            
            return {
                "last_hour": metrics_1h,
                "last_24_hours": metrics_24h,
                "last_7_days": metrics_7d,
                "current": await self.get_current_metrics()
            }
            
        except Exception as e:
            logger.error(f"Performance metrics retrieval failed: {e}")
            return {"error": str(e)}
    
    async def get_admin_metrics(self) -> Dict[str, Any]:
        """Get comprehensive admin metrics"""
        
        try:
            # System metrics
            system_metrics = await self.get_current_metrics()
            
            # Business metrics
            business_metrics = await self._get_business_metrics()
            
            # Service metrics
            service_metrics = await self._get_service_metrics()
            
            # Alert metrics
            alert_metrics = await self._get_alert_metrics()
            
            # Resource utilization trends
            utilization_trends = await self._get_utilization_trends()
            
            return {
                "system": system_metrics,
                "business": business_metrics,
                "services": service_metrics,
                "alerts": alert_metrics,
                "trends": utilization_trends,
                "uptime": await self._get_uptime_metrics()
            }
            
        except Exception as e:
            logger.error(f"Admin metrics retrieval failed: {e}")
            return {"error": str(e)}
    
    async def create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metric_value: float,
        threshold: float
    ):
        """Create system alert"""
        
        try:
            alert_id = f"{alert_type}_{int(time.time())}"
            
            # Check if similar alert is in cooldown
            if await self._is_alert_in_cooldown(alert_type):
                return
            
            alert = {
                "alert_id": alert_id,
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "metric_value": metric_value,
                "threshold": threshold,
                "timestamp": datetime.utcnow().isoformat(),
                "acknowledged": False,
                "resolved": False
            }
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            await self.redis.setex(
                f"alert:{alert_id}",
                86400 * 7,  # 7 days TTL
                json.dumps(alert)
            )
            
            # Set cooldown
            await self.redis.setex(
                f"alert_cooldown:{alert_type}",
                self.config["alert_cooldown_minutes"] * 60,
                "1"
            )
            
            logger.warning(f"Alert created: {alert_type} - {message}")
            
        except Exception as e:
            logger.error(f"Alert creation failed: {e}")
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id]["acknowledged"] = True
                self.active_alerts[alert_id]["acknowledged_at"] = datetime.utcnow().isoformat()
                
                # Update in Redis
                await self.redis.set(
                    f"alert:{alert_id}",
                    json.dumps(self.active_alerts[alert_id])
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Alert acknowledgment failed: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id]["resolved"] = True
                self.active_alerts[alert_id]["resolved_at"] = datetime.utcnow().isoformat()
                
                # Update in Redis
                await self.redis.set(
                    f"alert:{alert_id}",
                    json.dumps(self.active_alerts[alert_id])
                )
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Alert resolution failed: {e}")
            return False
    
    async def update_service_health(self, service_name: str, health_status: str, metrics: Dict[str, Any] = None):
        """Update service health status"""
        
        try:
            self.service_health[service_name] = {
                "status": health_status,
                "last_updated": datetime.utcnow().isoformat(),
                "metrics": metrics or {}
            }
            
            # Store in Redis
            await self.redis.setex(
                f"service_health:{service_name}",
                300,  # 5 minutes TTL
                json.dumps(self.service_health[service_name])
            )
            
        except Exception as e:
            logger.error(f"Service health update failed: {e}")
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        
        try:
            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Application metrics
            current_time = time.time()
            time_diff = current_time - self.last_metrics_time
            
            if time_diff > 0 and self.request_counter > 0:
                request_rate = self.request_counter / time_diff
                error_rate = (self.error_counter / self.request_counter) * 100
            else:
                request_rate = 0.0
                error_rate = 0.0
            
            avg_response_time = statistics.mean(self.response_times) if self.response_times else 0.0
            
            # Reset counters
            self.request_counter = 0
            self.error_counter = 0
            self.response_times = []
            self.last_metrics_time = current_time
            
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                network_in=network.bytes_recv,
                network_out=network.bytes_sent,
                active_connections=len(self.service_health),  # Simplified
                request_rate=request_rate,
                error_rate=error_rate,
                response_time_avg=avg_response_time
            )
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
            return None
    
    async def _store_metrics(self, metrics: SystemMetrics):
        """Store metrics in Redis"""
        
        try:
            timestamp = metrics.timestamp.timestamp()
            
            # Store in time series
            await self.redis.zadd(
                "system_metrics",
                {json.dumps(asdict(metrics), default=str): timestamp}
            )
            
            # Store aggregated metrics for different intervals
            for interval in self.config["aggregation_intervals"]:
                interval_key = f"system_metrics_{interval}s"
                
                # Get window start time
                window_start = timestamp - (timestamp % interval)
                
                # Store in interval bucket
                await self.redis.zadd(
                    interval_key,
                    {json.dumps(asdict(metrics), default=str): window_start}
                )
            
        except Exception as e:
            logger.error(f"Metrics storage failed: {e}")
    
    async def _get_aggregated_metrics(self, timeframe: str) -> Dict[str, Any]:
        """Get aggregated metrics for timeframe"""
        
        try:
            now = datetime.utcnow()
            
            if timeframe == "1h":
                start_time = now - timedelta(hours=1)
                key = "system_metrics_60s"
            elif timeframe == "24h":
                start_time = now - timedelta(hours=24)
                key = "system_metrics_3600s"
            elif timeframe == "7d":
                start_time = now - timedelta(days=7)
                key = "system_metrics_3600s"
            else:
                return {}
            
            # Get metrics from Redis
            metrics_data = await self.redis.zrangebyscore(
                key,
                start_time.timestamp(),
                now.timestamp()
            )
            
            if not metrics_data:
                return {}
            
            # Parse and aggregate
            parsed_metrics = []
            for data in metrics_data:
                metric = json.loads(data)
                parsed_metrics.append(metric)
            
            if not parsed_metrics:
                return {}
            
            # Calculate aggregations
            cpu_values = [m["cpu_usage"] for m in parsed_metrics]
            memory_values = [m["memory_usage"] for m in parsed_metrics]
            response_time_values = [m["response_time_avg"] for m in parsed_metrics]
            error_rate_values = [m["error_rate"] for m in parsed_metrics]
            
            return {
                "timeframe": timeframe,
                "data_points": len(parsed_metrics),
                "cpu_usage": {
                    "avg": statistics.mean(cpu_values),
                    "max": max(cpu_values),
                    "min": min(cpu_values)
                },
                "memory_usage": {
                    "avg": statistics.mean(memory_values),
                    "max": max(memory_values),
                    "min": min(memory_values)
                },
                "response_time": {
                    "avg": statistics.mean(response_time_values),
                    "max": max(response_time_values),
                    "min": min(response_time_values)
                },
                "error_rate": {
                    "avg": statistics.mean(error_rate_values),
                    "max": max(error_rate_values),
                    "min": min(error_rate_values)
                }
            }
            
        except Exception as e:
            logger.error(f"Aggregated metrics retrieval failed: {e}")
            return {}
    
    async def _get_business_metrics(self) -> Dict[str, Any]:
        """Get business-specific metrics"""
        
        try:
            # Get metrics from Redis
            agent_metrics = await self.redis.lrange("agent_analysis_metrics", 0, 99)
            trade_metrics = await self.redis.lrange("trade_execution_metrics", 0, 99)
            
            # Parse and aggregate
            total_agent_requests = len(agent_metrics)
            total_trades = len(trade_metrics)
            
            if agent_metrics:
                agent_costs = [json.loads(m)["cost_usd"] for m in agent_metrics]
                total_agent_cost = sum(agent_costs)
            else:
                total_agent_cost = 0.0
            
            if trade_metrics:
                trade_values = [json.loads(m)["trade_value"] for m in trade_metrics]
                total_trade_volume = sum(trade_values)
            else:
                total_trade_volume = 0.0
            
            return {
                "agent_analysis": {
                    "total_requests": total_agent_requests,
                    "total_cost_usd": total_agent_cost
                },
                "trading": {
                    "total_trades": total_trades,
                    "total_volume_usd": total_trade_volume
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Business metrics retrieval failed: {e}")
            return {}
    
    async def _get_service_metrics(self) -> Dict[str, Any]:
        """Get service health metrics"""
        
        try:
            service_metrics = {}
            
            for service_name, health_data in self.service_health.items():
                service_metrics[service_name] = {
                    "status": health_data["status"],
                    "last_updated": health_data["last_updated"],
                    "metrics": health_data.get("metrics", {})
                }
            
            return service_metrics
            
        except Exception as e:
            logger.error(f"Service metrics retrieval failed: {e}")
            return {}
    
    async def _get_alert_metrics(self) -> Dict[str, Any]:
        """Get alert metrics"""
        
        try:
            active_alert_count = len(self.active_alerts)
            total_alert_count = len(self.alert_history)
            
            # Count alerts by severity
            severity_counts = {}
            for alert in self.active_alerts.values():
                severity = alert.get("severity", "unknown")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            return {
                "active_alerts": active_alert_count,
                "total_alerts": total_alert_count,
                "alerts_by_severity": severity_counts,
                "recent_alerts": list(self.active_alerts.values())[-10:]  # Last 10 alerts
            }
            
        except Exception as e:
            logger.error(f"Alert metrics retrieval failed: {e}")
            return {}
    
    async def _get_utilization_trends(self) -> Dict[str, Any]:
        """Get resource utilization trends"""
        
        try:
            # Get hourly metrics for the last 24 hours
            now = datetime.utcnow()
            start_time = now - timedelta(hours=24)
            
            metrics_data = await self.redis.zrangebyscore(
                "system_metrics_3600s",
                start_time.timestamp(),
                now.timestamp()
            )
            
            if not metrics_data:
                return {}
            
            # Parse metrics
            hourly_data = []
            for data in metrics_data:
                metric = json.loads(data)
                hourly_data.append({
                    "timestamp": metric["timestamp"],
                    "cpu_usage": metric["cpu_usage"],
                    "memory_usage": metric["memory_usage"],
                    "response_time": metric["response_time_avg"]
                })
            
            return {
                "timeframe": "24h",
                "interval": "1h",
                "data": hourly_data
            }
            
        except Exception as e:
            logger.error(f"Utilization trends retrieval failed: {e}")
            return {}
    
    async def _get_uptime_metrics(self) -> Dict[str, Any]:
        """Get system uptime metrics"""
        
        try:
            # Get system boot time
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.utcnow() - boot_time
            
            # Get application uptime (simplified - would track application start time)
            app_uptime = uptime  # Placeholder
            
            return {
                "system_uptime_seconds": uptime.total_seconds(),
                "application_uptime_seconds": app_uptime.total_seconds(),
                "system_boot_time": boot_time.isoformat(),
                "application_start_time": boot_time.isoformat()  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Uptime metrics retrieval failed: {e}")
            return {}
    
    async def _is_alert_in_cooldown(self, alert_type: str) -> bool:
        """Check if alert type is in cooldown"""
        
        try:
            cooldown = await self.redis.get(f"alert_cooldown:{alert_type}")
            return cooldown is not None
            
        except Exception as e:
            logger.error(f"Alert cooldown check failed: {e}")
            return False
    
    async def _load_historical_metrics(self):
        """Load historical metrics from Redis"""
        
        try:
            # Load active alerts
            alert_keys = await self.redis.keys("alert:*")
            
            for key in alert_keys:
                alert_data = await self.redis.get(key)
                if alert_data:
                    alert = json.loads(alert_data)
                    alert_id = alert["alert_id"]
                    
                    if not alert.get("resolved", False):
                        self.active_alerts[alert_id] = alert
                    
                    self.alert_history.append(alert)
            
            # Load service health
            service_keys = await self.redis.keys("service_health:*")
            
            for key in service_keys:
                service_data = await self.redis.get(key)
                if service_data:
                    service_name = key.split(":", 1)[1]
                    self.service_health[service_name] = json.loads(service_data)
            
            logger.info(f"Loaded monitoring data: {len(self.active_alerts)} active alerts, {len(self.service_health)} services")
            
        except Exception as e:
            logger.error(f"Historical metrics loading failed: {e}")
    
    # Background tasks
    async def _metrics_collection_task(self):
        """Background task to collect system metrics"""
        
        while True:
            try:
                await asyncio.sleep(self.config["sampling_interval"])
                
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                if metrics:
                    # Store metrics
                    await self._store_metrics(metrics)
                    
                    # Add to buffer
                    self.system_metrics_buffer.append(metrics)
                    
                    # Keep buffer size manageable
                    if len(self.system_metrics_buffer) > 1000:
                        self.system_metrics_buffer = self.system_metrics_buffer[-1000:]
                
            except Exception as e:
                logger.error(f"Metrics collection task error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_aggregation_task(self):
        """Background task to aggregate metrics"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Aggregate metrics for different time windows
                # This is a simplified version - would implement proper time-window aggregation
                
                if self.system_metrics_buffer:
                    # Calculate 5-minute aggregates
                    recent_metrics = self.system_metrics_buffer[-5:]  # Last 5 samples
                    
                    if recent_metrics:
                        avg_cpu = statistics.mean([m.cpu_usage for m in recent_metrics])
                        avg_memory = statistics.mean([m.memory_usage for m in recent_metrics])
                        avg_response_time = statistics.mean([m.response_time_avg for m in recent_metrics])
                        
                        # Store aggregated metrics
                        aggregated = {
                            "timestamp": datetime.utcnow().isoformat(),
                            "window": "5min",
                            "cpu_usage_avg": avg_cpu,
                            "memory_usage_avg": avg_memory,
                            "response_time_avg": avg_response_time
                        }
                        
                        await self.redis.lpush(
                            "aggregated_metrics_5min",
                            json.dumps(aggregated)
                        )
                        
                        # Keep only recent aggregates
                        await self.redis.ltrim("aggregated_metrics_5min", 0, 287)  # 24 hours worth
                
            except Exception as e:
                logger.error(f"Metrics aggregation task error: {e}")
                await asyncio.sleep(300)
    
    async def _alert_monitoring_task(self):
        """Background task to monitor for alert conditions"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Get current metrics
                current_metrics = await self.get_current_metrics()
                
                if "system" in current_metrics:
                    system = current_metrics["system"]
                    
                    # Check CPU usage
                    if system["cpu_usage"] > self.config["alert_thresholds"]["cpu_usage"]:
                        await self.create_alert(
                            "high_cpu_usage",
                            "warning",
                            f"CPU usage is {system['cpu_usage']:.1f}%",
                            system["cpu_usage"],
                            self.config["alert_thresholds"]["cpu_usage"]
                        )
                    
                    # Check memory usage
                    if system["memory_usage"] > self.config["alert_thresholds"]["memory_usage"]:
                        await self.create_alert(
                            "high_memory_usage",
                            "warning",
                            f"Memory usage is {system['memory_usage']:.1f}%",
                            system["memory_usage"],
                            self.config["alert_thresholds"]["memory_usage"]
                        )
                    
                    # Check disk usage
                    if system["disk_usage"] > self.config["alert_thresholds"]["disk_usage"]:
                        await self.create_alert(
                            "high_disk_usage",
                            "critical",
                            f"Disk usage is {system['disk_usage']:.1f}%",
                            system["disk_usage"],
                            self.config["alert_thresholds"]["disk_usage"]
                        )
                
                if "api" in current_metrics:
                    api = current_metrics["api"]
                    
                    # Check error rate
                    if api["error_rate_percent"] > self.config["alert_thresholds"]["error_rate"]:
                        await self.create_alert(
                            "high_error_rate",
                            "critical",
                            f"Error rate is {api['error_rate_percent']:.1f}%",
                            api["error_rate_percent"],
                            self.config["alert_thresholds"]["error_rate"]
                        )
                    
                    # Check response time
                    if api["avg_response_time_ms"] > self.config["alert_thresholds"]["response_time_ms"]:
                        await self.create_alert(
                            "high_response_time",
                            "warning",
                            f"Average response time is {api['avg_response_time_ms']:.1f}ms",
                            api["avg_response_time_ms"],
                            self.config["alert_thresholds"]["response_time_ms"]
                        )
                
            except Exception as e:
                logger.error(f"Alert monitoring task error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_task(self):
        """Background task to cleanup old metrics and alerts"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Cleanup old metrics
                cutoff_time = datetime.utcnow() - timedelta(days=self.config["metrics_retention_days"])
                cutoff_timestamp = cutoff_time.timestamp()
                
                # Cleanup system metrics
                await self.redis.zremrangebyscore("system_metrics", 0, cutoff_timestamp)
                
                for interval in self.config["aggregation_intervals"]:
                    key = f"system_metrics_{interval}s"
                    await self.redis.zremrangebyscore(key, 0, cutoff_timestamp)
                
                # Cleanup old alerts
                current_time = datetime.utcnow()
                old_alert_keys = []
                for alert_id, alert in list(self.active_alerts.items()):
                    alert_time = datetime.fromisoformat(alert["timestamp"])
                    if current_time - alert_time > timedelta(days=7):  # Auto-resolve 7-day-old alerts
                        await self.resolve_alert(alert_id)
                        old_alert_keys.append(f"alert:{alert_id}")
                
                # Remove old alert records
                for key in old_alert_keys:
                    await self.redis.delete(key)
                
                logger.info("Monitoring cleanup completed")
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(3600)

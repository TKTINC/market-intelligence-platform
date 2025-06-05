"""
Real-time Metrics and Performance Monitoring
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import statistics
import aioredis

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int

@dataclass
class AgentMetrics:
    agent_name: str
    timestamp: datetime
    response_time_ms: float
    success_rate: float
    requests_per_minute: int
    error_rate: float
    availability: float
    health_score: float

@dataclass
class RequestMetrics:
    timestamp: datetime
    request_id: str
    symbols: List[str]
    processing_time_ms: int
    agents_used: List[str]
    confidence_score: float
    cache_hit: bool
    user_tier: str

@dataclass
class PerformanceAlert:
    alert_id: str
    timestamp: datetime
    severity: str  # low, medium, high, critical
    category: str  # system, agent, performance, error
    message: str
    details: Dict[str, Any]
    resolved: bool = False

class RealTimeMetrics:
    def __init__(self):
        self.redis = None
        
        # Metric storage
        self.system_metrics_history: List[SystemMetrics] = []
        self.agent_metrics_history: Dict[str, List[AgentMetrics]] = {}
        self.request_metrics_history: List[RequestMetrics] = []
        self.performance_alerts: List[PerformanceAlert] = []
        
        # Real-time counters
        self.counters = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "agent_calls": 0,
            "websocket_connections": 0,
            "alerts_generated": 0
        }
        
        # Performance thresholds
        self.thresholds = {
            "cpu_warning": 70.0,
            "cpu_critical": 85.0,
            "memory_warning": 80.0,
            "memory_critical": 90.0,
            "response_time_warning": 5000,  # 5 seconds
            "response_time_critical": 10000,  # 10 seconds
            "error_rate_warning": 0.05,  # 5%
            "error_rate_critical": 0.15,  # 15%
            "agent_availability_warning": 0.95,  # 95%
            "agent_availability_critical": 0.85  # 85%
        }
        
        # Configuration
        self.config = {
            "metrics_retention_hours": 24,
            "collection_interval": 30,  # seconds
            "aggregation_interval": 300,  # 5 minutes
            "alert_check_interval": 60,  # 1 minute
            "max_alerts": 1000
        }
        
        # Background tasks
        self.background_tasks = []
        
        # Performance data aggregation
        self.aggregated_metrics = {
            "hourly": {},
            "daily": {},
            "weekly": {}
        }
        
    async def start(self):
        """Start the monitoring system"""
        try:
            # Initialize Redis connection
            self.redis = aioredis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            
            # Start background tasks
            self.background_tasks = [
                asyncio.create_task(self._system_metrics_collector()),
                asyncio.create_task(self._agent_metrics_collector()),
                asyncio.create_task(self._metrics_aggregator()),
                asyncio.create_task(self._alert_monitor()),
                asyncio.create_task(self._metrics_cleanup())
            ]
            
            logger.info("Real-time monitoring started successfully")
            
        except Exception as e:
            logger.error(f"Monitoring startup failed: {e}")
            raise
    
    async def stop(self):
        """Stop the monitoring system"""
        try:
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Close Redis connection
            if self.redis:
                await self.redis.close()
            
            logger.info("Monitoring stopped")
            
        except Exception as e:
            logger.error(f"Monitoring shutdown error: {e}")
    
    async def record_unified_request(
        self,
        request_id: str,
        symbols: List[str],
        processing_time_ms: int,
        agents_used: List[str],
        confidence_score: float,
        cache_hit: bool = False,
        user_tier: str = "free"
    ):
        """Record metrics for a unified intelligence request"""
        
        try:
            # Create request metrics
            request_metrics = RequestMetrics(
                timestamp=datetime.utcnow(),
                request_id=request_id,
                symbols=symbols,
                processing_time_ms=processing_time_ms,
                agents_used=agents_used,
                confidence_score=confidence_score,
                cache_hit=cache_hit,
                user_tier=user_tier
            )
            
            # Store in history
            self.request_metrics_history.append(request_metrics)
            
            # Update counters
            self.counters["total_requests"] += 1
            if confidence_score > 0.5:
                self.counters["successful_requests"] += 1
            else:
                self.counters["failed_requests"] += 1
            
            if cache_hit:
                self.counters["cache_hits"] += 1
            else:
                self.counters["cache_misses"] += 1
            
            self.counters["agent_calls"] += len(agents_used)
            
            # Store in Redis for persistence
            await self._store_request_metrics(request_metrics)
            
            # Check for performance alerts
            await self._check_request_performance_alerts(request_metrics)
            
        except Exception as e:
            logger.error(f"Request metrics recording failed: {e}")
    
    async def record_agent_performance(
        self,
        agent_name: str,
        response_time_ms: float,
        success: bool
    ):
        """Record agent performance metrics"""
        
        try:
            current_time = datetime.utcnow()
            
            # Calculate metrics for this agent
            if agent_name not in self.agent_metrics_history:
                self.agent_metrics_history[agent_name] = []
            
            # Get recent metrics for calculations
            recent_metrics = [
                m for m in self.agent_metrics_history[agent_name]
                if (current_time - m.timestamp).total_seconds() < 300  # Last 5 minutes
            ]
            
            # Calculate success rate
            if recent_metrics:
                recent_successes = sum(1 for m in recent_metrics if m.success_rate > 0.9)
                success_rate = recent_successes / len(recent_metrics)
            else:
                success_rate = 1.0 if success else 0.0
            
            # Calculate requests per minute
            requests_per_minute = len(recent_metrics) * (60 / 300)  # Scale to per minute
            
            # Calculate error rate
            error_rate = 1.0 - success_rate
            
            # Calculate availability (simplified)
            availability = success_rate
            
            # Calculate health score
            health_score = self._calculate_agent_health_score(
                response_time_ms, success_rate, availability
            )
            
            # Create agent metrics
            agent_metrics = AgentMetrics(
                agent_name=agent_name,
                timestamp=current_time,
                response_time_ms=response_time_ms,
                success_rate=success_rate,
                requests_per_minute=requests_per_minute,
                error_rate=error_rate,
                availability=availability,
                health_score=health_score
            )
            
            # Store in history
            self.agent_metrics_history[agent_name].append(agent_metrics)
            
            # Store in Redis
            await self._store_agent_metrics(agent_metrics)
            
            # Check for agent alerts
            await self._check_agent_performance_alerts(agent_metrics)
            
        except Exception as e:
            logger.error(f"Agent performance recording failed: {e}")
    
    async def get_current_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        
        try:
            # Get latest system metrics
            latest_system = self.system_metrics_history[-1] if self.system_metrics_history else None
            
            # Calculate average response times
            recent_requests = [
                r for r in self.request_metrics_history
                if (datetime.utcnow() - r.timestamp).total_seconds() < 300
            ]
            
            avg_response_time = (
                statistics.mean(r.processing_time_ms for r in recent_requests)
                if recent_requests else 0
            )
            
            # Calculate cache hit ratio
            total_cache_requests = self.counters["cache_hits"] + self.counters["cache_misses"]
            cache_hit_ratio = (
                self.counters["cache_hits"] / total_cache_requests
                if total_cache_requests > 0 else 0
            )
            
            # Calculate success rate
            total_requests = self.counters["successful_requests"] + self.counters["failed_requests"]
            success_rate = (
                self.counters["successful_requests"] / total_requests
                if total_requests > 0 else 1.0
            )
            
            return {
                "system": {
                    "cpu_percent": latest_system.cpu_percent if latest_system else 0,
                    "memory_percent": latest_system.memory_percent if latest_system else 0,
                    "memory_used_mb": latest_system.memory_used_mb if latest_system else 0,
                    "disk_usage_percent": latest_system.disk_usage_percent if latest_system else 0,
                    "process_count": latest_system.process_count if latest_system else 0
                },
                "performance": {
                    "total_requests": self.counters["total_requests"],
                    "success_rate": round(success_rate, 3),
                    "avg_response_time_ms": round(avg_response_time, 1),
                    "cache_hit_ratio": round(cache_hit_ratio, 3),
                    "requests_last_5min": len(recent_requests)
                },
                "agents": self._get_agent_summary_stats(),
                "alerts": {
                    "active_alerts": len([a for a in self.performance_alerts if not a.resolved]),
                    "total_alerts": len(self.performance_alerts),
                    "critical_alerts": len([a for a in self.performance_alerts if a.severity == "critical" and not a.resolved])
                },
                "websockets": {
                    "active_connections": self.counters["websocket_connections"]
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Current stats retrieval failed: {e}")
            return {"error": str(e)}
    
    async def get_detailed_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        
        try:
            # Get historical data
            hourly_data = await self._get_hourly_metrics()
            daily_data = await self._get_daily_metrics()
            
            # Get agent performance details
            agent_details = {}
            for agent_name, metrics_list in self.agent_metrics_history.items():
                recent_metrics = [
                    m for m in metrics_list
                    if (datetime.utcnow() - m.timestamp).total_seconds() < 3600  # Last hour
                ]
                
                if recent_metrics:
                    agent_details[agent_name] = {
                        "avg_response_time_ms": statistics.mean(m.response_time_ms for m in recent_metrics),
                        "avg_success_rate": statistics.mean(m.success_rate for m in recent_metrics),
                        "avg_health_score": statistics.mean(m.health_score for m in recent_metrics),
                        "total_requests": len(recent_metrics),
                        "availability": statistics.mean(m.availability for m in recent_metrics)
                    }
            
            # Get recent alerts
            recent_alerts = [
                asdict(alert) for alert in self.performance_alerts[-50:]  # Last 50 alerts
            ]
            
            # Convert datetime objects to strings
            for alert in recent_alerts:
                alert["timestamp"] = alert["timestamp"].isoformat() if isinstance(alert["timestamp"], datetime) else alert["timestamp"]
            
            return {
                "current_stats": await self.get_current_stats(),
                "hourly_metrics": hourly_data,
                "daily_metrics": daily_data,
                "agent_details": agent_details,
                "recent_alerts": recent_alerts,
                "performance_trends": await self._calculate_performance_trends(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Detailed performance metrics retrieval failed: {e}")
            return {"error": str(e)}
    
    async def collect_agent_metrics(self):
        """Collect metrics from all agents"""
        
        try:
            # This would integrate with the agent coordinator
            # For now, generate sample metrics
            agent_names = ["sentiment", "llama_explanation", "gpt4_strategy", "tft_forecasting"]
            
            for agent_name in agent_names:
                # Simulate agent metrics collection
                await self.record_agent_performance(
                    agent_name=agent_name,
                    response_time_ms=self._get_agent_baseline_response_time(agent_name) * (0.8 + 0.4 * hash(agent_name + str(time.time())) % 100 / 100),
                    success=hash(agent_name + str(time.time())) % 10 > 1  # 90% success rate
                )
            
        except Exception as e:
            logger.error(f"Agent metrics collection failed: {e}")
    
    async def collect_system_metrics(self):
        """Collect system performance metrics"""
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            system_metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                process_count=len(psutil.pids())
            )
            
            # Store in history
            self.system_metrics_history.append(system_metrics)
            
            # Store in Redis
            await self._store_system_metrics(system_metrics)
            
            # Check for system alerts
            await self._check_system_performance_alerts(system_metrics)
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    def _calculate_agent_health_score(
        self,
        response_time_ms: float,
        success_rate: float,
        availability: float
    ) -> float:
        """Calculate agent health score"""
        
        try:
            # Response time score (0-1, lower is better)
            response_time_score = max(0, 1 - (response_time_ms / 10000))  # 10 second baseline
            
            # Weighted health score
            health_score = (
                success_rate * 0.4 +
                response_time_score * 0.3 +
                availability * 0.3
            )
            
            return max(0, min(1, health_score))
            
        except Exception as e:
            logger.error(f"Health score calculation failed: {e}")
            return 0.5
    
    def _get_agent_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all agents"""
        
        summary = {}
        
        for agent_name, metrics_list in self.agent_metrics_history.items():
            if not metrics_list:
                continue
                
            recent_metrics = [
                m for m in metrics_list
                if (datetime.utcnow() - m.timestamp).total_seconds() < 300  # Last 5 minutes
            ]
            
            if recent_metrics:
                latest = recent_metrics[-1]
                summary[agent_name] = {
                    "response_time_ms": round(latest.response_time_ms, 1),
                    "success_rate": round(latest.success_rate, 3),
                    "health_score": round(latest.health_score, 3),
                    "availability": round(latest.availability, 3),
                    "requests_per_minute": round(latest.requests_per_minute, 1)
                }
        
        return summary
    
    def _get_agent_baseline_response_time(self, agent_name: str) -> float:
        """Get baseline response time for agent"""
        
        baseline_times = {
            "sentiment": 800.0,
            "llama_explanation": 2500.0,
            "gpt4_strategy": 3200.0,
            "tft_forecasting": 1800.0
        }
        
        return baseline_times.get(agent_name, 1000.0)
    
    async def _store_request_metrics(self, metrics: RequestMetrics):
        """Store request metrics in Redis"""
        
        try:
            if self.redis:
                # Store in time series
                metrics_dict = asdict(metrics)
                metrics_dict["timestamp"] = metrics.timestamp.isoformat()
                
                await self.redis.zadd(
                    "metrics:requests",
                    {json.dumps(metrics_dict): metrics.timestamp.timestamp()}
                )
                
                # Cleanup old entries (keep last 24 hours)
                cutoff = (datetime.utcnow() - timedelta(hours=24)).timestamp()
                await self.redis.zremrangebyscore("metrics:requests", 0, cutoff)
                
        except Exception as e:
            logger.error(f"Request metrics storage failed: {e}")
    
    async def _store_agent_metrics(self, metrics: AgentMetrics):
        """Store agent metrics in Redis"""
        
        try:
            if self.redis:
                metrics_dict = asdict(metrics)
                metrics_dict["timestamp"] = metrics.timestamp.isoformat()
                
                await self.redis.zadd(
                    f"metrics:agents:{metrics.agent_name}",
                    {json.dumps(metrics_dict): metrics.timestamp.timestamp()}
                )
                
                # Cleanup old entries
                cutoff = (datetime.utcnow() - timedelta(hours=24)).timestamp()
                await self.redis.zremrangebyscore(
                    f"metrics:agents:{metrics.agent_name}",
                    0,
                    cutoff
                )
                
        except Exception as e:
            logger.error(f"Agent metrics storage failed: {e}")
    
    async def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in Redis"""
        
        try:
            if self.redis:
                metrics_dict = asdict(metrics)
                metrics_dict["timestamp"] = metrics.timestamp.isoformat()
                
                await self.redis.zadd(
                    "metrics:system",
                    {json.dumps(metrics_dict): metrics.timestamp.timestamp()}
                )
                
                # Cleanup old entries
                cutoff = (datetime.utcnow() - timedelta(hours=24)).timestamp()
                await self.redis.zremrangebyscore("metrics:system", 0, cutoff)
                
        except Exception as e:
            logger.error(f"System metrics storage failed: {e}")
    
    async def _check_request_performance_alerts(self, metrics: RequestMetrics):
        """Check for performance alerts based on request metrics"""
        
        try:
            # Check response time
            if metrics.processing_time_ms > self.thresholds["response_time_critical"]:
                await self._create_alert(
                    severity="critical",
                    category="performance",
                    message=f"Critical response time: {metrics.processing_time_ms}ms",
                    details={"request_id": metrics.request_id, "symbols": metrics.symbols}
                )
            elif metrics.processing_time_ms > self.thresholds["response_time_warning"]:
                await self._create_alert(
                    severity="medium",
                    category="performance",
                    message=f"High response time: {metrics.processing_time_ms}ms",
                    details={"request_id": metrics.request_id, "symbols": metrics.symbols}
                )
            
            # Check confidence score
            if metrics.confidence_score < 0.3:
                await self._create_alert(
                    severity="medium",
                    category="performance",
                    message=f"Low confidence score: {metrics.confidence_score:.2f}",
                    details={"request_id": metrics.request_id, "agents_used": metrics.agents_used}
                )
            
        except Exception as e:
            logger.error(f"Request performance alert check failed: {e}")
    
    async def _check_agent_performance_alerts(self, metrics: AgentMetrics):
        """Check for agent performance alerts"""
        
        try:
            # Check error rate
            if metrics.error_rate > self.thresholds["error_rate_critical"]:
                await self._create_alert(
                    severity="critical",
                    category="agent",
                    message=f"Critical error rate for {metrics.agent_name}: {metrics.error_rate:.1%}",
                    details={"agent_name": metrics.agent_name, "error_rate": metrics.error_rate}
                )
            elif metrics.error_rate > self.thresholds["error_rate_warning"]:
                await self._create_alert(
                    severity="medium",
                    category="agent",
                    message=f"High error rate for {metrics.agent_name}: {metrics.error_rate:.1%}",
                    details={"agent_name": metrics.agent_name, "error_rate": metrics.error_rate}
                )
            
            # Check availability
            if metrics.availability < self.thresholds["agent_availability_critical"]:
                await self._create_alert(
                    severity="critical",
                    category="agent",
                    message=f"Critical availability for {metrics.agent_name}: {metrics.availability:.1%}",
                    details={"agent_name": metrics.agent_name, "availability": metrics.availability}
                )
            elif metrics.availability < self.thresholds["agent_availability_warning"]:
                await self._create_alert(
                    severity="medium",
                    category="agent",
                    message=f"Low availability for {metrics.agent_name}: {metrics.availability:.1%}",
                    details={"agent_name": metrics.agent_name, "availability": metrics.availability}
                )
            
        except Exception as e:
            logger.error(f"Agent performance alert check failed: {e}")
    
    async def _check_system_performance_alerts(self, metrics: SystemMetrics):
        """Check for system performance alerts"""
        
        try:
            # Check CPU usage
            if metrics.cpu_percent > self.thresholds["cpu_critical"]:
                await self._create_alert(
                    severity="critical",
                    category="system",
                    message=f"Critical CPU usage: {metrics.cpu_percent:.1f}%",
                    details={"cpu_percent": metrics.cpu_percent}
                )
            elif metrics.cpu_percent > self.thresholds["cpu_warning"]:
                await self._create_alert(
                    severity="medium",
                    category="system",
                    message=f"High CPU usage: {metrics.cpu_percent:.1f}%",
                    details={"cpu_percent": metrics.cpu_percent}
                )
            
            # Check memory usage
            if metrics.memory_percent > self.thresholds["memory_critical"]:
                await self._create_alert(
                    severity="critical",
                    category="system",
                    message=f"Critical memory usage: {metrics.memory_percent:.1f}%",
                    details={"memory_percent": metrics.memory_percent, "memory_used_mb": metrics.memory_used_mb}
                )
            elif metrics.memory_percent > self.thresholds["memory_warning"]:
                await self._create_alert(
                    severity="medium",
                    category="system",
                    message=f"High memory usage: {metrics.memory_percent:.1f}%",
                    details={"memory_percent": metrics.memory_percent, "memory_used_mb": metrics.memory_used_mb}
                )
            
        except Exception as e:
            logger.error(f"System performance alert check failed: {e}")
    
    async def _create_alert(
        self,
        severity: str,
        category: str,
        message: str,
        details: Dict[str, Any]
    ):
        """Create a performance alert"""
        
        try:
            alert = PerformanceAlert(
                alert_id=f"alert_{int(time.time() * 1000)}",
                timestamp=datetime.utcnow(),
                severity=severity,
                category=category,
                message=message,
                details=details
            )
            
            self.performance_alerts.append(alert)
            self.counters["alerts_generated"] += 1
            
            # Log alert
            log_level = logging.CRITICAL if severity == "critical" else logging.WARNING
            logger.log(log_level, f"Performance alert: {message}")
            
            # Store alert in Redis
            if self.redis:
                alert_dict = asdict(alert)
                alert_dict["timestamp"] = alert.timestamp.isoformat()
                
                await self.redis.zadd(
                    "alerts:performance",
                    {json.dumps(alert_dict): alert.timestamp.timestamp()}
                )
            
            # Trim alerts if too many
            if len(self.performance_alerts) > self.config["max_alerts"]:
                self.performance_alerts = self.performance_alerts[-self.config["max_alerts"]//2:]
            
        except Exception as e:
            logger.error(f"Alert creation failed: {e}")
    
    async def _get_hourly_metrics(self) -> Dict[str, Any]:
        """Get hourly aggregated metrics"""
        
        try:
            # Aggregate request metrics by hour
            now = datetime.utcnow()
            hourly_data = {}
            
            for hour_offset in range(24):  # Last 24 hours
                hour_start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=hour_offset)
                hour_end = hour_start + timedelta(hours=1)
                
                hour_requests = [
                    r for r in self.request_metrics_history
                    if hour_start <= r.timestamp < hour_end
                ]
                
                if hour_requests:
                    hourly_data[hour_start.isoformat()] = {
                        "request_count": len(hour_requests),
                        "avg_response_time": statistics.mean(r.processing_time_ms for r in hour_requests),
                        "avg_confidence": statistics.mean(r.confidence_score for r in hour_requests),
                        "cache_hit_ratio": sum(1 for r in hour_requests if r.cache_hit) / len(hour_requests)
                    }
            
            return hourly_data
            
        except Exception as e:
            logger.error(f"Hourly metrics calculation failed: {e}")
            return {}
    
    async def _get_daily_metrics(self) -> Dict[str, Any]:
        """Get daily aggregated metrics"""
        
        try:
            # Calculate daily trends
            now = datetime.utcnow()
            daily_data = {}
            
            for day_offset in range(7):  # Last 7 days
                day_start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=day_offset)
                day_end = day_start + timedelta(days=1)
                
                day_requests = [
                    r for r in self.request_metrics_history
                    if day_start <= r.timestamp < day_end
                ]
                
                if day_requests:
                    daily_data[day_start.date().isoformat()] = {
                        "total_requests": len(day_requests),
                        "avg_response_time": statistics.mean(r.processing_time_ms for r in day_requests),
                        "success_rate": sum(1 for r in day_requests if r.confidence_score > 0.5) / len(day_requests),
                        "unique_symbols": len(set(symbol for r in day_requests for symbol in r.symbols))
                    }
            
            return daily_data
            
        except Exception as e:
            logger.error(f"Daily metrics calculation failed: {e}")
            return {}
    
    async def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends"""
        
        try:
            # Get recent data for trend calculation
            recent_requests = [
                r for r in self.request_metrics_history
                if (datetime.utcnow() - r.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            older_requests = [
                r for r in self.request_metrics_history
                if 3600 <= (datetime.utcnow() - r.timestamp).total_seconds() < 7200  # Previous hour
            ]
            
            trends = {}
            
            if recent_requests and older_requests:
                # Response time trend
                recent_avg_time = statistics.mean(r.processing_time_ms for r in recent_requests)
                older_avg_time = statistics.mean(r.processing_time_ms for r in older_requests)
                time_trend = (recent_avg_time - older_avg_time) / older_avg_time if older_avg_time > 0 else 0
                
                # Success rate trend
                recent_success_rate = sum(1 for r in recent_requests if r.confidence_score > 0.5) / len(recent_requests)
                older_success_rate = sum(1 for r in older_requests if r.confidence_score > 0.5) / len(older_requests)
                success_trend = recent_success_rate - older_success_rate
                
                trends = {
                    "response_time_trend": round(time_trend, 3),
                    "success_rate_trend": round(success_trend, 3),
                    "request_volume_trend": len(recent_requests) - len(older_requests)
                }
            
            return trends
            
        except Exception as e:
            logger.error(f"Performance trends calculation failed: {e}")
            return {}
    
    # Background tasks
    async def _system_metrics_collector(self):
        """Background task to collect system metrics"""
        
        while True:
            try:
                await asyncio.sleep(self.config["collection_interval"])
                await self.collect_system_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System metrics collector error: {e}")
                await asyncio.sleep(60)
    
    async def _agent_metrics_collector(self):
        """Background task to collect agent metrics"""
        
        while True:
            try:
                await asyncio.sleep(self.config["collection_interval"])
                await self.collect_agent_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Agent metrics collector error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_aggregator(self):
        """Background task to aggregate metrics"""
        
        while True:
            try:
                await asyncio.sleep(self.config["aggregation_interval"])
                
                # Aggregate hourly and daily metrics
                await self._get_hourly_metrics()
                await self._get_daily_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics aggregator error: {e}")
                await asyncio.sleep(300)
    
    async def _alert_monitor(self):
        """Background task to monitor for alerts"""
        
        while True:
            try:
                await asyncio.sleep(self.config["alert_check_interval"])
                
                # Check for any critical conditions that need immediate alerts
                # This is already handled in the individual metric recording functions
                
                # Auto-resolve old alerts
                current_time = datetime.utcnow()
                for alert in self.performance_alerts:
                    if (not alert.resolved and 
                        (current_time - alert.timestamp).total_seconds() > 3600):  # 1 hour
                        alert.resolved = True
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_cleanup(self):
        """Background task to cleanup old metrics"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                
                current_time = datetime.utcnow()
                retention_cutoff = current_time - timedelta(hours=self.config["metrics_retention_hours"])
                
                # Cleanup request metrics
                self.request_metrics_history = [
                    m for m in self.request_metrics_history
                    if m.timestamp > retention_cutoff
                ]
                
                # Cleanup agent metrics
                for agent_name in self.agent_metrics_history:
                    self.agent_metrics_history[agent_name] = [
                        m for m in self.agent_metrics_history[agent_name]
                        if m.timestamp > retention_cutoff
                    ]
                
                # Cleanup system metrics
                self.system_metrics_history = [
                    m for m in self.system_metrics_history
                    if m.timestamp > retention_cutoff
                ]
                
                # Cleanup alerts
                alert_cutoff = current_time - timedelta(days=7)  # Keep alerts for 7 days
                self.performance_alerts = [
                    a for a in self.performance_alerts
                    if a.timestamp > alert_cutoff
                ]
                
                logger.info("Metrics cleanup completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
                await asyncio.sleep(3600)

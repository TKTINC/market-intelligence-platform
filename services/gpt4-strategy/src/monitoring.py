"""
Performance Monitoring and Metrics for GPT-4 Strategy Service
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

class StrategyMetrics:
    def __init__(self):
        # Performance metrics
        self.request_times = deque(maxlen=1000)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.cost_tracking = deque(maxlen=1000)
        
        # Real-time stats
        self.current_stats = {
            "total_requests": 0,
            "total_errors": 0,
            "total_cost": 0.0,
            "avg_response_time": 0.0,
            "last_updated": datetime.utcnow()
        }
        
        # Rate limiting
        self.rate_windows = defaultdict(lambda: deque(maxlen=100))
        
        # Start background metrics collection
        asyncio.create_task(self._background_metrics_update())
    
    async def record_request(
        self,
        processing_time_ms: int,
        strategies_generated: int,
        cost_usd: float,
        user_id: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Record metrics for a strategy generation request"""
        
        try:
            timestamp = time.time()
            
            # Record timing
            self.request_times.append(processing_time_ms)
            
            # Record cost
            self.cost_tracking.append({
                "cost": cost_usd,
                "timestamp": timestamp,
                "strategies": strategies_generated
            })
            
            # Update counters
            self.current_stats["total_requests"] += 1
            self.current_stats["total_cost"] += cost_usd
            
            if error:
                self.current_stats["total_errors"] += 1
                self.error_counts[error] += 1
            
            # Update rate tracking
            if user_id:
                self.rate_windows[user_id].append(timestamp)
            
            # Update average response time
            if self.request_times:
                self.current_stats["avg_response_time"] = statistics.mean(self.request_times)
            
            self.current_stats["last_updated"] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")
    
    async def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        
        try:
            # Calculate recent performance
            now = time.time()
            recent_requests = [
                req for req in self.cost_tracking 
                if now - req["timestamp"] < 3600  # Last hour
            ]
            
            hourly_cost = sum(req["cost"] for req in recent_requests)
            hourly_requests = len(recent_requests)
            
            # Calculate percentiles
            response_times = list(self.request_times)
            percentiles = {}
            if response_times:
                percentiles = {
                    "p50": statistics.median(response_times),
                    "p95": self._calculate_percentile(response_times, 95),
                    "p99": self._calculate_percentile(response_times, 99)
                }
            
            return {
                **self.current_stats,
                "hourly_stats": {
                    "requests": hourly_requests,
                    "cost_usd": hourly_cost,
                    "avg_cost_per_request": hourly_cost / max(hourly_requests, 1)
                },
                "response_time_percentiles": percentiles,
                "error_rate": self.current_stats["total_errors"] / max(self.current_stats["total_requests"], 1)
            }
            
        except Exception as e:
            logger.error(f"Failed to get current stats: {e}")
            return {"error": "Stats unavailable"}
    
    async def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed performance analytics"""
        
        try:
            current_stats = await self.get_current_stats()
            
            # User rate analysis
            user_rates = {}
            now = time.time()
            for user_id, timestamps in self.rate_windows.items():
                recent_requests = [t for t in timestamps if now - t < 3600]
                user_rates[user_id] = len(recent_requests)
            
            # Top users by request volume
            top_users = sorted(
                user_rates.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            # Error analysis
            total_errors = sum(self.error_counts.values())
            error_breakdown = {
                error: {
                    "count": count,
                    "percentage": (count / total_errors * 100) if total_errors > 0 else 0
                }
                for error, count in self.error_counts.items()
            }
            
            # Cost analysis
            costs = [req["cost"] for req in self.cost_tracking]
            cost_stats = {}
            if costs:
                cost_stats = {
                    "total": sum(costs),
                    "average": statistics.mean(costs),
                    "median": statistics.median(costs),
                    "min": min(costs),
                    "max": max(costs)
                }
            
            return {
                "current_stats": current_stats,
                "user_analytics": {
                    "active_users_last_hour": len([u for u in user_rates.values() if u > 0]),
                    "top_users": top_users
                },
                "error_analytics": error_breakdown,
                "cost_analytics": cost_stats,
                "service_health": self._assess_service_health()
            }
            
        except Exception as e:
            logger.error(f"Failed to get detailed stats: {e}")
            return {"error": "Detailed stats unavailable"}
    
    def _calculate_percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
    
    def _assess_service_health(self) -> Dict[str, Any]:
        """Assess overall service health"""
        
        try:
            # Calculate health metrics
            error_rate = self.current_stats["total_errors"] / max(self.current_stats["total_requests"], 1)
            avg_response_time = self.current_stats["avg_response_time"]
            
            # Health score calculation
            health_score = 1.0
            
            # Error rate impact
            if error_rate > 0.1:  # >10% error rate
                health_score -= 0.4
            elif error_rate > 0.05:  # >5% error rate
                health_score -= 0.2
            
            # Response time impact
            if avg_response_time > 5000:  # >5 seconds
                health_score -= 0.3
            elif avg_response_time > 2000:  # >2 seconds
                health_score -= 0.1
            
            # Cost efficiency check
            recent_costs = [req["cost"] for req in self.cost_tracking if time.time() - req["timestamp"] < 3600]
            if recent_costs:
                avg_cost = statistics.mean(recent_costs)
                if avg_cost > 1.0:  # >$1 per request
                    health_score -= 0.2
            
            # Determine health status
            if health_score >= 0.8:
                status = "healthy"
            elif health_score >= 0.6:
                status = "degraded"
            else:
                status = "unhealthy"
            
            return {
                "status": status,
                "score": max(health_score, 0.0),
                "error_rate": error_rate,
                "avg_response_time_ms": avg_response_time,
                "recommendations": self._generate_health_recommendations(error_rate, avg_response_time)
            }
            
        except Exception as e:
            logger.error(f"Health assessment failed: {e}")
            return {"status": "unknown", "error": str(e)}
    
    def _generate_health_recommendations(
        self,
        error_rate: float,
        avg_response_time: float
    ) -> List[str]:
        """Generate recommendations for service improvement"""
        
        recommendations = []
        
        if error_rate > 0.1:
            recommendations.append("High error rate detected - investigate error patterns")
        
        if avg_response_time > 3000:
            recommendations.append("Slow response times - consider scaling or optimization")
        
        if len(self.request_times) > 900:  # Near capacity
            recommendations.append("High request volume - consider additional capacity")
        
        # Cost optimization
        recent_costs = [req["cost"] for req in self.cost_tracking if time.time() - req["timestamp"] < 3600]
        if recent_costs:
            avg_cost = statistics.mean(recent_costs)
            if avg_cost > 0.8:
                recommendations.append("High cost per request - review model usage and optimization")
        
        if not recommendations:
            recommendations.append("Service operating within normal parameters")
        
        return recommendations
    
    async def _background_metrics_update(self):
        """Background task to update metrics periodically"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Clean old data
                cutoff_time = time.time() - 86400  # 24 hours
                
                # Clean cost tracking
                self.cost_tracking = deque(
                    [req for req in self.cost_tracking if req["timestamp"] > cutoff_time],
                    maxlen=1000
                )
                
                # Clean rate windows
                for user_id in list(self.rate_windows.keys()):
                    self.rate_windows[user_id] = deque(
                        [t for t in self.rate_windows[user_id] if t > cutoff_time],
                        maxlen=100
                    )
                    
                    # Remove empty windows
                    if not self.rate_windows[user_id]:
                        del self.rate_windows[user_id]
                
                logger.debug("Metrics cleanup completed")
                
            except Exception as e:
                logger.error(f"Background metrics update failed: {e}")

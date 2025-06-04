"""
Advanced Rate Limiter for GPT-4 Strategy Service
"""

import redis
import asyncio
import json
import time
import logging
import os
from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_time: Optional[datetime]
    tier: str

class AdvancedRateLimiter:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )
        
        # Rate limits by tier (requests per hour)
        self.rate_limits = {
            "free": {"requests_per_hour": 10, "cost_per_hour": 1.0},
            "basic": {"requests_per_hour": 50, "cost_per_hour": 10.0},
            "premium": {"requests_per_hour": 200, "cost_per_hour": 50.0},
            "enterprise": {"requests_per_hour": 1000, "cost_per_hour": 200.0}
        }
        
        # Priority multipliers
        self.priority_multipliers = {
            "normal": 1.0,
            "high": 0.5,    # Uses 2x the quota
            "urgent": 0.25  # Uses 4x the quota
        }
    
    async def health_check(self) -> str:
        """Check Redis connectivity"""
        try:
            await asyncio.to_thread(self.redis_client.ping)
            return "healthy"
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return "unhealthy"
    
    async def check_limit(
        self,
        user_id: str,
        tier: str = "free",
        priority: str = "normal"
    ) -> RateLimitResult:
        """Check if user can make a request within rate limits"""
        
        try:
            current_time = time.time()
            hour_key = f"rate_limit:{user_id}:{int(current_time // 3600)}"
            cost_key = f"cost_limit:{user_id}:{int(current_time // 3600)}"
            
            # Get current usage
            current_requests = await asyncio.to_thread(
                self.redis_client.get, hour_key
            )
            current_cost = await asyncio.to_thread(
                self.redis_client.get, cost_key
            )
            
            current_requests = int(current_requests or 0)
            current_cost = float(current_cost or 0.0)
            
            # Get limits for tier
            limits = self.rate_limits.get(tier, self.rate_limits["free"])
            max_requests = limits["requests_per_hour"]
            max_cost = limits["cost_per_hour"]
            
            # Apply priority multiplier
            multiplier = self.priority_multipliers.get(priority, 1.0)
            request_cost = 1 / multiplier  # Higher priority uses more quota
            
            # Check both request count and cost limits
            if (current_requests + request_cost > max_requests):
                return RateLimitResult(
                    allowed=False,
                    remaining=max(0, int(max_requests - current_requests)),
                    reset_time=datetime.fromtimestamp(
                        (int(current_time // 3600) + 1) * 3600
                    ),
                    tier=tier
                )
            
            # For premium tiers, also check cost limits
            if tier in ["premium", "enterprise"]:
                estimated_cost = 0.5 * (1 / multiplier)  # Estimate with priority
                if current_cost + estimated_cost > max_cost:
                    return RateLimitResult(
                        allowed=False,
                        remaining=0,
                        reset_time=datetime.fromtimestamp(
                            (int(current_time // 3600) + 1) * 3600
                        ),
                        tier=tier
                    )
            
            # Update counters
            pipe = self.redis_client.pipeline()
            pipe.incrbyfloat(hour_key, request_cost)
            pipe.expire(hour_key, 3600)
            await asyncio.to_thread(pipe.execute)
            
            return RateLimitResult(
                allowed=True,
                remaining=int(max_requests - current_requests - request_cost),
                reset_time=None,
                tier=tier
            )
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open for service availability
            return RateLimitResult(
                allowed=True,
                remaining=999,
                reset_time=None,
                tier=tier
            )
    
    async def update_cost_usage(
        self,
        user_id: str,
        actual_cost: float
    ):
        """Update actual cost usage after request completion"""
        try:
            current_time = time.time()
            cost_key = f"cost_limit:{user_id}:{int(current_time // 3600)}"
            
            pipe = self.redis_client.pipeline()
            pipe.incrbyfloat(cost_key, actual_cost)
            pipe.expire(cost_key, 3600)
            await asyncio.to_thread(pipe.execute)
            
        except Exception as e:
            logger.error(f"Cost update failed: {e}")
    
    async def get_user_status(self, user_id: str) -> dict:
        """Get detailed rate limit status for user"""
        try:
            current_time = time.time()
            hour_key = f"rate_limit:{user_id}:{int(current_time // 3600)}"
            cost_key = f"cost_limit:{user_id}:{int(current_time // 3600)}"
            
            current_requests = await asyncio.to_thread(
                self.redis_client.get, hour_key
            )
            current_cost = await asyncio.to_thread(
                self.redis_client.get, cost_key
            )
            
            return {
                "current_hour_requests": int(current_requests or 0),
                "current_hour_cost": float(current_cost or 0.0),
                "reset_time": datetime.fromtimestamp(
                    (int(current_time // 3600) + 1) * 3600
                ).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get user status: {e}")
            return {"error": "Status unavailable"}

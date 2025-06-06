"""
Rate Limiter for API request throttling
"""

import asyncio
import aioredis
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class RateLimit:
    identifier: str
    limit: int
    window_seconds: int
    burst_limit: int
    current_count: int
    window_start: float
    last_request: float

@dataclass
class RateLimitRule:
    name: str
    path_pattern: str
    method: str
    limit: int
    window_seconds: int
    burst_limit: int
    applies_to: str  # 'user', 'ip', 'api_key'
    tier_overrides: Dict[str, Dict[str, int]]

class RateLimiter:
    def __init__(self):
        self.redis = None
        
        # Rate limiting configuration
        self.config = {
            "default_limits": {
                "free": {"requests_per_minute": 60, "burst": 10},
                "basic": {"requests_per_minute": 600, "burst": 50},
                "premium": {"requests_per_minute": 3000, "burst": 200},
                "enterprise": {"requests_per_minute": 15000, "burst": 1000}
            },
            "ip_limits": {
                "requests_per_minute": 1000,
                "burst": 100
            },
            "global_limits": {
                "requests_per_second": 10000,
                "requests_per_minute": 500000
            },
            "cleanup_interval": 300,  # Clean up expired entries every 5 minutes
            "window_precision": 60    # 1-minute sliding windows
        }
        
        # Rate limiting rules for different endpoints
        self.rules = [
            RateLimitRule(
                name="agent_analysis",
                path_pattern="/agents/analyze",
                method="POST",
                limit=10,
                window_seconds=60,
                burst_limit=3,
                applies_to="user",
                tier_overrides={
                    "premium": {"limit": 50, "burst_limit": 10},
                    "enterprise": {"limit": 200, "burst_limit": 50}
                }
            ),
            RateLimitRule(
                name="trading_execution",
                path_pattern="/trading/execute",
                method="POST",
                limit=30,
                window_seconds=60,
                burst_limit=10,
                applies_to="user",
                tier_overrides={
                    "premium": {"limit": 100, "burst_limit": 30},
                    "enterprise": {"limit": 500, "burst_limit": 100}
                }
            ),
            RateLimitRule(
                name="portfolio_creation",
                path_pattern="/portfolios/create",
                method="POST",
                limit=5,
                window_seconds=3600,  # 1 hour
                burst_limit=2,
                applies_to="user",
                tier_overrides={
                    "premium": {"limit": 20, "burst_limit": 5},
                    "enterprise": {"limit": 100, "burst_limit": 20}
                }
            ),
            RateLimitRule(
                name="market_data",
                path_pattern="/market/*",
                method="GET",
                limit=300,
                window_seconds=60,
                burst_limit=50,
                applies_to="user",
                tier_overrides={
                    "premium": {"limit": 1000, "burst_limit": 200},
                    "enterprise": {"limit": 5000, "burst_limit": 1000}
                }
            ),
            RateLimitRule(
                name="general_api",
                path_pattern="/*",
                method="*",
                limit=100,
                window_seconds=60,
                burst_limit=20,
                applies_to="user",
                tier_overrides={
                    "premium": {"limit": 500, "burst_limit": 100},
                    "enterprise": {"limit": 2000, "burst_limit": 400}
                }
            )
        ]
        
        # In-memory cache for frequently accessed limits
        self.limit_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "rate_limited_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
    async def initialize(self):
        """Initialize the rate limiter"""
        try:
            # Initialize Redis connection
            self.redis = aioredis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            
            # Load existing rate limit data
            await self._load_rate_limits()
            
            # Start background tasks
            asyncio.create_task(self._cleanup_task())
            asyncio.create_task(self._cache_refresh_task())
            
            logger.info("Rate limiter initialized")
            
        except Exception as e:
            logger.error(f"Rate limiter initialization failed: {e}")
            raise
    
    async def close(self):
        """Close the rate limiter"""
        if self.redis:
            await self.redis.close()
    
    async def check_rate_limit(
        self,
        identifier: str,
        tier: str = "free",
        path: str = "/",
        method: str = "GET",
        ip_address: str = None
    ) -> bool:
        """Check if request is within rate limits"""
        
        try:
            self.stats["total_requests"] += 1
            current_time = time.time()
            
            # Check global rate limits first
            if not await self._check_global_limits():
                self.stats["rate_limited_requests"] += 1
                return False
            
            # Check IP-based limits if provided
            if ip_address:
                if not await self._check_ip_limits(ip_address, current_time):
                    self.stats["rate_limited_requests"] += 1
                    return False
            
            # Find applicable rate limiting rule
            rule = self._find_applicable_rule(path, method)
            if not rule:
                return True  # No specific rule, allow request
            
            # Get limits for user tier
            limits = self._get_limits_for_tier(rule, tier)
            
            # Check rate limit
            allowed = await self._check_user_rate_limit(
                identifier, rule.name, limits, current_time
            )
            
            if not allowed:
                self.stats["rate_limited_requests"] += 1
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow on error to avoid blocking legitimate requests
    
    async def get_rate_limit_status(
        self,
        identifier: str,
        tier: str = "free",
        path: str = "/",
        method: str = "GET"
    ) -> Dict[str, Any]:
        """Get current rate limit status for identifier"""
        
        try:
            current_time = time.time()
            
            # Find applicable rule
            rule = self._find_applicable_rule(path, method)
            if not rule:
                return {"error": "No rate limit rule found"}
            
            # Get limits for tier
            limits = self._get_limits_for_tier(rule, tier)
            
            # Get current usage
            rate_limit = await self._get_rate_limit(identifier, rule.name, current_time)
            
            # Calculate remaining requests
            remaining = max(0, limits["limit"] - rate_limit.current_count)
            
            # Calculate reset time
            reset_time = rate_limit.window_start + rule.window_seconds
            
            return {
                "identifier": identifier,
                "rule": rule.name,
                "tier": tier,
                "limit": limits["limit"],
                "remaining": remaining,
                "used": rate_limit.current_count,
                "reset_time": reset_time,
                "reset_time_iso": datetime.fromtimestamp(reset_time).isoformat(),
                "burst_limit": limits["burst_limit"],
                "window_seconds": rule.window_seconds
            }
            
        except Exception as e:
            logger.error(f"Rate limit status retrieval failed: {e}")
            return {"error": str(e)}
    
    async def get_user_limits(self, user_id: str, tier: str = "free") -> Dict[str, Any]:
        """Get all rate limits for a user"""
        
        try:
            user_limits = {}
            
            for rule in self.rules:
                if rule.applies_to == "user":
                    limits = self._get_limits_for_tier(rule, tier)
                    status = await self.get_rate_limit_status(
                        user_id, tier, rule.path_pattern, rule.method
                    )
                    
                    user_limits[rule.name] = {
                        "path": rule.path_pattern,
                        "method": rule.method,
                        "limits": limits,
                        "status": status
                    }
            
            return {
                "user_id": user_id,
                "tier": tier,
                "limits": user_limits,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"User limits retrieval failed: {e}")
            return {"error": str(e)}
    
    async def reset_user_limits(self, user_id: str) -> bool:
        """Reset all rate limits for a user (admin function)"""
        
        try:
            # Find all rate limit keys for user
            pattern = f"rate_limit:{user_id}:*"
            keys = await self.redis.keys(pattern)
            
            if keys:
                await self.redis.delete(*keys)
            
            # Clear from cache
            cache_keys_to_remove = [
                key for key in self.limit_cache.keys()
                if key.startswith(f"{user_id}:")
            ]
            
            for key in cache_keys_to_remove:
                del self.limit_cache[key]
            
            logger.info(f"Reset rate limits for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"User limit reset failed: {e}")
            return False
    
    async def add_rate_limit_exception(
        self,
        identifier: str,
        rule_name: str,
        additional_requests: int,
        duration_minutes: int = 60
    ) -> bool:
        """Add temporary rate limit exception"""
        
        try:
            exception_key = f"rate_limit_exception:{identifier}:{rule_name}"
            exception_data = {
                "additional_requests": additional_requests,
                "expires_at": (datetime.utcnow() + timedelta(minutes=duration_minutes)).isoformat()
            }
            
            await self.redis.setex(
                exception_key,
                duration_minutes * 60,
                json.dumps(exception_data)
            )
            
            logger.info(f"Added rate limit exception for {identifier}:{rule_name}")
            return True
            
        except Exception as e:
            logger.error(f"Rate limit exception addition failed: {e}")
            return False
    
    async def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        
        try:
            # Calculate rates
            total_requests = self.stats["total_requests"]
            rate_limited = self.stats["rate_limited_requests"]
            
            if total_requests > 0:
                rate_limited_percentage = (rate_limited / total_requests) * 100
            else:
                rate_limited_percentage = 0.0
            
            # Get cache statistics
            cache_hits = self.stats["cache_hits"]
            cache_misses = self.stats["cache_misses"]
            total_cache_requests = cache_hits + cache_misses
            
            if total_cache_requests > 0:
                cache_hit_ratio = (cache_hits / total_cache_requests) * 100
            else:
                cache_hit_ratio = 0.0
            
            return {
                "total_requests": total_requests,
                "rate_limited_requests": rate_limited,
                "rate_limited_percentage": round(rate_limited_percentage, 2),
                "cache_hit_ratio": round(cache_hit_ratio, 2),
                "active_limits": len(self.limit_cache),
                "rules_count": len(self.rules),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Rate limit stats retrieval failed: {e}")
            return {"error": str(e)}
    
    def _find_applicable_rule(self, path: str, method: str) -> Optional[RateLimitRule]:
        """Find the most specific rate limiting rule for path and method"""
        
        try:
            # Sort rules by specificity (more specific patterns first)
            sorted_rules = sorted(self.rules, key=lambda r: (
                0 if r.path_pattern == path else 1,  # Exact matches first
                len(r.path_pattern.replace("*", "")),  # Longer patterns first
                0 if r.method == method else 1  # Exact method matches first
            ))
            
            for rule in sorted_rules:
                if self._path_matches_pattern(path, rule.path_pattern):
                    if rule.method == "*" or rule.method == method:
                        return rule
            
            return None
            
        except Exception as e:
            logger.error(f"Rule finding failed: {e}")
            return None
    
    def _path_matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches pattern with wildcard support"""
        
        try:
            if pattern == "/*":
                return True
            
            if "*" not in pattern:
                return path == pattern
            
            # Simple wildcard matching
            pattern_parts = pattern.split("*")
            
            if len(pattern_parts) == 2:
                prefix, suffix = pattern_parts
                return path.startswith(prefix) and path.endswith(suffix)
            
            # More complex patterns could be implemented here
            return False
            
        except Exception as e:
            logger.error(f"Path pattern matching failed: {e}")
            return False
    
    def _get_limits_for_tier(self, rule: RateLimitRule, tier: str) -> Dict[str, int]:
        """Get rate limits for specific user tier"""
        
        try:
            # Start with base rule limits
            limits = {
                "limit": rule.limit,
                "burst_limit": rule.burst_limit,
                "window_seconds": rule.window_seconds
            }
            
            # Apply tier overrides if available
            if tier in rule.tier_overrides:
                overrides = rule.tier_overrides[tier]
                limits.update(overrides)
            
            return limits
            
        except Exception as e:
            logger.error(f"Tier limits calculation failed: {e}")
            return {"limit": rule.limit, "burst_limit": rule.burst_limit, "window_seconds": rule.window_seconds}
    
    async def _check_global_limits(self) -> bool:
        """Check global system rate limits"""
        
        try:
            current_time = time.time()
            
            # Check requests per second
            rps_key = "global_rate_limit:rps"
            rps_window = int(current_time)
            
            current_rps = await self.redis.get(f"{rps_key}:{rps_window}")
            if current_rps and int(current_rps) >= self.config["global_limits"]["requests_per_second"]:
                return False
            
            # Check requests per minute
            rpm_key = "global_rate_limit:rpm"
            rpm_window = int(current_time / 60) * 60
            
            current_rpm = await self.redis.get(f"{rpm_key}:{rpm_window}")
            if current_rpm and int(current_rpm) >= self.config["global_limits"]["requests_per_minute"]:
                return False
            
            # Increment counters
            await self.redis.incr(f"{rps_key}:{rps_window}")
            await self.redis.expire(f"{rps_key}:{rps_window}", 2)
            
            await self.redis.incr(f"{rpm_key}:{rpm_window}")
            await self.redis.expire(f"{rpm_key}:{rpm_window}", 120)
            
            return True
            
        except Exception as e:
            logger.error(f"Global limits check failed: {e}")
            return True
    
    async def _check_ip_limits(self, ip_address: str, current_time: float) -> bool:
        """Check IP-based rate limits"""
        
        try:
            window = int(current_time / 60) * 60  # 1-minute windows
            ip_key = f"ip_rate_limit:{ip_address}:{window}"
            
            current_count = await self.redis.get(ip_key)
            current_count = int(current_count) if current_count else 0
            
            if current_count >= self.config["ip_limits"]["requests_per_minute"]:
                return False
            
            # Increment counter
            await self.redis.incr(ip_key)
            await self.redis.expire(ip_key, 120)  # 2 minutes TTL
            
            return True
            
        except Exception as e:
            logger.error(f"IP limits check failed: {e}")
            return True
    
    async def _check_user_rate_limit(
        self,
        identifier: str,
        rule_name: str,
        limits: Dict[str, int],
        current_time: float
    ) -> bool:
        """Check user-specific rate limits"""
        
        try:
            # Get current rate limit state
            rate_limit = await self._get_rate_limit(identifier, rule_name, current_time)
            
            # Check if we're in a new window
            window_duration = limits["window_seconds"]
            if current_time >= rate_limit.window_start + window_duration:
                # Reset for new window
                rate_limit.window_start = int(current_time / window_duration) * window_duration
                rate_limit.current_count = 0
            
            # Check burst limit (short-term)
            time_since_last = current_time - rate_limit.last_request
            if time_since_last < 1.0:  # Within 1 second
                burst_window_start = current_time - 60  # 1-minute burst window
                burst_count = await self._get_burst_count(identifier, rule_name, burst_window_start)
                
                if burst_count >= limits["burst_limit"]:
                    return False
            
            # Check main rate limit
            if rate_limit.current_count >= limits["limit"]:
                # Check for exceptions
                exception = await self._get_rate_limit_exception(identifier, rule_name)
                if exception:
                    additional_requests = exception.get("additional_requests", 0)
                    if rate_limit.current_count >= limits["limit"] + additional_requests:
                        return False
                else:
                    return False
            
            # Update rate limit state
            rate_limit.current_count += 1
            rate_limit.last_request = current_time
            
            await self._store_rate_limit(identifier, rule_name, rate_limit)
            await self._update_burst_count(identifier, rule_name, current_time)
            
            return True
            
        except Exception as e:
            logger.error(f"User rate limit check failed: {e}")
            return True
    
    async def _get_rate_limit(
        self,
        identifier: str,
        rule_name: str,
        current_time: float
    ) -> RateLimit:
        """Get current rate limit state"""
        
        try:
            cache_key = f"{identifier}:{rule_name}"
            
            # Try cache first
            if cache_key in self.limit_cache:
                cached_data, cache_time = self.limit_cache[cache_key]
                if current_time - cache_time < self.cache_ttl:
                    self.stats["cache_hits"] += 1
                    return RateLimit(**cached_data)
            
            self.stats["cache_misses"] += 1
            
            # Get from Redis
            redis_key = f"rate_limit:{identifier}:{rule_name}"
            rate_limit_data = await self.redis.get(redis_key)
            
            if rate_limit_data:
                data = json.loads(rate_limit_data)
                rate_limit = RateLimit(**data)
            else:
                # Create new rate limit
                window_duration = 60  # Default 1-minute window
                window_start = int(current_time / window_duration) * window_duration
                
                rate_limit = RateLimit(
                    identifier=identifier,
                    limit=0,  # Will be set by caller
                    window_seconds=window_duration,
                    burst_limit=0,  # Will be set by caller
                    current_count=0,
                    window_start=window_start,
                    last_request=0.0
                )
            
            # Update cache
            self.limit_cache[cache_key] = (asdict(rate_limit), current_time)
            
            return rate_limit
            
        except Exception as e:
            logger.error(f"Rate limit retrieval failed: {e}")
            # Return default rate limit
            return RateLimit(
                identifier=identifier,
                limit=100,
                window_seconds=60,
                burst_limit=10,
                current_count=0,
                window_start=current_time,
                last_request=0.0
            )
    
    async def _store_rate_limit(
        self,
        identifier: str,
        rule_name: str,
        rate_limit: RateLimit
    ):
        """Store rate limit state"""
        
        try:
            redis_key = f"rate_limit:{identifier}:{rule_name}"
            
            # Store in Redis with TTL
            ttl = rate_limit.window_seconds + 60  # Window duration + buffer
            await self.redis.setex(
                redis_key,
                ttl,
                json.dumps(asdict(rate_limit))
            )
            
            # Update cache
            cache_key = f"{identifier}:{rule_name}"
            self.limit_cache[cache_key] = (asdict(rate_limit), time.time())
            
        except Exception as e:
            logger.error(f"Rate limit storage failed: {e}")
    
    async def _get_burst_count(
        self,
        identifier: str,
        rule_name: str,
        window_start: float
    ) -> int:
        """Get burst count for recent time window"""
        
        try:
            burst_key = f"burst_count:{identifier}:{rule_name}"
            
            # Use sorted set to track timestamps
            count = await self.redis.zcount(burst_key, window_start, time.time())
            return count
            
        except Exception as e:
            logger.error(f"Burst count retrieval failed: {e}")
            return 0
    
    async def _update_burst_count(
        self,
        identifier: str,
        rule_name: str,
        timestamp: float
    ):
        """Update burst count with new timestamp"""
        
        try:
            burst_key = f"burst_count:{identifier}:{rule_name}"
            
            # Add timestamp to sorted set
            await self.redis.zadd(burst_key, {str(timestamp): timestamp})
            
            # Remove old timestamps (older than 1 minute)
            cutoff = timestamp - 60
            await self.redis.zremrangebyscore(burst_key, 0, cutoff)
            
            # Set TTL
            await self.redis.expire(burst_key, 120)
            
        except Exception as e:
            logger.error(f"Burst count update failed: {e}")
    
    async def _get_rate_limit_exception(
        self,
        identifier: str,
        rule_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get rate limit exception if it exists"""
        
        try:
            exception_key = f"rate_limit_exception:{identifier}:{rule_name}"
            exception_data = await self.redis.get(exception_key)
            
            if exception_data:
                return json.loads(exception_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Rate limit exception retrieval failed: {e}")
            return None
    
    async def _load_rate_limits(self):
        """Load existing rate limits from Redis"""
        
        try:
            # Load rate limits into cache
            pattern = "rate_limit:*"
            keys = await self.redis.keys(pattern)
            
            loaded_count = 0
            current_time = time.time()
            
            for key in keys:
                rate_limit_data = await self.redis.get(key)
                if rate_limit_data:
                    try:
                        data = json.loads(rate_limit_data)
                        
                        # Extract identifier and rule from key
                        key_parts = key.split(":")
                        if len(key_parts) >= 3:
                            identifier = key_parts[1]
                            rule_name = key_parts[2]
                            cache_key = f"{identifier}:{rule_name}"
                            
                            self.limit_cache[cache_key] = (data, current_time)
                            loaded_count += 1
                            
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Loaded {loaded_count} rate limits into cache")
            
        except Exception as e:
            logger.error(f"Rate limits loading failed: {e}")
    
    # Background tasks
    async def _cleanup_task(self):
        """Background task to cleanup expired rate limits"""
        
        while True:
            try:
                await asyncio.sleep(self.config["cleanup_interval"])
                
                current_time = time.time()
                
                # Clean up cache
                expired_keys = []
                for cache_key, (data, cache_time) in self.limit_cache.items():
                    if current_time - cache_time > self.cache_ttl * 2:  # Double TTL for cleanup
                        expired_keys.append(cache_key)
                
                for key in expired_keys:
                    del self.limit_cache[key]
                
                # Clean up old burst count keys
                burst_pattern = "burst_count:*"
                burst_keys = await self.redis.keys(burst_pattern)
                
                for key in burst_keys:
                    # Remove old timestamps
                    cutoff = current_time - 300  # 5 minutes
                    await self.redis.zremrangebyscore(key, 0, cutoff)
                    
                    # Remove empty keys
                    count = await self.redis.zcard(key)
                    if count == 0:
                        await self.redis.delete(key)
                
                if expired_keys or burst_keys:
                    logger.debug(f"Rate limiter cleanup: removed {len(expired_keys)} cache entries, processed {len(burst_keys)} burst keys")
                
            except Exception as e:
                logger.error(f"Rate limiter cleanup task error: {e}")
                await asyncio.sleep(300)
    
    async def _cache_refresh_task(self):
        """Background task to refresh frequently used cache entries"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                current_time = time.time()
                
                # Refresh cache entries that are frequently accessed but close to expiring
                refresh_threshold = self.cache_ttl * 0.8  # Refresh at 80% of TTL
                
                for cache_key, (data, cache_time) in list(self.limit_cache.items()):
                    age = current_time - cache_time
                    
                    if age > refresh_threshold:
                        # Refresh from Redis
                        identifier, rule_name = cache_key.split(":", 1)
                        redis_key = f"rate_limit:{identifier}:{rule_name}"
                        
                        rate_limit_data = await self.redis.get(redis_key)
                        if rate_limit_data:
                            data = json.loads(rate_limit_data)
                            self.limit_cache[cache_key] = (data, current_time)
                        else:
                            # Remove from cache if not in Redis
                            del self.limit_cache[cache_key]
                
            except Exception as e:
                logger.error(f"Cache refresh task error: {e}")
                await asyncio.sleep(60)

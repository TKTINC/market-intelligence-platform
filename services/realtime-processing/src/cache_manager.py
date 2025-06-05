"""
Distributed Cache Manager for intelligent caching with Redis
"""

import asyncio
import aioredis
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time
import pickle
import zlib
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    key: str
    value: Any
    ttl: int
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    compressed: bool = False

@dataclass
class CacheStats:
    hits: int
    misses: int
    hit_ratio: float
    total_keys: int
    memory_usage_mb: float
    evictions: int
    expired_keys: int

class DistributedCacheManager:
    def __init__(self):
        self.redis = None
        self.local_cache = {}  # L1 cache
        self.cache_stats = CacheStats(0, 0, 0.0, 0, 0.0, 0, 0)
        
        # Cache configuration
        self.config = {
            "redis_url": "redis://localhost:6379",
            "redis_db": 1,  # Separate DB for cache
            "max_local_cache_size": 1000,
            "compression_threshold": 1024,  # Compress values larger than 1KB
            "default_ttl": 300,  # 5 minutes
            "max_ttl": 86400,  # 24 hours
            "cleanup_interval": 300,  # 5 minutes
            "stats_interval": 60  # 1 minute
        }
        
        # Cache key prefixes
        self.prefixes = {
            "unified_intelligence": "ui:",
            "agent_response": "ar:",
            "market_data": "md:",
            "news": "news:",
            "options": "opt:",
            "sentiment": "sent:",
            "forecasts": "fc:",
            "strategies": "strat:",
            "explanations": "exp:"
        }
        
        # Cache policies
        self.cache_policies = {
            "unified_intelligence": {"ttl": 300, "compress": True},
            "agent_response": {"ttl": 600, "compress": True},
            "market_data": {"ttl": 60, "compress": False},
            "news": {"ttl": 3600, "compress": True},
            "sentiment": {"ttl": 900, "compress": False},
            "forecasts": {"ttl": 1800, "compress": True},
            "strategies": {"ttl": 1200, "compress": True}
        }
        
        # Background task handles
        self.background_tasks = []
        
    async def initialize(self):
        """Initialize the cache manager"""
        try:
            # Connect to Redis
            self.redis = aioredis.from_url(
                self.config["redis_url"],
                db=self.config["redis_db"],
                encoding="utf-8",
                decode_responses=False  # Handle binary data for compression
            )
            
            # Test connection
            await self.redis.ping()
            
            # Start background tasks
            self.background_tasks = [
                asyncio.create_task(self._cache_cleanup_task()),
                asyncio.create_task(self._stats_collection_task()),
                asyncio.create_task(self._memory_management_task())
            ]
            
            logger.info("Cache manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Cache manager initialization failed: {e}")
            raise
    
    async def close(self):
        """Close the cache manager"""
        try:
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Close Redis connection
            if self.redis:
                await self.redis.close()
            
            logger.info("Cache manager closed")
            
        except Exception as e:
            logger.error(f"Cache manager close error: {e}")
    
    async def health_check(self) -> str:
        """Check cache health"""
        try:
            if not self.redis:
                return "unhealthy - no redis connection"
            
            # Test Redis connection
            await self.redis.ping()
            
            # Check memory usage
            info = await self.redis.info("memory")
            memory_usage_mb = info.get("used_memory", 0) / (1024 * 1024)
            
            if memory_usage_mb > 1000:  # 1GB threshold
                return "warning - high memory usage"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return "unhealthy"
    
    async def generate_cache_key(
        self, 
        request: Any, 
        routing_plan: Any = None
    ) -> str:
        """Generate intelligent cache key for request"""
        
        try:
            # Extract relevant request attributes
            key_components = []
            
            # Add symbols (sorted for consistency)
            symbols = getattr(request, 'symbols', [])
            if symbols:
                key_components.append(f"symbols:{','.join(sorted(symbols))}")
            
            # Add analysis type
            analysis_type = getattr(request, 'analysis_type', 'standard')
            key_components.append(f"type:{analysis_type}")
            
            # Add time horizon
            time_horizon = getattr(request, 'time_horizon', 'intraday')
            key_components.append(f"horizon:{time_horizon}")
            
            # Add requested agents (if specified)
            agents_requested = getattr(request, 'agents_requested', None)
            if agents_requested:
                key_components.append(f"agents:{','.join(sorted(agents_requested))}")
            elif routing_plan:
                agents_to_call = getattr(routing_plan, 'agents_to_call', [])
                key_components.append(f"agents:{','.join(sorted(agents_to_call))}")
            
            # Add feature flags
            flags = []
            if getattr(request, 'include_explanations', True):
                flags.append("exp")
            if getattr(request, 'include_strategies', True):
                flags.append("strat")
            if getattr(request, 'include_forecasts', True):
                flags.append("fc")
            if flags:
                key_components.append(f"flags:{','.join(flags)}")
            
            # Add real-time data flag
            if getattr(request, 'real_time_data', True):
                key_components.append("rt:1")
            else:
                key_components.append("rt:0")
            
            # Create base key
            base_key = "|".join(key_components)
            
            # Generate hash for consistent key length
            key_hash = hashlib.md5(base_key.encode()).hexdigest()
            
            # Add prefix
            cache_key = f"{self.prefixes['unified_intelligence']}{key_hash}"
            
            return cache_key
            
        except Exception as e:
            logger.error(f"Cache key generation failed: {e}")
            # Fallback to simple hash
            return f"{self.prefixes['unified_intelligence']}{hash(str(request))}"
    
    async def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result with intelligent L1/L2 strategy"""
        
        try:
            start_time = time.time()
            
            # Check L1 cache first (local)
            if cache_key in self.local_cache:
                entry = self.local_cache[cache_key]
                
                # Check if expired
                if datetime.utcnow() - entry.created_at < timedelta(seconds=entry.ttl):
                    entry.last_accessed = datetime.utcnow()
                    entry.access_count += 1
                    
                    self.cache_stats.hits += 1
                    self._update_hit_ratio()
                    
                    logger.debug(f"L1 cache hit for key: {cache_key}")
                    return entry.value
                else:
                    # Remove expired entry
                    del self.local_cache[cache_key]
            
            # Check L2 cache (Redis)
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                # Deserialize data
                try:
                    # Check if compressed
                    if cached_data.startswith(b'compressed:'):
                        compressed_data = cached_data[11:]  # Remove 'compressed:' prefix
                        decompressed_data = zlib.decompress(compressed_data)
                        result = pickle.loads(decompressed_data)
                    else:
                        result = pickle.loads(cached_data)
                    
                    # Store in L1 cache for faster access
                    await self._store_in_local_cache(cache_key, result, 300)  # 5 min L1 TTL
                    
                    self.cache_stats.hits += 1
                    self._update_hit_ratio()
                    
                    logger.debug(f"L2 cache hit for key: {cache_key}, retrieval time: {(time.time() - start_time)*1000:.1f}ms")
                    return result
                    
                except Exception as e:
                    logger.error(f"Cache deserialization failed: {e}")
                    # Remove corrupted cache entry
                    await self.redis.delete(cache_key)
            
            # Cache miss
            self.cache_stats.misses += 1
            self._update_hit_ratio()
            
            logger.debug(f"Cache miss for key: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            self.cache_stats.misses += 1
            self._update_hit_ratio()
            return None
    
    async def cache_result(
        self, 
        cache_key: str, 
        result: Dict[str, Any], 
        ttl: Optional[int] = None,
        cache_type: str = "unified_intelligence"
    ):
        """Cache result with intelligent compression and TTL"""
        
        try:
            # Get cache policy
            policy = self.cache_policies.get(cache_type, {"ttl": self.config["default_ttl"], "compress": True})
            final_ttl = ttl if ttl is not None else policy["ttl"]
            final_ttl = min(final_ttl, self.config["max_ttl"])
            
            # Serialize data
            serialized_data = pickle.dumps(result)
            data_size = len(serialized_data)
            
            # Compress if needed
            compressed = False
            if policy.get("compress", False) and data_size > self.config["compression_threshold"]:
                compressed_data = zlib.compress(serialized_data)
                if len(compressed_data) < data_size * 0.8:  # Only use if 20%+ compression
                    serialized_data = b'compressed:' + compressed_data
                    compressed = True
                    logger.debug(f"Compressed cache entry from {data_size} to {len(compressed_data)} bytes")
            
            # Store in Redis (L2)
            await self.redis.setex(cache_key, final_ttl, serialized_data)
            
            # Store in local cache (L1) for frequently accessed items
            if cache_type in ["unified_intelligence", "market_data", "sentiment"]:
                await self._store_in_local_cache(cache_key, result, min(final_ttl, 300))
            
            logger.debug(f"Cached result with key: {cache_key}, size: {data_size} bytes, compressed: {compressed}, TTL: {final_ttl}s")
            
        except Exception as e:
            logger.error(f"Cache storage failed: {e}")
    
    async def invalidate_cache(self, pattern: str = None, cache_type: str = None):
        """Invalidate cache entries by pattern or type"""
        
        try:
            if cache_type and cache_type in self.prefixes:
                pattern = f"{self.prefixes[cache_type]}*"
            
            if pattern:
                # Get matching keys
                keys = await self.redis.keys(pattern)
                
                if keys:
                    # Delete from Redis
                    await self.redis.delete(*keys)
                    
                    # Delete from local cache
                    local_keys_to_delete = [k for k in self.local_cache.keys() if k.startswith(pattern.replace('*', ''))]
                    for k in local_keys_to_delete:
                        del self.local_cache[k]
                    
                    logger.info(f"Invalidated {len(keys)} cache entries matching pattern: {pattern}")
            
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
    
    async def clear_all_cache(self):
        """Clear all cached data"""
        
        try:
            # Clear Redis
            await self.redis.flushdb()
            
            # Clear local cache
            self.local_cache.clear()
            
            # Reset stats
            self.cache_stats = CacheStats(0, 0, 0.0, 0, 0.0, 0, 0)
            
            logger.info("All cache cleared")
            
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache performance statistics"""
        
        try:
            # Redis info
            redis_info = await self.redis.info("memory")
            redis_memory_mb = redis_info.get("used_memory", 0) / (1024 * 1024)
            
            # Count keys by type
            key_counts = {}
            for cache_type, prefix in self.prefixes.items():
                keys = await self.redis.keys(f"{prefix}*")
                key_counts[cache_type] = len(keys)
            
            # Local cache stats
            local_cache_size = len(self.local_cache)
            local_memory_estimate = sum(
                len(str(entry.value)) for entry in self.local_cache.values()
            ) / (1024 * 1024)  # Rough estimate in MB
            
            return {
                "hit_ratio": self.cache_stats.hit_ratio,
                "total_hits": self.cache_stats.hits,
                "total_misses": self.cache_stats.misses,
                "total_requests": self.cache_stats.hits + self.cache_stats.misses,
                "redis_memory_mb": round(redis_memory_mb, 2),
                "local_cache_size": local_cache_size,
                "local_memory_estimate_mb": round(local_memory_estimate, 2),
                "key_counts_by_type": key_counts,
                "total_keys": sum(key_counts.values()),
                "evictions": self.cache_stats.evictions,
                "expired_keys": self.cache_stats.expired_keys,
                "cache_policies": self.cache_policies
            }
            
        except Exception as e:
            logger.error(f"Cache statistics retrieval failed: {e}")
            return {"error": str(e)}
    
    async def cleanup_expired_entries(self):
        """Cleanup expired cache entries"""
        
        try:
            # Cleanup local cache
            current_time = datetime.utcnow()
            expired_local_keys = []
            
            for key, entry in self.local_cache.items():
                if current_time - entry.created_at > timedelta(seconds=entry.ttl):
                    expired_local_keys.append(key)
            
            for key in expired_local_keys:
                del self.local_cache[key]
                self.cache_stats.expired_keys += 1
            
            # Redis handles its own TTL expiration
            logger.debug(f"Cleaned up {len(expired_local_keys)} expired local cache entries")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    async def _store_in_local_cache(self, key: str, value: Any, ttl: int):
        """Store entry in local L1 cache"""
        
        try:
            # Check cache size limit
            if len(self.local_cache) >= self.config["max_local_cache_size"]:
                # Remove least recently used entry
                lru_key = min(
                    self.local_cache.keys(),
                    key=lambda k: self.local_cache[k].last_accessed
                )
                del self.local_cache[lru_key]
                self.cache_stats.evictions += 1
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                size_bytes=len(str(value)),
                compressed=False
            )
            
            self.local_cache[key] = entry
            
        except Exception as e:
            logger.error(f"Local cache storage failed: {e}")
    
    def _update_hit_ratio(self):
        """Update cache hit ratio"""
        total_requests = self.cache_stats.hits + self.cache_stats.misses
        if total_requests > 0:
            self.cache_stats.hit_ratio = self.cache_stats.hits / total_requests
    
    async def _cache_cleanup_task(self):
        """Background task for cache cleanup"""
        
        while True:
            try:
                await asyncio.sleep(self.config["cleanup_interval"])
                await self.cleanup_expired_entries()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup task error: {e}")
                await asyncio.sleep(60)
    
    async def _stats_collection_task(self):
        """Background task for stats collection"""
        
        while True:
            try:
                await asyncio.sleep(self.config["stats_interval"])
                
                # Update cache statistics
                stats = await self.get_cache_statistics()
                self.cache_stats.total_keys = stats.get("total_keys", 0)
                self.cache_stats.memory_usage_mb = stats.get("redis_memory_mb", 0.0)
                
                # Log stats periodically
                if self.cache_stats.hits + self.cache_stats.misses > 0:
                    logger.info(f"Cache stats - Hit ratio: {self.cache_stats.hit_ratio:.2%}, "
                              f"Memory: {self.cache_stats.memory_usage_mb:.1f}MB, "
                              f"Keys: {self.cache_stats.total_keys}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stats collection task error: {e}")
                await asyncio.sleep(60)
    
    async def _memory_management_task(self):
        """Background task for memory management"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Check Redis memory usage
                redis_info = await self.redis.info("memory")
                memory_usage_mb = redis_info.get("used_memory", 0) / (1024 * 1024)
                
                # If memory usage is high, start aggressive cleanup
                if memory_usage_mb > 800:  # 800MB threshold
                    logger.warning(f"High Redis memory usage: {memory_usage_mb:.1f}MB - starting cleanup")
                    
                    # Delete oldest entries from each cache type
                    for cache_type, prefix in self.prefixes.items():
                        keys = await self.redis.keys(f"{prefix}*")
                        
                        if len(keys) > 1000:  # If too many keys
                            # Delete 10% of oldest keys
                            keys_to_delete = keys[:len(keys)//10]
                            if keys_to_delete:
                                await self.redis.delete(*keys_to_delete)
                                logger.info(f"Deleted {len(keys_to_delete)} old cache entries for {cache_type}")
                
                # Check local cache memory
                if len(self.local_cache) > self.config["max_local_cache_size"] * 0.8:
                    # Remove 20% of LRU entries
                    sorted_entries = sorted(
                        self.local_cache.items(),
                        key=lambda x: x[1].last_accessed
                    )
                    
                    entries_to_remove = len(sorted_entries) // 5
                    for i in range(entries_to_remove):
                        key = sorted_entries[i][0]
                        del self.local_cache[key]
                        self.cache_stats.evictions += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory management task error: {e}")
                await asyncio.sleep(300)

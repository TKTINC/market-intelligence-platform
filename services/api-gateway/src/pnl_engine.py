"""
P&L Engine for real-time profit and loss calculations
"""

import asyncio
import aioredis
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import statistics
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class PnLSnapshot:
    portfolio_id: str
    timestamp: datetime
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    day_pnl: float
    cash_balance: float
    total_value: float
    positions_value: float

@dataclass
class PositionPnL:
    portfolio_id: str
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    day_pnl: float
    pnl_percentage: float

class PnLEngine:
    def __init__(self):
        self.redis = None
        
        # P&L calculation configuration
        self.config = {
            "snapshot_interval": 60,    # Take P&L snapshots every 60 seconds
            "history_retention_days": 90,  # Keep P&L history for 90 days
            "mark_to_market_interval": 5,  # Mark-to-market every 5 seconds
            "day_start_time": "09:30",  # Market day start time
            "day_end_time": "16:00"     # Market day end time
        }
        
        # P&L tracking
        self.portfolio_pnl_cache = {}
        self.position_pnl_cache = {}
        
        # Performance metrics
        self.calculation_stats = {
            "total_calculations": 0,
            "avg_calculation_time_ms": 0.0,
            "cache_hit_ratio": 0.0
        }
        
    async def initialize(self):
        """Initialize the P&L engine"""
        try:
            # Initialize Redis connection
            self.redis = aioredis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            
            # Load existing P&L cache
            await self._load_pnl_cache()
            
            # Start background tasks
            asyncio.create_task(self._pnl_snapshot_task())
            asyncio.create_task(self._mark_to_market_task())
            asyncio.create_task(self._pnl_history_cleanup())
            
            logger.info("P&L engine initialized")
            
        except Exception as e:
            logger.error(f"P&L engine initialization failed: {e}")
            raise
    
    async def close(self):
        """Close the P&L engine"""
        if self.redis:
            await self.redis.close()
    
    async def health_check(self) -> str:
        """Check health of P&L engine"""
        try:
            if not self.redis:
                return "unhealthy - no redis connection"
            
            # Test Redis connection
            await self.redis.ping()
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"P&L engine health check failed: {e}")
            return "unhealthy"
    
    async def update_portfolio_pnl(self, portfolio_id: str):
        """Update P&L for a specific portfolio"""
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Get portfolio data
            portfolio_data = await self._get_portfolio_data(portfolio_id)
            if not portfolio_data:
                return
            
            # Get current positions
            positions = await self._get_portfolio_positions(portfolio_id)
            
            # Calculate portfolio P&L
            portfolio_pnl = await self._calculate_portfolio_pnl(portfolio_data, positions)
            
            # Update cache
            self.portfolio_pnl_cache[portfolio_id] = portfolio_pnl
            
            # Store in Redis
            await self._store_portfolio_pnl(portfolio_pnl)
            
            # Update calculation stats
            calculation_time = (asyncio.get_event_loop().time() - start_time) * 1000
            await self._update_calculation_stats(calculation_time)
            
            logger.debug(f"Portfolio P&L updated for {portfolio_id}")
            
        except Exception as e:
            logger.error(f"Portfolio P&L update failed for {portfolio_id}: {e}")
    
    async def update_position_pnl(self, portfolio_id: str, symbol: str):
        """Update P&L for a specific position"""
        
        try:
            # Get position data
            position_data = await self._get_position_data(portfolio_id, symbol)
            if not position_data:
                return
            
            # Get current market price
            current_price = await self._get_current_price(symbol)
            
            # Calculate position P&L
            position_pnl = await self._calculate_position_pnl(position_data, current_price)
            
            # Update cache
            cache_key = f"{portfolio_id}:{symbol}"
            self.position_pnl_cache[cache_key] = position_pnl
            
            # Store in Redis
            await self._store_position_pnl(position_pnl)
            
        except Exception as e:
            logger.error(f"Position P&L update failed for {portfolio_id}:{symbol}: {e}")
    
    async def get_portfolio_pnl(
        self, 
        portfolio_id: str, 
        timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """Get P&L data for a portfolio"""
        
        try:
            # Get current P&L
            current_pnl = self.portfolio_pnl_cache.get(portfolio_id)
            if not current_pnl:
                await self.update_portfolio_pnl(portfolio_id)
                current_pnl = self.portfolio_pnl_cache.get(portfolio_id)
            
            if not current_pnl:
                raise Exception(f"Portfolio P&L not available for {portfolio_id}")
            
            # Get historical P&L
            historical_pnl = await self._get_historical_pnl(portfolio_id, timeframe)
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(
                portfolio_id, timeframe
            )
            
            return {
                "portfolio_id": portfolio_id,
                "current_pnl": asdict(current_pnl),
                "historical_pnl": historical_pnl,
                "performance_metrics": performance_metrics,
                "timeframe": timeframe,
                "last_updated": current_pnl.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Portfolio P&L retrieval failed: {e}")
            raise
    
    async def get_position_pnl(
        self, 
        portfolio_id: str, 
        symbol: str
    ) -> Dict[str, Any]:
        """Get P&L data for a specific position"""
        
        try:
            cache_key = f"{portfolio_id}:{symbol}"
            
            # Get current position P&L
            current_pnl = self.position_pnl_cache.get(cache_key)
            if not current_pnl:
                await self.update_position_pnl(portfolio_id, symbol)
                current_pnl = self.position_pnl_cache.get(cache_key)
            
            if not current_pnl:
                raise Exception(f"Position P&L not available for {portfolio_id}:{symbol}")
            
            # Get position P&L history
            pnl_history = await self._get_position_pnl_history(portfolio_id, symbol)
            
            return {
                "portfolio_id": portfolio_id,
                "symbol": symbol,
                "current_pnl": asdict(current_pnl),
                "pnl_history": pnl_history,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Position P&L retrieval failed: {e}")
            raise
    
    async def recalculate_all_pnl(self):
        """Recalculate P&L for all portfolios (admin function)"""
        
        try:
            # Get all active portfolios
            portfolio_keys = await self.redis.keys("portfolio:*")
            
            recalculated_count = 0
            
            for key in portfolio_keys:
                try:
                    portfolio_id = key.split(":")[-1]
                    await self.update_portfolio_pnl(portfolio_id)
                    recalculated_count += 1
                    
                    # Add small delay to avoid overwhelming the system
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Failed to recalculate P&L for portfolio {portfolio_id}: {e}")
            
            logger.info(f"Recalculated P&L for {recalculated_count} portfolios")
            
        except Exception as e:
            logger.error(f"P&L recalculation failed: {e}")
            raise
    
    async def _calculate_portfolio_pnl(
        self, 
        portfolio_data: Dict[str, Any], 
        positions: List[Dict[str, Any]]
    ) -> PnLSnapshot:
        """Calculate comprehensive P&L for a portfolio"""
        
        try:
            portfolio_id = portfolio_data["portfolio_id"]
            initial_balance = portfolio_data["initial_balance"]
            cash_balance = portfolio_data["cash_balance"]
            
            total_realized_pnl = 0.0
            total_unrealized_pnl = 0.0
            total_positions_value = 0.0
            
            # Calculate P&L for each position
            for position in positions:
                symbol = position["symbol"]
                quantity = position["quantity"]
                avg_cost = position["avg_cost"]
                
                # Get current market price
                current_price = await self._get_current_price(symbol)
                
                # Calculate position values
                market_value = quantity * current_price
                unrealized_pnl = (current_price - avg_cost) * quantity
                realized_pnl = position.get("realized_pnl", 0.0)
                
                total_positions_value += market_value
                total_unrealized_pnl += unrealized_pnl
                total_realized_pnl += realized_pnl
            
            # Calculate portfolio totals
            total_value = cash_balance + total_positions_value
            total_pnl = total_value - initial_balance
            
            # Calculate day P&L (simplified - would need intraday tracking)
            day_pnl = await self._calculate_day_pnl(portfolio_id)
            
            return PnLSnapshot(
                portfolio_id=portfolio_id,
                timestamp=datetime.utcnow(),
                total_pnl=total_pnl,
                realized_pnl=total_realized_pnl,
                unrealized_pnl=total_unrealized_pnl,
                day_pnl=day_pnl,
                cash_balance=cash_balance,
                total_value=total_value,
                positions_value=total_positions_value
            )
            
        except Exception as e:
            logger.error(f"Portfolio P&L calculation failed: {e}")
            raise
    
    async def _calculate_position_pnl(
        self, 
        position_data: Dict[str, Any], 
        current_price: float
    ) -> PositionPnL:
        """Calculate P&L for a specific position"""
        
        try:
            portfolio_id = position_data["portfolio_id"]
            symbol = position_data["symbol"]
            quantity = position_data["quantity"]
            avg_cost = position_data["avg_cost"]
            realized_pnl = position_data.get("realized_pnl", 0.0)
            
            # Calculate position metrics
            market_value = quantity * current_price
            unrealized_pnl = (current_price - avg_cost) * quantity
            
            # Calculate P&L percentage
            if avg_cost > 0:
                pnl_percentage = ((current_price - avg_cost) / avg_cost) * 100
            else:
                pnl_percentage = 0.0
            
            # Calculate day P&L for position
            day_pnl = await self._calculate_position_day_pnl(portfolio_id, symbol)
            
            return PositionPnL(
                portfolio_id=portfolio_id,
                symbol=symbol,
                quantity=quantity,
                avg_cost=avg_cost,
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                day_pnl=day_pnl,
                pnl_percentage=pnl_percentage
            )
            
        except Exception as e:
            logger.error(f"Position P&L calculation failed: {e}")
            raise
    
    async def _calculate_day_pnl(self, portfolio_id: str) -> float:
        """Calculate day P&L for portfolio"""
        
        try:
            # Get portfolio value at start of day
            day_start = datetime.utcnow().replace(hour=9, minute=30, second=0, microsecond=0)
            
            start_value_key = f"portfolio_value:{portfolio_id}:{day_start.date()}"
            start_value_data = await self.redis.get(start_value_key)
            
            if start_value_data:
                start_value = float(start_value_data)
                
                # Get current portfolio value
                current_pnl = self.portfolio_pnl_cache.get(portfolio_id)
                if current_pnl:
                    current_value = current_pnl.total_value
                    return current_value - start_value
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Day P&L calculation failed: {e}")
            return 0.0
    
    async def _calculate_position_day_pnl(self, portfolio_id: str, symbol: str) -> float:
        """Calculate day P&L for a specific position"""
        
        try:
            # Get position value at start of day
            day_start = datetime.utcnow().replace(hour=9, minute=30, second=0, microsecond=0)
            
            start_value_key = f"position_value:{portfolio_id}:{symbol}:{day_start.date()}"
            start_value_data = await self.redis.get(start_value_key)
            
            if start_value_data:
                start_value = float(start_value_data)
                
                # Get current position value
                cache_key = f"{portfolio_id}:{symbol}"
                current_pnl = self.position_pnl_cache.get(cache_key)
                if current_pnl:
                    current_value = current_pnl.market_value
                    return current_value - start_value
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Position day P&L calculation failed: {e}")
            return 0.0
    
    async def _get_portfolio_data(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio data from Redis"""
        
        try:
            portfolio_key = f"portfolio:{portfolio_id}"
            portfolio_data = await self.redis.get(portfolio_key)
            
            if portfolio_data:
                return json.loads(portfolio_data)
            return None
            
        except Exception as e:
            logger.error(f"Portfolio data retrieval failed: {e}")
            return None
    
    async def _get_portfolio_positions(self, portfolio_id: str) -> List[Dict[str, Any]]:
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
            logger.error(f"Portfolio positions retrieval failed: {e}")
            return []
    
    async def _get_position_data(self, portfolio_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position data from Redis"""
        
        try:
            position_key = f"positions:{portfolio_id}:{symbol}"
            position_data = await self.redis.get(position_key)
            
            if position_data:
                return json.loads(position_data)
            return None
            
        except Exception as e:
            logger.error(f"Position data retrieval failed: {e}")
            return None
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        
        try:
            # Try to get from market data service
            price_key = f"market_price:{symbol}"
            price_data = await self.redis.get(price_key)
            
            if price_data:
                return float(price_data)
            
            # Fallback to mock price
            import random
            base_prices = {"AAPL": 150, "MSFT": 300, "GOOGL": 2500, "AMZN": 3000, "TSLA": 200}
            base_price = base_prices.get(symbol, 100)
            
            return base_price + random.uniform(-5, 5)
            
        except Exception as e:
            logger.error(f"Current price retrieval failed for {symbol}: {e}")
            return 100.0  # Fallback price
    
    async def _store_portfolio_pnl(self, pnl_snapshot: PnLSnapshot):
        """Store portfolio P&L snapshot in Redis"""
        
        try:
            # Store current P&L
            current_key = f"pnl:portfolio:{pnl_snapshot.portfolio_id}:current"
            await self.redis.set(
                current_key,
                json.dumps(asdict(pnl_snapshot), default=str)
            )
            
            # Store in time series
            timestamp = pnl_snapshot.timestamp.timestamp()
            timeseries_key = f"pnl:portfolio:{pnl_snapshot.portfolio_id}:history"
            
            await self.redis.zadd(
                timeseries_key,
                {json.dumps(asdict(pnl_snapshot), default=str): timestamp}
            )
            
            # Store daily start value (for day P&L calculation)
            day_key = f"portfolio_value:{pnl_snapshot.portfolio_id}:{pnl_snapshot.timestamp.date()}"
            
            # Only set if it doesn't exist (preserves day start value)
            await self.redis.set(
                day_key, pnl_snapshot.total_value, nx=True, ex=86400  # 24 hour expiry
            )
            
        except Exception as e:
            logger.error(f"Portfolio P&L storage failed: {e}")
    
    async def _store_position_pnl(self, position_pnl: PositionPnL):
        """Store position P&L in Redis"""
        
        try:
            # Store current position P&L
            current_key = f"pnl:position:{position_pnl.portfolio_id}:{position_pnl.symbol}:current"
            await self.redis.set(
                current_key,
                json.dumps(asdict(position_pnl), default=str)
            )
            
            # Store in time series
            timestamp = datetime.utcnow().timestamp()
            timeseries_key = f"pnl:position:{position_pnl.portfolio_id}:{position_pnl.symbol}:history"
            
            await self.redis.zadd(
                timeseries_key,
                {json.dumps(asdict(position_pnl), default=str): timestamp}
            )
            
            # Store daily start value
            day_key = f"position_value:{position_pnl.portfolio_id}:{position_pnl.symbol}:{datetime.utcnow().date()}"
            await self.redis.set(
                day_key, position_pnl.market_value, nx=True, ex=86400
            )
            
        except Exception as e:
            logger.error(f"Position P&L storage failed: {e}")
    
    async def _get_historical_pnl(
        self, 
        portfolio_id: str, 
        timeframe: str
    ) -> List[Dict[str, Any]]:
        """Get historical P&L data for a portfolio"""
        
        try:
            # Calculate time range
            now = datetime.utcnow()
            
            if timeframe == "1d":
                start_time = now - timedelta(days=1)
            elif timeframe == "1w":
                start_time = now - timedelta(weeks=1)
            elif timeframe == "1m":
                start_time = now - timedelta(days=30)
            elif timeframe == "3m":
                start_time = now - timedelta(days=90)
            else:
                start_time = now - timedelta(days=1)
            
            # Get P&L history from Redis
            timeseries_key = f"pnl:portfolio:{portfolio_id}:history"
            
            history_data = await self.redis.zrangebyscore(
                timeseries_key,
                start_time.timestamp(),
                now.timestamp()
            )
            
            historical_pnl = []
            for data in history_data:
                pnl_point = json.loads(data)
                historical_pnl.append(pnl_point)
            
            return historical_pnl
            
        except Exception as e:
            logger.error(f"Historical P&L retrieval failed: {e}")
            return []
    
    async def _get_position_pnl_history(
        self, 
        portfolio_id: str, 
        symbol: str
    ) -> List[Dict[str, Any]]:
        """Get P&L history for a specific position"""
        
        try:
            # Get last 24 hours of position P&L
            now = datetime.utcnow()
            start_time = now - timedelta(days=1)
            
            timeseries_key = f"pnl:position:{portfolio_id}:{symbol}:history"
            
            history_data = await self.redis.zrangebyscore(
                timeseries_key,
                start_time.timestamp(),
                now.timestamp()
            )
            
            pnl_history = []
            for data in history_data:
                pnl_point = json.loads(data)
                pnl_history.append(pnl_point)
            
            return pnl_history
            
        except Exception as e:
            logger.error(f"Position P&L history retrieval failed: {e}")
            return []
    
    async def _calculate_performance_metrics(
        self, 
        portfolio_id: str, 
        timeframe: str
    ) -> Dict[str, Any]:
        """Calculate performance metrics for portfolio"""
        
        try:
            historical_pnl = await self._get_historical_pnl(portfolio_id, timeframe)
            
            if len(historical_pnl) < 2:
                return {"error": "Insufficient data for performance metrics"}
            
            # Extract P&L values
            pnl_values = [float(point["total_pnl"]) for point in historical_pnl]
            
            # Calculate metrics
            max_pnl = max(pnl_values)
            min_pnl = min(pnl_values)
            current_pnl = pnl_values[-1]
            
            # Calculate drawdown
            running_max = 0
            max_drawdown = 0
            
            for pnl in pnl_values:
                running_max = max(running_max, pnl)
                drawdown = running_max - pnl
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate volatility (simplified)
            if len(pnl_values) > 1:
                volatility = statistics.stdev(pnl_values)
            else:
                volatility = 0.0
            
            return {
                "max_pnl": max_pnl,
                "min_pnl": min_pnl,
                "current_pnl": current_pnl,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "data_points": len(pnl_values),
                "timeframe": timeframe
            }
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {"error": str(e)}
    
    async def _load_pnl_cache(self):
        """Load P&L cache from Redis"""
        
        try:
            # Load portfolio P&L cache
            portfolio_keys = await self.redis.keys("pnl:portfolio:*:current")
            
            for key in portfolio_keys:
                portfolio_id = key.split(":")[2]
                pnl_data = await self.redis.get(key)
                
                if pnl_data:
                    pnl_dict = json.loads(pnl_data)
                    pnl_snapshot = PnLSnapshot(
                        portfolio_id=pnl_dict["portfolio_id"],
                        timestamp=datetime.fromisoformat(pnl_dict["timestamp"]),
                        total_pnl=pnl_dict["total_pnl"],
                        realized_pnl=pnl_dict["realized_pnl"],
                        unrealized_pnl=pnl_dict["unrealized_pnl"],
                        day_pnl=pnl_dict["day_pnl"],
                        cash_balance=pnl_dict["cash_balance"],
                        total_value=pnl_dict["total_value"],
                        positions_value=pnl_dict["positions_value"]
                    )
                    self.portfolio_pnl_cache[portfolio_id] = pnl_snapshot
            
            # Load position P&L cache
            position_keys = await self.redis.keys("pnl:position:*:current")
            
            for key in position_keys:
                parts = key.split(":")
                portfolio_id = parts[2]
                symbol = parts[3]
                cache_key = f"{portfolio_id}:{symbol}"
                
                pnl_data = await self.redis.get(key)
                
                if pnl_data:
                    pnl_dict = json.loads(pnl_data)
                    position_pnl = PositionPnL(
                        portfolio_id=pnl_dict["portfolio_id"],
                        symbol=pnl_dict["symbol"],
                        quantity=pnl_dict["quantity"],
                        avg_cost=pnl_dict["avg_cost"],
                        current_price=pnl_dict["current_price"],
                        market_value=pnl_dict["market_value"],
                        unrealized_pnl=pnl_dict["unrealized_pnl"],
                        realized_pnl=pnl_dict["realized_pnl"],
                        day_pnl=pnl_dict["day_pnl"],
                        pnl_percentage=pnl_dict["pnl_percentage"]
                    )
                    self.position_pnl_cache[cache_key] = position_pnl
            
            logger.info(f"Loaded P&L cache: {len(self.portfolio_pnl_cache)} portfolios, {len(self.position_pnl_cache)} positions")
            
        except Exception as e:
            logger.error(f"P&L cache loading failed: {e}")
    
    async def _update_calculation_stats(self, calculation_time_ms: float):
        """Update P&L calculation statistics"""
        
        try:
            self.calculation_stats["total_calculations"] += 1
            
            # Update average calculation time (exponential moving average)
            alpha = 0.1
            current_avg = self.calculation_stats["avg_calculation_time_ms"]
            self.calculation_stats["avg_calculation_time_ms"] = (
                alpha * calculation_time_ms + (1 - alpha) * current_avg
            )
            
        except Exception as e:
            logger.error(f"Calculation stats update failed: {e}")
    
    # Background tasks
    async def _pnl_snapshot_task(self):
        """Background task to take regular P&L snapshots"""
        
        while True:
            try:
                await asyncio.sleep(self.config["snapshot_interval"])
                
                # Get all active portfolios
                portfolio_keys = await self.redis.keys("portfolio:*")
                
                for key in portfolio_keys:
                    portfolio_id = key.split(":")[-1]
                    try:
                        await self.update_portfolio_pnl(portfolio_id)
                    except Exception as e:
                        logger.error(f"Snapshot failed for portfolio {portfolio_id}: {e}")
                
            except Exception as e:
                logger.error(f"P&L snapshot task error: {e}")
                await asyncio.sleep(60)
    
    async def _mark_to_market_task(self):
        """Background task for mark-to-market updates"""
        
        while True:
            try:
                await asyncio.sleep(self.config["mark_to_market_interval"])
                
                # Update all cached positions with current market prices
                for cache_key, position_pnl in list(self.position_pnl_cache.items()):
                    portfolio_id, symbol = cache_key.split(":")
                    try:
                        await self.update_position_pnl(portfolio_id, symbol)
                    except Exception as e:
                        logger.error(f"Mark-to-market failed for {cache_key}: {e}")
                
            except Exception as e:
                logger.error(f"Mark-to-market task error: {e}")
                await asyncio.sleep(30)
    
    async def _pnl_history_cleanup(self):
        """Background task to cleanup old P&L history"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Calculate cutoff timestamp
                cutoff_time = datetime.utcnow() - timedelta(days=self.config["history_retention_days"])
                cutoff_timestamp = cutoff_time.timestamp()
                
                # Cleanup portfolio P&L history
                portfolio_history_keys = await self.redis.keys("pnl:portfolio:*:history")
                
                for key in portfolio_history_keys:
                    await self.redis.zremrangebyscore(key, 0, cutoff_timestamp)
                
                # Cleanup position P&L history
                position_history_keys = await self.redis.keys("pnl:position:*:history")
                
                for key in position_history_keys:
                    await self.redis.zremrangebyscore(key, 0, cutoff_timestamp)
                
                logger.info("P&L history cleanup completed")
                
            except Exception as e:
                logger.error(f"P&L history cleanup error: {e}")
                await asyncio.sleep(3600)

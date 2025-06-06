"""
Market Data Manager for real-time market data integration
"""

import asyncio
import aiohttp
import aioredis
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import time
import random
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class MarketQuote:
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    change: float
    change_percent: float
    high: float
    low: float
    open_price: float
    timestamp: datetime

@dataclass
class OptionsData:
    symbol: str
    option_type: str  # call or put
    strike: float
    expiry: str
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    timestamp: datetime

class MarketDataManager:
    def __init__(self):
        self.redis = None
        self.session = None
        
        # Market data configuration
        self.config = {
            "update_interval": 5,  # Update prices every 5 seconds
            "cache_ttl": 300,      # Cache for 5 minutes
            "max_symbols": 1000,   # Maximum symbols to track
            "data_sources": {
                "primary": "polygon",    # Primary data source
                "fallback": "mock"       # Fallback to mock data
            },
            "options_enabled": True,
            "market_hours": {
                "start": "09:30",
                "end": "16:00",
                "timezone": "US/Eastern"
            }
        }
        
        # Data source APIs
        self.data_sources = {
            "polygon": {
                "base_url": "https://api.polygon.io",
                "api_key": "your_polygon_api_key",
                "endpoints": {
                    "quotes": "/v2/aggs/ticker/{symbol}/range/1/minute/{date}/{date}",
                    "options": "/v3/reference/options/contracts"
                }
            },
            "alpha_vantage": {
                "base_url": "https://www.alphavantage.co",
                "api_key": "your_alpha_vantage_api_key",
                "endpoints": {
                    "quotes": "/query?function=GLOBAL_QUOTE&symbol={symbol}",
                    "intraday": "/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min"
                }
            }
        }
        
        # Market data cache
        self.price_cache = {}
        self.options_cache = {}
        self.subscribers = set()
        
        # Performance tracking
        self.stats = {
            "total_updates": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "avg_update_time_ms": 0.0,
            "data_source_performance": {}
        }
        
        # Mock data for development
        self.mock_base_prices = {
            "AAPL": 150.0, "MSFT": 300.0, "GOOGL": 2500.0, "AMZN": 3000.0,
            "TSLA": 200.0, "NVDA": 400.0, "META": 250.0, "NFLX": 400.0,
            "SPY": 450.0, "QQQ": 350.0, "IWM": 200.0, "VIX": 20.0
        }
        
    async def initialize(self):
        """Initialize the market data manager"""
        try:
            # Initialize Redis connection
            self.redis = aioredis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Load cached data
            await self._load_cached_data()
            
            # Test data source connections
            await self._test_data_sources()
            
            # Start background tasks
            asyncio.create_task(self._market_data_update_task())
            asyncio.create_task(self._options_data_update_task())
            asyncio.create_task(self._cache_cleanup_task())
            
            logger.info("Market data manager initialized")
            
        except Exception as e:
            logger.error(f"Market data manager initialization failed: {e}")
            raise
    
    async def close(self):
        """Close the market data manager"""
        if self.session:
            await self.session.close()
        if self.redis:
            await self.redis.close()
    
    async def get_status(self) -> str:
        """Get market data manager status"""
        try:
            if not self.redis or not self.session:
                return "unhealthy - missing connections"
            
            # Test Redis connection
            await self.redis.ping()
            
            # Check if we have recent data
            if self.price_cache:
                latest_update = max(
                    quote.timestamp for quote in self.price_cache.values()
                    if hasattr(quote, 'timestamp')
                )
                
                time_diff = datetime.utcnow() - latest_update
                if time_diff.total_seconds() > 300:  # 5 minutes
                    return "degraded - stale data"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Market data status check failed: {e}")
            return "unhealthy"
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        
        try:
            # Check cache first
            if symbol in self.price_cache:
                quote = self.price_cache[symbol]
                
                # Check if data is fresh (within cache TTL)
                if hasattr(quote, 'timestamp'):
                    age = (datetime.utcnow() - quote.timestamp).total_seconds()
                    if age < self.config["cache_ttl"]:
                        self.stats["cache_hits"] += 1
                        return quote.price
            
            # Fetch fresh data
            quote = await self._fetch_quote(symbol)
            if quote:
                return quote.price
            
            # Fallback to mock data
            return self._generate_mock_price(symbol)
            
        except Exception as e:
            logger.error(f"Current price retrieval failed for {symbol}: {e}")
            return self._generate_mock_price(symbol)
    
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols"""
        
        try:
            prices = {}
            
            # Use batch processing for efficiency
            tasks = []
            for symbol in symbols[:self.config["max_symbols"]]:
                task = self.get_current_price(symbol)
                tasks.append((symbol, task))
            
            # Execute all requests concurrently
            results = await asyncio.gather(
                *[task for _, task in tasks], return_exceptions=True
            )
            
            # Process results
            for i, (symbol, _) in enumerate(tasks):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"Price fetch failed for {symbol}: {result}")
                    prices[symbol] = self._generate_mock_price(symbol)
                else:
                    prices[symbol] = result
            
            return prices
            
        except Exception as e:
            logger.error(f"Batch price retrieval failed: {e}")
            return {symbol: self._generate_mock_price(symbol) for symbol in symbols}
    
    async def get_detailed_quote(self, symbol: str) -> Dict[str, Any]:
        """Get detailed quote information for a symbol"""
        
        try:
            # Try to get from cache first
            if symbol in self.price_cache:
                quote = self.price_cache[symbol]
                
                if hasattr(quote, 'timestamp'):
                    age = (datetime.utcnow() - quote.timestamp).total_seconds()
                    if age < self.config["cache_ttl"]:
                        return asdict(quote)
            
            # Fetch fresh detailed quote
            quote = await self._fetch_detailed_quote(symbol)
            if quote:
                return asdict(quote)
            
            # Fallback to mock detailed quote
            return self._generate_mock_detailed_quote(symbol)
            
        except Exception as e:
            logger.error(f"Detailed quote retrieval failed for {symbol}: {e}")
            return self._generate_mock_detailed_quote(symbol)
    
    async def get_options_data(
        self, 
        symbol: str, 
        expiry: Optional[str] = None,
        option_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get options data for a symbol"""
        
        try:
            if not self.config["options_enabled"]:
                return []
            
            cache_key = f"{symbol}_{expiry}_{option_type}"
            
            # Check options cache
            if cache_key in self.options_cache:
                cached_data = self.options_cache[cache_key]
                
                if isinstance(cached_data, list) and cached_data:
                    first_item = cached_data[0]
                    if hasattr(first_item, 'timestamp'):
                        age = (datetime.utcnow() - first_item.timestamp).total_seconds()
                        if age < self.config["cache_ttl"]:
                            return [asdict(option) for option in cached_data]
            
            # Fetch fresh options data
            options_data = await self._fetch_options_data(symbol, expiry, option_type)
            if options_data:
                return [asdict(option) for option in options_data]
            
            # Fallback to mock options data
            return self._generate_mock_options_data(symbol)
            
        except Exception as e:
            logger.error(f"Options data retrieval failed for {symbol}: {e}")
            return []
    
    async def update_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Update prices for specific symbols"""
        
        try:
            start_time = time.time()
            
            updated_prices = {}
            
            for symbol in symbols:
                try:
                    quote = await self._fetch_quote(symbol)
                    if quote:
                        self.price_cache[symbol] = quote
                        updated_prices[symbol] = quote.price
                        
                        # Store in Redis
                        await self._store_quote_in_redis(quote)
                    
                except Exception as e:
                    logger.error(f"Price update failed for {symbol}: {e}")
                    updated_prices[symbol] = self._generate_mock_price(symbol)
            
            # Update performance stats
            update_time = (time.time() - start_time) * 1000
            await self._update_performance_stats(update_time, len(symbols))
            
            return updated_prices
            
        except Exception as e:
            logger.error(f"Batch price update failed: {e}")
            return {}
    
    async def subscribe_to_symbol(self, symbol: str):
        """Subscribe to real-time updates for a symbol"""
        
        try:
            self.subscribers.add(symbol)
            
            # Store subscription in Redis
            await self.redis.sadd("market_data_subscriptions", symbol)
            
            logger.info(f"Subscribed to market data for {symbol}")
            
        except Exception as e:
            logger.error(f"Symbol subscription failed for {symbol}: {e}")
    
    async def unsubscribe_from_symbol(self, symbol: str):
        """Unsubscribe from real-time updates for a symbol"""
        
        try:
            self.subscribers.discard(symbol)
            
            # Remove subscription from Redis
            await self.redis.srem("market_data_subscriptions", symbol)
            
            logger.info(f"Unsubscribed from market data for {symbol}")
            
        except Exception as e:
            logger.error(f"Symbol unsubscription failed for {symbol}: {e}")
    
    async def _fetch_quote(self, symbol: str) -> Optional[MarketQuote]:
        """Fetch quote from primary data source"""
        
        try:
            # Try primary data source first
            quote = await self._fetch_from_polygon(symbol)
            if quote:
                return quote
            
            # Try fallback data source
            quote = await self._fetch_from_alpha_vantage(symbol)
            if quote:
                return quote
            
            # Generate mock data as last resort
            return self._generate_mock_quote(symbol)
            
        except Exception as e:
            logger.error(f"Quote fetch failed for {symbol}: {e}")
            return None
    
    async def _fetch_detailed_quote(self, symbol: str) -> Optional[MarketQuote]:
        """Fetch detailed quote with all fields"""
        
        try:
            # This would call a detailed quote endpoint
            # For now, return the same as regular quote
            return await self._fetch_quote(symbol)
            
        except Exception as e:
            logger.error(f"Detailed quote fetch failed for {symbol}: {e}")
            return None
    
    async def _fetch_options_data(
        self, 
        symbol: str, 
        expiry: Optional[str] = None,
        option_type: Optional[str] = None
    ) -> Optional[List[OptionsData]]:
        """Fetch options data from data source"""
        
        try:
            # This would call options data endpoints
            # For now, return mock options data
            return self._generate_mock_options_data(symbol)
            
        except Exception as e:
            logger.error(f"Options data fetch failed for {symbol}: {e}")
            return None
    
    async def _fetch_from_polygon(self, symbol: str) -> Optional[MarketQuote]:
        """Fetch quote from Polygon API"""
        
        try:
            polygon_config = self.data_sources["polygon"]
            
            # For development, we'll use mock data
            # In production, this would make actual API calls
            if polygon_config["api_key"] == "your_polygon_api_key":
                return self._generate_mock_quote(symbol)
            
            # Real API call would go here
            url = f"{polygon_config['base_url']}/v2/last/trade/{symbol}"
            headers = {"Authorization": f"Bearer {polygon_config['api_key']}"}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse Polygon response format
                    return MarketQuote(
                        symbol=symbol,
                        price=data.get("results", {}).get("p", 0.0),
                        bid=data.get("results", {}).get("p", 0.0) - 0.01,
                        ask=data.get("results", {}).get("p", 0.0) + 0.01,
                        volume=data.get("results", {}).get("s", 0),
                        change=0.0,  # Would calculate from previous close
                        change_percent=0.0,
                        high=data.get("results", {}).get("p", 0.0) + 2.0,
                        low=data.get("results", {}).get("p", 0.0) - 2.0,
                        open_price=data.get("results", {}).get("p", 0.0),
                        timestamp=datetime.utcnow()
                    )
                else:
                    logger.warning(f"Polygon API error {response.status} for {symbol}")
                    return None
            
        except Exception as e:
            logger.error(f"Polygon fetch failed for {symbol}: {e}")
            return None
    
    async def _fetch_from_alpha_vantage(self, symbol: str) -> Optional[MarketQuote]:
        """Fetch quote from Alpha Vantage API"""
        
        try:
            av_config = self.data_sources["alpha_vantage"]
            
            # For development, we'll use mock data
            if av_config["api_key"] == "your_alpha_vantage_api_key":
                return self._generate_mock_quote(symbol)
            
            # Real API call would go here
            url = av_config["base_url"] + av_config["endpoints"]["quotes"].format(symbol=symbol)
            params = {"apikey": av_config["api_key"]}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse Alpha Vantage response format
                    quote_data = data.get("Global Quote", {})
                    
                    return MarketQuote(
                        symbol=symbol,
                        price=float(quote_data.get("05. price", 0.0)),
                        bid=float(quote_data.get("05. price", 0.0)) - 0.01,
                        ask=float(quote_data.get("05. price", 0.0)) + 0.01,
                        volume=int(quote_data.get("06. volume", 0)),
                        change=float(quote_data.get("09. change", 0.0)),
                        change_percent=float(quote_data.get("10. change percent", "0%").replace("%", "")),
                        high=float(quote_data.get("03. high", 0.0)),
                        low=float(quote_data.get("04. low", 0.0)),
                        open_price=float(quote_data.get("02. open", 0.0)),
                        timestamp=datetime.utcnow()
                    )
                else:
                    logger.warning(f"Alpha Vantage API error {response.status} for {symbol}")
                    return None
            
        except Exception as e:
            logger.error(f"Alpha Vantage fetch failed for {symbol}: {e}")
            return None
    
    def _generate_mock_quote(self, symbol: str) -> MarketQuote:
        """Generate mock quote data for development"""
        
        try:
            base_price = self.mock_base_prices.get(symbol, 100.0)
            
            # Add some realistic random variation
            price_variation = random.uniform(-0.02, 0.02)  # ±2%
            current_price = base_price * (1 + price_variation)
            
            # Generate realistic bid/ask spread
            spread = current_price * 0.001  # 0.1% spread
            bid = current_price - (spread / 2)
            ask = current_price + (spread / 2)
            
            # Generate other fields
            volume = random.randint(100000, 10000000)
            daily_change = random.uniform(-0.05, 0.05)  # ±5% daily change
            change_amount = base_price * daily_change
            
            return MarketQuote(
                symbol=symbol,
                price=round(current_price, 2),
                bid=round(bid, 2),
                ask=round(ask, 2),
                volume=volume,
                change=round(change_amount, 2),
                change_percent=round(daily_change * 100, 2),
                high=round(current_price * 1.03, 2),
                low=round(current_price * 0.97, 2),
                open_price=round(base_price, 2),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Mock quote generation failed for {symbol}: {e}")
            return MarketQuote(
                symbol=symbol, price=100.0, bid=99.95, ask=100.05,
                volume=1000000, change=0.0, change_percent=0.0,
                high=102.0, low=98.0, open_price=100.0,
                timestamp=datetime.utcnow()
            )
    
    def _generate_mock_price(self, symbol: str) -> float:
        """Generate mock price for development"""
        
        try:
            base_price = self.mock_base_prices.get(symbol, 100.0)
            variation = random.uniform(-0.01, 0.01)  # ±1% variation
            return round(base_price * (1 + variation), 2)
            
        except Exception:
            return 100.0
    
    def _generate_mock_detailed_quote(self, symbol: str) -> Dict[str, Any]:
        """Generate mock detailed quote"""
        
        try:
            quote = self._generate_mock_quote(symbol)
            return asdict(quote)
            
        except Exception as e:
            logger.error(f"Mock detailed quote generation failed: {e}")
            return {}
    
    def _generate_mock_options_data(self, symbol: str) -> List[OptionsData]:
        """Generate mock options data"""
        
        try:
            current_price = self.mock_base_prices.get(symbol, 100.0)
            options_data = []
            
            # Generate options for different strikes and expiries
            strikes = [current_price * mult for mult in [0.9, 0.95, 1.0, 1.05, 1.1]]
            expiries = ["2024-03-15", "2024-04-19", "2024-06-21"]
            
            for expiry in expiries:
                for strike in strikes:
                    for option_type in ["call", "put"]:
                        # Calculate basic option values (simplified Black-Scholes)
                        moneyness = current_price / strike
                        
                        if option_type == "call":
                            intrinsic = max(0, current_price - strike)
                            delta = 0.5 + (moneyness - 1) * 0.3
                        else:
                            intrinsic = max(0, strike - current_price)
                            delta = -0.5 + (1 - moneyness) * 0.3
                        
                        time_value = random.uniform(0.5, 5.0)
                        option_price = intrinsic + time_value
                        
                        options_data.append(OptionsData(
                            symbol=f"{symbol}_{expiry}_{option_type}_{strike}",
                            option_type=option_type,
                            strike=round(strike, 2),
                            expiry=expiry,
                            bid=round(option_price - 0.05, 2),
                            ask=round(option_price + 0.05, 2),
                            last=round(option_price, 2),
                            volume=random.randint(10, 1000),
                            open_interest=random.randint(100, 10000),
                            implied_volatility=round(random.uniform(0.15, 0.4), 3),
                            delta=round(delta, 3),
                            gamma=round(random.uniform(0.001, 0.01), 4),
                            theta=round(random.uniform(-0.1, -0.01), 4),
                            vega=round(random.uniform(0.05, 0.2), 3),
                            timestamp=datetime.utcnow()
                        ))
            
            return options_data
            
        except Exception as e:
            logger.error(f"Mock options data generation failed: {e}")
            return []
    
    async def _store_quote_in_redis(self, quote: MarketQuote):
        """Store quote in Redis cache"""
        
        try:
            # Store current price
            await self.redis.setex(
                f"market_price:{quote.symbol}",
                self.config["cache_ttl"],
                quote.price
            )
            
            # Store detailed quote
            await self.redis.setex(
                f"market_quote:{quote.symbol}",
                self.config["cache_ttl"],
                json.dumps(asdict(quote), default=str)
            )
            
        except Exception as e:
            logger.error(f"Quote storage failed for {quote.symbol}: {e}")
    
    async def _load_cached_data(self):
        """Load cached market data from Redis"""
        
        try:
            # Load cached quotes
            quote_keys = await self.redis.keys("market_quote:*")
            
            for key in quote_keys:
                quote_data = await self.redis.get(key)
                if quote_data:
                    quote_dict = json.loads(quote_data)
                    
                    quote = MarketQuote(
                        symbol=quote_dict["symbol"],
                        price=quote_dict["price"],
                        bid=quote_dict["bid"],
                        ask=quote_dict["ask"],
                        volume=quote_dict["volume"],
                        change=quote_dict["change"],
                        change_percent=quote_dict["change_percent"],
                        high=quote_dict["high"],
                        low=quote_dict["low"],
                        open_price=quote_dict["open_price"],
                        timestamp=datetime.fromisoformat(quote_dict["timestamp"])
                    )
                    
                    self.price_cache[quote.symbol] = quote
            
            # Load subscriptions
            subscriptions = await self.redis.smembers("market_data_subscriptions")
            self.subscribers.update(subscriptions)
            
            logger.info(f"Loaded market data cache: {len(self.price_cache)} quotes, {len(self.subscribers)} subscriptions")
            
        except Exception as e:
            logger.error(f"Cached data loading failed: {e}")
    
    async def _test_data_sources(self):
        """Test connections to data sources"""
        
        try:
            # Test Polygon
            try:
                if self.data_sources["polygon"]["api_key"] != "your_polygon_api_key":
                    async with self.session.get(
                        f"{self.data_sources['polygon']['base_url']}/v2/last/trade/AAPL",
                        headers={"Authorization": f"Bearer {self.data_sources['polygon']['api_key']}"}
                    ) as response:
                        if response.status == 200:
                            logger.info("Polygon API connection successful")
                        else:
                            logger.warning(f"Polygon API test failed: {response.status}")
                else:
                    logger.info("Polygon API not configured - using mock data")
            except Exception as e:
                logger.warning(f"Polygon API test failed: {e}")
            
            # Test Alpha Vantage
            try:
                if self.data_sources["alpha_vantage"]["api_key"] != "your_alpha_vantage_api_key":
                    async with self.session.get(
                        f"{self.data_sources['alpha_vantage']['base_url']}/query",
                        params={"function": "GLOBAL_QUOTE", "symbol": "AAPL", "apikey": self.data_sources['alpha_vantage']['api_key']}
                    ) as response:
                        if response.status == 200:
                            logger.info("Alpha Vantage API connection successful")
                        else:
                            logger.warning(f"Alpha Vantage API test failed: {response.status}")
                else:
                    logger.info("Alpha Vantage API not configured - using mock data")
            except Exception as e:
                logger.warning(f"Alpha Vantage API test failed: {e}")
                
        except Exception as e:
            logger.error(f"Data source testing failed: {e}")
    
    async def _update_performance_stats(self, update_time_ms: float, symbol_count: int):
        """Update performance statistics"""
        
        try:
            self.stats["total_updates"] += 1
            self.stats["api_calls"] += symbol_count
            
            # Update average update time (exponential moving average)
            alpha = 0.1
            current_avg = self.stats["avg_update_time_ms"]
            self.stats["avg_update_time_ms"] = (
                alpha * update_time_ms + (1 - alpha) * current_avg
            )
            
        except Exception as e:
            logger.error(f"Performance stats update failed: {e}")
    
    # Background tasks
    async def _market_data_update_task(self):
        """Background task to update market data"""
        
        while True:
            try:
                await asyncio.sleep(self.config["update_interval"])
                
                # Update subscribed symbols
                if self.subscribers:
                    symbols_to_update = list(self.subscribers)
                    await self.update_prices(symbols_to_update)
                
            except Exception as e:
                logger.error(f"Market data update task error: {e}")
                await asyncio.sleep(30)
    
    async def _options_data_update_task(self):
        """Background task to update options data"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Update options every minute
                
                if not self.config["options_enabled"]:
                    continue
                
                # Update options data for subscribed symbols
                for symbol in list(self.subscribers):
                    try:
                        options_data = await self._fetch_options_data(symbol)
                        if options_data:
                            self.options_cache[symbol] = options_data
                    except Exception as e:
                        logger.error(f"Options update failed for {symbol}: {e}")
                
            except Exception as e:
                logger.error(f"Options data update task error: {e}")
                await asyncio.sleep(300)
    
    async def _cache_cleanup_task(self):
        """Background task to cleanup stale cache data"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Cleanup stale price cache
                current_time = datetime.utcnow()
                stale_symbols = []
                
                for symbol, quote in self.price_cache.items():
                    if hasattr(quote, 'timestamp'):
                        age = (current_time - quote.timestamp).total_seconds()
                        if age > self.config["cache_ttl"] * 2:  # Double the TTL
                            stale_symbols.append(symbol)
                
                for symbol in stale_symbols:
                    self.price_cache.pop(symbol, None)
                
                # Cleanup options cache
                stale_option_keys = []
                for key, options_list in self.options_cache.items():
                    if options_list and hasattr(options_list[0], 'timestamp'):
                        age = (current_time - options_list[0].timestamp).total_seconds()
                        if age > self.config["cache_ttl"] * 2:
                            stale_option_keys.append(key)
                
                for key in stale_option_keys:
                    self.options_cache.pop(key, None)
                
                logger.info(f"Cache cleanup: removed {len(stale_symbols)} stale quotes, {len(stale_option_keys)} stale options")
                
            except Exception as e:
                logger.error(f"Cache cleanup task error: {e}")
                await asyncio.sleep(3600)

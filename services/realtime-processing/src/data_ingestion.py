"""
Real-time Data Ingestion for streaming market data and news
"""

import asyncio
import aiohttp
import websockets
import aioredis
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable
from datetime import datetime, timedelta
import json
import time
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    symbol: str
    price: float
    volume: int
    bid: float
    ask: float
    timestamp: datetime
    source: str
    change: float = 0.0
    change_percent: float = 0.0

@dataclass
class NewsItem:
    id: str
    title: str
    content: str
    source: str
    timestamp: datetime
    symbols_mentioned: List[str]
    sentiment_score: Optional[float] = None
    relevance_score: float = 1.0

@dataclass
class OptionsData:
    symbol: str
    strike: float
    expiry: str
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    timestamp: datetime

class RealTimeDataIngestion:
    def __init__(self):
        self.redis = None
        self.session = None
        
        # Data sources configuration
        self.data_sources = {
            "market_data": {
                "polygon": {
                    "url": "wss://socket.polygon.io/stocks",
                    "api_key": "YOUR_POLYGON_API_KEY",
                    "enabled": True
                },
                "alpha_vantage": {
                    "url": "https://www.alphavantage.co/query",
                    "api_key": "YOUR_AV_API_KEY",
                    "enabled": True
                }
            },
            "news_sources": {
                "newsapi": {
                    "url": "https://newsapi.org/v2/everything",
                    "api_key": "YOUR_NEWS_API_KEY",
                    "enabled": True
                },
                "finnhub": {
                    "url": "https://finnhub.io/api/v1/news",
                    "api_key": "YOUR_FINNHUB_API_KEY",
                    "enabled": True
                }
            },
            "options_data": {
                "tradier": {
                    "url": "https://api.tradier.com/v1/markets/options/chains",
                    "api_key": "YOUR_TRADIER_API_KEY",
                    "enabled": False  # Premium feature
                }
            }
        }
        
        # WebSocket connections
        self.ws_connections = {}
        
        # Data buffers
        self.market_data_buffer = {}
        self.news_buffer = []
        self.options_buffer = {}
        
        # Event handlers
        self.data_handlers: Dict[str, List[Callable]] = {
            "market_data": [],
            "news": [],
            "options": []
        }
        
        # Background tasks
        self.background_tasks = []
        
        # Rate limiting
        self.rate_limits = {
            "polygon": {"requests_per_minute": 300, "current": 0, "reset_time": time.time()},
            "alpha_vantage": {"requests_per_minute": 5, "current": 0, "reset_time": time.time()},
            "newsapi": {"requests_per_minute": 100, "current": 0, "reset_time": time.time()}
        }
        
        # Configuration
        self.buffer_size = 10000
        self.batch_size = 100
        self.update_frequency = 1.0  # seconds
        
    async def start(self):
        """Start the data ingestion service"""
        try:
            # Initialize Redis connection
            self.redis = aioredis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            
            # Initialize HTTP session
            self.session = aiohttp.ClientSession()
            
            # Start background tasks
            self.background_tasks = [
                asyncio.create_task(self._market_data_poller()),
                asyncio.create_task(self._news_fetcher()),
                asyncio.create_task(self._options_data_fetcher()),
                asyncio.create_task(self._data_processor()),
                asyncio.create_task(self._buffer_manager()),
                asyncio.create_task(self._rate_limit_resetter())
            ]
            
            # Initialize WebSocket connections
            await self._initialize_websocket_connections()
            
            logger.info("Real-time data ingestion started successfully")
            
        except Exception as e:
            logger.error(f"Data ingestion startup failed: {e}")
            raise
    
    async def stop(self):
        """Stop the data ingestion service"""
        try:
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Close WebSocket connections
            for ws in self.ws_connections.values():
                if not ws.closed:
                    await ws.close()
            
            # Close HTTP session
            if self.session:
                await self.session.close()
            
            # Close Redis connection
            if self.redis:
                await self.redis.close()
            
            logger.info("Data ingestion stopped")
            
        except Exception as e:
            logger.error(f"Data ingestion shutdown error: {e}")
    
    async def health_check(self) -> str:
        """Check data ingestion health"""
        try:
            if not self.redis:
                return "unhealthy - no redis connection"
            
            # Test Redis connection
            await self.redis.ping()
            
            # Check WebSocket connections
            active_ws = sum(1 for ws in self.ws_connections.values() if not ws.closed)
            total_ws = len(self.ws_connections)
            
            # Check background tasks
            running_tasks = sum(1 for task in self.background_tasks if not task.done())
            
            if running_tasks == len(self.background_tasks) and active_ws >= total_ws * 0.5:
                return "healthy"
            else:
                return f"degraded - {running_tasks}/{len(self.background_tasks)} tasks, {active_ws}/{total_ws} ws"
                
        except Exception as e:
            logger.error(f"Data ingestion health check failed: {e}")
            return "unhealthy"
    
    async def get_current_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get current market data for specified symbols"""
        
        try:
            market_data = {}
            
            for symbol in symbols:
                # Get latest data from buffer
                if symbol in self.market_data_buffer:
                    latest_data = self.market_data_buffer[symbol][-1] if self.market_data_buffer[symbol] else None
                    
                    if latest_data:
                        market_data[symbol] = {
                            "price": latest_data.price,
                            "volume": latest_data.volume,
                            "bid": latest_data.bid,
                            "ask": latest_data.ask,
                            "change": latest_data.change,
                            "change_percent": latest_data.change_percent,
                            "timestamp": latest_data.timestamp.isoformat(),
                            "source": latest_data.source
                        }
                    else:
                        # Fetch fresh data
                        fresh_data = await self._fetch_symbol_data(symbol)
                        if fresh_data:
                            market_data[symbol] = fresh_data
                else:
                    # Fetch fresh data
                    fresh_data = await self._fetch_symbol_data(symbol)
                    if fresh_data:
                        market_data[symbol] = fresh_data
            
            # Add market context
            market_context = await self._get_market_context()
            
            return {
                "symbols": symbols,
                "market_data": market_data,
                "market_context": market_context,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Current market data retrieval failed: {e}")
            return {"symbols": symbols, "market_data": {}, "market_context": {}, "error": str(e)}
    
    async def get_recent_news(self, symbols: List[str], hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent news for specified symbols"""
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            relevant_news = []
            
            for news_item in self.news_buffer:
                if (news_item.timestamp >= cutoff_time and
                    any(symbol in news_item.symbols_mentioned for symbol in symbols)):
                    
                    relevant_news.append({
                        "id": news_item.id,
                        "title": news_item.title,
                        "content": news_item.content[:500] + "..." if len(news_item.content) > 500 else news_item.content,
                        "source": news_item.source,
                        "timestamp": news_item.timestamp.isoformat(),
                        "symbols_mentioned": news_item.symbols_mentioned,
                        "sentiment_score": news_item.sentiment_score,
                        "relevance_score": news_item.relevance_score
                    })
            
            # Sort by relevance and recency
            relevant_news.sort(key=lambda x: (x["relevance_score"], x["timestamp"]), reverse=True)
            
            return relevant_news[:50]  # Return top 50 most relevant
            
        except Exception as e:
            logger.error(f"Recent news retrieval failed: {e}")
            return []
    
    async def get_options_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get options data for specified symbols"""
        
        try:
            options_data = {}
            
            for symbol in symbols:
                if symbol in self.options_buffer:
                    symbol_options = []
                    
                    for option in self.options_buffer[symbol]:
                        symbol_options.append({
                            "strike": option.strike,
                            "expiry": option.expiry,
                            "option_type": option.option_type,
                            "bid": option.bid,
                            "ask": option.ask,
                            "volume": option.volume,
                            "open_interest": option.open_interest,
                            "implied_volatility": option.implied_volatility,
                            "greeks": {
                                "delta": option.delta,
                                "gamma": option.gamma,
                                "theta": option.theta,
                                "vega": option.vega
                            },
                            "timestamp": option.timestamp.isoformat()
                        })
                    
                    options_data[symbol] = symbol_options
            
            return options_data
            
        except Exception as e:
            logger.error(f"Options data retrieval failed: {e}")
            return {}
    
    async def add_data_handler(self, data_type: str, handler: Callable):
        """Add a data handler for specific data types"""
        
        if data_type in self.data_handlers:
            self.data_handlers[data_type].append(handler)
        else:
            logger.warning(f"Unknown data type: {data_type}")
    
    async def _initialize_websocket_connections(self):
        """Initialize WebSocket connections for real-time data"""
        
        try:
            # Initialize Polygon WebSocket for market data
            if self.data_sources["market_data"]["polygon"]["enabled"]:
                await self._connect_polygon_ws()
            
            logger.info("WebSocket connections initialized")
            
        except Exception as e:
            logger.error(f"WebSocket initialization failed: {e}")
    
    async def _connect_polygon_ws(self):
        """Connect to Polygon WebSocket for real-time market data"""
        
        try:
            url = self.data_sources["market_data"]["polygon"]["url"]
            api_key = self.data_sources["market_data"]["polygon"]["api_key"]
            
            # For demo purposes, use mock WebSocket
            # In production, implement actual Polygon WebSocket connection
            
            self.ws_connections["polygon"] = await self._mock_websocket_connection("polygon")
            
        except Exception as e:
            logger.error(f"Polygon WebSocket connection failed: {e}")
    
    async def _mock_websocket_connection(self, source: str):
        """Mock WebSocket connection for demo purposes"""
        
        class MockWebSocket:
            def __init__(self):
                self.closed = False
                
            async def close(self):
                self.closed = True
        
        return MockWebSocket()
    
    async def _market_data_poller(self):
        """Background task to poll market data from REST APIs"""
        
        while True:
            try:
                # Get list of symbols to track
                symbols = await self._get_tracked_symbols()
                
                if symbols:
                    # Fetch data for each symbol
                    for symbol in symbols:
                        if await self._check_rate_limit("alpha_vantage"):
                            await self._fetch_alpha_vantage_data(symbol)
                            await asyncio.sleep(0.1)  # Small delay between requests
                
                await asyncio.sleep(self.update_frequency)
                
            except Exception as e:
                logger.error(f"Market data polling error: {e}")
                await asyncio.sleep(5)
    
    async def _news_fetcher(self):
        """Background task to fetch news from various sources"""
        
        while True:
            try:
                # Fetch from NewsAPI
                if (self.data_sources["news_sources"]["newsapi"]["enabled"] and 
                    await self._check_rate_limit("newsapi")):
                    await self._fetch_newsapi_data()
                
                # Fetch from Finnhub
                if (self.data_sources["news_sources"]["finnhub"]["enabled"] and
                    await self._check_rate_limit("finnhub")):
                    await self._fetch_finnhub_news()
                
                await asyncio.sleep(300)  # Fetch news every 5 minutes
                
            except Exception as e:
                logger.error(f"News fetching error: {e}")
                await asyncio.sleep(60)
    
    async def _options_data_fetcher(self):
        """Background task to fetch options data"""
        
        while True:
            try:
                if self.data_sources["options_data"]["tradier"]["enabled"]:
                    symbols = await self._get_tracked_symbols()
                    
                    for symbol in symbols:
                        await self._fetch_tradier_options(symbol)
                        await asyncio.sleep(1)  # Rate limiting
                
                await asyncio.sleep(900)  # Fetch options every 15 minutes
                
            except Exception as e:
                logger.error(f"Options data fetching error: {e}")
                await asyncio.sleep(300)
    
    async def _data_processor(self):
        """Background task to process and enrich data"""
        
        while True:
            try:
                # Process market data
                await self._process_market_data_buffer()
                
                # Process news data
                await self._process_news_buffer()
                
                # Process options data
                await self._process_options_buffer()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Data processing error: {e}")
                await asyncio.sleep(5)
    
    async def _buffer_manager(self):
        """Background task to manage data buffer sizes"""
        
        while True:
            try:
                # Trim market data buffers
                for symbol in self.market_data_buffer:
                    if len(self.market_data_buffer[symbol]) > self.buffer_size:
                        self.market_data_buffer[symbol] = self.market_data_buffer[symbol][-self.buffer_size:]
                
                # Trim news buffer
                if len(self.news_buffer) > self.buffer_size:
                    self.news_buffer = self.news_buffer[-self.buffer_size:]
                
                # Trim options buffers
                for symbol in self.options_buffer:
                    if len(self.options_buffer[symbol]) > 1000:  # Keep fewer options records
                        self.options_buffer[symbol] = self.options_buffer[symbol][-1000:]
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                logger.error(f"Buffer management error: {e}")
                await asyncio.sleep(10)
    
    async def _rate_limit_resetter(self):
        """Background task to reset rate limit counters"""
        
        while True:
            try:
                current_time = time.time()
                
                for source, limits in self.rate_limits.items():
                    if current_time - limits["reset_time"] >= 60:  # Reset every minute
                        limits["current"] = 0
                        limits["reset_time"] = current_time
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Rate limit resetter error: {e}")
                await asyncio.sleep(5)
    
    async def _get_tracked_symbols(self) -> List[str]:
        """Get list of symbols currently being tracked"""
        
        try:
            # Get from Redis or use default symbols
            tracked = await self.redis.smembers("tracked_symbols")
            
            if tracked:
                return list(tracked)
            else:
                # Default symbols for demo
                return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "SPY", "QQQ"]
                
        except Exception as e:
            logger.error(f"Failed to get tracked symbols: {e}")
            return ["SPY", "QQQ"]  # Minimal fallback
    
    async def _check_rate_limit(self, source: str) -> bool:
        """Check if API call is within rate limit"""
        
        try:
            if source in self.rate_limits:
                limits = self.rate_limits[source]
                
                if limits["current"] < limits["requests_per_minute"]:
                    limits["current"] += 1
                    return True
                else:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return False
    
    async def _fetch_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current data for a specific symbol"""
        
        try:
            if await self._check_rate_limit("alpha_vantage"):
                return await self._fetch_alpha_vantage_data(symbol)
            else:
                # Return mock data if rate limited
                return await self._generate_mock_market_data(symbol)
                
        except Exception as e:
            logger.error(f"Symbol data fetch failed for {symbol}: {e}")
            return None
    
    async def _fetch_alpha_vantage_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch data from Alpha Vantage API"""
        
        try:
            # For demo purposes, generate mock data
            # In production, implement actual Alpha Vantage API calls
            return await self._generate_mock_market_data(symbol)
            
        except Exception as e:
            logger.error(f"Alpha Vantage fetch failed for {symbol}: {e}")
            return None
    
    async def _generate_mock_market_data(self, symbol: str) -> Dict[str, Any]:
        """Generate mock market data for demo purposes"""
        
        import random
        
        base_price = {"AAPL": 150, "MSFT": 300, "GOOGL": 2500, "AMZN": 3000, "TSLA": 200, 
                     "NVDA": 400, "META": 250, "SPY": 400, "QQQ": 350}.get(symbol, 100)
        
        current_price = base_price + random.uniform(-5, 5)
        change = random.uniform(-2, 2)
        
        return {
            "price": round(current_price, 2),
            "volume": random.randint(1000000, 10000000),
            "bid": round(current_price - 0.01, 2),
            "ask": round(current_price + 0.01, 2),
            "change": round(change, 2),
            "change_percent": round((change / current_price) * 100, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "mock"
        }
    
    async def _fetch_newsapi_data(self):
        """Fetch news from NewsAPI"""
        
        try:
            # For demo purposes, generate mock news
            # In production, implement actual NewsAPI calls
            mock_news = [
                {
                    "title": "Tech Stocks Rally on Strong Earnings",
                    "content": "Major technology companies posted strong quarterly earnings...",
                    "source": "Financial Times",
                    "symbols": ["AAPL", "MSFT", "GOOGL"]
                },
                {
                    "title": "Federal Reserve Maintains Interest Rates",
                    "content": "The Federal Reserve decided to keep interest rates unchanged...",
                    "source": "Reuters",
                    "symbols": ["SPY", "QQQ"]
                }
            ]
            
            for news_data in mock_news:
                news_item = NewsItem(
                    id=str(uuid.uuid4()),
                    title=news_data["title"],
                    content=news_data["content"],
                    source=news_data["source"],
                    timestamp=datetime.utcnow(),
                    symbols_mentioned=news_data["symbols"],
                    relevance_score=0.8
                )
                
                self.news_buffer.append(news_item)
                
                # Call news handlers
                for handler in self.data_handlers.get("news", []):
                    try:
                        await handler(news_item)
                    except Exception as e:
                        logger.error(f"News handler error: {e}")
            
        except Exception as e:
            logger.error(f"NewsAPI fetch failed: {e}")
    
    async def _fetch_finnhub_news(self):
        """Fetch news from Finnhub"""
        
        try:
            # Mock implementation
            pass
            
        except Exception as e:
            logger.error(f"Finnhub news fetch failed: {e}")
    
    async def _fetch_tradier_options(self, symbol: str):
        """Fetch options data from Tradier"""
        
        try:
            # Mock options data generation
            import random
            from datetime import datetime, timedelta
            
            expiry_dates = [
                (datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d"),
                (datetime.utcnow() + timedelta(days=14)).strftime("%Y-%m-%d"),
                (datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%d")
            ]
            
            base_price = {"AAPL": 150, "MSFT": 300, "GOOGL": 2500}.get(symbol, 100)
            
            if symbol not in self.options_buffer:
                self.options_buffer[symbol] = []
            
            for expiry in expiry_dates[:1]:  # Just one expiry for demo
                for i in range(3):  # 3 strikes around current price
                    strike = base_price + (i - 1) * 5
                    
                    for option_type in ["call", "put"]:
                        option = OptionsData(
                            symbol=symbol,
                            strike=strike,
                            expiry=expiry,
                            option_type=option_type,
                            bid=random.uniform(1, 10),
                            ask=random.uniform(1.05, 10.5),
                            volume=random.randint(0, 1000),
                            open_interest=random.randint(0, 5000),
                            implied_volatility=random.uniform(0.15, 0.8),
                            delta=random.uniform(-1, 1),
                            gamma=random.uniform(0, 0.1),
                            theta=random.uniform(-0.5, 0),
                            vega=random.uniform(0, 1),
                            timestamp=datetime.utcnow()
                        )
                        
                        self.options_buffer[symbol].append(option)
            
        except Exception as e:
            logger.error(f"Tradier options fetch failed for {symbol}: {e}")
    
    async def _process_market_data_buffer(self):
        """Process and enrich market data in buffer"""
        
        try:
            for symbol, data_points in self.market_data_buffer.items():
                if len(data_points) >= 2:
                    # Calculate technical indicators
                    latest = data_points[-1]
                    previous = data_points[-2]
                    
                    # Simple price change calculation
                    latest.change = latest.price - previous.price
                    latest.change_percent = (latest.change / previous.price) * 100
                    
                    # Store processed data in Redis
                    await self._store_processed_data(symbol, latest)
                    
                    # Call market data handlers
                    for handler in self.data_handlers.get("market_data", []):
                        try:
                            await handler(latest)
                        except Exception as e:
                            logger.error(f"Market data handler error: {e}")
            
        except Exception as e:
            logger.error(f"Market data processing error: {e}")
    
    async def _process_news_buffer(self):
        """Process and enrich news data in buffer"""
        
        try:
            # Simple news processing - in production, apply NLP sentiment analysis
            for news_item in self.news_buffer[-10:]:  # Process last 10 items
                if news_item.sentiment_score is None:
                    # Mock sentiment analysis
                    news_item.sentiment_score = await self._analyze_news_sentiment(news_item)
            
        except Exception as e:
            logger.error(f"News processing error: {e}")
    
    async def _process_options_buffer(self):
        """Process and enrich options data in buffer"""
        
        try:
            # Calculate implied volatility surface and other derived metrics
            # For demo purposes, just call handlers
            for symbol, options in self.options_buffer.items():
                for option in options[-5:]:  # Process last 5 options per symbol
                    for handler in self.data_handlers.get("options", []):
                        try:
                            await handler(option)
                        except Exception as e:
                            logger.error(f"Options handler error: {e}")
            
        except Exception as e:
            logger.error(f"Options processing error: {e}")
    
    async def _analyze_news_sentiment(self, news_item: NewsItem) -> float:
        """Analyze sentiment of news item"""
        
        try:
            # Mock sentiment analysis
            positive_words = ["rally", "gains", "strong", "growth", "bullish", "positive"]
            negative_words = ["decline", "falls", "weak", "bearish", "negative", "concerns"]
            
            text = (news_item.title + " " + news_item.content).lower()
            
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            
            if positive_count > negative_count:
                return 0.6 + (positive_count - negative_count) * 0.1
            elif negative_count > positive_count:
                return 0.4 - (negative_count - positive_count) * 0.1
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return 0.5
    
    async def _get_market_context(self) -> Dict[str, Any]:
        """Get overall market context"""
        
        try:
            # Calculate market-wide metrics
            spy_data = self.market_data_buffer.get("SPY", [])
            qqq_data = self.market_data_buffer.get("QQQ", [])
            
            context = {
                "market_session": "regular",  # regular, pre, post
                "market_status": "open",
                "volatility_regime": "normal",
                "sector_rotation": {},
                "market_breadth": {}
            }
            
            if spy_data:
                latest_spy = spy_data[-1]
                context["spy_price"] = latest_spy.price
                context["spy_change"] = latest_spy.change_percent
            
            if qqq_data:
                latest_qqq = qqq_data[-1]
                context["qqq_price"] = latest_qqq.price
                context["qqq_change"] = latest_qqq.change_percent
            
            return context
            
        except Exception as e:
            logger.error(f"Market context calculation failed: {e}")
            return {}
    
    async def _store_processed_data(self, symbol: str, data_point: MarketDataPoint):
        """Store processed data in Redis"""
        
        try:
            # Store latest data point
            data_dict = asdict(data_point)
            data_dict["timestamp"] = data_point.timestamp.isoformat()
            
            await self.redis.hset(
                f"market_data:{symbol}",
                "latest",
                json.dumps(data_dict)
            )
            
            # Store time series data
            await self.redis.zadd(
                f"market_data:{symbol}:timeseries",
                {json.dumps(data_dict): data_point.timestamp.timestamp()}
            )
            
            # Trim time series to last 24 hours
            cutoff = (datetime.utcnow() - timedelta(hours=24)).timestamp()
            await self.redis.zremrangebyscore(
                f"market_data:{symbol}:timeseries",
                0,
                cutoff
            )
            
        except Exception as e:
            logger.error(f"Data storage failed: {e}")

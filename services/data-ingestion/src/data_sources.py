"""
Enhanced data sources with options intelligence
Includes CBOE, ORATS, and other enhanced data collectors
"""

import asyncio
import aiohttp
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import boto3
from botocore.exceptions import ClientError

from config import settings

logger = logging.getLogger(__name__)

class BaseDataCollector(ABC):
    """Base class for all data collectors with enhanced error handling"""
    
    def __init__(self, name: str):
        self.name = name
        self.session = None
        self.secrets_client = boto3.client('secretsmanager', region_name=settings.AWS_REGION)
        self.credentials = None
        self.last_collection_time = None
        self.collection_count = 0
        
    async def initialize(self):
        """Initialize the data collector"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.REQUEST_TIMEOUT),
            headers={'User-Agent': f'MIP-DataCollector/{self.name}/1.0'}
        )
        await self._load_credentials()
    
    async def _load_credentials(self):
        """Load API credentials from AWS Secrets Manager"""
        try:
            secret_name = getattr(settings, f"{self.name.upper()}_SECRET_NAME", None)
            if not secret_name:
                logger.warning(f"No secret name configured for {self.name}")
                return
            
            response = self.secrets_client.get_secret_value(SecretId=secret_name)
            self.credentials = json.loads(response['SecretString'])
            logger.info(f"✅ Loaded credentials for {self.name}")
            
        except ClientError as e:
            logger.error(f"❌ Failed to load credentials for {self.name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Credential loading error for {self.name}: {str(e)}")
            raise
    
    @abstractmethod
    async def collect_data(self) -> List[Dict]:
        """Collect data from the source"""
        pass
    
    async def close(self):
        """Close the data collector and cleanup resources"""
        if self.session:
            await self.session.close()
        logger.info(f"🔌 Closed {self.name} data collector")

class CBOEOptionsCollector(BaseDataCollector):
    """CBOE Options data collector with enhanced options flow analysis"""
    
    def __init__(self):
        super().__init__("cboe")
        self.base_url = "https://www.cboe.com/us/options/market_statistics"
        self.supported_symbols = settings.SUPPORTED_OPTIONS_SYMBOLS
        
    async def collect_data(self) -> List[Dict]:
        """Collect options flow data from CBOE"""
        logger.debug(f"🔄 Collecting CBOE options data for {len(self.supported_symbols)} symbols...")
        
        if not self.session:
            await self.initialize()
        
        options_data = []
        
        try:
            # Collect data for each supported symbol
            tasks = []
            for symbol in self.supported_symbols:
                task = self._collect_symbol_options(symbol)
                tasks.append(task)
            
            # Execute all collections concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"❌ CBOE collection failed for {self.supported_symbols[i]}: {str(result)}")
                    continue
                
                if result:
                    options_data.extend(result)
            
            self.last_collection_time = datetime.utcnow()
            self.collection_count += 1
            
            logger.info(f"✅ CBOE: Collected {len(options_data)} options records")
            return options_data
            
        except Exception as e:
            logger.error(f"❌ CBOE collection failed: {str(e)}")
            raise

    async def _collect_symbol_options(self, symbol: str) -> List[Dict]:
        """Collect options data for a specific symbol"""
        try:
            # CBOE API endpoint for options data
            url = f"{self.base_url}/csv/options_volume/{symbol.lower()}.csv"
            
            headers = {
                'Accept': 'text/csv',
                'Authorization': f"Bearer {self.credentials.get('api_key')}" if self.credentials else None
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    csv_content = await response.text()
                    return self._parse_cboe_csv(symbol, csv_content)
                else:
                    logger.warning(f"⚠️  CBOE API returned status {response.status} for {symbol}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ CBOE symbol collection failed for {symbol}: {str(e)}")
            return []

    def _parse_cboe_csv(self, symbol: str, csv_content: str) -> List[Dict]:
        """Parse CBOE CSV data into structured format"""
        try:
            import csv
            import io
            
            options_records = []
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            
            for row in csv_reader:
                try:
                    # Parse CBOE CSV format
                    option_record = {
                        'symbol': symbol,
                        'option_symbol': row.get('symbol', ''),
                        'expiry': self._parse_expiry(row.get('expiry', '')),
                        'strike': float(row.get('strike', 0)),
                        'option_type': 'call' if row.get('call_put', '').lower() == 'c' else 'put',
                        'volume': int(row.get('volume', 0)),
                        'open_interest': int(row.get('open_interest', 0)),
                        'implied_volatility': float(row.get('iv', 0)) / 100,  # Convert percentage
                        'delta': float(row.get('delta', 0)),
                        'gamma': float(row.get('gamma', 0)),
                        'theta': float(row.get('theta', 0)),
                        'vega': float(row.get('vega', 0)),
                        'bid': float(row.get('bid', 0)) if row.get('bid') else None,
                        'ask': float(row.get('ask', 0)) if row.get('ask') else None,
                        'last_price': float(row.get('last', 0)) if row.get('last') else None,
                        'timestamp': datetime.utcnow(),
                        'source': 'cboe'
                    }
                    
                    # Only include options with minimum volume
                    if option_record['volume'] >= settings.MIN_OPTIONS_VOLUME:
                        options_records.append(option_record)
                
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping malformed CBOE record: {str(e)}")
                    continue
            
            return options_records
            
        except Exception as e:
            logger.error(f"❌ CBOE CSV parsing failed: {str(e)}")
            return []

    def _parse_expiry(self, expiry_str: str) -> str:
        """Parse CBOE expiry format to ISO date"""
        try:
            # Handle various CBOE date formats
            if len(expiry_str) == 6:  # YYMMDD
                year = 2000 + int(expiry_str[:2])
                month = int(expiry_str[2:4])
                day = int(expiry_str[4:6])
                return datetime(year, month, day).date().isoformat()
            else:
                # Try ISO format
                return datetime.fromisoformat(expiry_str).date().isoformat()
        except Exception:
            # Default to next Friday if parsing fails
            today = datetime.now().date()
            days_ahead = 4 - today.weekday()  # Friday is 4
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).isoformat()

class ORATSCollector(BaseDataCollector):
    """ORATS historical volatility and options analytics collector"""
    
    def __init__(self):
        super().__init__("orats")
        self.base_url = "https://api.orats.com/datav2"
        self.supported_symbols = settings.SUPPORTED_OPTIONS_SYMBOLS

    async def collect_data(self) -> List[Dict]:
        """Collect options analytics from ORATS"""
        logger.debug(f"🔄 Collecting ORATS analytics for {len(self.supported_symbols)} symbols...")
        
        if not self.session:
            await self.initialize()
        
        analytics_data = []
        
        try:
            # Collect volatility surfaces and analytics
            for symbol in self.supported_symbols:
                symbol_data = await self._collect_symbol_analytics(symbol)
                if symbol_data:
                    analytics_data.extend(symbol_data)
            
            self.last_collection_time = datetime.utcnow()
            self.collection_count += 1
            
            logger.info(f"✅ ORATS: Collected {len(analytics_data)} analytics records")
            return analytics_data
            
        except Exception as e:
            logger.error(f"❌ ORATS collection failed: {str(e)}")
            raise

    async def _collect_symbol_analytics(self, symbol: str) -> List[Dict]:
        """Collect ORATS analytics for a specific symbol"""
        try:
            # ORATS volatility surface endpoint
            url = f"{self.base_url}/volsurface"
            
            params = {
                'token': self.credentials.get('api_token') if self.credentials else '',
                'ticker': symbol,
                'fields': 'strike,expiry,dte,delta,gamma,theta,vega,iv,volOfVol,rho'
            }
            
            headers = {
                'Accept': 'application/json',
                'Authorization': f"Bearer {self.credentials.get('api_key')}" if self.credentials else None
            }
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_orats_response(symbol, data)
                else:
                    logger.warning(f"⚠️  ORATS API returned status {response.status} for {symbol}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ ORATS collection failed for {symbol}: {str(e)}")
            return []

    def _parse_orats_response(self, symbol: str, data: Dict) -> List[Dict]:
        """Parse ORATS API response"""
        try:
            analytics_records = []
            
            for record in data.get('data', []):
                try:
                    analytics_record = {
                        'symbol': symbol,
                        'strike': float(record.get('strike', 0)),
                        'expiry': record.get('expiry', ''),
                        'days_to_expiry': int(record.get('dte', 0)),
                        'delta': float(record.get('delta', 0)),
                        'gamma': float(record.get('gamma', 0)),
                        'theta': float(record.get('theta', 0)),
                        'vega': float(record.get('vega', 0)),
                        'implied_volatility': float(record.get('iv', 0)),
                        'vol_of_vol': float(record.get('volOfVol', 0)),
                        'rho': float(record.get('rho', 0)),
                        'timestamp': datetime.utcnow(),
                        'source': 'orats'
                    }
                    
                    analytics_records.append(analytics_record)
                
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping malformed ORATS record: {str(e)}")
                    continue
            
            return analytics_records
            
        except Exception as e:
            logger.error(f"❌ ORATS response parsing failed: {str(e)}")
            return []

class DarkPoolCollector(BaseDataCollector):
    """Dark pool and institutional options flow collector"""
    
    def __init__(self):
        super().__init__("darkpool")
        self.base_url = "https://api.darkpooldata.com/v1"  # Hypothetical endpoint
        self.supported_symbols = settings.SUPPORTED_OPTIONS_SYMBOLS

    async def collect_data(self) -> List[Dict]:
        """Collect dark pool options flow data"""
        logger.debug(f"🔄 Collecting dark pool data for {len(self.supported_symbols)} symbols...")
        
        if not self.session:
            await self.initialize()
        
        darkpool_data = []
        
        try:
            # Note: Dark pool data is typically more restricted
            # This is a simplified implementation
            
            for symbol in self.supported_symbols:
                symbol_data = await self._collect_symbol_darkpool(symbol)
                if symbol_data:
                    darkpool_data.extend(symbol_data)
            
            self.last_collection_time = datetime.utcnow()
            self.collection_count += 1
            
            logger.info(f"✅ Dark Pool: Collected {len(darkpool_data)} flow records")
            return darkpool_data
            
        except Exception as e:
            logger.error(f"❌ Dark pool collection failed: {str(e)}")
            raise

    async def _collect_symbol_darkpool(self, symbol: str) -> List[Dict]:
        """Collect dark pool data for a specific symbol"""
        try:
            url = f"{self.base_url}/options/flow"
            
            params = {
                'symbol': symbol,
                'min_premium': 10000,  # $10k+ trades only
                'timeframe': '1h'
            }
            
            headers = {
                'X-API-Key': self.credentials.get('api_key') if self.credentials else '',
                'Accept': 'application/json'
            }
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_darkpool_response(symbol, data)
                else:
                    logger.warning(f"⚠️  Dark pool API returned status {response.status} for {symbol}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Dark pool collection failed for {symbol}: {str(e)}")
            return []

    def _parse_darkpool_response(self, symbol: str, data: Dict) -> List[Dict]:
        """Parse dark pool API response"""
        try:
            flow_records = []
            
            for trade in data.get('trades', []):
                try:
                    flow_record = {
                        'symbol': symbol,
                        'trade_id': trade.get('id'),
                        'strike': float(trade.get('strike', 0)),
                        'expiry': trade.get('expiry', ''),
                        'option_type': trade.get('type', '').lower(),
                        'size': int(trade.get('size', 0)),
                        'premium': float(trade.get('premium', 0)),
                        'spot_price': float(trade.get('spot_price', 0)),
                        'trade_type': trade.get('trade_type', ''),  # buy/sell
                        'venue': trade.get('venue', 'unknown'),
                        'timestamp': datetime.fromisoformat(trade.get('timestamp', '')),
                        'source': 'darkpool'
                    }
                    
                    # Only include significant trades
                    if flow_record['premium'] >= 10000:  # $10k+ premium
                        flow_records.append(flow_record)
                
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping malformed dark pool record: {str(e)}")
                    continue
            
            return flow_records
            
        except Exception as e:
            logger.error(f"❌ Dark pool response parsing failed: {str(e)}")
            return []

class AlphaVantageCollector(BaseDataCollector):
    """Enhanced AlphaVantage collector with options support"""
    
    def __init__(self):
        super().__init__("alphavantage")
        self.base_url = "https://www.alphavantage.co/query"
        self.supported_symbols = settings.SUPPORTED_OPTIONS_SYMBOLS

    async def collect_data(self) -> List[Dict]:
        """Collect market data from AlphaVantage"""
        logger.debug(f"🔄 Collecting AlphaVantage data for {len(self.supported_symbols)} symbols...")
        
        if not self.session:
            await self.initialize()
        
        market_data = []
        
        try:
            # Collect intraday data for each symbol
            for symbol in self.supported_symbols:
                symbol_data = await self._collect_symbol_data(symbol)
                if symbol_data:
                    market_data.extend(symbol_data)
                
                # Rate limiting - AlphaVantage has strict limits
                await asyncio.sleep(0.2)  # 5 calls per second max
            
            self.last_collection_time = datetime.utcnow()
            self.collection_count += 1
            
            logger.info(f"✅ AlphaVantage: Collected {len(market_data)} market records")
            return market_data
            
        except Exception as e:
            logger.error(f"❌ AlphaVantage collection failed: {str(e)}")
            raise

    async def _collect_symbol_data(self, symbol: str) -> List[Dict]:
        """Collect market data for a specific symbol"""
        try:
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': '1min',
                'apikey': self.credentials.get('api_key') if self.credentials else '',
                'outputsize': 'compact'
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_alphavantage_response(symbol, data)
                else:
                    logger.warning(f"⚠️  AlphaVantage API returned status {response.status} for {symbol}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ AlphaVantage collection failed for {symbol}: {str(e)}")
            return []

    def _parse_alphavantage_response(self, symbol: str, data: Dict) -> List[Dict]:
        """Parse AlphaVantage API response"""
        try:
            market_records = []
            time_series = data.get('Time Series (1min)', {})
            
            for timestamp, ohlcv in time_series.items():
                try:
                    market_record = {
                        'symbol': symbol,
                        'timestamp': datetime.fromisoformat(timestamp),
                        'open': float(ohlcv.get('1. open', 0)),
                        'high': float(ohlcv.get('2. high', 0)),
                        'low': float(ohlcv.get('3. low', 0)),
                        'close': float(ohlcv.get('4. close', 0)),
                        'volume': int(ohlcv.get('5. volume', 0)),
                        'source': 'alphavantage'
                    }
                    
                    market_records.append(market_record)
                
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping malformed AlphaVantage record: {str(e)}")
                    continue
            
            return market_records
            
        except Exception as e:
            logger.error(f"❌ AlphaVantage response parsing failed: {str(e)}")
            return []

class NewsAPICollector(BaseDataCollector):
    """Enhanced NewsAPI collector with financial focus"""
    
    def __init__(self):
        super().__init__("newsapi")
        self.base_url = "https://newsapi.org/v2"

    async def collect_data(self) -> List[Dict]:
        """Collect financial news from NewsAPI"""
        logger.debug("🔄 Collecting financial news from NewsAPI...")
        
        if not self.session:
            await self.initialize()
        
        try:
            # Query for financial and market news
            queries = [
                "stock market options trading",
                "financial markets volatility", 
                "earnings reports",
                "Federal Reserve interest rates"
            ]
            
            news_data = []
            for query in queries:
                query_data = await self._collect_news_query(query)
                if query_data:
                    news_data.extend(query_data)
            
            self.last_collection_time = datetime.utcnow()
            self.collection_count += 1
            
            logger.info(f"✅ NewsAPI: Collected {len(news_data)} news articles")
            return news_data
            
        except Exception as e:
            logger.error(f"❌ NewsAPI collection failed: {str(e)}")
            raise

    async def _collect_news_query(self, query: str) -> List[Dict]:
        """Collect news for a specific query"""
        try:
            url = f"{self.base_url}/everything"
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': self.credentials.get('api_key') if self.credentials else '',
                'pageSize': 50
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_newsapi_response(data)
                else:
                    logger.warning(f"⚠️  NewsAPI returned status {response.status} for query: {query}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ NewsAPI query failed for '{query}': {str(e)}")
            return []

    def _parse_newsapi_response(self, data: Dict) -> List[Dict]:
        """Parse NewsAPI response"""
        try:
            news_records = []
            
            for article in data.get('articles', []):
                try:
                    news_record = {
                        'title': article.get('title', ''),
                        'content': article.get('content', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'published_at': datetime.fromisoformat(
                            article.get('publishedAt', '').replace('Z', '+00:00')
                        ),
                        'url': article.get('url', ''),
                        'author': article.get('author'),
                        'description': article.get('description', ''),
                        'source_type': 'news',
                        'source_name': 'newsapi'
                    }
                    
                    # Filter out articles without content
                    if len(news_record['content']) > 50:
                        news_records.append(news_record)
                
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping malformed NewsAPI record: {str(e)}")
                    continue
            
            return news_records
            
        except Exception as e:
            logger.error(f"❌ NewsAPI response parsing failed: {str(e)}")
            return []

class TwitterAPICollector(BaseDataCollector):
    """Enhanced Twitter API collector for financial sentiment"""
    
    def __init__(self):
        super().__init__("twitter")
        self.base_url = "https://api.twitter.com/2"

    async def collect_data(self) -> List[Dict]:
        """Collect financial tweets"""
        logger.debug("🔄 Collecting financial tweets...")
        
        if not self.session:
            await self.initialize()
        
        try:
            # Search for financial tweets
            financial_hashtags = [
                "#StockMarket", "#Options", "#Volatility", 
                "#Earnings", "#Fed", "#Trading"
            ]
            
            tweets_data = []
            for hashtag in financial_hashtags:
                hashtag_data = await self._collect_tweets_hashtag(hashtag)
                if hashtag_data:
                    tweets_data.extend(hashtag_data)
            
            self.last_collection_time = datetime.utcnow()
            self.collection_count += 1
            
            logger.info(f"✅ Twitter: Collected {len(tweets_data)} tweets")
            return tweets_data
            
        except Exception as e:
            logger.error(f"❌ Twitter collection failed: {str(e)}")
            raise

    async def _collect_tweets_hashtag(self, hashtag: str) -> List[Dict]:
        """Collect tweets for a specific hashtag"""
        try:
            url = f"{self.base_url}/tweets/search/recent"
            
            headers = {
                'Authorization': f"Bearer {self.credentials.get('bearer_token')}" if self.credentials else '',
                'Accept': 'application/json'
            }
            
            params = {
                'query': f"{hashtag} -is:retweet lang:en",
                'max_results': 100,
                'tweet.fields': 'created_at,author_id,public_metrics'
            }
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_twitter_response(data)
                else:
                    logger.warning(f"⚠️  Twitter API returned status {response.status} for hashtag: {hashtag}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Twitter collection failed for '{hashtag}': {str(e)}")
            return []

    def _parse_twitter_response(self, data: Dict) -> List[Dict]:
        """Parse Twitter API response"""
        try:
            tweet_records = []
            
            for tweet in data.get('data', []):
                try:
                    tweet_record = {
                        'title': tweet.get('text', '')[:100] + '...',  # Truncated title
                        'content': tweet.get('text', ''),
                        'source': 'Twitter',
                        'published_at': datetime.fromisoformat(
                            tweet.get('created_at', '').replace('Z', '+00:00')
                        ),
                        'url': f"https://twitter.com/i/web/status/{tweet.get('id', '')}",
                        'author': tweet.get('author_id', ''),
                        'engagement_metrics': tweet.get('public_metrics', {}),
                        'source_type': 'social',
                        'source_name': 'twitter'
                    }
                    
                    # Filter out short tweets
                    if len(tweet_record['content']) > 20:
                        tweet_records.append(tweet_record)
                
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping malformed Twitter record: {str(e)}")
                    continue
            
            return tweet_records
            
        except Exception as e:
            logger.error(f"❌ Twitter response parsing failed: {str(e)}")
            return []

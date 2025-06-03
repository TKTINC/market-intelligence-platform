"""
Enhanced Sentiment Analysis Service with Options Intelligence
Market Intelligence Platform - Sprint 2, Prompt 4

Features:
- Fine-tuned FinBERT with options-specific preprocessing
- Options entity extraction and sentiment weighting
- Real-time sentiment scoring with options bias detection
- Multi-source sentiment aggregation (news, social media, earnings calls)
- Sentiment-driven options strategy recommendations
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import uvloop

from config import settings
from models.enhanced_finbert import EnhancedFinBERTModel
from models.options_sentiment import OptionsSentimentAnalyzer
from processors.text_processor import EnhancedTextProcessor
from processors.entity_extractor import OptionsEntityExtractor
from kafka_consumer import EnhancedKafkaConsumer
from kafka_producer import SentimentKafkaProducer
from monitoring import SentimentMetricsCollector
from database import SentimentDatabase

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/mip/sentiment-analysis.log')
    ]
)
logger = logging.getLogger(__name__)

class EnhancedSentimentAnalysisService:
    """Enhanced sentiment analysis service with options intelligence"""
    
    def __init__(self):
        """Initialize the enhanced sentiment analysis service"""
        logger.info("🚀 Initializing Enhanced Sentiment Analysis Service...")
        
        # Initialize core components
        self.finbert_model = EnhancedFinBERTModel()
        self.options_analyzer = OptionsSentimentAnalyzer()
        self.text_processor = EnhancedTextProcessor()
        self.entity_extractor = OptionsEntityExtractor()
        
        # Initialize I/O components
        self.kafka_consumer = EnhancedKafkaConsumer()
        self.kafka_producer = SentimentKafkaProducer()
        self.database = SentimentDatabase()
        self.metrics = SentimentMetricsCollector()
        
        # Processing statistics
        self.stats = {
            'messages_processed': 0,
            'sentiment_analyses_completed': 0,
            'options_entities_extracted': 0,
            'sentiment_alerts_generated': 0,
            'processing_errors': 0,
            'average_processing_time': 0.0
        }
        
        # Sentiment cache for real-time queries
        self.sentiment_cache = {}
        self.cache_expiry = timedelta(minutes=15)
        
        # Shutdown flag
        self.shutdown_event = asyncio.Event()
        
        logger.info("✅ Enhanced Sentiment Analysis Service initialized")

    async def start(self):
        """Start the enhanced sentiment analysis service"""
        logger.info("▶️  Starting Enhanced Sentiment Analysis Service...")
        
        try:
            # Load models
            await self.finbert_model.load_model()
            await self.options_analyzer.load_model()
            
            # Initialize database connection
            await self.database.connect()
            
            # Start Kafka components
            await self.kafka_consumer.start()
            await self.kafka_producer.start()
            
            # Start metrics collection
            await self.metrics.start()
            
            # Create processing tasks
            tasks = []
            
            # News sentiment processing
            news_task = asyncio.create_task(
                self._process_news_sentiment(),
                name="news_sentiment_processing"
            )
            tasks.append(news_task)
            
            # Social media sentiment processing
            social_task = asyncio.create_task(
                self._process_social_sentiment(),
                name="social_sentiment_processing"
            )
            tasks.append(social_task)
            
            # Options-specific sentiment processing
            options_task = asyncio.create_task(
                self._process_options_sentiment(),
                name="options_sentiment_processing"
            )
            tasks.append(options_task)
            
            # Sentiment aggregation task
            aggregation_task = asyncio.create_task(
                self._aggregate_sentiment_signals(),
                name="sentiment_aggregation"
            )
            tasks.append(aggregation_task)
            
            # Cache maintenance task
            cache_task = asyncio.create_task(
                self._maintain_sentiment_cache(),
                name="cache_maintenance"
            )
            tasks.append(cache_task)
            
            # Metrics reporting task
            metrics_task = asyncio.create_task(
                self._metrics_reporting_loop(),
                name="metrics_reporting"
            )
            tasks.append(metrics_task)
            
            logger.info(f"🔄 Started {len(tasks)} sentiment processing tasks")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            logger.info("🛑 Shutdown signal received, stopping tasks...")
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"❌ Service startup failed: {str(e)}")
            raise
        finally:
            await self.cleanup()

    async def _process_news_sentiment(self):
        """Process news sentiment with options intelligence"""
        logger.info("🔄 Starting news sentiment processing loop")
        
        async for message in self.kafka_consumer.consume('raw-news'):
            try:
                processing_start = datetime.utcnow()
                self.stats['messages_processed'] += 1
                
                # Parse news message
                news_data = message.value
                
                # Enhanced text preprocessing for financial news
                processed_text = await self.text_processor.preprocess_news_content(
                    title=news_data.get('title', ''),
                    content=news_data.get('content', ''),
                    source=news_data.get('source', '')
                )
                
                if not processed_text:
                    continue
                
                # Extract options-related entities
                options_entities = await self.entity_extractor.extract_options_entities(
                    processed_text
                )
                
                # Analyze sentiment with FinBERT
                base_sentiment = await self.finbert_model.analyze_sentiment(
                    processed_text
                )
                
                # Apply options-specific sentiment analysis
                options_sentiment = await self.options_analyzer.analyze_options_sentiment(
                    text=processed_text,
                    entities=options_entities,
                    base_sentiment=base_sentiment
                )
                
                # Create comprehensive sentiment result
                sentiment_result = {
                    'source_id': news_data.get('_metadata', {}).get('ingestion_id'),
                    'source_type': 'news',
                    'content_type': 'article',
                    'timestamp': datetime.utcnow().isoformat(),
                    'processed_text': processed_text[:500],  # Truncated for storage
                    
                    # Base sentiment
                    'sentiment_score': base_sentiment['sentiment_score'],
                    'sentiment_label': base_sentiment['sentiment_label'],
                    'confidence': base_sentiment['confidence'],
                    
                    # Options-specific sentiment
                    'options_bias': options_sentiment['options_bias'],
                    'volatility_expectation': options_sentiment['volatility_expectation'],
                    'options_entities': options_entities,
                    
                    # Extracted symbols and relevance
                    'symbols_mentioned': options_entities.get('symbols', []),
                    'options_relevance_score': options_sentiment.get('relevance_score', 0.0),
                    
                    # Source metadata
                    'source': news_data.get('source', ''),
                    'published_at': news_data.get('published_at'),
                    'url': news_data.get('url', ''),
                    
                    # Processing metadata
                    'processing_time': (datetime.utcnow() - processing_start).total_seconds()
                }
                
                # Update statistics
                self.stats['sentiment_analyses_completed'] += 1
                if options_entities:
                    self.stats['options_entities_extracted'] += 1
                
                # Store in database
                await self.database.store_sentiment_analysis(sentiment_result)
                
                # Update cache for mentioned symbols
                await self._update_sentiment_cache(sentiment_result)
                
                # Publish to downstream services
                await self.kafka_producer.publish_sentiment(
                    topic='sentiment-analysis',
                    sentiment_data=sentiment_result
                )
                
                # Generate alerts for significant sentiment changes
                await self._check_sentiment_alerts(sentiment_result)
                
                # Update metrics
                processing_time = sentiment_result['processing_time']
                await self.metrics.record_sentiment_analysis(
                    source_type='news',
                    processing_time=processing_time,
                    symbols_count=len(sentiment_result['symbols_mentioned']),
                    options_relevance=sentiment_result['options_relevance_score']
                )
                
                logger.debug(f"✅ Processed news sentiment: {sentiment_result['sentiment_label']} "
                           f"({sentiment_result['sentiment_score']:.3f})")
                
            except Exception as e:
                self.stats['processing_errors'] += 1
                logger.error(f"❌ News sentiment processing failed: {str(e)}")
                
                await self.metrics.record_processing_error('news', str(e))
                continue
        
        logger.info("🛑 News sentiment processing loop stopped")

    async def _process_social_sentiment(self):
        """Process social media sentiment with enhanced filtering"""
        logger.info("🔄 Starting social media sentiment processing loop")
        
        async for message in self.kafka_consumer.consume('raw-social'):
            try:
                processing_start = datetime.utcnow()
                self.stats['messages_processed'] += 1
                
                # Parse social media message (Twitter, Reddit, etc.)
                social_data = message.value
                
                # Enhanced preprocessing for social media content
                processed_text = await self.text_processor.preprocess_social_content(
                    text=social_data.get('content', ''),
                    platform=social_data.get('source_name', ''),
                    engagement_metrics=social_data.get('engagement_metrics', {})
                )
                
                if not processed_text:
                    continue
                
                # Filter for financial relevance
                financial_relevance = await self.text_processor.calculate_financial_relevance(
                    processed_text
                )
                
                if financial_relevance < 0.3:  # Skip non-financial content
                    continue
                
                # Extract options entities
                options_entities = await self.entity_extractor.extract_options_entities(
                    processed_text
                )
                
                # Analyze sentiment
                base_sentiment = await self.finbert_model.analyze_sentiment(
                    processed_text
                )
                
                # Apply social media sentiment adjustments
                social_sentiment = await self.options_analyzer.analyze_social_sentiment(
                    text=processed_text,
                    entities=options_entities,
                    base_sentiment=base_sentiment,
                    engagement_metrics=social_data.get('engagement_metrics', {})
                )
                
                # Create sentiment result
                sentiment_result = {
                    'source_id': social_data.get('_metadata', {}).get('ingestion_id'),
                    'source_type': 'social',
                    'content_type': 'post',
                    'platform': social_data.get('source_name', ''),
                    'timestamp': datetime.utcnow().isoformat(),
                    'processed_text': processed_text[:200],  # Shorter for social media
                    
                    # Sentiment scores
                    'sentiment_score': social_sentiment['adjusted_sentiment_score'],
                    'sentiment_label': social_sentiment['sentiment_label'],
                    'confidence': social_sentiment['confidence'],
                    'social_amplification': social_sentiment.get('amplification_factor', 1.0),
                    
                    # Options analysis
                    'options_bias': social_sentiment['options_bias'],
                    'volatility_expectation': social_sentiment['volatility_expectation'],
                    'options_entities': options_entities,
                    
                    # Relevance and engagement
                    'financial_relevance': financial_relevance,
                    'engagement_score': social_sentiment.get('engagement_score', 0.0),
                    'viral_potential': social_sentiment.get('viral_potential', 0.0),
                    
                    # Symbols and metadata
                    'symbols_mentioned': options_entities.get('symbols', []),
                    'options_relevance_score': social_sentiment.get('relevance_score', 0.0),
                    'author': social_data.get('author', ''),
                    'published_at': social_data.get('published_at'),
                    
                    'processing_time': (datetime.utcnow() - processing_start).total_seconds()
                }
                
                # Update statistics
                self.stats['sentiment_analyses_completed'] += 1
                if options_entities:
                    self.stats['options_entities_extracted'] += 1
                
                # Store and publish
                await self.database.store_sentiment_analysis(sentiment_result)
                await self._update_sentiment_cache(sentiment_result)
                
                await self.kafka_producer.publish_sentiment(
                    topic='sentiment-analysis',
                    sentiment_data=sentiment_result
                )
                
                # Check for viral content alerts
                if sentiment_result['viral_potential'] > 0.8:
                    await self._generate_viral_content_alert(sentiment_result)
                
                # Update metrics
                await self.metrics.record_sentiment_analysis(
                    source_type='social',
                    processing_time=sentiment_result['processing_time'],
                    symbols_count=len(sentiment_result['symbols_mentioned']),
                    engagement_score=sentiment_result['engagement_score']
                )
                
                logger.debug(f"✅ Processed social sentiment: {sentiment_result['sentiment_label']} "
                           f"(viral: {sentiment_result['viral_potential']:.2f})")
                
            except Exception as e:
                self.stats['processing_errors'] += 1
                logger.error(f"❌ Social sentiment processing failed: {str(e)}")
                continue
        
        logger.info("🛑 Social sentiment processing loop stopped")

    async def _process_options_sentiment(self):
        """Process options-specific sentiment from financial forums and analysis"""
        logger.info("🔄 Starting options-specific sentiment processing loop")
        
        async for message in self.kafka_consumer.consume('raw-options-chatter'):
            try:
                processing_start = datetime.utcnow()
                self.stats['messages_processed'] += 1
                
                # Parse options-specific content
                options_data = message.value
                
                # Enhanced preprocessing for options discussions
                processed_text = await self.text_processor.preprocess_options_content(
                    text=options_data.get('content', ''),
                    title=options_data.get('title', ''),
                    context=options_data.get('context', {})
                )
                
                if not processed_text:
                    continue
                
                # Deep options entity extraction
                options_entities = await self.entity_extractor.extract_detailed_options_entities(
                    processed_text
                )
                
                # Specialized options sentiment analysis
                options_sentiment = await self.options_analyzer.analyze_comprehensive_options_sentiment(
                    text=processed_text,
                    entities=options_entities,
                    market_context=options_data.get('market_context', {})
                )
                
                # Create detailed sentiment result
                sentiment_result = {
                    'source_id': options_data.get('_metadata', {}).get('ingestion_id'),
                    'source_type': 'options_forum',
                    'content_type': 'options_analysis',
                    'timestamp': datetime.utcnow().isoformat(),
                    'processed_text': processed_text[:1000],  # Longer for options analysis
                    
                    # Options-specific sentiment
                    'options_sentiment_score': options_sentiment['options_sentiment_score'],
                    'volatility_sentiment': options_sentiment['volatility_sentiment'],
                    'direction_bias': options_sentiment['direction_bias'],  # bullish/bearish
                    'strategy_sentiment': options_sentiment['strategy_sentiment'],
                    
                    # Market regime analysis
                    'implied_volatility_sentiment': options_sentiment['iv_sentiment'],
                    'time_decay_sentiment': options_sentiment['theta_sentiment'],
                    'earnings_proximity_bias': options_sentiment.get('earnings_bias', 0.0),
                    
                    # Extracted options intelligence
                    'mentioned_strategies': options_entities.get('strategies', []),
                    'mentioned_strikes': options_entities.get('strikes', []),
                    'mentioned_expirations': options_entities.get('expirations', []),
                    'volatility_expectations': options_entities.get('volatility_expectations', {}),
                    
                    # Symbol-specific analysis
                    'symbols_mentioned': options_entities.get('symbols', []),
                    'symbol_sentiment_breakdown': options_sentiment.get('symbol_breakdown', {}),
                    
                    # Confidence and reliability
                    'analysis_confidence': options_sentiment['confidence'],
                    'author_credibility': options_sentiment.get('author_credibility', 0.5),
                    'source_reliability': options_sentiment.get('source_reliability', 0.5),
                    
                    # Metadata
                    'source': options_data.get('source', ''),
                    'author': options_data.get('author', ''),
                    'published_at': options_data.get('published_at'),
                    
                    'processing_time': (datetime.utcnow() - processing_start).total_seconds()
                }
                
                # Update statistics
                self.stats['sentiment_analyses_completed'] += 1
                self.stats['options_entities_extracted'] += len(options_entities.get('strategies', []))
                
                # Store and publish
                await self.database.store_sentiment_analysis(sentiment_result)
                await self._update_sentiment_cache(sentiment_result)
                
                await self.kafka_producer.publish_sentiment(
                    topic='options-sentiment-analysis',
                    sentiment_data=sentiment_result
                )
                
                # Generate strategy-specific alerts
                await self._check_options_strategy_alerts(sentiment_result)
                
                # Update metrics
                await self.metrics.record_sentiment_analysis(
                    source_type='options_forum',
                    processing_time=sentiment_result['processing_time'],
                    symbols_count=len(sentiment_result['symbols_mentioned']),
                    strategies_count=len(sentiment_result['mentioned_strategies'])
                )
                
                logger.debug(f"✅ Processed options sentiment: {sentiment_result['direction_bias']} "
                           f"(vol: {sentiment_result['volatility_sentiment']:.2f})")
                
            except Exception as e:
                self.stats['processing_errors'] += 1
                logger.error(f"❌ Options sentiment processing failed: {str(e)}")
                continue
        
        logger.info("🛑 Options sentiment processing loop stopped")

    async def _aggregate_sentiment_signals(self):
        """Aggregate sentiment signals across all sources for comprehensive analysis"""
        logger.info("🔄 Starting sentiment aggregation loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Get sentiment data from last hour for aggregation
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=1)
                
                # Fetch sentiment data by symbol
                for symbol in settings.SUPPORTED_OPTIONS_SYMBOLS:
                    sentiment_data = await self.database.get_sentiment_by_symbol(
                        symbol=symbol,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    if not sentiment_data:
                        continue
                    
                    # Aggregate sentiment across sources
                    aggregated_sentiment = await self._calculate_aggregated_sentiment(
                        symbol, sentiment_data
                    )
                    
                    # Store aggregated sentiment
                    await self.database.store_aggregated_sentiment(
                        symbol, aggregated_sentiment, end_time
                    )
                    
                    # Publish aggregated sentiment
                    await self.kafka_producer.publish_sentiment(
                        topic='aggregated-sentiment',
                        sentiment_data={
                            'symbol': symbol,
                            'timestamp': end_time.isoformat(),
                            **aggregated_sentiment
                        }
                    )
                    
                    # Update real-time cache
                    cache_key = f"aggregated_sentiment_{symbol}"
                    self.sentiment_cache[cache_key] = {
                        'data': aggregated_sentiment,
                        'timestamp': end_time
                    }
                
                # Sleep for 5 minutes before next aggregation
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"❌ Sentiment aggregation failed: {str(e)}")
                await asyncio.sleep(60)  # Wait before retry
        
        logger.info("🛑 Sentiment aggregation loop stopped")

    async def _calculate_aggregated_sentiment(
        self, 
        symbol: str, 
        sentiment_data: List[Dict]
    ) -> Dict:
        """Calculate comprehensive aggregated sentiment for a symbol"""
        try:
            if not sentiment_data:
                return self._get_neutral_sentiment()
            
            # Separate by source type for weighted aggregation
            news_sentiments = [s for s in sentiment_data if s['source_type'] == 'news']
            social_sentiments = [s for s in sentiment_data if s['source_type'] == 'social']
            options_sentiments = [s for s in sentiment_data if s['source_type'] == 'options_forum']
            
            # Calculate weighted sentiment scores
            news_weight = 0.4
            social_weight = 0.3
            options_weight = 0.3
            
            # News sentiment (weighted by source reliability)
            news_sentiment = self._calculate_weighted_sentiment(
                news_sentiments, 
                weight_key='confidence'
            ) if news_sentiments else 0.0
            
            # Social sentiment (weighted by engagement)
            social_sentiment = self._calculate_weighted_sentiment(
                social_sentiments, 
                weight_key='engagement_score'
            ) if social_sentiments else 0.0
            
            # Options sentiment (weighted by credibility)
            options_sentiment = self._calculate_weighted_sentiment(
                options_sentiments, 
                weight_key='analysis_confidence'
            ) if options_sentiments else 0.0
            
            # Calculate overall sentiment
            total_weight = 0
            weighted_sentiment = 0
            
            if news_sentiments:
                weighted_sentiment += news_sentiment * news_weight
                total_weight += news_weight
            
            if social_sentiments:
                weighted_sentiment += social_sentiment * social_weight
                total_weight += social_weight
            
            if options_sentiments:
                weighted_sentiment += options_sentiment * options_weight
                total_weight += options_weight
            
            overall_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
            
            # Calculate sentiment momentum (change over time)
            sentiment_momentum = self._calculate_sentiment_momentum(sentiment_data)
            
            # Calculate volatility expectations from options sentiment
            volatility_expectation = self._calculate_volatility_expectation(options_sentiments)
            
            # Determine sentiment label
            if overall_sentiment > 0.1:
                sentiment_label = 'positive'
            elif overall_sentiment < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return {
                'symbol': symbol,
                'overall_sentiment_score': round(overall_sentiment, 4),
                'sentiment_label': sentiment_label,
                'sentiment_momentum': round(sentiment_momentum, 4),
                
                # Source breakdown
                'news_sentiment': round(news_sentiment, 4),
                'social_sentiment': round(social_sentiment, 4),
                'options_sentiment': round(options_sentiment, 4),
                
                # Options-specific insights
                'volatility_expectation': round(volatility_expectation, 4),
                'options_bias': self._calculate_options_bias(options_sentiments),
                
                # Data quality metrics
                'confidence': self._calculate_overall_confidence(sentiment_data),
                'data_points': len(sentiment_data),
                'source_diversity': len(set(s['source_type'] for s in sentiment_data)),
                
                # Temporal analysis
                'sentiment_trend': self._analyze_sentiment_trend(sentiment_data),
                'peak_sentiment_time': self._find_peak_sentiment_time(sentiment_data),
                
                'aggregated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment aggregation calculation failed: {str(e)}")
            return self._get_neutral_sentiment()

    def _calculate_weighted_sentiment(
        self, 
        sentiments: List[Dict], 
        weight_key: str
    ) -> float:
        """Calculate weighted average sentiment"""
        if not sentiments:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for sentiment in sentiments:
            score = sentiment.get('sentiment_score', 0.0)
            weight = sentiment.get(weight_key, 1.0)
            
            total_weighted_score += score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def _calculate_sentiment_momentum(self, sentiment_data: List[Dict]) -> float:
        """Calculate sentiment momentum (rate of change)"""
        if len(sentiment_data) < 2:
            return 0.0
        
        # Sort by timestamp
        sorted_data = sorted(sentiment_data, key=lambda x: x['timestamp'])
        
        # Calculate momentum as change over time
        recent_sentiment = sum(s['sentiment_score'] for s in sorted_data[-5:]) / min(5, len(sorted_data))
        older_sentiment = sum(s['sentiment_score'] for s in sorted_data[:5]) / min(5, len(sorted_data))
        
        return recent_sentiment - older_sentiment

    def _calculate_volatility_expectation(self, options_sentiments: List[Dict]) -> float:
        """Calculate implied volatility expectation from options sentiment"""
        if not options_sentiments:
            return 0.0
        
        vol_expectations = []
        for sentiment in options_sentiments:
            vol_exp = sentiment.get('volatility_expectation', {})
            if isinstance(vol_exp, dict) and 'expected_iv' in vol_exp:
                vol_expectations.append(vol_exp['expected_iv'])
            elif isinstance(vol_exp, (int, float)):
                vol_expectations.append(vol_exp)
        
        return sum(vol_expectations) / len(vol_expectations) if vol_expectations else 0.0

    def _calculate_options_bias(self, options_sentiments: List[Dict]) -> str:
        """Calculate overall options bias (bullish/bearish/neutral)"""
        if not options_sentiments:
            return 'neutral'
        
        bias_scores = []
        for sentiment in options_sentiments:
            direction_bias = sentiment.get('direction_bias', 0.0)
            if isinstance(direction_bias, str):
                if direction_bias.lower() == 'bullish':
                    bias_scores.append(1.0)
                elif direction_bias.lower() == 'bearish':
                    bias_scores.append(-1.0)
                else:
                    bias_scores.append(0.0)
            else:
                bias_scores.append(direction_bias)
        
        avg_bias = sum(bias_scores) / len(bias_scores)
        
        if avg_bias > 0.2:
            return 'bullish'
        elif avg_bias < -0.2:
            return 'bearish'
        else:
            return 'neutral'

    def _calculate_overall_confidence(self, sentiment_data: List[Dict]) -> float:
        """Calculate overall confidence in the sentiment analysis"""
        if not sentiment_data:
            return 0.0
        
        confidences = []
        for sentiment in sentiment_data:
            confidence = sentiment.get('confidence', 0.5)
            confidences.append(confidence)
        
        return sum(confidences) / len(confidences)

    def _analyze_sentiment_trend(self, sentiment_data: List[Dict]) -> str:
        """Analyze sentiment trend direction"""
        if len(sentiment_data) < 3:
            return 'insufficient_data'
        
        # Sort by timestamp
        sorted_data = sorted(sentiment_data, key=lambda x: x['timestamp'])
        
        # Calculate trend using linear regression on sentiment scores
        timestamps = [(datetime.fromisoformat(s['timestamp']) - datetime(1970, 1, 1)).total_seconds() 
                     for s in sorted_data]
        sentiments = [s['sentiment_score'] for s in sorted_data]
        
        # Simple trend calculation
        n = len(timestamps)
        sum_x = sum(timestamps)
        sum_y = sum(sentiments)
        sum_xy = sum(t * s for t, s in zip(timestamps, sentiments))
        sum_x2 = sum(t * t for t in timestamps)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        
        if slope > 0.0001:
            return 'improving'
        elif slope < -0.0001:
            return 'deteriorating'
        else:
            return 'stable'

    def _find_peak_sentiment_time(self, sentiment_data: List[Dict]) -> Optional[str]:
        """Find timestamp of peak sentiment"""
        if not sentiment_data:
            return None
        
        peak_sentiment = max(sentiment_data, key=lambda x: abs(x.get('sentiment_score', 0)))
        return peak_sentiment.get('timestamp')

    def _get_neutral_sentiment(self) -> Dict:
        """Return neutral sentiment baseline"""
        return {
            'overall_sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'sentiment_momentum': 0.0,
            'news_sentiment': 0.0,
            'social_sentiment': 0.0,
            'options_sentiment': 0.0,
            'volatility_expectation': 0.0,
            'options_bias': 'neutral',
            'confidence': 0.0,
            'data_points': 0,
            'source_diversity': 0,
            'sentiment_trend': 'insufficient_data',
            'peak_sentiment_time': None,
            'aggregated_at': datetime.utcnow().isoformat()
        }

    async def _update_sentiment_cache(self, sentiment_result: Dict):
        """Update real-time sentiment cache"""
        try:
            symbols = sentiment_result.get('symbols_mentioned', [])
            for symbol in symbols:
                cache_key = f"latest_sentiment_{symbol}"
                self.sentiment_cache[cache_key] = {
                    'data': sentiment_result,
                    'timestamp': datetime.utcnow()
                }
            
            # Clean expired cache entries
            current_time = datetime.utcnow()
            expired_keys = [
                key for key, value in self.sentiment_cache.items()
                if current_time - value['timestamp'] > self.cache_expiry
            ]
            
            for key in expired_keys:
                del self.sentiment_cache[key]
                
        except Exception as e:
            logger.error(f"❌ Cache update failed: {str(e)}")

    async def _check_sentiment_alerts(self, sentiment_result: Dict):
        """Check for sentiment-based alerts and notifications"""
        try:
            # Check for extreme sentiment scores
            sentiment_score = abs(sentiment_result.get('sentiment_score', 0))
            if sentiment_score > 0.8:
                await self._generate_sentiment_alert(
                    'extreme_sentiment',
                    sentiment_result,
                    f"Extreme sentiment detected: {sentiment_result.get('sentiment_label')} "
                    f"({sentiment_score:.3f})"
                )
            
            # Check for high options relevance
            options_relevance = sentiment_result.get('options_relevance_score', 0)
            if options_relevance > 0.7:
                await self._generate_sentiment_alert(
                    'high_options_relevance',
                    sentiment_result,
                    f"High options relevance detected: {options_relevance:.3f}"
                )
            
            # Check for volatility expectations
            vol_expectation = sentiment_result.get('volatility_expectation', {})
            if isinstance(vol_expectation, dict) and vol_expectation.get('expected_change', 0) > 0.3:
                await self._generate_sentiment_alert(
                    'high_volatility_expectation',
                    sentiment_result,
                    f"High volatility expectation: {vol_expectation.get('expected_change', 0):.2f}"
                )
                
        except Exception as e:
            logger.error(f"❌ Sentiment alert check failed: {str(e)}")

    async def _check_options_strategy_alerts(self, sentiment_result: Dict):
        """Check for options strategy-specific alerts"""
        try:
            strategies = sentiment_result.get('mentioned_strategies', [])
            
            # Check for strategy consensus
            if len(strategies) >= 3:
                await self._generate_sentiment_alert(
                    'strategy_consensus',
                    sentiment_result,
                    f"Multiple strategies mentioned: {', '.join(strategies[:3])}"
                )
            
            # Check for directional bias strength
            direction_bias = sentiment_result.get('direction_bias', 0)
            if isinstance(direction_bias, (int, float)) and abs(direction_bias) > 0.7:
                await self._generate_sentiment_alert(
                    'strong_directional_bias',
                    sentiment_result,
                    f"Strong directional bias: {direction_bias:.2f}"
                )
                
        except Exception as e:
            logger.error(f"❌ Options strategy alert check failed: {str(e)}")

    async def _generate_sentiment_alert(
        self, 
        alert_type: str, 
        sentiment_result: Dict, 
        message: str
    ):
        """Generate and publish sentiment alerts"""
        try:
            alert = {
                'alert_id': f"{alert_type}_{int(datetime.utcnow().timestamp())}",
                'alert_type': alert_type,
                'severity': 'high' if alert_type.startswith('extreme') else 'medium',
                'message': message,
                'symbols': sentiment_result.get('symbols_mentioned', []),
                'sentiment_data': {
                    'sentiment_score': sentiment_result.get('sentiment_score'),
                    'sentiment_label': sentiment_result.get('sentiment_label'),
                    'confidence': sentiment_result.get('confidence'),
                    'source_type': sentiment_result.get('source_type')
                },
                'timestamp': datetime.utcnow().isoformat(),
                'expires_at': (datetime.utcnow() + timedelta(hours=4)).isoformat()
            }
            
            # Publish alert
            await self.kafka_producer.publish_alert(
                topic='sentiment-alerts',
                alert_data=alert
            )
            
            # Store in database
            await self.database.store_sentiment_alert(alert)
            
            # Update statistics
            self.stats['sentiment_alerts_generated'] += 1
            
            logger.info(f"🚨 Generated sentiment alert: {alert_type} - {message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate sentiment alert: {str(e)}")

    async def _generate_viral_content_alert(self, sentiment_result: Dict):
        """Generate alert for potentially viral content"""
        try:
            await self._generate_sentiment_alert(
                'viral_content',
                sentiment_result,
                f"Viral content detected: {sentiment_result.get('engagement_score', 0):.2f} engagement, "
                f"{sentiment_result.get('viral_potential', 0):.2f} viral potential"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to generate viral content alert: {str(e)}")

    async def _maintain_sentiment_cache(self):
        """Maintain sentiment cache and cleanup expired entries"""
        logger.info("🔄 Starting sentiment cache maintenance loop")
        
        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                
                # Clean expired entries
                expired_keys = [
                    key for key, value in self.sentiment_cache.items()
                    if current_time - value['timestamp'] > self.cache_expiry
                ]
                
                for key in expired_keys:
                    del self.sentiment_cache[key]
                
                if expired_keys:
                    logger.debug(f"🧹 Cleaned {len(expired_keys)} expired cache entries")
                
                # Update cache statistics
                await self.metrics.record_cache_statistics(
                    cache_size=len(self.sentiment_cache),
                    expired_entries=len(expired_keys)
                )
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"❌ Cache maintenance failed: {str(e)}")
                await asyncio.sleep(60)
        
        logger.info("🛑 Sentiment cache maintenance loop stopped")

    async def _metrics_reporting_loop(self):
        """Report service metrics periodically"""
        logger.info("📊 Starting metrics reporting loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Calculate average processing time
                if self.stats['sentiment_analyses_completed'] > 0:
                    # This would be calculated from actual timing data in production
                    self.stats['average_processing_time'] = 0.5  # placeholder
                
                # Report comprehensive metrics
                await self.metrics.report_service_statistics(self.stats)
                
                # Log statistics
                logger.info(
                    f"📊 Sentiment Service Stats - "
                    f"Processed: {self.stats['messages_processed']}, "
                    f"Analyzed: {self.stats['sentiment_analyses_completed']}, "
                    f"Entities: {self.stats['options_entities_extracted']}, "
                    f"Alerts: {self.stats['sentiment_alerts_generated']}, "
                    f"Errors: {self.stats['processing_errors']}"
                )
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"❌ Metrics reporting failed: {str(e)}")
                await asyncio.sleep(300)
        
        logger.info("🛑 Metrics reporting loop stopped")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up resources...")
        
        try:
            # Stop Kafka components
            await self.kafka_consumer.stop()
            await self.kafka_producer.stop()
            
            # Close database connection
            await self.database.close()
            
            # Stop metrics collection
            await self.metrics.stop()
            
            # Clear cache
            self.sentiment_cache.clear()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {str(e)}")

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()


async def main():
    """Main entry point for the enhanced sentiment analysis service"""
    # Set up uvloop for better async performance
    uvloop.install()
    
    # Create service instance
    service = EnhancedSentimentAnalysisService()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, service.handle_shutdown)
    signal.signal(signal.SIGTERM, service.handle_shutdown)
    
    try:
        # Start the service
        await service.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ Service failed: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("👋 Enhanced Sentiment Analysis Service shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())

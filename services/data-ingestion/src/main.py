"""
Enhanced Data Ingestion Service with Options Intelligence
Market Intelligence Platform - Sprint 1, Prompt 2

Features:
- Multi-source data collection (News, Market Data, Options Flow)
- CBOE and ORATS options data integration
- Enhanced validation schemas with options-specific validation
- Circuit breaker pattern with exponential backoff
- Kafka producer with enhanced topics
- Comprehensive error handling and monitoring
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uvloop

from config import settings
from data_sources import (
    NewsAPICollector,
    TwitterAPICollector, 
    AlphaVantageCollector,
    CBOEOptionsCollector,
    ORATSCollector,
    DarkPoolCollector
)
from kafka_producer import EnhancedKafkaProducer
from circuit_breaker import CircuitBreaker
from monitoring import MetricsCollector
from validators import DataValidationEngine

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/mip/data-ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

class EnhancedDataIngestionService:
    """Enhanced data ingestion service with options intelligence"""
    
    def __init__(self):
        """Initialize the enhanced data ingestion service"""
        logger.info("🚀 Initializing Enhanced Data Ingestion Service...")
        
        # Initialize components
        self.kafka_producer = EnhancedKafkaProducer()
        self.metrics = MetricsCollector()
        self.validator = DataValidationEngine()
        
        # Initialize data collectors with circuit breakers
        self.collectors = {
            'news': {
                'newsapi': NewsAPICollector(),
                'twitter': TwitterAPICollector()
            },
            'market': {
                'alphavantage': AlphaVantageCollector()
            },
            'options': {
                'cboe': CBOEOptionsCollector(),
                'orats': ORATSCollector(),
                'darkpool': DarkPoolCollector()
            }
        }
        
        # Circuit breakers for each data source
        self.circuit_breakers = {
            source_type: {
                name: CircuitBreaker(
                    name=f"{source_type}_{name}",
                    failure_threshold=settings.CIRCUIT_BREAKER_THRESHOLD,
                    timeout=settings.CIRCUIT_BREAKER_TIMEOUT,
                    expected_exception=Exception
                )
                for name in sources.keys()
            }
            for source_type, sources in self.collectors.items()
        }
        
        # Collection intervals (in seconds)
        self.collection_intervals = {
            'news': 300,      # 5 minutes
            'market': 60,     # 1 minute  
            'options': 30     # 30 seconds - more frequent for options
        }
        
        # Track collection statistics
        self.stats = {
            'collections_started': 0,
            'collections_completed': 0,
            'validation_errors': 0,
            'kafka_publish_errors': 0,
            'circuit_breaker_opens': 0
        }
        
        # Shutdown flag
        self.shutdown_event = asyncio.Event()
        
        logger.info("✅ Enhanced Data Ingestion Service initialized")

    async def start(self):
        """Start the enhanced data ingestion service"""
        logger.info("▶️  Starting Enhanced Data Ingestion Service...")
        
        try:
            # Start Kafka producer
            await self.kafka_producer.start()
            
            # Start metrics collection
            await self.metrics.start()
            
            # Create collection tasks
            tasks = []
            
            # News collection tasks
            for name, collector in self.collectors['news'].items():
                task = asyncio.create_task(
                    self._collection_loop('news', name, collector),
                    name=f"news_{name}_collection"
                )
                tasks.append(task)
            
            # Market data collection tasks
            for name, collector in self.collectors['market'].items():
                task = asyncio.create_task(
                    self._collection_loop('market', name, collector),
                    name=f"market_{name}_collection"
                )
                tasks.append(task)
            
            # Options data collection tasks (enhanced frequency)
            for name, collector in self.collectors['options'].items():
                task = asyncio.create_task(
                    self._collection_loop('options', name, collector),
                    name=f"options_{name}_collection"
                )
                tasks.append(task)
            
            # Health check task
            health_task = asyncio.create_task(
                self._health_check_loop(),
                name="health_check"
            )
            tasks.append(health_task)
            
            # Metrics reporting task
            metrics_task = asyncio.create_task(
                self._metrics_reporting_loop(),
                name="metrics_reporting"
            )
            tasks.append(metrics_task)
            
            logger.info(f"🔄 Started {len(tasks)} collection tasks")
            
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

    async def _collection_loop(self, source_type: str, source_name: str, collector):
        """Enhanced collection loop with circuit breaker and exponential backoff"""
        logger.info(f"🔄 Starting {source_type}_{source_name} collection loop")
        
        circuit_breaker = self.circuit_breakers[source_type][source_name]
        interval = self.collection_intervals[source_type]
        consecutive_failures = 0
        max_backoff = 300  # 5 minutes maximum backoff
        
        while not self.shutdown_event.is_set():
            try:
                self.stats['collections_started'] += 1
                collection_start = datetime.utcnow()
                
                # Collect data through circuit breaker
                async with circuit_breaker:
                    data = await collector.collect_data()
                
                if data:
                    # Validate data
                    validated_data = await self._validate_and_enrich_data(
                        source_type, source_name, data
                    )
                    
                    if validated_data:
                        # Publish to Kafka
                        await self._publish_to_kafka(
                            source_type, source_name, validated_data
                        )
                        
                        # Update metrics
                        collection_time = (datetime.utcnow() - collection_start).total_seconds()
                        await self.metrics.record_collection_success(
                            source_type, source_name, len(validated_data), collection_time
                        )
                        
                        logger.debug(f"✅ {source_type}_{source_name}: Collected {len(validated_data)} records")
                        
                        # Reset failure counter on success
                        consecutive_failures = 0
                        
                    else:
                        logger.warning(f"⚠️  {source_type}_{source_name}: No valid data after validation")
                
                self.stats['collections_completed'] += 1
                
                # Wait for next collection
                await asyncio.sleep(interval)
                
            except CircuitBreaker.CircuitBreakerOpenError:
                self.stats['circuit_breaker_opens'] += 1
                logger.warning(f"🔴 {source_type}_{source_name}: Circuit breaker open, waiting...")
                await asyncio.sleep(interval * 2)  # Wait longer when circuit breaker is open
                
            except Exception as e:
                consecutive_failures += 1
                
                # Calculate exponential backoff
                backoff_time = min(interval * (2 ** consecutive_failures), max_backoff)
                
                logger.error(
                    f"❌ {source_type}_{source_name} collection failed "
                    f"(attempt {consecutive_failures}): {str(e)}"
                )
                
                # Record failure metrics
                await self.metrics.record_collection_failure(
                    source_type, source_name, str(e)
                )
                
                # Wait with exponential backoff
                logger.info(f"⏳ Backing off for {backoff_time}s before retry...")
                await asyncio.sleep(backoff_time)
        
        logger.info(f"🛑 {source_type}_{source_name} collection loop stopped")

    async def _validate_and_enrich_data(
        self, 
        source_type: str, 
        source_name: str, 
        data: List[Dict]
    ) -> List[Dict]:
        """Enhanced validation and enrichment with options-specific processing"""
        try:
            validated_data = []
            
            for record in data:
                try:
                    # Add metadata
                    record['_metadata'] = {
                        'source_type': source_type,
                        'source_name': source_name,
                        'collected_at': datetime.utcnow().isoformat(),
                        'ingestion_id': f"{source_type}_{source_name}_{int(datetime.utcnow().timestamp())}"
                    }
                    
                    # Validate based on source type
                    if source_type == 'options':
                        validated_record = await self.validator.validate_options_data(record)
                    elif source_type == 'market':
                        validated_record = await self.validator.validate_market_data(record)
                    elif source_type == 'news':
                        validated_record = await self.validator.validate_news_data(record)
                    else:
                        validated_record = await self.validator.validate_generic_data(record)
                    
                    if validated_record:
                        # Options-specific enrichment
                        if source_type == 'options':
                            validated_record = await self._enrich_options_data(validated_record)
                        
                        validated_data.append(validated_record)
                        
                except Exception as e:
                    self.stats['validation_errors'] += 1
                    logger.error(f"❌ Validation failed for {source_type}_{source_name} record: {str(e)}")
                    continue
            
            logger.debug(f"✅ Validated {len(validated_data)}/{len(data)} records for {source_type}_{source_name}")
            return validated_data
            
        except Exception as e:
            logger.error(f"❌ Validation batch failed for {source_type}_{source_name}: {str(e)}")
            return []

    async def _enrich_options_data(self, options_record: Dict) -> Dict:
        """Enhanced options data enrichment with Greeks calculation and analytics"""
        try:
            # Calculate additional options metrics
            if all(key in options_record for key in ['strike', 'expiry', 'underlying_price', 'implied_volatility']):
                
                # Calculate time to expiry
                expiry_date = datetime.fromisoformat(options_record['expiry'].replace('Z', '+00:00'))
                time_to_expiry = (expiry_date - datetime.utcnow()).total_seconds() / (365.25 * 24 * 3600)
                options_record['time_to_expiry'] = time_to_expiry
                
                # Calculate moneyness
                strike = float(options_record['strike'])
                underlying_price = float(options_record['underlying_price'])
                moneyness = underlying_price / strike
                options_record['moneyness'] = moneyness
                
                # Determine option category
                if moneyness > 1.02:
                    options_record['moneyness_category'] = 'ITM' if options_record['option_type'] == 'call' else 'OTM'
                elif moneyness < 0.98:
                    options_record['moneyness_category'] = 'OTM' if options_record['option_type'] == 'call' else 'ITM'
                else:
                    options_record['moneyness_category'] = 'ATM'
                
                # Calculate IV percentile (simplified - in production, use historical data)
                iv = float(options_record['implied_volatility'])
                options_record['iv_rank'] = min(iv * 100 / 2.0, 100)  # Simplified calculation
                
                # Flag unusual activity
                volume = int(options_record.get('volume', 0))
                open_interest = int(options_record.get('open_interest', 1))
                volume_oi_ratio = volume / max(open_interest, 1)
                
                options_record['volume_oi_ratio'] = volume_oi_ratio
                options_record['unusual_activity'] = volume_oi_ratio > 0.5  # Flag if volume > 50% of OI
                
                # Add analytics flags
                options_record['analytics'] = {
                    'high_iv': iv > 0.5,
                    'short_term': time_to_expiry < 0.0833,  # < 30 days
                    'liquid': volume > 100,
                    'large_oi': open_interest > 1000
                }
            
            return options_record
            
        except Exception as e:
            logger.error(f"❌ Options enrichment failed: {str(e)}")
            return options_record

    async def _publish_to_kafka(
        self, 
        source_type: str, 
        source_name: str, 
        data: List[Dict]
    ):
        """Enhanced Kafka publishing with topic routing and error handling"""
        try:
            # Determine Kafka topic based on source type
            topic_mapping = {
                'news': 'raw-news',
                'market': 'raw-market-data',
                'options': 'raw-options-flow'  # New enhanced topic
            }
            
            base_topic = topic_mapping.get(source_type, 'raw-data')
            
            # Publish data in batches
            batch_size = 100
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                # For options data, also publish to enriched topic
                if source_type == 'options':
                    await self.kafka_producer.publish_batch(
                        topic='options-enriched',
                        data=batch,
                        source=f"{source_type}_{source_name}"
                    )
                
                # Publish to base topic
                await self.kafka_producer.publish_batch(
                    topic=base_topic,
                    data=batch,
                    source=f"{source_type}_{source_name}"
                )
            
            logger.debug(f"📤 Published {len(data)} records to Kafka for {source_type}_{source_name}")
            
        except Exception as e:
            self.stats['kafka_publish_errors'] += 1
            logger.error(f"❌ Kafka publish failed for {source_type}_{source_name}: {str(e)}")
            raise

    async def _health_check_loop(self):
        """Health check loop to monitor service health"""
        logger.info("❤️  Starting health check loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Check circuit breaker states
                for source_type, breakers in self.circuit_breakers.items():
                    for name, breaker in breakers.items():
                        if breaker.is_open:
                            logger.warning(f"🔴 Circuit breaker {source_type}_{name} is OPEN")
                
                # Check Kafka producer health
                if not await self.kafka_producer.health_check():
                    logger.error("❌ Kafka producer health check failed")
                
                # Log statistics
                logger.info(f"📊 Stats: Collections {self.stats['collections_completed']}/{self.stats['collections_started']}, "
                          f"Validation errors: {self.stats['validation_errors']}, "
                          f"Kafka errors: {self.stats['kafka_publish_errors']}")
                
                await asyncio.sleep(60)  # Health check every minute
                
            except Exception as e:
                logger.error(f"❌ Health check failed: {str(e)}")
                await asyncio.sleep(60)
        
        logger.info("🛑 Health check loop stopped")

    async def _metrics_reporting_loop(self):
        """Metrics reporting loop for monitoring and observability"""
        logger.info("📊 Starting metrics reporting loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Report collection metrics
                await self.metrics.report_collection_stats(self.stats)
                
                # Report circuit breaker states
                breaker_states = {}
                for source_type, breakers in self.circuit_breakers.items():
                    for name, breaker in breakers.items():
                        breaker_states[f"{source_type}_{name}"] = {
                            'state': 'OPEN' if breaker.is_open else 'CLOSED',
                            'failure_count': breaker.failure_count,
                            'last_failure': breaker.last_failure_time
                        }
                
                await self.metrics.report_circuit_breaker_states(breaker_states)
                
                # Report Kafka producer metrics
                kafka_metrics = await self.kafka_producer.get_metrics()
                await self.metrics.report_kafka_metrics(kafka_metrics)
                
                await asyncio.sleep(300)  # Report every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Metrics reporting failed: {str(e)}")
                await asyncio.sleep(300)
        
        logger.info("🛑 Metrics reporting loop stopped")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up resources...")
        
        try:
            # Stop Kafka producer
            await self.kafka_producer.stop()
            
            # Stop metrics collection
            await self.metrics.stop()
            
            # Close data collectors
            for source_type, sources in self.collectors.items():
                for name, collector in sources.items():
                    if hasattr(collector, 'close'):
                        await collector.close()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {str(e)}")

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()


async def main():
    """Main entry point for the enhanced data ingestion service"""
    # Set up uvloop for better async performance
    uvloop.install()
    
    # Create service instance
    service = EnhancedDataIngestionService()
    
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
        logger.info("👋 Enhanced Data Ingestion Service shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())

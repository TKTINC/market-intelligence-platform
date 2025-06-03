"""
Enhanced Kafka producer with options-specific topics and error handling
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from kafka import KafkaProducer
from kafka.errors import KafkaError
import ssl

from config import settings

logger = logging.getLogger(__name__)

class EnhancedKafkaProducer:
    """Enhanced Kafka producer with robust error handling and monitoring"""
    
    def __init__(self):
        """Initialize the enhanced Kafka producer"""
        self.producer: Optional[KafkaProducer] = None
        self.is_connected = False
        
        # Producer metrics
        self.metrics = {
            'total_messages': 0,
            'successful_messages': 0,
            'failed_messages': 0,
            'total_batches': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'connection_errors': 0,
            'last_success_time': None,
            'last_error_time': None
        }
        
        # Topic configurations
        self.topic_configs = {
            'raw-news': {'partitions': 50, 'replication_factor': 3},
            'raw-market-data': {'partitions': 100, 'replication_factor': 3},
            'raw-options-flow': {'partitions': 50, 'replication_factor': 3},
            'options-enriched': {'partitions': 100, 'replication_factor': 3},
            'cleaned-data': {'partitions': 100, 'replication_factor': 3}
        }
    
    async def start(self):
        """Start the Kafka producer with enhanced configuration"""
        try:
            logger.info("🚀 Starting enhanced Kafka producer...")
            
            # SSL/TLS configuration for MSK
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Enhanced producer configuration
            producer_config = {
                'bootstrap_servers': settings.KAFKA_BOOTSTRAP_SERVERS.split(','),
                'security_protocol': settings.KAFKA_SECURITY_PROTOCOL,
                'ssl_context': ssl_context,
                
                # Serialization
                'key_serializer': lambda x: json.dumps(x).encode('utf-8') if x else None,
                'value_serializer': lambda x: json.dumps(x, default=str).encode('utf-8'),
                
                # Performance and reliability
                'acks': 'all',  # Wait for all replicas
                'retries': 5,
                'max_in_flight_requests_per_connection': 1,  # Ensure ordering
                'batch_size': 16384,
                'linger_ms': 10,  # Small batch delay for throughput
                'compression_type': 'lz4',  # Fast compression
                
                # Buffering and timeouts
                'buffer_memory': 33554432,  # 32MB buffer
                'request_timeout_ms': 30000,
                'delivery_timeout_ms': 120000,
                
                # Error handling
                'retry_backoff_ms': 1000,
                'reconnect_backoff_ms': 50,
                'reconnect_backoff_max_ms': 1000,
            }
            
            # Add SSL certificate configuration if available
            if hasattr(settings, 'KAFKA_SSL_CA_LOCATION'):
                producer_config.update({
                    'ssl_cafile': settings.KAFKA_SSL_CA_LOCATION,
                    'ssl_certfile': settings.KAFKA_SSL_CERTIFICATE_LOCATION,
                    'ssl_keyfile': settings.KAFKA_SSL_KEY_LOCATION,
                })
            
            self.producer = KafkaProducer(**producer_config)
            self.is_connected = True
            
            logger.info("✅ Enhanced Kafka producer started successfully")
            
        except Exception as e:
            self.metrics['connection_errors'] += 1
            logger.error(f"❌ Failed to start Kafka producer: {str(e)}")
            raise
    
    async def publish_batch(
        self, 
        topic: str, 
        data: List[Dict], 
        source: str,
        partition_key: Optional[str] = None
    ):
        """Publish a batch of messages to Kafka with enhanced error handling"""
        if not self.is_connected:
            raise RuntimeError("Kafka producer not connected")
        
        batch_start_time = datetime.utcnow()
        self.metrics['total_batches'] += 1
        
        try:
            # Prepare messages
            messages = []
            for record in data:
                # Add batch metadata
                enhanced_record = {
                    **record,
                    '_batch_info': {
                        'batch_id': f"{source}_{int(batch_start_time.timestamp())}",
                        'batch_size': len(data),
                        'source': source,
                        'published_at': batch_start_time.isoformat()
                    }
                }
                
                # Determine partition key
                key = partition_key or record.get('symbol') or source
                
                messages.append((topic, key, enhanced_record))
            
            # Send messages asynchronously
            futures = []
            for topic, key, message in messages:
                future = self.producer.send(
                    topic=topic,
                    key=key,
                    value=message,
                    timestamp_ms=int(datetime.utcnow().timestamp() * 1000)
                )
                futures.append(future)
            
            # Wait for all messages to be sent
            successful_count = 0
            failed_count = 0
            
            for future in futures:
                try:
                    record_metadata = future.get(timeout=30)
                    successful_count += 1
                    logger.debug(f"✅ Message sent to {record_metadata.topic}:{record_metadata.partition}:{record_metadata.offset}")
                    
                except KafkaError as e:
                    failed_count += 1
                    logger.error(f"❌ Message send failed: {str(e)}")
            
            # Update metrics
            self.metrics['total_messages'] += len(data)
            self.metrics['successful_messages'] += successful_count
            self.metrics['failed_messages'] += failed_count
            
            if failed_count == 0:
                self.metrics['successful_batches'] += 1
                self.metrics['last_success_time'] = datetime.utcnow()
                
                batch_duration = (datetime.utcnow() - batch_start_time).total_seconds()
                logger.info(f"📤 Published batch: {len(data)} messages to {topic} in {batch_duration:.2f}s")
                
            else:
                self.metrics['failed_batches'] += 1
                self.metrics['last_error_time'] = datetime.utcnow()
                logger.error(f"❌ Batch partially failed: {successful_count}/{len(data)} messages sent")
            
            # Flush to ensure delivery
            self.producer.flush()
            
        except Exception as e:
            self.metrics['failed_batches'] += 1
            self.metrics['failed_messages'] += len(data)
            self.metrics['last_error_time'] = datetime.utcnow()
            
            logger.error(f"❌ Batch publish failed for {topic}: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Perform health check on Kafka producer"""
        try:
            if not self.is_connected or not self.producer:
                return False
            
            # Try to get cluster metadata
            metadata = self.producer.partitions_for('__consumer_offsets')
            return metadata is not None
            
        except Exception as e:
            logger.error(f"❌ Kafka health check failed: {str(e)}")
            return False
    
    async def get_metrics(self) -> Dict:
        """Get producer metrics for monitoring"""
        return {
            **self.metrics,
            'is_connected': self.is_connected,
            'success_rate': (
                self.metrics['successful_messages'] / 
                max(self.metrics['total_messages'], 1)
            ) * 100,
            'batch_success_rate': (
                self.metrics['successful_batches'] / 
                max(self.metrics['total_batches'], 1)
            ) * 100
        }
    
    async def stop(self):
        """Stop the Kafka producer"""
        try:
            if self.producer:
                logger.info("🛑 Stopping Kafka producer...")
                self.producer.flush()
                self.producer.close()
                self.is_connected = False
                logger.info("✅ Kafka producer stopped")
                
        except Exception as e:
            logger.error(f"❌ Error stopping Kafka producer: {str(e)}")

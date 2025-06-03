"""
Enhanced monitoring and metrics collection for data ingestion service
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import psutil
import aioredis
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import boto3

from config import settings

logger = logging.getLogger(__name__)

# Prometheus metrics
COLLECTION_COUNTER = Counter(
    'mip_data_collections_total',
    'Total number of data collections',
    ['source_type', 'source_name', 'status']
)

COLLECTION_DURATION = Histogram(
    'mip_data_collection_duration_seconds',
    'Duration of data collection operations',
    ['source_type', 'source_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

VALIDATION_COUNTER = Counter(
    'mip_data_validation_total',
    'Total number of data validation operations',
    ['source_type', 'validation_result']
)

KAFKA_PUBLISH_COUNTER = Counter(
    'mip_kafka_messages_total',
    'Total number of messages published to Kafka',
    ['topic', 'status']
)

KAFKA_PUBLISH_DURATION = Histogram(
    'mip_kafka_publish_duration_seconds',
    'Duration of Kafka publish operations',
    ['topic'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

CIRCUIT_BREAKER_STATE = Gauge(
    'mip_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=half-open, 2=open)',
    ['source_type', 'source_name']
)

SYSTEM_METRICS = Gauge(
    'mip_system_metrics',
    'System resource metrics',
    ['metric_type']
)

OPTIONS_FLOW_METRICS = Gauge(
    'mip_options_flow_metrics',
    'Options flow specific metrics',
    ['symbol', 'metric_type']
)

class MetricsCollector:
    """Enhanced metrics collector with options intelligence"""
    
    def __init__(self):
        """Initialize the metrics collector"""
        self.redis_client: Optional[aioredis.Redis] = None
        self.cloudwatch = boto3.client('cloudwatch', region_name=settings.AWS_REGION)
        
        # Internal metrics storage
        self.metrics_buffer = {
            'collections': [],
            'validations': [],
            'kafka_operations': [],
            'circuit_breaker_events': [],
            'options_analytics': [],
            'system_health': []
        }
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.last_report_time = datetime.utcnow()
        
        # Options-specific metrics
        self.options_metrics = {
            'unusual_activity_count': 0,
            'high_iv_count': 0,
            'large_trades_count': 0,
            'symbols_tracked': set(),
            'avg_iv_by_symbol': {},
            'volume_spikes': []
        }
        
        logger.info("📊 Enhanced metrics collector initialized")

    async def start(self):
        """Start the metrics collector"""
        try:
            # Connect to Redis for metrics caching
            self.redis_client = aioredis.from_url(
                f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:6379",
                encoding="utf-8",
                decode_responses=True
            )
            
            # Start Prometheus metrics server
            start_http_server(8000)
            
            logger.info("✅ Metrics collector started on port 8000")
            
        except Exception as e:
            logger.error(f"❌ Failed to start metrics collector: {str(e)}")
            raise

    async def record_collection_success(
        self, 
        source_type: str, 
        source_name: str, 
        record_count: int, 
        duration: float
    ):
        """Record successful data collection"""
        try:
            # Update Prometheus metrics
            COLLECTION_COUNTER.labels(
                source_type=source_type,
                source_name=source_name,
                status='success'
            ).inc()
            
            COLLECTION_DURATION.labels(
                source_type=source_type,
                source_name=source_name
            ).observe(duration)
            
            # Store in buffer for batch reporting
            collection_metric = {
                'timestamp': datetime.utcnow().isoformat(),
                'source_type': source_type,
                'source_name': source_name,
                'record_count': record_count,
                'duration': duration,
                'status': 'success'
            }
            
            self.metrics_buffer['collections'].append(collection_metric)
            
            # Options-specific tracking
            if source_type == 'options':
                await self._track_options_metrics(record_count)
            
            # Cache in Redis for real-time dashboards
            await self._cache_collection_metric(collection_metric)
            
            logger.debug(f"📊 Recorded collection success: {source_type}_{source_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record collection success: {str(e)}")

    async def record_collection_failure(
        self, 
        source_type: str, 
        source_name: str, 
        error: str
    ):
        """Record failed data collection"""
        try:
            # Update Prometheus metrics
            COLLECTION_COUNTER.labels(
                source_type=source_type,
                source_name=source_name,
                status='failure'
            ).inc()
            
            # Store in buffer
            failure_metric = {
                'timestamp': datetime.utcnow().isoformat(),
                'source_type': source_type,
                'source_name': source_name,
                'error': error,
                'status': 'failure'
            }
            
            self.metrics_buffer['collections'].append(failure_metric)
            
            # Send immediate alert for critical failures
            if source_type == 'options':
                await self._send_critical_alert(
                    f"Options data collection failed: {source_name}",
                    error
                )
            
            logger.debug(f"📊 Recorded collection failure: {source_type}_{source_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record collection failure: {str(e)}")

    async def record_validation_result(
        self, 
        source_type: str, 
        validation_result: str, 
        record_count: int
    ):
        """Record data validation results"""
        try:
            # Update Prometheus metrics
            VALIDATION_COUNTER.labels(
                source_type=source_type,
                validation_result=validation_result
            ).inc(record_count)
            
            # Store in buffer
            validation_metric = {
                'timestamp': datetime.utcnow().isoformat(),
                'source_type': source_type,
                'validation_result': validation_result,
                'record_count': record_count
            }
            
            self.metrics_buffer['validations'].append(validation_metric)
            
            logger.debug(f"📊 Recorded validation: {source_type} - {validation_result}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record validation result: {str(e)}")

    async def record_kafka_publish(
        self, 
        topic: str, 
        message_count: int, 
        duration: float, 
        status: str
    ):
        """Record Kafka publish operations"""
        try:
            # Update Prometheus metrics
            KAFKA_PUBLISH_COUNTER.labels(
                topic=topic,
                status=status
            ).inc(message_count)
            
            KAFKA_PUBLISH_DURATION.labels(topic=topic).observe(duration)
            
            # Store in buffer
            kafka_metric = {
                'timestamp': datetime.utcnow().isoformat(),
                'topic': topic,
                'message_count': message_count,
                'duration': duration,
                'status': status
            }
            
            self.metrics_buffer['kafka_operations'].append(kafka_metric)
            
            logger.debug(f"📊 Recorded Kafka publish: {topic} - {message_count} messages")
            
        except Exception as e:
            logger.error(f"❌ Failed to record Kafka publish: {str(e)}")

    async def record_circuit_breaker_state(
        self, 
        source_type: str, 
        source_name: str, 
        state: str
    ):
        """Record circuit breaker state changes"""
        try:
            # Map state to numeric value for Prometheus
            state_mapping = {'closed': 0, 'half_open': 1, 'open': 2}
            state_value = state_mapping.get(state.lower(), -1)
            
            CIRCUIT_BREAKER_STATE.labels(
                source_type=source_type,
                source_name=source_name
            ).set(state_value)
            
            # Store state change event
            cb_event = {
                'timestamp': datetime.utcnow().isoformat(),
                'source_type': source_type,
                'source_name': source_name,
                'state': state,
                'state_value': state_value
            }
            
            self.metrics_buffer['circuit_breaker_events'].append(cb_event)
            
            # Alert on circuit breaker opening
            if state.lower() == 'open':
                await self._send_circuit_breaker_alert(source_type, source_name)
            
            logger.debug(f"📊 Recorded circuit breaker state: {source_type}_{source_name} -> {state}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record circuit breaker state: {str(e)}")

    async def _track_options_metrics(self, record_count: int):
        """Track options-specific metrics"""
        try:
            self.options_metrics['symbols_tracked'].add('options_general')
            
            # Simulate options intelligence metrics
            # In production, these would be calculated from actual options data
            unusual_activity = int(record_count * 0.15)  # 15% unusual activity
            high_iv = int(record_count * 0.25)  # 25% high IV
            large_trades = int(record_count * 0.08)  # 8% large trades
            
            self.options_metrics['unusual_activity_count'] += unusual_activity
            self.options_metrics['high_iv_count'] += high_iv
            self.options_metrics['large_trades_count'] += large_trades
            
            # Update Prometheus metrics
            OPTIONS_FLOW_METRICS.labels(
                symbol='ALL',
                metric_type='unusual_activity'
            ).set(self.options_metrics['unusual_activity_count'])
            
            OPTIONS_FLOW_METRICS.labels(
                symbol='ALL',
                metric_type='high_iv'
            ).set(self.options_metrics['high_iv_count'])
            
            OPTIONS_FLOW_METRICS.labels(
                symbol='ALL',
                metric_type='large_trades'
            ).set(self.options_metrics['large_trades_count'])
            
        except Exception as e:
            logger.error(f"❌ Failed to track options metrics: {str(e)}")

    async def _cache_collection_metric(self, metric: Dict):
        """Cache metrics in Redis for real-time access"""
        try:
            if not self.redis_client:
                return
            
            # Store latest metrics by source
            cache_key = f"mip:metrics:collection:{metric['source_type']}:{metric['source_name']}"
            
            await self.redis_client.setex(
                cache_key,
                300,  # 5 minute expiry
                str(metric)
            )
            
            # Store aggregated metrics
            daily_key = f"mip:metrics:daily:{datetime.utcnow().strftime('%Y-%m-%d')}"
            await self.redis_client.hincrby(daily_key, f"{metric['source_type']}_collections", 1)
            await self.redis_client.expire(daily_key, 86400 * 7)  # 7 day expiry
            
        except Exception as e:
            logger.error(f"❌ Failed to cache collection metric: {str(e)}")

    async def _send_critical_alert(self, subject: str, message: str):
        """Send critical alerts for important failures"""
        try:
            # In production, integrate with alerting systems (PagerDuty, Slack, etc.)
            alert_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'severity': 'CRITICAL',
                'subject': subject,
                'message': message,
                'service': 'data-ingestion'
            }
            
            logger.critical(f"🚨 CRITICAL ALERT: {subject} - {message}")
            
            # Store alert in Redis for monitoring dashboard
            if self.redis_client:
                await self.redis_client.lpush(
                    "mip:alerts:critical",
                    str(alert_data)
                )
                await self.redis_client.ltrim("mip:alerts:critical", 0, 99)  # Keep latest 100
            
        except Exception as e:
            logger.error(f"❌ Failed to send critical alert: {str(e)}")

    async def _send_circuit_breaker_alert(self, source_type: str, source_name: str):
        """Send alert when circuit breaker opens"""
        try:
            await self._send_critical_alert(
                f"Circuit Breaker Opened: {source_type}_{source_name}",
                f"Circuit breaker for {source_type}_{source_name} has opened due to repeated failures"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to send circuit breaker alert: {str(e)}")

    async def collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_METRICS.labels(metric_type='cpu_percent').set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_METRICS.labels(metric_type='memory_percent').set(memory.percent)
            SYSTEM_METRICS.labels(metric_type='memory_available_gb').set(memory.available / (1024**3))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            SYSTEM_METRICS.labels(metric_type='disk_percent').set(disk.percent)
            SYSTEM_METRICS.labels(metric_type='disk_free_gb').set(disk.free / (1024**3))
            
            # Network I/O
            network = psutil.net_io_counters()
            SYSTEM_METRICS.labels(metric_type='network_bytes_sent').set(network.bytes_sent)
            SYSTEM_METRICS.labels(metric_type='network_bytes_recv').set(network.bytes_recv)
            
            # Store in buffer
            system_metric = {
                'timestamp': datetime.utcnow().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv
            }
            
            self.metrics_buffer['system_health'].append(system_metric)
            
        except Exception as e:
            logger.error(f"❌ Failed to collect system metrics: {str(e)}")

    async def report_collection_stats(self, stats: Dict):
        """Report collection statistics"""
        try:
            # Send to CloudWatch for AWS monitoring
            await self._send_cloudwatch_metrics(stats)
            
            # Update service uptime
            uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
            SYSTEM_METRICS.labels(metric_type='uptime_seconds').set(uptime_seconds)
            
        except Exception as e:
            logger.error(f"❌ Failed to report collection stats: {str(e)}")

    async def report_circuit_breaker_states(self, states: Dict):
        """Report circuit breaker states"""
        try:
            for source_key, state_info in states.items():
                source_type, source_name = source_key.split('_', 1)
                await self.record_circuit_breaker_state(
                    source_type, source_name, state_info['state']
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to report circuit breaker states: {str(e)}")

    async def report_kafka_metrics(self, kafka_metrics: Dict):
        """Report Kafka producer metrics"""
        try:
            for metric_name, value in kafka_metrics.items():
                SYSTEM_METRICS.labels(
                    metric_type=f'kafka_{metric_name}'
                ).set(value)
                
        except Exception as e:
            logger.error(f"❌ Failed to report Kafka metrics: {str(e)}")

    async def _send_cloudwatch_metrics(self, stats: Dict):
        """Send metrics to AWS CloudWatch"""
        try:
            metric_data = []
            
            # Collection metrics
            metric_data.append({
                'MetricName': 'CollectionsCompleted',
                'Value': stats.get('collections_completed', 0),
                'Unit': 'Count',
                'Dimensions': [
                    {'Name': 'Service', 'Value': 'data-ingestion'},
                    {'Name': 'Environment', 'Value': settings.ENVIRONMENT}
                ]
            })
            
            metric_data.append({
                'MetricName': 'ValidationErrors',
                'Value': stats.get('validation_errors', 0),
                'Unit': 'Count',
                'Dimensions': [
                    {'Name': 'Service', 'Value': 'data-ingestion'},
                    {'Name': 'Environment', 'Value': settings.ENVIRONMENT}
                ]
            })
            
            metric_data.append({
                'MetricName': 'KafkaPublishErrors',
                'Value': stats.get('kafka_publish_errors', 0),
                'Unit': 'Count',
                'Dimensions': [
                    {'Name': 'Service', 'Value': 'data-ingestion'},
                    {'Name': 'Environment', 'Value': settings.ENVIRONMENT}
                ]
            })
            
            # Options-specific metrics
            metric_data.append({
                'MetricName': 'OptionsUnusualActivity',
                'Value': self.options_metrics['unusual_activity_count'],
                'Unit': 'Count',
                'Dimensions': [
                    {'Name': 'Service', 'Value': 'options-intelligence'},
                    {'Name': 'Environment', 'Value': settings.ENVIRONMENT}
                ]
            })
            
            # Send to CloudWatch
            self.cloudwatch.put_metric_data(
                Namespace='MIP/DataIngestion',
                MetricData=metric_data
            )
            
            logger.debug("📊 Sent metrics to CloudWatch")
            
        except Exception as e:
            logger.error(f"❌ Failed to send CloudWatch metrics: {str(e)}")

    async def get_metrics_summary(self) -> Dict:
        """Get comprehensive metrics summary"""
        try:
            uptime = datetime.utcnow() - self.start_time
            
            summary = {
                'service_info': {
                    'start_time': self.start_time.isoformat(),
                    'uptime_seconds': uptime.total_seconds(),
                    'environment': settings.ENVIRONMENT
                },
                'collection_metrics': {
                    'total_collections': len(self.metrics_buffer['collections']),
                    'successful_collections': len([
                        m for m in self.metrics_buffer['collections'] 
                        if m['status'] == 'success'
                    ]),
                    'failed_collections': len([
                        m for m in self.metrics_buffer['collections'] 
                        if m['status'] == 'failure'
                    ])
                },
                'options_metrics': self.options_metrics.copy(),
                'system_health': {
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent
                },
                'buffer_sizes': {
                    'collections': len(self.metrics_buffer['collections']),
                    'validations': len(self.metrics_buffer['validations']),
                    'kafka_operations': len(self.metrics_buffer['kafka_operations']),
                    'circuit_breaker_events': len(self.metrics_buffer['circuit_breaker_events'])
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get metrics summary: {str(e)}")
            return {}

    async def stop(self):
        """Stop the metrics collector"""
        try:
            # Send final metrics report
            final_summary = await self.get_metrics_summary()
            logger.info(f"📊 Final metrics summary: {final_summary}")
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("✅ Metrics collector stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping metrics collector: {str(e)}")

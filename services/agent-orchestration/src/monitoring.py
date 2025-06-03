import logging
import asyncio
from typing import Dict, Any
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Info
import time

logger = logging.getLogger(__name__)

# Prometheus metrics
ORCHESTRATOR_REQUESTS = Counter(
    'orchestrator_requests_total',
    'Total requests to orchestrator',
    ['request_type', 'user_tier', 'status']
)

ORCHESTRATOR_DURATION = Histogram(
    'orchestrator_request_duration_seconds',
    'Time spent processing orchestrator requests',
    ['request_type', 'user_tier']
)

AGENT_INVOCATIONS = Counter(
    'agent_invocations_total',
    'Total agent invocations',
    ['agent_type', 'status']
)

AGENT_DURATION = Histogram(
    'agent_processing_duration_seconds',
    'Time spent in agent processing',
    ['agent_type']
)

CIRCUIT_BREAKER_STATE = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=half_open, 2=open)',
    ['agent_type']
)

FALLBACK_TRIGGERS = Counter(
    'fallback_triggers_total',
    'Total fallback triggers',
    ['primary_agent', 'fallback_agent']
)

COST_TRACKING = Counter(
    'cost_usd_total',
    'Total cost in USD',
    ['agent_type', 'user_tier']
)

BUDGET_VIOLATIONS = Counter(
    'budget_violations_total',
    'Budget violations by type',
    ['violation_type', 'user_tier']
)

ACTIVE_USERS = Gauge(
    'active_users',
    'Number of active users',
    ['user_tier']
)

QUEUE_DEPTH = Gauge(
    'queue_depth',
    'Current queue depth by agent',
    ['agent_type']
)

SYSTEM_HEALTH = Gauge(
    'system_health_score',
    'Overall system health score (0-1)'
)

# Service info
SERVICE_INFO = Info(
    'orchestrator_service_info',
    'Information about the orchestrator service'
)

class MetricsCollector:
    """Collects and manages metrics for the orchestration service"""
    
    def __init__(self):
        self.start_time = time.time()
        self.health_scores = {}
        
        # Set service info
        SERVICE_INFO.info({
            'version': '1.0.0',
            'service': 'agent-orchestration',
            'environment': 'development'
        })
        
        logger.info("Metrics collector initialized")
    
    def record_request(
        self,
        request_type: str,
        user_tier: str,
        status: str,
        duration_seconds: float,
        cost_usd: float = 0.0
    ):
        """Record orchestrator request metrics"""
        ORCHESTRATOR_REQUESTS.labels(
            request_type=request_type,
            user_tier=user_tier,
            status=status
        ).inc()
        
        ORCHESTRATOR_DURATION.labels(
            request_type=request_type,
            user_tier=user_tier
        ).observe(duration_seconds)
        
        if cost_usd > 0:
            COST_TRACKING.labels(
                agent_type='orchestrator',
                user_tier=user_tier
            ).inc(cost_usd)
    
    def record_agent_invocation(
        self,
        agent_type: str,
        status: str,
        duration_seconds: float,
        cost_usd: float = 0.0,
        user_tier: str = 'unknown'
    ):
        """Record agent invocation metrics"""
        AGENT_INVOCATIONS.labels(
            agent_type=agent_type,
            status=status
        ).inc()
        
        AGENT_DURATION.labels(
            agent_type=agent_type
        ).observe(duration_seconds)
        
        if cost_usd > 0:
            COST_TRACKING.labels(
                agent_type=agent_type,
                user_tier=user_tier
            ).inc(cost_usd)
    
    def record_circuit_breaker_state(self, agent_type: str, state: str):
        """Record circuit breaker state"""
        state_mapping = {
            'CLOSED': 0,
            'HALF_OPEN': 1,
            'OPEN': 2
        }
        
        CIRCUIT_BREAKER_STATE.labels(
            agent_type=agent_type
        ).set(state_mapping.get(state, 0))
    
    def record_fallback_trigger(self, primary_agent: str, fallback_agent: str):
        """Record fallback trigger"""
        FALLBACK_TRIGGERS.labels(
            primary_agent=primary_agent,
            fallback_agent=fallback_agent
        ).inc()
    
    def record_budget_violation(self, violation_type: str, user_tier: str):
        """Record budget violation"""
        BUDGET_VIOLATIONS.labels(
            violation_type=violation_type,
            user_tier=user_tier
        ).inc()
    
    def update_active_users(self, counts_by_tier: Dict[str, int]):
        """Update active user counts"""
        for tier, count in counts_by_tier.items():
            ACTIVE_USERS.labels(user_tier=tier).set(count)
    
    def update_queue_depth(self, agent_type: str, depth: int):
        """Update queue depth for agent"""
        QUEUE_DEPTH.labels(agent_type=agent_type).set(depth)
    
    def update_health_score(self, component: str, score: float):
        """Update health score for component"""
        self.health_scores[component] = score
        
        # Calculate overall system health
        if self.health_scores:
            overall_health = sum(self.health_scores.values()) / len(self.health_scores)
            SYSTEM_HEALTH.set(overall_health)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics"""
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'health_scores': self.health_scores,
            'overall_health': sum(self.health_scores.values()) / len(self.health_scores) if self.health_scores else 0.0,
            'metrics_collected': True
        }

# Global metrics collector instance
metrics_collector = MetricsCollector()

def setup_prometheus_metrics():
    """Setup Prometheus metrics endpoint"""
    try:
        # Start Prometheus metrics server
        prometheus_client.start_http_server(9090)
        logger.info("Prometheus metrics server started on port 9090")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to start Prometheus metrics server: {str(e)}")
        return False

async def health_monitor_task(orchestrator):
    """Background task to monitor system health"""
    while True:
        try:
            # Check agent health
            agent_health = await orchestrator.get_agent_health()
            
            for agent_type, health in agent_health.items():
                # Calculate health score (0.0 to 1.0)
                if health.get('status') == 'healthy':
                    health_score = min(1.0, health.get('success_rate', 0.0))
                elif health.get('status') == 'degraded':
                    health_score = 0.5
                else:
                    health_score = 0.0
                
                metrics_collector.update_health_score(agent_type, health_score)
                
                # Update circuit breaker state
                cb_state = health.get('circuit_breaker_state', 'CLOSED')
                metrics_collector.record_circuit_breaker_state(agent_type, cb_state)
                
                # Update queue depth
                queue_depth = health.get('queue_depth', 0)
                metrics_collector.update_queue_depth(agent_type, queue_depth)
            
            # Update orchestrator health
            orchestrator_health = 1.0 if len(agent_health) > 0 else 0.0
            metrics_collector.update_health_score('orchestrator', orchestrator_health)
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Health monitor error: {str(e)}")
            await asyncio.sleep(60)  # Longer delay on error

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, metric_name: str, labels: Dict[str, str] = None):
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            
            # Record to appropriate metric based on name
            if self.metric_name == 'orchestrator_request':
                ORCHESTRATOR_DURATION.labels(**self.labels).observe(duration)
            elif self.metric_name == 'agent_processing':
                AGENT_DURATION.labels(**self.labels).observe(duration)

# Decorator for timing functions
def timed_operation(metric_name: str, labels: Dict[str, str] = None):
    """Decorator to time function execution"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            with PerformanceTimer(metric_name, labels):
                return await func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage
async def example_monitoring_usage():
    """Example of how to use monitoring"""
    
    # Record a request
    metrics_collector.record_request(
        request_type="comprehensive_analysis",
        user_tier="premium", 
        status="success",
        duration_seconds=1.5,
        cost_usd=0.045
    )
    
    # Record agent invocation
    metrics_collector.record_agent_invocation(
        agent_type="gpt-4-turbo",
        status="success",
        duration_seconds=0.85,
        cost_usd=0.03,
        user_tier="premium"
    )
    
    # Record fallback trigger
    metrics_collector.record_fallback_trigger(
        primary_agent="llama-7b",
        fallback_agent="finbert_explainer"
    )
    
    # Update health scores
    metrics_collector.update_health_score("gpt-4-turbo", 0.95)
    metrics_collector.update_health_score("finbert", 0.98)
    
    # Get metrics summary
    summary = metrics_collector.get_metrics_summary()
    logger.info(f"Metrics summary: {summary}")

if __name__ == "__main__":
    # Test metrics setup
    setup_prometheus_metrics()
    asyncio.run(example_monitoring_usage())

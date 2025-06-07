# Enhanced Monitoring Service for Market Intelligence Platform
import time
import logging
import asyncio
import psutil
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for agent monitoring"""
    response_time: float
    cpu_usage: float
    memory_usage: float
    tokens_used: int
    cost_incurred: float
    error_count: int
    success_rate: float

@dataclass
class SecurityAlert:
    """Security alert structure"""
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    user_id: Optional[str]
    service_name: str
    metadata: Dict[str, Any]

class EnhancedMonitoringService:
    """Enhanced monitoring with multi-agent performance tracking"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.registry = CollectorRegistry()
        self._setup_metrics()
        
        # Performance thresholds
        self.performance_thresholds = {
            'response_time_p95': 2.0,
            'error_rate': 0.05,
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'cost_per_request': 0.10
        }
        
        self.security_alerts: List[SecurityAlert] = []
        
    def _setup_metrics(self):
        """Setup Prometheus metrics for comprehensive monitoring"""
        
        self.agent_request_duration = Histogram(
            'agent_request_duration_seconds',
            'Agent request duration in seconds',
            ['service_name', 'agent_type', 'model'],
            registry=self.registry,
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.agent_requests_total = Counter(
            'agent_requests_total',
            'Total agent requests',
            ['service_name', 'agent_type', 'status', 'user_id'],
            registry=self.registry
        )
        
        self.agent_memory_usage = Gauge(
            'agent_memory_usage_bytes',
            'Agent memory usage in bytes',
            ['service_name', 'agent_type'],
            registry=self.registry
        )
        
        self.agent_cpu_usage = Gauge(
            'agent_cpu_usage_percent',
            'Agent CPU usage percentage',
            ['service_name', 'agent_type'],
            registry=self.registry
        )
        
        self.llm_tokens_total = Counter(
            'llm_tokens_total',
            'Total LLM tokens used',
            ['model', 'service_name', 'token_type'],
            registry=self.registry
        )
        
        self.agent_api_cost = Counter(
            'agent_api_cost_total',
            'Total API costs incurred',
            ['service_name', 'provider', 'model'],
            registry=self.registry
        )
        
        # Trading metrics
        self.portfolio_value = Gauge(
            'virtual_trading_portfolio_value',
            'Current portfolio value',
            ['strategy_type'],
            registry=self.registry
        )
        
        self.portfolio_pnl_daily = Gauge(
            'virtual_trading_portfolio_pnl_daily',
            'Daily portfolio P&L',
            ['strategy_type'],
            registry=self.registry
        )
        
        self.position_weight = Gauge(
            'virtual_trading_position_weight',
            'Position weight in portfolio',
            ['symbol', 'strategy_type'],
            registry=self.registry
        )
        
        self.portfolio_var_95 = Gauge(
            'virtual_trading_portfolio_var_95',
            'Portfolio 95% Value at Risk',
            ['strategy_type'],
            registry=self.registry
        )
        
        self.options_strategy_executions = Counter(
            'options_strategy_executions_total',
            'Total options strategy executions',
            ['strategy_name', 'status', 'symbol'],
            registry=self.registry
        )
        
        # Security metrics
        self.security_alerts_total = Counter(
            'security_alerts_total',
            'Total security alerts',
            ['alert_type', 'severity', 'service_name'],
            registry=self.registry
        )
        
    async def record_agent_performance(
        self, 
        service_name: str,
        agent_type: str,
        model: str,
        response: Any,
        user_id: str,
        start_time: float
    ):
        """Record comprehensive agent performance metrics"""
        
        duration = time.time() - start_time
        
        self.agent_request_duration.labels(
            service_name=service_name,
            agent_type=agent_type,
            model=model
        ).observe(duration)
        
        status = 'success' if response.success else 'error'
        self.agent_requests_total.labels(
            service_name=service_name,
            agent_type=agent_type,
            status=status,
            user_id=user_id
        ).inc()
        
        # Update system metrics
        await self._update_system_metrics(service_name, agent_type)
        
    async def _update_system_metrics(self, service_name: str, agent_type: str):
        """Update system resource metrics"""
        
        try:
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            
            self.agent_cpu_usage.labels(
                service_name=service_name,
                agent_type=agent_type
            ).set(cpu_percent)
            
            self.agent_memory_usage.labels(
                service_name=service_name,
                agent_type=agent_type
            ).set(memory_info.rss)
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
            
    async def record_security_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        user_id: Optional[str] = None,
        service_name: str = 'unknown',
        metadata: Dict[str, Any] = None
    ):
        """Record security alerts"""
        
        alert = SecurityAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            service_name=service_name,
            metadata=metadata or {}
        )
        
        self.security_alerts.append(alert)
        
        self.security_alerts_total.labels(
            alert_type=alert_type,
            severity=severity,
            service_name=service_name
        ).inc()
        
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry).decode()
        
    async def start_monitoring_loop(self):
        """Start continuous monitoring loop"""
        
        while True:
            try:
                # Clean up old security alerts
                if len(self.security_alerts) > 1000:
                    self.security_alerts = self.security_alerts[-1000:]
                    
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

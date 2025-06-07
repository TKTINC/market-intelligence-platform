# Monitoring Integration for Market Intelligence Platform
import asyncio
import logging
from typing import Dict, Any, Optional
from .monitoring.enhanced_monitoring import EnhancedMonitoringService

logger = logging.getLogger(__name__)

class MonitoringIntegration:
    """Integration layer for monitoring across all services"""
    
    def __init__(self, db_manager):
        self.monitoring_service = EnhancedMonitoringService(db_manager)
        self.background_tasks = set()
        
    async def start(self):
        """Start monitoring services"""
        
        task = asyncio.create_task(self.monitoring_service.start_monitoring_loop())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        logger.info("Enhanced monitoring started")
        
    async def stop(self):
        """Stop monitoring services"""
        
        for task in self.background_tasks:
            task.cancel()
            
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        logger.info("Enhanced monitoring stopped")
        
    async def record_agent_metrics(
        self,
        service_name: str,
        agent_type: str,
        model: str,
        response: Any,
        user_id: str,
        start_time: float,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """Record comprehensive agent metrics"""
        
        await self.monitoring_service.record_agent_performance(
            service_name=service_name,
            agent_type=agent_type,
            model=model,
            response=response,
            user_id=user_id,
            start_time=start_time
        )
        
    async def record_security_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        user_id: Optional[str] = None,
        service_name: str = 'unknown',
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record security events"""
        
        await self.monitoring_service.record_security_alert(
            alert_type=event_type,
            severity=severity,
            message=message,
            user_id=user_id,
            service_name=service_name,
            metadata=metadata
        )
        
    def get_metrics_endpoint(self) -> str:
        """Get Prometheus metrics for endpoint"""
        return self.monitoring_service.get_metrics()

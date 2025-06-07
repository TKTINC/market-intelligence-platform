# Monitoring Middleware for API Gateway
import time
import logging
from typing import Callable
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

class MonitoringMiddleware(BaseHTTPMiddleware):
    """Enhanced monitoring middleware for API Gateway"""
    
    def __init__(self, app, monitoring_service=None):
        super().__init__(app)
        self.monitoring_service = monitoring_service
        self._setup_metrics()
        
    def _setup_metrics(self):
        """Setup API Gateway specific metrics"""
        
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code', 'user_id']
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Enhanced request monitoring with security tracking"""
        
        start_time = time.time()
        endpoint = request.url.path
        method = request.method
        user_id = getattr(request.state, 'user_id', 'anonymous')
        
        try:
            # Security monitoring
            await self._check_security_patterns(request)
            
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            status_code = response.status_code
            
            self.http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                user_id=user_id
            ).inc()
            
            self.http_request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            return response
            
        except Exception as e:
            self.http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=500,
                user_id=user_id
            ).inc()
            
            if self.monitoring_service:
                await self.monitoring_service.record_security_alert(
                    alert_type='request_error',
                    severity='warning',
                    message=f"Request error on {endpoint}: {str(e)}",
                    user_id=user_id,
                    service_name='api-gateway'
                )
            
            raise
            
    async def _check_security_patterns(self, request: Request):
        """Check for suspicious security patterns"""
        
        if not self.monitoring_service:
            return
            
        user_id = getattr(request.state, 'user_id', 'anonymous')
        
        # Check for SQL injection patterns
        query_params = str(request.query_params)
        if any(pattern in query_params.lower() for pattern in ['union select', 'drop table']):
            await self.monitoring_service.record_security_alert(
                alert_type='sql_injection_attempt',
                severity='critical',
                message=f"Potential SQL injection detected: {query_params}",
                user_id=user_id,
                service_name='api-gateway'
            )

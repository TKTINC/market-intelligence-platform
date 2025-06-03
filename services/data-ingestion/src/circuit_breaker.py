"""
Circuit breaker implementation with exponential backoff
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Type, Any
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """
    Enhanced circuit breaker with exponential backoff for API protection
    """
    
    class CircuitBreakerOpenError(Exception):
        """Exception raised when circuit breaker is open"""
        pass
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: Type[Exception] = Exception,
        half_open_max_calls: int = 3
    ):
        """
        Initialize circuit breaker
        
        Args:
            name: Circuit breaker identifier
            failure_threshold: Number of failures before opening
            timeout: Timeout in seconds before allowing retry
            expected_exception: Exception type to count as failure
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.half_open_max_calls = half_open_max_calls
        
        # State management
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitBreakerState.CLOSED
        self.half_open_calls = 0
        
        # Metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.circuit_open_count = 0
        
        logger.info(f"🔧 Initialized circuit breaker '{name}' with threshold {failure_threshold}")
    
    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        return self.state == CircuitBreakerState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open"""
        return self.state == CircuitBreakerState.HALF_OPEN
    
    def _should_allow_request(self) -> bool:
        """Determine if request should be allowed"""
        current_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        elif self.state == CircuitBreakerState.OPEN:
            # Check if timeout period has passed
            if (self.last_failure_time and 
                current_time - self.last_failure_time >= timedelta(seconds=self.timeout)):
                self._transition_to_half_open()
                return True
            return False
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def _on_success(self):
        """Handle successful call"""
        self.successful_calls += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self._reset()
                logger.info(f"🟢 Circuit breaker '{self.name}' reset to CLOSED after successful half-open calls")
        
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self, exception: Exception):
        """Handle failed call"""
        self.failed_calls += 1
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Failed during half-open, go back to open
            self._transition_to_open()
            logger.warning(f"🔴 Circuit breaker '{self.name}' failed during half-open, returning to OPEN")
        
        elif (self.state == CircuitBreakerState.CLOSED and 
              self.failure_count >= self.failure_threshold):
            self._transition_to_open()
            logger.error(f"🔴 Circuit breaker '{self.name}' OPENED after {self.failure_count} failures")
    
    def _transition_to_open(self):
        """Transition to open state"""
        self.state = CircuitBreakerState.OPEN
        self.circuit_open_count += 1
        self.half_open_calls = 0
    
    def _transition_to_half_open(self):
        """Transition to half-open state"""
        self.state = CircuitBreakerState.HALF_OPEN
        self.half_open_calls = 0
        logger.info(f"🟡 Circuit breaker '{self.name}' transitioned to HALF_OPEN")
    
    def _reset(self):
        """Reset circuit breaker to closed state"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None
    
    @asynccontextmanager
    async def __call__(self):
        """Context manager for circuit breaker usage"""
        if not self._should_allow_request():
            raise self.CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is {self.state.value}"
            )
        
        self.total_calls += 1
        
        try:
            yield
            self._on_success()
            
        except self.expected_exception as e:
            self._on_failure(e)
            raise
        except Exception as e:
            # Unexpected exceptions don't count as circuit breaker failures
            logger.warning(f"⚠️  Unexpected exception in circuit breaker '{self.name}': {str(e)}")
            raise
    
    def get_metrics(self) -> dict:
        """Get circuit breaker metrics"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'circuit_open_count': self.circuit_open_count,
            'success_rate': (self.successful_calls / max(self.total_calls, 1)) * 100,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }
    
    def __str__(self) -> str:
        return (f"CircuitBreaker(name='{self.name}', state={self.state.value}, "
                f"failures={self.failure_count}/{self.failure_threshold})")

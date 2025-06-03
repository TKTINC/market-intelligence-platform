# services/agent-orchestration/src/circuit_breaker.py
import asyncio
import time
import logging
from typing import Any, Callable, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Circuit is open, requests are rejected
    HALF_OPEN = "HALF_OPEN"  # Testing if service has recovered

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreakerTimeoutError(Exception):
    """Exception raised when request times out"""
    pass

class CircuitBreaker:
    """
    Circuit breaker implementation for protecting agent calls
    Prevents cascading failures by temporarily stopping calls to failing agents
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
        name: str = "unknown"
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying half-open
            half_open_max_calls: Max calls allowed in half-open state
            name: Name for logging and monitoring
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        
        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.circuit_opened_count = 0
        
        logger.info(f"Circuit breaker '{name}' initialized with threshold={failure_threshold}")
    
    async def call(
        self,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Async function to execute
            *args: Function arguments
            timeout: Optional timeout in seconds
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            CircuitBreakerTimeoutError: When call times out
            Any exception from the wrapped function
        """
        self.total_calls += 1
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
            else:
                logger.debug(f"Circuit breaker '{self.name}' is OPEN, rejecting call")
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open. "
                    f"Failure count: {self.failure_count}, "
                    f"Threshold: {self.failure_threshold}"
                )
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                logger.debug(f"Circuit breaker '{self.name}' half-open limit reached")
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' half-open limit exceeded"
                )
        
        # Execute the function
        try:
            if timeout:
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            else:
                result = await func(*args, **kwargs)
            
            # Success - reset circuit breaker
            await self._on_success()
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Circuit breaker '{self.name}' call timed out after {timeout}s")
            await self._on_failure()
            raise CircuitBreakerTimeoutError(
                f"Call to '{self.name}' timed out after {timeout} seconds"
            )
            
        except Exception as e:
            logger.warning(f"Circuit breaker '{self.name}' call failed: {str(e)}")
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Handle successful call"""
        self.successful_calls += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            
            # If we've had enough successful calls in half-open, close the circuit
            if self.half_open_calls >= self.half_open_max_calls:
                logger.info(f"Circuit breaker '{self.name}' closing after successful recovery")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.last_failure_time = None
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on successful call
            self.failure_count = 0
            self.last_failure_time = None
    
    async def _on_failure(self):
        """Handle failed call"""
        self.failed_calls += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open state - go back to open
            logger.warning(f"Circuit breaker '{self.name}' failed in HALF_OPEN, reopening")
            self.state = CircuitState.OPEN
            
        elif self.state == CircuitState.CLOSED:
            # Check if we should open the circuit
            if self.failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker '{self.name}' opening due to {self.failure_count} failures "
                    f"(threshold: {self.failure_threshold})"
                )
                self.state = CircuitState.OPEN
                self.circuit_opened_count += 1
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout
    
    def get_state(self) -> str:
        """Get current circuit breaker state"""
        return self.state.value
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics"""
        success_rate = 0.0
        if self.total_calls > 0:
            success_rate = self.successful_calls / self.total_calls
        
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": success_rate,
            "circuit_opened_count": self.circuit_opened_count,
            "last_failure_time": self.last_failure_time,
            "time_since_last_failure": (
                time.time() - self.last_failure_time if self.last_failure_time else None
            )
        }
    
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        logger.info(f"Manually resetting circuit breaker '{self.name}'")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
    
    def force_open(self):
        """Manually force circuit breaker to open state"""
        logger.warning(f"Manually forcing circuit breaker '{self.name}' to OPEN")
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()

class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers
    Provides centralized monitoring and control
    """
    
    def __init__(self):
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
    
    def register(self, name: str, circuit_breaker: CircuitBreaker):
        """Register a circuit breaker with the manager"""
        self.circuit_breakers[name] = circuit_breaker
        logger.info(f"Registered circuit breaker: {name}")
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def get_all_stats(self) -> dict:
        """Get statistics for all circuit breakers"""
        stats = {}
        for name, cb in self.circuit_breakers.items():
            stats[name] = cb.get_stats()
        return stats
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for name, cb in self.circuit_breakers.items():
            cb.reset()
            logger.info(f"Reset circuit breaker: {name}")
    
    def get_open_circuits(self) -> list[str]:
        """Get list of circuit breakers in open state"""
        open_circuits = []
        for name, cb in self.circuit_breakers.items():
            if cb.state == CircuitState.OPEN:
                open_circuits.append(name)
        return open_circuits
    
    def get_degraded_circuits(self) -> list[str]:
        """Get list of circuit breakers with high failure rates"""
        degraded_circuits = []
        for name, cb in self.circuit_breakers.items():
            stats = cb.get_stats()
            if stats['success_rate'] < 0.8 and stats['total_calls'] > 10:
                degraded_circuits.append(name)
        return degraded_circuits

# Example usage for testing
async def example_flaky_function(should_fail: bool = False):
    """Example function that sometimes fails"""
    if should_fail:
        raise Exception("Simulated failure")
    await asyncio.sleep(0.1)  # Simulate some work
    return "Success"

async def test_circuit_breaker():
    """Test circuit breaker functionality"""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5, name="test")
    
    print("Testing successful calls...")
    for i in range(5):
        try:
            result = await cb.call(example_flaky_function, should_fail=False)
            print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1} failed: {e}")
    
    print(f"\nStats: {cb.get_stats()}")
    
    print("\nTesting failing calls...")
    for i in range(5):
        try:
            result = await cb.call(example_flaky_function, should_fail=True)
            print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1} failed: {e}")
    
    print(f"\nStats after failures: {cb.get_stats()}")
    
    print("\nTesting calls while circuit is open...")
    for i in range(3):
        try:
            result = await cb.call(example_flaky_function, should_fail=False)
            print(f"Call {i+1}: {result}")
        except CircuitBreakerOpenError as e:
            print(f"Call {i+1} rejected: {e}")
    
    print(f"\nFinal stats: {cb.get_stats()}")

if __name__ == "__main__":
    asyncio.run(test_circuit_breaker())

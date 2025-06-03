# services/agent-orchestration/tests/test_circuit_breaker.py
import pytest
import asyncio
from src.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitState

@pytest.mark.asyncio
async def test_circuit_breaker_normal_operation():
    """Test circuit breaker in normal operation"""
    cb = CircuitBreaker(failure_threshold=3, name="test")
    
    async def success_func():
        return "success"
    
    result = await cb.call(success_func)
    assert result == "success"
    assert cb.state == CircuitState.CLOSED

@pytest.mark.asyncio
async def test_circuit_breaker_failure_threshold():
    """Test circuit breaker opens after failure threshold"""
    cb = CircuitBreaker(failure_threshold=3, name="test")
    
    async def failure_func():
        raise Exception("Test failure")
    
    # Trigger failures to reach threshold
    for _ in range(3):
        with pytest.raises(Exception):
            await cb.call(failure_func)
    
    assert cb.state == CircuitState.OPEN
    
    # Next call should be rejected immediately
    with pytest.raises(CircuitBreakerOpenError):
        await cb.call(failure_func)

@pytest.mark.asyncio
async def test_circuit_breaker_timeout():
    """Test circuit breaker timeout functionality"""
    cb = CircuitBreaker(failure_threshold=2, name="test")
    
    async def slow_func():
        await asyncio.sleep(2)
        return "slow"
    
    with pytest.raises(Exception):  # Should timeout
        await cb.call(slow_func, timeout=0.1)

@pytest.mark.asyncio
async def test_circuit_breaker_recovery():
    """Test circuit breaker recovery after timeout"""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1, name="test")
    
    async def failure_func():
        raise Exception("Test failure")
    
    async def success_func():
        return "success"
    
    # Trigger failures to open circuit
    for _ in range(2):
        with pytest.raises(Exception):
            await cb.call(failure_func)
    
    assert cb.state == CircuitState.OPEN
    
    # Wait for recovery timeout
    await asyncio.sleep(1.1)
    
    # Should transition to half-open and then closed on success
    result = await cb.call(success_func)
    assert result == "success"

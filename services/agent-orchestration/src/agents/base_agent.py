# services/agent-orchestration/src/agents/base_agent.py
import asyncio
import logging
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import prometheus_client

logger = logging.getLogger(__name__)

@dataclass
class AgentResult:
    """Standard result format for all agents"""
    data: Dict[str, Any]
    confidence: float
    processing_time_ms: int
    model_version: str
    agent_type: str
    metadata: Optional[Dict[str, Any]] = None

class AgentError(Exception):
    """Base exception for agent errors"""
    pass

class AgentTimeoutError(AgentError):
    """Raised when agent processing times out"""
    pass

class AgentValidationError(AgentError):
    """Raised when input validation fails"""
    pass

class BaseAgent(ABC):
    """
    Abstract base class for all AI agents
    Provides common functionality for monitoring, validation, and result formatting
    """
    
    def __init__(self, agent_type: str, model_version: str = "1.0.0"):
        self.agent_type = agent_type
        self.model_version = model_version
        self.is_initialized = False
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        # Prometheus metrics
        self.request_counter = prometheus_client.Counter(
            f'{agent_type}_requests_total',
            f'Total requests to {agent_type} agent',
            ['status']
        )
        
        self.processing_time_histogram = prometheus_client.Histogram(
            f'{agent_type}_processing_seconds',
            f'Processing time for {agent_type} agent'
        )
        
        self.error_counter = prometheus_client.Counter(
            f'{agent_type}_errors_total',
            f'Total errors from {agent_type} agent',
            ['error_type']
        )
        
        logger.info(f"Initialized {agent_type} agent v{model_version}")
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent (load models, connect to services, etc.)"""
        pass
    
    @abstractmethod
    async def _process_internal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal processing method to be implemented by each agent
        
        Args:
            payload: Input data for processing
            
        Returns:
            Processing result as dictionary
        """
        pass
    
    @abstractmethod
    def validate_input(self, payload: Dict[str, Any]) -> bool:
        """
        Validate input payload
        
        Args:
            payload: Input data to validate
            
        Returns:
            True if valid, raises AgentValidationError if invalid
        """
        pass
    
    async def process(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> AgentResult:
        """
        Main processing method with monitoring and error handling
        
        Args:
            payload: Input data for processing
            timeout: Optional timeout in seconds
            
        Returns:
            AgentResult with standardized output format
        """
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Validate input
            if not self.validate_input(payload):
                raise AgentValidationError(f"Invalid input for {self.agent_type} agent")
            
            # Check if agent is initialized
            if not self.is_initialized:
                raise AgentError(f"{self.agent_type} agent not initialized")
            
            # Process with timeout if specified
            if timeout:
                result_data = await asyncio.wait_for(
                    self._process_internal(payload),
                    timeout=timeout
                )
            else:
                result_data = await self._process_internal(payload)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            self.total_processing_time += processing_time_ms
            
            # Create standardized result
            result = AgentResult(
                data=result_data,
                confidence=result_data.get('confidence', 1.0),
                processing_time_ms=processing_time_ms,
                model_version=self.model_version,
                agent_type=self.agent_type,
                metadata={
                    'request_id': payload.get('request_id'),
                    'timestamp': datetime.utcnow().isoformat(),
                    'processing_node': 'node-1'  # Could be dynamic
                }
            )
            
            # Update metrics
            self.request_counter.labels(status='success').inc()
            self.processing_time_histogram.observe(processing_time_ms / 1000.0)
            
            logger.debug(
                f"{self.agent_type} agent processed request in {processing_time_ms}ms"
            )
            
            return result
            
        except asyncio.TimeoutError:
            self.error_count += 1
            self.error_counter.labels(error_type='timeout').inc()
            self.request_counter.labels(status='timeout').inc()
            
            logger.error(f"{self.agent_type} agent timed out after {timeout}s")
            raise AgentTimeoutError(f"{self.agent_type} agent timed out")
            
        except AgentValidationError as e:
            self.error_count += 1
            self.error_counter.labels(error_type='validation').inc()
            self.request_counter.labels(status='validation_error').inc()
            
            logger.error(f"{self.agent_type} agent validation error: {str(e)}")
            raise
            
        except Exception as e:
            self.error_count += 1
            self.error_counter.labels(error_type='processing').inc()
            self.request_counter.labels(status='error').inc()
            
            logger.error(f"{self.agent_type} agent processing error: {str(e)}")
            raise AgentError(f"{self.agent_type} agent failed: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for the agent
        
        Returns:
            Health status information
        """
        try:
            # Perform a simple test operation
            test_payload = self.get_test_payload()
            
            start_time = time.time()
            await self._process_internal(test_payload)
            response_time = (time.time() - start_time) * 1000
            
            avg_processing_time = (
                self.total_processing_time / self.request_count
                if self.request_count > 0 else 0
            )
            
            error_rate = (
                self.error_count / self.request_count
                if self.request_count > 0 else 0
            )
            
            return {
                'status': 'healthy',
                'agent_type': self.agent_type,
                'model_version': self.model_version,
                'is_initialized': self.is_initialized,
                'request_count': self.request_count,
                'error_count': self.error_count,
                'error_rate': error_rate,
                'avg_processing_time_ms': avg_processing_time,
                'last_health_check_ms': response_time,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"{self.agent_type} health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'agent_type': self.agent_type,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_test_payload(self) -> Dict[str, Any]:
        """
        Get a test payload for health checks
        Should be overridden by each agent type
        """
        return {
            'test': True,
            'data': 'health_check'
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        avg_processing_time = (
            self.total_processing_time / self.request_count
            if self.request_count > 0 else 0
        )
        
        error_rate = (
            self.error_count / self.request_count
            if self.request_count > 0 else 0
        )
        
        return {
            'agent_type': self.agent_type,
            'model_version': self.model_version,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'avg_processing_time_ms': avg_processing_time,
            'total_processing_time_ms': self.total_processing_time,
            'is_initialized': self.is_initialized
        }
    
    async def warm_up(self, iterations: int = 3) -> Dict[str, Any]:
        """
        Warm up the agent with test requests
        Useful for loading models into memory
        """
        logger.info(f"Warming up {self.agent_type} agent with {iterations} iterations")
        
        warm_up_times = []
        test_payload = self.get_test_payload()
        
        for i in range(iterations):
            try:
                start_time = time.time()
                await self._process_internal(test_payload)
                warm_up_time = (time.time() - start_time) * 1000
                warm_up_times.append(warm_up_time)
                
                logger.debug(f"Warm-up iteration {i+1}: {warm_up_time:.2f}ms")
                
            except Exception as e:
                logger.error(f"Warm-up iteration {i+1} failed: {str(e)}")
        
        if warm_up_times:
            avg_warm_up_time = sum(warm_up_times) / len(warm_up_times)
            logger.info(f"{self.agent_type} warm-up completed. Avg time: {avg_warm_up_time:.2f}ms")
            
            return {
                'status': 'completed',
                'iterations': len(warm_up_times),
                'avg_time_ms': avg_warm_up_time,
                'times_ms': warm_up_times
            }
        else:
            return {
                'status': 'failed',
                'iterations': 0,
                'error': 'All warm-up iterations failed'
            }
    
    def reset_stats(self) -> None:
        """Reset agent statistics"""
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        logger.info(f"Reset statistics for {self.agent_type} agent")
    
    async def shutdown(self) -> None:
        """
        Shutdown the agent and cleanup resources
        Should be overridden by agents that need cleanup
        """
        logger.info(f"Shutting down {self.agent_type} agent")
        self.is_initialized = False

class MockAgent(BaseAgent):
    """
    Mock agent for testing purposes
    Simulates processing with configurable delays and failure rates
    """
    
    def __init__(
        self,
        agent_type: str = "mock",
        processing_delay: float = 0.1,
        failure_rate: float = 0.0,
        model_version: str = "1.0.0"
    ):
        super().__init__(agent_type, model_version)
        self.processing_delay = processing_delay
        self.failure_rate = failure_rate
    
    async def initialize(self) -> None:
        """Initialize mock agent"""
        await asyncio.sleep(0.1)  # Simulate initialization time
        self.is_initialized = True
        logger.info(f"Mock agent {self.agent_type} initialized")
    
    def validate_input(self, payload: Dict[str, Any]) -> bool:
        """Validate input - mock always accepts input"""
        return isinstance(payload, dict)
    
    async def _process_internal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Mock processing with configurable delay and failure rate"""
        
        # Simulate processing time
        await asyncio.sleep(self.processing_delay)
        
        # Simulate random failures
        import random
        if random.random() < self.failure_rate:
            raise Exception(f"Simulated failure in {self.agent_type} agent")
        
        # Return mock result
        return {
            'result': f"Processed by {self.agent_type}",
            'input_size': len(str(payload)),
            'confidence': 0.95,
            'mock_data': True
        }
    
    def get_test_payload(self) -> Dict[str, Any]:
        """Get test payload for mock agent"""
        return {
            'test': True,
            'mock_input': 'health_check_data'
        }

# Example usage and testing
async def test_base_agent():
    """Test the base agent functionality"""
    
    # Create mock agent
    agent = MockAgent(
        agent_type="test_mock",
        processing_delay=0.05,
        failure_rate=0.1
    )
    
    # Initialize
    await agent.initialize()
    
    # Test successful processing
    payload = {'data': 'test_input', 'value': 123}
    
    try:
        result = await agent.process(payload)
        print(f"Result: {result}")
        print(f"Stats: {agent.get_stats()}")
        
    except Exception as e:
        print(f"Processing failed: {e}")
    
    # Test health check
    health = await agent.health_check()
    print(f"Health: {health}")
    
    # Test warm-up
    warm_up_result = await agent.warm_up(iterations=5)
    print(f"Warm-up: {warm_up_result}")
    
    # Test multiple requests
    print("\nTesting multiple requests...")
    for i in range(10):
        try:
            result = await agent.process({'request': i})
            print(f"Request {i}: Success")
        except Exception as e:
            print(f"Request {i}: Failed - {e}")
    
    print(f"\nFinal stats: {agent.get_stats()}")
    
    # Shutdown
    await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(test_base_agent())

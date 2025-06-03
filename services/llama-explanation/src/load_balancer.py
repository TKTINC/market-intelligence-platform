# services/llama-explanation/src/load_balancer.py
import asyncio
import logging
import time
import heapq
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)

class Priority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class Request:
    """Request data structure with priority handling"""
    id: str
    context: Dict[str, Any]
    max_tokens: int
    temperature: float
    system_prompt: Optional[str]
    priority: Priority
    created_at: float
    timeout: float = 30.0
    future: Optional[asyncio.Future] = field(default=None)
    
    def __lt__(self, other):
        """Priority queue ordering (higher priority first, then FIFO)"""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at

class LoadBalancer:
    """
    Load balancer for Llama explanation requests
    Handles request queuing, priority, and concurrent execution management
    """
    
    def __init__(self, llama_engine, max_concurrent: int = 3, max_queue_size: int = 100):
        self.llama_engine = llama_engine
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        
        # Request management
        self.request_queue: List[Request] = []
        self.active_requests: Dict[str, Request] = {}
        self.request_semaphore = asyncio.Semaphore(max_concurrent)
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'timeout_requests': 0,
            'queue_rejections': 0,
            'average_wait_time': 0.0,
            'average_processing_time': 0.0
        }
        
        # Priority mapping
        self.priority_map = {
            'low': Priority.LOW,
            'normal': Priority.NORMAL,
            'high': Priority.HIGH,
            'urgent': Priority.URGENT
        }
        
        # Background task management
        self.queue_processor_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info(f"Load balancer initialized: max_concurrent={max_concurrent}, max_queue={max_queue_size}")
    
    async def start(self):
        """Start the load balancer background tasks"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.queue_processor_task = asyncio.create_task(self._queue_processor())
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_requests())
        
        logger.info("Load balancer started")
    
    async def stop(self):
        """Stop the load balancer and cleanup"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        if self.queue_processor_task:
            self.queue_processor_task.cancel()
            try:
                await self.queue_processor_task
            except asyncio.CancelledError:
                pass
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel any remaining requests
        for request in self.request_queue:
            if request.future and not request.future.done():
                request.future.cancel()
        
        for request in self.active_requests.values():
            if request.future and not request.future.done():
                request.future.cancel()
        
        logger.info("Load balancer stopped")
    
    async def generate_explanation(
        self,
        context: Dict[str, Any],
        max_tokens: int = 256,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
        priority: str = "normal",
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Generate explanation through load balancer"""
        
        if not self.is_running:
            raise RuntimeError("Load balancer not running")
        
        # Check queue capacity
        if len(self.request_queue) >= self.max_queue_size:
            self.stats['queue_rejections'] += 1
            raise RuntimeError(f"Request queue full (max {self.max_queue_size})")
        
        # Create request
        request = Request(
            id=str(uuid.uuid4()),
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            priority=self.priority_map.get(priority, Priority.NORMAL),
            created_at=time.time(),
            timeout=timeout,
            future=asyncio.Future()
        )
        
        # Add to queue
        await self._enqueue_request(request)
        
        try:
            # Wait for completion
            result = await asyncio.wait_for(request.future, timeout=timeout)
            
            # Update statistics
            self.stats['completed_requests'] += 1
            self._update_wait_time_stats(request)
            
            return result
            
        except asyncio.TimeoutError:
            self.stats['timeout_requests'] += 1
            self._remove_request(request)
            raise RuntimeError(f"Request {request.id} timed out after {timeout}s")
        
        except Exception as e:
            self.stats['failed_requests'] += 1
            self._remove_request(request)
            raise
    
    async def generate_explanations_batch(
        self,
        requests: List[Dict[str, Any]],
        default_priority: str = "normal"
    ) -> List[Dict[str, Any]]:
        """Generate multiple explanations in batch"""
        
        if not requests:
            return []
        
        # Create futures for all requests
        tasks = []
        
        for req_data in requests:
            task = asyncio.create_task(
                self.generate_explanation(
                    context=req_data['context'],
                    max_tokens=req_data.get('max_tokens', 256),
                    temperature=req_data.get('temperature', 0.1),
                    system_prompt=req_data.get('system_prompt'),
                    priority=req_data.get('priority', default_priority),
                    timeout=req_data.get('timeout', 30.0)
                )
            )
            tasks.append(task)
        
        # Wait for all to complete (with individual timeouts)
        results = []
        
        for i, task in enumerate(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'request_index': i
                })
        
        return results
    
    async def _enqueue_request(self, request: Request):
        """Add request to priority queue"""
        
        self.stats['total_requests'] += 1
        
        # Add to priority queue
        heapq.heappush(self.request_queue, request)
        
        logger.debug(f"Enqueued request {request.id} with priority {request.priority.name}")
    
    def _remove_request(self, request: Request):
        """Remove request from tracking"""
        
        # Remove from active requests
        if request.id in self.active_requests:
            del self.active_requests[request.id]
        
        # Note: We don't remove from request_queue as it's managed by heapq
        # The queue processor will skip completed/cancelled requests
    
    async def _queue_processor(self):
        """Background task to process request queue"""
        
        while self.is_running:
            try:
                # Check if we can process more requests
                if not self.request_queue:
                    await asyncio.sleep(0.1)
                    continue
                
                # Try to acquire semaphore (non-blocking check)
                try:
                    await asyncio.wait_for(self.request_semaphore.acquire(), timeout=0.1)
                except asyncio.TimeoutError:
                    # No available slots, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                
                # Get next request from queue
                if self.request_queue:
                    request = heapq.heappop(self.request_queue)
                    
                    # Check if request is still valid
                    if request.future.done() or request.future.cancelled():
                        self.request_semaphore.release()
                        continue
                    
                    # Check timeout
                    if time.time() - request.created_at > request.timeout:
                        request.future.set_exception(
                            RuntimeError(f"Request {request.id} expired in queue")
                        )
                        self.request_semaphore.release()
                        self.stats['timeout_requests'] += 1
                        continue
                    
                    # Process request
                    self.active_requests[request.id] = request
                    asyncio.create_task(self._process_request(request))
                else:
                    self.request_semaphore.release()
                
            except Exception as e:
                logger.error(f"Queue processor error: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _process_request(self, request: Request):
        """Process individual request"""
        
        start_time = time.time()
        
        try:
            logger.debug(f"Processing request {request.id}")
            
            # Generate explanation
            result = await self.llama_engine.generate_explanation(
                context=request.context,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system_prompt=request.system_prompt
            )
            
            # Add processing metadata
            processing_time = time.time() - start_time
            result['load_balancer_info'] = {
                'request_id': request.id,
                'priority': request.priority.name,
                'queue_wait_time': start_time - request.created_at,
                'processing_time': processing_time
            }
            
            # Complete the request
            if not request.future.done():
                request.future.set_result(result)
            
            # Update statistics
            self._update_processing_time_stats(processing_time)
            
            logger.debug(f"Completed request {request.id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Request {request.id} failed: {str(e)}")
            
            if not request.future.done():
                request.future.set_exception(e)
        
        finally:
            # Cleanup
            self._remove_request(request)
            self.request_semaphore.release()
    
    async def _cleanup_expired_requests(self):
        """Background task to cleanup expired requests"""
        
        while self.is_running:
            try:
                current_time = time.time()
                expired_requests = []
                
                # Check active requests for expiration
                for request_id, request in list(self.active_requests.items()):
                    if current_time - request.created_at > request.timeout:
                        expired_requests.append(request)
                
                # Cancel expired requests
                for request in expired_requests:
                    if not request.future.done():
                        request.future.set_exception(
                            RuntimeError(f"Request {request.id} expired")
                        )
                    self._remove_request(request)
                    self.stats['timeout_requests'] += 1
                
                if expired_requests:
                    logger.warning(f"Cleaned up {len(expired_requests)} expired requests")
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Cleanup task error: {str(e)}")
                await asyncio.sleep(10.0)
    
    def _update_wait_time_stats(self, request: Request):
        """Update average wait time statistics"""
        
        wait_time = time.time() - request.created_at
        
        if self.stats['completed_requests'] > 1:
            # Running average
            n = self.stats['completed_requests']
            self.stats['average_wait_time'] = (
                (self.stats['average_wait_time'] * (n - 1) + wait_time) / n
            )
        else:
            self.stats['average_wait_time'] = wait_time
    
    def _update_processing_time_stats(self, processing_time: float):
        """Update average processing time statistics"""
        
        if self.stats['completed_requests'] > 1:
            # Running average
            n = self.stats['completed_requests']
            self.stats['average_processing_time'] = (
                (self.stats['average_processing_time'] * (n - 1) + processing_time) / n
            )
        else:
            self.stats['average_processing_time'] = processing_time
    
    def get_queue_depth(self) -> int:
        """Get current queue depth"""
        return len(self.request_queue)
    
    def get_active_requests_count(self) -> int:
        """Get number of active requests"""
        return len(self.active_requests)
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Get detailed queue information"""
        
        # Count requests by priority
        priority_counts = defaultdict(int)
        for request in self.request_queue:
            priority_counts[request.priority.name] += 1
        
        return {
            'queue_depth': len(self.request_queue),
            'active_requests': len(self.active_requests),
            'max_concurrent': self.max_concurrent,
            'max_queue_size': self.max_queue_size,
            'available_slots': self.request_semaphore._value,
            'priority_breakdown': dict(priority_counts)
        }
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed load balancer statistics"""
        
        return {
            'statistics': dict(self.stats),
            'queue_info': self.get_queue_info(),
            'configuration': {
                'max_concurrent': self.max_concurrent,
                'max_queue_size': self.max_queue_size,
                'is_running': self.is_running
            },
            'performance': {
                'throughput_requests_per_minute': (
                    self.stats['completed_requests'] / 
                    max(1, self.stats['average_processing_time'] / 60)
                ) if self.stats['average_processing_time'] > 0 else 0,
                'success_rate': (
                    self.stats['completed_requests'] / 
                    max(1, self.stats['total_requests'])
                ) if self.stats['total_requests'] > 0 else 0
            }
        }

class RequestRateLimiter:
    """Rate limiter for requests by user/IP"""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests_per_minute = max_requests_per_minute
        self.request_history: Dict[str, List[float]] = defaultdict(list)
        self.cleanup_interval = 60.0  # Clean up every minute
        self.last_cleanup = time.time()
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for identifier"""
        
        current_time = time.time()
        
        # Cleanup old entries
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time
        
        # Get request history for identifier
        history = self.request_history[identifier]
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60.0
        self.request_history[identifier] = [
            timestamp for timestamp in history if timestamp > cutoff_time
        ]
        
        # Check if under limit
        if len(self.request_history[identifier]) < self.max_requests_per_minute:
            self.request_history[identifier].append(current_time)
            return True
        
        return False
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove old entries from all identifiers"""
        
        cutoff_time = current_time - 60.0
        
        for identifier in list(self.request_history.keys()):
            self.request_history[identifier] = [
                timestamp for timestamp in self.request_history[identifier]
                if timestamp > cutoff_time
            ]
            
            # Remove empty entries
            if not self.request_history[identifier]:
                del self.request_history[identifier]

# Example usage and testing
async def test_load_balancer():
    """Test load balancer functionality"""
    
    # Mock Llama engine for testing
    class MockLlamaEngine:
        async def generate_explanation(self, **kwargs):
            await asyncio.sleep(0.5)  # Simulate processing time
            return {
                'explanation': 'Test explanation',
                'tokens_used': 100,
                'confidence_score': 0.85
            }
    
    # Create load balancer
    mock_engine = MockLlamaEngine()
    load_balancer = LoadBalancer(mock_engine, max_concurrent=2, max_queue_size=10)
    
    await load_balancer.start()
    
    try:
        # Test single request
        result = await load_balancer.generate_explanation(
            context={'analysis_type': 'test', 'symbol': 'AAPL'},
            priority='normal'
        )
        print(f"Single request result: {result}")
        
        # Test batch requests
        batch_requests = [
            {'context': {'analysis_type': 'test', 'symbol': f'TEST{i}'}}
            for i in range(5)
        ]
        
        batch_results = await load_balancer.generate_explanations_batch(batch_requests)
        print(f"Batch results: {len(batch_results)} completed")
        
        # Get statistics
        stats = load_balancer.get_detailed_stats()
        print(f"Load balancer stats: {stats}")
        
    finally:
        await load_balancer.stop()

if __name__ == "__main__":
    asyncio.run(test_load_balancer())

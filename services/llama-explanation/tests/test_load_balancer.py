# services/llama-explanation/tests/test_load_balancer.py
import pytest
import asyncio
from unittest.mock import AsyncMock
from src.load_balancer import LoadBalancer, Priority

@pytest.fixture
async def mock_llama_engine():
    """Mock LlamaEngine for testing"""
    engine = AsyncMock()
    engine.generate_explanation.return_value = {
        'explanation': 'Test explanation',
        'tokens_used': 100,
        'confidence_score': 0.85
    }
    return engine

@pytest.fixture
async def load_balancer(mock_llama_engine):
    """Create LoadBalancer for testing"""
    lb = LoadBalancer(mock_llama_engine, max_concurrent=2, max_queue_size=10)
    await lb.start()
    yield lb
    await lb.stop()

@pytest.mark.asyncio
async def test_load_balancer_single_request(load_balancer):
    """Test single request through load balancer"""
    result = await load_balancer.generate_explanation(
        context={'test': 'data'},
        priority='normal'
    )
    
    assert 'explanation' in result
    assert 'load_balancer_info' in result
    assert result['load_balancer_info']['priority'] == 'NORMAL'

@pytest.mark.asyncio
async def test_load_balancer_priority_queue(load_balancer):
    """Test priority queue functionality"""
    # Submit requests with different priorities
    tasks = []
    
    # Low priority requests
    for i in range(3):
        task = asyncio.create_task(
            load_balancer.generate_explanation(
                context={'request': f'low_{i}'},
                priority='low'
            )
        )
        tasks.append(task)
    
    # High priority request
    high_priority_task = asyncio.create_task(
        load_balancer.generate_explanation(
            context={'request': 'high'},
            priority='high'
        )
    )
    tasks.append(high_priority_task)
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 4
    assert all('explanation' in result for result in results)

@pytest.mark.asyncio
async def test_load_balancer_batch_requests(load_balancer):
    """Test batch request processing"""
    batch_requests = [
        {'context': {'request': f'batch_{i}'}}
        for i in range(5)
    ]
    
    results = await load_balancer.generate_explanations_batch(batch_requests)
    
    assert len(results) == 5
    assert all('explanation' in result for result in results if 'error' not in result)

@pytest.mark.asyncio
async def test_load_balancer_queue_overflow(load_balancer):
    """Test queue overflow handling"""
    # Fill the queue beyond capacity
    tasks = []
    
    # Submit more requests than queue can handle
    for i in range(15):  # Queue max is 10
        try:
            task = asyncio.create_task(
                load_balancer.generate_explanation(
                    context={'request': f'overflow_{i}'}
                )
            )
            tasks.append(task)
        except RuntimeError as e:
            # Expected for requests beyond queue capacity
            assert "queue full" in str(e).lower()
    
    # Some tasks should succeed, others should be rejected
    if tasks:
        completed_tasks = []
        for task in tasks:
            try:
                result = await task
                completed_tasks.append(result)
            except RuntimeError:
                pass  # Expected for overflow
        
        # Should have some successful completions
        assert len(completed_tasks) > 0

@pytest.mark.asyncio
async def test_load_balancer_timeout(load_balancer):
    """Test request timeout handling"""
    # Mock engine to simulate slow response
    load_balancer.llama_engine.generate_explanation = AsyncMock(
        side_effect=lambda **kwargs: asyncio.sleep(2)  # 2 second delay
    )
    
    with pytest.raises(RuntimeError, match="timed out"):
        await load_balancer.generate_explanation(
            context={'test': 'timeout'},
            timeout=0.5  # 0.5 second timeout
        )

@pytest.mark.asyncio
async def test_load_balancer_stats(load_balancer):
    """Test load balancer statistics"""
    # Generate some requests
    await load_balancer.generate_explanation(context={'test': '1'})
    await load_balancer.generate_explanation(context={'test': '2'})
    
    stats = load_balancer.get_detailed_stats()
    
    assert 'statistics' in stats
    assert 'queue_info' in stats
    assert 'performance' in stats
    assert stats['statistics']['total_requests'] >= 2
    assert stats['statistics']['completed_requests'] >= 2

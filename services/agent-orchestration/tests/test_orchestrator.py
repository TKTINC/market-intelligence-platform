# services/agent-orchestration/tests/test_orchestrator.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.orchestrator import AgentOrchestrator, WorkflowResult, UserPreferences
from src.circuit_breaker import CircuitBreaker

@pytest.fixture
async def mock_redis():
    """Mock Redis client"""
    redis_mock = AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.setex.return_value = True
    redis_mock.incrbyfloat.return_value = 1.0
    return redis_mock

@pytest.fixture
async def orchestrator(mock_redis):
    """Create orchestrator instance for testing"""
    orchestrator = AgentOrchestrator(mock_redis)
    
    # Mock database
    orchestrator.db = AsyncMock()
    orchestrator.db.initialize = AsyncMock()
    orchestrator.db.get_user_llm_settings.return_value = None
    orchestrator.db.log_agent_performance = AsyncMock()
    
    # Mock agents
    for agent_name in orchestrator.agent_configs:
        mock_agent = AsyncMock()
        mock_agent.process.return_value = {
            'result': f'mock_result_{agent_name}',
            'confidence': 0.85
        }
        mock_agent.initialize = AsyncMock()
        mock_agent.health_check = AsyncMock(return_value={'status': 'healthy'})
        orchestrator.agents[agent_name] = mock_agent
    
    await orchestrator.initialize()
    return orchestrator

@pytest.mark.asyncio
async def test_orchestrator_initialization(orchestrator):
    """Test orchestrator initialization"""
    assert len(orchestrator.agents) > 0
    assert len(orchestrator.circuit_breakers) > 0
    assert orchestrator.routing_table is not None

@pytest.mark.asyncio
async def test_execute_workflow_success(orchestrator):
    """Test successful workflow execution"""
    result = await orchestrator.execute_workflow(
        request_type="news_analysis",
        user_id="test_user",
        payload={"text": "Apple stock is doing well"}
    )
    
    assert isinstance(result, WorkflowResult)
    assert result.analysis_id is not None
    assert 'sentiment' in result.agent_outputs
    assert result.total_cost >= 0
    assert result.duration_ms > 0

@pytest.mark.asyncio
async def test_route_request_free_tier(orchestrator):
    """Test request routing for free tier users"""
    agents = orchestrator._route_request("options_recommendation", "free")
    
    # Free tier should get lite models
    assert 'strategy_lite' in agents or 'options_strategy' in agents

@pytest.mark.asyncio
async def test_route_request_premium_tier(orchestrator):
    """Test request routing for premium tier users"""
    agents = orchestrator._route_request("options_recommendation", "premium")
    
    # Premium tier should get advanced models
    assert 'options_strategy' in agents
    assert 'risk_analysis' in agents

@pytest.mark.asyncio
async def test_circuit_breaker_integration(orchestrator):
    """Test circuit breaker integration"""
    agent_name = "sentiment"
    circuit_breaker = orchestrator.circuit_breakers[agent_name]
    
    # Simulate failures to open circuit breaker
    for _ in range(6):  # More than threshold
        circuit_breaker.on_failure()
    
    assert circuit_breaker.state.value == "OPEN"

@pytest.mark.asyncio
async def test_cost_tracking(orchestrator):
    """Test cost tracking functionality"""
    # Mock cost tracker
    orchestrator.cost_tracker.check_user_budget = AsyncMock(return_value=True)
    orchestrator.cost_tracker.record_cost = AsyncMock(return_value=True)
    
    result = await orchestrator.execute_workflow(
        request_type="price_prediction",
        user_id="test_user",
        payload={"symbol": "AAPL", "current_price": 150.0}
    )
    
    assert result.total_cost >= 0

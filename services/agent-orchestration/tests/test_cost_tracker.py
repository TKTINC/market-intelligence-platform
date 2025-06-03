# services/agent-orchestration/tests/test_cost_tracker.py
import pytest
from datetime import datetime, date
from unittest.mock import AsyncMock
from src.cost_tracker import CostTracker, CostEvent, UserBudget

@pytest.fixture
async def cost_tracker():
    """Create cost tracker for testing"""
    tracker = CostTracker()
    
    # Mock Redis and database
    tracker.redis = AsyncMock()
    tracker.db = AsyncMock()
    tracker.db.initialize = AsyncMock()
    
    await tracker.initialize()
    return tracker

@pytest.mark.asyncio
async def test_calculate_cost(cost_tracker):
    """Test cost calculation"""
    cost = await cost_tracker.calculate_cost("gpt-4-turbo", 1000)
    assert cost == 0.03  # $0.03 per 1K tokens
    
    cost_finbert = await cost_tracker.calculate_cost("finbert", 1000)
    assert cost_finbert == 0.0001  # Much cheaper

@pytest.mark.asyncio
async def test_record_cost_event(cost_tracker):
    """Test recording cost events"""
    cost_event = CostEvent(
        user_id="test_user",
        agent_type="gpt-4-turbo",
        cost_usd=0.045,
        tokens_used=1500,
        timestamp=datetime.utcnow(),
        analysis_id="test_analysis"
    )
    
    result = await cost_tracker.record_cost(
        cost_event.user_id,
        cost_event.agent_type,
        cost_event.cost_usd,
        cost_event.tokens_used,
        cost_event.analysis_id
    )
    
    assert result is True

@pytest.mark.asyncio
async def test_budget_enforcement(cost_tracker):
    """Test budget enforcement"""
    # Mock user budget
    budget = UserBudget(
        user_id="test_user",
        monthly_limit=100.0,
        daily_limit=10.0,
        per_request_limit=2.0,
        user_tier="premium"
    )
    
    cost_tracker.get_user_budget = AsyncMock(return_value=budget)
    cost_tracker.get_daily_usage = AsyncMock(return_value=5.0)
    
    # Should pass budget check
    can_afford = await cost_tracker.check_user_budget("test_user", 1.0)
    assert can_afford is True
    
    # Should fail budget check
    can_afford = await cost_tracker.check_user_budget("test_user", 10.0)
    assert can_afford is False

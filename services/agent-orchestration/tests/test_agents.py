# services/agent-orchestration/tests/test_agents.py
import pytest
from src.agents.finbert_agent import FinBertAgent
from src.agents.gpt4_strategy_agent import GPT4StrategyAgent
from src.agents.llama_explanation_agent import LlamaExplanationAgent

@pytest.mark.asyncio
async def test_finbert_agent():
    """Test FinBERT agent"""
    agent = FinBertAgent()
    await agent.initialize()
    
    payload = {"text": "Apple stock is performing well"}
    result = await agent.process(payload)
    
    assert result.data['sentiment'] in ['positive', 'negative', 'neutral']
    assert 'confidence' in result.data
    assert result.processing_time_ms > 0

@pytest.mark.asyncio
async def test_gpt4_strategy_agent():
    """Test GPT-4 strategy agent"""
    agent = GPT4StrategyAgent()
    await agent.initialize()
    
    payload = {
        "symbol": "AAPL",
        "current_price": 150.0,
        "iv_rank": 65
    }
    result = await agent.process(payload)
    
    assert 'strategy' in result.data
    assert 'reasoning' in result.data
    assert 'risk_assessment' in result.data

@pytest.mark.asyncio
async def test_llama_explanation_agent():
    """Test Llama explanation agent"""
    agent = LlamaExplanationAgent()
    await agent.initialize()
    
    payload = {
        "context": {
            "analysis_type": "sentiment",
            "symbol": "AAPL",
            "sentiment_score": 0.75
        }
    }
    result = await agent.process(payload)
    
    assert 'explanation' in result.data
    assert 'key_points' in result.data
    assert result.data['confidence'] > 0

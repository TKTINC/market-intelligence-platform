# services/llama-explanation/tests/test_llama_engine.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.llama_engine import LlamaEngine

@pytest.fixture
async def llama_engine():
    """Create LlamaEngine instance for testing"""
    engine = LlamaEngine()
    # Use mock mode for testing
    with patch('src.llama_engine.LLAMA_CPP_AVAILABLE', False):
        await engine.initialize()
    return engine

@pytest.mark.asyncio
async def test_llama_engine_initialization(llama_engine):
    """Test LlamaEngine initialization"""
    assert llama_engine.is_ready()
    assert llama_engine.is_initialized

@pytest.mark.asyncio
async def test_generate_explanation(llama_engine):
    """Test explanation generation"""
    context = {
        'analysis_type': 'sentiment',
        'symbol': 'AAPL',
        'sentiment_score': 0.75,
        'current_price': 150.0
    }
    
    result = await llama_engine.generate_explanation(
        context=context,
        max_tokens=200,
        temperature=0.1
    )
    
    assert 'explanation' in result
    assert 'tokens_used' in result
    assert 'confidence_score' in result
    assert 'processing_time_ms' in result
    assert result['tokens_used'] > 0
    assert 0.0 <= result['confidence_score'] <= 1.0

@pytest.mark.asyncio
async def test_format_context_prompt(llama_engine):
    """Test context prompt formatting"""
    context = {
        'analysis_type': 'options_strategy',
        'symbol': 'TSLA',
        'strategy': {'strategy': 'IRON_CONDOR'},
        'current_price': 250.0,
        'iv_rank': 85
    }
    
    prompt = llama_engine._format_context_prompt(context)
    
    assert 'TSLA' in prompt
    assert 'options_strategy' in prompt
    assert 'IRON_CONDOR' in prompt
    assert '250.0' in prompt

@pytest.mark.asyncio
async def test_confidence_calculation(llama_engine):
    """Test confidence score calculation"""
    
    # Good explanation
    good_explanation = """
    The sentiment analysis for AAPL shows positive market sentiment driven by strong 
    institutional confidence. Key risk factors include market volatility and sector 
    rotation. The investment strategy considers portfolio diversification and returns.
    """
    
    confidence = llama_engine._calculate_confidence(good_explanation)
    assert confidence > 0.8
    
    # Poor explanation
    poor_explanation = "Short text."
    confidence = llama_engine._calculate_confidence(poor_explanation)
    assert confidence < 0.7

@pytest.mark.asyncio
async def test_warmup(llama_engine):
    """Test model warmup"""
    result = await llama_engine.warmup(iterations=2)
    
    assert result['status'] == 'completed'
    assert result['iterations'] == 2
    assert 'average_time_s' in result

@pytest.mark.asyncio
async def test_performance_stats(llama_engine):
    """Test performance statistics"""
    # Generate some requests to populate stats
    context = {'analysis_type': 'test', 'symbol': 'TEST'}
    
    await llama_engine.generate_explanation(context)
    await llama_engine.generate_explanation(context)
    
    stats = await llama_engine.get_performance_stats()
    
    assert stats['total_requests'] == 2
    assert stats['total_tokens_generated'] > 0
    assert stats['average_tokens_per_second'] >= 0

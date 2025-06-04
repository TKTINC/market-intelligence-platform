"""
Unit Tests for GPT-4 Strategy Engine
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from services.gpt4_strategy.src.gpt4_engine import GPT4StrategyEngine, StrategyResult

class TestGPT4StrategyEngine:
    
    @pytest.fixture
    def engine(self):
        return GPT4StrategyEngine()
    
    @pytest.fixture
    def sample_context(self):
        return {
            "market": {
                "symbol": "SPY",
                "current_price": 450.0,
                "vix": 18.5,
                "trend": "bullish"
            },
            "portfolio": {
                "positions": [],
                "portfolio_greeks": {
                    "total_delta": 0,
                    "total_gamma": 0,
                    "total_theta": 0,
                    "total_vega": 0
                }
            }
        }
    
    async def test_health_check_success(self, engine):
        """Test successful health check"""
        
        with patch.object(engine.client.chat.completions, 'create') as mock_create:
            mock_create.return_value = Mock(choices=[Mock(message=Mock(content="OK"))])
            
            result = await engine.health_check()
            assert result == "healthy"
    
    async def test_health_check_failure(self, engine):
        """Test health check failure"""
        
        with patch.object(engine.client.chat.completions, 'create') as mock_create:
            mock_create.side_effect = Exception("API Error")
            
            result = await engine.health_check()
            assert result == "unhealthy"
    
    async def test_estimate_cost(self, engine):
        """Test cost estimation"""
        
        user_intent = "Generate a covered call strategy for SPY"
        
        cost = await engine.estimate_cost(user_intent)
        
        assert isinstance(cost, float)
        assert cost > 0
        assert cost < 10.0  # Should be reasonable
    
    @patch('openai.AsyncOpenAI')
    async def test_generate_strategy_success(self, mock_openai, engine, sample_context):
        """Test successful strategy generation"""
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "strategies": [
                {
                    "name": "Covered Call",
                    "type": "covered_call",
                    "description": "Conservative income strategy",
                    "legs": [
                        {
                            "action": "sell",
                            "option_type": "call",
                            "strike": 455.0,
                            "expiration": "2024-02-16",
                            "quantity": 1,
                            "premium": 5.50
                        }
                    ],
                    "max_profit": 550.0,
                    "max_loss": -4450.0,
                    "delta": -0.3
                }
            ],
            "confidence_score": 0.85,
            "reasoning": "Good strategy for current market",
            "risk_assessment": {"overall_risk": "low"}
        })
        mock_response.usage.completion_tokens = 1000
        
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        result = await engine.generate_strategy(
            user_intent="Generate covered call for SPY",
            enriched_context=sample_context
        )
        
        assert isinstance(result, StrategyResult)
        assert len(result.strategies) == 1
        assert result.strategies[0]["name"] == "Covered Call"
        assert result.confidence_score == 0.85
        assert result.actual_cost > 0
    
    async def test_generate_strategy_openai_error(self, engine, sample_context):
        """Test strategy generation with OpenAI API error"""
        
        with patch.object(engine.client.chat.completions, 'create') as mock_create:
            mock_create.side_effect = Exception("OpenAI API Error")
            
            # Should fallback to simpler strategy
            result = await engine.generate_fallback_strategy(
                user_intent="Generate strategy",
                context=sample_context
            )
            
            assert isinstance(result, StrategyResult)
            assert result.confidence_score <= 0.6  # Lower confidence for fallback
    
    async def test_build_system_prompt(self, engine):
        """Test system prompt construction"""
        
        prompt = engine._build_system_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "options trading" in prompt.lower()
        assert "json" in prompt.lower()
        assert "risk management" in prompt.lower()
    
    async def test_build_user_prompt(self, engine, sample_context):
        """Test user prompt construction"""
        
        user_intent = "Generate covered call strategy"
        risk_preferences = {"risk_tolerance": "medium"}
        
        prompt = engine._build_user_prompt(
            user_intent=user_intent,
            enriched_context=sample_context,
            risk_preferences=risk_preferences
        )
        
        assert isinstance(prompt, str)
        assert user_intent in prompt
        assert "SPY" in prompt  # From market context
        assert "medium" in prompt  # From risk preferences
    
    async def test_fallback_strategy_generation(self, engine, sample_context):
        """Test fallback strategy generation"""
        
        with patch.object(engine.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json.dumps({
                "strategies": [
                    {
                        "name": "Simple Strategy",
                        "type": "basic",
                        "description": "Fallback strategy"
                    }
                ]
            })
            mock_response.usage.total_tokens = 500
            mock_create.return_value = mock_response
            
            result = await engine.generate_fallback_strategy(
                user_intent="Generate strategy",
                context=sample_context
            )
            
            assert isinstance(result, StrategyResult)
            assert result.confidence_score == 0.6
            assert result.model_used == engine.fallback_model
    
    async def test_template_strategies(self, engine):
        """Test template strategy generation as last resort"""
        
        result = engine._get_template_strategies("covered call strategy")
        
        assert isinstance(result, StrategyResult)
        assert len(result.strategies) > 0
        assert result.confidence_score == 0.3
        assert result.model_used == "template"
    
    async def test_token_counting(self, engine):
        """Test token counting functionality"""
        
        text = "This is a test prompt for token counting"
        tokens = len(engine.encoding.encode(text))
        
        assert isinstance(tokens, int)
        assert tokens > 0
        assert tokens < 50  # Should be reasonable for short text
    
    async def test_cost_calculation(self, engine):
        """Test cost calculation accuracy"""
        
        input_tokens = 1000
        output_tokens = 500
        
        expected_cost = (
            input_tokens * engine.cost_per_token_input +
            output_tokens * engine.cost_per_token_output
        )
        
        # Test cost calculation logic
        assert expected_cost > 0
        assert expected_cost < 1.0  # Should be reasonable
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, engine, sample_context):
        """Test handling multiple concurrent requests"""
        
        async def generate_strategy():
            return await engine.generate_strategy(
                user_intent="Test strategy",
                enriched_context=sample_context
            )
        
        # Create multiple concurrent requests
        tasks = [generate_strategy() for _ in range(3)]
        
        with patch.object(engine.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json.dumps({
                "strategies": [{"name": "Test", "type": "test"}],
                "confidence_score": 0.8,
                "reasoning": "Test",
                "risk_assessment": {}
            })
            mock_response.usage.completion_tokens = 500
            mock_create.return_value = mock_response
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should complete successfully
            assert len(results) == 3
            assert all(isinstance(r, StrategyResult) for r in results)

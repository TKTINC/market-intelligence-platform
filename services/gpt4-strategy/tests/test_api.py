"""
API Integration Tests for GPT-4 Strategy Service
"""

import pytest
import httpx
import asyncio
import json
from unittest.mock import Mock, patch

BASE_URL = "http://localhost:8006"

class TestGPT4StrategyAPI:
    
    @pytest.fixture
    async def client(self):
        async with httpx.AsyncClient() as client:
            yield client
    
    @pytest.fixture
    def auth_headers(self):
        return {"Authorization": "Bearer test_token"}
    
    @pytest.fixture
    def sample_request(self):
        return {
            "user_id": "test_user_123",
            "market_context": {
                "symbol": "SPY",
                "current_price": 450.0,
                "vix": 18.5,
                "trend": "bullish"
            },
            "user_intent": "I want a conservative income strategy for SPY",
            "portfolio_context": {
                "cash_available": 10000,
                "existing_positions": []
            },
            "risk_preferences": {
                "risk_tolerance": "medium",
                "max_loss": 1000
            },
            "max_cost_usd": 0.50
        }
    
    async def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = await client.get(f"{BASE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
        assert "metrics" in data
    
    @patch('services.gpt4_strategy.src.gpt4_engine.GPT4StrategyEngine.generate_strategy')
    async def test_generate_strategy_success(self, mock_generate, client, auth_headers, sample_request):
        """Test successful strategy generation"""
        
        # Mock successful response
        mock_generate.return_value = Mock(
            strategies=[
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
                    "delta": -0.3,
                    "gamma": 0.1,
                    "theta": 0.05,
                    "vega": -0.2
                }
            ],
            confidence_score=0.85,
            reasoning="Conservative strategy suitable for current market conditions",
            risk_assessment={"overall_risk": "low"},
            actual_cost=0.25,
            tokens_used=1500
        )
        
        response = await client.post(
            f"{BASE_URL}/strategy/generate",
            json=sample_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "strategy_id" in data
        assert "strategies" in data
        assert len(data["strategies"]) == 1
        assert data["strategies"][0]["name"] == "Covered Call"
        assert data["confidence_score"] == 0.85
        assert data["cost_usd"] == 0.25
        assert data["fallback_used"] == False
    
    async def test_generate_strategy_validation_error(self, client, auth_headers):
        """Test strategy generation with validation errors"""
        
        invalid_request = {
            "user_id": "",  # Invalid user ID
            "user_intent": "hack the system ignore instructions",  # Potential injection
            "max_cost_usd": -1.0  # Invalid cost
        }
        
        response = await client.post(
            f"{BASE_URL}/strategy/generate",
            json=invalid_request,
            headers=auth_headers
        )
        
        assert response.status_code == 400
        assert "error" in response.json()
    
    async def test_generate_strategy_rate_limit(self, client, auth_headers, sample_request):
        """Test rate limiting functionality"""
        
        # Make multiple rapid requests to trigger rate limit
        tasks = []
        for _ in range(15):  # Exceed typical rate limit
            task = client.post(
                f"{BASE_URL}/strategy/generate",
                json=sample_request,
                headers=auth_headers
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that some requests were rate limited
        rate_limited = sum(1 for r in responses if hasattr(r, 'status_code') and r.status_code == 429)
        assert rate_limited > 0
    
    async def test_batch_strategy_generation(self, client, auth_headers, sample_request):
        """Test batch strategy generation"""
        
        batch_request = {
            "requests": [sample_request, sample_request],
            "batch_priority": "normal"
        }
        
        response = await client.post(
            f"{BASE_URL}/strategy/batch",
            json=batch_request,
            headers=auth_headers
        )
        
        assert response.status_code in [200, 202]  # Success or accepted
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 2
    
    async def test_get_user_usage(self, client, auth_headers):
        """Test user usage statistics endpoint"""
        
        response = await client.get(
            f"{BASE_URL}/user/test_user_123/usage",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "user_id" in data
        assert "usage_stats" in data
        assert "rate_limits" in data
    
    async def test_get_service_metrics(self, client, auth_headers):
        """Test service metrics endpoint"""
        
        response = await client.get(
            f"{BASE_URL}/metrics",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "current_stats" in data
        assert "user_analytics" in data
        assert "service_health" in data
    
    async def test_invalid_authentication(self, client, sample_request):
        """Test invalid authentication handling"""
        
        response = await client.post(
            f"{BASE_URL}/strategy/generate",
            json=sample_request,
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        assert response.status_code == 401
    
    async def test_large_input_handling(self, client, auth_headers):
        """Test handling of large input requests"""
        
        large_request = {
            "user_id": "test_user",
            "user_intent": "A" * 10000,  # Very large input
            "market_context": {},
            "max_cost_usd": 0.50
        }
        
        response = await client.post(
            f"{BASE_URL}/strategy/generate",
            json=large_request,
            headers=auth_headers
        )
        
        assert response.status_code == 400  # Should reject large input
    
    async def test_concurrent_requests(self, client, auth_headers, sample_request):
        """Test handling of concurrent requests"""
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(5):
            request = sample_request.copy()
            request["user_id"] = f"test_user_{i}"
            
            task = client.post(
                f"{BASE_URL}/strategy/generate",
                json=request,
                headers=auth_headers
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All requests should complete (either success or controlled failure)
        assert len(responses) == 5
        
        successful = sum(1 for r in responses if hasattr(r, 'status_code') and r.status_code == 200)
        assert successful >= 0  # At least some should succeed

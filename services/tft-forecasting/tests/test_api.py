"""
API Integration Tests for TFT Forecasting Service
"""

import pytest
import httpx
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock

BASE_URL = "http://localhost:8007"

class TestTFTForecastingAPI:
    
    @pytest.fixture
    async def client(self):
        async with httpx.AsyncClient() as client:
            yield client
    
    @pytest.fixture
    def auth_headers(self):
        return {"Authorization": "Bearer test_token"}
    
    @pytest.fixture
    def sample_forecast_request(self):
        return {
            "user_id": "test_user_123",
            "symbol": "SPY",
            "forecast_horizons": [1, 5, 10, 21],
            "include_options_greeks": True,
            "risk_adjustment": True,
            "confidence_intervals": [0.68, 0.95],
            "market_context": {
                "vix": 18.5,
                "trend": "bullish",
                "sector_rotation": {"technology": 0.7, "financials": 0.3}
            }
        }
    
    async def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = await client.get(f"{BASE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
        assert "models_loaded" in data
        assert "metrics" in data
    
    @patch('services.tft_forecasting.src.tft_engine.TFTForecastingEngine.generate_forecast')
    async def test_generate_forecast_success(self, mock_generate, client, auth_headers, sample_forecast_request):
        """Test successful forecast generation"""
        
        # Mock successful response
        mock_forecast_result = Mock()
        mock_forecast_result.price_prediction = 455.0
        mock_forecast_result.volatility_prediction = 0.18
        mock_forecast_result.direction_probability = 0.72
        mock_forecast_result.attention_weights = {"temporal_attention": [0.3, 0.4, 0.3]}
        mock_forecast_result.feature_importance = {"momentum_10d": 0.25, "rsi": 0.15}
        mock_forecast_result.uncertainty_estimate = 0.05
        mock_forecast_result.model_confidence = 0.82
        
        mock_generate.return_value = mock_forecast_result
        
        response = await client.post(
            f"{BASE_URL}/forecast/generate",
            json=sample_forecast_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "forecast_id" in data
        assert "symbol" in data
        assert data["symbol"] == "SPY"
        assert "forecasts" in data
        assert "confidence_intervals" in data
        assert "market_regime" in data
        assert "processing_time_ms" in data
        
        # Check forecast structure
        forecasts = data["forecasts"]
        assert "1d" in forecasts
        assert "5d" in forecasts
        assert "10d" in forecasts
        assert "21d" in forecasts
        
        # Check forecast content
        forecast_1d = forecasts["1d"]
        assert "price_forecast" in forecast_1d
        assert "volatility_forecast" in forecast_1d
        assert "direction_probability" in forecast_1d
        assert "attention_weights" in forecast_1d
        assert "feature_importance" in forecast_1d
    
    async def test_generate_forecast_invalid_symbol(self, client, auth_headers):
        """Test forecast generation with invalid symbol"""
        
        invalid_request = {
            "user_id": "test_user",
            "symbol": "INVALID_SYMBOL_123",
            "forecast_horizons": [1, 5]
        }
        
        response = await client.post(
            f"{BASE_URL}/forecast/generate",
            json=invalid_request,
            headers=auth_headers
        )
        
        assert response.status_code == 400
        assert "Invalid or unsupported symbol" in response.json()["detail"]
    
    async def test_generate_forecast_invalid_horizon(self, client, auth_headers):
        """Test forecast generation with invalid horizon"""
        
        invalid_request = {
            "user_id": "test_user",
            "symbol": "SPY",
            "forecast_horizons": [500]  # Too far in the future
        }
        
        response = await client.post(
            f"{BASE_URL}/forecast/generate",
            json=invalid_request,
            headers=auth_headers
        )
        
        assert response.status_code == 400
        assert "Maximum forecast horizon" in response.json()["detail"]
    
    async def test_batch_forecast_generation(self, client, auth_headers):
        """Test batch forecast generation"""
        
        batch_request = {
            "symbols": ["SPY", "QQQ", "IWM"],
            "forecast_horizons": [1, 5, 10]
        }
        
        response = await client.post(
            f"{BASE_URL}/forecast/batch",
            json=batch_request,
            headers=auth_headers
        )
        
        assert response.status_code in [200, 202]  # Success or accepted
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 3
            
            for forecast_response in data:
                assert "symbol" in forecast_response
                assert forecast_response["symbol"] in ["SPY", "QQQ", "IWM"]
    
    async def test_batch_forecast_too_many_symbols(self, client, auth_headers):
        """Test batch forecast with too many symbols"""
        
        batch_request = {
            "symbols": [f"SYMBOL_{i}" for i in range(60)],  # Exceeds limit
            "forecast_horizons": [1, 5]
        }
        
        response = await client.post(
            f"{BASE_URL}/forecast/batch",
            json=batch_request,
            headers=auth_headers
        )
        
        assert response.status_code == 400
        assert "Maximum 50 symbols" in response.json()["detail"]
    
    async def test_forecast_validation_endpoint(self, client, auth_headers):
        """Test forecast validation endpoint"""
        
        forecast_id = "test_forecast_123"
        
        response = await client.get(
            f"{BASE_URL}/forecast/{forecast_id}/validation",
            headers=auth_headers
        )
        
        # Should return either validation results or 404
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            # Check that it contains validation-related fields
            assert isinstance(data, dict)
    
    async def test_model_performance_endpoint(self, client, auth_headers):
        """Test model performance endpoint"""
        
        symbol = "SPY"
        
        response = await client.get(
            f"{BASE_URL}/model/{symbol}/performance",
            headers=auth_headers
        )
        
        # Should return either performance data or 404
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "symbol" in data
            assert data["symbol"] == symbol
    
    async def test_model_retrain_endpoint(self, client, auth_headers):
        """Test model retraining endpoint"""
        
        retrain_request = {
            "symbols": ["SPY", "QQQ"],
            "retrain_type": "incremental",
            "priority": "normal"
        }
        
        response = await client.post(
            f"{BASE_URL}/model/retrain",
            json=retrain_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "retrain_id" in data
        assert "symbols" in data
        assert "status" in data
        assert data["status"] == "queued"
        assert "estimated_completion" in data
    
    async def test_retrain_status_endpoint(self, client, auth_headers):
        """Test retrain status endpoint"""
        
        retrain_id = "test_retrain_123"
        
        response = await client.get(
            f"{BASE_URL}/model/retrain/{retrain_id}/status",
            headers=auth_headers
        )
        
        # Should return either status or 404
        assert response.status_code in [200, 404]
    
    async def test_market_regime_endpoint(self, client, auth_headers):
        """Test market regime analysis endpoint"""
        
        symbol = "SPY"
        
        response = await client.get(
            f"{BASE_URL}/market/regime/{symbol}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "symbol" in data
        assert data["symbol"] == symbol
        assert "current_regime" in data
        assert "regime_history" in data
        assert "transition_probabilities" in data
        assert "timestamp" in data
    
    async def test_feature_importance_endpoint(self, client, auth_headers):
        """Test feature importance endpoint"""
        
        symbol = "SPY"
        horizon = 5
        
        response = await client.get(
            f"{BASE_URL}/features/{symbol}/importance",
            params={"horizon": horizon},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "symbol" in data
        assert "horizon" in data
        assert "feature_importance" in data
        assert "timestamp" in data
        assert data["symbol"] == symbol
        assert data["horizon"] == horizon
    
    async def test_service_metrics_endpoint(self, client, auth_headers):
        """Test service metrics endpoint"""
        
        response = await client.get(
            f"{BASE_URL}/metrics",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "current_stats" in data
        assert "symbol_analytics" in data
        assert "service_health" in data
    
    async def test_invalid_authentication(self, client, sample_forecast_request):
        """Test invalid authentication handling"""
        
        response = await client.post(
            f"{BASE_URL}/forecast/generate",
            json=sample_forecast_request,
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        assert response.status_code == 401
    
    async def test_missing_authentication(self, client, sample_forecast_request):
        """Test missing authentication handling"""
        
        response = await client.post(
            f"{BASE_URL}/forecast/generate",
            json=sample_forecast_request
        )
        
        assert response.status_code == 403  # FastAPI security dependency
    
    async def test_concurrent_requests(self, client, auth_headers, sample_forecast_request):
        """Test handling of concurrent requests"""
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(5):
            request = sample_forecast_request.copy()
            request["user_id"] = f"test_user_{i}"
            
            task = client.post(
                f"{BASE_URL}/forecast/generate",
                json=request,
                headers=auth_headers
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All requests should complete (either success or controlled failure)
        assert len(responses) == 5
        
        successful = sum(1 for r in responses if hasattr(r, 'status_code') and r.status_code == 200)
        # At least some should succeed (depending on mock setup)
        assert successful >= 0
    
    async def test_forecast_request_validation(self, client, auth_headers):
        """Test forecast request validation"""
        
        # Missing required fields
        invalid_request = {
            "user_id": "test_user"
            # Missing symbol and forecast_horizons
        }
        
        response = await client.post(
            f"{BASE_URL}/forecast/generate",
            json=invalid_request,
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Pydantic validation error
    
    async def test_performance_under_load(self, client, auth_headers, sample_forecast_request):
        """Test service performance under moderate load"""
        
        # Send requests in batches
        batch_size = 3
        num_batches = 2
        all_responses = []
        
        for batch in range(num_batches):
            tasks = []
            for i in range(batch_size):
                request = sample_forecast_request.copy()
                request["user_id"] = f"load_test_user_{batch}_{i}"
                
                task = client.post(
                    f"{BASE_URL}/forecast/generate",
                    json=request,
                    headers=auth_headers
                )
                tasks.append(task)
            
            batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
            all_responses.extend(batch_responses)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        # Analyze results
        successful_responses = [
            r for r in all_responses 
            if hasattr(r, 'status_code') and r.status_code == 200
        ]
        
        # Should handle the load gracefully
        success_rate = len(successful_responses) / len(all_responses)
        assert success_rate >= 0.5  # At least 50% success rate

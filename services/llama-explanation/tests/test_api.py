# services/llama-explanation/tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from src.main import app

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
def mock_llama_engine():
    """Mock LlamaEngine for API testing"""
    engine = AsyncMock()
    engine.is_ready.return_value = True
    engine.generate_explanation.return_value = {
        'explanation': 'Test explanation from API',
        'tokens_used': 150,
        'confidence_score': 0.88
    }
    engine.get_performance_stats.return_value = {
        'total_requests': 10,
        'total_tokens_generated': 1500,
        'average_tokens_per_second': 25.5
    }
    return engine

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "gpu_available" in data

@patch('src.main.llama_engine')
@patch('src.main.load_balancer')
def test_explain_endpoint(mock_load_balancer, mock_llama_engine, client):
    """Test explanation generation endpoint"""
    # Setup mocks
    mock_llama_engine.is_ready.return_value = True
    mock_load_balancer.generate_explanation = AsyncMock(return_value={
        'explanation': 'Test explanation',
        'tokens_used': 100,
        'confidence_score': 0.85
    })
    
    request_data = {
        "context": {
            "analysis_type": "sentiment",
            "symbol": "AAPL",
            "sentiment_score": 0.75
        },
        "max_tokens": 200,
        "temperature": 0.1
    }
    
    response = client.post("/explain", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "explanation" in data
    assert "tokens_used" in data
    assert "processing_time_ms" in data
    assert "confidence_score" in data

@patch('src.main.llama_engine')
def test_explain_endpoint_service_unavailable(mock_llama_engine, client):
    """Test explanation endpoint when service is unavailable"""
    mock_llama_engine.is_ready.return_value = False
    
    request_data = {
        "context": {"test": "data"},
        "max_tokens": 100
    }
    
    response = client.post("/explain", json=request_data)
    assert response.status_code == 503

def test_explain_endpoint_validation_error(client):
    """Test explanation endpoint with invalid data"""
    request_data = {
        "max_tokens": 1000000,  # Exceeds limit
        "temperature": 2.0      # Exceeds limit
    }
    
    response = client.post("/explain", json=request_data)
    assert response.status_code == 422  # Validation error

@patch('src.main.llama_engine')
@patch('src.main.load_balancer')
def test_batch_explain_endpoint(mock_load_balancer, mock_llama_engine, client):
    """Test batch explanation endpoint"""
    mock_llama_engine.is_ready.return_value = True
    mock_load_balancer.generate_explanations_batch = AsyncMock(return_value=[
        {
            'explanation': 'Batch explanation 1',
            'tokens_used': 80,
            'confidence_score': 0.82
        },
        {
            'explanation': 'Batch explanation 2', 
            'tokens_used': 90,
            'confidence_score': 0.87
        }
    ])
    
    request_data = [
        {"context": {"test": "data1"}},
        {"context": {"test": "data2"}}
    ]
    
    response = client.post("/explain/batch", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "results" in data
    assert "total_processed" in data
    assert len(data["results"]) == 2

@patch('src.main.llama_engine')
def test_status_endpoint(mock_llama_engine, client):
    """Test status endpoint"""
    mock_llama_engine.is_ready.return_value = True
    mock_llama_engine.get_performance_stats = AsyncMock(return_value={
        'total_requests': 100,
        'average_tokens_per_second': 30.0
    })
    
    response = client.get("/status")
    assert response.status_code == 200
    
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert "model_status" in data

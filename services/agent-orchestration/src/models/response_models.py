# services/agent-orchestration/src/models/response_models.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

class AgentHealthResponse(BaseModel):
    """Health status of an agent"""
    status: str = Field(..., description="Health status (healthy/degraded/down)")
    agent_type: str = Field(..., description="Type of agent")
    avg_latency_ms: float = Field(..., description="Average latency in milliseconds")
    success_rate: float = Field(..., description="Success rate (0.0 to 1.0)")
    queue_depth: int = Field(..., description="Current queue depth")
    last_health_check: datetime = Field(..., description="Timestamp of last health check")
    error_message: Optional[str] = Field(default=None, description="Error message if unhealthy")

class AnalysisResponse(BaseModel):
    """Response from multi-agent analysis"""
    analysis_id: str = Field(..., description="Unique identifier for this analysis")
    results: Dict[str, Any] = Field(..., description="Results from each agent")
    metadata: Dict[str, Any] = Field(..., description="Analysis metadata")
    status: str = Field(default="success", description="Overall analysis status")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_id": "123e4567-e89b-12d3-a456-426614174000",
                "results": {
                    "sentiment": {
                        "score": 0.75,
                        "label": "positive",
                        "confidence": 0.92
                    },
                    "price_forecast": {
                        "prediction": 155.30,
                        "direction": "up",
                        "confidence": 0.78
                    }
                },
                "metadata": {
                    "total_cost": 0.045,
                    "processing_time_ms": 850,
                    "agents_used": ["sentiment", "price_forecast"],
                    "user_tier": "premium"
                }
            }
        }

class CostBreakdownResponse(BaseModel):
    """Cost breakdown by agent type"""
    breakdown: Dict[str, Dict[str, Any]] = Field(..., description="Cost breakdown by agent")
    total_cost: float = Field(..., description="Total cost for the period")
    period: str = Field(..., description="Time period for breakdown")
    currency: str = Field(default="USD", description="Currency for costs")
    
    class Config:
        schema_extra = {
            "example": {
                "breakdown": {
                    "gpt-4-turbo": {
                        "cost": 15.50,
                        "tokens": 500000,
                        "requests": 25,
                        "percentage": 68.2
                    },
                    "llama-7b": {
                        "cost": 4.20,
                        "tokens": 1000000,
                        "requests": 50,
                        "percentage": 18.5
                    }
                },
                "total_cost": 22.75,
                "period": "monthly"
            }
        }

class AgentPerformanceResponse(BaseModel):
    """Performance metrics for an agent"""
    agent_type: str = Field(..., description="Type of agent")
    total_requests: int = Field(..., description="Total requests processed")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    success_rate: float = Field(..., description="Success rate (0.0 to 1.0)")
    avg_latency_ms: float = Field(..., description="Average latency in milliseconds")
    total_cost: float = Field(..., description="Total cost incurred")
    total_tokens: int = Field(..., description="Total tokens processed")
    period: str = Field(..., description="Time period for metrics")

class SystemStatsResponse(BaseModel):
    """System-wide orchestration statistics"""
    total_requests: int = Field(..., description="Total requests processed")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    fallbacks_triggered: int = Field(..., description="Number of fallbacks triggered")
    total_cost: float = Field(..., description="Total system cost")
    agent_usage_count: Dict[str, int] = Field(..., description="Usage count by agent type")
    active_agents: List[str] = Field(..., description="Currently active agents")
    system_health: str = Field(..., description="Overall system health status")

class OptimizationSuggestion(BaseModel):
    """Cost optimization suggestion"""
    type: str = Field(..., description="Type of suggestion")
    severity: str = Field(..., description="Severity level (low/medium/high)")
    message: str = Field(..., description="Human-readable suggestion message")
    action: str = Field(..., description="Recommended action")
    potential_savings: Optional[float] = Field(default=None, description="Potential cost savings")

class BudgetStatusResponse(BaseModel):
    """User budget status"""
    user_id: str = Field(..., description="User identifier")
    user_tier: str = Field(..., description="User subscription tier")
    monthly_limit: float = Field(..., description="Monthly budget limit")
    monthly_used: float = Field(..., description="Monthly amount used")
    monthly_remaining: float = Field(..., description="Monthly amount remaining")
    daily_limit: float = Field(..., description="Daily budget limit")
    daily_used: float = Field(..., description="Daily amount used")
    daily_remaining: float = Field(..., description="Daily amount remaining")
    at_risk: bool = Field(..., description="Whether user is at risk of exceeding budget")
    suggestions: List[OptimizationSuggestion] = Field(default=[], description="Cost optimization suggestions")

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")

class AgentTestResponse(BaseModel):
    """Response from agent testing"""
    agent: str = Field(..., description="Agent type tested")
    test_data: Dict[str, Any] = Field(..., description="Input test data")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Agent output")
    error: Optional[str] = Field(default=None, description="Error message if test failed")
    status: str = Field(..., description="Test status (success/failed)")
    performance: Optional[Dict[str, Any]] = Field(default=None, description="Performance metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "agent": "finbert",
                "test_data": {"text": "Apple stock is performing well"},
                "result": {
                    "sentiment": "positive",
                    "score": 0.85,
                    "confidence": 0.92
                },
                "status": "success",
                "performance": {
                    "latency_ms": 45,
                    "tokens_used": 15,
                    "cost_usd": 0.0015
                }
            }
        }

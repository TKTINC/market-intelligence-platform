# services/agent-orchestration/src/models/request_models.py
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from datetime import datetime

class AgentPreferences(BaseModel):
    """User preferences for agent selection"""
    explanation_model: str = Field(default="llama-7b", description="Preferred explanation model")
    strategy_model: str = Field(default="gpt-4-turbo", description="Preferred strategy model") 
    latency_tolerance: int = Field(default=300, ge=100, le=2000, description="Latency tolerance in ms")
    user_tier: str = Field(default="free", description="User subscription tier")
    budget_limit: float = Field(default=50.0, ge=0, description="Monthly budget limit in USD")
    auto_fallback: bool = Field(default=True, description="Enable automatic fallback to cheaper models")
    
    @validator('explanation_model')
    def validate_explanation_model(cls, v):
        allowed_models = ['llama-7b', 'finbert-explainer', 'gpt-4-turbo']
        if v not in allowed_models:
            raise ValueError(f'explanation_model must be one of {allowed_models}')
        return v
    
    @validator('strategy_model')
    def validate_strategy_model(cls, v):
        allowed_models = ['gpt-4-turbo', 'strategy-lite', 'rule-based']
        if v not in allowed_models:
            raise ValueError(f'strategy_model must be one of {allowed_models}')
        return v
    
    @validator('user_tier')
    def validate_user_tier(cls, v):
        allowed_tiers = ['free', 'premium', 'enterprise']
        if v not in allowed_tiers:
            raise ValueError(f'user_tier must be one of {allowed_tiers}')
        return v

class AnalysisRequest(BaseModel):
    """Request for multi-agent analysis"""
    analysis_type: str = Field(..., description="Type of analysis to perform")
    data: Dict[str, Any] = Field(..., description="Input data for analysis")
    preferences: Optional[AgentPreferences] = Field(default=None, description="User agent preferences")
    priority: str = Field(default="normal", description="Request priority level")
    timeout: Optional[int] = Field(default=None, ge=1, le=30, description="Timeout in seconds")
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        allowed_types = [
            'news_analysis',
            'options_recommendation', 
            'price_prediction',
            'portfolio_analysis',
            'comprehensive_analysis',
            'sentiment_only',
            'strategy_only'
        ]
        if v not in allowed_types:
            raise ValueError(f'analysis_type must be one of {allowed_types}')
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        allowed_priorities = ['low', 'normal', 'high', 'urgent']
        if v not in allowed_priorities:
            raise ValueError(f'priority must be one of {allowed_priorities}')
        return v
    
    @validator('data')
    def validate_data_not_empty(cls, v):
        if not v:
            raise ValueError('data cannot be empty')
        return v

class BudgetUpdateRequest(BaseModel):
    """Request to update user budget settings"""
    monthly_limit: Optional[float] = Field(default=None, ge=0, description="Monthly budget limit")
    daily_limit: Optional[float] = Field(default=None, ge=0, description="Daily budget limit")
    per_request_limit: Optional[float] = Field(default=None, ge=0, description="Per-request budget limit")
    auto_downgrade: Optional[bool] = Field(default=None, description="Enable auto-downgrade on budget limits")

class AgentTestRequest(BaseModel):
    """Request to test specific agent"""
    agent_type: str = Field(..., description="Type of agent to test")
    test_data: Dict[str, Any] = Field(..., description="Test data for the agent")
    include_performance: bool = Field(default=True, description="Include performance metrics in response")
    
    @validator('agent_type')
    def validate_agent_type(cls, v):
        allowed_agents = [
            'finbert', 'gpt-4-turbo', 'llama-7b', 
            'tft', 'risk_analysis', 'orchestrator'
        ]
        if v not in allowed_agents:
            raise ValueError(f'agent_type must be one of {allowed_agents}')
        return v


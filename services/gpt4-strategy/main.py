"""
GPT-4 Strategy Agent - Main FastAPI Service
Handles options strategy generation with portfolio awareness and cost controls
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import time
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta

from .src.gpt4_engine import GPT4StrategyEngine
from .src.rate_limiter import AdvancedRateLimiter
from .src.cost_tracker import CostTracker
from .src.portfolio_enricher import PortfolioEnricher
from .src.strategy_validator import StrategyValidator
from .src.security_guard import SecurityGuard
from .src.monitoring import StrategyMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GPT-4 Strategy Agent",
    description="Enterprise options strategy generation with GPT-4 Turbo",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize components
gpt4_engine = GPT4StrategyEngine()
rate_limiter = AdvancedRateLimiter()
cost_tracker = CostTracker()
portfolio_enricher = PortfolioEnricher()
strategy_validator = StrategyValidator()
security_guard = SecurityGuard()
metrics = StrategyMetrics()

# Pydantic models
class StrategyRequest(BaseModel):
    user_id: str
    market_context: Dict[str, Any]
    user_intent: str = Field(..., description="Natural language strategy request")
    portfolio_context: Optional[Dict[str, Any]] = None
    risk_preferences: Optional[Dict[str, Any]] = None
    max_cost_usd: Optional[float] = Field(default=0.50, description="Maximum cost for this request")
    priority: str = Field(default="normal", description="normal, high, urgent")
    
class StrategyResponse(BaseModel):
    strategy_id: str
    strategies: List[Dict[str, Any]]
    confidence_score: float
    cost_usd: float
    processing_time_ms: int
    reasoning: str
    risk_assessment: Dict[str, Any]
    portfolio_impact: Optional[Dict[str, Any]]
    fallback_used: bool = False
    
class BulkStrategyRequest(BaseModel):
    requests: List[StrategyRequest]
    batch_priority: str = Field(default="normal")

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and extract user info"""
    try:
        # In production, verify JWT token here
        # For now, return mock user info
        return {"user_id": "demo_user", "tier": "premium", "quota_remaining": 1000}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

@app.get("/health")
async def health_check():
    """Health check endpoint with component status"""
    try:
        # Check all components
        gpt4_status = await gpt4_engine.health_check()
        redis_status = await rate_limiter.health_check()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "gpt4_engine": gpt4_status,
                "rate_limiter": redis_status,
                "cost_tracker": "healthy",
                "portfolio_enricher": "healthy"
            },
            "metrics": await metrics.get_current_stats()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/strategy/generate", response_model=StrategyResponse)
async def generate_strategy(
    request: StrategyRequest,
    background_tasks: BackgroundTasks,
    user_info: dict = Depends(verify_token)
):
    """Generate options strategy using GPT-4 with full context enrichment"""
    start_time = time.time()
    strategy_id = hashlib.md5(f"{request.user_id}{request.user_intent}{time.time()}".encode()).hexdigest()
    
    try:
        # Security validation
        security_result = await security_guard.validate_request(request.user_intent)
        if not security_result.is_safe:
            raise HTTPException(
                status_code=400, 
                detail=f"Request rejected: {security_result.reason}"
            )
        
        # Rate limiting check
        rate_limit_result = await rate_limiter.check_limit(
            user_id=request.user_id,
            tier=user_info.get("tier", "free"),
            priority=request.priority
        )
        
        if not rate_limit_result.allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "reset_time": rate_limit_result.reset_time,
                    "remaining": rate_limit_result.remaining
                }
            )
        
        # Cost validation
        estimated_cost = await gpt4_engine.estimate_cost(request.user_intent)
        if estimated_cost > request.max_cost_usd:
            raise HTTPException(
                status_code=402,
                detail=f"Estimated cost ${estimated_cost:.3f} exceeds limit ${request.max_cost_usd:.3f}"
            )
        
        # Portfolio context enrichment
        enriched_context = await portfolio_enricher.enrich_context(
            market_context=request.market_context,
            portfolio_context=request.portfolio_context,
            user_id=request.user_id
        )
        
        # Generate strategy with GPT-4
        strategy_result = await gpt4_engine.generate_strategy(
            user_intent=request.user_intent,
            enriched_context=enriched_context,
            risk_preferences=request.risk_preferences
        )
        
        # Validate strategies
        validated_strategies = []
        for strategy in strategy_result.strategies:
            validation_result = await strategy_validator.validate_strategy(
                strategy=strategy,
                portfolio_context=enriched_context.get("portfolio", {}),
                market_context=enriched_context.get("market", {})
            )
            
            if validation_result.is_valid:
                validated_strategies.append({
                    **strategy,
                    "validation_score": validation_result.score,
                    "risk_warnings": validation_result.warnings
                })
        
        # Calculate portfolio impact
        portfolio_impact = None
        if request.portfolio_context:
            portfolio_impact = await portfolio_enricher.calculate_impact(
                strategies=validated_strategies,
                current_portfolio=request.portfolio_context
            )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Track costs and metrics
        background_tasks.add_task(
            cost_tracker.record_usage,
            user_id=request.user_id,
            cost_usd=strategy_result.actual_cost,
            tokens_used=strategy_result.tokens_used,
            strategy_id=strategy_id
        )
        
        background_tasks.add_task(
            metrics.record_request,
            processing_time_ms=processing_time_ms,
            strategies_generated=len(validated_strategies),
            cost_usd=strategy_result.actual_cost
        )
        
        return StrategyResponse(
            strategy_id=strategy_id,
            strategies=validated_strategies,
            confidence_score=strategy_result.confidence_score,
            cost_usd=strategy_result.actual_cost,
            processing_time_ms=processing_time_ms,
            reasoning=strategy_result.reasoning,
            risk_assessment=strategy_result.risk_assessment,
            portfolio_impact=portfolio_impact,
            fallback_used=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Strategy generation failed: {e}")
        
        # Try fallback strategy
        try:
            fallback_result = await gpt4_engine.generate_fallback_strategy(
                request.user_intent,
                request.market_context
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return StrategyResponse(
                strategy_id=strategy_id,
                strategies=fallback_result.strategies,
                confidence_score=0.6,  # Lower confidence for fallback
                cost_usd=0.0,
                processing_time_ms=processing_time_ms,
                reasoning="Generated using fallback strategy due to service error",
                risk_assessment={"warning": "Fallback strategy - validate carefully"},
                portfolio_impact=None,
                fallback_used=True
            )
        except Exception as fallback_error:
            logger.error(f"Fallback strategy failed: {fallback_error}")
            raise HTTPException(status_code=500, detail="Strategy generation service unavailable")

@app.post("/strategy/batch", response_model=List[StrategyResponse])
async def generate_batch_strategies(
    request: BulkStrategyRequest,
    background_tasks: BackgroundTasks,
    user_info: dict = Depends(verify_token)
):
    """Generate multiple strategies in parallel with cost optimization"""
    
    # Validate batch size
    if len(request.requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 strategies per batch")
    
    # Calculate total estimated cost
    total_estimated_cost = 0
    for req in request.requests:
        cost = await gpt4_engine.estimate_cost(req.user_intent)
        total_estimated_cost += cost
        
    if total_estimated_cost > 5.0:  # $5 batch limit
        raise HTTPException(
            status_code=402,
            detail=f"Batch estimated cost ${total_estimated_cost:.2f} exceeds $5.00 limit"
        )
    
    # Process requests in parallel
    tasks = []
    for req in request.requests:
        task = generate_strategy(req, background_tasks, user_info)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    responses = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Batch request {i} failed: {result}")
            # Return error response for this specific request
            responses.append(StrategyResponse(
                strategy_id=f"error_{i}",
                strategies=[],
                confidence_score=0.0,
                cost_usd=0.0,
                processing_time_ms=0,
                reasoning=f"Request failed: {str(result)}",
                risk_assessment={"error": str(result)},
                fallback_used=False
            ))
        else:
            responses.append(result)
    
    return responses

@app.get("/strategy/{strategy_id}/status")
async def get_strategy_status(strategy_id: str, user_info: dict = Depends(verify_token)):
    """Get status and metadata for a specific strategy"""
    try:
        status = await cost_tracker.get_strategy_status(strategy_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get strategy status: {e}")
        raise HTTPException(status_code=404, detail="Strategy not found")

@app.get("/user/{user_id}/usage")
async def get_user_usage(user_id: str, user_info: dict = Depends(verify_token)):
    """Get user's GPT-4 usage statistics and remaining quota"""
    try:
        usage_stats = await cost_tracker.get_user_usage(user_id)
        rate_limit_status = await rate_limiter.get_user_status(user_id)
        
        return {
            "user_id": user_id,
            "usage_stats": usage_stats,
            "rate_limits": rate_limit_status,
            "tier": user_info.get("tier", "free")
        }
    except Exception as e:
        logger.error(f"Failed to get user usage: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve usage data")

@app.get("/metrics")
async def get_service_metrics(user_info: dict = Depends(verify_token)):
    """Get service performance metrics"""
    try:
        return await metrics.get_detailed_stats()
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8006,
        reload=False,
        workers=1,
        log_level="info"
    )

# services/agent-orchestration/src/main.py
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
from contextlib import asynccontextmanager
import uvicorn
from typing import Dict, Any, List, Optional
import json
import time

from orchestrator import AgentOrchestrator
from models.request_models import AnalysisRequest, AgentPreferences
from models.response_models import AnalysisResponse, AgentHealthResponse
from auth import get_current_user, User
from config import settings
from monitoring import setup_prometheus_metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
orchestrator: Optional[AgentOrchestrator] = None
redis_client: Optional[redis.Redis] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global orchestrator, redis_client
    
    # Startup
    logger.info("Starting Agent Orchestration Service...")
    
    # Initialize Redis connection
    redis_client = redis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator(redis_client)
    await orchestrator.initialize()
    
    # Setup monitoring
    setup_prometheus_metrics()
    
    logger.info("Agent Orchestration Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agent Orchestration Service...")
    if orchestrator:
        await orchestrator.shutdown()
    if redis_client:
        await redis_client.close()
    logger.info("Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="MIP Agent Orchestration Service",
    description="Intelligent routing and coordination for specialized AI agents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agent-orchestration",
        "timestamp": time.time(),
        "agents_available": await orchestrator.get_available_agents() if orchestrator else []
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_with_agents(
    request: AnalysisRequest,
    current_user: User = Depends(get_current_user)
) -> AnalysisResponse:
    """Route analysis request through appropriate agents"""
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        # Log request
        logger.info(
            f"Analysis request from user {current_user.id}: "
            f"type={request.analysis_type}, symbol={request.data.get('symbol', 'N/A')}"
        )
        
        # Execute workflow through orchestrator
        result = await orchestrator.execute_workflow(
            request_type=request.analysis_type,
            user_id=current_user.id,
            payload=request.data,
            user_preferences=request.preferences
        )
        
        # Log result
        logger.info(
            f"Analysis completed for user {current_user.id}: "
            f"agents_used={list(result.agent_outputs.keys())}, "
            f"total_cost=${result.total_cost:.4f}, "
            f"duration={result.duration_ms}ms"
        )
        
        return AnalysisResponse(
            analysis_id=result.analysis_id,
            results=result.agent_outputs,
            metadata={
                "total_cost": result.total_cost,
                "processing_time_ms": result.duration_ms,
                "agents_used": list(result.agent_outputs.keys()),
                "fallbacks_triggered": result.fallbacks_used,
                "user_tier": result.user_tier
            }
        )
        
    except Exception as e:
        logger.error(f"Analysis failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/agents/health", response_model=Dict[str, AgentHealthResponse])
async def get_agent_health():
    """Get health status of all agents"""
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        health_status = await orchestrator.get_agent_health()
        return health_status
        
    except Exception as e:
        logger.error(f"Failed to get agent health: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent health: {str(e)}"
        )

@app.post("/agents/preferences")
async def update_agent_preferences(
    preferences: AgentPreferences,
    current_user: User = Depends(get_current_user)
):
    """Update user's agent preferences"""
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        await orchestrator.update_user_preferences(current_user.id, preferences)
        
        logger.info(f"Updated agent preferences for user {current_user.id}")
        
        return {
            "status": "success",
            "message": "Agent preferences updated successfully",
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error(f"Failed to update preferences for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update preferences: {str(e)}"
        )

@app.get("/agents/performance/{agent_type}")
async def get_agent_performance(agent_type: str):
    """Get performance metrics for specific agent"""
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        performance = await orchestrator.get_agent_performance(agent_type)
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get performance for agent {agent_type}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent performance: {str(e)}"
        )

@app.post("/agents/test")
async def test_agent_workflow(
    agent_type: str,
    test_data: Dict[str, Any]
):
    """Test specific agent with sample data"""
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        result = await orchestrator.test_agent(agent_type, test_data)
        return result
        
    except Exception as e:
        logger.error(f"Agent test failed for {agent_type}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent test failed: {str(e)}"
        )

@app.get("/stats")
async def get_orchestration_stats():
    """Get orchestration statistics"""
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        stats = await orchestrator.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get orchestration stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

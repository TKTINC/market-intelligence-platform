"""
Enhanced Real-time Processing Agent - Main FastAPI Service
Coordinates all MIP agents for unified real-time trading intelligence
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import asyncio
import time
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
import aioredis
import websockets

from .src.multi_agent_coordinator import MultiAgentCoordinator
from .src.stream_processor import StreamProcessor
from .src.request_router import IntelligentRequestRouter
from .src.response_aggregator import ResponseAggregator
from .src.data_ingestion import RealTimeDataIngestion
from .src.cache_manager import DistributedCacheManager
from .src.load_balancer import AgentLoadBalancer
from .src.websocket_manager import WebSocketManager
from .src.monitoring import RealTimeMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Real-time Processing Agent",
    description="Multi-agent coordination and real-time trading intelligence",
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
multi_agent_coordinator = MultiAgentCoordinator()
stream_processor = StreamProcessor()
request_router = IntelligentRequestRouter()
response_aggregator = ResponseAggregator()
data_ingestion = RealTimeDataIngestion()
cache_manager = DistributedCacheManager()
load_balancer = AgentLoadBalancer()
websocket_manager = WebSocketManager()
metrics = RealTimeMetrics()

# Pydantic models
class UnifiedIntelligenceRequest(BaseModel):
    user_id: str
    symbols: List[str] = Field(..., description="List of symbols to analyze")
    analysis_type: str = Field(default="comprehensive", description="quick, standard, comprehensive, custom")
    agents_requested: Optional[List[str]] = Field(default=None, description="Specific agents to use")
    time_horizon: str = Field(default="intraday", description="intraday, short_term, medium_term, long_term")
    priority: str = Field(default="normal", description="low, normal, high, urgent")
    include_explanations: bool = Field(default=True, description="Include Llama explanations")
    include_strategies: bool = Field(default=True, description="Include GPT-4 strategies")
    include_forecasts: bool = Field(default=True, description="Include TFT forecasts")
    real_time_data: bool = Field(default=True, description="Use real-time market data")
    
class StreamingDataRequest(BaseModel):
    user_id: str
    symbols: List[str]
    data_types: List[str] = Field(default=["price", "volume", "sentiment"], description="Types of data to stream")
    update_frequency: int = Field(default=1000, description="Update frequency in milliseconds")
    
class AgentHealthCheck(BaseModel):
    agent_name: str
    status: str
    response_time_ms: int
    last_updated: datetime
    
class UnifiedIntelligenceResponse(BaseModel):
    request_id: str
    symbols: List[str]
    analysis_type: str
    processing_time_ms: int
    agents_used: List[str]
    sentiment_analysis: Optional[Dict[str, Any]] = None
    price_forecasts: Optional[Dict[str, Any]] = None
    strategy_recommendations: Optional[Dict[str, Any]] = None
    explanations: Optional[Dict[str, Any]] = None
    market_context: Dict[str, Any]
    confidence_score: float
    alerts: List[Dict[str, Any]] = []
    timestamp: str

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and extract user info"""
    try:
        # In production, verify JWT token here
        return {"user_id": "demo_user", "tier": "premium", "permissions": ["read", "write"]}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize Redis connection
        await cache_manager.initialize()
        
        # Start data ingestion
        await data_ingestion.start()
        
        # Start stream processor
        await stream_processor.start()
        
        # Initialize agent connections
        await multi_agent_coordinator.initialize_agents()
        
        # Start background tasks
        asyncio.create_task(background_health_monitor())
        asyncio.create_task(background_cache_cleanup())
        asyncio.create_task(background_metrics_collection())
        
        logger.info("Real-time processing service started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        await stream_processor.stop()
        await data_ingestion.stop()
        await cache_manager.close()
        await websocket_manager.close_all_connections()
        
        logger.info("Real-time processing service shutdown complete")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

@app.get("/health")
async def health_check():
    """Comprehensive health check for all components"""
    try:
        # Check all agent health
        agent_health = await multi_agent_coordinator.check_all_agents_health()
        
        # Check stream processor
        stream_health = await stream_processor.health_check()
        
        # Check data ingestion
        ingestion_health = await data_ingestion.health_check()
        
        # Check cache
        cache_health = await cache_manager.health_check()
        
        # Overall health determination
        all_healthy = all([
            agent_health.get("overall_healthy", False),
            stream_health == "healthy",
            ingestion_health == "healthy",
            cache_health == "healthy"
        ])
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "agents": agent_health,
                "stream_processor": stream_health,
                "data_ingestion": ingestion_health,
                "cache_manager": cache_health,
                "websocket_manager": "healthy"
            },
            "metrics": await metrics.get_current_stats(),
            "active_connections": websocket_manager.get_connection_count()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")

@app.post("/intelligence/unified", response_model=UnifiedIntelligenceResponse)
async def get_unified_intelligence(
    request: UnifiedIntelligenceRequest,
    background_tasks: BackgroundTasks,
    user_info: dict = Depends(verify_token)
):
    """Get unified multi-agent trading intelligence"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Processing unified intelligence request {request_id} for symbols: {request.symbols}")
        
        # Route request to appropriate agents
        routing_plan = await request_router.create_routing_plan(request, user_info)
        
        # Check cache for recent results
        cache_key = await cache_manager.generate_cache_key(request, routing_plan)
        cached_result = await cache_manager.get_cached_result(cache_key)
        
        if cached_result and not request.real_time_data:
            logger.info(f"Returning cached result for request {request_id}")
            cached_result["request_id"] = request_id
            return UnifiedIntelligenceResponse(**cached_result)
        
        # Get real-time market data
        market_data = await data_ingestion.get_current_market_data(request.symbols)
        
        # Coordinate agents based on routing plan
        agent_responses = await multi_agent_coordinator.execute_coordinated_request(
            routing_plan, market_data, request_id
        )
        
        # Aggregate responses
        unified_response = await response_aggregator.aggregate_agent_responses(
            agent_responses, request, market_data
        )
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Prepare final response
        final_response = UnifiedIntelligenceResponse(
            request_id=request_id,
            symbols=request.symbols,
            analysis_type=request.analysis_type,
            processing_time_ms=processing_time_ms,
            agents_used=list(agent_responses.keys()),
            sentiment_analysis=unified_response.get("sentiment_analysis"),
            price_forecasts=unified_response.get("price_forecasts"),
            strategy_recommendations=unified_response.get("strategy_recommendations"),
            explanations=unified_response.get("explanations"),
            market_context=unified_response.get("market_context", {}),
            confidence_score=unified_response.get("confidence_score", 0.5),
            alerts=unified_response.get("alerts", []),
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Cache result for future requests
        if request.analysis_type in ["standard", "comprehensive"]:
            background_tasks.add_task(
                cache_manager.cache_result,
                cache_key, final_response.dict(), ttl=300  # 5 minutes
            )
        
        # Record metrics
        background_tasks.add_task(
            metrics.record_unified_request,
            request_id, request.symbols, processing_time_ms, 
            len(agent_responses), unified_response.get("confidence_score", 0.5)
        )
        
        # Broadcast to WebSocket subscribers
        if websocket_manager.has_subscribers(request.symbols):
            background_tasks.add_task(
                websocket_manager.broadcast_intelligence_update,
                request.symbols, final_response.dict()
            )
        
        return final_response
        
    except Exception as e:
        logger.error(f"Unified intelligence request failed: {e}")
        raise HTTPException(status_code=500, detail="Intelligence processing failed")

@app.post("/intelligence/stream/start")
async def start_intelligence_stream(
    request: StreamingDataRequest,
    user_info: dict = Depends(verify_token)
):
    """Start streaming intelligence updates"""
    try:
        stream_id = str(uuid.uuid4())
        
        # Register stream
        await stream_processor.register_stream(
            stream_id, request.user_id, request.symbols, 
            request.data_types, request.update_frequency
        )
        
        return {
            "stream_id": stream_id,
            "symbols": request.symbols,
            "data_types": request.data_types,
            "update_frequency": request.update_frequency,
            "status": "started",
            "websocket_url": f"ws://localhost:8008/ws/intelligence/{stream_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start intelligence stream: {e}")
        raise HTTPException(status_code=500, detail="Stream initialization failed")

@app.delete("/intelligence/stream/{stream_id}")
async def stop_intelligence_stream(
    stream_id: str,
    user_info: dict = Depends(verify_token)
):
    """Stop streaming intelligence updates"""
    try:
        await stream_processor.unregister_stream(stream_id)
        
        return {
            "stream_id": stream_id,
            "status": "stopped"
        }
        
    except Exception as e:
        logger.error(f"Failed to stop intelligence stream: {e}")
        raise HTTPException(status_code=404, detail="Stream not found")

@app.websocket("/ws/intelligence/{stream_id}")
async def intelligence_websocket(websocket: WebSocket, stream_id: str):
    """WebSocket endpoint for real-time intelligence streaming"""
    await websocket_manager.handle_intelligence_connection(websocket, stream_id)

@app.websocket("/ws/market/{symbols}")
async def market_data_websocket(websocket: WebSocket, symbols: str):
    """WebSocket endpoint for real-time market data"""
    symbol_list = symbols.split(",")
    await websocket_manager.handle_market_data_connection(websocket, symbol_list)

@app.get("/agents/status")
async def get_agents_status(user_info: dict = Depends(verify_token)):
    """Get status of all agents"""
    try:
        return await multi_agent_coordinator.get_detailed_agent_status()
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        raise HTTPException(status_code=500, detail="Agent status unavailable")

@app.post("/agents/{agent_name}/restart")
async def restart_agent(
    agent_name: str,
    user_info: dict = Depends(verify_token)
):
    """Restart a specific agent"""
    try:
        if "admin" not in user_info.get("permissions", []):
            raise HTTPException(status_code=403, detail="Admin permission required")
        
        result = await multi_agent_coordinator.restart_agent(agent_name)
        return result
        
    except Exception as e:
        logger.error(f"Failed to restart agent {agent_name}: {e}")
        raise HTTPException(status_code=500, detail="Agent restart failed")

@app.get("/cache/status")
async def get_cache_status(user_info: dict = Depends(verify_token)):
    """Get cache performance statistics"""
    try:
        return await cache_manager.get_cache_statistics()
    except Exception as e:
        logger.error(f"Failed to get cache status: {e}")
        raise HTTPException(status_code=500, detail="Cache status unavailable")

@app.delete("/cache/clear")
async def clear_cache(
    user_info: dict = Depends(verify_token)
):
    """Clear all cached data"""
    try:
        if "admin" not in user_info.get("permissions", []):
            raise HTTPException(status_code=403, detail="Admin permission required")
        
        await cache_manager.clear_all_cache()
        return {"status": "cache_cleared", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail="Cache clear failed")

@app.get("/metrics/performance")
async def get_performance_metrics(user_info: dict = Depends(verify_token)):
    """Get detailed performance metrics"""
    try:
        return await metrics.get_detailed_performance_metrics()
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")

@app.get("/load-balancer/status")
async def get_load_balancer_status(user_info: dict = Depends(verify_token)):
    """Get load balancer status and routing statistics"""
    try:
        return await load_balancer.get_status()
    except Exception as e:
        logger.error(f"Failed to get load balancer status: {e}")
        raise HTTPException(status_code=500, detail="Load balancer status unavailable")

@app.post("/alerts/subscribe")
async def subscribe_to_alerts(
    symbols: List[str],
    alert_types: List[str],
    user_info: dict = Depends(verify_token)
):
    """Subscribe to real-time alerts for symbols"""
    try:
        subscription_id = str(uuid.uuid4())
        
        await stream_processor.register_alert_subscription(
            subscription_id, user_info["user_id"], symbols, alert_types
        )
        
        return {
            "subscription_id": subscription_id,
            "symbols": symbols,
            "alert_types": alert_types,
            "status": "subscribed"
        }
        
    except Exception as e:
        logger.error(f"Failed to subscribe to alerts: {e}")
        raise HTTPException(status_code=500, detail="Alert subscription failed")

@app.delete("/alerts/unsubscribe/{subscription_id}")
async def unsubscribe_from_alerts(
    subscription_id: str,
    user_info: dict = Depends(verify_token)
):
    """Unsubscribe from alerts"""
    try:
        await stream_processor.unregister_alert_subscription(subscription_id)
        return {"subscription_id": subscription_id, "status": "unsubscribed"}
        
    except Exception as e:
        logger.error(f"Failed to unsubscribe from alerts: {e}")
        raise HTTPException(status_code=404, detail="Subscription not found")

# Background tasks
async def background_health_monitor():
    """Background task to monitor agent health"""
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            # Check agent health
            unhealthy_agents = await multi_agent_coordinator.identify_unhealthy_agents()
            
            if unhealthy_agents:
                logger.warning(f"Unhealthy agents detected: {unhealthy_agents}")
                
                # Attempt to restart unhealthy agents
                for agent_name in unhealthy_agents:
                    try:
                        await multi_agent_coordinator.restart_agent(agent_name)
                        logger.info(f"Restarted unhealthy agent: {agent_name}")
                    except Exception as e:
                        logger.error(f"Failed to restart agent {agent_name}: {e}")
            
        except Exception as e:
            logger.error(f"Health monitor error: {e}")

async def background_cache_cleanup():
    """Background task to clean expired cache entries"""
    while True:
        try:
            await asyncio.sleep(300)  # Clean every 5 minutes
            await cache_manager.cleanup_expired_entries()
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

async def background_metrics_collection():
    """Background task to collect and aggregate metrics"""
    while True:
        try:
            await asyncio.sleep(60)  # Collect every minute
            await metrics.collect_agent_metrics()
            await metrics.collect_system_metrics()
            
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8008,
        reload=False,
        workers=1,
        log_level="info"
    )

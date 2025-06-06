"""
Enhanced FastAPI Gateway - Main Application
Central API hub with agent orchestration and virtual trading
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import asyncio
import time
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
import aioredis
import websockets
import httpx

from .src.agent_orchestrator import AgentOrchestrator
from .src.virtual_trading import VirtualTradingEngine
from .src.portfolio_manager import PortfolioManager
from .src.pnl_engine import PnLEngine
from .src.risk_manager import RiskManager
from .src.market_data import MarketDataManager
from .src.websocket_handler import WebSocketHandler
from .src.auth_manager import AuthManager
from .src.monitoring import GatewayMonitoring
from .src.rate_limiter import RateLimiter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced MIP API Gateway",
    description="Central API hub with agent orchestration and virtual trading",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
agent_orchestrator = AgentOrchestrator()
virtual_trading = VirtualTradingEngine()
portfolio_manager = PortfolioManager()
pnl_engine = PnLEngine()
risk_manager = RiskManager()
market_data = MarketDataManager()
websocket_handler = WebSocketHandler()
auth_manager = AuthManager()
monitoring = GatewayMonitoring()
rate_limiter = RateLimiter()

# Pydantic models
class AgentAnalysisRequest(BaseModel):
    user_id: str
    symbols: List[str] = Field(..., description="Symbols to analyze")
    agents: List[str] = Field(default=["sentiment", "forecasting", "strategy"], description="Agents to use")
    analysis_depth: str = Field(default="standard", description="quick, standard, comprehensive")
    include_explanations: bool = Field(default=True)
    max_cost_usd: float = Field(default=1.0, description="Maximum analysis cost")
    
class VirtualTradeRequest(BaseModel):
    user_id: str
    portfolio_id: str
    symbol: str
    action: str = Field(..., description="buy, sell, close")
    quantity: int = Field(..., gt=0)
    order_type: str = Field(default="market", description="market, limit, stop")
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = Field(default="day", description="day, gtc, ioc")
    
class PortfolioCreateRequest(BaseModel):
    user_id: str
    name: str
    initial_balance: float = Field(default=100000.0, gt=0)
    risk_tolerance: str = Field(default="medium", description="low, medium, high")
    max_position_size: float = Field(default=0.1, description="Max position as % of portfolio")
    
class AgentAnalysisResponse(BaseModel):
    request_id: str
    symbols: List[str]
    agents_used: List[str]
    processing_time_ms: int
    total_cost_usd: float
    sentiment_analysis: Optional[Dict[str, Any]] = None
    price_forecasts: Optional[Dict[str, Any]] = None
    strategy_recommendations: Optional[Dict[str, Any]] = None
    explanations: Optional[Dict[str, Any]] = None
    overall_confidence: float
    timestamp: str

class VirtualTradeResponse(BaseModel):
    trade_id: str
    user_id: str
    portfolio_id: str
    symbol: str
    action: str
    quantity: int
    executed_price: float
    total_value: float
    commission: float
    timestamp: str
    status: str

class PortfolioSnapshot(BaseModel):
    portfolio_id: str
    user_id: str
    name: str
    total_value: float
    cash_balance: float
    total_pnl: float
    day_pnl: float
    positions: List[Dict[str, Any]]
    risk_metrics: Dict[str, Any]
    last_updated: str

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and extract user info"""
    try:
        user_info = await auth_manager.verify_token(credentials.credentials)
        return user_info
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

# Rate limiting dependency
async def check_rate_limit(user_info: dict = Depends(verify_token)):
    """Check if user is within rate limits"""
    try:
        if not await rate_limiter.check_rate_limit(user_info["user_id"], user_info.get("tier", "free")):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        return user_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rate limit check failed: {e}")
        return user_info

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize all components
        await agent_orchestrator.initialize()
        await virtual_trading.initialize()
        await portfolio_manager.initialize()
        await pnl_engine.initialize()
        await risk_manager.initialize()
        await market_data.initialize()
        await websocket_handler.initialize()
        await auth_manager.initialize()
        await monitoring.initialize()
        await rate_limiter.initialize()
        
        # Start background tasks
        asyncio.create_task(background_pnl_updates())
        asyncio.create_task(background_risk_monitoring())
        asyncio.create_task(background_market_data_updates())
        asyncio.create_task(background_health_monitoring())
        
        logger.info("Enhanced API Gateway started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        await websocket_handler.close_all_connections()
        await agent_orchestrator.close()
        await virtual_trading.close()
        await monitoring.close()
        
        logger.info("Enhanced API Gateway shutdown complete")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# Health and status endpoints
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "agent_orchestrator": await agent_orchestrator.health_check(),
                "virtual_trading": await virtual_trading.health_check(),
                "portfolio_manager": await portfolio_manager.health_check(),
                "pnl_engine": await pnl_engine.health_check(),
                "market_data": await market_data.health_check(),
                "websocket_handler": await websocket_handler.health_check()
            },
            "metrics": await monitoring.get_current_metrics(),
            "active_connections": websocket_handler.get_connection_count()
        }
        
        # Check if any component is unhealthy
        unhealthy_components = [
            name for name, status in health_status["components"].items()
            if status != "healthy"
        ]
        
        if unhealthy_components:
            health_status["status"] = "degraded"
            health_status["unhealthy_components"] = unhealthy_components
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")

@app.get("/status")
async def get_system_status(user_info: dict = Depends(verify_token)):
    """Get detailed system status"""
    try:
        return {
            "agents": await agent_orchestrator.get_agent_status(),
            "trading": await virtual_trading.get_system_status(),
            "portfolios": await portfolio_manager.get_portfolio_statistics(),
            "market_data": await market_data.get_status(),
            "performance": await monitoring.get_performance_metrics(),
            "rate_limits": await rate_limiter.get_user_limits(user_info["user_id"])
        }
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Status retrieval failed")

# Agent orchestration endpoints
@app.post("/agents/analyze", response_model=AgentAnalysisResponse)
async def analyze_with_agents(
    request: AgentAnalysisRequest,
    background_tasks: BackgroundTasks,
    user_info: dict = Depends(check_rate_limit)
):
    """Get multi-agent analysis for symbols"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting agent analysis {request_id} for user {user_info['user_id']}")
        
        # Execute agent analysis
        analysis_result = await agent_orchestrator.execute_analysis(
            request_id=request_id,
            user_id=user_info["user_id"],
            symbols=request.symbols,
            agents=request.agents,
            analysis_depth=request.analysis_depth,
            include_explanations=request.include_explanations,
            max_cost_usd=request.max_cost_usd,
            user_tier=user_info.get("tier", "free")
        )
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Create response
        response = AgentAnalysisResponse(
            request_id=request_id,
            symbols=request.symbols,
            agents_used=analysis_result["agents_used"],
            processing_time_ms=processing_time_ms,
            total_cost_usd=analysis_result["total_cost"],
            sentiment_analysis=analysis_result.get("sentiment_analysis"),
            price_forecasts=analysis_result.get("price_forecasts"),
            strategy_recommendations=analysis_result.get("strategy_recommendations"),
            explanations=analysis_result.get("explanations"),
            overall_confidence=analysis_result["overall_confidence"],
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Record metrics
        background_tasks.add_task(
            monitoring.record_agent_analysis,
            request_id, user_info["user_id"], request.symbols,
            processing_time_ms, analysis_result["total_cost"]
        )
        
        # Broadcast to WebSocket subscribers
        if websocket_handler.has_subscribers(user_info["user_id"]):
            background_tasks.add_task(
                websocket_handler.broadcast_analysis_result,
                user_info["user_id"], response.dict()
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Agent analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.get("/agents/status")
async def get_agent_status(user_info: dict = Depends(verify_token)):
    """Get status of all AI agents"""
    try:
        return await agent_orchestrator.get_detailed_agent_status()
    except Exception as e:
        logger.error(f"Agent status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Agent status unavailable")

# Portfolio management endpoints
@app.post("/portfolios/create")
async def create_portfolio(
    request: PortfolioCreateRequest,
    user_info: dict = Depends(check_rate_limit)
):
    """Create a new virtual trading portfolio"""
    try:
        # Verify user owns the portfolio
        if request.user_id != user_info["user_id"] and user_info.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        portfolio = await portfolio_manager.create_portfolio(
            user_id=request.user_id,
            name=request.name,
            initial_balance=request.initial_balance,
            risk_tolerance=request.risk_tolerance,
            max_position_size=request.max_position_size
        )
        
        return portfolio
        
    except Exception as e:
        logger.error(f"Portfolio creation failed: {e}")
        raise HTTPException(status_code=500, detail="Portfolio creation failed")

@app.get("/portfolios/{portfolio_id}", response_model=PortfolioSnapshot)
async def get_portfolio(
    portfolio_id: str,
    user_info: dict = Depends(verify_token)
):
    """Get portfolio details and current positions"""
    try:
        portfolio = await portfolio_manager.get_portfolio(portfolio_id)
        
        # Verify user owns the portfolio
        if portfolio["user_id"] != user_info["user_id"] and user_info.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get current market data for positions
        symbols = [pos["symbol"] for pos in portfolio["positions"]]
        if symbols:
            market_prices = await market_data.get_current_prices(symbols)
            
            # Update portfolio with current prices
            portfolio = await portfolio_manager.update_portfolio_with_prices(
                portfolio_id, market_prices
            )
        
        return PortfolioSnapshot(**portfolio)
        
    except Exception as e:
        logger.error(f"Portfolio retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Portfolio retrieval failed")

@app.get("/portfolios/user/{user_id}")
async def get_user_portfolios(
    user_id: str,
    user_info: dict = Depends(verify_token)
):
    """Get all portfolios for a user"""
    try:
        # Verify user access
        if user_id != user_info["user_id"] and user_info.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        portfolios = await portfolio_manager.get_user_portfolios(user_id)
        return {"portfolios": portfolios}
        
    except Exception as e:
        logger.error(f"User portfolios retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Portfolio retrieval failed")

# Virtual trading endpoints
@app.post("/trading/execute", response_model=VirtualTradeResponse)
async def execute_virtual_trade(
    request: VirtualTradeRequest,
    background_tasks: BackgroundTasks,
    user_info: dict = Depends(check_rate_limit)
):
    """Execute a virtual trade"""
    try:
        # Verify user owns the portfolio
        portfolio = await portfolio_manager.get_portfolio(request.portfolio_id)
        if portfolio["user_id"] != user_info["user_id"] and user_info.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get current market price
        current_price = await market_data.get_current_price(request.symbol)
        
        # Pre-trade risk checks
        risk_check = await risk_manager.validate_trade(
            portfolio_id=request.portfolio_id,
            symbol=request.symbol,
            action=request.action,
            quantity=request.quantity,
            price=current_price,
            portfolio=portfolio
        )
        
        if not risk_check["approved"]:
            raise HTTPException(status_code=400, detail=f"Trade rejected: {risk_check['reason']}")
        
        # Execute the trade
        trade_result = await virtual_trading.execute_trade(
            user_id=request.user_id,
            portfolio_id=request.portfolio_id,
            symbol=request.symbol,
            action=request.action,
            quantity=request.quantity,
            order_type=request.order_type,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            current_market_price=current_price
        )
        
        # Update portfolio
        await portfolio_manager.update_portfolio_after_trade(
            request.portfolio_id, trade_result
        )
        
        # Create response
        response = VirtualTradeResponse(
            trade_id=trade_result["trade_id"],
            user_id=request.user_id,
            portfolio_id=request.portfolio_id,
            symbol=request.symbol,
            action=request.action,
            quantity=request.quantity,
            executed_price=trade_result["executed_price"],
            total_value=trade_result["total_value"],
            commission=trade_result["commission"],
            timestamp=trade_result["timestamp"],
            status=trade_result["status"]
        )
        
        # Record metrics and update P&L
        background_tasks.add_task(
            monitoring.record_trade_execution,
            trade_result["trade_id"], request.user_id, request.symbol,
            trade_result["total_value"]
        )
        
        background_tasks.add_task(
            pnl_engine.update_position_pnl,
            request.portfolio_id, request.symbol
        )
        
        # Broadcast to WebSocket subscribers
        if websocket_handler.has_subscribers(request.user_id):
            background_tasks.add_task(
                websocket_handler.broadcast_trade_execution,
                request.user_id, response.dict()
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        raise HTTPException(status_code=500, detail="Trade execution failed")

@app.get("/trading/history/{portfolio_id}")
async def get_trading_history(
    portfolio_id: str,
    limit: int = 100,
    offset: int = 0,
    user_info: dict = Depends(verify_token)
):
    """Get trading history for a portfolio"""
    try:
        # Verify user owns the portfolio
        portfolio = await portfolio_manager.get_portfolio(portfolio_id)
        if portfolio["user_id"] != user_info["user_id"] and user_info.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        trades = await virtual_trading.get_trading_history(
            portfolio_id, limit=limit, offset=offset
        )
        
        return {"trades": trades, "total_count": len(trades)}
        
    except Exception as e:
        logger.error(f"Trading history retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Trading history retrieval failed")

# P&L and performance endpoints
@app.get("/pnl/portfolio/{portfolio_id}")
async def get_portfolio_pnl(
    portfolio_id: str,
    timeframe: str = "1d",
    user_info: dict = Depends(verify_token)
):
    """Get P&L breakdown for a portfolio"""
    try:
        # Verify user owns the portfolio
        portfolio = await portfolio_manager.get_portfolio(portfolio_id)
        if portfolio["user_id"] != user_info["user_id"] and user_info.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        pnl_data = await pnl_engine.get_portfolio_pnl(portfolio_id, timeframe)
        return pnl_data
        
    except Exception as e:
        logger.error(f"P&L retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="P&L retrieval failed")

@app.get("/pnl/position/{portfolio_id}/{symbol}")
async def get_position_pnl(
    portfolio_id: str,
    symbol: str,
    user_info: dict = Depends(verify_token)
):
    """Get P&L for a specific position"""
    try:
        # Verify user owns the portfolio
        portfolio = await portfolio_manager.get_portfolio(portfolio_id)
        if portfolio["user_id"] != user_info["user_id"] and user_info.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        pnl_data = await pnl_engine.get_position_pnl(portfolio_id, symbol)
        return pnl_data
        
    except Exception as e:
        logger.error(f"Position P&L retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Position P&L retrieval failed")

# Risk management endpoints
@app.get("/risk/portfolio/{portfolio_id}")
async def get_portfolio_risk_metrics(
    portfolio_id: str,
    user_info: dict = Depends(verify_token)
):
    """Get risk metrics for a portfolio"""
    try:
        # Verify user owns the portfolio
        portfolio = await portfolio_manager.get_portfolio(portfolio_id)
        if portfolio["user_id"] != user_info["user_id"] and user_info.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        risk_metrics = await risk_manager.calculate_portfolio_risk(portfolio_id)
        return risk_metrics
        
    except Exception as e:
        logger.error(f"Risk metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Risk metrics retrieval failed")

@app.post("/risk/alerts/subscribe")
async def subscribe_to_risk_alerts(
    portfolio_id: str,
    alert_types: List[str],
    user_info: dict = Depends(check_rate_limit)
):
    """Subscribe to risk alerts for a portfolio"""
    try:
        # Verify user owns the portfolio
        portfolio = await portfolio_manager.get_portfolio(portfolio_id)
        if portfolio["user_id"] != user_info["user_id"] and user_info.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        subscription = await risk_manager.subscribe_to_alerts(
            user_id=user_info["user_id"],
            portfolio_id=portfolio_id,
            alert_types=alert_types
        )
        
        return subscription
        
    except Exception as e:
        logger.error(f"Risk alert subscription failed: {e}")
        raise HTTPException(status_code=500, detail="Risk alert subscription failed")

# Market data endpoints
@app.get("/market/prices")
async def get_market_prices(
    symbols: str,
    user_info: dict = Depends(verify_token)
):
    """Get current market prices for symbols"""
    try:
        symbol_list = symbols.split(",")
        prices = await market_data.get_current_prices(symbol_list)
        return {"prices": prices, "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Market prices retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Market prices retrieval failed")

@app.get("/market/quotes/{symbol}")
async def get_symbol_quote(
    symbol: str,
    user_info: dict = Depends(verify_token)
):
    """Get detailed quote for a symbol"""
    try:
        quote = await market_data.get_detailed_quote(symbol)
        return quote
        
    except Exception as e:
        logger.error(f"Symbol quote retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Quote retrieval failed")

# WebSocket endpoints
@app.websocket("/ws/portfolio/{portfolio_id}")
async def portfolio_websocket(websocket: WebSocket, portfolio_id: str):
    """WebSocket endpoint for real-time portfolio updates"""
    await websocket_handler.handle_portfolio_connection(websocket, portfolio_id)

@app.websocket("/ws/market/{symbols}")
async def market_data_websocket(websocket: WebSocket, symbols: str):
    """WebSocket endpoint for real-time market data"""
    symbol_list = symbols.split(",")
    await websocket_handler.handle_market_data_connection(websocket, symbol_list)

@app.websocket("/ws/user/{user_id}")
async def user_websocket(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for user-specific updates"""
    await websocket_handler.handle_user_connection(websocket, user_id)

# Admin endpoints
@app.get("/admin/metrics")
async def get_admin_metrics(user_info: dict = Depends(verify_token)):
    """Get system-wide metrics (admin only)"""
    try:
        if user_info.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        return await monitoring.get_admin_metrics()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics retrieval failed")

@app.post("/admin/maintenance")
async def trigger_maintenance(
    maintenance_type: str,
    user_info: dict = Depends(verify_token)
):
    """Trigger system maintenance tasks (admin only)"""
    try:
        if user_info.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        if maintenance_type == "cache_clear":
            await agent_orchestrator.clear_cache()
        elif maintenance_type == "pnl_recalculation":
            await pnl_engine.recalculate_all_pnl()
        elif maintenance_type == "risk_reassessment":
            await risk_manager.reassess_all_portfolios()
        else:
            raise HTTPException(status_code=400, detail="Invalid maintenance type")
        
        return {"status": "maintenance_triggered", "type": maintenance_type}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Maintenance trigger failed: {e}")
        raise HTTPException(status_code=500, detail="Maintenance failed")

# Background tasks
async def background_pnl_updates():
    """Background task to update P&L for all portfolios"""
    while True:
        try:
            await asyncio.sleep(30)  # Update every 30 seconds
            
            active_portfolios = await portfolio_manager.get_active_portfolios()
            
            for portfolio_id in active_portfolios:
                try:
                    await pnl_engine.update_portfolio_pnl(portfolio_id)
                except Exception as e:
                    logger.error(f"P&L update failed for portfolio {portfolio_id}: {e}")
            
        except Exception as e:
            logger.error(f"Background P&L updates error: {e}")

async def background_risk_monitoring():
    """Background task to monitor portfolio risks"""
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            
            active_portfolios = await portfolio_manager.get_active_portfolios()
            
            for portfolio_id in active_portfolios:
                try:
                    await risk_manager.monitor_portfolio_risk(portfolio_id)
                except Exception as e:
                    logger.error(f"Risk monitoring failed for portfolio {portfolio_id}: {e}")
            
        except Exception as e:
            logger.error(f"Background risk monitoring error: {e}")

async def background_market_data_updates():
    """Background task to update market data"""
    while True:
        try:
            await asyncio.sleep(5)  # Update every 5 seconds
            
            # Get all symbols from active portfolios
            symbols = await portfolio_manager.get_all_symbols()
            
            if symbols:
                prices = await market_data.update_prices(symbols)
                
                # Broadcast price updates to WebSocket clients
                await websocket_handler.broadcast_price_updates(prices)
            
        except Exception as e:
            logger.error(f"Background market data updates error: {e}")

async def background_health_monitoring():
    """Background task to monitor system health"""
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            # Check component health
            health_status = await health_check()
            
            if health_status["status"] != "healthy":
                logger.warning(f"System health degraded: {health_status}")
                
                # Send alerts to admin users
                await websocket_handler.broadcast_admin_alert({
                    "type": "health_alert",
                    "status": health_status["status"],
                    "components": health_status.get("unhealthy_components", []),
                    "timestamp": datetime.utcnow().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Background health monitoring error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )

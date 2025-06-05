"""
TFT Price Forecasting Agent - Main FastAPI Service
Handles multi-horizon financial time series forecasting with attention mechanisms
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import asyncio
import time
import hashlib
import json
import logging
import os
import numpy as np
from datetime import datetime, timedelta

from .src.tft_engine import TFTForecastingEngine
from .src.data_preprocessor import FinancialDataPreprocessor
from .src.feature_engineer import MultiScaleFeatureEngineer
from .src.market_regime_detector import MarketRegimeDetector
from .src.options_integrator import OptionsGreeksIntegrator
from .src.risk_adjuster import RiskAdjustedForecaster
from .src.model_manager import TFTModelManager
from .src.monitoring import ForecastingMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TFT Price Forecasting Agent",
    description="Financial time series forecasting with Temporal Fusion Transformers",
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
tft_engine = TFTForecastingEngine()
data_preprocessor = FinancialDataPreprocessor()
feature_engineer = MultiScaleFeatureEngineer()
regime_detector = MarketRegimeDetector()
options_integrator = OptionsGreeksIntegrator()
risk_adjuster = RiskAdjustedForecaster()
model_manager = TFTModelManager()
metrics = ForecastingMetrics()

# Pydantic models
class ForecastRequest(BaseModel):
    user_id: str
    symbol: str
    forecast_horizons: List[int] = Field(default=[1, 5, 10, 21], description="Days to forecast")
    include_options_greeks: bool = Field(default=True, description="Include options Greeks in forecast")
    risk_adjustment: bool = Field(default=True, description="Apply risk adjustments")
    confidence_intervals: List[float] = Field(default=[0.68, 0.95], description="Confidence levels")
    market_context: Optional[Dict[str, Any]] = None
    
class BatchForecastRequest(BaseModel):
    symbols: List[str]
    forecast_horizons: List[int] = Field(default=[1, 5, 10, 21])
    batch_id: Optional[str] = None

class ForecastResponse(BaseModel):
    forecast_id: str
    symbol: str
    forecasts: Dict[str, Any]  # Keyed by horizon
    confidence_intervals: Dict[str, Any]
    market_regime: Dict[str, Any]
    options_impact: Optional[Dict[str, Any]]
    risk_metrics: Dict[str, Any]
    model_performance: Dict[str, Any]
    processing_time_ms: int
    timestamp: str

class ModelRetrainRequest(BaseModel):
    symbols: List[str]
    retrain_type: str = Field(default="incremental", description="full or incremental")
    priority: str = Field(default="normal", description="low, normal, high")

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and extract user info"""
    try:
        # In production, verify JWT token here
        return {"user_id": "demo_user", "tier": "premium"}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

@app.get("/health")
async def health_check():
    """Health check endpoint with TFT model status"""
    try:
        # Check all components
        tft_status = await tft_engine.health_check()
        model_status = await model_manager.get_model_status()
        data_status = await data_preprocessor.health_check()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "tft_engine": tft_status,
                "model_manager": model_status,
                "data_preprocessor": data_status,
                "feature_engineer": "healthy",
                "regime_detector": "healthy"
            },
            "models_loaded": await model_manager.get_loaded_models(),
            "metrics": await metrics.get_current_stats()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/forecast/generate", response_model=ForecastResponse)
async def generate_forecast(
    request: ForecastRequest,
    background_tasks: BackgroundTasks,
    user_info: dict = Depends(verify_token)
):
    """Generate multi-horizon price forecasts with attention analysis"""
    start_time = time.time()
    forecast_id = hashlib.md5(f"{request.symbol}{request.user_id}{time.time()}".encode()).hexdigest()
    
    try:
        # Validate symbol and horizons
        if not await data_preprocessor.validate_symbol(request.symbol):
            raise HTTPException(status_code=400, detail=f"Invalid or unsupported symbol: {request.symbol}")
        
        if max(request.forecast_horizons) > 252:  # 1 year max
            raise HTTPException(status_code=400, detail="Maximum forecast horizon is 252 days")
        
        # Get and preprocess historical data
        logger.info(f"Generating forecast for {request.symbol}, horizons: {request.forecast_horizons}")
        
        historical_data = await data_preprocessor.get_historical_data(
            symbol=request.symbol,
            lookback_days=1260  # 5 years of data
        )
        
        # Feature engineering
        features = await feature_engineer.engineer_features(
            historical_data=historical_data,
            symbol=request.symbol,
            market_context=request.market_context
        )
        
        # Market regime detection
        market_regime = await regime_detector.detect_regime(
            features=features,
            symbol=request.symbol
        )
        
        # Generate forecasts for each horizon
        forecasts = {}
        confidence_intervals = {}
        
        for horizon in request.forecast_horizons:
            horizon_forecast = await tft_engine.generate_forecast(
                features=features,
                horizon=horizon,
                symbol=request.symbol,
                market_regime=market_regime
            )
            
            forecasts[f"{horizon}d"] = {
                "price_forecast": horizon_forecast.price_prediction,
                "volatility_forecast": horizon_forecast.volatility_prediction,
                "direction_probability": horizon_forecast.direction_probability,
                "attention_weights": horizon_forecast.attention_weights,
                "feature_importance": horizon_forecast.feature_importance
            }
            
            # Calculate confidence intervals
            if request.confidence_intervals:
                intervals = await tft_engine.calculate_confidence_intervals(
                    forecast=horizon_forecast,
                    confidence_levels=request.confidence_intervals
                )
                confidence_intervals[f"{horizon}d"] = intervals
        
        # Options Greeks integration
        options_impact = None
        if request.include_options_greeks:
            options_impact = await options_integrator.integrate_options_data(
                symbol=request.symbol,
                price_forecasts=forecasts,
                market_regime=market_regime
            )
        
        # Risk adjustment
        risk_metrics = {}
        if request.risk_adjustment:
            risk_adjusted_forecasts = await risk_adjuster.adjust_forecasts(
                forecasts=forecasts,
                market_regime=market_regime,
                symbol=request.symbol
            )
            
            # Update forecasts with risk adjustments
            for horizon in forecasts:
                forecasts[horizon]["risk_adjusted_price"] = risk_adjusted_forecasts[horizon]["adjusted_price"]
                forecasts[horizon]["risk_score"] = risk_adjusted_forecasts[horizon]["risk_score"]
            
            risk_metrics = risk_adjusted_forecasts.get("overall_risk_metrics", {})
        
        # Model performance metrics
        model_performance = await tft_engine.get_model_performance(request.symbol)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Background tasks
        background_tasks.add_task(
            metrics.record_forecast,
            symbol=request.symbol,
            horizons=request.forecast_horizons,
            processing_time_ms=processing_time_ms,
            forecast_id=forecast_id
        )
        
        # Store forecast for future validation
        background_tasks.add_task(
            tft_engine.store_forecast_for_validation,
            forecast_id=forecast_id,
            symbol=request.symbol,
            forecasts=forecasts,
            timestamp=datetime.utcnow()
        )
        
        return ForecastResponse(
            forecast_id=forecast_id,
            symbol=request.symbol,
            forecasts=forecasts,
            confidence_intervals=confidence_intervals,
            market_regime=market_regime.to_dict(),
            options_impact=options_impact,
            risk_metrics=risk_metrics,
            model_performance=model_performance,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}")
        raise HTTPException(status_code=500, detail="Forecast generation service unavailable")

@app.post("/forecast/batch", response_model=List[ForecastResponse])
async def generate_batch_forecasts(
    request: BatchForecastRequest,
    background_tasks: BackgroundTasks,
    user_info: dict = Depends(verify_token)
):
    """Generate forecasts for multiple symbols in parallel"""
    
    if len(request.symbols) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 symbols per batch")
    
    batch_id = request.batch_id or hashlib.md5(f"batch_{time.time()}".encode()).hexdigest()
    
    try:
        # Create individual forecast requests
        forecast_requests = []
        for symbol in request.symbols:
            forecast_req = ForecastRequest(
                user_id=user_info["user_id"],
                symbol=symbol,
                forecast_horizons=request.forecast_horizons
            )
            forecast_requests.append(forecast_req)
        
        # Process requests in parallel
        tasks = []
        for req in forecast_requests:
            task = generate_forecast(req, background_tasks, user_info)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch forecast failed for {request.symbols[i]}: {result}")
                # Create error response
                error_response = ForecastResponse(
                    forecast_id=f"error_{i}",
                    symbol=request.symbols[i],
                    forecasts={},
                    confidence_intervals={},
                    market_regime={"error": str(result)},
                    options_impact=None,
                    risk_metrics={"error": str(result)},
                    model_performance={"error": str(result)},
                    processing_time_ms=0,
                    timestamp=datetime.utcnow().isoformat()
                )
                responses.append(error_response)
            else:
                responses.append(result)
        
        # Record batch metrics
        background_tasks.add_task(
            metrics.record_batch_forecast,
            batch_id=batch_id,
            symbols=request.symbols,
            successful_forecasts=len([r for r in results if not isinstance(r, Exception)])
        )
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch forecast failed: {e}")
        raise HTTPException(status_code=500, detail="Batch forecast service unavailable")

@app.get("/forecast/{forecast_id}/validation")
async def get_forecast_validation(forecast_id: str, user_info: dict = Depends(verify_token)):
    """Get validation results for a previous forecast"""
    try:
        validation_results = await tft_engine.get_forecast_validation(forecast_id)
        return validation_results
    except Exception as e:
        logger.error(f"Failed to get forecast validation: {e}")
        raise HTTPException(status_code=404, detail="Forecast validation not found")

@app.get("/model/{symbol}/performance")
async def get_model_performance(symbol: str, user_info: dict = Depends(verify_token)):
    """Get detailed model performance metrics for a symbol"""
    try:
        performance = await tft_engine.get_detailed_model_performance(symbol)
        return performance
    except Exception as e:
        logger.error(f"Failed to get model performance: {e}")
        raise HTTPException(status_code=404, detail="Model performance data not found")

@app.post("/model/retrain")
async def retrain_models(
    request: ModelRetrainRequest,
    background_tasks: BackgroundTasks,
    user_info: dict = Depends(verify_token)
):
    """Trigger model retraining for specified symbols"""
    try:
        retrain_id = hashlib.md5(f"retrain_{time.time()}".encode()).hexdigest()
        
        # Queue retraining task
        background_tasks.add_task(
            model_manager.retrain_models,
            symbols=request.symbols,
            retrain_type=request.retrain_type,
            retrain_id=retrain_id,
            priority=request.priority
        )
        
        return {
            "retrain_id": retrain_id,
            "symbols": request.symbols,
            "retrain_type": request.retrain_type,
            "status": "queued",
            "estimated_completion": (datetime.utcnow() + timedelta(hours=2)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise HTTPException(status_code=500, detail="Model retraining service unavailable")

@app.get("/model/retrain/{retrain_id}/status")
async def get_retrain_status(retrain_id: str, user_info: dict = Depends(verify_token)):
    """Get status of model retraining job"""
    try:
        status = await model_manager.get_retrain_status(retrain_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get retrain status: {e}")
        raise HTTPException(status_code=404, detail="Retrain job not found")

@app.get("/market/regime/{symbol}")
async def get_market_regime(symbol: str, user_info: dict = Depends(verify_token)):
    """Get current market regime analysis for a symbol"""
    try:
        # Get recent data for regime detection
        historical_data = await data_preprocessor.get_historical_data(
            symbol=symbol,
            lookback_days=252  # 1 year for regime detection
        )
        
        features = await feature_engineer.engineer_features(
            historical_data=historical_data,
            symbol=symbol
        )
        
        regime = await regime_detector.detect_regime(
            features=features,
            symbol=symbol
        )
        
        return {
            "symbol": symbol,
            "current_regime": regime.to_dict(),
            "regime_history": await regime_detector.get_regime_history(symbol),
            "transition_probabilities": await regime_detector.get_transition_probabilities(symbol),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Market regime analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Market regime analysis unavailable")

@app.get("/features/{symbol}/importance")
async def get_feature_importance(
    symbol: str,
    horizon: int = 5,
    user_info: dict = Depends(verify_token)
):
    """Get feature importance analysis for forecasting"""
    try:
        importance = await tft_engine.get_feature_importance(symbol, horizon)
        return {
            "symbol": symbol,
            "horizon": horizon,
            "feature_importance": importance,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Feature importance analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Feature importance analysis unavailable")

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
        port=8007,
        reload=False,
        workers=1,
        log_level="info"
    )

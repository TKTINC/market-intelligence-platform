# services/llama-explanation/src/main.py
import asyncio
import logging
import time
import json
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import psutil
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

from llama_engine import LlamaEngine
from load_balancer import LoadBalancer
from config import settings
from monitoring import setup_metrics, LlamaMetrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
llama_engine: Optional[LlamaEngine] = None
load_balancer: Optional[LoadBalancer] = None
metrics: Optional[LlamaMetrics] = None

class ExplanationRequest(BaseModel):
    """Request for explanation generation"""
    context: Dict[str, Any] = Field(..., description="Context for explanation")
    max_tokens: int = Field(default=256, ge=50, le=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="Generation temperature")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")
    priority: str = Field(default="normal", description="Request priority")

class ExplanationResponse(BaseModel):
    """Response from explanation generation"""
    explanation: str = Field(..., description="Generated explanation")
    tokens_used: int = Field(..., description="Number of tokens used")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    confidence_score: float = Field(..., description="Confidence in explanation quality")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, float]
    queue_depth: int
    uptime_seconds: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global llama_engine, load_balancer, metrics
    
    # Startup
    logger.info("Starting Llama 2-7B Explanation Service...")
    
    try:
        # Initialize metrics
        metrics = LlamaMetrics()
        setup_metrics()
        
        # Initialize Llama engine
        llama_engine = LlamaEngine()
        await llama_engine.initialize()
        
        # Initialize load balancer
        load_balancer = LoadBalancer(llama_engine)
        await load_balancer.start()
        
        # Warm up model
        await llama_engine.warmup()
        
        logger.info("Llama 2-7B service started successfully")
        
        yield
        
        # Shutdown
        logger.info("Shutting down Llama 2-7B service...")
        if load_balancer:
            await load_balancer.stop()
        if llama_engine:
            await llama_engine.shutdown()
        
    except Exception as e:
        logger.error(f"Failed to start Llama service: {str(e)}")
        raise

# Create FastAPI app
app = FastAPI(
    title="MIP Llama 2-7B Explanation Service",
    description="GPU-optimized Llama 2-7B for financial explanations",
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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    try:
        # Check if model is loaded
        model_loaded = llama_engine and llama_engine.is_ready()
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        
        # Get memory usage
        memory_info = psutil.virtual_memory()
        memory_usage = {
            "system_memory_percent": memory_info.percent,
            "system_memory_available_gb": memory_info.available / (1024**3)
        }
        
        if gpu_available:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_used = torch.cuda.memory_allocated(0)
            memory_usage.update({
                "gpu_memory_percent": (gpu_memory_used / gpu_memory) * 100,
                "gpu_memory_available_gb": (gpu_memory - gpu_memory_used) / (1024**3)
            })
        
        # Get queue depth
        queue_depth = load_balancer.get_queue_depth() if load_balancer else 0
        
        # Calculate uptime
        uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        
        status = "healthy" if model_loaded and gpu_available else "degraded"
        
        return HealthResponse(
            status=status,
            model_loaded=model_loaded,
            gpu_available=gpu_available,
            memory_usage=memory_usage,
            queue_depth=queue_depth,
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
async def generate_explanation(
    request: ExplanationRequest,
    background_tasks: BackgroundTasks
):
    """Generate explanation from context"""
    
    if not llama_engine or not llama_engine.is_ready():
        raise HTTPException(status_code=503, detail="Llama engine not ready")
    
    start_time = time.time()
    
    try:
        # Record request
        if metrics:
            metrics.record_request()
        
        # Route through load balancer
        result = await load_balancer.generate_explanation(
            context=request.context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=request.system_prompt,
            priority=request.priority
        )
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Record metrics
        if metrics:
            metrics.record_success(processing_time_ms, result.get('tokens_used', 0))
        
        # Log successful request
        logger.info(
            f"Generated explanation: {processing_time_ms}ms, "
            f"{result.get('tokens_used', 0)} tokens"
        )
        
        return ExplanationResponse(
            explanation=result['explanation'],
            tokens_used=result.get('tokens_used', 0),
            processing_time_ms=processing_time_ms,
            model_info={
                "model_name": "llama-2-7b-explanations",
                "quantization": "Q4_K_M",
                "gpu_layers": settings.N_GPU_LAYERS,
                "context_size": settings.N_CTX
            },
            confidence_score=result.get('confidence_score', 0.85)
        )
        
    except Exception as e:
        # Record error
        if metrics:
            metrics.record_error(str(e))
        
        logger.error(f"Explanation generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Explanation generation failed: {str(e)}"
        )

@app.post("/explain/batch")
async def generate_explanations_batch(
    requests: List[ExplanationRequest]
):
    """Generate multiple explanations in batch"""
    
    if not llama_engine or not llama_engine.is_ready():
        raise HTTPException(status_code=503, detail="Llama engine not ready")
    
    if len(requests) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(requests)} exceeds maximum {settings.MAX_BATCH_SIZE}"
        )
    
    results = []
    
    try:
        # Process batch through load balancer
        batch_results = await load_balancer.generate_explanations_batch([
            {
                'context': req.context,
                'max_tokens': req.max_tokens,
                'temperature': req.temperature,
                'system_prompt': req.system_prompt,
                'priority': req.priority
            }
            for req in requests
        ])
        
        # Format responses
        for i, result in enumerate(batch_results):
            if 'error' in result:
                results.append({'error': result['error'], 'index': i})
            else:
                results.append(ExplanationResponse(
                    explanation=result['explanation'],
                    tokens_used=result.get('tokens_used', 0),
                    processing_time_ms=result.get('processing_time_ms', 0),
                    model_info={
                        "model_name": "llama-2-7b-explanations",
                        "quantization": "Q4_K_M"
                    },
                    confidence_score=result.get('confidence_score', 0.85)
                ))
        
        return {"results": results, "total_processed": len(results)}
        
    except Exception as e:
        logger.error(f"Batch explanation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )

@app.get("/status")
async def get_status():
    """Get detailed service status"""
    try:
        status = {
            "service": "llama-explanation",
            "version": "1.0.0",
            "model_status": {
                "loaded": llama_engine.is_ready() if llama_engine else False,
                "model_name": "llama-2-7b-explanations",
                "quantization": "Q4_K_M",
                "gpu_layers": settings.N_GPU_LAYERS
            },
            "performance": {},
            "queue_info": {}
        }
        
        if llama_engine:
            status["performance"] = await llama_engine.get_performance_stats()
        
        if load_balancer:
            status["queue_info"] = load_balancer.get_queue_info()
        
        return status
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics"""
    return prometheus_client.generate_latest()

@app.post("/admin/warmup")
async def warmup_model():
    """Manually trigger model warmup"""
    if not llama_engine:
        raise HTTPException(status_code=503, detail="Llama engine not initialized")
    
    try:
        warmup_result = await llama_engine.warmup()
        return {
            "status": "completed",
            "warmup_time_ms": warmup_result.get('time_ms', 0),
            "iterations": warmup_result.get('iterations', 0)
        }
        
    except Exception as e:
        logger.error(f"Manual warmup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/reload")
async def reload_model():
    """Reload the Llama model"""
    if not llama_engine:
        raise HTTPException(status_code=503, detail="Llama engine not initialized")
    
    try:
        await llama_engine.reload()
        return {"status": "reloaded", "timestamp": time.time()}
        
    except Exception as e:
        logger.error(f"Model reload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/stats")
async def get_admin_stats():
    """Get detailed admin statistics"""
    try:
        stats = {}
        
        if llama_engine:
            stats["engine"] = await llama_engine.get_detailed_stats()
        
        if load_balancer:
            stats["load_balancer"] = load_balancer.get_detailed_stats()
        
        if metrics:
            stats["metrics"] = metrics.get_summary()
        
        # System resources
        stats["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            stats["gpu"] = {
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_gb": torch.cuda.memory_allocated(0) / (1024**3),
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"Admin stats failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Store start time for uptime calculation
@app.on_event("startup")
async def startup_event():
    app.state.start_time = time.time()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for GPU model
        log_level="info"
    )

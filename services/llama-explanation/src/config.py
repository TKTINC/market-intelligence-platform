# services/llama-explanation/src/config.py
import os
from typing import Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Configuration settings for Llama Explanation Service"""
    
    # Service Configuration
    SERVICE_NAME: str = "llama-explanation"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    
    # Model Configuration
    MODEL_PATH: str = Field(
        default="/models/llama-2-7b-explanations.Q4_K_M.gguf",
        env="MODEL_PATH"
    )
    
    # Llama-specific settings
    N_CTX: int = Field(default=4096, env="N_CTX")  # Context window size
    N_BATCH: int = Field(default=512, env="N_BATCH")  # Batch size
    N_GPU_LAYERS: int = Field(default=35, env="N_GPU_LAYERS")  # GPU layers
    N_THREADS: int = Field(default=8, env="N_THREADS")  # CPU threads
    
    # Performance settings
    MAX_CONCURRENT_REQUESTS: int = Field(default=3, env="MAX_CONCURRENT_REQUESTS")
    MAX_QUEUE_SIZE: int = Field(default=100, env="MAX_QUEUE_SIZE")
    MAX_BATCH_SIZE: int = Field(default=10, env="MAX_BATCH_SIZE")
    REQUEST_TIMEOUT: int = Field(default=60, env="REQUEST_TIMEOUT")
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    # GPU Configuration
    CUDA_VISIBLE_DEVICES: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    GPU_MEMORY_FRACTION: float = Field(default=0.9, env="GPU_MEMORY_FRACTION")
    
    # Optimization flags
    USE_MLOCK: bool = Field(default=True, env="USE_MLOCK")  # Lock model in memory
    USE_MMAP: bool = Field(default=True, env="USE_MMAP")    # Memory-map model file
    F16_KV: bool = Field(default=True, env="F16_KV")        # Use float16 for cache
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

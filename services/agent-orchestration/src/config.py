import os
from typing import List, Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Configuration settings for Agent Orchestration Service"""
    
    # Service Configuration
    SERVICE_NAME: str = "agent-orchestration"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_PREFIX: str = Field(default="/api/v1", env="API_PREFIX")
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql://mip_user:mip_password@localhost:5432/mip",
        env="DATABASE_URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=20, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=0, env="DATABASE_MAX_OVERFLOW")
    
    # Redis Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    REDIS_POOL_SIZE: int = Field(default=20, env="REDIS_POOL_SIZE")
    
    # Agent Configuration
    AGENT_TIMEOUT_DEFAULT: int = Field(default=30, env="AGENT_TIMEOUT_DEFAULT")
    CIRCUIT_BREAKER_THRESHOLD: int = Field(default=5, env="CIRCUIT_BREAKER_THRESHOLD")
    CIRCUIT_BREAKER_TIMEOUT: int = Field(default=60, env="CIRCUIT_BREAKER_TIMEOUT")
    
    # External API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    
    # Agent Service URLs
    FINBERT_SERVICE_URL: str = Field(default="http://finbert-service:8000", env="FINBERT_SERVICE_URL")
    LLAMA_SERVICE_URL: str = Field(default="http://llama-service:8000", env="LLAMA_SERVICE_URL") 
    GPT4_SERVICE_URL: str = Field(default="http://gpt4-service:8000", env="GPT4_SERVICE_URL")
    TFT_SERVICE_URL: str = Field(default="http://tft-service:8000", env="TFT_SERVICE_URL")
    RISK_SERVICE_URL: str = Field(default="http://risk-service:8000", env="RISK_SERVICE_URL")
    
    # Cost and Budget Configuration
    DEFAULT_FREE_MONTHLY_LIMIT: float = Field(default=5.0, env="DEFAULT_FREE_MONTHLY_LIMIT")
    DEFAULT_PREMIUM_MONTHLY_LIMIT: float = Field(default=100.0, env="DEFAULT_PREMIUM_MONTHLY_LIMIT")
    DEFAULT_ENTERPRISE_MONTHLY_LIMIT: float = Field(default=1000.0, env="DEFAULT_ENTERPRISE_MONTHLY_LIMIT")
    
    # Security Configuration
    JWT_SECRET_KEY: str = Field(default="dev_secret_key_change_in_production", env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    JWT_EXPIRATION_HOURS: int = Field(default=24, env="JWT_EXPIRATION_HOURS")
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    
    # Monitoring Configuration
    PROMETHEUS_ENABLED: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    HEALTH_CHECK_INTERVAL: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Performance Tuning
    MAX_CONCURRENT_REQUESTS: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    REQUEST_QUEUE_SIZE: int = Field(default=1000, env="REQUEST_QUEUE_SIZE")
    AGENT_POOL_SIZE: int = Field(default=10, env="AGENT_POOL_SIZE")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

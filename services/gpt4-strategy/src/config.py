"""
Configuration management for GPT-4 Strategy Service
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    openai_fallback_model: str = "gpt-3.5-turbo"
    max_tokens: int = 4000
    temperature: float = 0.3
    
    # Rate Limiting
    requests_per_minute: int = 50
    cost_per_hour_limit: float = 100.0
    
    # Database
    database_url: str = ""
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    
    # Service Settings
    max_input_length: int = 5000
    max_strategies_per_request: int = 5
    max_batch_size: int = 10
    
    # Security
    enable_security_validation: bool = True
    enable_rate_limiting: bool = True
    enable_cost_tracking: bool = True
    
    # Monitoring
    metrics_retention_hours: int = 24
    health_check_interval: int = 60
    
    # Performance
    request_timeout: int = 30
    max_concurrent_requests: int = 100

class ConfigManager:
    def __init__(self):
        self.config = self._load_config()
        
    def _load_config(self) -> ServiceConfig:
        """Load configuration from environment variables and files"""
        
        try:
            # Load from environment variables
            config = ServiceConfig(
                # OpenAI
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                openai_model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
                openai_fallback_model=os.getenv("OPENAI_FALLBACK_MODEL", "gpt-3.5-turbo"),
                max_tokens=int(os.getenv("MAX_TOKENS", "4000")),
                temperature=float(os.getenv("TEMPERATURE", "0.3")),
                
                # Rate Limiting
                requests_per_minute=int(os.getenv("REQUESTS_PER_MINUTE", "50")),
                cost_per_hour_limit=float(os.getenv("COST_PER_HOUR_LIMIT", "100.0")),
                
                # Database
                database_url=os.getenv("DATABASE_URL", ""),
                
                # Redis
                redis_host=os.getenv("REDIS_HOST", "localhost"),
                redis_port=int(os.getenv("REDIS_PORT", "6379")),
                redis_password=os.getenv("REDIS_PASSWORD"),
                
                # Service Settings
                max_input_length=int(os.getenv("MAX_INPUT_LENGTH", "5000")),
                max_strategies_per_request=int(os.getenv("MAX_STRATEGIES_PER_REQUEST", "5")),
                max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "10")),
                
                # Feature Flags
                enable_security_validation=os.getenv("ENABLE_SECURITY_VALIDATION", "true").lower() == "true",
                enable_rate_limiting=os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true",
                enable_cost_tracking=os.getenv("ENABLE_COST_TRACKING", "true").lower() == "true",
                
                # Monitoring
                metrics_retention_hours=int(os.getenv("METRICS_RETENTION_HOURS", "24")),
                health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "60")),
                
                # Performance
                request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
                max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "100"))
            )
            
            # Validate required configuration
            self._validate_config(config)
            
            # Load additional config from file if exists
            config_file = os.getenv("CONFIG_FILE")
            if config_file and os.path.exists(config_file):
                file_config = self._load_config_file(config_file)
                config = self._merge_configs(config, file_config)
            
            logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _validate_config(self, config: ServiceConfig):
        """Validate configuration values"""
        
        errors = []
        
        # Required fields
        if not config.openai_api_key:
            errors.append("OPENAI_API_KEY is required")
        
        if not config.database_url:
            errors.append("DATABASE_URL is required")
        
        # Range validations
        if config.max_tokens < 100 or config.max_tokens > 8000:
            errors.append("MAX_TOKENS must be between 100 and 8000")
        
        if config.temperature < 0 or config.temperature > 2:
            errors.append("TEMPERATURE must be between 0 and 2")
        
        if config.requests_per_minute < 1 or config.requests_per_minute > 1000:
            errors.append("REQUESTS_PER_MINUTE must be between 1 and 1000")
        
        if config.max_input_length < 100 or config.max_input_length > 50000:
            errors.append("MAX_INPUT_LENGTH must be between 100 and 50000")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def _load_config_file(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")
            return {}
    
    def _merge_configs(self, base_config: ServiceConfig, file_config: Dict[str, Any]) -> ServiceConfig:
        """Merge file configuration with base configuration"""
        
        try:
            # Create a new config with file overrides
            config_dict = base_config.__dict__.copy()
            
            # Apply file overrides
            for key, value in file_config.items():
                if hasattr(base_config, key):
                    config_dict[key] = value
                else:
                    logger.warning(f"Unknown config key in file: {key}")
            
            return ServiceConfig(**config_dict)
            
        except Exception as e:
            logger.error(f"Failed to merge configurations: {e}")
            return base_config
    
    def get_config(self) -> ServiceConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration at runtime"""
        
        try:
            config_dict = self.config.__dict__.copy()
            config_dict.update(updates)
            
            new_config = ServiceConfig(**config_dict)
            self._validate_config(new_config)
            
            self.config = new_config
            logger.info(f"Configuration updated: {list(updates.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            raise
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI-specific configuration"""
        return {
            "api_key": self.config.openai_api_key,
            "model": self.config.openai_model,
            "fallback_model": self.config.openai_fallback_model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration"""
        return {
            "requests_per_minute": self.config.requests_per_minute,
            "cost_per_hour_limit": self.config.cost_per_hour_limit,
            "enabled": self.config.enable_rate_limiting
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            "max_input_length": self.config.max_input_length,
            "validation_enabled": self.config.enable_security_validation
        }

# Global config manager instance
config_manager = ConfigManager()

def get_config() -> ServiceConfig:
    """Get current service configuration"""
    return config_manager.get_config()

def update_config(updates: Dict[str, Any]):
    """Update service configuration"""
    config_manager.update_config(updates)

"""
Enhanced data validation schemas with options-specific validation
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator, ValidationError
import logging

logger = logging.getLogger(__name__)

class OptionsDataSchema(BaseModel):
    """Enhanced validation schema for options flow data"""
    
    symbol: str = Field(..., regex=r'^[A-Z]{1,5}$', description="Underlying symbol")
    expiry: date = Field(..., description="Option expiry date")
    strike: float = Field(..., gt=0, description="Strike price")
    option_type: str = Field(..., regex=r'^(call|put)$', description="Option type")
    volume: int = Field(..., ge=0, description="Trading volume")
    open_interest: int = Field(..., ge=0, description="Open interest")
    implied_volatility: float = Field(..., ge=0, le=5.0, description="Implied volatility")
    
    # Greeks validation
    delta: float = Field(..., ge=-1.0, le=1.0, description="Delta")
    gamma: float = Field(..., ge=0, description="Gamma")
    theta: float = Field(..., le=0, description="Theta")
    vega: float = Field(..., ge=0, description="Vega")
    
    # Price information
    bid: Optional[float] = Field(None, ge=0, description="Bid price")
    ask: Optional[float] = Field(None, ge=0, description="Ask price")
    last_price: Optional[float] = Field(None, ge=0, description="Last trade price")
    
    # Market data
    underlying_price: Optional[float] = Field(None, gt=0, description="Underlying asset price")
    timestamp: datetime = Field(..., description="Data timestamp")
    
    @validator('expiry')
    def validate_expiry(cls, v):
        """Validate that expiry is not in the past"""
        if v < date.today():
            raise ValueError("Option expiry cannot be in the past")
        return v
    
    @validator('ask')
    def validate_bid_ask_spread(cls, v, values):
        """Validate bid-ask spread is reasonable"""
        if v is not None and 'bid' in values and values['bid'] is not None:
            spread = v - values['bid']
            if spread < 0:
                raise ValueError("Ask price cannot be lower than bid price")
            if spread > values['bid'] * 2:  # Spread > 200% of bid
                raise ValueError("Bid-ask spread too wide")
        return v
    
    @validator('delta')
    def validate_delta_by_option_type(cls, v, values):
        """Validate delta sign based on option type"""
        if 'option_type' in values:
            if values['option_type'] == 'call' and v < 0:
                raise ValueError("Call option delta cannot be negative")
            if values['option_type'] == 'put' and v > 0:
                raise ValueError("Put option delta cannot be positive")
        return v

class MarketDataSchema(BaseModel):
    """Enhanced validation schema for market data"""
    
    symbol: str = Field(..., regex=r'^[A-Z]{1,5}$')
    timestamp: datetime = Field(...)
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: int = Field(..., ge=0)
    
    # Enhanced market data fields
    implied_volatility: Optional[float] = Field(None, ge=0, le=5.0)
    average_volume: Optional[int] = Field(None, ge=0)
    market_cap: Optional[float] = Field(None, ge=0)
    
    @validator('high')
    def validate_high_price(cls, v, values):
        """Validate high >= open, low, close"""
        if 'open' in values and v < values['open']:
            raise ValueError("High price cannot be lower than open price")
        if 'low' in values and v < values['low']:
            raise ValueError("High price cannot be lower than low price")
        if 'close' in values and v < values['close']:
            raise ValueError("High price cannot be lower than close price")
        return v
    
    @validator('low')
    def validate_low_price(cls, v, values):
        """Validate low <= open, close"""
        if 'open' in values and v > values['open']:
            raise ValueError("Low price cannot be higher than open price")
        if 'close' in values and v > values['close']:
            raise ValueError("Low price cannot be higher than close price")
        return v

class NewsDataSchema(BaseModel):
    """Enhanced validation schema for news data"""
    
    title: str = Field(..., min_length=10, max_length=500)
    content: str = Field(..., min_length=50, max_length=10000)
    source: str = Field(..., min_length=1, max_length=100)
    published_at: datetime = Field(...)
    url: str = Field(..., regex=r'^https?://.+')
    
    # Enhanced news fields
    symbols_mentioned: Optional[List[str]] = Field(None, description="Mentioned stock symbols")
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @validator('published_at')
    def validate_published_date(cls, v):
        """Validate published date is not too far in the future"""
        if v > datetime.utcnow():
            # Allow small time differences due to timezone issues
            if (v - datetime.utcnow()).total_seconds() > 3600:  # 1 hour
                raise ValueError("Published date cannot be more than 1 hour in the future")
        return v

class VirtualTradeSchema(BaseModel):
    """Validation schema for virtual trading data"""
    
    strategy_type: str = Field(
        ..., 
        regex=r'^(COVERED_CALL|CASH_SECURED_PUT|IRON_CONDOR|STRANGLE|BUTTERFLY)$'
    )
    entry_conditions: Dict[str, Any] = Field(...)
    risk_parameters: Dict[str, float] = Field(...)
    expected_return: float = Field(..., ge=-1.0, le=10.0)
    
    @validator('risk_parameters')
    def validate_risk_parameters(cls, v):
        """Validate required risk parameters"""
        required_params = ['max_loss', 'probability_profit', 'break_even']
        for param in required_params:
            if param not in v:
                raise ValueError(f"Missing required risk parameter: {param}")
        return v

class DataValidationEngine:
    """Enhanced data validation engine with options-specific validation"""
    
    def __init__(self):
        self.schemas = {
            'options': OptionsDataSchema,
            'market': MarketDataSchema,
            'news': NewsDataSchema,
            'virtual_trade': VirtualTradeSchema
        }
        
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'validation_errors': 0,
            'schema_errors': {}
        }
    
    async def validate_options_data(self, data: Dict) -> Optional[Dict]:
        """Validate options flow data with enhanced validation"""
        try:
            self.validation_stats['total_validations'] += 1
            
            # Additional business logic validation for options
            if not self._is_valid_options_symbol(data.get('symbol')):
                logger.warning(f"Unsupported options symbol: {data.get('symbol')}")
                return None
            
            # Validate minimum volume threshold
            if data.get('volume', 0) < 10:
                logger.debug(f"Options volume too low: {data.get('volume')}")
                return None
            
            # Validate market hours (simplified)
            if not self._is_market_hours(data.get('timestamp')):
                logger.debug("Options data outside market hours")
                return None
            
            # Run schema validation
            validated = OptionsDataSchema(**data)
            
            self.validation_stats['successful_validations'] += 1
            return validated.dict()
            
        except ValidationError as e:
            self.validation_stats['validation_errors'] += 1
            self._record_schema_error('options', str(e))
            logger.error(f"Options data validation failed: {str(e)}")
            return None
        except Exception as e:
            self.validation_stats['validation_errors'] += 1
            logger.error(f"Options data validation error: {str(e)}")
            return None
    
    async def validate_market_data(self, data: Dict) -> Optional[Dict]:
        """Validate market data with enhanced checks"""
        try:
            self.validation_stats['total_validations'] += 1
            
            # Check for reasonable price movements (circuit breaker style)
            if self._has_extreme_price_movement(data):
                logger.warning(f"Extreme price movement detected for {data.get('symbol')}")
                return None
            
            # Run schema validation
            validated = MarketDataSchema(**data)
            
            self.validation_stats['successful_validations'] += 1
            return validated.dict()
            
        except ValidationError as e:
            self.validation_stats['validation_errors'] += 1
            self._record_schema_error('market', str(e))
            logger.error(f"Market data validation failed: {str(e)}")
            return None
    
    async def validate_news_data(self, data: Dict) -> Optional[Dict]:
        """Validate news data with content analysis"""
        try:
            self.validation_stats['total_validations'] += 1
            
            # Check for duplicate content
            if self._is_duplicate_news(data):
                logger.debug("Duplicate news content detected")
                return None
            
            # Run schema validation
            validated = NewsDataSchema(**data)
            
            self.validation_stats['successful_validations'] += 1
            return validated.dict()
            
        except ValidationError as e:
            self.validation_stats['validation_errors'] += 1
            self._record_schema_error('news', str(e))
            logger.error(f"News data validation failed: {str(e)}")
            return None
    
    async def validate_generic_data(self, data: Dict) -> Optional[Dict]:
        """Generic validation for unspecified data types"""
        try:
            self.validation_stats['total_validations'] += 1
            
            # Basic checks
            if not isinstance(data, dict) or not data:
                return None
            
            # Ensure required metadata
            if '_metadata' not in data:
                logger.warning("Missing metadata in generic data")
                return None
            
            self.validation_stats['successful_validations'] += 1
            return data
            
        except Exception as e:
            self.validation_stats['validation_errors'] += 1
            logger.error(f"Generic data validation error: {str(e)}")
            return None
    
    def _is_valid_options_symbol(self, symbol: str) -> bool:
        """Check if symbol is supported for options trading"""
        if not symbol:
            return False
        
        # Major stocks and ETFs supported for options
        supported_symbols = {
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
            'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'TLT', 'XLE', 'XLF'
        }
        
        return symbol.upper() in supported_symbols
    
    def _is_market_hours(self, timestamp) -> bool:
        """Check if timestamp is during market hours (simplified)"""
        if not timestamp:
            return False
        
        # Convert to datetime if string
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return False
        
        # Simple check: Monday-Friday, 9:30 AM - 4:00 PM ET
        # In production, use proper market calendar
        weekday = timestamp.weekday()
        hour = timestamp.hour
        
        return 0 <= weekday <= 4 and 9 <= hour <= 16
    
    def _has_extreme_price_movement(self, data: Dict) -> bool:
        """Check for extreme price movements that might indicate bad data"""
        try:
            open_price = float(data.get('open', 0))
            high_price = float(data.get('high', 0))
            low_price = float(data.get('low', 0))
            close_price = float(data.get('close', 0))
            
            if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                return True
            
            # Check for extreme intraday movements (>50%)
            intraday_range = (high_price - low_price) / open_price
            if intraday_range > 0.5:
                return True
            
            # Check for extreme gap movements (>30%)
            if abs(open_price - close_price) / open_price > 0.3:
                return True
            
            return False
            
        except (ValueError, ZeroDivisionError):
            return True
    
    def _is_duplicate_news(self, data: Dict) -> bool:
        """Check for duplicate news content (simplified)"""
        # In production, implement proper duplicate detection
        # using content hashing, title similarity, etc.
        return False
    
    def _record_schema_error(self, schema_type: str, error: str):
        """Record schema validation errors for monitoring"""
        if schema_type not in self.validation_stats['schema_errors']:
            self.validation_stats['schema_errors'][schema_type] = {}
        
        error_key = error[:100]  # Truncate long errors
        if error_key in self.validation_stats['schema_errors'][schema_type]:
            self.validation_stats['schema_errors'][schema_type][error_key] += 1
        else:
            self.validation_stats['schema_errors'][schema_type][error_key] = 1
    
    def get_validation_stats(self) -> Dict:
        """Get validation statistics for monitoring"""
        return self.validation_stats.copy()

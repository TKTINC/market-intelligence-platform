"""
Portfolio Manager for virtual trading portfolios
"""

import aioredis
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import uuid
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class Portfolio:
    portfolio_id: str
    user_id: str
    name: str
    initial_balance: float
    cash_balance: float
    total_value: float
    total_pnl: float
    day_pnl: float
    risk_tolerance: str
    max_position_size: float
    created_at: datetime
    last_updated: datetime
    status: str

class PortfolioManager:
    def __init__(self):
        self.redis = None
        
        # Portfolio configuration
        self.config = {
            "default_initial_balance": 100000.0,
            "min_cash_balance": 0.0,
            "max_portfolios_per_user": 10,
            "risk_tolerance_options": ["low", "medium", "high"],
            "default_max_position_size": 0.1  # 10% of portfolio
        }
        
        # Portfolio statistics
        self.stats = {
            "total_portfolios": 0,
            "total_value": 0.0,
            "active_users": 0
        }
        
    async def initialize(self):
        """Initialize the portfolio manager"""
        try:
            # Initialize Redis connection
            self.redis = aioredis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            
            # Load portfolio statistics
            await self._load_portfolio_stats()
            
            logger.info("Portfolio manager initialized")
            
        except Exception as e:
            logger.error(f"Portfolio manager initialization failed: {e}")
            raise
    
    async def close(self):
        """Close the portfolio manager"""
        if self.redis:
            await self.redis.close()
    
    async def health_check(self) -> str:
        """Check health of portfolio manager"""
        try:
            if not self.redis:
                return "unhealthy - no redis connection"
            
            # Test Redis connection
            await self.redis.ping()
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Portfolio manager health check failed: {e}")
            return "unhealthy"
    
    async def create_portfolio(
        self,
        user_id: str,
        name: str,
        initial_balance: float = None,
        risk_tolerance: str = "medium",
        max_position_size: float = None
    ) -> Dict[str, Any]:
        """Create a new virtual trading portfolio"""
        
        try:
            # Validate parameters
            if initial_balance is None:
                initial_balance = self.config["default_initial_balance"]
            
            if max_position_size is None:
                max_position_size = self.config["default_max_position_size"]
            
            if risk_tolerance not in self.config["risk_tolerance_options"]:
                raise ValueError(f"Invalid risk tolerance: {risk_tolerance}")
            
            # Check user portfolio limit
            user_portfolios = await self.get_user_portfolios(user_id)
            if len(user_portfolios) >= self.config["max_portfolios_per_user"]:
                raise Exception(f"User has reached maximum portfolio limit: {self.config['max_portfolios_per_user']}")
            
            # Create portfolio
            portfolio_id = str(uuid.uuid4())
            current_time = datetime.utcnow()
            
            portfolio = Portfolio(
                portfolio_id=portfolio_id,
                user_id=user_id,
                name=name,
                initial_balance=initial_balance,
                cash_balance=initial_balance,
                total_value=initial_balance,
                total_pnl=0.0,
                day_pnl=0.0,
                risk_tolerance=risk_tolerance,
                max_position_size=max_position_size,
                created_at=current_time,
                last_updated=current_time,
                status="active"
            )
            
            # Store portfolio
            await self._store_portfolio(portfolio)
            
            # Update statistics
            self.stats["total_portfolios"] += 1
            await self._update_portfolio_stats()
            
            logger.info(f"Portfolio created: {portfolio_id} for user {user_id}")
            
            return asdict(portfolio)
            
        except Exception as e:
            logger.error(f"Portfolio creation failed: {e}")
            raise
    
    async def get_portfolio(self, portfolio_id: str) -> Dict[str, Any]:
        """Get portfolio details"""
        
        try:
            portfolio_key = f"portfolio:{portfolio_id}"
            portfolio_data = await self.redis.get(portfolio_key)
            
            if not portfolio_data:
                raise Exception(f"Portfolio not found: {portfolio_id}")
            
            portfolio = json.loads(portfolio_data)
            
            # Get current positions
            positions = await self._get_portfolio_positions(portfolio_id)
            portfolio["positions"] = positions
            
            # Calculate current metrics
            portfolio = await self._calculate_portfolio_metrics(portfolio)
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Portfolio retrieval failed: {e}")
            raise
    
    async def get_user_portfolios(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all portfolios for a user"""
        
        try:
            # Get portfolio IDs for user
            user_portfolio_key = f"user_portfolios:{user_id}"
            portfolio_ids = await self.redis.smembers(user_portfolio_key)
            
            portfolios = []
            for portfolio_id in portfolio_ids:
                try:
                    portfolio = await self.get_portfolio(portfolio_id)
                    portfolios.append(portfolio)
                except Exception as e:
                    logger.error(f"Failed to load portfolio {portfolio_id}: {e}")
            
            # Sort by creation date (newest first)
            portfolios.sort(key=lambda p: p["created_at"], reverse=True)
            
            return portfolios
            
        except Exception as e:
            logger.error(f"User portfolios retrieval failed: {e}")
            return []
    
    async def update_portfolio_after_trade(
        self,
        portfolio_id: str,
        trade_result: Dict[str, Any]
    ):
        """Update portfolio after trade execution"""
        
        try:
            portfolio = await self.get_portfolio(portfolio_id)
            
            # Update cash balance
            trade_value = trade_result["total_value"]
            commission = trade_result["commission"]
            action = trade_result.get("action", "buy")
            
            if action == "buy":
                portfolio["cash_balance"] -= (trade_value + commission)
            elif action == "sell":
                portfolio["cash_balance"] += (trade_value - commission)
            
            # Update timestamp
            portfolio["last_updated"] = datetime.utcnow().isoformat()
            
            # Recalculate portfolio metrics
            portfolio = await self._calculate_portfolio_metrics(portfolio)
            
            # Store updated portfolio
            portfolio_obj = Portfolio(**{
                k: v for k, v in portfolio.items() 
                if k in Portfolio.__dataclass_fields__
            })
            await self._store_portfolio(portfolio_obj)
            
            logger.info(f"Portfolio {portfolio_id} updated after trade")
            
        except Exception as e:
            logger.error(f"Portfolio update after trade failed: {e}")
            raise
    
    async def update_portfolio_with_prices(
        self,
        portfolio_id: str,
        market_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """Update portfolio with current market prices"""
        
        try:
            portfolio = await self.get_portfolio(portfolio_id)
            
            # Update position values with current prices
            total_position_value = 0.0
            
            for position in portfolio["positions"]:
                symbol = position["symbol"]
                if symbol in market_prices:
                    current_price = market_prices[symbol]
                    quantity = position["quantity"]
                    
                    position["current_price"] = current_price
                    position["market_value"] = quantity * current_price
                    position["unrealized_pnl"] = (current_price - position["avg_cost"]) * quantity
                    
                    total_position_value += position["market_value"]
            
            # Update portfolio total value
            portfolio["total_value"] = portfolio["cash_balance"] + total_position_value
            portfolio["total_pnl"] = portfolio["total_value"] - portfolio["initial_balance"]
            
            # Calculate day P&L (would need historical data)
            portfolio["day_pnl"] = 0.0  # Placeholder
            
            portfolio["last_updated"] = datetime.utcnow().isoformat()
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Portfolio price update failed: {e}")
            raise
    
    async def get_active_portfolios(self) -> List[str]:
        """Get list of active portfolio IDs"""
        
        try:
            # Get all portfolio keys
            portfolio_keys = await self.redis.keys("portfolio:*")
            
            active_portfolios = []
            for key in portfolio_keys:
                portfolio_data = await self.redis.get(key)
                if portfolio_data:
                    portfolio = json.loads(portfolio_data)
                    if portfolio.get("status") == "active":
                        active_portfolios.append(portfolio["portfolio_id"])
            
            return active_portfolios
            
        except Exception as e:
            logger.error(f"Active portfolios retrieval failed: {e}")
            return []
    
    async def get_all_symbols(self) -> List[str]:
        """Get all symbols across all portfolios"""
        
        try:
            # Get all position keys
            position_keys = await self.redis.keys("positions:*")
            
            symbols = set()
            for key in position_keys:
                # Extract symbol from key format: positions:portfolio_id:symbol
                parts = key.split(":")
                if len(parts) >= 3:
                    symbol = parts[2]
                    symbols.add(symbol)
            
            return list(symbols)
            
        except Exception as e:
            logger.error(f"All symbols retrieval failed: {e}")
            return []
    
    async def get_portfolio_statistics(self) -> Dict[str, Any]:
        """Get portfolio system statistics"""
        
        try:
            # Update statistics
            await self._load_portfolio_stats()
            
            return {
                "total_portfolios": self.stats["total_portfolios"],
                "total_value": self.stats["total_value"],
                "active_users": self.stats["active_users"],
                "average_portfolio_size": (
                    self.stats["total_value"] / max(1, self.stats["total_portfolios"])
                ),
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Portfolio statistics retrieval failed: {e}")
            return {"error": str(e)}
    
    async def _store_portfolio(self, portfolio: Portfolio):
        """Store portfolio in Redis"""
        
        try:
            portfolio_key = f"portfolio:{portfolio.portfolio_id}"
            
            # Convert datetime objects to strings
            portfolio_dict = asdict(portfolio)
            portfolio_dict["created_at"] = portfolio.created_at.isoformat()
            portfolio_dict["last_updated"] = portfolio.last_updated.isoformat()
            
            await self.redis.set(
                portfolio_key,
                json.dumps(portfolio_dict, default=str)
            )
            
            # Add to user's portfolio set
            user_portfolio_key = f"user_portfolios:{portfolio.user_id}"
            await self.redis.sadd(user_portfolio_key, portfolio.portfolio_id)
            
        except Exception as e:
            logger.error(f"Portfolio storage failed: {e}")
            raise
    
    async def _get_portfolio_positions(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Get all positions for a portfolio"""
        
        try:
            position_keys = await self.redis.keys(f"positions:{portfolio_id}:*")
            positions = []
            
            for key in position_keys:
                position_data = await self.redis.get(key)
                if position_data:
                    position = json.loads(position_data)
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Portfolio positions retrieval failed: {e}")
            return []
    
    async def _calculate_portfolio_metrics(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        
        try:
            # Calculate total position value
            total_position_value = sum(
                position.get("market_value", 0) for position in portfolio.get("positions", [])
            )
            
            # Update total value
            portfolio["total_value"] = portfolio["cash_balance"] + total_position_value
            
            # Calculate total P&L
            portfolio["total_pnl"] = portfolio["total_value"] - portfolio["initial_balance"]
            
            # Calculate additional metrics
            portfolio["cash_percentage"] = (
                portfolio["cash_balance"] / max(1, portfolio["total_value"])
            ) * 100
            
            portfolio["positions_value"] = total_position_value
            portfolio["positions_count"] = len(portfolio.get("positions", []))
            
            # Risk metrics
            portfolio["risk_metrics"] = await self._calculate_risk_metrics(portfolio)
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {e}")
            return portfolio
    
    async def _calculate_risk_metrics(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics for portfolio"""
        
        try:
            positions = portfolio.get("positions", [])
            total_value = portfolio["total_value"]
            
            if not positions or total_value <= 0:
                return {
                    "largest_position_pct": 0.0,
                    "concentration_risk": "low",
                    "position_count": 0,
                    "leverage": 0.0
                }
            
            # Calculate largest position percentage
            largest_position_value = max(
                abs(position.get("market_value", 0)) for position in positions
            )
            largest_position_pct = (largest_position_value / total_value) * 100
            
            # Determine concentration risk
            if largest_position_pct > 50:
                concentration_risk = "high"
            elif largest_position_pct > 25:
                concentration_risk = "medium"
            else:
                concentration_risk = "low"
            
            # Calculate leverage (simplified)
            total_long_value = sum(
                position.get("market_value", 0) for position in positions
                if position.get("market_value", 0) > 0
            )
            leverage = total_long_value / max(1, total_value)
            
            return {
                "largest_position_pct": round(largest_position_pct, 2),
                "concentration_risk": concentration_risk,
                "position_count": len(positions),
                "leverage": round(leverage, 2),
                "total_exposure": round(total_long_value, 2)
            }
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return {}
    
    async def _load_portfolio_stats(self):
        """Load portfolio statistics"""
        
        try:
            # Count total portfolios
            portfolio_keys = await self.redis.keys("portfolio:*")
            self.stats["total_portfolios"] = len(portfolio_keys)
            
            # Calculate total value and active users
            total_value = 0.0
            active_users = set()
            
            for key in portfolio_keys:
                portfolio_data = await self.redis.get(key)
                if portfolio_data:
                    portfolio = json.loads(portfolio_data)
                    if portfolio.get("status") == "active":
                        total_value += portfolio.get("total_value", 0)
                        active_users.add(portfolio.get("user_id"))
            
            self.stats["total_value"] = total_value
            self.stats["active_users"] = len(active_users)
            
        except Exception as e:
            logger.error(f"Portfolio stats loading failed: {e}")
    
    async def _update_portfolio_stats(self):
        """Update and store portfolio statistics"""
        
        try:
            await self._load_portfolio_stats()
            
            # Store in Redis
            await self.redis.set(
                "portfolio_stats",
                json.dumps(self.stats, default=str)
            )
            
        except Exception as e:
            logger.error(f"Portfolio stats update failed: {e}")

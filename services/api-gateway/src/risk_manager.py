"""
Risk Manager for portfolio risk assessment and monitoring
"""

import asyncio
import aioredis
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import statistics
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class RiskAlert:
    alert_id: str
    portfolio_id: str
    user_id: str
    alert_type: str
    severity: str
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    acknowledged: bool

@dataclass
class RiskMetrics:
    portfolio_id: str
    var_1d: float  # Value at Risk (1 day)
    var_5d: float  # Value at Risk (5 days)
    max_drawdown: float
    concentration_risk: float
    leverage_ratio: float
    sharpe_ratio: float
    beta: float
    correlation_risk: float
    liquidity_risk: float
    position_count: int
    largest_position_pct: float
    cash_ratio: float
    last_updated: datetime

class RiskManager:
    def __init__(self):
        self.redis = None
        
        # Risk management configuration
        self.config = {
            "var_confidence_level": 0.95,  # 95% confidence for VaR
            "max_portfolio_drawdown": 0.20,  # 20% max drawdown
            "max_position_concentration": 0.25,  # 25% max single position
            "max_leverage": 2.0,  # 2x maximum leverage
            "min_cash_ratio": 0.05,  # 5% minimum cash
            "liquidity_threshold": 0.1,  # 10% liquidity risk threshold
            "correlation_threshold": 0.8,  # 80% correlation warning
            "alert_cooldown_minutes": 30  # 30 min between same alerts
        }
        
        # Risk alert subscriptions
        self.alert_subscriptions = {}
        
        # Risk calculation cache
        self.risk_metrics_cache = {}
        
        # Performance tracking
        self.monitoring_stats = {
            "risk_assessments": 0,
            "alerts_generated": 0,
            "avg_assessment_time_ms": 0.0
        }
        
    async def initialize(self):
        """Initialize the risk manager"""
        try:
            # Initialize Redis connection
            self.redis = aioredis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            
            # Load alert subscriptions
            await self._load_alert_subscriptions()
            
            # Load risk metrics cache
            await self._load_risk_metrics_cache()
            
            # Start background monitoring
            asyncio.create_task(self._risk_monitoring_task())
            asyncio.create_task(self._alert_cleanup_task())
            
            logger.info("Risk manager initialized")
            
        except Exception as e:
            logger.error(f"Risk manager initialization failed: {e}")
            raise
    
    async def close(self):
        """Close the risk manager"""
        if self.redis:
            await self.redis.close()
    
    async def health_check(self) -> str:
        """Check health of risk manager"""
        try:
            if not self.redis:
                return "unhealthy - no redis connection"
            
            # Test Redis connection
            await self.redis.ping()
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Risk manager health check failed: {e}")
            return "unhealthy"
    
    async def validate_trade(
        self,
        portfolio_id: str,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate trade against risk limits"""
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Calculate trade impact
            trade_value = abs(quantity) * price
            total_value = portfolio.get("total_value", 0)
            
            # Risk checks
            risk_checks = []
            
            # 1. Position concentration check
            if total_value > 0:
                position_impact = trade_value / total_value
                if position_impact > self.config["max_position_concentration"]:
                    risk_checks.append({
                        "check": "position_concentration",
                        "passed": False,
                        "reason": f"Trade would create {position_impact:.1%} concentration (max: {self.config['max_position_concentration']:.1%})"
                    })
                else:
                    risk_checks.append({
                        "check": "position_concentration",
                        "passed": True,
                        "value": position_impact
                    })
            
            # 2. Cash balance check
            cash_balance = portfolio.get("cash_balance", 0)
            if action == "buy" and trade_value > cash_balance:
                risk_checks.append({
                    "check": "cash_balance",
                    "passed": False,
                    "reason": f"Insufficient cash: ${cash_balance:,.2f} available, ${trade_value:,.2f} required"
                })
            else:
                risk_checks.append({
                    "check": "cash_balance",
                    "passed": True,
                    "value": cash_balance
                })
            
            # 3. Leverage check
            leverage = await self._calculate_leverage_impact(portfolio, action, trade_value)
            if leverage > self.config["max_leverage"]:
                risk_checks.append({
                    "check": "leverage",
                    "passed": False,
                    "reason": f"Trade would create {leverage:.2f}x leverage (max: {self.config['max_leverage']:.2f}x)"
                })
            else:
                risk_checks.append({
                    "check": "leverage",
                    "passed": True,
                    "value": leverage
                })
            
            # 4. Portfolio risk check
            portfolio_risk = await self._assess_trade_risk_impact(portfolio_id, symbol, action, quantity, price)
            if portfolio_risk["risk_score"] > 0.8:
                risk_checks.append({
                    "check": "portfolio_risk",
                    "passed": False,
                    "reason": f"High portfolio risk score: {portfolio_risk['risk_score']:.2f}"
                })
            else:
                risk_checks.append({
                    "check": "portfolio_risk",
                    "passed": True,
                    "value": portfolio_risk["risk_score"]
                })
            
            # Determine overall approval
            failed_checks = [check for check in risk_checks if not check["passed"]]
            approved = len(failed_checks) == 0
            
            # Update monitoring stats
            assessment_time = (asyncio.get_event_loop().time() - start_time) * 1000
            await self._update_monitoring_stats(assessment_time)
            
            return {
                "approved": approved,
                "reason": failed_checks[0]["reason"] if failed_checks else "Trade approved",
                "risk_checks": risk_checks,
                "risk_score": portfolio_risk.get("risk_score", 0.0),
                "assessment_time_ms": round(assessment_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Trade validation failed: {e}")
            return {
                "approved": False,
                "reason": f"Risk validation error: {str(e)}",
                "risk_checks": [],
                "risk_score": 1.0
            }
    
    async def calculate_portfolio_risk(self, portfolio_id: str) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics for portfolio"""
        
        try:
            # Get portfolio data
            portfolio_data = await self._get_portfolio_data(portfolio_id)
            if not portfolio_data:
                raise Exception(f"Portfolio not found: {portfolio_id}")
            
            # Get positions
            positions = await self._get_portfolio_positions(portfolio_id)
            
            # Get historical P&L data
            historical_pnl = await self._get_historical_pnl(portfolio_id)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(
                portfolio_data, positions, historical_pnl
            )
            
            # Cache the results
            self.risk_metrics_cache[portfolio_id] = risk_metrics
            
            # Store in Redis
            await self._store_risk_metrics(risk_metrics)
            
            return asdict(risk_metrics)
            
        except Exception as e:
            logger.error(f"Portfolio risk calculation failed: {e}")
            raise
    
    async def monitor_portfolio_risk(self, portfolio_id: str):
        """Monitor portfolio for risk alerts"""
        
        try:
            # Calculate current risk metrics
            risk_metrics = await self.calculate_portfolio_risk(portfolio_id)
            
            # Check for risk threshold breaches
            alerts = await self._check_risk_thresholds(portfolio_id, risk_metrics)
            
            # Process and send alerts
            for alert in alerts:
                await self._process_risk_alert(alert)
            
        except Exception as e:
            logger.error(f"Portfolio risk monitoring failed for {portfolio_id}: {e}")
    
    async def subscribe_to_alerts(
        self,
        user_id: str,
        portfolio_id: str,
        alert_types: List[str]
    ) -> Dict[str, Any]:
        """Subscribe to risk alerts for a portfolio"""
        
        try:
            subscription_key = f"{user_id}:{portfolio_id}"
            
            # Validate alert types
            valid_alert_types = [
                "drawdown", "concentration", "leverage", "var_breach", 
                "liquidity", "correlation", "cash_low"
            ]
            
            validated_types = [
                alert_type for alert_type in alert_types 
                if alert_type in valid_alert_types
            ]
            
            # Store subscription
            self.alert_subscriptions[subscription_key] = {
                "user_id": user_id,
                "portfolio_id": portfolio_id,
                "alert_types": validated_types,
                "created_at": datetime.utcnow().isoformat(),
                "active": True
            }
            
            # Store in Redis
            await self.redis.set(
                f"risk_alert_subscription:{subscription_key}",
                json.dumps(self.alert_subscriptions[subscription_key])
            )
            
            return {
                "subscription_id": subscription_key,
                "alert_types": validated_types,
                "status": "active"
            }
            
        except Exception as e:
            logger.error(f"Alert subscription failed: {e}")
            raise
    
    async def reassess_all_portfolios(self):
        """Reassess risk for all portfolios (admin function)"""
        
        try:
            # Get all active portfolios
            portfolio_keys = await self.redis.keys("portfolio:*")
            
            reassessed_count = 0
            
            for key in portfolio_keys:
                try:
                    portfolio_id = key.split(":")[-1]
                    await self.calculate_portfolio_risk(portfolio_id)
                    reassessed_count += 1
                    
                    # Add delay to avoid overwhelming the system
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Failed to reassess portfolio {portfolio_id}: {e}")
            
            logger.info(f"Reassessed risk for {reassessed_count} portfolios")
            
        except Exception as e:
            logger.error(f"Portfolio risk reassessment failed: {e}")
            raise
    
    async def _calculate_risk_metrics(
        self,
        portfolio_data: Dict[str, Any],
        positions: List[Dict[str, Any]],
        historical_pnl: List[Dict[str, Any]]
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        try:
            portfolio_id = portfolio_data["portfolio_id"]
            total_value = portfolio_data.get("total_value", 0)
            cash_balance = portfolio_data.get("cash_balance", 0)
            
            # Basic metrics
            position_count = len(positions)
            cash_ratio = cash_balance / max(1, total_value)
            
            # Position concentration
            if positions and total_value > 0:
                position_values = [abs(pos.get("market_value", 0)) for pos in positions]
                largest_position_pct = max(position_values) / total_value
                concentration_risk = sum(val**2 for val in position_values) / (total_value**2)
            else:
                largest_position_pct = 0.0
                concentration_risk = 0.0
            
            # Leverage calculation
            long_value = sum(pos.get("market_value", 0) for pos in positions if pos.get("market_value", 0) > 0)
            leverage_ratio = long_value / max(1, total_value)
            
            # Historical metrics (if available)
            if len(historical_pnl) > 1:
                pnl_values = [point.get("total_pnl", 0) for point in historical_pnl]
                
                # VaR calculation
                var_1d = await self._calculate_var(pnl_values, 1)
                var_5d = await self._calculate_var(pnl_values, 5)
                
                # Maximum drawdown
                max_drawdown = await self._calculate_max_drawdown(pnl_values)
                
                # Sharpe ratio (simplified)
                if len(pnl_values) > 1:
                    returns = [(pnl_values[i] - pnl_values[i-1]) / max(1, abs(pnl_values[i-1])) 
                              for i in range(1, len(pnl_values))]
                    
                    if statistics.stdev(returns) > 0:
                        sharpe_ratio = statistics.mean(returns) / statistics.stdev(returns)
                    else:
                        sharpe_ratio = 0.0
                else:
                    sharpe_ratio = 0.0
                
            else:
                var_1d = 0.0
                var_5d = 0.0
                max_drawdown = 0.0
                sharpe_ratio = 0.0
            
            # Market beta (simplified - would need market data)
            beta = 1.0  # Default to market beta
            
            # Correlation and liquidity risk (simplified)
            correlation_risk = await self._calculate_correlation_risk(positions)
            liquidity_risk = await self._calculate_liquidity_risk(positions)
            
            return RiskMetrics(
                portfolio_id=portfolio_id,
                var_1d=var_1d,
                var_5d=var_5d,
                max_drawdown=max_drawdown,
                concentration_risk=concentration_risk,
                leverage_ratio=leverage_ratio,
                sharpe_ratio=sharpe_ratio,
                beta=beta,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                position_count=position_count,
                largest_position_pct=largest_position_pct,
                cash_ratio=cash_ratio,
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            raise
    
    async def _calculate_var(self, pnl_values: List[float], days: int) -> float:
        """Calculate Value at Risk"""
        
        try:
            if len(pnl_values) < days:
                return 0.0
            
            # Calculate daily returns
            returns = []
            for i in range(days, len(pnl_values)):
                start_value = pnl_values[i-days]
                end_value = pnl_values[i]
                
                if start_value != 0:
                    return_pct = (end_value - start_value) / abs(start_value)
                    returns.append(return_pct)
            
            if not returns:
                return 0.0
            
            # Calculate VaR at confidence level
            returns.sort()
            var_index = int((1 - self.config["var_confidence_level"]) * len(returns))
            
            return abs(returns[var_index]) if var_index < len(returns) else 0.0
            
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            return 0.0
    
    async def _calculate_max_drawdown(self, pnl_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        
        try:
            if len(pnl_values) < 2:
                return 0.0
            
            running_max = pnl_values[0]
            max_drawdown = 0.0
            
            for pnl in pnl_values[1:]:
                running_max = max(running_max, pnl)
                drawdown = (running_max - pnl) / max(1, abs(running_max))
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Max drawdown calculation failed: {e}")
            return 0.0
    
    async def _calculate_correlation_risk(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate position correlation risk"""
        
        try:
            # Simplified correlation risk based on sector/asset class
            # In a real implementation, this would use historical price correlations
            
            if len(positions) < 2:
                return 0.0
            
            # Group positions by symbol (simplified)
            symbols = [pos.get("symbol", "") for pos in positions]
            unique_symbols = len(set(symbols))
            
            # Higher correlation risk with fewer unique assets
            correlation_risk = 1.0 - (unique_symbols / len(positions))
            
            return min(1.0, max(0.0, correlation_risk))
            
        except Exception as e:
            logger.error(f"Correlation risk calculation failed: {e}")
            return 0.0
    
    async def _calculate_liquidity_risk(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate portfolio liquidity risk"""
        
        try:
            # Simplified liquidity risk - in reality would use volume data
            
            if not positions:
                return 0.0
            
            # Assume larger positions have higher liquidity risk
            total_value = sum(abs(pos.get("market_value", 0)) for pos in positions)
            
            if total_value == 0:
                return 0.0
            
            # Calculate weighted average position size
            avg_position_size = total_value / len(positions)
            
            # Higher liquidity risk for larger average positions
            liquidity_risk = min(1.0, avg_position_size / 50000)  # $50k threshold
            
            return liquidity_risk
            
        except Exception as e:
            logger.error(f"Liquidity risk calculation failed: {e}")
            return 0.0
    
    async def _calculate_leverage_impact(
        self,
        portfolio: Dict[str, Any],
        action: str,
        trade_value: float
    ) -> float:
        """Calculate leverage impact of trade"""
        
        try:
            total_value = portfolio.get("total_value", 0)
            
            if total_value <= 0:
                return 0.0
            
            # Calculate current long exposure
            positions = portfolio.get("positions", [])
            current_long_value = sum(
                pos.get("market_value", 0) for pos in positions
                if pos.get("market_value", 0) > 0
            )
            
            # Calculate new long exposure after trade
            if action == "buy":
                new_long_value = current_long_value + trade_value
            else:
                new_long_value = max(0, current_long_value - trade_value)
            
            # Calculate leverage
            return new_long_value / total_value
            
        except Exception as e:
            logger.error(f"Leverage impact calculation failed: {e}")
            return 0.0
    
    async def _assess_trade_risk_impact(
        self,
        portfolio_id: str,
        symbol: str,
        action: str,
        quantity: int,
        price: float
    ) -> Dict[str, Any]:
        """Assess risk impact of a specific trade"""
        
        try:
            # Get current risk metrics
            current_risk = self.risk_metrics_cache.get(portfolio_id)
            
            if not current_risk:
                await self.calculate_portfolio_risk(portfolio_id)
                current_risk = self.risk_metrics_cache.get(portfolio_id)
            
            if not current_risk:
                return {"risk_score": 0.5, "factors": []}
            
            # Calculate trade impact factors
            trade_value = abs(quantity) * price
            risk_factors = []
            
            # Concentration impact
            if hasattr(current_risk, 'largest_position_pct'):
                concentration_impact = trade_value / max(1, trade_value * 10)  # Simplified
                risk_factors.append({
                    "factor": "concentration",
                    "impact": concentration_impact,
                    "weight": 0.3
                })
            
            # Leverage impact  
            if hasattr(current_risk, 'leverage_ratio'):
                leverage_impact = min(1.0, current_risk.leverage_ratio / self.config["max_leverage"])
                risk_factors.append({
                    "factor": "leverage",
                    "impact": leverage_impact,
                    "weight": 0.3
                })
            
            # Liquidity impact
            if hasattr(current_risk, 'liquidity_risk'):
                liquidity_impact = current_risk.liquidity_risk
                risk_factors.append({
                    "factor": "liquidity",
                    "impact": liquidity_impact,
                    "weight": 0.2
                })
            
            # Correlation impact
            if hasattr(current_risk, 'correlation_risk'):
                correlation_impact = current_risk.correlation_risk
                risk_factors.append({
                    "factor": "correlation",
                    "impact": correlation_impact,
                    "weight": 0.2
                })
            
            # Calculate weighted risk score
            total_weight = sum(factor["weight"] for factor in risk_factors)
            if total_weight > 0:
                risk_score = sum(
                    factor["impact"] * factor["weight"] for factor in risk_factors
                ) / total_weight
            else:
                risk_score = 0.5
            
            return {
                "risk_score": min(1.0, max(0.0, risk_score)),
                "factors": risk_factors
            }
            
        except Exception as e:
            logger.error(f"Trade risk impact assessment failed: {e}")
            return {"risk_score": 0.5, "factors": []}
    
    async def _check_risk_thresholds(
        self,
        portfolio_id: str,
        risk_metrics: Dict[str, Any]
    ) -> List[RiskAlert]:
        """Check risk metrics against thresholds"""
        
        try:
            alerts = []
            portfolio_data = await self._get_portfolio_data(portfolio_id)
            user_id = portfolio_data.get("user_id", "") if portfolio_data else ""
            
            # Check max drawdown
            if risk_metrics.get("max_drawdown", 0) > self.config["max_portfolio_drawdown"]:
                alert = RiskAlert(
                    alert_id=f"{portfolio_id}_drawdown_{int(datetime.utcnow().timestamp())}",
                    portfolio_id=portfolio_id,
                    user_id=user_id,
                    alert_type="drawdown",
                    severity="high",
                    message=f"Portfolio drawdown {risk_metrics['max_drawdown']:.1%} exceeds limit {self.config['max_portfolio_drawdown']:.1%}",
                    current_value=risk_metrics["max_drawdown"],
                    threshold_value=self.config["max_portfolio_drawdown"],
                    timestamp=datetime.utcnow(),
                    acknowledged=False
                )
                alerts.append(alert)
            
            # Check concentration risk
            if risk_metrics.get("largest_position_pct", 0) > self.config["max_position_concentration"]:
                alert = RiskAlert(
                    alert_id=f"{portfolio_id}_concentration_{int(datetime.utcnow().timestamp())}",
                    portfolio_id=portfolio_id,
                    user_id=user_id,
                    alert_type="concentration",
                    severity="medium",
                    message=f"Largest position {risk_metrics['largest_position_pct']:.1%} exceeds concentration limit {self.config['max_position_concentration']:.1%}",
                    current_value=risk_metrics["largest_position_pct"],
                    threshold_value=self.config["max_position_concentration"],
                    timestamp=datetime.utcnow(),
                    acknowledged=False
                )
                alerts.append(alert)
            
            # Check leverage
            if risk_metrics.get("leverage_ratio", 0) > self.config["max_leverage"]:
                alert = RiskAlert(
                    alert_id=f"{portfolio_id}_leverage_{int(datetime.utcnow().timestamp())}",
                    portfolio_id=portfolio_id,
                    user_id=user_id,
                    alert_type="leverage",
                    severity="high",
                    message=f"Portfolio leverage {risk_metrics['leverage_ratio']:.2f}x exceeds limit {self.config['max_leverage']:.2f}x",
                    current_value=risk_metrics["leverage_ratio"],
                    threshold_value=self.config["max_leverage"],
                    timestamp=datetime.utcnow(),
                    acknowledged=False
                )
                alerts.append(alert)
            
            # Check cash ratio
            if risk_metrics.get("cash_ratio", 0) < self.config["min_cash_ratio"]:
                alert = RiskAlert(
                    alert_id=f"{portfolio_id}_cash_low_{int(datetime.utcnow().timestamp())}",
                    portfolio_id=portfolio_id,
                    user_id=user_id,
                    alert_type="cash_low",
                    severity="medium",
                    message=f"Cash ratio {risk_metrics['cash_ratio']:.1%} below minimum {self.config['min_cash_ratio']:.1%}",
                    current_value=risk_metrics["cash_ratio"],
                    threshold_value=self.config["min_cash_ratio"],
                    timestamp=datetime.utcnow(),
                    acknowledged=False
                )
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Risk threshold checking failed: {e}")
            return []
    
    async def _process_risk_alert(self, alert: RiskAlert):
        """Process and send risk alert"""
        
        try:
            # Check if alert is subscribed to
            subscription_key = f"{alert.user_id}:{alert.portfolio_id}"
            subscription = self.alert_subscriptions.get(subscription_key)
            
            if not subscription or not subscription.get("active"):
                return
            
            if alert.alert_type not in subscription.get("alert_types", []):
                return
            
            # Check cooldown period
            if await self._is_alert_in_cooldown(alert):
                return
            
            # Store alert
            await self._store_risk_alert(alert)
            
            # Update monitoring stats
            self.monitoring_stats["alerts_generated"] += 1
            
            logger.info(f"Risk alert generated: {alert.alert_type} for portfolio {alert.portfolio_id}")
            
        except Exception as e:
            logger.error(f"Risk alert processing failed: {e}")
    
    async def _is_alert_in_cooldown(self, alert: RiskAlert) -> bool:
        """Check if similar alert is in cooldown period"""
        
        try:
            cooldown_key = f"alert_cooldown:{alert.portfolio_id}:{alert.alert_type}"
            last_alert_time = await self.redis.get(cooldown_key)
            
            if last_alert_time:
                last_time = datetime.fromisoformat(last_alert_time)
                time_diff = datetime.utcnow() - last_time
                
                if time_diff.total_seconds() < (self.config["alert_cooldown_minutes"] * 60):
                    return True
            
            # Set cooldown for this alert type
            await self.redis.setex(
                cooldown_key,
                self.config["alert_cooldown_minutes"] * 60,
                datetime.utcnow().isoformat()
            )
            
            return False
            
        except Exception as e:
            logger.error(f"Alert cooldown check failed: {e}")
            return False
    
    async def _store_risk_alert(self, alert: RiskAlert):
        """Store risk alert in Redis"""
        
        try:
            alert_key = f"risk_alert:{alert.alert_id}"
            
            await self.redis.setex(
                alert_key,
                86400 * 7,  # 7 days TTL
                json.dumps(asdict(alert), default=str)
            )
            
            # Add to user's alert list
            user_alerts_key = f"user_alerts:{alert.user_id}"
            await self.redis.lpush(user_alerts_key, alert.alert_id)
            await self.redis.ltrim(user_alerts_key, 0, 99)  # Keep last 100 alerts
            
        except Exception as e:
            logger.error(f"Risk alert storage failed: {e}")
    
    async def _store_risk_metrics(self, risk_metrics: RiskMetrics):
        """Store risk metrics in Redis"""
        
        try:
            # Store current metrics
            current_key = f"risk_metrics:{risk_metrics.portfolio_id}:current"
            await self.redis.set(
                current_key,
                json.dumps(asdict(risk_metrics), default=str)
            )
            
            # Store in time series for historical tracking
            timestamp = risk_metrics.last_updated.timestamp()
            history_key = f"risk_metrics:{risk_metrics.portfolio_id}:history"
            
            await self.redis.zadd(
                history_key,
                {json.dumps(asdict(risk_metrics), default=str): timestamp}
            )
            
            # Cleanup old history (keep 30 days)
            cutoff_time = datetime.utcnow() - timedelta(days=30)
            await self.redis.zremrangebyscore(
                history_key, 0, cutoff_time.timestamp()
            )
            
        except Exception as e:
            logger.error(f"Risk metrics storage failed: {e}")
    
    async def _get_portfolio_data(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio data from Redis"""
        
        try:
            portfolio_key = f"portfolio:{portfolio_id}"
            portfolio_data = await self.redis.get(portfolio_key)
            
            if portfolio_data:
                return json.loads(portfolio_data)
            return None
            
        except Exception as e:
            logger.error(f"Portfolio data retrieval failed: {e}")
            return None
    
    async def _get_portfolio_positions(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Get portfolio positions from Redis"""
        
        try:
            position_keys = await self.redis.keys(f"positions:{portfolio_id}:*")
            positions = []
            
            for key in position_keys:
                position_data = await self.redis.get(key)
                if position_data:
                    positions.append(json.loads(position_data))
            
            return positions
            
        except Exception as e:
            logger.error(f"Portfolio positions retrieval failed: {e}")
            return []
    
    async def _get_historical_pnl(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Get historical P&L data for portfolio"""
        
        try:
            # Get last 30 days of P&L data
            now = datetime.utcnow()
            start_time = now - timedelta(days=30)
            
            history_key = f"pnl:portfolio:{portfolio_id}:history"
            
            history_data = await self.redis.zrangebyscore(
                history_key,
                start_time.timestamp(),
                now.timestamp()
            )
            
            pnl_history = []
            for data in history_data:
                pnl_point = json.loads(data)
                pnl_history.append(pnl_point)
            
            return pnl_history
            
        except Exception as e:
            logger.error(f"Historical P&L retrieval failed: {e}")
            return []
    
    async def _load_alert_subscriptions(self):
        """Load alert subscriptions from Redis"""
        
        try:
            subscription_keys = await self.redis.keys("risk_alert_subscription:*")
            
            for key in subscription_keys:
                subscription_data = await self.redis.get(key)
                if subscription_data:
                    subscription = json.loads(subscription_data)
                    subscription_key = f"{subscription['user_id']}:{subscription['portfolio_id']}"
                    self.alert_subscriptions[subscription_key] = subscription
            
            logger.info(f"Loaded {len(self.alert_subscriptions)} alert subscriptions")
            
        except Exception as e:
            logger.error(f"Alert subscriptions loading failed: {e}")
    
    async def _load_risk_metrics_cache(self):
        """Load risk metrics cache from Redis"""
        
        try:
            metrics_keys = await self.redis.keys("risk_metrics:*:current")
            
            for key in metrics_keys:
                portfolio_id = key.split(":")[1]
                metrics_data = await self.redis.get(key)
                
                if metrics_data:
                    metrics_dict = json.loads(metrics_data)
                    risk_metrics = RiskMetrics(
                        portfolio_id=metrics_dict["portfolio_id"],
                        var_1d=metrics_dict["var_1d"],
                        var_5d=metrics_dict["var_5d"],
                        max_drawdown=metrics_dict["max_drawdown"],
                        concentration_risk=metrics_dict["concentration_risk"],
                        leverage_ratio=metrics_dict["leverage_ratio"],
                        sharpe_ratio=metrics_dict["sharpe_ratio"],
                        beta=metrics_dict["beta"],
                        correlation_risk=metrics_dict["correlation_risk"],
                        liquidity_risk=metrics_dict["liquidity_risk"],
                        position_count=metrics_dict["position_count"],
                        largest_position_pct=metrics_dict["largest_position_pct"],
                        cash_ratio=metrics_dict["cash_ratio"],
                        last_updated=datetime.fromisoformat(metrics_dict["last_updated"])
                    )
                    self.risk_metrics_cache[portfolio_id] = risk_metrics
            
            logger.info(f"Loaded risk metrics cache for {len(self.risk_metrics_cache)} portfolios")
            
        except Exception as e:
            logger.error(f"Risk metrics cache loading failed: {e}")
    
    async def _update_monitoring_stats(self, assessment_time_ms: float):
        """Update risk monitoring statistics"""
        
        try:
            self.monitoring_stats["risk_assessments"] += 1
            
            # Update average assessment time (exponential moving average)
            alpha = 0.1
            current_avg = self.monitoring_stats["avg_assessment_time_ms"]
            self.monitoring_stats["avg_assessment_time_ms"] = (
                alpha * assessment_time_ms + (1 - alpha) * current_avg
            )
            
        except Exception as e:
            logger.error(f"Monitoring stats update failed: {e}")
    
    # Background tasks
    async def _risk_monitoring_task(self):
        """Background task for continuous risk monitoring"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Get all active portfolios
                portfolio_keys = await self.redis.keys("portfolio:*")
                
                for key in portfolio_keys:
                    portfolio_id = key.split(":")[-1]
                    try:
                        await self.monitor_portfolio_risk(portfolio_id)
                        await asyncio.sleep(1)  # Small delay between portfolios
                    except Exception as e:
                        logger.error(f"Risk monitoring failed for portfolio {portfolio_id}: {e}")
                
            except Exception as e:
                logger.error(f"Risk monitoring task error: {e}")
                await asyncio.sleep(60)
    
    async def _alert_cleanup_task(self):
        """Background task to cleanup old alerts"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Cleanup old alerts (older than 7 days)
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                
                alert_keys = await self.redis.keys("risk_alert:*")
                
                for key in alert_keys:
                    alert_data = await self.redis.get(key)
                    if alert_data:
                        alert = json.loads(alert_data)
                        alert_time = datetime.fromisoformat(alert["timestamp"])
                        
                        if alert_time < cutoff_time:
                            await self.redis.delete(key)
                
                logger.info("Risk alert cleanup completed")
                
            except Exception as e:
                logger.error(f"Alert cleanup task error: {e}")
                await asyncio.sleep(3600)

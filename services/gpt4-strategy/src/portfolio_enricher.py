"""
Portfolio Context Enrichment for Strategy Generation
"""

import asyncio
import asyncpg
import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PortfolioEnricher:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        
    async def get_db_connection(self):
        """Get database connection"""
        try:
            return await asyncpg.connect(self.db_url)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    async def enrich_context(
        self,
        market_context: Dict[str, Any],
        portfolio_context: Optional[Dict[str, Any]],
        user_id: str
    ) -> Dict[str, Any]:
        """Enrich context with portfolio and market data"""
        
        try:
            conn = await self.get_db_connection()
            
            enriched_context = {
                "market": await self._enrich_market_context(conn, market_context),
                "portfolio": await self._enrich_portfolio_context(conn, portfolio_context, user_id),
                "user_profile": await self._get_user_profile(conn, user_id),
                "correlations": await self._calculate_correlations(conn, portfolio_context)
            }
            
            await conn.close()
            return enriched_context
            
        except Exception as e:
            logger.error(f"Context enrichment failed: {e}")
            return {
                "market": market_context or {},
                "portfolio": portfolio_context or {},
                "user_profile": {},
                "correlations": {}
            }
    
    async def _enrich_market_context(
        self,
        conn,
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add real-time market data and analysis"""
        
        try:
            # Get current VIX and market indicators
            market_data = await conn.fetchrow("""
                SELECT 
                    vix_level,
                    spy_price,
                    qqq_price,
                    market_trend,
                    sector_rotation,
                    options_volume,
                    timestamp
                FROM market_indicators 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            
            # Get upcoming earnings
            upcoming_earnings = await conn.fetch("""
                SELECT symbol, earnings_date, expected_move
                FROM earnings_calendar 
                WHERE earnings_date BETWEEN NOW() AND NOW() + INTERVAL '14 days'
                ORDER BY earnings_date
            """)
            
            enriched = dict(market_context) if market_context else {}
            
            if market_data:
                enriched.update({
                    "vix": float(market_data["vix_level"]),
                    "spy_price": float(market_data["spy_price"]),
                    "market_trend": market_data["market_trend"],
                    "sector_rotation": market_data["sector_rotation"],
                    "options_volume": market_data["options_volume"],
                    "last_updated": market_data["timestamp"].isoformat()
                })
            
            enriched["upcoming_earnings"] = [
                {
                    "symbol": row["symbol"],
                    "date": row["earnings_date"].isoformat(),
                    "expected_move": float(row["expected_move"]) if row["expected_move"] else None
                }
                for row in upcoming_earnings
            ]
            
            return enriched
            
        except Exception as e:
            logger.error(f"Market context enrichment failed: {e}")
            return market_context or {}
    
    async def _enrich_portfolio_context(
        self,
        conn,
        portfolio_context: Optional[Dict[str, Any]],
        user_id: str
    ) -> Dict[str, Any]:
        """Enrich with current portfolio positions and Greeks"""
        
        try:
            # Get current positions
            positions = await conn.fetch("""
                SELECT 
                    symbol,
                    position_type,
                    quantity,
                    strike_price,
                    expiration_date,
                    premium_paid,
                    current_value,
                    delta,
                    gamma,
                    theta,
                    vega,
                    last_updated
                FROM user_positions 
                WHERE user_id = $1 AND quantity != 0
            """, user_id)
            
            # Calculate portfolio Greeks
            total_delta = sum(float(pos["delta"] or 0) * pos["quantity"] for pos in positions)
            total_gamma = sum(float(pos["gamma"] or 0) * pos["quantity"] for pos in positions)
            total_theta = sum(float(pos["theta"] or 0) * pos["quantity"] for pos in positions)
            total_vega = sum(float(pos["vega"] or 0) * pos["quantity"] for pos in positions)
            
            # Calculate portfolio value and P&L
            total_value = sum(float(pos["current_value"] or 0) for pos in positions)
            total_cost = sum(float(pos["premium_paid"] or 0) * pos["quantity"] for pos in positions)
            unrealized_pnl = total_value - total_cost
            
            enriched = dict(portfolio_context) if portfolio_context else {}
            
            enriched.update({
                "positions": [
                    {
                        "symbol": pos["symbol"],
                        "type": pos["position_type"],
                        "quantity": pos["quantity"],
                        "strike": float(pos["strike_price"]) if pos["strike_price"] else None,
                        "expiration": pos["expiration_date"].isoformat() if pos["expiration_date"] else None,
                        "current_value": float(pos["current_value"]) if pos["current_value"] else 0,
                        "delta": float(pos["delta"]) if pos["delta"] else 0,
                        "gamma": float(pos["gamma"]) if pos["gamma"] else 0,
                        "theta": float(pos["theta"]) if pos["theta"] else 0,
                        "vega": float(pos["vega"]) if pos["vega"] else 0
                    }
                    for pos in positions
                ],
                "portfolio_greeks": {
                    "total_delta": total_delta,
                    "total_gamma": total_gamma,
                    "total_theta": total_theta,
                    "total_vega": total_vega
                },
                "portfolio_summary": {
                    "total_positions": len(positions),
                    "total_value": total_value,
                    "unrealized_pnl": unrealized_pnl,
                    "pnl_percentage": (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
                }
            })
            
            return enriched
            
        except Exception as e:
            logger.error(f"Portfolio context enrichment failed: {e}")
            return portfolio_context or {}
    
    async def _get_user_profile(self, conn, user_id: str) -> Dict[str, Any]:
        """Get user trading profile and preferences"""
        
        try:
            profile = await conn.fetchrow("""
                SELECT 
                    risk_tolerance,
                    preferred_strategies,
                    max_position_size,
                    trading_experience,
                    account_type,
                    created_at
                FROM user_profiles 
                WHERE user_id = $1
            """, user_id)
            
            if profile:
                return {
                    "risk_tolerance": profile["risk_tolerance"],
                    "preferred_strategies": json.loads(profile["preferred_strategies"] or "[]"),
                    "max_position_size": float(profile["max_position_size"]) if profile["max_position_size"] else 10000,
                    "trading_experience": profile["trading_experience"],
                    "account_type": profile["account_type"],
                    "member_since": profile["created_at"].isoformat() if profile["created_at"] else None
                }
            else:
                return {"risk_tolerance": "medium", "max_position_size": 5000}
                
        except Exception as e:
            logger.error(f"User profile fetch failed: {e}")
            return {}
    
    async def _calculate_correlations(
        self,
        conn,
        portfolio_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate portfolio correlations and concentration risk"""
        
        try:
            if not portfolio_context or not portfolio_context.get("positions"):
                return {}
            
            # Get symbols from portfolio
            symbols = [pos.get("symbol") for pos in portfolio_context.get("positions", [])]
            
            if len(symbols) < 2:
                return {"concentration_risk": "low", "diversification_score": 1.0}
            
            # Calculate sector concentrations
            sector_data = await conn.fetch("""
                SELECT symbol, sector, industry
                FROM stock_sectors 
                WHERE symbol = ANY($1)
            """, symbols)
            
            sector_counts = {}
            for row in sector_data:
                sector = row["sector"]
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            # Calculate concentration risk
            total_positions = len(symbols)
            max_sector_concentration = max(sector_counts.values()) / total_positions if sector_counts else 0
            
            if max_sector_concentration > 0.6:
                concentration_risk = "high"
            elif max_sector_concentration > 0.4:
                concentration_risk = "medium"
            else:
                concentration_risk = "low"
            
            return {
                "sector_breakdown": sector_counts,
                "concentration_risk": concentration_risk,
                "diversification_score": 1 - max_sector_concentration,
                "total_symbols": len(set(symbols))
            }
            
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return {}
    
    async def calculate_impact(
        self,
        strategies: List[Dict[str, Any]],
        current_portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate impact of new strategies on existing portfolio"""
        
        try:
            # Extract current portfolio Greeks
            current_greeks = current_portfolio.get("portfolio_greeks", {})
            current_delta = current_greeks.get("total_delta", 0)
            current_gamma = current_greeks.get("total_gamma", 0)
            current_theta = current_greeks.get("total_theta", 0)
            current_vega = current_greeks.get("total_vega", 0)
            
            impacts = []
            
            for strategy in strategies:
                # Calculate strategy Greeks impact
                strategy_delta = strategy.get("delta", 0)
                strategy_gamma = strategy.get("gamma", 0)
                strategy_theta = strategy.get("theta", 0)
                strategy_vega = strategy.get("vega", 0)
                
                # New portfolio Greeks
                new_delta = current_delta + strategy_delta
                new_gamma = current_gamma + strategy_gamma
                new_theta = current_theta + strategy_theta
                new_vega = current_vega + strategy_vega
                
                # Calculate impact metrics
                delta_change_pct = (strategy_delta / abs(current_delta)) * 100 if current_delta != 0 else 0
                
                impact = {
                    "strategy_name": strategy.get("name", "Unknown"),
                    "greek_changes": {
                        "delta_change": strategy_delta,
                        "delta_change_pct": delta_change_pct,
                        "new_portfolio_delta": new_delta,
                        "gamma_change": strategy_gamma,
                        "theta_change": strategy_theta,
                        "vega_change": strategy_vega
                    },
                    "risk_impact": self._assess_risk_impact(
                        current_delta, new_delta,
                        current_vega, new_vega
                    ),
                    "diversification_impact": self._assess_diversification_impact(
                        strategy, current_portfolio
                    )
                }
                
                impacts.append(impact)
            
            return {
                "strategy_impacts": impacts,
                "overall_assessment": self._overall_portfolio_assessment(impacts)
            }
            
        except Exception as e:
            logger.error(f"Portfolio impact calculation failed: {e}")
            return {"error": "Impact calculation unavailable"}
    
    def _assess_risk_impact(
        self,
        current_delta: float,
        new_delta: float,
        current_vega: float,
        new_vega: float
    ) -> Dict[str, Any]:
        """Assess risk impact of adding strategy"""
        
        delta_change = abs(new_delta - current_delta)
        vega_change = abs(new_vega - current_vega)
        
        if delta_change < 10 and vega_change < 50:
            risk_level = "low"
        elif delta_change < 25 and vega_change < 100:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return {
            "risk_level": risk_level,
            "delta_impact": "increased" if new_delta > current_delta else "decreased",
            "vega_impact": "increased" if new_vega > current_vega else "decreased"
        }
    
    def _assess_diversification_impact(
        self,
        strategy: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> str:
        """Assess diversification impact"""
        
        # Simple assessment based on strategy type
        strategy_type = strategy.get("type", "").lower()
        
        # Get existing strategy types
        existing_positions = portfolio.get("positions", [])
        existing_types = [pos.get("type", "").lower() for pos in existing_positions]
        
        if strategy_type in existing_types:
            return "neutral"  # Same type already exists
        else:
            return "positive"  # New strategy type adds diversification
    
    def _overall_portfolio_assessment(
        self,
        impacts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Provide overall assessment of portfolio impact"""
        
        if not impacts:
            return {"assessment": "no_impact"}
        
        risk_levels = [impact["risk_impact"]["risk_level"] for impact in impacts]
        high_risk_count = risk_levels.count("high")
        
        if high_risk_count > 0:
            overall_risk = "high"
        elif risk_levels.count("medium") > len(impacts) / 2:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        return {
            "overall_risk": overall_risk,
            "strategies_analyzed": len(impacts),
            "recommendation": "review_carefully" if overall_risk == "high" else "proceed_with_caution" if overall_risk == "medium" else "looks_good"
        }

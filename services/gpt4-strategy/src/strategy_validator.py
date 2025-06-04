"""
Strategy Validation Engine for Options Strategies
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    is_valid: bool
    score: float
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]

class StrategyValidator:
    def __init__(self):
        # Validation thresholds
        self.max_position_size = 100000  # $100k default max
        self.min_liquidity_threshold = 100  # Min daily volume
        self.max_days_to_expiration = 365
        self.min_days_to_expiration = 1
        
        # Risk limits
        self.max_delta_exposure = 500
        self.max_vega_exposure = 1000
        self.max_theta_exposure = -100
        
    async def validate_strategy(
        self,
        strategy: Dict[str, Any],
        portfolio_context: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> ValidationResult:
        """Comprehensive strategy validation"""
        
        warnings = []
        errors = []
        recommendations = []
        score = 1.0
        
        try:
            # Basic structure validation
            structure_result = self._validate_structure(strategy)
            if not structure_result.is_valid:
                return structure_result
            
            # Risk validation
            risk_score, risk_warnings = self._validate_risk_parameters(
                strategy, portfolio_context
            )
            warnings.extend(risk_warnings)
            score *= risk_score
            
            # Market condition validation
            market_score, market_warnings = self._validate_market_conditions(
                strategy, market_context
            )
            warnings.extend(market_warnings)
            score *= market_score
            
            # Liquidity validation
            liquidity_score, liquidity_warnings = await self._validate_liquidity(strategy)
            warnings.extend(liquidity_warnings)
            score *= liquidity_score
            
            # Portfolio fit validation
            fit_score, fit_warnings = self._validate_portfolio_fit(
                strategy, portfolio_context
            )
            warnings.extend(fit_warnings)
            score *= fit_score
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                strategy, warnings, portfolio_context
            )
            
            is_valid = score >= 0.6 and len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                score=score,
                warnings=warnings,
                errors=errors,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Strategy validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                score=0.0,
                warnings=[],
                errors=[f"Validation error: {str(e)}"],
                recommendations=["Manual review required"]
            )
    
    def _validate_structure(self, strategy: Dict[str, Any]) -> ValidationResult:
        """Validate basic strategy structure"""
        
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ["name", "type", "legs", "max_profit", "max_loss"]
        for field in required_fields:
            if field not in strategy:
                errors.append(f"Missing required field: {field}")
        
        # Validate legs
        legs = strategy.get("legs", [])
        if not legs:
            errors.append("Strategy must have at least one leg")
        
        for i, leg in enumerate(legs):
            leg_errors = self._validate_leg(leg, i)
            errors.extend(leg_errors)
        
        # Validate profit/loss calculations
        max_profit = strategy.get("max_profit")
        max_loss = strategy.get("max_loss")
        
        if max_profit is not None and max_loss is not None:
            if max_profit <= 0 and max_loss >= 0:
                warnings.append("Strategy shows no profit potential with unlimited loss")
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            score=1.0 if is_valid else 0.0,
            warnings=warnings,
            errors=errors,
            recommendations=[]
        )
    
    def _validate_leg(self, leg: Dict[str, Any], leg_index: int) -> List[str]:
        """Validate individual option leg"""
        
        errors = []
        
        # Required leg fields
        required_leg_fields = ["action", "option_type", "strike", "expiration", "quantity"]
        for field in required_leg_fields:
            if field not in leg:
                errors.append(f"Leg {leg_index}: Missing required field '{field}'")
        
        # Validate action
        if leg.get("action") not in ["buy", "sell"]:
            errors.append(f"Leg {leg_index}: Action must be 'buy' or 'sell'")
        
        # Validate option type
        if leg.get("option_type") not in ["call", "put"]:
            errors.append(f"Leg {leg_index}: Option type must be 'call' or 'put'")
        
        # Validate strike price
        strike = leg.get("strike")
        if strike is not None and (not isinstance(strike, (int, float)) or strike <= 0):
            errors.append(f"Leg {leg_index}: Strike price must be positive number")
        
        # Validate quantity
        quantity = leg.get("quantity")
        if quantity is not None and (not isinstance(quantity, int) or quantity <= 0):
            errors.append(f"Leg {leg_index}: Quantity must be positive integer")
        
        # Validate expiration
        expiration = leg.get("expiration")
        if expiration:
            try:
                exp_date = datetime.fromisoformat(expiration.replace("Z", "+00:00"))
                days_to_exp = (exp_date - datetime.now()).days
                
                if days_to_exp < self.min_days_to_expiration:
                    errors.append(f"Leg {leg_index}: Expiration too close ({days_to_exp} days)")
                elif days_to_exp > self.max_days_to_expiration:
                    errors.append(f"Leg {leg_index}: Expiration too far ({days_to_exp} days)")
                    
            except (ValueError, TypeError):
                errors.append(f"Leg {leg_index}: Invalid expiration date format")
        
        return errors
    
    def _validate_risk_parameters(
        self,
        strategy: Dict[str, Any],
        portfolio_context: Dict[str, Any]
    ) -> tuple[float, List[str]]:
        """Validate risk parameters and Greeks"""
        
        warnings = []
        score = 1.0
        
        # Check Greeks limits
        delta = strategy.get("delta", 0)
        gamma = strategy.get("gamma", 0)
        theta = strategy.get("theta", 0)
        vega = strategy.get("vega", 0)
        
        # Current portfolio Greeks
        portfolio_greeks = portfolio_context.get("portfolio_greeks", {})
        current_delta = portfolio_greeks.get("total_delta", 0)
        current_vega = portfolio_greeks.get("total_vega", 0)
        
        # Delta exposure check
        new_delta = current_delta + delta
        if abs(new_delta) > self.max_delta_exposure:
            warnings.append(f"High delta exposure: {new_delta:.2f} (limit: {self.max_delta_exposure})")
            score *= 0.8
        
        # Vega exposure check
        new_vega = current_vega + vega
        if abs(new_vega) > self.max_vega_exposure:
            warnings.append(f"High vega exposure: {new_vega:.2f} (limit: {self.max_vega_exposure})")
            score *= 0.8
        
        # Theta decay check
        if theta < self.max_theta_exposure:
            warnings.append(f"High theta decay: {theta:.2f} per day")
            score *= 0.9
        
        # Capital requirement check
        capital_required = strategy.get("capital_required", 0)
        max_position = portfolio_context.get("max_position_size", self.max_position_size)
        
        if capital_required > max_position:
            warnings.append(f"Capital required (${capital_required:,.0f}) exceeds limit (${max_position:,.0f})")
            score *= 0.7
        
        # Risk-reward ratio
        max_profit = strategy.get("max_profit", 0)
        max_loss = abs(strategy.get("max_loss", 0))
        
        if max_loss > 0:
            risk_reward_ratio = max_profit / max_loss
            if risk_reward_ratio < 0.5:
                warnings.append(f"Poor risk-reward ratio: {risk_reward_ratio:.2f}")
                score *= 0.8
        
        return score, warnings
    
    def _validate_market_conditions(
        self,
        strategy: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> tuple[float, List[str]]:
        """Validate strategy against current market conditions"""
        
        warnings = []
        score = 1.0
        
        vix_level = market_context.get("vix", 20)
        market_trend = market_context.get("market_trend", "neutral")
        
        # VIX-based validation
        strategy_type = strategy.get("type", "").lower()
        ideal_condition = strategy.get("ideal_market_condition", "").lower()
        
        # High VIX warnings
        if vix_level > 30:
            if "sell" in strategy_type and "volatility" in strategy_type:
                warnings.append("High VIX environment - volatility selling strategies risky")
                score *= 0.8
        
        # Low VIX warnings
        if vix_level < 15:
            if "buy" in strategy_type and "volatility" in strategy_type:
                warnings.append("Low VIX environment - volatility buying strategies expensive")
                score *= 0.9
        
        # Trend alignment
        if ideal_condition and market_trend:
            if "bullish" in ideal_condition and market_trend == "bearish":
                warnings.append(f"Strategy expects {ideal_condition} but market is {market_trend}")
                score *= 0.8
            elif "bearish" in ideal_condition and market_trend == "bullish":
                warnings.append(f"Strategy expects {ideal_condition} but market is {market_trend}")
                score *= 0.8
        
        return score, warnings
    
    async def _validate_liquidity(
        self,
        strategy: Dict[str, Any]
    ) -> tuple[float, List[str]]:
        """Validate liquidity of strategy components"""
        
        warnings = []
        score = 1.0
        
        try:
            legs = strategy.get("legs", [])
            
            for i, leg in enumerate(legs):
                # In production, check actual option chain liquidity
                # For now, use basic validation
                
                strike = leg.get("strike", 0)
                symbol = leg.get("symbol", "SPY")  # Default to SPY
                
                # Check if strike is reasonable (within 20% of current price)
                # This is a simplified check - in production would use real data
                if symbol == "SPY":  # Example validation
                    current_price = 450  # Would get from market data
                    if abs(strike - current_price) / current_price > 0.2:
                        warnings.append(f"Leg {i}: Strike {strike} may have low liquidity (far from current price)")
                        score *= 0.9
            
            return score, warnings
            
        except Exception as e:
            logger.error(f"Liquidity validation failed: {e}")
            return 0.8, ["Unable to validate liquidity"]
    
    def _validate_portfolio_fit(
        self,
        strategy: Dict[str, Any],
        portfolio_context: Dict[str, Any]
    ) -> tuple[float, List[str]]:
        """Validate how well strategy fits with existing portfolio"""
        
        warnings = []
        score = 1.0
        
        # Check for position concentration
        current_positions = portfolio_context.get("positions", [])
        strategy_symbols = self._extract_symbols_from_strategy(strategy)
        
        existing_symbols = [pos.get("symbol") for pos in current_positions]
        
        for symbol in strategy_symbols:
            symbol_count = existing_symbols.count(symbol)
            if symbol_count >= 3:
                warnings.append(f"High concentration in {symbol} (already {symbol_count} positions)")
                score *= 0.8
        
        # Check account type compatibility
        account_type = portfolio_context.get("account_type", "margin")
        strategy_type = strategy.get("type", "").lower()
        
        if account_type == "cash" and "naked" in strategy_type:
            warnings.append("Naked options not allowed in cash accounts")
            score *= 0.6
        
        # Check experience level
        experience = portfolio_context.get("trading_experience", "intermediate")
        complexity_score = self._assess_strategy_complexity(strategy)
        
        if experience == "beginner" and complexity_score > 0.7:
            warnings.append("Complex strategy for beginner trader")
            score *= 0.7
        
        return score, warnings
    
    def _extract_symbols_from_strategy(self, strategy: Dict[str, Any]) -> List[str]:
        """Extract underlying symbols from strategy legs"""
        symbols = []
        for leg in strategy.get("legs", []):
            symbol = leg.get("symbol")
            if symbol:
                symbols.append(symbol)
        return symbols
    
    def _assess_strategy_complexity(self, strategy: Dict[str, Any]) -> float:
        """Assess strategy complexity (0=simple, 1=very complex)"""
        
        leg_count = len(strategy.get("legs", []))
        strategy_type = strategy.get("type", "").lower()
        
        # Base complexity from leg count
        complexity = min(leg_count / 4, 1.0)
        
        # Adjust for strategy type
        complex_strategies = ["iron_condor", "butterfly", "calendar", "diagonal"]
        if any(complex_type in strategy_type for complex_type in complex_strategies):
            complexity += 0.3
        
        return min(complexity, 1.0)
    
    def _generate_recommendations(
        self,
        strategy: Dict[str, Any],
        warnings: List[str],
        portfolio_context: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Based on warnings
        if any("vega" in warning.lower() for warning in warnings):
            recommendations.append("Consider hedging vega exposure with opposite volatility trades")
        
        if any("delta" in warning.lower() for warning in warnings):
            recommendations.append("Consider delta hedging with underlying stock or futures")
        
        if any("concentration" in warning.lower() for warning in warnings):
            recommendations.append("Diversify across different underlying assets")
        
        if any("liquidity" in warning.lower() for warning in warnings):
            recommendations.append("Check bid-ask spreads before execution")
        
        # Strategy-specific recommendations
        strategy_type = strategy.get("type", "").lower()
        
        if "covered_call" in strategy_type:
            recommendations.append("Monitor for early assignment risk near ex-dividend dates")
        
        if "iron_condor" in strategy_type:
            recommendations.append("Plan exit strategy at 25-50% profit to avoid gamma risk")
        
        if "straddle" in strategy_type or "strangle" in strategy_type:
            recommendations.append("Monitor implied volatility levels for optimal entry/exit")
        
        return recommendations

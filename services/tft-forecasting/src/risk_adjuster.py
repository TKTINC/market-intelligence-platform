"""
Risk-Adjusted Forecasting for TFT Price Predictions
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class RiskAdjustment:
    adjustment_factor: float
    risk_score: float
    confidence_penalty: float
    volatility_adjustment: float
    regime_adjustment: float
    liquidity_adjustment: float

class RiskAdjustedForecaster:
    def __init__(self):
        self.risk_models = {}
        self.adjustment_history = {}
        
        # Risk adjustment parameters
        self.max_adjustment = 0.15  # Maximum 15% adjustment
        self.confidence_threshold = 0.7
        self.volatility_threshold = 0.25
        
        # Risk factor weights
        self.risk_weights = {
            'market_regime': 0.3,
            'volatility_regime': 0.25,
            'liquidity_risk': 0.2,
            'model_uncertainty': 0.15,
            'tail_risk': 0.1
        }
    
    async def adjust_forecasts(
        self,
        forecasts: Dict[str, Any],
        market_regime: Any,
        symbol: str
    ) -> Dict[str, Any]:
        """Apply risk adjustments to forecasts"""
        
        try:
            logger.info(f"Applying risk adjustments for {symbol}")
            
            # Calculate risk factors
            risk_factors = await self._calculate_risk_factors(
                forecasts, market_regime, symbol
            )
            
            # Calculate adjustments for each horizon
            adjusted_forecasts = {}
            risk_details = {}
            
            for horizon, forecast_data in forecasts.items():
                # Calculate horizon-specific risk adjustment
                risk_adjustment = await self._calculate_risk_adjustment(
                    forecast_data, risk_factors, horizon
                )
                
                # Apply adjustments
                adjusted_price = await self._apply_price_adjustment(
                    forecast_data['price_forecast'], risk_adjustment
                )
                
                adjusted_volatility = await self._apply_volatility_adjustment(
                    forecast_data.get('volatility_forecast', 0.2), risk_adjustment
                )
                
                adjusted_confidence = await self._apply_confidence_adjustment(
                    forecast_data.get('direction_probability', 0.5), risk_adjustment
                )
                
                # Store adjusted forecast
                adjusted_forecasts[horizon] = {
                    'original_price': forecast_data['price_forecast'],
                    'adjusted_price': adjusted_price,
                    'original_volatility': forecast_data.get('volatility_forecast', 0.2),
                    'adjusted_volatility': adjusted_volatility,
                    'original_confidence': forecast_data.get('direction_probability', 0.5),
                    'adjusted_confidence': adjusted_confidence,
                    'risk_score': risk_adjustment.risk_score,
                    'adjustment_factor': risk_adjustment.adjustment_factor
                }
                
                risk_details[horizon] = risk_adjustment
            
            # Calculate overall risk metrics
            overall_risk = await self._calculate_overall_risk_metrics(
                risk_factors, adjusted_forecasts
            )
            
            # Store adjustment history
            await self._store_adjustment_history(symbol, risk_details, overall_risk)
            
            return {
                **adjusted_forecasts,
                'overall_risk_metrics': overall_risk,
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            logger.error(f"Risk adjustment failed for {symbol}: {e}")
            return forecasts  # Return original forecasts if adjustment fails
    
    async def _calculate_risk_factors(
        self,
        forecasts: Dict[str, Any],
        market_regime: Any,
        symbol: str
    ) -> Dict[str, float]:
        """Calculate various risk factors"""
        
        try:
            risk_factors = {}
            
            # Market regime risk
            regime_risk = await self._calculate_regime_risk(market_regime)
            risk_factors['market_regime_risk'] = regime_risk
            
            # Volatility regime risk
            volatility_risk = await self._calculate_volatility_risk(forecasts, market_regime)
            risk_factors['volatility_risk'] = volatility_risk
            
            # Model uncertainty risk
            model_uncertainty = await self._calculate_model_uncertainty(forecasts, symbol)
            risk_factors['model_uncertainty'] = model_uncertainty
            
            # Liquidity risk
            liquidity_risk = await self._calculate_liquidity_risk(symbol)
            risk_factors['liquidity_risk'] = liquidity_risk
            
            # Tail risk
            tail_risk = await self._calculate_tail_risk(forecasts, market_regime)
            risk_factors['tail_risk'] = tail_risk
            
            # Correlation risk
            correlation_risk = await self._calculate_correlation_risk(symbol)
            risk_factors['correlation_risk'] = correlation_risk
            
            # Event risk
            event_risk = await self._calculate_event_risk(symbol)
            risk_factors['event_risk'] = event_risk
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Risk factor calculation failed: {e}")
            return {}
    
    async def _calculate_regime_risk(self, market_regime: Any) -> float:
        """Calculate risk based on market regime"""
        
        try:
            regime_name = getattr(market_regime, 'regime_name', 'Sideways Market')
            regime_confidence = getattr(market_regime, 'confidence', 0.5)
            transition_probability = getattr(market_regime, 'transition_probability', 0.2)
            
            # Base risk by regime type
            regime_risk_scores = {
                'Bull Market': 0.2,
                'Bear Market': 0.6,
                'Sideways Market': 0.4,
                'Crisis/High Volatility': 0.9
            }
            
            base_risk = regime_risk_scores.get(regime_name, 0.5)
            
            # Adjust for regime confidence and transition probability
            confidence_adjustment = (1 - regime_confidence) * 0.3
            transition_adjustment = transition_probability * 0.2
            
            total_regime_risk = min(base_risk + confidence_adjustment + transition_adjustment, 1.0)
            
            return float(total_regime_risk)
            
        except Exception as e:
            logger.error(f"Regime risk calculation failed: {e}")
            return 0.5
    
    async def _calculate_volatility_risk(
        self,
        forecasts: Dict[str, Any],
        market_regime: Any
    ) -> float:
        """Calculate risk based on volatility levels"""
        
        try:
            # Get volatility forecasts
            vol_forecasts = [
                data.get('volatility_forecast', 0.2) 
                for data in forecasts.values()
            ]
            
            if not vol_forecasts:
                return 0.3  # Default volatility risk
            
            # Calculate volatility risk metrics
            mean_vol = np.mean(vol_forecasts)
            vol_std = np.std(vol_forecasts)
            max_vol = max(vol_forecasts)
            
            # Risk increases with higher volatility
            vol_risk = min(mean_vol / 0.3, 1.0)  # Normalize to 30% baseline
            
            # Add risk for volatility uncertainty
            uncertainty_risk = min(vol_std / 0.1, 0.3)  # Cap at 30%
            
            # Add risk for extreme volatility
            extreme_risk = max(0, (max_vol - 0.4) / 0.2)  # Risk above 40% vol
            
            total_vol_risk = min(vol_risk + uncertainty_risk + extreme_risk, 1.0)
            
            return float(total_vol_risk)
            
        except Exception as e:
            logger.error(f"Volatility risk calculation failed: {e}")
            return 0.3
    
    async def _calculate_model_uncertainty(
        self,
        forecasts: Dict[str, Any],
        symbol: str
    ) -> float:
        """Calculate model uncertainty risk"""
        
        try:
            # Check forecast consistency across horizons
            price_forecasts = [
                data['price_forecast'] for data in forecasts.values()
            ]
            
            if len(price_forecasts) < 2:
                return 0.3  # Default uncertainty
            
            # Calculate returns implied by forecasts
            returns = []
            for i in range(1, len(price_forecasts)):
                ret = (price_forecasts[i] - price_forecasts[i-1]) / price_forecasts[i-1]
                returns.append(abs(ret))
            
            # Uncertainty based on forecast volatility
            forecast_volatility = np.std(returns) if returns else 0.1
            uncertainty = min(forecast_volatility / 0.05, 1.0)  # Normalize to 5% baseline
            
            # Add uncertainty based on model confidence
            confidence_scores = [
                data.get('model_confidence', 0.5) for data in forecasts.values()
            ]
            avg_confidence = np.mean(confidence_scores)
            confidence_uncertainty = (1 - avg_confidence) * 0.5
            
            total_uncertainty = min(uncertainty + confidence_uncertainty, 1.0)
            
            return float(total_uncertainty)
            
        except Exception as e:
            logger.error(f"Model uncertainty calculation failed: {e}")
            return 0.3
    
    async def _calculate_liquidity_risk(self, symbol: str) -> float:
        """Calculate liquidity risk for the symbol"""
        
        try:
            # In production, this would use actual volume and spread data
            # For now, use symbol-based heuristics
            
            # Major ETFs and large caps have lower liquidity risk
            low_risk_symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            
            if symbol in low_risk_symbols:
                return 0.1  # Low liquidity risk
            
            # Check if it's a major index or large cap (simplified)
            if len(symbol) <= 4 and symbol.isupper():
                return 0.2  # Medium liquidity risk
            
            # Default to higher liquidity risk for other symbols
            return 0.4
            
        except Exception as e:
            logger.error(f"Liquidity risk calculation failed: {e}")
            return 0.3
    
    async def _calculate_tail_risk(
        self,
        forecasts: Dict[str, Any],
        market_regime: Any
    ) -> float:
        """Calculate tail risk (extreme event probability)"""
        
        try:
            # Base tail risk from market regime
            regime_name = getattr(market_regime, 'regime_name', 'Sideways Market')
            
            regime_tail_risk = {
                'Bull Market': 0.1,
                'Bear Market': 0.4,
                'Sideways Market': 0.2,
                'Crisis/High Volatility': 0.7
            }.get(regime_name, 0.3)
            
            # Additional tail risk from forecast volatility
            vol_forecasts = [
                data.get('volatility_forecast', 0.2) 
                for data in forecasts.values()
            ]
            
            max_vol = max(vol_forecasts) if vol_forecasts else 0.2
            vol_tail_risk = max(0, (max_vol - 0.3) / 0.2)  # Risk above 30% vol
            
            total_tail_risk = min(regime_tail_risk + vol_tail_risk, 1.0)
            
            return float(total_tail_risk)
            
        except Exception as e:
            logger.error(f"Tail risk calculation failed: {e}")
            return 0.3
    
    async def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation/contagion risk"""
        
        try:
            # Simplified correlation risk based on asset type
            # In production, this would use actual correlation data
            
            # ETFs have higher correlation risk
            if symbol in ['SPY', 'QQQ', 'IWM']:
                return 0.6
            
            # Tech stocks have medium-high correlation
            tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA']
            if symbol in tech_symbols:
                return 0.5
            
            # Other symbols have medium correlation risk
            return 0.3
            
        except Exception as e:
            logger.error(f"Correlation risk calculation failed: {e}")
            return 0.3
    
    async def _calculate_event_risk(self, symbol: str) -> float:
        """Calculate event risk (earnings, announcements, etc.)"""
        
        try:
            # In production, this would check earnings calendar and news
            # For now, use simplified heuristics
            
            # All stocks have some event risk
            base_event_risk = 0.2
            
            # Individual stocks have higher event risk than ETFs
            if symbol in ['SPY', 'QQQ', 'IWM']:
                return 0.1  # ETFs have lower event risk
            
            return base_event_risk
            
        except Exception as e:
            logger.error(f"Event risk calculation failed: {e}")
            return 0.2
    
    async def _calculate_risk_adjustment(
        self,
        forecast_data: Dict[str, Any],
        risk_factors: Dict[str, float],
        horizon: str
    ) -> RiskAdjustment:
        """Calculate overall risk adjustment for a forecast"""
        
        try:
            # Weighted risk score
            weighted_risk = sum(
                risk_factors.get(factor, 0.3) * weight
                for factor, weight in self.risk_weights.items()
                if factor in risk_factors
            )
            
            # Time decay adjustment (longer horizons = higher risk)
            horizon_days = int(horizon.replace('d', ''))
            time_adjustment = min(horizon_days / 252, 0.5)  # Max 50% adjustment for 1 year
            
            # Total risk score
            total_risk = min(weighted_risk + time_adjustment, 1.0)
            
            # Convert risk score to adjustment factor
            adjustment_factor = total_risk * self.max_adjustment
            
            # Confidence penalty based on model uncertainty
            model_confidence = forecast_data.get('model_confidence', 0.5)
            confidence_penalty = (1 - model_confidence) * 0.1
            
            # Volatility adjustment
            forecast_vol = forecast_data.get('volatility_forecast', 0.2)
            volatility_adjustment = min(forecast_vol / 0.3, 1.0) * 0.05
            
            # Regime adjustment
            regime_adjustment = risk_factors.get('market_regime_risk', 0.3) * 0.03
            
            # Liquidity adjustment
            liquidity_adjustment = risk_factors.get('liquidity_risk', 0.3) * 0.02
            
            return RiskAdjustment(
                adjustment_factor=adjustment_factor,
                risk_score=total_risk,
                confidence_penalty=confidence_penalty,
                volatility_adjustment=volatility_adjustment,
                regime_adjustment=regime_adjustment,
                liquidity_adjustment=liquidity_adjustment
            )
            
        except Exception as e:
            logger.error(f"Risk adjustment calculation failed: {e}")
            return RiskAdjustment(0.05, 0.3, 0.02, 0.01, 0.01, 0.01)
    
    async def _apply_price_adjustment(
        self,
        original_price: float,
        risk_adjustment: RiskAdjustment
    ) -> float:
        """Apply risk adjustment to price forecast"""
        
        try:
            # Conservative adjustment - reduce extreme forecasts
            total_adjustment = (
                risk_adjustment.adjustment_factor +
                risk_adjustment.confidence_penalty +
                risk_adjustment.volatility_adjustment
            )
            
            # Apply adjustment towards mean reversion
            # High risk = more conservative (closer to current price)
            adjusted_price = original_price * (1 - total_adjustment)
            
            return float(adjusted_price)
            
        except Exception as e:
            logger.error(f"Price adjustment failed: {e}")
            return original_price
    
    async def _apply_volatility_adjustment(
        self,
        original_volatility: float,
        risk_adjustment: RiskAdjustment
    ) -> float:
        """Apply risk adjustment to volatility forecast"""
        
        try:
            # Increase volatility forecast in high-risk environments
            vol_increase = (
                risk_adjustment.volatility_adjustment +
                risk_adjustment.regime_adjustment
            )
            
            adjusted_volatility = original_volatility * (1 + vol_increase)
            
            return float(min(adjusted_volatility, 1.0))  # Cap at 100%
            
        except Exception as e:
            logger.error(f"Volatility adjustment failed: {e}")
            return original_volatility
    
    async def _apply_confidence_adjustment(
        self,
        original_confidence: float,
        risk_adjustment: RiskAdjustment
    ) -> float:
        """Apply risk adjustment to confidence/direction probability"""
        
        try:
            # Reduce confidence in high-risk environments
            confidence_reduction = (
                risk_adjustment.confidence_penalty +
                risk_adjustment.liquidity_adjustment
            )
            
            adjusted_confidence = original_confidence * (1 - confidence_reduction)
            
            return float(max(adjusted_confidence, 0.0))  # Floor at 0%
            
        except Exception as e:
            logger.error(f"Confidence adjustment failed: {e}")
            return original_confidence
    
    async def _calculate_overall_risk_metrics(
        self,
        risk_factors: Dict[str, float],
        adjusted_forecasts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall risk metrics for the forecast set"""
        
        try:
            # Overall risk score
            overall_risk = np.mean(list(risk_factors.values()))
            
            # Risk distribution
            risk_distribution = {
                'low_risk': sum(1 for v in risk_factors.values() if v < 0.3),
                'medium_risk': sum(1 for v in risk_factors.values() if 0.3 <= v < 0.6),
                'high_risk': sum(1 for v in risk_factors.values() if v >= 0.6)
            }
            
            # Average adjustments
            avg_adjustment = np.mean([
                data['adjustment_factor'] for data in adjusted_forecasts.values()
                if 'adjustment_factor' in data
            ])
            
            # Risk regime
            if overall_risk < 0.3:
                risk_regime = 'low'
            elif overall_risk < 0.6:
                risk_regime = 'medium'
            else:
                risk_regime = 'high'
            
            return {
                'overall_risk_score': float(overall_risk),
                'risk_regime': risk_regime,
                'risk_distribution': risk_distribution,
                'average_adjustment': float(avg_adjustment),
                'dominant_risk_factors': await self._identify_dominant_risks(risk_factors),
                'risk_recommendation': await self._generate_risk_recommendation(overall_risk)
            }
            
        except Exception as e:
            logger.error(f"Overall risk metrics calculation failed: {e}")
            return {}
    
    async def _identify_dominant_risks(self, risk_factors: Dict[str, float]) -> List[str]:
        """Identify the dominant risk factors"""
        
        try:
            # Sort risk factors by magnitude
            sorted_risks = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
            
            # Return top 3 risk factors
            return [factor for factor, score in sorted_risks[:3]]
            
        except Exception as e:
            logger.error(f"Dominant risk identification failed: {e}")
            return []
    
    async def _generate_risk_recommendation(self, overall_risk: float) -> str:
        """Generate risk-based recommendation"""
        
        try:
            if overall_risk < 0.3:
                return "Low risk environment - forecasts can be used with standard position sizing"
            elif overall_risk < 0.6:
                return "Medium risk environment - consider reduced position sizing and tighter stops"
            else:
                return "High risk environment - use minimal position sizing and focus on risk management"
                
        except Exception as e:
            logger.error(f"Risk recommendation generation failed: {e}")
            return "Risk assessment unavailable - proceed with caution"
    
    async def _store_adjustment_history(
        self,
        symbol: str,
        risk_details: Dict[str, RiskAdjustment],
        overall_risk: Dict[str, Any]
    ):
        """Store risk adjustment history for analysis"""
        
        try:
            if symbol not in self.adjustment_history:
                self.adjustment_history[symbol] = []
            
            adjustment_record = {
                'timestamp': datetime.utcnow(),
                'risk_details': {
                    horizon: {
                        'risk_score': adj.risk_score,
                        'adjustment_factor': adj.adjustment_factor
                    }
                    for horizon, adj in risk_details.items()
                },
                'overall_risk': overall_risk
            }
            
            self.adjustment_history[symbol].append(adjustment_record)
            
            # Keep only last 100 records
            if len(self.adjustment_history[symbol]) > 100:
                self.adjustment_history[symbol] = self.adjustment_history[symbol][-100:]
            
        except Exception as e:
            logger.error(f"Failed to store adjustment history for {symbol}: {e}")

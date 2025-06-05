"""
Options Greeks Integration for Price Forecasting
"""

import pandas as pd
import numpy as np
import asyncio
import asyncpg
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize_scalar
import warnings
import os
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class OptionsGreeksIntegrator:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.risk_free_rate = 0.05  # 5% default risk-free rate
        
    async def integrate_options_data(
        self,
        symbol: str,
        price_forecasts: Dict[str, Any],
        market_regime: Any
    ) -> Dict[str, Any]:
        """Integrate options data with price forecasts"""
        
        try:
            # Get current options data
            options_data = await self._get_options_data(symbol)
            
            if not options_data:
                return await self._generate_synthetic_options_impact(price_forecasts, market_regime)
            
            # Calculate implied volatility surface
            iv_surface = await self._calculate_iv_surface(options_data)
            
            # Calculate Greeks for forecasted prices
            greeks_impact = await self._calculate_greeks_impact(
                symbol, price_forecasts, iv_surface, market_regime
            )
            
            # Calculate options flow impact
            flow_impact = await self._calculate_options_flow_impact(options_data, price_forecasts)
            
            # Calculate gamma exposure
            gamma_exposure = await self._calculate_gamma_exposure(options_data, price_forecasts)
            
            # Calculate volatility impact
            vol_impact = await self._calculate_volatility_impact(iv_surface, price_forecasts)
            
            return {
                'implied_volatility_surface': iv_surface,
                'greeks_impact': greeks_impact,
                'options_flow_impact': flow_impact,
                'gamma_exposure': gamma_exposure,
                'volatility_impact': vol_impact,
                'options_sentiment': await self._calculate_options_sentiment(options_data),
                'pin_risk': await self._calculate_pin_risk(options_data, price_forecasts)
            }
            
        except Exception as e:
            logger.error(f"Options integration failed for {symbol}: {e}")
            return await self._generate_synthetic_options_impact(price_forecasts, market_regime)
    
    async def _get_options_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get current options data from database"""
        
        try:
            conn = await asyncpg.connect(self.db_url)
            
            # Get options data for the symbol
            options_data = await conn.fetch("""
                SELECT 
                    strike_price,
                    expiration_date,
                    option_type,
                    bid_price,
                    ask_price,
                    last_price,
                    volume,
                    open_interest,
                    implied_volatility,
                    delta,
                    gamma,
                    theta,
                    vega,
                    rho,
                    timestamp
                FROM options_data 
                WHERE symbol = $1 
                AND expiration_date > $2
                AND timestamp >= $3
                ORDER BY expiration_date, strike_price
            """, symbol, datetime.utcnow(), datetime.utcnow() - timedelta(hours=24))
            
            await conn.close()
            
            if options_data:
                df = pd.DataFrame(options_data)
                return df
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get options data for {symbol}: {e}")
            return None
    
    async def _calculate_iv_surface(self, options_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate implied volatility surface"""
        
        try:
            # Group by expiration and option type
            iv_surface = {}
            
            for exp_date in options_data['expiration_date'].unique():
                exp_data = options_data[options_data['expiration_date'] == exp_date]
                
                calls = exp_data[exp_data['option_type'] == 'call']
                puts = exp_data[exp_data['option_type'] == 'put']
                
                if len(calls) > 0:
                    iv_surface[f"calls_{exp_date}"] = {
                        'strikes': calls['strike_price'].tolist(),
                        'implied_vols': calls['implied_volatility'].tolist(),
                        'volumes': calls['volume'].tolist()
                    }
                
                if len(puts) > 0:
                    iv_surface[f"puts_{exp_date}"] = {
                        'strikes': puts['strike_price'].tolist(),
                        'implied_vols': puts['implied_volatility'].tolist(),
                        'volumes': puts['volume'].tolist()
                    }
            
            # Calculate surface statistics
            all_ivs = options_data['implied_volatility'].dropna()
            if len(all_ivs) > 0:
                iv_surface['statistics'] = {
                    'mean_iv': float(all_ivs.mean()),
                    'iv_skew': float(all_ivs.skew()),
                    'iv_term_structure': await self._calculate_term_structure(options_data),
                    'put_call_iv_spread': await self._calculate_put_call_iv_spread(options_data)
                }
            
            return iv_surface
            
        except Exception as e:
            logger.error(f"IV surface calculation failed: {e}")
            return {}
    
    async def _calculate_greeks_impact(
        self,
        symbol: str,
        price_forecasts: Dict[str, Any],
        iv_surface: Dict[str, Any],
        market_regime: Any
    ) -> Dict[str, Any]:
        """Calculate how Greeks will impact price movements"""
        
        try:
            greeks_impact = {}
            
            # Get current stock price (approximate from forecasts)
            current_price = 100.0  # Placeholder - would get from current market data
            
            for horizon, forecast_data in price_forecasts.items():
                forecast_price = forecast_data['price_forecast']
                price_change = (forecast_price - current_price) / current_price
                
                # Calculate delta impact
                delta_impact = await self._calculate_delta_impact(
                    price_change, iv_surface, horizon
                )
                
                # Calculate gamma impact
                gamma_impact = await self._calculate_gamma_impact(
                    price_change, iv_surface, horizon
                )
                
                # Calculate vega impact
                vega_impact = await self._calculate_vega_impact(
                    forecast_data.get('volatility_forecast', 0.2), iv_surface, horizon
                )
                
                # Calculate theta decay
                theta_impact = await self._calculate_theta_impact(iv_surface, horizon)
                
                greeks_impact[horizon] = {
                    'delta_impact': delta_impact,
                    'gamma_impact': gamma_impact,
                    'vega_impact': vega_impact,
                    'theta_impact': theta_impact,
                    'net_greeks_pressure': delta_impact + gamma_impact + vega_impact + theta_impact
                }
            
            return greeks_impact
            
        except Exception as e:
            logger.error(f"Greeks impact calculation failed: {e}")
            return {}
    
    async def _calculate_options_flow_impact(
        self,
        options_data: pd.DataFrame,
        price_forecasts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate impact of options flow on price movements"""
        
        try:
            # Calculate put-call ratio
            calls_volume = options_data[options_data['option_type'] == 'call']['volume'].sum()
            puts_volume = options_data[options_data['option_type'] == 'put']['volume'].sum()
            
            put_call_ratio = puts_volume / (calls_volume + 1e-8)
            
            # Calculate unusual options activity
            volume_mean = options_data['volume'].mean()
            unusual_volume = options_data[options_data['volume'] > volume_mean * 3]
            
            # Calculate dark pool activity indicator
            total_volume = options_data['volume'].sum()
            dark_pool_indicator = min(total_volume / 1000000, 1.0)  # Normalize
            
            # Calculate options sentiment
            sentiment_score = await self._calculate_flow_sentiment(options_data)
            
            return {
                'put_call_ratio': float(put_call_ratio),
                'unusual_activity_count': len(unusual_volume),
                'dark_pool_indicator': float(dark_pool_indicator),
                'flow_sentiment': sentiment_score,
                'net_flow_pressure': self._interpret_flow_pressure(put_call_ratio, sentiment_score)
            }
            
        except Exception as e:
            logger.error(f"Options flow impact calculation failed: {e}")
            return {}
    
    async def _calculate_gamma_exposure(
        self,
        options_data: pd.DataFrame,
        price_forecasts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate gamma exposure and its impact on price stability"""
        
        try:
            # Calculate total gamma exposure
            total_gamma_calls = options_data[options_data['option_type'] == 'call']['gamma'].sum()
            total_gamma_puts = options_data[options_data['option_type'] == 'put']['gamma'].sum()
            
            # Gamma exposure levels
            net_gamma = total_gamma_calls - total_gamma_puts
            
            # Calculate gamma levels at different price points
            current_price = 100.0  # Placeholder
            gamma_levels = {}
            
            for delta_pct in [-0.1, -0.05, 0, 0.05, 0.1]:
                test_price = current_price * (1 + delta_pct)
                gamma_at_price = await self._calculate_gamma_at_price(options_data, test_price)
                gamma_levels[f"{delta_pct*100:+.0f}%"] = gamma_at_price
            
            # Determine gamma impact on forecasts
            gamma_impact_on_volatility = abs(net_gamma) * 0.001  # Simplified calculation
            
            return {
                'total_gamma_calls': float(total_gamma_calls),
                'total_gamma_puts': float(total_gamma_puts),
                'net_gamma_exposure': float(net_gamma),
                'gamma_levels': gamma_levels,
                'gamma_impact_on_volatility': float(gamma_impact_on_volatility),
                'gamma_regime': 'high' if abs(net_gamma) > 1000 else 'normal'
            }
            
        except Exception as e:
            logger.error(f"Gamma exposure calculation failed: {e}")
            return {}
    
    async def _calculate_volatility_impact(
        self,
        iv_surface: Dict[str, Any],
        price_forecasts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate impact of implied volatility on forecasts"""
        
        try:
            if 'statistics' not in iv_surface:
                return {}
            
            mean_iv = iv_surface['statistics']['mean_iv']
            iv_skew = iv_surface['statistics']['iv_skew']
            
            vol_impact = {}
            
            for horizon, forecast_data in price_forecasts.items():
                forecast_vol = forecast_data.get('volatility_forecast', 0.2)
                
                # Calculate vol premium/discount
                vol_premium = (mean_iv - forecast_vol) / forecast_vol
                
                # Impact on price forecast
                vol_adjustment = vol_premium * 0.1  # 10% sensitivity
                
                vol_impact[horizon] = {
                    'implied_vol': float(mean_iv),
                    'forecast_vol': float(forecast_vol),
                    'vol_premium': float(vol_premium),
                    'price_adjustment': float(vol_adjustment),
                    'vol_regime': 'high' if mean_iv > 0.3 else 'normal'
                }
            
            return vol_impact
            
        except Exception as e:
            logger.error(f"Volatility impact calculation failed: {e}")
            return {}
    
    async def _calculate_options_sentiment(self, options_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate options-based sentiment indicators"""
        
        try:
            # Call vs Put volume sentiment
            call_volume = options_data[options_data['option_type'] == 'call']['volume'].sum()
            put_volume = options_data[options_data['option_type'] == 'put']['volume'].sum()
            
            volume_sentiment = (call_volume - put_volume) / (call_volume + put_volume + 1e-8)
            
            # Open interest sentiment
            call_oi = options_data[options_data['option_type'] == 'call']['open_interest'].sum()
            put_oi = options_data[options_data['option_type'] == 'put']['open_interest'].sum()
            
            oi_sentiment = (call_oi - put_oi) / (call_oi + put_oi + 1e-8)
            
            # Skew sentiment
            iv_skew = 0  # Would get from iv_surface
            skew_sentiment = -iv_skew / 2.0  # Negative skew is bullish
            
            # Combined sentiment
            combined_sentiment = (volume_sentiment + oi_sentiment + skew_sentiment) / 3
            
            return {
                'volume_sentiment': float(volume_sentiment),
                'open_interest_sentiment': float(oi_sentiment),
                'skew_sentiment': float(skew_sentiment),
                'combined_sentiment': float(combined_sentiment),
                'sentiment_strength': float(abs(combined_sentiment))
            }
            
        except Exception as e:
            logger.error(f"Options sentiment calculation failed: {e}")
            return {}
    
    async def _calculate_pin_risk(
        self,
        options_data: pd.DataFrame,
        price_forecasts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate pin risk at major strike levels"""
        
        try:
            # Find strikes with high open interest
            strike_oi = options_data.groupby('strike_price')['open_interest'].sum()
            high_oi_strikes = strike_oi[strike_oi > strike_oi.quantile(0.8)].index.tolist()
            
            pin_risk = {}
            current_price = 100.0  # Placeholder
            
            for horizon, forecast_data in price_forecasts.items():
                forecast_price = forecast_data['price_forecast']
                
                # Find nearest high OI strikes
                distances = [abs(forecast_price - strike) for strike in high_oi_strikes]
                nearest_strike = high_oi_strikes[np.argmin(distances)] if high_oi_strikes else forecast_price
                
                # Calculate pin probability
                distance_to_pin = abs(forecast_price - nearest_strike) / forecast_price
                pin_probability = max(0, 1 - distance_to_pin * 10)  # Simplified calculation
                
                pin_risk[horizon] = {
                    'nearest_pin_strike': float(nearest_strike),
                    'distance_to_pin': float(distance_to_pin),
                    'pin_probability': float(pin_probability),
                    'pin_impact': 'high' if pin_probability > 0.3 else 'low'
                }
            
            return pin_risk
            
        except Exception as e:
            logger.error(f"Pin risk calculation failed: {e}")
            return {}
    
    async def _generate_synthetic_options_impact(
        self,
        price_forecasts: Dict[str, Any],
        market_regime: Any
    ) -> Dict[str, Any]:
        """Generate synthetic options impact when real data is unavailable"""
        
        try:
            # Use market regime to estimate options impact
            regime_name = getattr(market_regime, 'regime_name', 'Sideways Market')
            volatility_level = getattr(market_regime, 'volatility_level', 'medium')
            
            # Synthetic implied volatility
            base_iv = {
                'low': 0.15,
                'medium': 0.20,
                'high': 0.30,
                'very_high': 0.45
            }.get(volatility_level, 0.20)
            
            # Synthetic put-call ratio
            put_call_ratio = {
                'Bull Market': 0.7,
                'Bear Market': 1.3,
                'Sideways Market': 1.0,
                'Crisis/High Volatility': 1.5
            }.get(regime_name, 1.0)
            
            return {
                'synthetic_data': True,
                'implied_volatility_surface': {
                    'statistics': {
                        'mean_iv': base_iv,
                        'iv_skew': -0.1 if 'Bull' in regime_name else 0.1
                    }
                },
                'options_flow_impact': {
                    'put_call_ratio': put_call_ratio,
                    'flow_sentiment': -0.2 if 'Bear' in regime_name else 0.1
                },
                'gamma_exposure': {
                    'net_gamma_exposure': 0,
                    'gamma_regime': 'normal'
                },
                'volatility_impact': {
                    horizon: {
                        'implied_vol': base_iv,
                        'vol_premium': 0.0,
                        'price_adjustment': 0.0
                    } for horizon in price_forecasts.keys()
                }
            }
            
        except Exception as e:
            logger.error(f"Synthetic options impact generation failed: {e}")
            return {}
    
    # Helper methods
    async def _calculate_term_structure(self, options_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate implied volatility term structure"""
        
        term_structure = {}
        current_date = datetime.utcnow()
        
        for exp_date in options_data['expiration_date'].unique():
            exp_data = options_data[options_data['expiration_date'] == exp_date]
            days_to_exp = (exp_date - current_date).days
            mean_iv = exp_data['implied_volatility'].mean()
            
            if days_to_exp > 0:
                term_structure[f"{days_to_exp}d"] = float(mean_iv)
        
        return term_structure
    
    async def _calculate_put_call_iv_spread(self, options_data: pd.DataFrame) -> float:
        """Calculate spread between put and call implied volatilities"""
        
        calls = options_data[options_data['option_type'] == 'call']
        puts = options_data[options_data['option_type'] == 'put']
        
        call_iv = calls['implied_volatility'].mean() if len(calls) > 0 else 0.2
        put_iv = puts['implied_volatility'].mean() if len(puts) > 0 else 0.2
        
        return float(put_iv - call_iv)
    
    async def _calculate_delta_impact(self, price_change: float, iv_surface: Dict, horizon: str) -> float:
        """Calculate delta impact on price"""
        return price_change * 0.5  # Simplified
    
    async def _calculate_gamma_impact(self, price_change: float, iv_surface: Dict, horizon: str) -> float:
        """Calculate gamma impact on price"""
        return (price_change ** 2) * 0.1  # Simplified
    
    async def _calculate_vega_impact(self, vol_forecast: float, iv_surface: Dict, horizon: str) -> float:
        """Calculate vega impact on price"""
        return (vol_forecast - 0.2) * 0.1  # Simplified
    
    async def _calculate_theta_impact(self, iv_surface: Dict, horizon: str) -> float:
        """Calculate theta decay impact"""
        return -0.01  # Simplified negative theta
    
    async def _calculate_flow_sentiment(self, options_data: pd.DataFrame) -> float:
        """Calculate sentiment from options flow"""
        # Simplified sentiment calculation
        call_volume = options_data[options_data['option_type'] == 'call']['volume'].sum()
        put_volume = options_data[options_data['option_type'] == 'put']['volume'].sum()
        
        return float((call_volume - put_volume) / (call_volume + put_volume + 1e-8))
    
    def _interpret_flow_pressure(self, put_call_ratio: float, sentiment_score: float) -> str:
        """Interpret overall flow pressure"""
        if put_call_ratio > 1.2 and sentiment_score < -0.1:
            return "bearish"
        elif put_call_ratio < 0.8 and sentiment_score > 0.1:
            return "bullish"
        else:
            return "neutral"
    
    async def _calculate_gamma_at_price(self, options_data: pd.DataFrame, price: float) -> float:
        """Calculate gamma exposure at a specific price level"""
        # Simplified gamma calculation
        relevant_options = options_data[
            (options_data['strike_price'] >= price * 0.9) &
            (options_data['strike_price'] <= price * 1.1)
        ]
        
        return float(relevant_options['gamma'].sum())

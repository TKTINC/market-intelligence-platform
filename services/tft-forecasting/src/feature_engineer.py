"""
Multi-Scale Feature Engineering for TFT Forecasting
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import ta
from sklearn.decomposition import PCA
from scipy import stats

logger = logging.getLogger(__name__)

class MultiScaleFeatureEngineer:
    def __init__(self):
        self.feature_cache = {}
        self.pca_models = {}  # For dimensionality reduction
        
    async def engineer_features(
        self,
        historical_data: pd.DataFrame,
        symbol: str,
        market_context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Engineer comprehensive feature set for TFT model"""
        
        try:
            logger.info(f"Engineering features for {symbol}")
            
            # Start with preprocessed data
            features = historical_data.copy()
            
            # Add multi-timeframe technical features
            features = await self._add_multi_timeframe_features(features)
            
            # Add market microstructure features
            features = await self._add_microstructure_features(features)
            
            # Add cross-asset features
            features = await self._add_cross_asset_features(features, symbol, market_context)
            
            # Add macro economic features
            features = await self._add_macro_features(features, market_context)
            
            # Add options-specific features
            features = await self._add_options_features(features, symbol)
            
            # Add time-based features
            features = await self._add_temporal_features(features)
            
            # Add interaction features
            features = await self._add_interaction_features(features)
            
            # Feature selection and dimensionality reduction
            features = await self._feature_selection(features, symbol)
            
            # Clean and finalize
            features = await self._finalize_features(features)
            
            logger.info(f"Feature engineering completed for {symbol}. Shape: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Feature engineering failed for {symbol}: {e}")
            raise
    
    async def _add_multi_timeframe_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add features across multiple timeframes"""
        
        try:
            # Price momentum across different horizons
            for window in [3, 5, 10, 20, 50]:
                data[f'momentum_{window}d'] = data['close_price'].pct_change(window)
                data[f'momentum_rank_{window}d'] = data[f'momentum_{window}d'].rolling(252).rank(pct=True)
            
            # Multi-timeframe moving averages
            for window in [5, 10, 20, 50, 200]:
                data[f'sma_{window}'] = data['close_price'].rolling(window).mean()
                data[f'price_vs_sma_{window}'] = (data['close_price'] / data[f'sma_{window}']) - 1
                
                # Slope of moving average
                data[f'sma_{window}_slope'] = data[f'sma_{window}'].diff(5) / data[f'sma_{window}'].shift(5)
            
            # Exponential moving averages with different decay rates
            for alpha in [0.1, 0.2, 0.3, 0.5]:
                span = int(2/alpha - 1)
                data[f'ema_alpha_{alpha}'] = data['close_price'].ewm(alpha=alpha).mean()
                data[f'price_vs_ema_{alpha}'] = (data['close_price'] / data[f'ema_alpha_{alpha}']) - 1
            
            # Multi-timeframe RSI
            for window in [7, 14, 21, 28]:
                delta = data['close_price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                data[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            # Williams %R across different periods
            for window in [14, 21, 28]:
                highest_high = data['high_price'].rolling(window).max()
                lowest_low = data['low_price'].rolling(window).min()
                data[f'williams_r_{window}'] = -100 * (highest_high - data['close_price']) / (highest_high - lowest_low)
            
            return data
            
        except Exception as e:
            logger.error(f"Multi-timeframe feature engineering failed: {e}")
            raise
    
    async def _add_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        
        try:
            # Bid-ask spread proxies
            data['hl_spread'] = (data['high_price'] - data['low_price']) / data['close_price']
            data['hl_spread_ma'] = data['hl_spread'].rolling(20).mean()
            data['hl_spread_std'] = data['hl_spread'].rolling(20).std()
            
            # Volume-price analysis
            data['volume_weighted_price'] = (data['volume'] * data['close_price']).rolling(20).sum() / data['volume'].rolling(20).sum()
            data['volume_profile'] = data['volume'] / data['volume'].rolling(20).mean()
            
            # Tick analysis (using price changes as proxy for ticks)
            data['upticks'] = (data['close_price'] > data['close_price'].shift(1)).astype(int)
            data['downticks'] = (data['close_price'] < data['close_price'].shift(1)).astype(int)
            data['tick_imbalance'] = (data['upticks'] - data['downticks']).rolling(20).mean()
            
            # Order flow imbalance proxies
            data['buying_pressure'] = data['volume'] * (2 * (data['close_price'] - data['low_price']) / (data['high_price'] - data['low_price']) - 1)
            data['selling_pressure'] = data['volume'] * (2 * (data['high_price'] - data['close_price']) / (data['high_price'] - data['low_price']) - 1)
            data['order_imbalance'] = (data['buying_pressure'] - data['selling_pressure']).rolling(10).mean()
            
            # Amihud illiquidity measure
            data['amihud_illiquidity'] = abs(data['close_price'].pct_change()) / (data['volume'] * data['close_price'])
            data['amihud_illiquidity'] = data['amihud_illiquidity'].rolling(20).mean()
            
            # Price impact measures
            data['price_impact'] = abs(data['close_price'].pct_change()) / np.log(data['volume'] + 1)
            
            return data
            
        except Exception as e:
            logger.error(f"Microstructure feature engineering failed: {e}")
            raise
    
    async def _add_cross_asset_features(
        self,
        data: pd.DataFrame,
        symbol: str,
        market_context: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Add cross-asset and market-relative features"""
        
        try:
            # Market benchmark features (using SPY as proxy)
            # In production, fetch actual market data
            
            # VIX features (implied volatility)
            if market_context and 'vix' in market_context:
                vix_level = market_context['vix']
                data['vix_level'] = vix_level
                data['vix_regime'] = 1 if vix_level > 20 else 0
                data['vix_spike'] = 1 if vix_level > 30 else 0
            
            # Sector rotation features
            if market_context and 'sector_rotation' in market_context:
                sector_data = market_context['sector_rotation']
                # Add sector strength indicators
                for sector, strength in sector_data.items():
                    data[f'sector_{sector}_strength'] = strength
            
            # Currency features (for international exposure)
            # Dollar index proxy
            data['dollar_strength'] = np.sin(2 * np.pi * np.arange(len(data)) / 252)  # Placeholder
            
            # Interest rate features
            # 10-year yield proxy
            data['yield_curve_slope'] = np.random.normal(0, 0.1, len(data))  # Placeholder
            
            # Commodity features
            # Oil prices, gold prices as market risk indicators
            data['oil_proxy'] = np.sin(2 * np.pi * np.arange(len(data)) / 100)  # Placeholder
            data['gold_proxy'] = np.cos(2 * np.pi * np.arange(len(data)) / 150)  # Placeholder
            
            return data
            
        except Exception as e:
            logger.error(f"Cross-asset feature engineering failed: {e}")
            raise
    
    async def _add_macro_features(
        self,
        data: pd.DataFrame,
        market_context: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Add macroeconomic features"""
        
        try:
            # Economic calendar features
            if market_context and 'economic_calendar' in market_context:
                calendar_data = market_context['economic_calendar']
                # Add time-to-event features for major announcements
                for event in calendar_data:
                    data[f'days_to_{event}'] = 0  # Placeholder
            
            # Seasonality features
            data['month'] = data.index.month
            data['quarter'] = data.index.quarter
            data['day_of_week'] = data.index.dayofweek
            data['day_of_month'] = data.index.day
            data['week_of_year'] = data.index.isocalendar().week
            
            # Monthly/quarterly effects
            data['january_effect'] = (data['month'] == 1).astype(int)
            data['december_effect'] = (data['month'] == 12).astype(int)
            data['quarter_end'] = data.index.is_quarter_end.astype(int)
            data['month_end'] = data.index.is_month_end.astype(int)
            
            # Holiday effects
            data['holiday_proximity'] = 0  # Placeholder for holiday calendar
            
            # Earnings season effects
            data['earnings_season'] = ((data['month'] % 3 == 1) & (data.index.day <= 31)).astype(int)
            
            return data
            
        except Exception as e:
            logger.error(f"Macro feature engineering failed: {e}")
            raise
    
    async def _add_options_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add options-specific features"""
        
        try:
            # Implied volatility features
            # In production, fetch actual options data
            
            # Put-call ratio
            data['put_call_ratio'] = np.random.uniform(0.5, 1.5, len(data))  # Placeholder
            
            # Options volume
            data['options_volume'] = np.random.uniform(0.1, 2.0, len(data))  # Placeholder
            
            # Skew indicators
            data['volatility_skew'] = np.random.normal(0, 0.1, len(data))  # Placeholder
            
            # Term structure indicators
            data['vol_term_structure'] = np.random.normal(0, 0.05, len(data))  # Placeholder
            
            # Gamma exposure
            data['gamma_exposure'] = np.random.normal(0, 0.1, len(data))  # Placeholder
            
            # Dark pool activity proxy
            data['dark_pool_ratio'] = np.random.uniform(0.1, 0.4, len(data))  # Placeholder
            
            return data
            
        except Exception as e:
            logger.error(f"Options feature engineering failed: {e}")
            raise
    
    async def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based cyclical features"""
        
        try:
            # Cyclical encoding of time features
            data['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
            data['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
            
            data['day_of_week_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
            data['day_of_week_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
            
            data['day_of_month_sin'] = np.sin(2 * np.pi * data.index.day / 31)
            data['day_of_month_cos'] = np.cos(2 * np.pi * data.index.day / 31)
            
            # Annual cycle (captures yearly seasonality)
            day_of_year = data.index.dayofyear
            data['year_progress_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
            data['year_progress_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
            
            # Business cycle features
            data['business_days_from_month_start'] = data.index.to_series().groupby(
                [data.index.year, data.index.month]
            ).cumcount() + 1
            
            # Time since major market events (approximate)
            reference_date = pd.Timestamp('2020-03-23')  # COVID low
            data['days_since_covid_low'] = (data.index - reference_date).days
            data['years_since_covid_low'] = data['days_since_covid_low'] / 365.25
            
            return data
            
        except Exception as e:
            logger.error(f"Temporal feature engineering failed: {e}")
            raise
    
    async def _add_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add interaction and polynomial features"""
        
        try:
            # Volume-price interactions
            data['volume_price_interaction'] = data['volume'] * data['close_price'].pct_change()
            data['volume_volatility_interaction'] = data['volume'] * data['realized_vol_21d']
            
            # Technical indicator interactions
            data['rsi_bb_interaction'] = data['rsi'] * data['bb_position']
            data['macd_momentum_interaction'] = data['macd'] * data['momentum_10d']
            
            # Volatility interactions
            data['vol_regime_momentum'] = data['vol_regime'] * data['momentum_21d']
            data['vol_skew_interaction'] = data['realized_vol_21d'] * data.get('volatility_skew', 0)
            
            # Polynomial features for key indicators
            data['rsi_squared'] = data['rsi'] ** 2
            data['momentum_squared'] = data['momentum_10d'] ** 2
            data['volume_ratio_squared'] = data['volume_ratio'] ** 2
            
            # Ratios and relative features
            data['momentum_vol_ratio'] = data['momentum_10d'] / (data['realized_vol_21d'] + 1e-8)
            data['price_volume_efficiency'] = data['close_price'].pct_change() / (np.log(data['volume'] + 1) + 1e-8)
            
            return data
            
        except Exception as e:
            logger.error(f"Interaction feature engineering failed: {e}")
            raise
    
    async def _feature_selection(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Select most relevant features and reduce dimensionality"""
        
        try:
            # Remove features with too many missing values
            missing_threshold = 0.1
            missing_pct = data.isnull().sum() / len(data)
            valid_features = missing_pct[missing_pct <= missing_threshold].index
            data = data[valid_features]
            
            # Remove features with zero variance
            numeric_data = data.select_dtypes(include=[np.number])
            zero_var_features = numeric_data.columns[numeric_data.var() == 0]
            data = data.drop(columns=zero_var_features)
            
            # Remove highly correlated features
            corr_matrix = data.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            high_corr_features = [
                column for column in upper_triangle.columns 
                if any(upper_triangle[column] > 0.95)
            ]
            data = data.drop(columns=high_corr_features)
            
            # Apply PCA for dimensionality reduction if needed
            if len(data.columns) > 100:
                data = await self._apply_pca(data, symbol, n_components=80)
            
            return data
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return data
    
    async def _apply_pca(self, data: pd.DataFrame, symbol: str, n_components: int) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction"""
        
        try:
            if symbol not in self.pca_models:
                self.pca_models[symbol] = PCA(n_components=n_components)
                transformed_data = self.pca_models[symbol].fit_transform(data)
            else:
                transformed_data = self.pca_models[symbol].transform(data)
            
            # Create new DataFrame with PCA features
            pca_columns = [f'pca_{i}' for i in range(n_components)]
            pca_df = pd.DataFrame(
                transformed_data,
                index=data.index,
                columns=pca_columns
            )
            
            return pca_df
            
        except Exception as e:
            logger.error(f"PCA application failed: {e}")
            return data
    
    async def _finalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Final cleaning and preparation of features"""
        
        try:
            # Fill any remaining missing values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Replace infinite values
            data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Ensure all features are numeric
            data = data.select_dtypes(include=[np.number])
            
            # Remove any remaining NaN values
            data = data.dropna()
            
            return data
            
        except Exception as e:
            logger.error(f"Feature finalization failed: {e}")
            raise

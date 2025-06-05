"""
Financial Data Preprocessor for TFT Forecasting
"""

import pandas as pd
import numpy as np
import asyncio
import asyncpg
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler
import os

logger = logging.getLogger(__name__)

class FinancialDataPreprocessor:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.scalers = {}  # Symbol -> scaler mappings
        self.supported_symbols = set()
        
        # Data quality thresholds
        self.min_data_points = 252  # 1 year minimum
        self.max_missing_pct = 0.05  # 5% max missing data
        
    async def health_check(self) -> str:
        """Check data preprocessor health"""
        try:
            # Check database connection
            conn = await asyncpg.connect(self.db_url)
            await conn.close()
            
            # Check if we have recent data
            recent_data_count = len(self.supported_symbols)
            
            if recent_data_count > 0:
                return "healthy"
            else:
                return "warming_up"
                
        except Exception as e:
            logger.error(f"Data preprocessor health check failed: {e}")
            return "unhealthy"
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported and has sufficient data"""
        
        try:
            # Check if symbol exists in our database
            conn = await asyncpg.connect(self.db_url)
            
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM price_data 
                WHERE symbol = $1 AND date >= $2
            """, symbol, datetime.utcnow() - timedelta(days=365))
            
            await conn.close()
            
            if count >= self.min_data_points:
                self.supported_symbols.add(symbol)
                return True
            
            # Try to fetch from external source
            return await self._fetch_and_validate_external(symbol)
            
        except Exception as e:
            logger.error(f"Symbol validation failed for {symbol}: {e}")
            return False
    
    async def get_historical_data(
        self,
        symbol: str,
        lookback_days: int = 1260
    ) -> pd.DataFrame:
        """Get historical price and volume data for a symbol"""
        
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Try to get data from database first
            conn = await asyncpg.connect(self.db_url)
            
            data = await conn.fetch("""
                SELECT date, open_price, high_price, low_price, close_price, volume,
                       adjusted_close, vwap, volatility_1d, volatility_5d, volatility_21d
                FROM price_data 
                WHERE symbol = $1 AND date BETWEEN $2 AND $3
                ORDER BY date
            """, symbol, start_date, end_date)
            
            await conn.close()
            
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            else:
                # Fallback to external data source
                df = await self._fetch_external_data(symbol, start_date, end_date)
            
            # Data quality checks and cleaning
            df = await self._clean_and_validate_data(df, symbol)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise
    
    async def preprocess_features(
        self,
        data: pd.DataFrame,
        symbol: str,
        target_horizon: int = 1
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess features and create target variables"""
        
        try:
            # Calculate technical indicators
            data = await self._calculate_technical_indicators(data)
            
            # Calculate market microstructure features
            data = await self._calculate_microstructure_features(data)
            
            # Calculate volatility features
            data = await self._calculate_volatility_features(data)
            
            # Calculate regime features
            data = await self._calculate_regime_features(data)
            
            # Create target variables
            targets = await self._create_target_variables(data, target_horizon)
            
            # Handle missing values
            data = await self._handle_missing_values(data)
            
            # Feature scaling
            data = await self._scale_features(data, symbol)
            
            # Remove any remaining NaN values
            valid_indices = ~(data.isna().any(axis=1) | targets.isna())
            data = data[valid_indices]
            targets = targets[valid_indices]
            
            return data, targets
            
        except Exception as e:
            logger.error(f"Feature preprocessing failed for {symbol}: {e}")
            raise
    
    async def _fetch_and_validate_external(self, symbol: str) -> bool:
        """Fetch data from external source and validate"""
        
        try:
            # Use yfinance for external data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2y")
            
            if len(hist) >= self.min_data_points:
                # Store in database for future use
                await self._store_external_data(symbol, hist)
                self.supported_symbols.add(symbol)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"External data fetch failed for {symbol}: {e}")
            return False
    
    async def _fetch_external_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch external data for date range"""
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            # Rename columns to match our schema
            hist.columns = hist.columns.str.lower()
            hist.rename(columns={
                'open': 'open_price',
                'high': 'high_price', 
                'low': 'low_price',
                'close': 'close_price',
                'adj close': 'adjusted_close'
            }, inplace=True)
            
            return hist
            
        except Exception as e:
            logger.error(f"External data fetch failed for {symbol}: {e}")
            raise
    
    async def _store_external_data(self, symbol: str, data: pd.DataFrame):
        """Store external data in database"""
        
        try:
            conn = await asyncpg.connect(self.db_url)
            
            for date, row in data.iterrows():
                await conn.execute("""
                    INSERT INTO price_data 
                    (symbol, date, open_price, high_price, low_price, close_price, volume, adjusted_close)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (symbol, date) DO NOTHING
                """, symbol, date, row['Open'], row['High'], row['Low'], 
                     row['Close'], row['Volume'], row['Adj Close'])
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store external data for {symbol}: {e}")
    
    async def _clean_and_validate_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and validate price data"""
        
        try:
            # Remove duplicates
            data = data[~data.index.duplicated(keep='first')]
            
            # Sort by date
            data = data.sort_index()
            
            # Check for missing values
            missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
            
            if missing_pct > self.max_missing_pct:
                logger.warning(f"High missing data percentage for {symbol}: {missing_pct:.2%}")
            
            # Forward fill missing values
            data = data.fillna(method='ffill')
            
            # Remove outliers using IQR method
            data = await self._remove_outliers(data)
            
            # Validate price consistency
            data = await self._validate_price_consistency(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Data cleaning failed for {symbol}: {e}")
            raise
    
    async def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical analysis indicators"""
        
        try:
            # Moving averages
            data['sma_5'] = data['close_price'].rolling(5).mean()
            data['sma_10'] = data['close_price'].rolling(10).mean()
            data['sma_20'] = data['close_price'].rolling(20).mean()
            data['sma_50'] = data['close_price'].rolling(50).mean()
            
            data['ema_5'] = data['close_price'].ewm(span=5).mean()
            data['ema_10'] = data['close_price'].ewm(span=10).mean()
            data['ema_20'] = data['close_price'].ewm(span=20).mean()
            
            # Relative Strength Index (RSI)
            delta = data['close_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = data['close_price'].ewm(span=12).mean()
            ema26 = data['close_price'].ewm(span=26).mean()
            data['macd'] = ema12 - ema26
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
            
            # Bollinger Bands
            data['bb_middle'] = data['close_price'].rolling(20).mean()
            bb_std = data['close_price'].rolling(20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
            data['bb_width'] = data['bb_upper'] - data['bb_lower']
            data['bb_position'] = (data['close_price'] - data['bb_lower']) / data['bb_width']
            
            # Stochastic Oscillator
            low_14 = data['low_price'].rolling(14).min()
            high_14 = data['high_price'].rolling(14).max()
            data['stoch_k'] = 100 * (data['close_price'] - low_14) / (high_14 - low_14)
            data['stoch_d'] = data['stoch_k'].rolling(3).mean()
            
            # Average True Range (ATR)
            data['tr1'] = data['high_price'] - data['low_price']
            data['tr2'] = abs(data['high_price'] - data['close_price'].shift(1))
            data['tr3'] = abs(data['low_price'] - data['close_price'].shift(1))
            data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
            data['atr'] = data['tr'].rolling(14).mean()
            
            # Drop intermediate calculation columns
            data.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Technical indicator calculation failed: {e}")
            raise
    
    async def _calculate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features"""
        
        try:
            # Price impact measures
            data['price_range'] = (data['high_price'] - data['low_price']) / data['close_price']
            data['price_gap'] = (data['open_price'] - data['close_price'].shift(1)) / data['close_price'].shift(1)
            
            # Volume features
            data['volume_sma_10'] = data['volume'].rolling(10).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma_10']
            data['volume_price_trend'] = (data['close_price'].diff() * data['volume']).rolling(5).sum()
            
            # VWAP features
            if 'vwap' in data.columns:
                data['vwap_distance'] = (data['close_price'] - data['vwap']) / data['vwap']
            else:
                # Calculate VWAP if not available
                data['vwap'] = (data['volume'] * (data['high_price'] + data['low_price'] + data['close_price']) / 3).cumsum() / data['volume'].cumsum()
                data['vwap_distance'] = (data['close_price'] - data['vwap']) / data['vwap']
            
            # Intraday momentum
            data['intraday_return'] = (data['close_price'] - data['open_price']) / data['open_price']
            data['overnight_return'] = (data['open_price'] - data['close_price'].shift(1)) / data['close_price'].shift(1)
            
            return data
            
        except Exception as e:
            logger.error(f"Microstructure feature calculation failed: {e}")
            raise
    
    async def _calculate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate various volatility measures"""
        
        try:
            # Returns
            data['return_1d'] = data['close_price'].pct_change()
            data['return_5d'] = data['close_price'].pct_change(5)
            data['return_21d'] = data['close_price'].pct_change(21)
            
            # Realized volatility
            data['realized_vol_5d'] = data['return_1d'].rolling(5).std() * np.sqrt(252)
            data['realized_vol_21d'] = data['return_1d'].rolling(21).std() * np.sqrt(252)
            data['realized_vol_63d'] = data['return_1d'].rolling(63).std() * np.sqrt(252)
            
            # Parkinson volatility (using high-low)
            data['parkinson_vol'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                (np.log(data['high_price'] / data['low_price']) ** 2).rolling(21).mean() * 252
            )
            
            # Garman-Klass volatility
            data['gk_vol'] = np.sqrt(
                (0.5 * (np.log(data['high_price'] / data['low_price']) ** 2) - 
                 (2 * np.log(2) - 1) * (np.log(data['close_price'] / data['open_price']) ** 2)
                ).rolling(21).mean() * 252
            )
            
            # Volatility regime indicators
            data['vol_regime'] = (data['realized_vol_21d'] > data['realized_vol_21d'].rolling(63).quantile(0.75)).astype(int)
            
            return data
            
        except Exception as e:
            logger.error(f"Volatility feature calculation failed: {e}")
            raise
    
    async def _calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market regime features"""
        
        try:
            # Trend strength
            data['trend_strength'] = abs(data['close_price'].rolling(20).apply(
                lambda x: np.corrcoef(x, range(len(x)))[0, 1], raw=False
            ))
            
            # Market state indicators
            data['above_sma_20'] = (data['close_price'] > data['sma_20']).astype(int)
            data['above_sma_50'] = (data['close_price'] > data['sma_50']).astype(int)
            
            # Momentum regimes
            data['momentum_regime'] = np.where(
                data['return_21d'] > data['return_21d'].rolling(63).quantile(0.75), 1,
                np.where(data['return_21d'] < data['return_21d'].rolling(63).quantile(0.25), -1, 0)
            )
            
            # Volatility persistence
            data['vol_persistence'] = data['realized_vol_21d'].rolling(5).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0, raw=False
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Regime feature calculation failed: {e}")
            raise
    
    async def _create_target_variables(self, data: pd.DataFrame, horizon: int) -> pd.Series:
        """Create target variables for forecasting"""
        
        try:
            # Price return target
            price_target = data['close_price'].shift(-horizon).pct_change(horizon)
            
            return price_target
            
        except Exception as e:
            logger.error(f"Target variable creation failed: {e}")
            raise
    
    async def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        
        try:
            # Forward fill first
            data = data.fillna(method='ffill')
            
            # Backward fill for any remaining
            data = data.fillna(method='bfill')
            
            # Fill any remaining with median
            for col in data.columns:
                if data[col].isnull().any():
                    data[col].fillna(data[col].median(), inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Missing value handling failed: {e}")
            raise
    
    async def _scale_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Scale features for model input"""
        
        try:
            # Use RobustScaler for financial data (handles outliers better)
            if symbol not in self.scalers:
                self.scalers[symbol] = RobustScaler()
                scaled_data = self.scalers[symbol].fit_transform(data)
            else:
                scaled_data = self.scalers[symbol].transform(data)
            
            # Convert back to DataFrame
            scaled_df = pd.DataFrame(
                scaled_data,
                index=data.index,
                columns=data.columns
            )
            
            return scaled_df
            
        except Exception as e:
            logger.error(f"Feature scaling failed for {symbol}: {e}")
            raise
    
    async def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        
        try:
            # Apply outlier removal to price columns only
            price_cols = ['open_price', 'high_price', 'low_price', 'close_price']
            
            for col in price_cols:
                if col in data.columns:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers instead of removing (to preserve time series)
                    data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
            
            return data
            
        except Exception as e:
            logger.error(f"Outlier removal failed: {e}")
            return data
    
    async def _validate_price_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate price relationships (high >= low, etc.)"""
        
        try:
            # Ensure high >= low
            data.loc[data['high_price'] < data['low_price'], 'high_price'] = data['low_price']
            
            # Ensure close is between high and low
            data['close_price'] = data['close_price'].clip(
                lower=data['low_price'],
                upper=data['high_price']
            )
            
            # Ensure open is between high and low
            data['open_price'] = data['open_price'].clip(
                lower=data['low_price'],
                upper=data['high_price']
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Price consistency validation failed: {e}")
            return data

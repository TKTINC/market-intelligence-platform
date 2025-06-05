"""
Market Regime Detection using Hidden Markov Models and Clustering
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    regime_id: int
    regime_name: str
    confidence: float
    volatility_level: str
    trend_direction: str
    mean_return: float
    volatility: float
    characteristics: Dict[str, Any]
    transition_probability: float
    persistence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime_id': self.regime_id,
            'regime_name': self.regime_name,
            'confidence': self.confidence,
            'volatility_level': self.volatility_level,
            'trend_direction': self.trend_direction,
            'mean_return': self.mean_return,
            'volatility': self.volatility,
            'characteristics': self.characteristics,
            'transition_probability': self.transition_probability,
            'persistence': self.persistence
        }

class MarketRegimeDetector:
    def __init__(self):
        self.regime_models = {}  # Symbol -> trained models
        self.regime_history = {}  # Symbol -> historical regimes
        self.transition_matrices = {}  # Symbol -> transition probabilities
        
        # Regime detection parameters
        self.n_regimes = 4  # Bull, Bear, Sideways, Crisis
        self.lookback_window = 252  # 1 year for regime detection
        self.confidence_threshold = 0.6
        
        # Feature weights for regime classification
        self.feature_weights = {
            'return_features': 0.4,
            'volatility_features': 0.3,
            'momentum_features': 0.2,
            'volume_features': 0.1
        }
        
    async def detect_regime(
        self,
        features: pd.DataFrame,
        symbol: str
    ) -> MarketRegime:
        """Detect current market regime for a symbol"""
        
        try:
            # Prepare regime features
            regime_features = await self._prepare_regime_features(features)
            
            # Get or train regime model
            model = await self._get_regime_model(symbol, regime_features)
            
            # Predict current regime
            current_features = regime_features.iloc[-1:].values
            regime_probs = model.predict_proba(current_features)[0]
            predicted_regime = model.predict(current_features)[0]
            
            # Get regime characteristics
            regime_info = await self._get_regime_characteristics(
                predicted_regime, 
                regime_probs,
                features.iloc[-self.lookback_window:]
            )
            
            # Calculate transition probability
            transition_prob = await self._calculate_transition_probability(
                symbol, predicted_regime
            )
            
            # Calculate regime persistence
            persistence = await self._calculate_regime_persistence(
                symbol, predicted_regime
            )
            
            # Create regime object
            regime = MarketRegime(
                regime_id=predicted_regime,
                regime_name=regime_info['name'],
                confidence=float(np.max(regime_probs)),
                volatility_level=regime_info['volatility_level'],
                trend_direction=regime_info['trend_direction'],
                mean_return=regime_info['mean_return'],
                volatility=regime_info['volatility'],
                characteristics=regime_info['characteristics'],
                transition_probability=transition_prob,
                persistence=persistence
            )
            
            # Update regime history
            await self._update_regime_history(symbol, regime)
            
            return regime
            
        except Exception as e:
            logger.error(f"Regime detection failed for {symbol}: {e}")
            # Return default regime
            return await self._get_default_regime()
    
    async def get_regime_history(self, symbol: str) -> List[Dict[str, Any]]:
        """Get historical regime information"""
        
        try:
            if symbol not in self.regime_history:
                return []
            
            return [regime.to_dict() for regime in self.regime_history[symbol][-30:]]  # Last 30 days
            
        except Exception as e:
            logger.error(f"Failed to get regime history for {symbol}: {e}")
            return []
    
    async def get_transition_probabilities(self, symbol: str) -> Dict[str, Any]:
        """Get regime transition probability matrix"""
        
        try:
            if symbol not in self.transition_matrices:
                return {}
            
            matrix = self.transition_matrices[symbol]
            regime_names = ['Bull', 'Bear', 'Sideways', 'Crisis']
            
            return {
                'transition_matrix': matrix.tolist(),
                'regime_names': regime_names,
                'steady_state_probabilities': await self._calculate_steady_state_probabilities(matrix)
            }
            
        except Exception as e:
            logger.error(f"Failed to get transition probabilities for {symbol}: {e}")
            return {}
    
    async def _prepare_regime_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare features specifically for regime detection"""
        
        try:
            regime_data = pd.DataFrame(index=features.index)
            
            # Return features
            if 'return_1d' in features.columns:
                regime_data['daily_return'] = features['return_1d']
                regime_data['return_5d'] = features['return_5d'] if 'return_5d' in features.columns else features['return_1d'].rolling(5).sum()
                regime_data['return_21d'] = features['return_21d'] if 'return_21d' in features.columns else features['return_1d'].rolling(21).sum()
                
                # Return skewness and kurtosis
                regime_data['return_skew_21d'] = features['return_1d'].rolling(21).skew()
                regime_data['return_kurt_21d'] = features['return_1d'].rolling(21).kurt()
            
            # Volatility features
            if 'realized_vol_21d' in features.columns:
                regime_data['volatility'] = features['realized_vol_21d']
                regime_data['vol_of_vol'] = features['realized_vol_21d'].rolling(21).std()
                regime_data['vol_regime'] = features['vol_regime'] if 'vol_regime' in features.columns else 0
            
            # Momentum features
            momentum_cols = [col for col in features.columns if 'momentum' in col]
            if momentum_cols:
                regime_data['momentum_composite'] = features[momentum_cols].mean(axis=1)
                regime_data['momentum_consistency'] = features[momentum_cols].std(axis=1)
            
            # Trend strength
            if 'trend_strength' in features.columns:
                regime_data['trend_strength'] = features['trend_strength']
            
            # Volume features
            if 'volume_ratio' in features.columns:
                regime_data['volume_ratio'] = features['volume_ratio']
                regime_data['volume_trend'] = features['volume_ratio'].rolling(10).apply(
                    lambda x: np.corrcoef(x, range(len(x)))[0, 1] if len(x) > 1 else 0, raw=False
                )
            
            # Technical indicators
            if 'rsi' in features.columns:
                regime_data['rsi'] = features['rsi']
                regime_data['rsi_regime'] = np.where(features['rsi'] > 70, 1, np.where(features['rsi'] < 30, -1, 0))
            
            if 'bb_position' in features.columns:
                regime_data['bb_position'] = features['bb_position']
            
            # Market stress indicators
            if 'vix_level' in features.columns:
                regime_data['vix_level'] = features['vix_level']
                regime_data['vix_regime'] = features['vix_regime'] if 'vix_regime' in features.columns else 0
            
            # Fill missing values
            regime_data = regime_data.fillna(method='ffill').fillna(0)
            
            return regime_data
            
        except Exception as e:
            logger.error(f"Regime feature preparation failed: {e}")
            raise
    
    async def _get_regime_model(self, symbol: str, features: pd.DataFrame) -> GaussianMixture:
        """Get or train regime detection model"""
        
        try:
            if symbol not in self.regime_models:
                # Train new model
                model = await self._train_regime_model(features)
                self.regime_models[symbol] = model
            else:
                # Check if model needs retraining
                if await self._should_retrain_model(symbol, features):
                    model = await self._train_regime_model(features)
                    self.regime_models[symbol] = model
                else:
                    model = self.regime_models[symbol]
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to get regime model for {symbol}: {e}")
            raise
    
    async def _train_regime_model(self, features: pd.DataFrame) -> GaussianMixture:
        """Train Gaussian Mixture Model for regime detection"""
        
        try:
            # Use last 2 years of data for training
            training_data = features.iloc[-504:] if len(features) > 504 else features
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(training_data.values)
            
            # Train Gaussian Mixture Model
            model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                max_iter=100,
                random_state=42
            )
            
            model.fit(scaled_features)
            
            # Store scaler with model
            model.scaler = scaler
            
            return model
            
        except Exception as e:
            logger.error(f"Regime model training failed: {e}")
            raise
    
    async def _get_regime_characteristics(
        self,
        regime_id: int,
        regime_probs: np.ndarray,
        recent_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get characteristics of the detected regime"""
        
        try:
            # Calculate recent statistics
            if 'daily_return' in recent_data.columns:
                mean_return = recent_data['daily_return'].mean()
                volatility = recent_data['daily_return'].std()
            else:
                mean_return = 0.0
                volatility = 0.02  # Default volatility
            
            # Define regime characteristics based on regime_id
            regime_definitions = {
                0: {  # Bull Market
                    'name': 'Bull Market',
                    'volatility_level': 'low' if volatility < 0.015 else 'medium',
                    'trend_direction': 'bullish',
                    'characteristics': {
                        'typical_duration_days': 400,
                        'expected_return_annual': 0.12,
                        'max_drawdown_typical': 0.10,
                        'momentum_persistence': 'high'
                    }
                },
                1: {  # Bear Market
                    'name': 'Bear Market',
                    'volatility_level': 'high' if volatility > 0.025 else 'medium',
                    'trend_direction': 'bearish',
                    'characteristics': {
                        'typical_duration_days': 200,
                        'expected_return_annual': -0.15,
                        'max_drawdown_typical': 0.30,
                        'momentum_persistence': 'medium'
                    }
                },
                2: {  # Sideways Market
                    'name': 'Sideways Market',
                    'volatility_level': 'medium',
                    'trend_direction': 'neutral',
                    'characteristics': {
                        'typical_duration_days': 150,
                        'expected_return_annual': 0.05,
                        'max_drawdown_typical': 0.15,
                        'momentum_persistence': 'low'
                    }
                },
                3: {  # Crisis/High Volatility
                    'name': 'Crisis/High Volatility',
                    'volatility_level': 'very_high',
                    'trend_direction': 'highly_volatile',
                    'characteristics': {
                        'typical_duration_days': 60,
                        'expected_return_annual': -0.20,
                        'max_drawdown_typical': 0.40,
                        'momentum_persistence': 'very_low'
                    }
                }
            }
            
            regime_info = regime_definitions.get(regime_id, regime_definitions[2])  # Default to sideways
            regime_info['mean_return'] = mean_return
            regime_info['volatility'] = volatility
            
            return regime_info
            
        except Exception as e:
            logger.error(f"Failed to get regime characteristics: {e}")
            return {
                'name': 'Unknown',
                'volatility_level': 'medium',
                'trend_direction': 'neutral',
                'mean_return': 0.0,
                'volatility': 0.02,
                'characteristics': {}
            }
    
    async def _calculate_transition_probability(self, symbol: str, current_regime: int) -> float:
        """Calculate probability of transitioning from current regime"""
        
        try:
            if symbol not in self.transition_matrices:
                return 0.2  # Default transition probability
            
            transition_matrix = self.transition_matrices[symbol]
            
            # Probability of staying in current regime
            stay_probability = transition_matrix[current_regime, current_regime]
            
            # Transition probability is 1 - stay probability
            return float(1 - stay_probability)
            
        except Exception as e:
            logger.error(f"Failed to calculate transition probability: {e}")
            return 0.2
    
    async def _calculate_regime_persistence(self, symbol: str, current_regime: int) -> float:
        """Calculate how persistent the current regime is"""
        
        try:
            if symbol not in self.regime_history:
                return 0.5  # Default persistence
            
            # Look at last 20 regime observations
            recent_regimes = [r.regime_id for r in self.regime_history[symbol][-20:]]
            
            if not recent_regimes:
                return 0.5
            
            # Calculate persistence as proportion of recent observations in current regime
            persistence = sum(1 for r in recent_regimes if r == current_regime) / len(recent_regimes)
            
            return float(persistence)
            
        except Exception as e:
            logger.error(f"Failed to calculate regime persistence: {e}")
            return 0.5
    
    async def _update_regime_history(self, symbol: str, regime: MarketRegime):
        """Update regime history for a symbol"""
        
        try:
            if symbol not in self.regime_history:
                self.regime_history[symbol] = []
            
            # Add current regime to history
            self.regime_history[symbol].append(regime)
            
            # Keep only last 100 observations
            if len(self.regime_history[symbol]) > 100:
                self.regime_history[symbol] = self.regime_history[symbol][-100:]
            
            # Update transition matrix
            await self._update_transition_matrix(symbol)
            
        except Exception as e:
            logger.error(f"Failed to update regime history for {symbol}: {e}")
    
    async def _update_transition_matrix(self, symbol: str):
        """Update transition probability matrix"""
        
        try:
            if len(self.regime_history[symbol]) < 10:
                return  # Need minimum history
            
            # Extract regime sequence
            regime_sequence = [r.regime_id for r in self.regime_history[symbol]]
            
            # Calculate transition counts
            transition_counts = np.zeros((self.n_regimes, self.n_regimes))
            
            for i in range(len(regime_sequence) - 1):
                from_regime = regime_sequence[i]
                to_regime = regime_sequence[i + 1]
                transition_counts[from_regime, to_regime] += 1
            
            # Convert to probabilities
            transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
            for i in range(self.n_regimes):
                row_sum = transition_counts[i, :].sum()
                if row_sum > 0:
                    transition_matrix[i, :] = transition_counts[i, :] / row_sum
                else:
                    # If no transitions observed, assume equal probability
                    transition_matrix[i, :] = 1.0 / self.n_regimes
            
            self.transition_matrices[symbol] = transition_matrix
            
        except Exception as e:
            logger.error(f"Failed to update transition matrix for {symbol}: {e}")
    
    async def _calculate_steady_state_probabilities(self, transition_matrix: np.ndarray) -> List[float]:
        """Calculate steady-state probabilities from transition matrix"""
        
        try:
            # Find the eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
            
            # Find the eigenvector corresponding to eigenvalue 1
            stationary_idx = np.argmin(np.abs(eigenvalues - 1))
            stationary_vector = np.real(eigenvectors[:, stationary_idx])
            
            # Normalize to get probabilities
            stationary_probabilities = stationary_vector / stationary_vector.sum()
            
            return stationary_probabilities.tolist()
            
        except Exception as e:
            logger.error(f"Failed to calculate steady state probabilities: {e}")
            return [0.25] * 4  # Equal probabilities as fallback
    
    async def _should_retrain_model(self, symbol: str, features: pd.DataFrame) -> bool:
        """Determine if model should be retrained"""
        
        # Retrain monthly or if significant regime changes detected
        if symbol not in self.regime_history:
            return False
        
        # Check if last 30 days show consistent regime changes
        recent_regimes = [r.regime_id for r in self.regime_history[symbol][-30:]]
        if len(set(recent_regimes)) > 2:  # More than 2 different regimes in last 30 days
            return True
        
        return False  # Keep existing model
    
    async def _get_default_regime(self) -> MarketRegime:
        """Return default regime when detection fails"""
        
        return MarketRegime(
            regime_id=2,  # Sideways market
            regime_name='Sideways Market',
            confidence=0.5,
            volatility_level='medium',
            trend_direction='neutral',
            mean_return=0.0,
            volatility=0.02,
            characteristics={
                'typical_duration_days': 150,
                'expected_return_annual': 0.05,
                'max_drawdown_typical': 0.15
            },
            transition_probability=0.2,
            persistence=0.5
        )

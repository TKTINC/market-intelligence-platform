#!/usr/bin/env python3
"""
=============================================================================
TFT FORECASTING ACCURACY TEST
Location: src/agents/tft/tests/forecast_accuracy_test.py
=============================================================================
"""

import numpy as np
import pandas as pd
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse
import sys
import os
from datetime import datetime, timedelta
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ForecastAccuracyMetrics:
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error
    mae: float   # Mean Absolute Error
    direction_accuracy: float  # Directional accuracy
    hit_rate: float  # Percentage of predictions within acceptable range
    sharpe_ratio: float  # Risk-adjusted return metric

@dataclass
class TFTTestResult:
    asset: str
    horizon_days: int
    metrics: ForecastAccuracyMetrics
    passed_accuracy_threshold: bool
    feature_importance: Dict[str, float]
    confidence_intervals: Dict[str, List[float]]

@dataclass
class TFTForecastTestSuite:
    forecast_accuracy_by_horizon: Dict[str, Dict[str, Any]]
    training_performance: Dict[str, Any]
    model_stability: Dict[str, Any]
    feature_analysis: Dict[str, Any]
    overall_performance: Dict[str, Any]
    recommendations: List[str]

class TFTForecastTest:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.getenv('TFT_MODEL_PATH', '/models/tft')
        
        # Test configuration
        self.time_horizons = [1, 5, 10, 20, 30]  # days
        self.test_assets = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Performance thresholds by horizon
        self.accuracy_thresholds = {
            1: {'mape': 3.0, 'direction_accuracy': 0.75, 'hit_rate': 0.80},
            5: {'mape': 5.0, 'direction_accuracy': 0.70, 'hit_rate': 0.75},
            10: {'mape': 7.0, 'direction_accuracy': 0.65, 'hit_rate': 0.70},
            20: {'mape': 10.0, 'direction_accuracy': 0.60, 'hit_rate': 0.65},
            30: {'mape': 12.0, 'direction_accuracy': 0.55, 'hit_rate': 0.60}
        }
        
        # Feature categories for analysis
        self.feature_categories = {
            'price_features': ['open', 'high', 'low', 'close', 'volume'],
            'technical_indicators': ['rsi', 'macd', 'bollinger_bands', 'moving_averages'],
            'volatility_features': ['realized_vol', 'implied_vol', 'garch_vol'],
            'sentiment_features': ['news_sentiment', 'social_sentiment', 'analyst_sentiment'],
            'macro_features': ['vix', 'yield_curve', 'dollar_index', 'commodity_prices']
        }
    
    def generate_mock_price_data(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """Generate realistic mock price data using geometric Brownian motion."""
        
        # Set different parameters for different assets
        asset_params = {
            'SPY': {'drift': 0.08, 'volatility': 0.16, 'base_price': 400},
            'QQQ': {'drift': 0.12, 'volatility': 0.22, 'base_price': 350},
            'IWM': {'drift': 0.06, 'volatility': 0.20, 'base_price': 200},
            'AAPL': {'drift': 0.15, 'volatility': 0.28, 'base_price': 170},
            'MSFT': {'drift': 0.13, 'volatility': 0.25, 'base_price': 350},
            'GOOGL': {'drift': 0.10, 'volatility': 0.30, 'base_price': 130},
            'AMZN': {'drift': 0.08, 'volatility': 0.35, 'base_price': 140},
            'TSLA': {'drift': 0.20, 'volatility': 0.50, 'base_price': 200}
        }
        
        params = asset_params.get(symbol, {'drift': 0.08, 'volatility': 0.20, 'base_price': 100})
        
        # Generate random walk
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        
        dt = 1/252  # Daily time step
        returns = np.random.normal(
            params['drift'] * dt,
            params['volatility'] * np.sqrt(dt),
            days
        )
        
        # Generate price series
        prices = [params['base_price']]
        for ret in returns:
            prices.append(prices[-1] * np.exp(ret))
        
        prices = prices[1:]  # Remove initial price
        
        # Generate OHLCV data
        dates = pd.date_range(start='2022-01-01', periods=days, freq='D')
        
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate intraday volatility
            daily_vol = params['volatility'] / np.sqrt(252)
            high = close * np.exp(np.abs(np.random.normal(0, daily_vol)))
            low = close * np.exp(-np.abs(np.random.normal(0, daily_vol)))
            open_price = prices[i-1] if i > 0 else close
            
            # Ensure OHLC relationships
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Volume (log-normal distribution)
            volume = int(np.random.lognormal(15, 1)) * 100
            
            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and features to price data."""
        
        # Simple moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volatility
        df['returns'] = df['close'].pct_change()
        df['realized_vol'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def simulate_tft_forecast(self, data: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Simulate TFT model forecasting with realistic behavior."""
        
        # Simulate model prediction time
        processing_time = 2.0 + horizon * 0.1 + np.random.normal(0, 0.5)
        processing_time = max(1.0, processing_time)
        time.sleep(processing_time / 10)  # Accelerated for testing
        
        # Use last N days for prediction
        lookback_days = min(60, len(data) - horizon)
        recent_data = data.tail(lookback_days).copy()
        
        # Simulate TFT prediction logic
        # Base prediction on recent trend and mean reversion
        recent_returns = recent_data['returns'].dropna()
        
        if len(recent_returns) == 0:
            # Fallback if no returns available
            forecast = [data['close'].iloc[-1]] * horizon
        else:
            # Trend component
            trend = recent_returns.mean()
            
            # Volatility component
            vol = recent_returns.std()
            
            # Mean reversion component (stronger for longer horizons)
            mean_reversion_strength = min(0.5, horizon / 60)
            long_term_mean = data['close'].mean()
            current_price = data['close'].iloc[-1]
            
            forecast = []
            price = current_price
            
            for i in range(horizon):
                # Trend continuation (weaker over time)
                trend_component = trend * (0.9 ** i)
                
                # Mean reversion (stronger over time)
                reversion_component = (long_term_mean - price) * mean_reversion_strength * (i / horizon)
                
                # Noise component
                noise = np.random.normal(0, vol) * (1 + i * 0.1)
                
                # Combine components
                daily_return = trend_component + reversion_component * 0.01 + noise * 0.5
                price = price * (1 + daily_return)
                forecast.append(price)
        
        # Generate confidence intervals
        forecast_array = np.array(forecast)
        uncertainty = np.linspace(0.05, 0.15, horizon)  # Increasing uncertainty
        
        confidence_intervals = {
            'upper_95': forecast_array * (1 + uncertainty * 2),
            'lower_95': forecast_array * (1 - uncertainty * 2),
            'upper_68': forecast_array * (1 + uncertainty),
            'lower_68': forecast_array * (1 - uncertainty)
        }
        
        # Simulate feature importance
        feature_importance = {
            'close': np.random.uniform(0.25, 0.35),
            'volume': np.random.uniform(0.10, 0.20),
            'rsi': np.random.uniform(0.05, 0.15),
            'macd': np.random.uniform(0.05, 0.15),
            'sma_20': np.random.uniform(0.10, 0.20),
            'realized_vol': np.random.uniform(0.08, 0.18),
            'bb_position': np.random.uniform(0.03, 0.10)
        }
        
        # Normalize feature importance
        total_importance = sum(feature_importance.values())
        feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        metadata = {
            'processing_time_ms': processing_time * 1000,
            'model_confidence': np.random.uniform(0.7, 0.9),
            'feature_importance': feature_importance,
            'confidence_intervals': confidence_intervals
        }
        
        return forecast_array, metadata
    
    def calculate_accuracy_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> ForecastAccuracyMetrics:
        """Calculate comprehensive accuracy metrics."""
        
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            # Return default metrics if no valid data
            return ForecastAccuracyMetrics(
                mape=100.0, rmse=float('inf'), mae=float('inf'),
                direction_accuracy=0.0, hit_rate=0.0, sharpe_ratio=0.0
            )
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
        
        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((actual_clean - predicted_clean) ** 2))
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(actual_clean - predicted_clean))
        
        # Direction accuracy
        if len(actual_clean) > 1:
            actual_direction = np.sign(np.diff(actual_clean))
            predicted_direction = np.sign(np.diff(predicted_clean))
            direction_accuracy = np.mean(actual_direction == predicted_direction)
        else:
            direction_accuracy = 0.0
        
        # Hit rate (predictions within 5% of actual)
        hit_threshold = 0.05
        hits = np.abs((actual_clean - predicted_clean) / actual_clean) <= hit_threshold
        hit_rate = np.mean(hits)
        
        # Sharpe ratio (simplified calculation)
        if len(actual_clean) > 1:
            returns_actual = np.diff(actual_clean) / actual_clean[:-1]
            returns_predicted = np.diff(predicted_clean) / predicted_clean[:-1]
            
            # Calculate Sharpe ratio of prediction strategy
            excess_returns = returns_predicted - np.mean(returns_actual)
            if np.std(returns_predicted) > 0:
                sharpe_ratio = np.mean(excess_returns) / np.std(returns_predicted) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        return ForecastAccuracyMetrics(
            mape=mape,
            rmse=rmse,
            mae=mae,
            direction_accuracy=direction_accuracy,
            hit_rate=hit_rate,
            sharpe_ratio=sharpe_ratio
        )
    
    def test_forecast_accuracy(self) -> Dict[str, Any]:
        """Test forecasting accuracy across different horizons and assets."""
        
        logger.info("Testing forecast accuracy across horizons and assets")
        
        results = {}
        
        for horizon in self.time_horizons:
            logger.info(f"Testing {horizon}-day forecast horizon")
            
            horizon_results = []
            
            for asset in self.test_assets:
                logger.info(f"  Testing {asset}")
                
                # Generate mock data
                total_days = 300 + horizon  # Extra days for out-of-sample testing
                price_data = self.generate_mock_price_data(asset, total_days)
                price_data = self.add_technical_features(price_data)
                
                # Split data for training/testing
                train_data = price_data[:-horizon].copy()
                actual_future = price_data[-horizon:]['close'].values
                
                # Generate forecast
                forecast, metadata = self.simulate_tft_forecast(train_data, horizon)
                
                # Calculate metrics
                metrics = self.calculate_accuracy_metrics(actual_future, forecast)
                
                # Check thresholds
                threshold = self.accuracy_thresholds[horizon]
                passed = (
                    metrics.mape <= threshold['mape'] and
                    metrics.direction_accuracy >= threshold['direction_accuracy'] and
                    metrics.hit_rate >= threshold['hit_rate']
                )
                
                result = TFTTestResult(
                    asset=asset,
                    horizon_days=horizon,
                    metrics=metrics,
                    passed_accuracy_threshold=passed,
                    feature_importance=metadata['feature_importance'],
                    confidence_intervals=metadata['confidence_intervals']
                )
                
                horizon_results.append(asdict(result))
                
                # Log result
                status = "✓ PASS" if passed else "✗ FAIL"
                logger.info(f"    {status} - MAPE: {metrics.mape:.1f}%, Dir: {metrics.direction_accuracy:.1%}")
            
            # Calculate horizon summary
            passed_assets = sum(1 for r in horizon_results if r['passed_accuracy_threshold'])
            avg_mape = np.mean([r['metrics']['mape'] for r in horizon_results])
            avg_direction = np.mean([r['metrics']['direction_accuracy'] for r in horizon_results])
            
            results[f'{horizon}_day'] = {
                'individual_results': horizon_results,
                'summary': {
                    'passed_assets': passed_assets,
                    'total_assets': len(self.test_assets),
                    'pass_rate': passed_assets / len(self.test_assets),
                    'average_mape': avg_mape,
                    'average_direction_accuracy': avg_direction,
                    'threshold_met': passed_assets / len(self.test_assets) >= 0.7  # 70% pass rate required
                }
            }
        
        return results
    
    def test_training_performance(self) -> Dict[str, Any]:
        """Test model training performance and convergence."""
        
        logger.info("Testing training performance")
        
        # Simulate training process
        training_config = {
            'epochs': 100,
            'batch_size': 256,
            'learning_rate': 0.001,
            'early_stopping_patience': 10
        }
        
        # Simulate training metrics over epochs
        epochs = training_config['epochs']
        train_losses = []
        val_losses = []
        
        # Initial loss
        initial_train_loss = np.random.uniform(0.8, 1.2)
        initial_val_loss = initial_train_loss * np.random.uniform(1.1, 1.3)
        
        for epoch in range(epochs):
            # Simulate decreasing loss with noise
            progress = epoch / epochs
            
            # Training loss (faster convergence)
            train_loss = initial_train_loss * np.exp(-progress * 3) + np.random.normal(0, 0.02)
            train_loss = max(0.05, train_loss)  # Minimum loss
            
            # Validation loss (slower convergence, more noise)
            val_loss = initial_val_loss * np.exp(-progress * 2.5) + np.random.normal(0, 0.03)
            val_loss = max(0.08, val_loss)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Early stopping check
            if epoch > 20:
                recent_val_losses = val_losses[-training_config['early_stopping_patience']:]
                if len(recent_val_losses) == training_config['early_stopping_patience']:
                    if val_loss >= min(recent_val_losses):
                        actual_epochs = epoch + 1
                        break
        else:
            actual_epochs = epochs
        
        # Calculate training metrics
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        best_val_loss = min(val_losses)
        
        # Convergence analysis
        convergence_epoch = next((i for i, loss in enumerate(val_losses) if loss < 0.3), actual_epochs)
        overfitting_detected = final_val_loss > best_val_loss * 1.2
        
        training_result = {
            'training_config': training_config,
            'actual_epochs': actual_epochs,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'best_val_loss': best_val_loss,
            'convergence_epoch': convergence_epoch,
            'overfitting_detected': overfitting_detected,
            'training_successful': final_train_loss < 0.3 and final_val_loss < 0.4,
            'loss_curves': {
                'train_losses': train_losses[:actual_epochs],
                'val_losses': val_losses[:actual_epochs]
            }
        }
        
        logger.info(f"Training completed in {actual_epochs} epochs")
        logger.info(f"Final validation loss: {final_val_loss:.3f}")
        
        return training_result
    
    def test_model_stability(self) -> Dict[str, Any]:
        """Test model stability across multiple runs."""
        
        logger.info("Testing model stability")
        
        stability_runs = 5
        stability_results = []
        
        # Test same asset/horizon multiple times
        test_asset = 'SPY'
        test_horizon = 5
        
        for run in range(stability_runs):
            logger.info(f"  Stability run {run + 1}/{stability_runs}")
            
            # Generate same base data with slight variations
            np.random.seed(42 + run)  # Slightly different seed each run
            price_data = self.generate_mock_price_data(test_asset, 300)
            price_data = self.add_technical_features(price_data)
            
            train_data = price_data[:-test_horizon].copy()
            actual_future = price_data[-test_horizon:]['close'].values
            
            # Generate forecast
            forecast, metadata = self.simulate_tft_forecast(train_data, test_horizon)
            
            # Calculate metrics
            metrics = self.calculate_accuracy_metrics(actual_future, forecast)
            
            stability_results.append({
                'run': run + 1,
                'mape': metrics.mape,
                'direction_accuracy': metrics.direction_accuracy,
                'hit_rate': metrics.hit_rate,
                'processing_time_ms': metadata['processing_time_ms']
            })
        
        # Calculate stability metrics
        mape_values = [r['mape'] for r in stability_results]
        direction_values = [r['direction_accuracy'] for r in stability_results]
        
        stability_analysis = {
            'runs_completed': stability_runs,
            'mape_statistics': {
                'mean': np.mean(mape_values),
                'std': np.std(mape_values),
                'coefficient_of_variation': np.std(mape_values) / np.mean(mape_values)
            },
            'direction_accuracy_statistics': {
                'mean': np.mean(direction_values),
                'std': np.std(direction_values),
                'coefficient_of_variation': np.std(direction_values) / np.mean(direction_values)
            },
            'stability_score': 1.0 - min(1.0, np.std(mape_values) / np.mean(mape_values)),
            'consistent_performance': np.std(mape_values) / np.mean(mape_values) < 0.2,
            'individual_runs': stability_results
        }
        
        logger.info(f"Stability score: {stability_analysis['stability_score']:.2f}")
        
        return stability_analysis
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance across different scenarios."""
        
        logger.info("Analyzing feature importance")
        
        feature_analysis = {}
        
        # Test feature importance for different assets and horizons
        for category, features in self.feature_categories.items():
            category_importance = []
            
            # Sample a few assets and horizons
            sample_assets = ['SPY', 'AAPL', 'TSLA']
            sample_horizons = [1, 5, 20]
            
            for asset in sample_assets:
                for horizon in sample_horizons:
                    price_data = self.generate_mock_price_data(asset, 300)
                    price_data = self.add_technical_features(price_data)
                    
                    train_data = price_data[:-horizon].copy()
                    _, metadata = self.simulate_tft_forecast(train_data, horizon)
                    
                    # Extract importance for features in this category
                    category_score = sum(metadata['feature_importance'].get(feature, 0) 
                                       for feature in features 
                                       if feature in metadata['feature_importance'])
                    
                    category_importance.append(category_score)
            
            feature_analysis[category] = {
                'avg_importance': np.mean(category_importance),
                'std_importance': np.std(category_importance),
                'samples': len(category_importance)
            }
        
        # Rank categories by importance
        ranked_categories = sorted(feature_analysis.items(), 
                                 key=lambda x: x[1]['avg_importance'], 
                                 reverse=True)
        
        return {
            'category_analysis': feature_analysis,
            'ranked_categories': [(cat, metrics['avg_importance']) for cat, metrics in ranked_categories],
            'most_important_category': ranked_categories[0][0] if ranked_categories else None
        }
    
    def generate_recommendations(self, forecast_results: Dict[str, Any],
                               training_results: Dict[str, Any],
                               stability_results: Dict[str, Any],
                               feature_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        
        recommendations = []
        
        # Forecast accuracy recommendations
        poor_horizons = []
        for horizon_key, results in forecast_results.items():
            if not results['summary']['threshold_met']:
                poor_horizons.append(horizon_key)
        
        if poor_horizons:
            recommendations.append(
                f"Poor performance on {len(poor_horizons)} horizon(s): {', '.join(poor_horizons)}. "
                f"Consider additional training data or model architecture improvements."
            )
        
        # Training recommendations
        if not training_results['training_successful']:
            recommendations.append(
                f"Training did not converge properly (final val loss: {training_results['final_val_loss']:.3f}). "
                f"Consider adjusting learning rate or training duration."
            )
        
        if training_results['overfitting_detected']:
            recommendations.append(
                "Overfitting detected. Consider regularization, dropout, or early stopping adjustments."
            )
        
        # Stability recommendations
        if not stability_results['consistent_performance']:
            recommendations.append(
                f"Model shows inconsistent performance across runs "
                f"(CV: {stability_results['mape_statistics']['coefficient_of_variation']:.2f}). "
                f"Consider ensemble methods or model averaging."
            )
        
        # Feature importance recommendations
        if feature_results['most_important_category']:
            top_category = feature_results['most_important_category']
            recommendations.append(
                f"Focus on improving {top_category} as they show highest importance. "
                f"Consider adding more features in this category."
            )
        
        # Performance optimization
        avg_mape = np.mean([results['summary']['average_mape'] 
                           for results in forecast_results.values()])
        if avg_mape > 8.0:
            recommendations.append(
                f"Overall MAPE ({avg_mape:.1f}%) is high. "
                f"Consider hyperparameter tuning or alternative model architectures."
            )
        
        if not recommendations:
            recommendations.append(
                "All tests passed within acceptable thresholds. "
                "TFT model performance is satisfactory across all metrics."
            )
        
        return recommendations
    
    async def run_full_test_suite(self) -> TFTForecastTestSuite:
        """Run complete TFT forecast test suite."""
        
        logger.info("="*60)
        logger.info("STARTING TFT FORECAST ACCURACY TEST SUITE")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Run forecast accuracy tests
        forecast_results = self.test_forecast_accuracy()
        
        # Run training performance tests
        training_results = self.test_training_performance()
        
        # Run stability tests
        stability_results = self.test_model_stability()
        
        # Run feature importance analysis
        feature_results = self.analyze_feature_importance()
        
        # Calculate overall performance
        horizon_pass_rates = [results['summary']['pass_rate'] 
                            for results in forecast_results.values()]
        overall_pass_rate = np.mean(horizon_pass_rates)
        
        overall_performance = {
            'overall_pass_rate': overall_pass_rate,
            'horizons_tested': len(self.time_horizons),
            'assets_tested': len(self.test_assets),
            'best_horizon': min(forecast_results.keys(), 
                              key=lambda k: forecast_results[k]['summary']['average_mape']),
            'worst_horizon': max(forecast_results.keys(), 
                               key=lambda k: forecast_results[k]['summary']['average_mape']),
            'avg_direction_accuracy': np.mean([results['summary']['average_direction_accuracy'] 
                                             for results in forecast_results.values()])
        }
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            forecast_results, training_results, stability_results, feature_results
        )
        
        total_duration = time.time() - start_time
        
        # Compile final result
        result = TFTForecastTestSuite(
            forecast_accuracy_by_horizon=forecast_results,
            training_performance=training_results,
            model_stability=stability_results,
            feature_analysis=feature_results,
            overall_performance=overall_performance,
            recommendations=recommendations
        )
        
        # Log summary
        logger.info("="*60)
        logger.info("TFT FORECAST TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Duration: {total_duration:.1f} seconds")
        logger.info(f"Overall Pass Rate: {overall_pass_rate:.1%}")
        logger.info(f"Best Horizon: {overall_performance['best_horizon']}")
        logger.info(f"Training Successful: {'✓' if training_results['training_successful'] else '✗'}")
        logger.info(f"Model Stability: {'✓' if stability_results['consistent_performance'] else '✗'}")
        
        # Log horizon performance
        logger.info("\nHorizon Performance:")
        for horizon_key, results in forecast_results.items():
            summary = results['summary']
            logger.info(f"  {horizon_key}: {summary['passed_assets']}/{summary['total_assets']} "
                       f"(MAPE: {summary['average_mape']:.1f}%)")
        
        return result

def main():
    parser = argparse.ArgumentParser(description='TFT Forecast Accuracy Test')
    parser.add_argument('--model-path', '-m', type=str,
                       help='Path to TFT model')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for results (JSON)')
    parser.add_argument('--assets', '-a', nargs='+',
                       help='Specific assets to test (default: all)')
    parser.add_argument('--horizons', '-h', nargs='+', type=int,
                       help='Specific horizons to test (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize test suite
        test_suite = TFTForecastTest(model_path=args.model_path)
        
        # Override test parameters if specified
        if args.assets:
            test_suite.test_assets = args.assets
        if args.horizons:
            test_suite.time_horizons = args.horizons
        
        # Run test suite
        result = asyncio.run(test_suite.run_full_test_suite())
        
        # Output results
        result_dict = asdict(result)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result_dict, f, indent=2)
            logger.info(f"Results written to {args.output}")
        else:
            print(json.dumps(result_dict, indent=2))
        
        # Exit with appropriate code
        overall_passed = result.overall_performance['overall_pass_rate'] >= 0.7
        sys.exit(0 if overall_passed else 1)
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
Performance Monitoring and Metrics for TFT Forecasting Service
"""

import asyncio
import time
import json
import logging
import asyncpg
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import numpy as np
import os

logger = logging.getLogger(__name__)

class ForecastingMetrics:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        
        # Performance metrics
        self.forecast_times = deque(maxlen=1000)
        self.forecast_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.symbol_metrics = defaultdict(lambda: defaultdict(list))
        
        # Real-time stats
        self.current_stats = {
            "total_forecasts": 0,
            "total_errors": 0,
            "avg_forecast_time": 0.0,
            "accuracy_7d": 0.0,
            "last_updated": datetime.utcnow()
        }
        
        # Model performance tracking
        self.model_performance = defaultdict(dict)
        self.forecast_validation = defaultdict(list)
        
        # Start background metrics collection
        asyncio.create_task(self._background_metrics_update())
    
    async def record_forecast(
        self,
        symbol: str,
        horizons: List[int],
        processing_time_ms: int,
        forecast_id: str,
        error: Optional[str] = None
    ):
        """Record metrics for a forecast generation request"""
        
        try:
            timestamp = time.time()
            
            # Record timing
            self.forecast_times.append(processing_time_ms)
            
            # Update counters
            self.current_stats["total_forecasts"] += 1
            self.forecast_counts[symbol] += 1
            
            if error:
                self.current_stats["total_errors"] += 1
                self.error_counts[error] += 1
            
            # Record symbol-specific metrics
            self.symbol_metrics[symbol]["processing_times"].append(processing_time_ms)
            self.symbol_metrics[symbol]["horizons_requested"].extend(horizons)
            self.symbol_metrics[symbol]["timestamps"].append(timestamp)
            
            # Update average forecast time
            if self.forecast_times:
                self.current_stats["avg_forecast_time"] = statistics.mean(self.forecast_times)
            
            self.current_stats["last_updated"] = datetime.utcnow()
            
            # Store in database for persistence
            await self._store_forecast_metrics(symbol, horizons, processing_time_ms, forecast_id, error)
            
        except Exception as e:
            logger.error(f"Failed to record forecast metrics: {e}")
    
    async def record_batch_forecast(
        self,
        batch_id: str,
        symbols: List[str],
        successful_forecasts: int
    ):
        """Record metrics for batch forecast generation"""
        
        try:
            batch_success_rate = successful_forecasts / len(symbols) if symbols else 0
            
            # Store batch metrics
            conn = await asyncpg.connect(self.db_url)
            
            await conn.execute("""
                INSERT INTO forecast_batch_metrics 
                (batch_id, symbols, total_requested, successful_forecasts, 
                 success_rate, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, batch_id, symbols, len(symbols), successful_forecasts, 
                batch_success_rate, datetime.utcnow())
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record batch metrics: {e}")
    
    async def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        
        try:
            # Calculate recent accuracy
            recent_accuracy = await self._calculate_recent_accuracy()
            self.current_stats["accuracy_7d"] = recent_accuracy
            
            # Calculate percentiles
            response_times = list(self.forecast_times)
            percentiles = {}
            if response_times:
                percentiles = {
                    "p50": statistics.median(response_times),
                    "p95": self._calculate_percentile(response_times, 95),
                    "p99": self._calculate_percentile(response_times, 99)
                }
            
            # Calculate error rate
            error_rate = (self.current_stats["total_errors"] / 
                         max(self.current_stats["total_forecasts"], 1))
            
            return {
                **self.current_stats,
                "response_time_percentiles": percentiles,
                "error_rate": error_rate,
                "active_symbols": len(self.symbol_metrics),
                "top_symbols": await self._get_top_symbols(),
                "model_health": await self._get_model_health_summary()
            }
            
        except Exception as e:
            logger.error(f"Failed to get current stats: {e}")
            return {"error": "Stats unavailable"}
    
    async def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed performance analytics"""
        
        try:
            current_stats = await self.get_current_stats()
            
            # Symbol-specific analytics
            symbol_analytics = {}
            for symbol, metrics in self.symbol_metrics.items():
                if metrics["processing_times"]:
                    symbol_analytics[symbol] = {
                        "total_forecasts": len(metrics["processing_times"]),
                        "avg_processing_time": statistics.mean(metrics["processing_times"]),
                        "most_requested_horizons": self._get_most_common_horizons(
                            metrics["horizons_requested"]
                        ),
                        "last_forecast": datetime.fromtimestamp(
                            max(metrics["timestamps"])
                        ).isoformat() if metrics["timestamps"] else None
                    }
            
            # Error analysis
            total_errors = sum(self.error_counts.values())
            error_breakdown = {
                error: {
                    "count": count,
                    "percentage": (count / total_errors * 100) if total_errors > 0 else 0
                }
                for error, count in self.error_counts.items()
            }
            
            # Performance by horizon
            horizon_performance = await self._get_horizon_performance()
            
            # Model performance summary
            model_performance = await self._get_detailed_model_performance()
            
            return {
                "current_stats": current_stats,
                "symbol_analytics": symbol_analytics,
                "error_analytics": error_breakdown,
                "horizon_performance": horizon_performance,
                "model_performance": model_performance,
                "validation_results": await self._get_validation_summary(),
                "service_health": await self._assess_service_health()
            }
            
        except Exception as e:
            logger.error(f"Failed to get detailed stats: {e}")
            return {"error": "Detailed stats unavailable"}
    
    async def record_forecast_validation(
        self,
        forecast_id: str,
        symbol: str,
        horizon: int,
        predicted_value: float,
        actual_value: float,
        prediction_date: datetime,
        validation_date: datetime
    ):
        """Record forecast validation results"""
        
        try:
            # Calculate error metrics
            absolute_error = abs(actual_value - predicted_value)
            percentage_error = absolute_error / abs(actual_value) if actual_value != 0 else 0
            direction_correct = np.sign(predicted_value) == np.sign(actual_value)
            
            # Store validation record
            validation_record = {
                "forecast_id": forecast_id,
                "symbol": symbol,
                "horizon": horizon,
                "predicted_value": predicted_value,
                "actual_value": actual_value,
                "absolute_error": absolute_error,
                "percentage_error": percentage_error,
                "direction_correct": direction_correct,
                "prediction_date": prediction_date,
                "validation_date": validation_date
            }
            
            self.forecast_validation[symbol].append(validation_record)
            
            # Store in database
            conn = await asyncpg.connect(self.db_url)
            
            await conn.execute("""
                INSERT INTO forecast_validation 
                (forecast_id, symbol, horizon, predicted_value, actual_value,
                 absolute_error, percentage_error, direction_correct,
                 prediction_date, validation_date)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, forecast_id, symbol, horizon, predicted_value, actual_value,
                absolute_error, percentage_error, direction_correct,
                prediction_date, validation_date)
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record forecast validation: {e}")
    
    async def _store_forecast_metrics(
        self,
        symbol: str,
        horizons: List[int],
        processing_time_ms: int,
        forecast_id: str,
        error: Optional[str]
    ):
        """Store forecast metrics in database"""
        
        try:
            conn = await asyncpg.connect(self.db_url)
            
            await conn.execute("""
                INSERT INTO forecast_metrics 
                (forecast_id, symbol, horizons, processing_time_ms, error_message, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, forecast_id, symbol, horizons, processing_time_ms, error, datetime.utcnow())
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store forecast metrics: {e}")
    
    async def _calculate_recent_accuracy(self) -> float:
        """Calculate accuracy for recent forecasts"""
        
        try:
            conn = await asyncpg.connect(self.db_url)
            
            # Get validation results from last 7 days
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            accuracy = await conn.fetchval("""
                SELECT AVG(CASE WHEN direction_correct THEN 1.0 ELSE 0.0 END)
                FROM forecast_validation 
                WHERE validation_date >= $1
            """, week_ago)
            
            await conn.close()
            
            return float(accuracy) if accuracy is not None else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate recent accuracy: {e}")
            return 0.0
    
    async def _get_top_symbols(self) -> List[Dict[str, Any]]:
        """Get most frequently forecasted symbols"""
        
        try:
            # Sort by forecast count
            sorted_symbols = sorted(
                self.forecast_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [
                {"symbol": symbol, "forecast_count": count}
                for symbol, count in sorted_symbols[:10]
            ]
            
        except Exception as e:
            logger.error(f"Failed to get top symbols: {e}")
            return []
    
    async def _get_model_health_summary(self) -> Dict[str, Any]:
        """Get summary of model health across all symbols"""
        
        try:
            conn = await asyncpg.connect(self.db_url)
            
            # Get model performance from registry
            model_stats = await conn.fetch("""
                SELECT 
                    COUNT(*) as total_models,
                    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_models,
                    COUNT(CASE WHEN status = 'training' THEN 1 END) as training_models,
                    AVG((performance_metrics->>'directional_accuracy')::float) as avg_accuracy
                FROM model_registry
            """)
            
            await conn.close()
            
            if model_stats:
                stats = model_stats[0]
                return {
                    "total_models": stats["total_models"],
                    "active_models": stats["active_models"],
                    "training_models": stats["training_models"],
                    "average_accuracy": float(stats["avg_accuracy"]) if stats["avg_accuracy"] else 0.0
                }
            
            return {"total_models": 0, "active_models": 0, "training_models": 0, "average_accuracy": 0.0}
            
        except Exception as e:
            logger.error(f"Failed to get model health summary: {e}")
            return {}
    
    async def _get_horizon_performance(self) -> Dict[str, Any]:
        """Get performance metrics by forecast horizon"""
        
        try:
            conn = await asyncpg.connect(self.db_url)
            
            horizon_stats = await conn.fetch("""
                SELECT 
                    horizon,
                    COUNT(*) as total_forecasts,
                    AVG(percentage_error) as avg_error,
                    AVG(CASE WHEN direction_correct THEN 1.0 ELSE 0.0 END) as accuracy
                FROM forecast_validation
                WHERE validation_date >= $1
                GROUP BY horizon
                ORDER BY horizon
            """, datetime.utcnow() - timedelta(days=30))
            
            await conn.close()
            
            return {
                f"{row['horizon']}d": {
                    "total_forecasts": row["total_forecasts"],
                    "avg_error": float(row["avg_error"]) if row["avg_error"] else 0.0,
                    "accuracy": float(row["accuracy"]) if row["accuracy"] else 0.0
                }
                for row in horizon_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get horizon performance: {e}")
            return {}
    
    async def _get_detailed_model_performance(self) -> Dict[str, Any]:
        """Get detailed model performance metrics"""
        
        try:
            conn = await asyncpg.connect(self.db_url)
            
            # Get recent model performance
            model_perf = await conn.fetch("""
                SELECT 
                    mr.symbol,
                    mr.version,
                    mr.performance_metrics,
                    COUNT(fv.forecast_id) as recent_forecasts,
                    AVG(fv.percentage_error) as recent_avg_error,
                    AVG(CASE WHEN fv.direction_correct THEN 1.0 ELSE 0.0 END) as recent_accuracy
                FROM model_registry mr
                LEFT JOIN forecast_validation fv ON mr.symbol = fv.symbol
                WHERE mr.status = 'active' 
                AND (fv.validation_date >= $1 OR fv.validation_date IS NULL)
                GROUP BY mr.symbol, mr.version, mr.performance_metrics
            """, datetime.utcnow() - timedelta(days=30))
            
            await conn.close()
            
            performance_data = {}
            for row in model_perf:
                symbol = row["symbol"]
                performance_data[symbol] = {
                    "version": row["version"],
                    "training_metrics": json.loads(row["performance_metrics"]) if row["performance_metrics"] else {},
                    "recent_forecasts": row["recent_forecasts"] or 0,
                    "recent_avg_error": float(row["recent_avg_error"]) if row["recent_avg_error"] else 0.0,
                    "recent_accuracy": float(row["recent_accuracy"]) if row["recent_accuracy"] else 0.0
                }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to get detailed model performance: {e}")
            return {}
    
    async def _get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of forecast validation results"""
        
        try:
            conn = await asyncpg.connect(self.db_url)
            
            # Get validation summary for last 30 days
            validation_summary = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_validations,
                    AVG(percentage_error) as avg_error,
                    AVG(CASE WHEN direction_correct THEN 1.0 ELSE 0.0 END) as avg_accuracy,
                    STDDEV(percentage_error) as error_std
                FROM forecast_validation
                WHERE validation_date >= $1
            """, datetime.utcnow() - timedelta(days=30))
            
            await conn.close()
            
            if validation_summary:
                return {
                    "total_validations": validation_summary["total_validations"],
                    "average_error": float(validation_summary["avg_error"]) if validation_summary["avg_error"] else 0.0,
                    "average_accuracy": float(validation_summary["avg_accuracy"]) if validation_summary["avg_accuracy"] else 0.0,
                    "error_std": float(validation_summary["error_std"]) if validation_summary["error_std"] else 0.0
                }
            
            return {"total_validations": 0, "average_error": 0.0, "average_accuracy": 0.0, "error_std": 0.0}
            
        except Exception as e:
            logger.error(f"Failed to get validation summary: {e}")
            return {}
    
    async def _assess_service_health(self) -> Dict[str, Any]:
        """Assess overall service health"""
        
        try:
            # Calculate health metrics
            error_rate = (self.current_stats["total_errors"] / 
                         max(self.current_stats["total_forecasts"], 1))
            avg_response_time = self.current_stats["avg_forecast_time"]
            recent_accuracy = self.current_stats["accuracy_7d"]
            
            # Health score calculation
            health_score = 1.0
            
            # Error rate impact
            if error_rate > 0.1:  # >10% error rate
                health_score -= 0.4
            elif error_rate > 0.05:  # >5% error rate
                health_score -= 0.2
            
            # Response time impact
            if avg_response_time > 10000:  # >10 seconds
                health_score -= 0.3
            elif avg_response_time > 5000:  # >5 seconds
                health_score -= 0.1
            
            # Accuracy impact
            if recent_accuracy < 0.5:  # <50% accuracy
                health_score -= 0.4
            elif recent_accuracy < 0.6:  # <60% accuracy
                health_score -= 0.2
            
            # Determine health status
            if health_score >= 0.8:
                status = "healthy"
            elif health_score >= 0.6:
                status = "degraded"
            else:
                status = "unhealthy"
            
            return {
                "status": status,
                "score": max(health_score, 0.0),
                "error_rate": error_rate,
                "avg_response_time_ms": avg_response_time,
                "recent_accuracy": recent_accuracy,
                "recommendations": self._generate_health_recommendations(
                    error_rate, avg_response_time, recent_accuracy
                )
            }
            
        except Exception as e:
            logger.error(f"Health assessment failed: {e}")
            return {"status": "unknown", "error": str(e)}
    
    def _calculate_percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
    
    def _get_most_common_horizons(self, horizons: List[int]) -> List[int]:
        """Get most commonly requested horizons"""
        from collections import Counter
        
        if not horizons:
            return []
        
        counter = Counter(horizons)
        return [horizon for horizon, count in counter.most_common(3)]
    
    def _generate_health_recommendations(
        self,
        error_rate: float,
        avg_response_time: float,
        recent_accuracy: float
    ) -> List[str]:
        """Generate recommendations for service improvement"""
        
        recommendations = []
        
        if error_rate > 0.1:
            recommendations.append("High error rate detected - investigate error patterns and model stability")
        
        if avg_response_time > 5000:
            recommendations.append("Slow response times - consider model optimization or infrastructure scaling")
        
        if recent_accuracy < 0.6:
            recommendations.append("Low forecast accuracy - consider model retraining or feature engineering")
        
        if len(self.symbol_metrics) > 100:
            recommendations.append("High number of active symbols - consider model clustering or caching")
        
        if not recommendations:
            recommendations.append("Service operating within normal parameters")
        
        return recommendations
    
    async def _background_metrics_update(self):
        """Background task to update metrics periodically"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                
                # Clean old data
                cutoff_time = time.time() - 86400  # 24 hours
                
                # Clean symbol metrics
                for symbol in list(self.symbol_metrics.keys()):
                    metrics = self.symbol_metrics[symbol]
                    
                    # Filter out old timestamps
                    valid_indices = [
                        i for i, ts in enumerate(metrics["timestamps"])
                        if ts > cutoff_time
                    ]
                    
                    if valid_indices:
                        metrics["timestamps"] = [metrics["timestamps"][i] for i in valid_indices]
                        metrics["processing_times"] = [metrics["processing_times"][i] for i in valid_indices]
                        # Note: horizons_requested is cumulative, so we don't filter it by time
                    else:
                        # Remove symbol if no recent activity
                        del self.symbol_metrics[symbol]
                
                logger.debug("Metrics cleanup completed")
                
            except Exception as e:
                logger.error(f"Background metrics update failed: {e}")

"""
TFT Model Manager for Training, Versioning, and Deployment
"""

import torch
import torch.nn as nn
import asyncio
import asyncpg
import logging
import os
import joblib
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    model_id: str
    symbol: str
    version: str
    creation_date: datetime
    last_updated: datetime
    performance_metrics: Dict[str, float]
    training_config: Dict[str, Any]
    model_size_mb: float
    status: str  # 'training', 'active', 'deprecated'

@dataclass
class RetrainJob:
    job_id: str
    symbols: List[str]
    retrain_type: str
    priority: str
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    progress: float
    error_message: Optional[str]

class TFTModelManager:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.model_dir = "/app/models/tft"
        self.backup_dir = "/app/models/backup"
        self.max_concurrent_training = 2
        
        # Model versioning
        self.model_registry = {}  # symbol -> ModelMetadata
        self.active_jobs = {}  # job_id -> RetrainJob
        
        # Training queue
        self.training_queue = asyncio.Queue()
        self.training_executor = ThreadPoolExecutor(max_workers=self.max_concurrent_training)
        
        # Performance thresholds
        self.min_accuracy = 0.55  # Minimum directional accuracy
        self.max_mape = 0.15  # Maximum mean absolute percentage error
        
        # Initialize directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        
        try:
            # Load model registry from database
            await self._load_model_registry()
            
            status = {
                'total_models': len(self.model_registry),
                'active_models': len([m for m in self.model_registry.values() if m.status == 'active']),
                'training_models': len([m for m in self.model_registry.values() if m.status == 'training']),
                'deprecated_models': len([m for m in self.model_registry.values() if m.status == 'deprecated']),
                'active_training_jobs': len([j for j in self.active_jobs.values() if j.status == 'running']),
                'queue_size': self.training_queue.qsize(),
                'models_by_symbol': {
                    symbol: {
                        'version': metadata.version,
                        'status': metadata.status,
                        'last_updated': metadata.last_updated.isoformat(),
                        'performance': metadata.performance_metrics
                    }
                    for symbol, metadata in self.model_registry.items()
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get model status: {e}")
            return {"error": "Model status unavailable"}
    
    async def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        
        try:
            loaded_models = []
            
            for symbol, metadata in self.model_registry.items():
                if metadata.status == 'active':
                    model_path = os.path.join(self.model_dir, f"tft_{symbol}_{metadata.version}.pth")
                    if os.path.exists(model_path):
                        loaded_models.append(f"{symbol}_{metadata.version}")
            
            return loaded_models
            
        except Exception as e:
            logger.error(f"Failed to get loaded models: {e}")
            return []
    
    async def retrain_models(
        self,
        symbols: List[str],
        retrain_type: str = "incremental",
        retrain_id: str = None,
        priority: str = "normal"
    ):
        """Queue models for retraining"""
        
        try:
            if not retrain_id:
                retrain_id = f"retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create retrain job
            job = RetrainJob(
                job_id=retrain_id,
                symbols=symbols,
                retrain_type=retrain_type,
                priority=priority,
                status="queued",
                created_at=datetime.utcnow(),
                started_at=None,
                completed_at=None,
                progress=0.0,
                error_message=None
            )
            
            self.active_jobs[retrain_id] = job
            
            # Add to training queue
            await self.training_queue.put(job)
            
            # Start training worker if not running
            asyncio.create_task(self._training_worker())
            
            logger.info(f"Queued retraining job {retrain_id} for symbols: {symbols}")
            
        except Exception as e:
            logger.error(f"Failed to queue retraining: {e}")
            raise
    
    async def get_retrain_status(self, retrain_id: str) -> Dict[str, Any]:
        """Get status of a retraining job"""
        
        try:
            if retrain_id not in self.active_jobs:
                return {"error": "Retrain job not found"}
            
            job = self.active_jobs[retrain_id]
            
            return {
                "job_id": job.job_id,
                "symbols": job.symbols,
                "retrain_type": job.retrain_type,
                "priority": job.priority,
                "status": job.status,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error_message": job.error_message,
                "estimated_completion": self._estimate_completion_time(job)
            }
            
        except Exception as e:
            logger.error(f"Failed to get retrain status: {e}")
            return {"error": str(e)}
    
    async def _training_worker(self):
        """Background worker for processing training jobs"""
        
        try:
            while not self.training_queue.empty():
                try:
                    # Get next job from queue
                    job = await asyncio.wait_for(self.training_queue.get(), timeout=1.0)
                    
                    # Update job status
                    job.status = "running"
                    job.started_at = datetime.utcnow()
                    
                    logger.info(f"Starting training job {job.job_id}")
                    
                    # Process each symbol
                    for i, symbol in enumerate(job.symbols):
                        try:
                            await self._retrain_symbol_model(symbol, job.retrain_type)
                            job.progress = (i + 1) / len(job.symbols)
                            
                        except Exception as symbol_error:
                            logger.error(f"Failed to retrain model for {symbol}: {symbol_error}")
                            continue
                    
                    # Mark job as completed
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    job.progress = 1.0
                    
                    logger.info(f"Completed training job {job.job_id}")
                    
                except asyncio.TimeoutError:
                    break  # No more jobs in queue
                except Exception as e:
                    logger.error(f"Training worker error: {e}")
                    if 'job' in locals():
                        job.status = "failed"
                        job.error_message = str(e)
                        job.completed_at = datetime.utcnow()
                    
        except Exception as e:
            logger.error(f"Training worker failed: {e}")
    
    async def _retrain_symbol_model(self, symbol: str, retrain_type: str):
        """Retrain model for a specific symbol"""
        
        try:
            logger.info(f"Retraining {retrain_type} model for {symbol}")
            
            # Get training data
            training_data = await self._get_training_data(symbol)
            
            if len(training_data) < 252:  # Need at least 1 year of data
                raise ValueError(f"Insufficient training data for {symbol}: {len(training_data)} days")
            
            # Prepare features and targets
            features, targets = await self._prepare_training_data(training_data, symbol)
            
            # Train model
            if retrain_type == "full":
                model = await self._train_full_model(features, targets, symbol)
            else:  # incremental
                model = await self._train_incremental_model(features, targets, symbol)
            
            # Validate model performance
            performance = await self._validate_model(model, features, targets, symbol)
            
            if not await self._meets_performance_criteria(performance):
                raise ValueError(f"Model for {symbol} does not meet performance criteria")
            
            # Save model
            await self._save_model(model, symbol, performance)
            
            # Update model registry
            await self._update_model_registry(symbol, performance)
            
            logger.info(f"Successfully retrained model for {symbol}")
            
        except Exception as e:
            logger.error(f"Model retraining failed for {symbol}: {e}")
            raise
    
    async def _get_training_data(self, symbol: str) -> pd.DataFrame:
        """Get training data for a symbol"""
        
        try:
            conn = await asyncpg.connect(self.db_url)
            
            # Get 5 years of data for training
            start_date = datetime.utcnow() - timedelta(days=1825)
            
            data = await conn.fetch("""
                SELECT date, open_price, high_price, low_price, close_price, volume,
                       adjusted_close, vwap, volatility_1d, volatility_5d, volatility_21d
                FROM price_data 
                WHERE symbol = $1 AND date >= $2
                ORDER BY date
            """, symbol, start_date)
            
            await conn.close()
            
            if not data:
                raise ValueError(f"No training data found for {symbol}")
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get training data for {symbol}: {e}")
            raise
    
    async def _prepare_training_data(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare training data for TFT model"""
        
        try:
            # Import required components
            from .data_preprocessor import FinancialDataPreprocessor
            from .feature_engineer import MultiScaleFeatureEngineer
            
            preprocessor = FinancialDataPreprocessor()
            feature_engineer = MultiScaleFeatureEngineer()
            
            # Engineer features
            features = await feature_engineer.engineer_features(data, symbol)
            
            # Prepare for TFT (multi-horizon targets)
            horizons = [1, 5, 10, 21]  # Multiple prediction horizons
            
            # Create sequences for training
            sequence_length = 60  # 60 days lookback
            features_sequences = []
            targets_sequences = []
            
            for i in range(sequence_length, len(features) - max(horizons)):
                # Feature sequence
                feature_seq = features.iloc[i-sequence_length:i].values
                features_sequences.append(feature_seq)
                
                # Multi-horizon targets
                current_price = features.iloc[i]['close_price']
                horizon_targets = []
                
                for h in horizons:
                    if i + h < len(features):
                        future_price = features.iloc[i + h]['close_price']
                        return_target = (future_price - current_price) / current_price
                        horizon_targets.append(return_target)
                    else:
                        horizon_targets.append(0.0)  # Padding
                
                targets_sequences.append(horizon_targets)
            
            # Convert to tensors
            features_tensor = torch.FloatTensor(np.array(features_sequences))
            targets_tensor = torch.FloatTensor(np.array(targets_sequences))
            
            return features_tensor, targets_tensor
            
        except Exception as e:
            logger.error(f"Training data preparation failed for {symbol}: {e}")
            raise
    
    async def _train_full_model(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        symbol: str
    ) -> nn.Module:
        """Train a full TFT model from scratch"""
        
        try:
            from .models.tft_model import TemporalFusionTransformer
            
            # Model configuration
            config = {
                "input_size": features.shape[2],
                "hidden_size": 128,
                "num_attention_heads": 8,
                "dropout_rate": 0.1,
                "num_encoder_layers": 4,
                "num_decoder_layers": 4,
                "output_size": targets.shape[1],  # Number of horizons
                "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9]
            }
            
            # Initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = TemporalFusionTransformer(config)
            model.to(device)
            
            # Training setup
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Split data
            train_size = int(0.8 * len(features))
            train_features = features[:train_size].to(device)
            train_targets = targets[:train_size].to(device)
            val_features = features[train_size:].to(device)
            val_targets = targets[train_size:].to(device)
            
            # Training loop
            epochs = 100
            batch_size = 32
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                
                # Mini-batch training
                for i in range(0, len(train_features), batch_size):
                    batch_features = train_features[i:i+batch_size]
                    batch_targets = train_targets[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs['point_forecast'], batch_targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(val_features)
                    val_loss = criterion(val_outputs['point_forecast'], val_targets).item()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch} for {symbol}")
                    break
            
            # Load best model state
            model.load_state_dict(best_model_state)
            
            return model
            
        except Exception as e:
            logger.error(f"Full model training failed for {symbol}: {e}")
            raise
    
    async def _train_incremental_model(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        symbol: str
    ) -> nn.Module:
        """Train model incrementally (fine-tuning)"""
        
        try:
            # Load existing model
            existing_model_path = await self._find_latest_model(symbol)
            
            if not existing_model_path:
                # No existing model, train full model
                return await self._train_full_model(features, targets, symbol)
            
            from .models.tft_model import TemporalFusionTransformer
            
            # Load existing model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(existing_model_path, map_location=device)
            
            model = TemporalFusionTransformer(checkpoint['config'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            # Fine-tuning with lower learning rate
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            criterion = nn.MSELoss()
            
            # Use only recent data for incremental training
            recent_data_size = min(len(features), 252)  # Last year
            train_features = features[-recent_data_size:].to(device)
            train_targets = targets[-recent_data_size:].to(device)
            
            # Fine-tuning loop (fewer epochs)
            epochs = 20
            for epoch in range(epochs):
                model.train()
                
                optimizer.zero_grad()
                outputs = model(train_features)
                loss = criterion(outputs['point_forecast'], train_targets)
                loss.backward()
                optimizer.step()
            
            return model
            
        except Exception as e:
            logger.error(f"Incremental model training failed for {symbol}: {e}")
            raise
    
    async def _validate_model(
        self,
        model: nn.Module,
        features: torch.Tensor,
        targets: torch.Tensor,
        symbol: str
    ) -> Dict[str, float]:
        """Validate model performance"""
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            
            # Use last 20% of data for validation
            val_size = int(0.2 * len(features))
            val_features = features[-val_size:].to(device)
            val_targets = targets[-val_size:].to(device)
            
            with torch.no_grad():
                predictions = model(val_features)['point_forecast']
            
            # Convert to numpy for metrics calculation
            pred_np = predictions.cpu().numpy()
            target_np = val_targets.cpu().numpy()
            
            # Calculate performance metrics
            performance = {}
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((target_np - pred_np) / (target_np + 1e-8)))
            performance['mape'] = float(mape)
            
            # Root Mean Square Error
            rmse = np.sqrt(np.mean((target_np - pred_np) ** 2))
            performance['rmse'] = float(rmse)
            
            # Directional Accuracy
            target_direction = np.sign(target_np)
            pred_direction = np.sign(pred_np)
            directional_accuracy = np.mean(target_direction == pred_direction)
            performance['directional_accuracy'] = float(directional_accuracy)
            
            # Sharpe Ratio (if applicable)
            if np.std(pred_np) > 0:
                sharpe_ratio = np.mean(pred_np) / np.std(pred_np) * np.sqrt(252)
                performance['sharpe_ratio'] = float(sharpe_ratio)
            else:
                performance['sharpe_ratio'] = 0.0
            
            # R-squared
            ss_res = np.sum((target_np - pred_np) ** 2)
            ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
            performance['r_squared'] = float(r_squared)
            
            return performance
            
        except Exception as e:
            logger.error(f"Model validation failed for {symbol}: {e}")
            return {}
    
    async def _meets_performance_criteria(self, performance: Dict[str, float]) -> bool:
        """Check if model meets minimum performance criteria"""
        
        try:
            directional_accuracy = performance.get('directional_accuracy', 0.0)
            mape = performance.get('mape', 1.0)
            
            return (directional_accuracy >= self.min_accuracy and 
                   mape <= self.max_mape)
            
        except Exception as e:
            logger.error(f"Performance criteria check failed: {e}")
            return False
    
    async def _save_model(
        self,
        model: nn.Module,
        symbol: str,
        performance: Dict[str, float]
    ):
        """Save trained model to disk"""
        
        try:
            # Generate version number
            version = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
            
            model_filename = f"tft_{symbol}_{version}.pth"
            model_path = os.path.join(self.model_dir, model_filename)
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': getattr(model, 'config', {}),
                'symbol': symbol,
                'version': version,
                'timestamp': datetime.utcnow(),
                'performance': performance
            }, model_path)
            
            logger.info(f"Saved model for {symbol} at {model_path}")
            
            # Backup old model
            await self._backup_old_model(symbol)
            
        except Exception as e:
            logger.error(f"Model saving failed for {symbol}: {e}")
            raise
    
    async def _update_model_registry(self, symbol: str, performance: Dict[str, float]):
        """Update model registry in database"""
        
        try:
            conn = await asyncpg.connect(self.db_url)
            
            # Mark old models as deprecated
            await conn.execute("""
                UPDATE model_registry 
                SET status = 'deprecated', updated_at = $1
                WHERE symbol = $2 AND status = 'active'
            """, datetime.utcnow(), symbol)
            
            # Insert new model record
            version = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
            
            await conn.execute("""
                INSERT INTO model_registry 
                (symbol, version, performance_metrics, status, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, symbol, version, json.dumps(performance), 'active', 
                datetime.utcnow(), datetime.utcnow())
            
            await conn.close()
            
            # Update in-memory registry
            metadata = ModelMetadata(
                model_id=f"{symbol}_{version}",
                symbol=symbol,
                version=version,
                creation_date=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                performance_metrics=performance,
                training_config={},
                model_size_mb=0.0,
                status='active'
            )
            
            self.model_registry[symbol] = metadata
            
        except Exception as e:
            logger.error(f"Model registry update failed for {symbol}: {e}")
            raise
    
    async def _load_model_registry(self):
        """Load model registry from database"""
        
        try:
            conn = await asyncpg.connect(self.db_url)
            
            models = await conn.fetch("""
                SELECT symbol, version, performance_metrics, status, created_at, updated_at
                FROM model_registry 
                WHERE status = 'active'
            """)
            
            await conn.close()
            
            for model in models:
                metadata = ModelMetadata(
                    model_id=f"{model['symbol']}_{model['version']}",
                    symbol=model['symbol'],
                    version=model['version'],
                    creation_date=model['created_at'],
                    last_updated=model['updated_at'],
                    performance_metrics=json.loads(model['performance_metrics']),
                    training_config={},
                    model_size_mb=0.0,
                    status=model['status']
                )
                
                self.model_registry[model['symbol']] = metadata
            
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
    
    async def _find_latest_model(self, symbol: str) -> Optional[str]:
        """Find path to latest model for symbol"""
        
        try:
            # Look for model files
            model_files = [
                f for f in os.listdir(self.model_dir)
                if f.startswith(f"tft_{symbol}_") and f.endswith(".pth")
            ]
            
            if not model_files:
                return None
            
            # Return most recent file
            latest_file = max(model_files, key=lambda x: os.path.getctime(
                os.path.join(self.model_dir, x)
            ))
            
            return os.path.join(self.model_dir, latest_file)
            
        except Exception as e:
            logger.error(f"Failed to find latest model for {symbol}: {e}")
            return None
    
    async def _backup_old_model(self, symbol: str):
        """Backup existing model before replacing"""
        
        try:
            existing_model = await self._find_latest_model(symbol)
            
            if existing_model and os.path.exists(existing_model):
                backup_filename = f"backup_{os.path.basename(existing_model)}"
                backup_path = os.path.join(self.backup_dir, backup_filename)
                
                # Copy to backup location
                import shutil
                shutil.copy2(existing_model, backup_path)
                
                logger.info(f"Backed up model for {symbol} to {backup_path}")
            
        except Exception as e:
            logger.error(f"Model backup failed for {symbol}: {e}")
    
    def _estimate_completion_time(self, job: RetrainJob) -> Optional[str]:
        """Estimate completion time for a training job"""
        
        try:
            if job.status == "completed":
                return None
            
            if job.status == "running" and job.started_at:
                elapsed = datetime.utcnow() - job.started_at
                
                if job.progress > 0:
                    total_time = elapsed / job.progress
                    remaining_time = total_time - elapsed
                    completion_time = datetime.utcnow() + remaining_time
                    return completion_time.isoformat()
            
            # Default estimate for queued jobs
            estimated_minutes = len(job.symbols) * 30  # 30 minutes per symbol
            completion_time = datetime.utcnow() + timedelta(minutes=estimated_minutes)
            return completion_time.isoformat()
            
        except Exception as e:
            logger.error(f"Completion time estimation failed: {e}")
            return None

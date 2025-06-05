-- TFT Forecasting Service Database Schema
-- Tables for forecasting metrics, model management, and validation

-- Forecast metrics tracking
CREATE TABLE IF NOT EXISTS forecast_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    forecast_id VARCHAR(64) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    horizons INTEGER[] NOT NULL,
    processing_time_ms INTEGER NOT NULL,
    error_message TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    INDEX idx_forecast_metrics_symbol (symbol),
    INDEX idx_forecast_metrics_timestamp (timestamp),
    INDEX idx_forecast_metrics_forecast_id (forecast_id)
);

-- Batch forecast metrics
CREATE TABLE IF NOT EXISTS forecast_batch_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id VARCHAR(64) NOT NULL,
    symbols TEXT[] NOT NULL,
    total_requested INTEGER NOT NULL,
    successful_forecasts INTEGER NOT NULL,
    success_rate NUMERIC(5,4) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    INDEX idx_batch_metrics_batch_id (batch_id),
    INDEX idx_batch_metrics_timestamp (timestamp)
);

-- Model registry for TFT models
CREATE TABLE IF NOT EXISTS model_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    version VARCHAR(50) NOT NULL,
    performance_metrics JSONB NOT NULL,
    training_config JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    model_size_mb NUMERIC(10,2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    UNIQUE(symbol, version),
    INDEX idx_model_registry_symbol (symbol),
    INDEX idx_model_registry_status (status),
    INDEX idx_model_registry_updated (updated_at)
);

-- Forecast validation results
CREATE TABLE IF NOT EXISTS forecast_validation (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    forecast_id VARCHAR(64) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    horizon INTEGER NOT NULL,
    predicted_value NUMERIC(15,8) NOT NULL,
    actual_value NUMERIC(15,8) NOT NULL,
    absolute_error NUMERIC(15,8) NOT NULL,
    percentage_error NUMERIC(8,6) NOT NULL,
    direction_correct BOOLEAN NOT NULL,
    prediction_date TIMESTAMPTZ NOT NULL,
    validation_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    INDEX idx_forecast_validation_symbol (symbol),
    INDEX idx_forecast_validation_horizon (horizon),
    INDEX idx_forecast_validation_validation_date (validation_date),
    INDEX idx_forecast_validation_forecast_id (forecast_id)
);

-- Market regime tracking
CREATE TABLE IF NOT EXISTS market_regime_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    regime_id INTEGER NOT NULL,
    regime_name VARCHAR(50) NOT NULL,
    confidence NUMERIC(4,3) NOT NULL,
    volatility_level VARCHAR(20) NOT NULL,
    trend_direction VARCHAR(20) NOT NULL,
    characteristics JSONB,
    transition_probability NUMERIC(4,3),
    persistence NUMERIC(4,3),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    INDEX idx_regime_history_symbol (symbol),
    INDEX idx_regime_history_timestamp (timestamp),
    INDEX idx_regime_history_regime_id (regime_id)
);

-- Model training jobs
CREATE TABLE IF NOT EXISTS model_training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id VARCHAR(64) NOT NULL UNIQUE,
    symbols TEXT[] NOT NULL,
    retrain_type VARCHAR(20) NOT NULL,
    priority VARCHAR(10) NOT NULL DEFAULT 'normal',
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    progress NUMERIC(4,3) DEFAULT 0.0,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    
    INDEX idx_training_jobs_job_id (job_id),
    INDEX idx_training_jobs_status (status),
    INDEX idx_training_jobs_created_at (created_at)
);

-- Feature importance tracking
CREATE TABLE IF NOT EXISTS feature_importance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    horizon INTEGER NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    importance_score NUMERIC(8,6) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    INDEX idx_feature_importance_symbol (symbol),
    INDEX idx_feature_importance_horizon (horizon),
    INDEX idx_feature_importance_timestamp (timestamp)
);

-- Risk adjustment history
CREATE TABLE IF NOT EXISTS risk_adjustment_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    forecast_id VARCHAR(64) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    horizon INTEGER NOT NULL,
    original_price NUMERIC(15,8) NOT NULL,
    adjusted_price NUMERIC(15,8) NOT NULL,
    risk_score NUMERIC(4,3) NOT NULL,
    adjustment_factor NUMERIC(6,4) NOT NULL,
    risk_factors JSONB NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    INDEX idx_risk_adjustment_symbol (symbol),
    INDEX idx_risk_adjustment_forecast_id (forecast_id),
    INDEX idx_risk_adjustment_timestamp (timestamp)
);

-- Create partitioning for large tables (PostgreSQL 11+)
-- Partition forecast_metrics by month
CREATE TABLE forecast_metrics_template (LIKE forecast_metrics INCLUDING ALL);

-- Function to create monthly partitions
CREATE OR REPLACE FUNCTION create_monthly_partition_tft(table_name TEXT, start_date DATE)
RETURNS VOID AS $
DECLARE
    partition_name TEXT;
    end_date DATE;
BEGIN
    partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
    end_date := start_date + INTERVAL '1 month';
    
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF %I 
                    FOR VALUES FROM (%L) TO (%L)',
                    partition_name, table_name, start_date, end_date);
END;
$ LANGUAGE plpgsql;

-- Create initial partitions for current and next month
SELECT create_monthly_partition_tft('forecast_metrics', date_trunc('month', CURRENT_DATE));
SELECT create_monthly_partition_tft('forecast_metrics', date_trunc('month', CURRENT_DATE + INTERVAL '1 month'));

-- Views for common queries
CREATE VIEW forecast_accuracy_summary AS
SELECT 
    symbol,
    horizon,
    COUNT(*) as total_forecasts,
    AVG(CASE WHEN direction_correct THEN 1.0 ELSE 0.0 END) as directional_accuracy,
    AVG(percentage_error) as avg_percentage_error,
    STDDEV(percentage_error) as error_std,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY percentage_error) as median_error,
    MIN(validation_date) as first_validation,
    MAX(validation_date) as last_validation
FROM forecast_validation
WHERE validation_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY symbol, horizon;

CREATE VIEW model_performance_summary AS
SELECT 
    mr.symbol,
    mr.version,
    mr.status,
    (mr.performance_metrics->>'directional_accuracy')::float as training_accuracy,
    (mr.performance_metrics->>'mape')::float as training_mape,
    fas.directional_accuracy as validation_accuracy,
    fas.avg_percentage_error as validation_mape,
    mr.updated_at as model_updated,
    fas.last_validation
FROM model_registry mr
LEFT JOIN forecast_accuracy_summary fas ON mr.symbol = fas.symbol
WHERE mr.status = 'active';

CREATE VIEW recent_forecast_activity AS
SELECT 
    symbol,
    COUNT(*) as forecasts_last_24h,
    AVG(processing_time_ms) as avg_processing_time,
    COUNT(CASE WHEN error_message IS NOT NULL THEN 1 END) as error_count,
    MAX(timestamp) as last_forecast
FROM forecast_metrics
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY symbol
ORDER BY forecasts_last_24h DESC;

-- Triggers for automatic maintenance
CREATE OR REPLACE FUNCTION update_model_registry_timestamp()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_model_registry_timestamp
    BEFORE UPDATE ON model_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_model_registry_timestamp();

-- Data retention cleanup function
CREATE OR REPLACE FUNCTION cleanup_old_tft_data()
RETURNS VOID AS $
BEGIN
    -- Delete old forecast metrics (keep 6 months)
    DELETE FROM forecast_metrics 
    WHERE timestamp < NOW() - INTERVAL '6 months';
    
    -- Delete old validation data (keep 1 year)
    DELETE FROM forecast_validation 
    WHERE validation_date < NOW() - INTERVAL '1 year';
    
    -- Delete old regime history (keep 6 months)
    DELETE FROM market_regime_history 
    WHERE timestamp < NOW() - INTERVAL '6 months';
    
    -- Delete completed training jobs (keep 3 months)
    DELETE FROM model_training_jobs 
    WHERE status IN ('completed', 'failed') 
    AND completed_at < NOW() - INTERVAL '3 months';
    
    -- Delete old feature importance (keep 3 months)
    DELETE FROM feature_importance 
    WHERE timestamp < NOW() - INTERVAL '3 months';
    
    -- Delete old risk adjustment history (keep 6 months)
    DELETE FROM risk_adjustment_history 
    WHERE timestamp < NOW() - INTERVAL '6 months';
END;
$ LANGUAGE plpgsql;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO mip_service;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO mip_service;
GRANT SELECT ON ALL VIEWS IN SCHEMA public TO mip_service;

-- Indexes for performance optimization
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_forecast_metrics_symbol_timestamp 
ON forecast_metrics (symbol, timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_forecast_validation_symbol_horizon_date 
ON forecast_validation (symbol, horizon, validation_date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_registry_symbol_status_updated 
ON model_registry (symbol, status, updated_at DESC);

-- Schedule cleanup (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-tft-data', '0 3 * * 0', 'SELECT cleanup_old_tft_data();');

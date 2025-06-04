-- GPT-4 Strategy Service Database Schema
-- Run this script to set up required tables

-- GPT-4 usage tracking
CREATE TABLE IF NOT EXISTS gpt4_usage_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    strategy_id VARCHAR(64) NOT NULL,
    cost_usd NUMERIC(10,6) NOT NULL,
    tokens_used INTEGER NOT NULL,
    model_used VARCHAR(50) NOT NULL DEFAULT 'gpt-4-turbo',
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Additional metadata
    input_hash CHAR(64),
    output_hash CHAR(64),
    processing_time_ms INTEGER,
    strategies_generated INTEGER DEFAULT 1,
    validation_score FLOAT,
    
    -- Indexing for performance
    INDEX idx_gpt4_usage_user_time (user_id, timestamp),
    INDEX idx_gpt4_usage_strategy (strategy_id),
    INDEX idx_gpt4_usage_timestamp (timestamp)
);

-- Budget alerts tracking
CREATE TABLE IF NOT EXISTS budget_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    threshold_usd NUMERIC(10,2) NOT NULL,
    actual_spending_usd NUMERIC(10,2) NOT NULL,
    alert_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    alert_type VARCHAR(20) DEFAULT 'daily_threshold',
    acknowledged BOOLEAN DEFAULT FALSE,
    
    INDEX idx_budget_alerts_user (user_id),
    INDEX idx_budget_alerts_timestamp (alert_timestamp)
);

-- User LLM preferences (enhanced from base schema)
CREATE TABLE IF NOT EXISTS user_llm_settings (
    user_id UUID PRIMARY KEY REFERENCES users(id),
    strategy_model VARCHAR(30) DEFAULT 'gpt-4-turbo',
    fallback_model VARCHAR(30) DEFAULT 'gpt-3.5-turbo',
    max_cost_per_request NUMERIC(8,4) DEFAULT 0.50,
    latency_tolerance INTEGER DEFAULT 3000,
    user_tier VARCHAR(20) DEFAULT 'free',
    enable_portfolio_analysis BOOLEAN DEFAULT TRUE,
    enable_risk_validation BOOLEAN DEFAULT TRUE,
    preferred_strategy_types TEXT[], -- Array of preferred strategy types
    risk_tolerance VARCHAR(20) DEFAULT 'medium',
    
    -- Rate limiting preferences
    hourly_request_limit INTEGER DEFAULT 10,
    daily_cost_limit NUMERIC(10,2) DEFAULT 5.00,
    
    -- Feature flags
    enable_batch_requests BOOLEAN DEFAULT FALSE,
    enable_advanced_strategies BOOLEAN DEFAULT FALSE,
    
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Strategy performance tracking
CREATE TABLE IF NOT EXISTS strategy_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id VARCHAR(64) NOT NULL,
    user_id UUID NOT NULL REFERENCES users(id),
    strategy_name VARCHAR(100),
    strategy_type VARCHAR(50),
    
    -- Performance metrics
    confidence_score FLOAT,
    validation_score FLOAT,
    user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
    user_feedback TEXT,
    
    -- Strategy details
    strategies_generated INTEGER DEFAULT 1,
    max_profit NUMERIC(12,2),
    max_loss NUMERIC(12,2),
    capital_required NUMERIC(12,2),
    
    -- Greeks summary
    total_delta NUMERIC(8,4),
    total_gamma NUMERIC(8,4),
    total_theta NUMERIC(8,4),
    total_vega NUMERIC(8,4),
    
    -- Timestamps
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    rated_at TIMESTAMPTZ,
    
    INDEX idx_strategy_perf_user (user_id),
    INDEX idx_strategy_perf_type (strategy_type),
    INDEX idx_strategy_perf_rating (user_rating)
);

-- Market conditions snapshot for strategy context
CREATE TABLE IF NOT EXISTS market_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id VARCHAR(64) NOT NULL,
    
    -- Market indicators at time of strategy generation
    vix_level NUMERIC(6,2),
    spy_price NUMERIC(8,2),
    qqq_price NUMERIC(8,2),
    market_trend VARCHAR(20),
    sector_rotation VARCHAR(50),
    
    -- Options market data
    options_volume_ratio NUMERIC(6,4),
    put_call_ratio NUMERIC(6,4),
    implied_volatility_rank NUMERIC(6,2),
    
    snapshot_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    INDEX idx_market_snapshots_strategy (strategy_id),
    INDEX idx_market_snapshots_timestamp (snapshot_timestamp)
);

-- Agent audit log for compliance
CREATE TABLE IF NOT EXISTS agent_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_type VARCHAR(30) NOT NULL DEFAULT 'gpt4_strategy',
    user_id UUID NOT NULL REFERENCES users(id),
    
    -- Request details
    request_hash CHAR(64) NOT NULL,
    input_sanitized BOOLEAN DEFAULT FALSE,
    security_validated BOOLEAN DEFAULT FALSE,
    rate_limit_checked BOOLEAN DEFAULT FALSE,
    
    -- Processing details
    inference_time_ms INTEGER,
    tokens_used INTEGER,
    cost_usd NUMERIC(10,6),
    
    -- Results
    strategies_generated INTEGER DEFAULT 0,
    validation_passed BOOLEAN DEFAULT FALSE,
    fallback_used BOOLEAN DEFAULT FALSE,
    
    -- Compliance
    compliance_checked BOOLEAN DEFAULT FALSE,
    risk_warnings TEXT[],
    
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Compliance indexing
    INDEX idx_audit_user_time (user_id, timestamp),
    INDEX idx_audit_agent_type (agent_type),
    INDEX idx_audit_compliance (compliance_checked, timestamp)
);

-- Create partitioning for large tables (PostgreSQL 11+)
-- Partition gpt4_usage_log by month for performance
CREATE TABLE gpt4_usage_log_template (LIKE gpt4_usage_log INCLUDING ALL);

-- Function to create monthly partitions
CREATE OR REPLACE FUNCTION create_monthly_partition(table_name TEXT, start_date DATE)
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

-- Create initial partitions
SELECT create_monthly_partition('gpt4_usage_log', date_trunc('month', CURRENT_DATE));
SELECT create_monthly_partition('gpt4_usage_log', date_trunc('month', CURRENT_DATE + INTERVAL '1 month'));

-- Indexes for optimal performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gpt4_usage_user_cost_time 
ON gpt4_usage_log (user_id, cost_usd, timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gpt4_usage_model_time 
ON gpt4_usage_log (model_used, timestamp DESC);

-- Views for common queries
CREATE VIEW user_daily_usage AS
SELECT 
    user_id,
    DATE(timestamp) as usage_date,
    COUNT(*) as requests_count,
    SUM(cost_usd) as total_cost,
    AVG(processing_time_ms) as avg_processing_time,
    SUM(tokens_used) as total_tokens,
    SUM(strategies_generated) as total_strategies
FROM gpt4_usage_log
GROUP BY user_id, DATE(timestamp);

CREATE VIEW service_hourly_stats AS
SELECT 
    date_trunc('hour', timestamp) as hour,
    COUNT(*) as requests,
    SUM(cost_usd) as total_cost,
    AVG(cost_usd) as avg_cost,
    AVG(processing_time_ms) as avg_response_time,
    COUNT(DISTINCT user_id) as unique_users,
    model_used
FROM gpt4_usage_log
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY date_trunc('hour', timestamp), model_used
ORDER BY hour DESC;

-- Triggers for automatic data management
CREATE OR REPLACE FUNCTION update_user_llm_settings_timestamp()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_user_llm_settings_timestamp
    BEFORE UPDATE ON user_llm_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_user_llm_settings_timestamp();

-- Data retention policy (delete old audit logs)
CREATE OR REPLACE FUNCTION cleanup_old_audit_logs()
RETURNS VOID AS $
BEGIN
    DELETE FROM agent_audit_log 
    WHERE timestamp < NOW() - INTERVAL '90 days';
    
    DELETE FROM gpt4_usage_log 
    WHERE timestamp < NOW() - INTERVAL '1 year';
    
    DELETE FROM budget_alerts 
    WHERE alert_timestamp < NOW() - INTERVAL '6 months' AND acknowledged = TRUE;
END;
$ LANGUAGE plpgsql;

-- Schedule cleanup (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-audit-logs', '0 2 * * 0', 'SELECT cleanup_old_audit_logs();');

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON gpt4_usage_log TO mip_service;
GRANT SELECT, INSERT, UPDATE ON budget_alerts TO mip_service;
GRANT SELECT, INSERT, UPDATE, DELETE ON user_llm_settings TO mip_service;
GRANT SELECT, INSERT, UPDATE ON strategy_performance TO mip_service;
GRANT SELECT, INSERT ON market_snapshots TO mip_service;
GRANT SELECT, INSERT ON agent_audit_log TO mip_service;

-- Sequences
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO mip_service;

-- Views
GRANT SELECT ON user_daily_usage TO mip_service;
GRANT SELECT ON service_hourly_stats TO mip_service;

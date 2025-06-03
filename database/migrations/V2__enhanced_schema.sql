-- Enhanced PostgreSQL Schema for Market Intelligence Platform
-- Sprint 1, Prompt 3: User Customization & Options Intelligence
-- Author: MIP Development Team
-- Version: 2.0

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Set up tablespaces for performance optimization
-- CREATE TABLESPACE options_data LOCATION '/var/lib/postgresql/options_data';
-- CREATE TABLESPACE time_series_data LOCATION '/var/lib/postgresql/time_series';

-- ====================================
-- 1. ENHANCED CORE TABLES
-- ====================================

-- Enhanced assets table with options support
CREATE TABLE IF NOT EXISTS assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    asset_type VARCHAR(20) NOT NULL CHECK (asset_type IN ('stock', 'etf', 'index', 'crypto', 'forex')),
    exchange VARCHAR(10) NOT NULL,
    sector VARCHAR(50),
    industry VARCHAR(100),
    market_cap BIGINT,
    
    -- Options-specific fields
    options_enabled BOOLEAN DEFAULT false,
    avg_daily_volume BIGINT DEFAULT 0,
    beta NUMERIC(6,4),
    dividend_yield NUMERIC(6,4),
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    
    -- Indexes
    CONSTRAINT assets_symbol_format CHECK (symbol ~ '^[A-Z]{1,5}$')
);

-- Enhanced market data table with options support
DROP TABLE IF EXISTS market_data CASCADE;
CREATE TABLE market_data (
    asset_id UUID REFERENCES assets(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- OHLCV data
    open NUMERIC(18,8) NOT NULL CHECK (open > 0),
    high NUMERIC(18,8) NOT NULL CHECK (high > 0),
    low NUMERIC(18,8) NOT NULL CHECK (low > 0),
    close NUMERIC(18,8) NOT NULL CHECK (close > 0),
    volume BIGINT NOT NULL CHECK (volume >= 0),
    
    -- Enhanced market data
    vwap NUMERIC(18,8), -- Volume Weighted Average Price
    trades_count INTEGER,
    
    -- Options-specific market data (NEW)
    implied_volatility NUMERIC(8,4) CHECK (implied_volatility >= 0 AND implied_volatility <= 5.0),
    historical_volatility NUMERIC(8,4),
    iv_rank NUMERIC(5,2), -- IV percentile rank
    
    -- Market microstructure
    bid NUMERIC(18,8),
    ask NUMERIC(18,8),
    bid_size INTEGER,
    ask_size INTEGER,
    
    -- Metadata
    source VARCHAR(50) NOT NULL,
    ingestion_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Primary key and constraints
    PRIMARY KEY (asset_id, timestamp),
    CONSTRAINT market_data_price_validation CHECK (high >= low AND high >= open AND high >= close AND low <= open AND low <= close),
    CONSTRAINT market_data_bid_ask_validation CHECK (ask IS NULL OR bid IS NULL OR ask >= bid)
) PARTITION BY RANGE (timestamp);

-- Create partitions for market data (monthly partitions)
CREATE TABLE market_data_2024_01 PARTITION OF market_data
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE market_data_2024_02 PARTITION OF market_data
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
CREATE TABLE market_data_2024_03 PARTITION OF market_data
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');
CREATE TABLE market_data_2024_04 PARTITION OF market_data
    FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');
CREATE TABLE market_data_2024_05 PARTITION OF market_data
    FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');
CREATE TABLE market_data_2024_06 PARTITION OF market_data
    FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');
CREATE TABLE market_data_2024_07 PARTITION OF market_data
    FOR VALUES FROM ('2024-07-01') TO ('2024-08-01');
CREATE TABLE market_data_2024_08 PARTITION OF market_data
    FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');
CREATE TABLE market_data_2024_09 PARTITION OF market_data
    FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');
CREATE TABLE market_data_2024_10 PARTITION OF market_data
    FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');
CREATE TABLE market_data_2024_11 PARTITION OF market_data
    FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');
CREATE TABLE market_data_2024_12 PARTITION OF market_data
    FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');

-- ====================================
-- 2. NEW: OPTIONS INTELLIGENCE TABLES
-- ====================================

-- Options flow data table (NEW)
CREATE TABLE options_flow (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID REFERENCES assets(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Option contract details
    option_symbol VARCHAR(20) NOT NULL,
    expiry DATE NOT NULL,
    strike NUMERIC(10,2) NOT NULL CHECK (strike > 0),
    option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('call', 'put')),
    
    -- Trading data
    volume BIGINT NOT NULL CHECK (volume >= 0),
    open_interest BIGINT NOT NULL CHECK (open_interest >= 0),
    volume_oi_ratio NUMERIC(8,4) GENERATED ALWAYS AS (
        CASE WHEN open_interest > 0 THEN volume::NUMERIC / open_interest ELSE 0 END
    ) STORED,
    
    -- Pricing
    bid NUMERIC(10,4) CHECK (bid >= 0),
    ask NUMERIC(10,4) CHECK (ask >= 0),
    last_price NUMERIC(10,4) CHECK (last_price >= 0),
    
    -- Greeks
    implied_volatility NUMERIC(8,4) NOT NULL CHECK (implied_volatility >= 0 AND implied_volatility <= 5.0),
    delta NUMERIC(6,4) NOT NULL CHECK (delta >= -1.0 AND delta <= 1.0),
    gamma NUMERIC(6,4) NOT NULL CHECK (gamma >= 0),
    theta NUMERIC(6,4) NOT NULL CHECK (theta <= 0),
    vega NUMERIC(6,4) NOT NULL CHECK (vega >= 0),
    rho NUMERIC(6,4),
    
    -- Enhanced analytics (NEW)
    underlying_price NUMERIC(10,4),
    moneyness NUMERIC(6,4) GENERATED ALWAYS AS (
        CASE WHEN strike > 0 AND underlying_price > 0 
        THEN underlying_price / strike ELSE NULL END
    ) STORED,
    time_to_expiry NUMERIC(8,6), -- Years to expiry
    intrinsic_value NUMERIC(10,4),
    time_value NUMERIC(10,4),
    
    -- Market context
    iv_rank NUMERIC(5,2), -- 0-100 percentile
    unusual_activity BOOLEAN GENERATED ALWAYS AS (volume_oi_ratio > 0.5) STORED,
    large_trade BOOLEAN GENERATED ALWAYS AS (volume > 1000) STORED,
    
    -- Metadata
    source VARCHAR(50) NOT NULL,
    trade_type VARCHAR(20), -- 'buy', 'sell', 'sweep', etc.
    venue VARCHAR(50),
    ingestion_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT options_flow_bid_ask_check CHECK (ask IS NULL OR bid IS NULL OR ask >= bid),
    CONSTRAINT options_flow_expiry_check CHECK (expiry >= CURRENT_DATE),
    CONSTRAINT options_flow_delta_direction CHECK (
        (option_type = 'call' AND delta >= 0) OR 
        (option_type = 'put' AND delta <= 0)
    )
) PARTITION BY RANGE (expiry);

-- Create partitions for options data by expiry date
CREATE TABLE options_flow_2024_q1 PARTITION OF options_flow
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
CREATE TABLE options_flow_2024_q2 PARTITION OF options_flow
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');
CREATE TABLE options_flow_2024_q3 PARTITION OF options_flow
    FOR VALUES FROM ('2024-07-01') TO ('2024-10-01');
CREATE TABLE options_flow_2024_q4 PARTITION OF options_flow
    FOR VALUES FROM ('2024-10-01') TO ('2025-01-01');
CREATE TABLE options_flow_2025_q1 PARTITION OF options_flow
    FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');

-- ====================================
-- 3. USER MANAGEMENT & CUSTOMIZATION
-- ====================================

-- Enhanced users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    
    -- Profile information
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    timezone VARCHAR(50) DEFAULT 'UTC',
    
    -- Trading preferences
    risk_tolerance VARCHAR(20) DEFAULT 'moderate' CHECK (risk_tolerance IN ('conservative', 'moderate', 'aggressive')),
    experience_level VARCHAR(20) DEFAULT 'beginner' CHECK (experience_level IN ('beginner', 'intermediate', 'advanced', 'expert')),
    
    -- Account status
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    
    -- Subscription/tier information
    subscription_tier VARCHAR(20) DEFAULT 'free' CHECK (subscription_tier IN ('free', 'basic', 'premium', 'enterprise')),
    subscription_expires TIMESTAMPTZ
);

-- NEW: Custom Mag-7 weighting table for user customization
CREATE TABLE user_watchlist (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    
    -- Customization options
    custom_weight NUMERIC(4,2) DEFAULT 1.0 CHECK (custom_weight >= 0 AND custom_weight <= 10.0),
    display_order INTEGER DEFAULT 0,
    is_options_tracked BOOLEAN DEFAULT false,
    
    -- Position sizing preferences
    position_size_preference NUMERIC(10,2), -- Dollar amount or percentage
    position_sizing_type VARCHAR(10) DEFAULT 'dollar' CHECK (position_sizing_type IN ('dollar', 'percent')),
    
    -- Alert preferences
    price_alert_enabled BOOLEAN DEFAULT false,
    price_alert_threshold NUMERIC(5,2), -- Percentage change
    iv_alert_enabled BOOLEAN DEFAULT false,
    iv_alert_threshold NUMERIC(5,2), -- IV rank threshold
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (user_id, symbol),
    
    -- Ensure symbol exists in assets
    FOREIGN KEY (symbol) REFERENCES assets(symbol) ON UPDATE CASCADE
);

-- NEW: Virtual trading journal for performance tracking
CREATE TABLE virtual_trades (
    trade_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Trade identification
    recommendation_id UUID, -- Link to AI recommendation
    strategy_type VARCHAR(50) NOT NULL CHECK (strategy_type IN (
        'COVERED_CALL', 'CASH_SECURED_PUT', 'IRON_CONDOR', 'STRANGLE', 
        'BUTTERFLY', 'STRADDLE', 'COLLAR', 'CALENDAR_SPREAD'
    )),
    
    -- Asset information
    underlying_symbol VARCHAR(10) NOT NULL,
    
    -- Trade details
    strategy_details JSONB NOT NULL, -- Specific strategy configuration
    
    -- Execution information
    entry_price NUMERIC(18,4),
    exit_price NUMERIC(18,4),
    entry_date TIMESTAMPTZ DEFAULT NOW(),
    exit_date TIMESTAMPTZ,
    
    -- Performance metrics
    pnl NUMERIC(18,4),
    pnl_percentage NUMERIC(8,4),
    max_profit NUMERIC(18,4),
    max_loss NUMERIC(18,4),
    
    -- Risk metrics
    probability_profit NUMERIC(5,2), -- Expected probability of profit
    break_even_points JSONB, -- Array of break-even prices
    
    -- Trade status
    status VARCHAR(20) DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED', 'EXPIRED', 'ASSIGNED')),
    
    -- Market conditions at entry
    entry_iv_rank NUMERIC(5,2),
    entry_underlying_price NUMERIC(10,4),
    entry_dte INTEGER, -- Days to expiration
    
    -- Performance attribution
    alpha NUMERIC(8,4), -- Excess return vs benchmark
    sharpe_ratio NUMERIC(8,4),
    
    -- Metadata
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- NEW: Brokerage accounts integration
CREATE TABLE brokerage_accounts (
    account_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Broker information
    broker_name VARCHAR(50) NOT NULL CHECK (broker_name IN (
        'TD_AMERITRADE', 'INTERACTIVE_BROKERS', 'CHARLES_SCHWAB', 
        'E_TRADE', 'FIDELITY', 'ROBINHOOD'
    )),
    broker_account_id VARCHAR(100) NOT NULL, -- External account ID
    
    -- Account details
    account_type VARCHAR(20) DEFAULT 'margin' CHECK (account_type IN ('cash', 'margin', 'ira', 'roth_ira')),
    account_nickname VARCHAR(100),
    
    -- Integration status
    connection_status VARCHAR(20) DEFAULT 'disconnected' CHECK (connection_status IN (
        'connected', 'disconnected', 'expired', 'error'
    )),
    permissions JSONB DEFAULT '{"read_positions": true, "execute_trades": false}'::jsonb,
    
    -- Security
    oauth_token_hash VARCHAR(255), -- Hashed OAuth token
    oauth_expires TIMESTAMPTZ,
    last_sync TIMESTAMPTZ,
    
    -- Account balance information (cached)
    cash_balance NUMERIC(12,2),
    buying_power NUMERIC(12,2),
    total_value NUMERIC(12,2),
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Ensure unique broker account per user
    UNIQUE(user_id, broker_name, broker_account_id)
);

-- NEW: Options strategies catalog
CREATE TABLE options_strategies (
    strategy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID REFERENCES assets(id) ON DELETE CASCADE,
    
    -- Strategy definition
    strategy_name VARCHAR(50) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL,
    strategy_details JSONB NOT NULL,
    
    -- Market conditions
    underlying_price NUMERIC(10,4) NOT NULL,
    current_iv_rank NUMERIC(5,2),
    
    -- Strategy metrics
    max_profit NUMERIC(18,4),
    max_loss NUMERIC(18,4),
    probability_profit NUMERIC(5,2),
    expected_return NUMERIC(8,4),
    
    -- Risk assessment
    risk_level VARCHAR(20) CHECK (risk_level IN ('low', 'medium', 'high')),
    capital_required NUMERIC(18,4),
    margin_requirement NUMERIC(18,4),
    
    -- Greeks summary
    net_delta NUMERIC(8,4),
    net_gamma NUMERIC(8,4),
    net_theta NUMERIC(8,4),
    net_vega NUMERIC(8,4),
    
    -- Recommendation context
    market_regime VARCHAR(20), -- 'bullish', 'bearish', 'neutral', 'volatile'
    recommended_dte_min INTEGER,
    recommended_dte_max INTEGER,
    
    -- Performance tracking
    backtest_score NUMERIC(5,2), -- Historical success rate
    confidence_score NUMERIC(5,2), -- AI confidence level
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true
);

-- ====================================
-- 4. ENHANCED INDEXES FOR PERFORMANCE
-- ====================================

-- Market data indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_time 
    ON market_data (asset_id, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_timestamp 
    ON market_data (timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_source 
    ON market_data (source, timestamp DESC);

-- Options flow indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_flow_asset_time 
    ON options_flow (asset_id, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_flow_expiry 
    ON options_flow (expiry DESC, asset_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_flow_strike_type 
    ON options_flow (asset_id, strike, option_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_flow_unusual_activity 
    ON options_flow (unusual_activity, timestamp DESC) WHERE unusual_activity = true;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_flow_large_trades 
    ON options_flow (large_trade, volume DESC) WHERE large_trade = true;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_flow_iv_rank 
    ON options_flow (iv_rank DESC, asset_id) WHERE iv_rank IS NOT NULL;

-- User watchlist indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_watchlist_user 
    ON user_watchlist (user_id, display_order);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_watchlist_symbol 
    ON user_watchlist (symbol, custom_weight DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_watchlist_options_tracked 
    ON user_watchlist (user_id, is_options_tracked) WHERE is_options_tracked = true;

-- Virtual trades indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_virtual_trades_user 
    ON virtual_trades (user_id, entry_date DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_virtual_trades_symbol 
    ON virtual_trades (underlying_symbol, entry_date DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_virtual_trades_strategy 
    ON virtual_trades (strategy_type, entry_date DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_virtual_trades_status 
    ON virtual_trades (status, user_id) WHERE status = 'OPEN';
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_virtual_trades_performance 
    ON virtual_trades (pnl_percentage DESC, entry_date DESC) WHERE pnl_percentage IS NOT NULL;

-- Brokerage accounts indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brokerage_accounts_user 
    ON brokerage_accounts (user_id, connection_status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brokerage_accounts_broker 
    ON brokerage_accounts (broker_name, is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brokerage_accounts_sync 
    ON brokerage_accounts (last_sync DESC) WHERE connection_status = 'connected';

-- Options strategies indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_strategies_asset 
    ON options_strategies (asset_id, created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_strategies_type 
    ON options_strategies (strategy_type, confidence_score DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_strategies_risk 
    ON options_strategies (risk_level, expected_return DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_strategies_active 
    ON options_strategies (is_active, expires_at) WHERE is_active = true;

-- Composite indexes for common queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_flow_symbol_expiry_strike 
    ON options_flow (asset_id, expiry, strike, option_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_virtual_trades_user_strategy_status 
    ON virtual_trades (user_id, strategy_type, status);

-- ====================================
-- 5. PARTITIONING TRIGGERS
-- ====================================

-- Function to create market data partitions automatically
CREATE OR REPLACE FUNCTION create_market_data_partition()
RETURNS TRIGGER AS $$
DECLARE
    partition_date TEXT;
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    -- Extract year-month from timestamp
    partition_date := to_char(NEW.timestamp, 'YYYY_MM');
    partition_name := 'market_data_' || partition_date;
    
    -- Calculate partition bounds
    start_date := date_trunc('month', NEW.timestamp)::DATE;
    end_date := (start_date + INTERVAL '1 month')::DATE;
    
    -- Check if partition exists
    IF NOT EXISTS (
        SELECT 1 FROM pg_class WHERE relname = partition_name
    ) THEN
        -- Create partition
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF market_data FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
        
        -- Create indexes on new partition
        EXECUTE format(
            'CREATE INDEX %I ON %I (asset_id, timestamp DESC)',
            'idx_' || partition_name || '_asset_time', partition_name
        );
        
        RAISE NOTICE 'Created partition % for date range % to %', 
            partition_name, start_date, end_date;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to create options flow partitions automatically
CREATE OR REPLACE FUNCTION create_options_partition()
RETURNS TRIGGER AS $$
DECLARE
    partition_quarter TEXT;
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
    quarter_num INTEGER;
    year_num INTEGER;
BEGIN
    -- Extract year and quarter from expiry
    year_num := EXTRACT(YEAR FROM NEW.expiry);
    quarter_num := EXTRACT(QUARTER FROM NEW.expiry);
    
    partition_quarter := year_num || '_q' || quarter_num;
    partition_name := 'options_flow_' || partition_quarter;
    
    -- Calculate quarter bounds
    start_date := make_date(year_num, (quarter_num - 1) * 3 + 1, 1);
    end_date := make_date(year_num, quarter_num * 3 + 1, 1);
    
    -- Check if partition exists
    IF NOT EXISTS (
        SELECT 1 FROM pg_class WHERE relname = partition_name
    ) THEN
        -- Create partition
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF options_flow FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
        
        -- Create indexes on new partition
        EXECUTE format(
            'CREATE INDEX %I ON %I (asset_id, timestamp DESC)',
            'idx_' || partition_name || '_asset_time', partition_name
        );
        
        RAISE NOTICE 'Created options partition % for date range % to %', 
            partition_name, start_date, end_date;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic partitioning
CREATE TRIGGER market_data_partition_trigger
    BEFORE INSERT ON market_data
    FOR EACH ROW EXECUTE FUNCTION create_market_data_partition();

CREATE TRIGGER options_flow_partition_trigger
    BEFORE INSERT ON options_flow
    FOR EACH ROW EXECUTE FUNCTION create_options_partition();

-- ====================================
-- 6. ROW LEVEL SECURITY POLICIES
-- ====================================

-- Enable RLS on user-specific tables
ALTER TABLE user_watchlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE virtual_trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE brokerage_accounts ENABLE ROW LEVEL SECURITY;

-- RLS policies for user watchlist
CREATE POLICY user_watchlist_policy ON user_watchlist
    FOR ALL TO authenticated_users
    USING (user_id = current_setting('app.current_user_id')::UUID);

-- RLS policies for virtual trades
CREATE POLICY virtual_trades_policy ON virtual_trades
    FOR ALL TO authenticated_users
    USING (user_id = current_setting('app.current_user_id')::UUID);

-- RLS policies for brokerage accounts
CREATE POLICY brokerage_accounts_policy ON brokerage_accounts
    FOR ALL TO authenticated_users
    USING (user_id = current_setting('app.current_user_id')::UUID);

-- ====================================
-- 7. UPDATED_AT TRIGGERS
-- ====================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at triggers
CREATE TRIGGER assets_updated_at BEFORE UPDATE ON assets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER user_watchlist_updated_at BEFORE UPDATE ON user_watchlist
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER virtual_trades_updated_at BEFORE UPDATE ON virtual_trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER brokerage_accounts_updated_at BEFORE UPDATE ON brokerage_accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ====================================
-- 8. MONITORING AND ANALYTICS VIEWS
-- ====================================

-- View for options flow analytics
CREATE OR REPLACE VIEW options_analytics AS
SELECT 
    a.symbol,
    a.name,
    COUNT(*) as total_options_records,
    SUM(volume) as total_volume,
    AVG(implied_volatility) as avg_iv,
    AVG(iv_rank) as avg_iv_rank,
    COUNT(*) FILTER (WHERE unusual_activity) as unusual_activity_count,
    COUNT(*) FILTER (WHERE large_trade) as large_trade_count,
    MAX(timestamp) as last_update
FROM options_flow of
JOIN assets a ON of.asset_id = a.id
WHERE of.timestamp >= NOW() - INTERVAL '1 day'
GROUP BY a.symbol, a.name
ORDER BY total_volume DESC;

-- View for user trading performance
CREATE OR REPLACE VIEW user_trading_performance AS
SELECT 
    u.username,
    vt.strategy_type,
    COUNT(*) as total_trades,
    COUNT(*) FILTER (WHERE status = 'CLOSED' AND pnl > 0) as winning_trades,
    COUNT(*) FILTER (WHERE status = 'CLOSED' AND pnl <= 0) as losing_trades,
    ROUND(
        COUNT(*) FILTER (WHERE status = 'CLOSED' AND pnl > 0)::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE status = 'CLOSED'), 0) * 100, 2
    ) as win_rate,
    ROUND(AVG(pnl_percentage), 4) as avg_return_pct,
    ROUND(SUM(pnl), 2) as total_pnl,
    MAX(entry_date) as last_trade_date
FROM virtual_trades vt
JOIN users u ON vt.user_id = u.id
WHERE vt.entry_date >= NOW() - INTERVAL '30 days'
GROUP BY u.username, vt.strategy_type
ORDER BY total_pnl DESC;

-- View for market data summary
CREATE OR REPLACE VIEW market_summary AS
SELECT 
    a.symbol,
    a.name,
    md.close as current_price,
    md.volume as current_volume,
    md.implied_volatility as current_iv,
    md.iv_rank,
    LAG(md.close) OVER (PARTITION BY a.symbol ORDER BY md.timestamp) as prev_close,
    ROUND(
        ((md.close / LAG(md.close) OVER (PARTITION BY a.symbol ORDER BY md.timestamp)) - 1) * 100, 2
    ) as price_change_pct,
    md.timestamp as last_update
FROM market_data md
JOIN assets a ON md.asset_id = a.id
WHERE md.timestamp >= NOW() - INTERVAL '1 hour'
AND md.timestamp = (
    SELECT MAX(timestamp) 
    FROM market_data md2 
    WHERE md2.asset_id = md.asset_id
)
ORDER BY a.symbol;

-- ====================================
-- 9. DATA CLEANUP FUNCTIONS
-- ====================================

-- Function to clean old partitions
CREATE OR REPLACE FUNCTION cleanup_old_partitions()
RETURNS INTEGER AS $$
DECLARE
    partition_record RECORD;
    cleanup_date DATE;
    dropped_count INTEGER := 0;
BEGIN
    -- Clean partitions older than 2 years
    cleanup_date := CURRENT_DATE - INTERVAL '2 years';
    
    -- Drop old market data partitions
    FOR partition_record IN
        SELECT schemaname, tablename 
        FROM pg_tables 
        WHERE tablename LIKE 'market_data_%'
        AND tablename < 'market_data_' || to_char(cleanup_date, 'YYYY_MM')
    LOOP
        EXECUTE 'DROP TABLE IF EXISTS ' || partition_record.tablename || ' CASCADE';
        dropped_count := dropped_count + 1;
        RAISE NOTICE 'Dropped partition: %', partition_record.tablename;
    END LOOP;
    
    -- Drop old options partitions
    FOR partition_record IN
        SELECT schemaname, tablename 
        FROM pg_tables 
        WHERE tablename LIKE 'options_flow_%'
        AND tablename < 'options_flow_' || to_char(cleanup_date, 'YYYY_"q"Q')
    LOOP
        EXECUTE 'DROP TABLE IF EXISTS ' || partition_record.tablename || ' CASCADE';
        dropped_count := dropped_count + 1;
        RAISE NOTICE 'Dropped options partition: %', partition_record.tablename;
    END LOOP;
    
    RETURN dropped_count;
END;
$$ LANGUAGE plpgsql;

-- ====================================
-- 10. PERFORMANCE OPTIMIZATION
-- ====================================

-- Analyze tables for query optimization
ANALYZE assets;
ANALYZE market_data;
ANALYZE options_flow;
ANALYZE user_watchlist;
ANALYZE virtual_trades;
ANALYZE brokerage_accounts;
ANALYZE options_strategies;

-- Set table statistics targets for better query planning
ALTER TABLE market_data ALTER COLUMN timestamp SET STATISTICS 1000;
ALTER TABLE options_flow ALTER COLUMN timestamp SET STATISTICS 1000;
ALTER TABLE options_flow ALTER COLUMN asset_id SET STATISTICS 1000;

-- Create partial indexes for better performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_flow_recent 
    ON options_flow (asset_id, timestamp DESC) 
    WHERE timestamp >= NOW() - INTERVAL '30 days';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_virtual_trades_recent_open 
    ON virtual_trades (user_id, entry_date DESC) 
    WHERE status = 'OPEN' AND entry_date >= NOW() - INTERVAL '90 days';

-- ====================================
-- 11. COMMENTS FOR DOCUMENTATION
-- ====================================

COMMENT ON TABLE assets IS 'Enhanced assets table with options trading support';
COMMENT ON TABLE market_data IS 'Enhanced market data with options-specific fields and partitioning';
COMMENT ON TABLE options_flow IS 'Real-time options flow data with Greeks and analytics';
COMMENT ON TABLE user_watchlist IS 'Custom user watchlists with Mag-7 weighting support';
COMMENT ON TABLE virtual_trades IS 'Virtual trading journal for strategy performance tracking';
COMMENT ON TABLE brokerage_accounts IS 'Brokerage account integration for real trading';
COMMENT ON TABLE options_strategies IS 'AI-generated options strategies catalog';

COMMENT ON COLUMN options_flow.moneyness IS 'Underlying price / strike price ratio';
COMMENT ON COLUMN options_flow.volume_oi_ratio IS 'Volume to open interest ratio for unusual activity detection';
COMMENT ON COLUMN options_flow.unusual_activity IS 'Flag for unusual options activity (volume/OI > 0.5)';
COMMENT ON COLUMN user_watchlist.custom_weight IS 'User-defined weighting for Mag-7 prioritization';
COMMENT ON COLUMN virtual_trades.strategy_details IS 'JSON structure containing specific strategy parameters';

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO mip_application;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO mip_application;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO mip_application;

-- Final optimization
VACUUM ANALYZE;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Enhanced MIP database schema v2.0 created successfully!';
    RAISE NOTICE 'Tables created: assets, market_data, options_flow, user_watchlist, virtual_trades, brokerage_accounts, options_strategies';
    RAISE NOTICE 'Features: Partitioning, RLS, Auto-partitioning triggers, Performance indexes, Analytics views';
    RAISE NOTICE 'Options intelligence: Greeks tracking, IV analytics, Unusual activity detection';
    RAISE NOTICE 'User customization: Mag-7 weighting, Virtual trading, Brokerage integration';
END $$;

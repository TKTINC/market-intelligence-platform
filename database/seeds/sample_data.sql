-- Sample data for Enhanced MIP Database
-- Provides realistic test data for development and testing

-- Insert sample assets (Mag-7 + major ETFs)
INSERT INTO assets (symbol, name, asset_type, exchange, sector, industry, options_enabled, avg_daily_volume, beta, market_cap) VALUES
('AAPL', 'Apple Inc.', 'stock', 'NASDAQ', 'Technology', 'Consumer Electronics', true, 50000000, 1.2, 3000000000000),
('MSFT', 'Microsoft Corporation', 'stock', 'NASDAQ', 'Technology', 'Software', true, 30000000, 1.1, 2800000000000),
('GOOGL', 'Alphabet Inc.', 'stock', 'NASDAQ', 'Technology', 'Internet Services', true, 25000000, 1.3, 1700000000000),
('AMZN', 'Amazon.com Inc.', 'stock', 'NASDAQ', 'Consumer Discretionary', 'E-commerce', true, 35000000, 1.4, 1500000000000),
('NVDA', 'NVIDIA Corporation', 'stock', 'NASDAQ', 'Technology', 'Semiconductors', true, 45000000, 1.8, 1200000000000),
('TSLA', 'Tesla Inc.', 'stock', 'NASDAQ', 'Consumer Discretionary', 'Electric Vehicles', true, 75000000, 2.1, 800000000000),
('META', 'Meta Platforms Inc.', 'stock', 'NASDAQ', 'Technology', 'Social Media', true, 20000000, 1.3, 750000000000),
('SPY', 'SPDR S&P 500 ETF Trust', 'etf', 'ARCA', 'Diversified', 'Broad Market ETF', true, 80000000, 1.0, 400000000000),
('QQQ', 'Invesco QQQ Trust', 'etf', 'NASDAQ', 'Technology', 'Tech ETF', true, 40000000, 1.2, 200000000000),
('IWM', 'iShares Russell 2000 ETF', 'etf', 'ARCA', 'Diversified', 'Small Cap ETF', true, 25000000, 1.3, 50000000000);

-- Insert sample users with different profiles
INSERT INTO users (email, username, first_name, last_name, risk_tolerance, experience_level, subscription_tier) VALUES
('demo@mip.com', 'demo_user', 'Demo', 'User', 'moderate', 'intermediate', 'premium'),
('trader1@example.com', 'options_trader', 'Alex', 'Smith', 'aggressive', 'advanced', 'premium'),
('conservative@example.com', 'safe_investor', 'Mary', 'Johnson', 'conservative', 'beginner', 'basic'),
('pro_trader@example.com', 'market_pro', 'David', 'Wilson', 'aggressive', 'expert', 'enterprise');

-- Insert sample user watchlists with custom Mag-7 weighting
INSERT INTO user_watchlist (user_id, symbol, custom_weight, display_order, is_options_tracked, position_size_preference, position_sizing_type, price_alert_enabled, price_alert_threshold, iv_alert_enabled, iv_alert_threshold) VALUES
-- Demo user's custom Mag-7 weighting
((SELECT id FROM users WHERE username = 'demo_user'), 'NVDA', 2.5, 1, true, 10000, 'dollar', true, 5.0, true, 80),
((SELECT id FROM users WHERE username = 'demo_user'), 'AAPL', 2.0, 2, true, 8000, 'dollar', true, 3.0, false, null),
((SELECT id FROM users WHERE username = 'demo_user'), 'TSLA', 1.8, 3, true, 6000, 'dollar', true, 7.0, true, 75),
((SELECT id FROM users WHERE username = 'demo_user'), 'MSFT', 1.5, 4, true, 7000, 'dollar', false, null, false, null),
((SELECT id FROM users WHERE username = 'demo_user'), 'GOOGL', 1.3, 5, false, 5000, 'dollar', false, null, false, null),
((SELECT id FROM users WHERE username = 'demo_user'), 'META', 1.2, 6, true, 4000, 'dollar', true, 6.0, true, 70),
((SELECT id FROM users WHERE username = 'demo_user'), 'AMZN', 1.0, 7, false, 3000, 'dollar', false, null, false, null),
((SELECT id FROM users WHERE username = 'demo_user'), 'SPY', 0.8, 8, true, 15000, 'dollar', false, null, false, null),

-- Options trader's aggressive weighting
((SELECT id FROM users WHERE username = 'options_trader'), 'NVDA', 3.0, 1, true, 20000, 'dollar', true, 3.0, true, 85),
((SELECT id FROM users WHERE username = 'options_trader'), 'TSLA', 2.8, 2, true, 18000, 'dollar', true, 5.0, true, 80),
((SELECT id FROM users WHERE username = 'options_trader'), 'AAPL', 2.2, 3, true, 15000, 'dollar', true, 2.0, true, 75),
((SELECT id FROM users WHERE username = 'options_trader'), 'QQQ', 1.5, 4, true, 12000, 'dollar', false, null, true, 70),

-- Conservative investor's weighting
((SELECT id FROM users WHERE username = 'safe_investor'), 'AAPL', 1.2, 1, false, 5000, 'dollar', true, 2.0, false, null),
((SELECT id FROM users WHERE username = 'safe_investor'), 'MSFT', 1.1, 2, false, 4500, 'dollar', true, 2.0, false, null),
((SELECT id FROM users WHERE username = 'safe_investor'), 'SPY', 1.0, 3, false, 10000, 'dollar', false, null, false, null);

-- Insert sample market data (recent)
INSERT INTO market_data (asset_id, timestamp, open, high, low, close, volume, implied_volatility, iv_rank, bid, ask, source) VALUES
-- AAPL data
((SELECT id FROM assets WHERE symbol = 'AAPL'), NOW() - INTERVAL '1 hour', 150.25, 151.80, 149.90, 151.45, 45231000, 0.28, 65, 151.44, 151.46, 'alphavantage'),
((SELECT id FROM assets WHERE symbol = 'AAPL'), NOW() - INTERVAL '2 hours', 149.80, 150.50, 149.20, 150.25, 38742000, 0.29, 67, 150.24, 150.26, 'alphavantage'),

-- NVDA data
((SELECT id FROM assets WHERE symbol = 'NVDA'), NOW() - INTERVAL '1 hour', 285.50, 289.20, 284.10, 287.80, 52847000, 0.45, 78, 287.78, 287.82, 'alphavantage'),
((SELECT id FROM assets WHERE symbol = 'NVDA'), NOW() - INTERVAL '2 hours', 282.90, 286.40, 281.50, 285.50, 48932000, 0.46, 79, 285.48, 285.52, 'alphavantage'),

-- TSLA data
((SELECT id FROM assets WHERE symbol = 'TSLA'), NOW() - INTERVAL '1 hour', 195.40, 198.50, 194.20, 197.30, 68254000, 0.52, 82, 197.28, 197.32, 'alphavantage'),
((SELECT id FROM assets WHERE symbol = 'TSLA'), NOW() - INTERVAL '2 hours', 192.80, 196.10, 191.90, 195.40, 72109000, 0.53, 83, 195.38, 195.42, 'alphavantage');

-- Insert sample options flow data
INSERT INTO options_flow (asset_id, timestamp, option_symbol, expiry, strike, option_type, volume, open_interest, bid, ask, last_price, implied_volatility, delta, gamma, theta, vega, underlying_price, time_to_expiry, iv_rank, source, trade_type) VALUES
-- AAPL options
((SELECT id FROM assets WHERE symbol = 'AAPL'), NOW() - INTERVAL '30 minutes', 'AAPL240315C00150000', '2024-03-15', 150.00, 'call', 2500, 15420, 2.85, 2.95, 2.90, 0.32, 0.52, 0.045, -0.08, 0.15, 151.45, 0.0822, 68, 'cboe', 'buy'),
((SELECT id FROM assets WHERE symbol = 'AAPL'), NOW() - INTERVAL '45 minutes', 'AAPL240315P00145000', '2024-03-15', 145.00, 'put', 1800, 8930, 1.25, 1.35, 1.30, 0.28, -0.25, 0.038, -0.06, 0.12, 151.45, 0.0822, 58, 'cboe', 'sell'),

-- NVDA options (high IV)
((SELECT id FROM assets WHERE symbol = 'NVDA'), NOW() - INTERVAL '20 minutes', 'NVDA240322C00290000', '2024-03-22', 290.00, 'call', 5200, 22150, 8.50, 8.80, 8.65, 0.48, 0.65, 0.025, -0.12, 0.28, 287.80, 0.1370, 85, 'cboe', 'buy'),
((SELECT id FROM assets WHERE symbol = 'NVDA'), NOW() - INTERVAL '35 minutes', 'NVDA240329P00280000', '2024-03-29', 280.00, 'put', 3100, 18760, 5.20, 5.45, 5.32, 0.44, -0.35, 0.022, -0.10, 0.25, 287.80, 0.1589, 78, 'cboe', 'buy'),

-- TSLA options (unusual activity)
((SELECT id FROM assets WHERE symbol = 'TSLA'), NOW() - INTERVAL '15 minutes', 'TSLA240315C00200000', '2024-03-15', 200.00, 'call', 8500, 12400, 3.40, 3.60, 3.50, 0.55, 0.45, 0.032, -0.15, 0.22, 197.30, 0.0822, 88, 'cboe', 'sweep'),
((SELECT id FROM assets WHERE symbol = 'TSLA'), NOW() - INTERVAL '25 minutes', 'TSLA240408P00190000', '2024-04-08', 190.00, 'put', 4200, 9850, 6.80, 7.10, 6.95, 0.58, -0.42, 0.028, -0.18, 0.20, 197.30, 0.1507, 92, 'cboe', 'buy');

-- Insert sample virtual trades
INSERT INTO virtual_trades (user_id, strategy_type, underlying_symbol, strategy_details, entry_price, entry_date, pnl, pnl_percentage, max_profit, max_loss, probability_profit, break_even_points, status, entry_iv_rank, entry_underlying_price, entry_dte, notes) VALUES
-- Demo user's trades
((SELECT id FROM users WHERE username = 'demo_user'), 'COVERED_CALL', 'AAPL', 
 '{"long_stock": 100, "short_call": {"strike": 155, "expiry": "2024-03-22", "premium": 2.50}}'::jsonb, 
 2.50, NOW() - INTERVAL '5 days', 125.00, 5.0, 250.00, -1500.00, 68.5, '[152.50]'::jsonb, 'OPEN', 
 65, 150.00, 17, 'Conservative covered call on AAPL position'),

((SELECT id FROM users WHERE username = 'demo_user'), 'IRON_CONDOR', 'NVDA', 
 '{"short_put": {"strike": 270, "premium": 3.20}, "long_put": {"strike": 260, "premium": 1.80}, "short_call": {"strike": 300, "premium": 4.10}, "long_call": {"strike": 310, "premium": 2.30}}'::jsonb, 
 3.20, NOW() - INTERVAL '3 days', 210.00, 6.7, 320.00, -680.00, 72.3, '[273.20, 296.80]'::jsonb, 'OPEN', 
 78, 285.00, 19, 'High IV iron condor on NVDA'),

-- Options trader's trades
((SELECT id FROM users WHERE username = 'options_trader'), 'STRANGLE', 'TSLA', 
 '{"long_call": {"strike": 210, "premium": 8.50}, "long_put": {"strike": 180, "premium": 6.20}}'::jsonb, 
 14.70, NOW() - INTERVAL '7 days', -245.00, -16.7, 1000.00, -1470.00, 45.2, '[165.30, 224.70]'::jsonb, 'OPEN', 
 85, 195.00, 24, 'Long volatility play before earnings'),

((SELECT id FROM users WHERE username = 'options_trader'), 'CASH_SECURED_PUT', 'AAPL', 
 '{"short_put": {"strike": 145, "expiry": "2024-03-08", "premium": 1.85}}'::jsonb, 
 1.85, NOW() - INTERVAL '10 days', 185.00, 100.0, 185.00, -1435.00, 78.5, '[143.15]'::jsonb, 'CLOSED', 
 62, 148.00, 14, 'Expired worthless - full profit'),

-- Conservative investor's trade
((SELECT id FROM users WHERE username = 'safe_investor'), 'COVERED_CALL', 'MSFT', 
 '{"long_stock": 100, "short_call": {"strike": 420, "expiry": "2024-04-19", "premium": 3.80}}'::jsonb, 
 3.80, NOW() - INTERVAL '2 days', 95.00, 2.5, 380.00, -2000.00, 81.2, '[423.80]'::jsonb, 'OPEN', 
 45, 415.00, 48, 'Conservative income generation on MSFT holdings');

-- Insert sample brokerage accounts
INSERT INTO brokerage_accounts (user_id, broker_name, broker_account_id, account_type, account_nickname, connection_status, permissions, cash_balance, buying_power, total_value, last_sync) VALUES
((SELECT id FROM users WHERE username = 'demo_user'), 'TD_AMERITRADE', 'TDA123456789', 'margin', 'Main Trading Account', 'connected', 
 '{"read_positions": true, "execute_trades": false, "read_orders": true}'::jsonb, 25000.00, 75000.00, 125000.00, NOW() - INTERVAL '1 hour'),

((SELECT id FROM users WHERE username = 'options_trader'), 'INTERACTIVE_BROKERS', 'IB987654321', 'margin', 'Options Trading', 'connected', 
 '{"read_positions": true, "execute_trades": true, "read_orders": true, "place_orders": true}'::jsonb, 50000.00, 200000.00, 275000.00, NOW() - INTERVAL '30 minutes'),

((SELECT id FROM users WHERE username = 'safe_investor'), 'CHARLES_SCHWAB', 'SCHW456789123', 'ira', 'Retirement Account', 'connected', 
 '{"read_positions": true, "execute_trades": false, "read_orders": true}'::jsonb, 15000.00, 15000.00, 85000.00, NOW() - INTERVAL '2 hours');

-- Insert sample options strategies
INSERT INTO options_strategies (asset_id, strategy_name, strategy_type, strategy_details, underlying_price, current_iv_rank, max_profit, max_loss, probability_profit, expected_return, risk_level, capital_required, net_delta, net_gamma, net_theta, net_vega, market_regime, recommended_dte_min, recommended_dte_max, backtest_score, confidence_score) VALUES
-- AAPL strategies
((SELECT id FROM assets WHERE symbol = 'AAPL'), 'Conservative Income', 'COVERED_CALL', 
 '{"action": "sell", "strike": 155, "expiry": "2024-04-19", "premium": 3.20, "underlying_shares": 100}'::jsonb, 
 151.45, 65, 320.00, -1500.00, 72.5, 2.1, 'low', 15145.00, 0.48, 0.02, -0.08, -0.12, 'neutral', 14, 45, 78.3, 85.2),

-- NVDA strategies
((SELECT id FROM assets WHERE symbol = 'NVDA'), 'High IV Iron Condor', 'IRON_CONDOR', 
 '{"short_put": {"strike": 275, "premium": 4.50}, "long_put": {"strike": 265, "premium": 2.30}, "short_call": {"strike": 300, "premium": 5.20}, "long_call": {"strike": 310, "premium": 2.80}}'::jsonb, 
 287.80, 78, 460.00, -540.00, 68.7, 4.6, 'medium', 1000.00, 0.05, 0.01, -0.15, -0.08, 'neutral', 21, 35, 72.1, 88.9),

-- TSLA strategies
((SELECT id FROM assets WHERE symbol = 'TSLA'), 'Volatility Expansion', 'STRANGLE', 
 '{"long_call": {"strike": 210, "premium": 9.20}, "long_put": {"strike": 180, "premium": 7.40}}'::jsonb, 
 197.30, 82, 2000.00, -1660.00, 42.8, 15.3, 'high', 1660.00, 0.15, 0.08, -0.25, 0.45, 'volatile', 14, 28, 58.4, 76.3);

-- Display success message
SELECT 'Sample data inserted successfully!' as message,
       (SELECT COUNT(*) FROM assets) as assets_count,
       (SELECT COUNT(*) FROM users) as users_count,
       (SELECT COUNT(*) FROM user_watchlist) as watchlist_entries,
       (SELECT COUNT(*) FROM market_data) as market_data_records,
       (SELECT COUNT(*) FROM options_flow) as options_records,
       (SELECT COUNT(*) FROM virtual_trades) as virtual_trades_count,
       (SELECT COUNT(*) FROM brokerage_accounts) as brokerage_accounts_count,
       (SELECT COUNT(*) FROM options_strategies) as strategies_count;

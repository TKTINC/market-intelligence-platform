-- Advanced Partition Management for MIP Database
-- Handles automatic partition creation, maintenance, and cleanup

-- ====================================
-- 1. ADVANCED PARTITION CREATION
-- ====================================

-- Enhanced function to create market data partitions with optimization
CREATE OR REPLACE FUNCTION create_market_data_partition_enhanced(target_date DATE)
RETURNS TEXT AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
    index_name TEXT;
    result TEXT;
BEGIN
    -- Generate partition name
    partition_name := 'market_data_' || to_char(target_date, 'YYYY_MM');
    
    -- Calculate partition bounds (monthly)
    start_date := date_trunc('month', target_date)::DATE;
    end_date := (start_date + INTERVAL '1 month')::DATE;
    
    -- Check if partition already exists
    IF EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = partition_name AND n.nspname = 'public'
    ) THEN
        RETURN 'Partition ' || partition_name || ' already exists';
    END IF;
    
    -- Create the partition
    EXECUTE format(
        'CREATE TABLE %I PARTITION OF market_data FOR VALUES FROM (%L) TO (%L)',
        partition_name, start_date, end_date
    );
    
    -- Create optimized indexes
    index_name := 'idx_' || partition_name || '_asset_time';
    EXECUTE format(
        'CREATE INDEX %I ON %I (asset_id, timestamp DESC)',
        index_name, partition_name
    );
    
    index_name := 'idx_' || partition_name || '_timestamp';
    EXECUTE format(
        'CREATE INDEX %I ON %I USING BRIN (timestamp)',
        index_name, partition_name
    );
    
    index_name := 'idx_' || partition_name || '_source';
    EXECUTE format(
        'CREATE INDEX %I ON %I (source) WHERE source IS NOT NULL',
        index_name, partition_name
    );
    
    -- Set table-specific settings for performance
    EXECUTE format(
        'ALTER TABLE %I SET (fillfactor = 90, autovacuum_vacuum_scale_factor = 0.1)',
        partition_name
    );
    
    -- Add table comment
    EXECUTE format(
        'COMMENT ON TABLE %I IS ''Market data partition for %s to %s''',
        partition_name, start_date, end_date
    );
    
    result := 'Created partition ' || partition_name || ' for period ' || start_date || ' to ' || end_date;
    RAISE NOTICE '%', result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Enhanced function to create options flow partitions by expiry quarters
CREATE OR REPLACE FUNCTION create_options_partition_enhanced(target_date DATE)
RETURNS TEXT AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
    quarter_num INTEGER;
    year_num INTEGER;
    index_name TEXT;
    result TEXT;
BEGIN
    -- Calculate quarter information
    year_num := EXTRACT(YEAR FROM target_date);
    quarter_num := EXTRACT(QUARTER FROM target_date);
    
    -- Generate partition name
    partition_name := 'options_flow_' || year_num || '_q' || quarter_num;
    
    -- Calculate quarter bounds
    start_date := make_date(year_num, (quarter_num - 1) * 3 + 1, 1);
    end_date := start_date + INTERVAL '3 months';
    
    -- Check if partition already exists
    IF EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = partition_name AND n.nspname = 'public'
    ) THEN
        RETURN 'Partition ' || partition_name || ' already exists';
    END IF;
    
    -- Create the partition
    EXECUTE format(
        'CREATE TABLE %I PARTITION OF options_flow FOR VALUES FROM (%L) TO (%L)',
        partition_name, start_date, end_date
    );
    
    -- Create comprehensive indexes for options queries
    
    -- Primary lookup index
    index_name := 'idx_' || partition_name || '_asset_time';
    EXECUTE format(
        'CREATE INDEX %I ON %I (asset_id, timestamp DESC)',
        index_name, partition_name
    );
    
    -- Options chain index
    index_name := 'idx_' || partition_name || '_chain';
    EXECUTE format(
        'CREATE INDEX %I ON %I (asset_id, expiry, strike, option_type)',
        index_name, partition_name
    );
    
    -- Unusual activity index
    index_name := 'idx_' || partition_name || '_unusual';
    EXECUTE format(
        'CREATE INDEX %I ON %I (unusual_activity, volume DESC) WHERE unusual_activity = true',
        index_name, partition_name
    );
    
    -- IV rank index for high IV options
    index_name := 'idx_' || partition_name || '_iv_rank';
    EXECUTE format(
        'CREATE INDEX %I ON %I (iv_rank DESC, asset_id) WHERE iv_rank >= 70',
        index_name, partition_name
    );
    
    -- Large trades index
    index_name := 'idx_' || partition_name || '_large_trades';
    EXECUTE format(
        'CREATE INDEX %I ON %I (large_trade, timestamp DESC) WHERE large_trade = true',
        index_name, partition_name
    );
    
    -- Expiry-based index for options chain queries
    index_name := 'idx_' || partition_name || '_expiry_strike';
    EXECUTE format(
        'CREATE INDEX %I ON %I (expiry, asset_id, strike)',
        index_name, partition_name
    );
    
    -- Set table-specific settings
    EXECUTE format(
        'ALTER TABLE %I SET (fillfactor = 85, autovacuum_vacuum_scale_factor = 0.05)',
        partition_name
    );
    
    -- Add table comment
    EXECUTE format(
        'COMMENT ON TABLE %I IS ''Options flow partition for %s Q%s (%s to %s)''',
        partition_name, year_num, quarter_num, start_date, end_date
    );
    
    result := 'Created options partition ' || partition_name || ' for ' || year_num || ' Q' || quarter_num;
    RAISE NOTICE '%', result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- ====================================
-- 2. AUTOMATIC PARTITION MAINTENANCE
-- ====================================

-- Function to create future partitions proactively
CREATE OR REPLACE FUNCTION create_future_partitions()
RETURNS TABLE(partition_type TEXT, partition_name TEXT, result TEXT) AS $$
DECLARE
    current_month DATE;
    target_month DATE;
    current_quarter DATE;
    target_quarter DATE;
    i INTEGER;
BEGIN
    -- Create market data partitions for next 3 months
    current_month := date_trunc('month', CURRENT_DATE);
    
    FOR i IN 0..2 LOOP
        target_month := current_month + (i || ' months')::INTERVAL;
        
        SELECT 'market_data', 'market_data_' || to_char(target_month, 'YYYY_MM'), 
               create_market_data_partition_enhanced(target_month)
        INTO partition_type, partition_name, result;
        
        RETURN NEXT;
    END LOOP;
    
    -- Create options partitions for next 2 quarters
    current_quarter := date_trunc('quarter', CURRENT_DATE);
    
    FOR i IN 0..1 LOOP
        target_quarter := current_quarter + (i * 3 || ' months')::INTERVAL;
        
        SELECT 'options_flow', 
               'options_flow_' || EXTRACT(YEAR FROM target_quarter) || '_q' || EXTRACT(QUARTER FROM target_quarter),
               create_options_partition_enhanced(target_quarter)
        INTO partition_type, partition_name, result;
        
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- ====================================
-- 3. PARTITION CLEANUP AND ARCHIVAL
-- ====================================

-- Enhanced cleanup function with archival options
CREATE OR REPLACE FUNCTION cleanup_old_partitions_enhanced(
    retention_months INTEGER DEFAULT 24,
    archive_to_s3 BOOLEAN DEFAULT false
)
RETURNS TABLE(action TEXT, partition_name TEXT, status TEXT, details TEXT) AS $$
DECLARE
    partition_record RECORD;
    cleanup_date DATE;
    archive_date DATE;
    dropped_count INTEGER := 0;
    archived_count INTEGER := 0;
    partition_size TEXT;
    table_oid OID;
BEGIN
    -- Calculate cleanup and archive dates
    cleanup_date := CURRENT_DATE - (retention_months || ' months')::INTERVAL;
    archive_date := CURRENT_DATE - ((retention_months - 6) || ' months')::INTERVAL;
    
    RAISE NOTICE 'Starting partition cleanup: retention=%s months, cleanup_date=%s', 
                 retention_months, cleanup_date;
    
    -- Handle market data partitions
    FOR partition_record IN
        SELECT schemaname, tablename, 
               substring(tablename from 'market_data_(\d{4}_\d{2})') as date_part
        FROM pg_tables 
        WHERE tablename LIKE 'market_data_____\_\_\_' 
        AND schemaname = 'public'
    LOOP
        -- Parse date from partition name
        BEGIN
            DECLARE
                partition_date DATE;
            BEGIN
                partition_date := to_date(partition_record.date_part, 'YYYY_MM');
                
                -- Get partition size for reporting
                SELECT oid INTO table_oid FROM pg_class WHERE relname = partition_record.tablename;
                SELECT pg_size_pretty(pg_total_relation_size(table_oid)) INTO partition_size;
                
                IF partition_date < cleanup_date THEN
                    -- Archive before dropping if requested
                    IF archive_to_s3 AND partition_date >= archive_date THEN
                        -- In production, this would trigger S3 export
                        SELECT 'ARCHIVE', partition_record.tablename, 'SUCCESS', 
                               'Archived partition (size: ' || partition_size || ')'
                        INTO action, partition_name, status, details;
                        
                        archived_count := archived_count + 1;
                        RETURN NEXT;
                    END IF;
                    
                    -- Drop the partition
                    EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(partition_record.tablename) || ' CASCADE';
                    
                    SELECT 'DROP', partition_record.tablename, 'SUCCESS', 
                           'Dropped partition (was size: ' || partition_size || ')'
                    INTO action, partition_name, status, details;
                    
                    dropped_count := dropped_count + 1;
                    RETURN NEXT;
                    
                ELSIF partition_date < archive_date AND archive_to_s3 THEN
                    -- Candidate for archival
                    SELECT 'ARCHIVE_CANDIDATE', partition_record.tablename, 'PENDING', 
                           'Candidate for archival (size: ' || partition_size || ')'
                    INTO action, partition_name, status, details;
                    
                    RETURN NEXT;
                END IF;
            END;
        EXCEPTION
            WHEN others THEN
                SELECT 'ERROR', partition_record.tablename, 'FAILED', 
                       'Could not parse date from partition name: ' || SQLERRM
                INTO action, partition_name, status, details;
                RETURN NEXT;
        END;
    END LOOP;
    
    -- Handle options flow partitions
    FOR partition_record IN
        SELECT schemaname, tablename,
               substring(tablename from 'options_flow_(\d{4})_q(\d)') as year_part,
               substring(tablename from 'options_flow_\d{4}_q(\d)') as quarter_part
        FROM pg_tables 
        WHERE tablename LIKE 'options_flow_____\_q_' 
        AND schemaname = 'public'
    LOOP
        BEGIN
            DECLARE
                partition_date DATE;
                year_num INTEGER;
                quarter_num INTEGER;
            BEGIN
                year_num := partition_record.year_part::INTEGER;
                quarter_num := partition_record.quarter_part::INTEGER;
                partition_date := make_date(year_num, (quarter_num - 1) * 3 + 1, 1);
                
                -- Get partition size
                SELECT oid INTO table_oid FROM pg_class WHERE relname = partition_record.tablename;
                SELECT pg_size_pretty(pg_total_relation_size(table_oid)) INTO partition_size;
                
                IF partition_date < cleanup_date THEN
                    -- Archive before dropping if requested
                    IF archive_to_s3 AND partition_date >= archive_date THEN
                        SELECT 'ARCHIVE', partition_record.tablename, 'SUCCESS', 
                               'Archived options partition (size: ' || partition_size || ')'
                        INTO action, partition_name, status, details;
                        
                        archived_count := archived_count + 1;
                        RETURN NEXT;
                    END IF;
                    
                    -- Drop the partition
                    EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(partition_record.tablename) || ' CASCADE';
                    
                    SELECT 'DROP', partition_record.tablename, 'SUCCESS', 
                           'Dropped options partition (was size: ' || partition_size || ')'
                    INTO action, partition_name, status, details;
                    
                    dropped_count := dropped_count + 1;
                    RETURN NEXT;
                END IF;
            END;
        EXCEPTION
            WHEN others THEN
                SELECT 'ERROR', partition_record.tablename, 'FAILED', 
                       'Could not process options partition: ' || SQLERRM
                INTO action, partition_name, status, details;
                RETURN NEXT;
        END;
    END LOOP;
    
    -- Summary
    SELECT 'SUMMARY', 'cleanup_complete', 'SUCCESS', 
           format('Dropped: %s partitions, Archived: %s partitions', dropped_count, archived_count)
    INTO action, partition_name, status, details;
    RETURN NEXT;
    
    RAISE NOTICE 'Partition cleanup completed: dropped=%, archived=%', dropped_count, archived_count;
END;
$$ LANGUAGE plpgsql;

-- ====================================
-- 4. PARTITION STATISTICS AND MONITORING
-- ====================================

-- Function to get partition statistics
CREATE OR REPLACE FUNCTION get_partition_statistics()
RETURNS TABLE(
    table_type TEXT,
    partition_name TEXT,
    row_count BIGINT,
    size_pretty TEXT,
    size_bytes BIGINT,
    date_range TEXT,
    last_vacuum TIMESTAMPTZ,
    last_analyze TIMESTAMPTZ
) AS $$
BEGIN
    -- Market data partition stats
    RETURN QUERY
    SELECT 
        'market_data'::TEXT as table_type,
        c.relname::TEXT as partition_name,
        c.reltuples::BIGINT as row_count,
        pg_size_pretty(pg_total_relation_size(c.oid)) as size_pretty,
        pg_total_relation_size(c.oid) as size_bytes,
        pg_get_expr(c.relpartbound, c.oid) as date_range,
        s.last_vacuum,
        s.last_analyze
    FROM pg_class c
    JOIN pg_inherits i ON i.inhrelid = c.oid
    JOIN pg_class p ON p.oid = i.inhparent
    LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
    WHERE p.relname = 'market_data'
    AND c.relkind = 'r'
    
    UNION ALL
    
    -- Options flow partition stats
    SELECT 
        'options_flow'::TEXT as table_type,
        c.relname::TEXT as partition_name,
        c.reltuples::BIGINT as row_count,
        pg_size_pretty(pg_total_relation_size(c.oid)) as size_pretty,
        pg_total_relation_size(c.oid) as size_bytes,
        pg_get_expr(c.relpartbound, c.oid) as date_range,
        s.last_vacuum,
        s.last_analyze
    FROM pg_class c
    JOIN pg_inherits i ON i.inhrelid = c.oid
    JOIN pg_class p ON p.oid = i.inhparent
    LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
    WHERE p.relname = 'options_flow'
    AND c.relkind = 'r'
    
    ORDER BY table_type, partition_name;
END;
$$ LANGUAGE plpgsql;

-- ====================================
-- 5. MAINTENANCE SCHEDULER
-- ====================================

-- Function to perform comprehensive partition maintenance
CREATE OR REPLACE FUNCTION perform_partition_maintenance(
    create_future BOOLEAN DEFAULT true,
    cleanup_old BOOLEAN DEFAULT false,
    vacuum_partitions BOOLEAN DEFAULT true,
    analyze_partitions BOOLEAN DEFAULT true
)
RETURNS TABLE(operation TEXT, status TEXT, details TEXT) AS $$
DECLARE
    maintenance_start TIMESTAMPTZ;
    partition_record RECORD;
    operation_count INTEGER := 0;
BEGIN
    maintenance_start := NOW();
    
    SELECT 'MAINTENANCE_START', 'STARTED', 
           'Partition maintenance started at ' || maintenance_start
    INTO operation, status, details;
    RETURN NEXT;
    
    -- Create future partitions
    IF create_future THEN
        FOR partition_record IN 
            SELECT * FROM create_future_partitions()
        LOOP
            SELECT 'CREATE_FUTURE', 'SUCCESS', 
                   partition_record.partition_type || ': ' || partition_record.result
            INTO operation, status, details;
            RETURN NEXT;
            operation_count := operation_count + 1;
        END LOOP;
    END IF;
    
    -- Cleanup old partitions
    IF cleanup_old THEN
        FOR partition_record IN 
            SELECT * FROM cleanup_old_partitions_enhanced()
        LOOP
            SELECT 'CLEANUP', partition_record.status, 
                   partition_record.action || ': ' || partition_record.partition_name || ' - ' || partition_record.details
            INTO operation, status, details;
            RETURN NEXT;
            operation_count := operation_count + 1;
        END LOOP;
    END IF;
    
    -- Vacuum partitions
    IF vacuum_partitions THEN
        FOR partition_record IN
            SELECT relname 
            FROM pg_class c
            JOIN pg_inherits i ON i.inhrelid = c.oid
            JOIN pg_class p ON p.oid = i.inhparent
            WHERE p.relname IN ('market_data', 'options_flow')
            AND c.relkind = 'r'
        LOOP
            BEGIN
                EXECUTE 'VACUUM (ANALYZE) ' || quote_ident(partition_record.relname);
                
                SELECT 'VACUUM', 'SUCCESS', 
                       'Vacuumed partition: ' || partition_record.relname
                INTO operation, status, details;
                RETURN NEXT;
                
            EXCEPTION WHEN others THEN
                SELECT 'VACUUM', 'ERROR', 
                       'Failed to vacuum ' || partition_record.relname || ': ' || SQLERRM
                INTO operation, status, details;
                RETURN NEXT;
            END;
            
            operation_count := operation_count + 1;
        END LOOP;
    END IF;
    
    -- Final summary
    SELECT 'MAINTENANCE_COMPLETE', 'SUCCESS', 
           'Completed ' || operation_count || ' operations in ' || 
           EXTRACT(EPOCH FROM (NOW() - maintenance_start)) || ' seconds'
    INTO operation, status, details;
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- ====================================
-- 6. MONITORING VIEWS
-- ====================================

-- View for partition health monitoring
CREATE OR REPLACE VIEW partition_health_monitor AS
SELECT 
    table_type,
    COUNT(*) as partition_count,
    SUM(row_count) as total_rows,
    pg_size_pretty(SUM(size_bytes)) as total_size,
    AVG(row_count) as avg_rows_per_partition,
    MIN(last_vacuum) as oldest_vacuum,
    MIN(last_analyze) as oldest_analyze,
    COUNT(*) FILTER (WHERE last_vacuum < NOW() - INTERVAL '7 days') as partitions_need_vacuum,
    COUNT(*) FILTER (WHERE last_analyze < NOW() - INTERVAL '7 days') as partitions_need_analyze
FROM get_partition_statistics()
GROUP BY table_type;

-- Grant necessary permissions
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO mip_application;
GRANT SELECT ON partition_health_monitor TO mip_application;

-- Success notification
SELECT 'Advanced partition management functions created successfully!' as message;

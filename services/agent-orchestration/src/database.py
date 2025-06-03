# services/agent-orchestration/src/database.py
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, date
import asyncpg
import json
import hashlib
from dataclasses import asdict

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database manager for agent orchestration
    Handles all database operations for agent performance, costs, and user settings
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or "postgresql://mip_user:mip_password@localhost:5432/mip"
        self.pool: Optional[asyncpg.Pool] = None
        logger.info("Database manager initialized")
    
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Ensure required tables exist
            await self._ensure_tables()
            
            logger.info("Database connection pool established")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    async def _ensure_tables(self):
        """Ensure all required tables exist"""
        
        # Agent audit log table
        await self._execute_query("""
            CREATE TABLE IF NOT EXISTS agent_audit_log (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                agent_type VARCHAR(30) NOT NULL,
                user_id UUID,
                analysis_id UUID,
                inference_time_ms INT,
                input_hash CHAR(64),
                output_hash CHAR(64),
                tokens_used INT,
                cost_usd NUMERIC(10,6),
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                status VARCHAR(20) DEFAULT 'success',
                error_message TEXT
            );
        """)
        
        # User LLM settings table
        await self._execute_query("""
            CREATE TABLE IF NOT EXISTS user_llm_settings (
                user_id UUID PRIMARY KEY,
                explanation_model VARCHAR(20) DEFAULT 'llama-7b',
                strategy_model VARCHAR(20) DEFAULT 'gpt-4-turbo',
                latency_tolerance INT DEFAULT 300,
                user_tier VARCHAR(20) DEFAULT 'free',
                budget_limit NUMERIC(10,2) DEFAULT 50.00,
                auto_fallback BOOLEAN DEFAULT true,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Cost events table
        await self._execute_query("""
            CREATE TABLE IF NOT EXISTS cost_events (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL,
                agent_type VARCHAR(30) NOT NULL,
                cost_usd NUMERIC(10,6) NOT NULL,
                tokens_used INT NOT NULL,
                analysis_id UUID,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # User budgets table
        await self._execute_query("""
            CREATE TABLE IF NOT EXISTS user_budgets (
                user_id UUID PRIMARY KEY,
                monthly_limit NUMERIC(10,2) NOT NULL,
                daily_limit NUMERIC(10,2) NOT NULL,
                per_request_limit NUMERIC(10,2) NOT NULL,
                user_tier VARCHAR(20) NOT NULL,
                auto_downgrade BOOLEAN DEFAULT true,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Workflow executions table
        await self._execute_query("""
            CREATE TABLE IF NOT EXISTS workflow_executions (
                analysis_id UUID PRIMARY KEY,
                user_id UUID NOT NULL,
                request_type VARCHAR(50) NOT NULL,
                agent_outputs JSONB,
                total_cost NUMERIC(10,6),
                duration_ms INT,
                fallbacks_used TEXT[],
                status VARCHAR(20) DEFAULT 'success',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Agent performance snapshots table
        await self._execute_query("""
            CREATE TABLE IF NOT EXISTS agent_performance_snapshots (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                agent_type VARCHAR(30) NOT NULL,
                avg_latency_ms NUMERIC(8,2),
                success_rate NUMERIC(5,4),
                request_count INT,
                error_count INT,
                snapshot_time TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create indexes for better performance
        await self._execute_query("""
            CREATE INDEX IF NOT EXISTS idx_agent_audit_log_agent_type 
            ON agent_audit_log (agent_type, timestamp DESC);
        """)
        
        await self._execute_query("""
            CREATE INDEX IF NOT EXISTS idx_agent_audit_log_user_id 
            ON agent_audit_log (user_id, timestamp DESC);
        """)
        
        await self._execute_query("""
            CREATE INDEX IF NOT EXISTS idx_cost_events_user_timestamp 
            ON cost_events (user_id, timestamp DESC);
        """)
        
        await self._execute_query("""
            CREATE INDEX IF NOT EXISTS idx_cost_events_agent_timestamp 
            ON cost_events (agent_type, timestamp DESC);
        """)
        
        logger.info("Database tables verified/created")
    
    async def _execute_query(self, query: str, *args) -> Any:
        """Execute a database query"""
        if not self.pool:
            raise Exception("Database not initialized")
        
        async with self.pool.acquire() as connection:
            return await connection.execute(query, *args)
    
    async def _fetch_query(self, query: str, *args) -> List[Any]:
        """Fetch results from a database query"""
        if not self.pool:
            raise Exception("Database not initialized")
        
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
    
    async def _fetchrow_query(self, query: str, *args) -> Optional[Any]:
        """Fetch single row from a database query"""
        if not self.pool:
            raise Exception("Database not initialized")
        
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *args)
    
    async def log_agent_performance(
        self,
        agent_type: str,
        analysis_id: str,
        inference_time_ms: int,
        input_hash: str,
        output_hash: str,
        tokens_used: int,
        cost_usd: float,
        user_id: Optional[str] = None,
        status: str = 'success',
        error_message: Optional[str] = None
    ):
        """Log agent performance data"""
        try:
            await self._execute_query("""
                INSERT INTO agent_audit_log (
                    agent_type, user_id, analysis_id, inference_time_ms,
                    input_hash, output_hash, tokens_used, cost_usd,
                    status, error_message
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, agent_type, user_id, analysis_id, inference_time_ms,
                input_hash, output_hash, tokens_used, cost_usd,
                status, error_message)
            
            logger.debug(f"Logged performance for {agent_type}: {inference_time_ms}ms, ${cost_usd:.6f}")
            
        except Exception as e:
            logger.error(f"Failed to log agent performance: {str(e)}")
            raise
    
    async def record_cost_event(self, cost_event):
        """Record a cost event"""
        try:
            await self._execute_query("""
                INSERT INTO cost_events (
                    user_id, agent_type, cost_usd, tokens_used, analysis_id, timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, cost_event.user_id, cost_event.agent_type, cost_event.cost_usd,
                cost_event.tokens_used, cost_event.analysis_id, cost_event.timestamp)
            
            logger.debug(f"Recorded cost event: {cost_event.user_id}, ${cost_event.cost_usd:.6f}")
            
        except Exception as e:
            logger.error(f"Failed to record cost event: {str(e)}")
            raise
    
    async def log_workflow_execution(
        self,
        analysis_id: str,
        user_id: str,
        request_type: str,
        agent_outputs: Dict[str, Any],
        total_cost: float,
        duration_ms: int,
        fallbacks_used: List[str]
    ):
        """Log complete workflow execution"""
        try:
            await self._execute_query("""
                INSERT INTO workflow_executions (
                    analysis_id, user_id, request_type, agent_outputs,
                    total_cost, duration_ms, fallbacks_used
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, analysis_id, user_id, request_type, json.dumps(agent_outputs),
                total_cost, duration_ms, fallbacks_used)
            
            logger.debug(f"Logged workflow execution: {analysis_id}, ${total_cost:.6f}")
            
        except Exception as e:
            logger.error(f"Failed to log workflow execution: {str(e)}")
            raise
    
    async def get_user_llm_settings(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user LLM settings"""
        try:
            row = await self._fetchrow_query("""
                SELECT explanation_model, strategy_model, latency_tolerance,
                       user_tier, budget_limit, auto_fallback
                FROM user_llm_settings 
                WHERE user_id = $1
            """, user_id)
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user LLM settings: {str(e)}")
            return None
    
    async def update_user_llm_settings(self, user_id: str, settings: Dict[str, Any]):
        """Update user LLM settings"""
        try:
            # Upsert user settings
            await self._execute_query("""
                INSERT INTO user_llm_settings (
                    user_id, explanation_model, strategy_model, latency_tolerance,
                    user_tier, budget_limit, auto_fallback, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (user_id) DO UPDATE SET
                    explanation_model = EXCLUDED.explanation_model,
                    strategy_model = EXCLUDED.strategy_model,
                    latency_tolerance = EXCLUDED.latency_tolerance,
                    user_tier = EXCLUDED.user_tier,
                    budget_limit = EXCLUDED.budget_limit,
                    auto_fallback = EXCLUDED.auto_fallback,
                    updated_at = NOW()
            """, user_id, settings.get('explanation_model', 'llama-7b'),
                settings.get('strategy_model', 'gpt-4-turbo'),
                settings.get('latency_tolerance', 300),
                settings.get('user_tier', 'free'),
                settings.get('budget_limit', 50.0),
                settings.get('auto_fallback', True))
            
            logger.info(f"Updated LLM settings for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to update user LLM settings: {str(e)}")
            raise
    
    async def get_user_tier(self, user_id: str) -> Optional[str]:
        """Get user tier"""
        try:
            row = await self._fetchrow_query("""
                SELECT user_tier FROM user_llm_settings WHERE user_id = $1
            """, user_id)
            
            return row['user_tier'] if row else None
            
        except Exception as e:
            logger.error(f"Failed to get user tier: {str(e)}")
            return None
    
    async def get_user_budget(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user budget configuration"""
        try:
            row = await self._fetchrow_query("""
                SELECT monthly_limit, daily_limit, per_request_limit,
                       user_tier, auto_downgrade
                FROM user_budgets 
                WHERE user_id = $1
            """, user_id)
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user budget: {str(e)}")
            return None
    
    async def save_user_budget(self, budget):
        """Save user budget configuration"""
        try:
            await self._execute_query("""
                INSERT INTO user_budgets (
                    user_id, monthly_limit, daily_limit, per_request_limit,
                    user_tier, auto_downgrade
                ) VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id) DO UPDATE SET
                    monthly_limit = EXCLUDED.monthly_limit,
                    daily_limit = EXCLUDED.daily_limit,
                    per_request_limit = EXCLUDED.per_request_limit,
                    user_tier = EXCLUDED.user_tier,
                    auto_downgrade = EXCLUDED.auto_downgrade,
                    updated_at = NOW()
            """, budget.user_id, budget.monthly_limit, budget.daily_limit,
                budget.per_request_limit, budget.user_tier, budget.auto_downgrade)
            
            logger.debug(f"Saved budget for user {budget.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to save user budget: {str(e)}")
            raise
    
    async def get_daily_usage(self, user_id: str, target_date: date) -> float:
        """Get user's daily cost usage"""
        try:
            row = await self._fetchrow_query("""
                SELECT COALESCE(SUM(cost_usd), 0) as daily_cost
                FROM cost_events 
                WHERE user_id = $1 
                AND DATE(timestamp) = $2
            """, user_id, target_date)
            
            return float(row['daily_cost']) if row else 0.0
            
        except Exception as e:
            logger.error(f"Failed to get daily usage: {str(e)}")
            return 0.0
    
    async def get_monthly_usage(self, user_id: str, year_month: str) -> float:
        """Get user's monthly cost usage"""
        try:
            row = await self._fetchrow_query("""
                SELECT COALESCE(SUM(cost_usd), 0) as monthly_cost
                FROM cost_events 
                WHERE user_id = $1 
                AND TO_CHAR(timestamp, 'YYYY-MM') = $2
            """, user_id, year_month)
            
            return float(row['monthly_cost']) if row else 0.0
            
        except Exception as e:
            logger.error(f"Failed to get monthly usage: {str(e)}")
            return 0.0
    
    async def get_cost_breakdown(
        self,
        user_id: str,
        period: str = "monthly"
    ) -> Dict[str, Any]:
        """Get detailed cost breakdown by agent type"""
        try:
            if period == "daily":
                time_filter = "DATE(timestamp) = CURRENT_DATE"
            elif period == "weekly":
                time_filter = "timestamp >= CURRENT_DATE - INTERVAL '7 days'"
            else:  # monthly
                time_filter = "TO_CHAR(timestamp, 'YYYY-MM') = TO_CHAR(CURRENT_DATE, 'YYYY-MM')"
            
            rows = await self._fetch_query(f"""
                SELECT 
                    agent_type,
                    SUM(cost_usd) as total_cost,
                    SUM(tokens_used) as total_tokens,
                    COUNT(*) as request_count,
                    AVG(cost_usd) as avg_cost_per_request
                FROM cost_events 
                WHERE user_id = $1 AND {time_filter}
                GROUP BY agent_type
                ORDER BY total_cost DESC
            """, user_id)
            
            breakdown = {}
            total_cost = 0.0
            
            for row in rows:
                agent_type = row['agent_type']
                cost = float(row['total_cost'])
                total_cost += cost
                
                breakdown[agent_type] = {
                    'cost': cost,
                    'tokens': int(row['total_tokens']),
                    'requests': int(row['request_count']),
                    'avg_cost_per_request': float(row['avg_cost_per_request'])
                }
            
            # Add percentages
            for agent_type in breakdown:
                if total_cost > 0:
                    breakdown[agent_type]['percentage'] = (breakdown[agent_type]['cost'] / total_cost) * 100
                else:
                    breakdown[agent_type]['percentage'] = 0.0
            
            return {
                'breakdown': breakdown,
                'total_cost': total_cost,
                'period': period
            }
            
        except Exception as e:
            logger.error(f"Failed to get cost breakdown: {str(e)}")
            return {}
    
    async def get_agent_performance_metrics(self, agent_type: str) -> Dict[str, Any]:
        """Get performance metrics for specific agent"""
        try:
            # Get recent performance data
            row = await self._fetchrow_query("""
                SELECT 
                    COUNT(*) as total_requests,
                    AVG(inference_time_ms) as avg_latency,
                    COUNT(*) FILTER (WHERE status = 'success') as successful_requests,
                    COUNT(*) FILTER (WHERE status != 'success') as failed_requests,
                    SUM(cost_usd) as total_cost,
                    SUM(tokens_used) as total_tokens
                FROM agent_audit_log 
                WHERE agent_type = $1 
                AND timestamp >= NOW() - INTERVAL '24 hours'
            """, agent_type)
            
            if not row or row['total_requests'] == 0:
                return {
                    'agent_type': agent_type,
                    'status': 'no_data',
                    'total_requests': 0
                }
            
            total_requests = int(row['total_requests'])
            successful_requests = int(row['successful_requests'])
            
            return {
                'agent_type': agent_type,
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': int(row['failed_requests']),
                'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
                'avg_latency_ms': float(row['avg_latency']) if row['avg_latency'] else 0,
                'total_cost': float(row['total_cost']) if row['total_cost'] else 0,
                'total_tokens': int(row['total_tokens']) if row['total_tokens'] else 0,
                'period': '24_hours'
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent performance metrics: {str(e)}")
            return {}
    
    async def store_performance_snapshot(
        self,
        agent_type: str,
        avg_latency: float,
        success_rate: float
    ):
        """Store performance snapshot for historical tracking"""
        try:
            # Get current request and error counts
            row = await self._fetchrow_query("""
                SELECT 
                    COUNT(*) as request_count,
                    COUNT(*) FILTER (WHERE status != 'success') as error_count
                FROM agent_audit_log 
                WHERE agent_type = $1 
                AND timestamp >= NOW() - INTERVAL '5 minutes'
            """, agent_type)
            
            request_count = int(row['request_count']) if row else 0
            error_count = int(row['error_count']) if row else 0
            
            await self._execute_query("""
                INSERT INTO agent_performance_snapshots (
                    agent_type, avg_latency_ms, success_rate, 
                    request_count, error_count
                ) VALUES ($1, $2, $3, $4, $5)
            """, agent_type, avg_latency, success_rate, request_count, error_count)
            
        except Exception as e:
            logger.error(f"Failed to store performance snapshot: {str(e)}")
    
    async def get_system_cost_stats(self, period: str = "daily") -> Dict[str, Any]:
        """Get system-wide cost statistics"""
        try:
            if period == "daily":
                time_filter = "DATE(timestamp) = CURRENT_DATE"
            elif period == "weekly":
                time_filter = "timestamp >= CURRENT_DATE - INTERVAL '7 days'"
            else:  # monthly
                time_filter = "TO_CHAR(timestamp, 'YYYY-MM') = TO_CHAR(CURRENT_DATE, 'YYYY-MM')"
            
            row = await self._fetchrow_query(f"""
                SELECT 
                    SUM(cost_usd) as total_cost,
                    COUNT(DISTINCT user_id) as active_users,
                    COUNT(*) as total_requests,
                    SUM(tokens_used) as total_tokens
                FROM cost_events 
                WHERE {time_filter}
            """)
            
            agent_breakdown = await self._fetch_query(f"""
                SELECT 
                    agent_type,
                    SUM(cost_usd) as cost,
                    COUNT(*) as requests
                FROM cost_events 
                WHERE {time_filter}
                GROUP BY agent_type
                ORDER BY cost DESC
            """)
            
            return {
                'period': period,
                'total_cost': float(row['total_cost']) if row['total_cost'] else 0,
                'active_users': int(row['active_users']) if row['active_users'] else 0,
                'total_requests': int(row['total_requests']) if row['total_requests'] else 0,
                'total_tokens': int(row['total_tokens']) if row['total_tokens'] else 0,
                'agent_breakdown': [
                    {
                        'agent_type': r['agent_type'],
                        'cost': float(r['cost']),
                        'requests': int(r['requests'])
                    }
                    for r in agent_breakdown
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get system cost stats: {str(e)}")
            return {}
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Cleanup old audit and cost data"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Clean up old audit logs
            deleted_audit = await self._execute_query("""
                DELETE FROM agent_audit_log 
                WHERE timestamp < $1
            """, cutoff_date)
            
            # Clean up old performance snapshots
            deleted_snapshots = await self._execute_query("""
                DELETE FROM agent_performance_snapshots 
                WHERE snapshot_time < $1
            """, cutoff_date)
            
            logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
    
    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connections closed")

# Example usage and testing
async def test_database():
    """Test database functionality"""
    
    db = DatabaseManager()
    await db.initialize()
    
    user_id = "test_user_123"
    analysis_id = "test_analysis_456"
    
    # Test user settings
    settings = {
        'explanation_model': 'llama-7b',
        'strategy_model': 'gpt-4-turbo',
        'latency_tolerance': 500,
        'user_tier': 'premium',
        'budget_limit': 100.0
    }
    
    await db.update_user_llm_settings(user_id, settings)
    retrieved_settings = await db.get_user_llm_settings(user_id)
    print(f"User settings: {retrieved_settings}")
    
    # Test performance logging
    await db.log_agent_performance(
        agent_type="gpt-4-turbo",
        analysis_id=analysis_id,
        inference_time_ms=850,
        input_hash="abc123",
        output_hash="def456",
        tokens_used=1500,
        cost_usd=0.045,
        user_id=user_id
    )
    
    # Test cost tracking
    from cost_tracker import CostEvent
    cost_event = CostEvent(
        user_id=user_id,
        agent_type="gpt-4-turbo",
        cost_usd=0.045,
        tokens_used=1500,
        timestamp=datetime.utcnow(),
        analysis_id=analysis_id
    )
    
    await db.record_cost_event(cost_event)
    
    # Test metrics retrieval
    performance = await db.get_agent_performance_metrics("gpt-4-turbo")
    print(f"Agent performance: {performance}")
    
    breakdown = await db.get_cost_breakdown(user_id, "monthly")
    print(f"Cost breakdown: {breakdown}")
    
    await db.close()

if __name__ == "__main__":
    asyncio.run(test_database())

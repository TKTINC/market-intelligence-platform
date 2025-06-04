"""
Cost Tracking and Budget Management for GPT-4 Strategy Service
"""

import asyncio
import asyncpg
import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class UsageRecord:
    user_id: str
    strategy_id: str
    cost_usd: float
    tokens_used: int
    model_used: str
    timestamp: datetime

class CostTracker:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.daily_budget_alerts = [10.0, 25.0, 50.0]  # Alert thresholds
        
    async def get_db_connection(self):
        """Get database connection"""
        try:
            return await asyncpg.connect(self.db_url)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    async def record_usage(
        self,
        user_id: str,
        cost_usd: float,
        tokens_used: int,
        strategy_id: str,
        model_used: str = "gpt-4-turbo"
    ):
        """Record usage and cost for tracking and billing"""
        
        try:
            conn = await self.get_db_connection()
            
            await conn.execute("""
                INSERT INTO gpt4_usage_log 
                (user_id, strategy_id, cost_usd, tokens_used, model_used, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, user_id, strategy_id, cost_usd, tokens_used, model_used, datetime.utcnow())
            
            # Check for budget alerts
            await self._check_budget_alerts(conn, user_id, cost_usd)
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record usage: {e}")
    
    async def get_user_usage(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get user's usage statistics"""
        
        try:
            conn = await self.get_db_connection()
            
            since_date = datetime.utcnow() - timedelta(days=days)
            
            # Get total usage
            total_cost = await conn.fetchval("""
                SELECT COALESCE(SUM(cost_usd), 0) 
                FROM gpt4_usage_log 
                WHERE user_id = $1 AND timestamp >= $2
            """, user_id, since_date)
            
            total_requests = await conn.fetchval("""
                SELECT COUNT(*) 
                FROM gpt4_usage_log 
                WHERE user_id = $1 AND timestamp >= $2
            """, user_id, since_date)
            
            # Get daily breakdown
            daily_usage = await conn.fetch("""
                SELECT 
                    DATE(timestamp) as date,
                    SUM(cost_usd) as daily_cost,
                    COUNT(*) as daily_requests,
                    AVG(tokens_used) as avg_tokens
                FROM gpt4_usage_log 
                WHERE user_id = $1 AND timestamp >= $2
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, user_id, since_date)
            
            # Get model breakdown
            model_usage = await conn.fetch("""
                SELECT 
                    model_used,
                    SUM(cost_usd) as model_cost,
                    COUNT(*) as model_requests
                FROM gpt4_usage_log 
                WHERE user_id = $1 AND timestamp >= $2
                GROUP BY model_used
            """, user_id, since_date)
            
            await conn.close()
            
            return {
                "period_days": days,
                "total_cost_usd": float(total_cost),
                "total_requests": total_requests,
                "average_cost_per_request": float(total_cost / max(total_requests, 1)),
                "daily_usage": [dict(row) for row in daily_usage],
                "model_breakdown": [dict(row) for row in model_usage]
            }
            
        except Exception as e:
            logger.error(f"Failed to get user usage: {e}")
            return {"error": "Usage data unavailable"}
    
    async def get_strategy_status(self, strategy_id: str) -> Dict[str, Any]:
        """Get status and cost for specific strategy"""
        
        try:
            conn = await self.get_db_connection()
            
            strategy_data = await conn.fetchrow("""
                SELECT 
                    user_id,
                    cost_usd,
                    tokens_used,
                    model_used,
                    timestamp
                FROM gpt4_usage_log 
                WHERE strategy_id = $1
            """, strategy_id)
            
            await conn.close()
            
            if strategy_data:
                return {
                    "strategy_id": strategy_id,
                    "user_id": strategy_data["user_id"],
                    "cost_usd": float(strategy_data["cost_usd"]),
                    "tokens_used": strategy_data["tokens_used"],
                    "model_used": strategy_data["model_used"],
                    "timestamp": strategy_data["timestamp"].isoformat(),
                    "status": "completed"
                }
            else:
                return {"strategy_id": strategy_id, "status": "not_found"}
                
        except Exception as e:
            logger.error(f"Failed to get strategy status: {e}")
            return {"strategy_id": strategy_id, "status": "error"}
    
    async def _check_budget_alerts(
        self,
        conn,
        user_id: str,
        new_cost: float
    ):
        """Check if user has exceeded budget thresholds"""
        
        try:
            # Get today's spending
            today = datetime.utcnow().date()
            today_spending = await conn.fetchval("""
                SELECT COALESCE(SUM(cost_usd), 0)
                FROM gpt4_usage_log 
                WHERE user_id = $1 AND DATE(timestamp) = $2
            """, user_id, today)
            
            total_today = float(today_spending) + new_cost
            
            # Check alert thresholds
            for threshold in self.daily_budget_alerts:
                if today_spending < threshold <= total_today:
                    await self._send_budget_alert(conn, user_id, total_today, threshold)
                    
        except Exception as e:
            logger.error(f"Budget alert check failed: {e}")
    
    async def _send_budget_alert(
        self,
        conn,
        user_id: str,
        current_spending: float,
        threshold: float
    ):
        """Send budget alert notification"""
        
        try:
            # Record alert
            await conn.execute("""
                INSERT INTO budget_alerts 
                (user_id, threshold_usd, actual_spending_usd, alert_timestamp)
                VALUES ($1, $2, $3, $4)
            """, user_id, threshold, current_spending, datetime.utcnow())
            
            # In production, send email/notification here
            logger.warning(f"Budget alert: User {user_id} spent ${current_spending:.2f} (threshold: ${threshold:.2f})")
            
        except Exception as e:
            logger.error(f"Failed to send budget alert: {e}")
    
    async def get_service_costs(self, hours: int = 24) -> Dict[str, Any]:
        """Get service-wide cost analysis"""
        
        try:
            conn = await self.get_db_connection()
            
            since_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Total service costs
            total_cost = await conn.fetchval("""
                SELECT COALESCE(SUM(cost_usd), 0)
                FROM gpt4_usage_log 
                WHERE timestamp >= $1
            """, since_time)
            
            # Cost by model
            model_costs = await conn.fetch("""
                SELECT 
                    model_used,
                    SUM(cost_usd) as total_cost,
                    COUNT(*) as request_count,
                    AVG(cost_usd) as avg_cost_per_request
                FROM gpt4_usage_log 
                WHERE timestamp >= $1
                GROUP BY model_used
            """, since_time)
            
            # Top spending users
            top_users = await conn.fetch("""
                SELECT 
                    user_id,
                    SUM(cost_usd) as user_cost,
                    COUNT(*) as user_requests
                FROM gpt4_usage_log 
                WHERE timestamp >= $1
                GROUP BY user_id
                ORDER BY user_cost DESC
                LIMIT 10
            """, since_time)
            
            await conn.close()
            
            return {
                "period_hours": hours,
                "total_service_cost": float(total_cost),
                "model_breakdown": [dict(row) for row in model_costs],
                "top_users": [dict(row) for row in top_users]
            }
            
        except Exception as e:
            logger.error(f"Failed to get service costs: {e}")
            return {"error": "Service cost data unavailable"}

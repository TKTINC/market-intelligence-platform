# services/agent-orchestration/src/cost_tracker.py
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import redis.asyncio as redis
from database import DatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class CostEvent:
    """Individual cost event"""
    user_id: str
    agent_type: str
    cost_usd: float
    tokens_used: int
    timestamp: datetime
    analysis_id: str

@dataclass
class UserBudget:
    """User budget configuration"""
    user_id: str
    monthly_limit: float
    daily_limit: float
    per_request_limit: float
    user_tier: str
    auto_downgrade: bool = True

class CostTracker:
    """
    Tracks and manages costs for AI agent usage
    Provides budget enforcement and cost optimization
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self.db = DatabaseManager()
        
        # Cost configuration per agent type (per 1K tokens)
        self.agent_costs = {
            'finbert': 0.0001,
            'gpt-4-turbo': 0.03,
            'llama-7b': 0.0004,
            'tft': 0.0002,
            'risk_analysis': 0.0001,
            
            # Fallback models (cheaper)
            'finbert_lite': 0.00005,
            'strategy_lite': 0.002,
            'finbert_explainer': 0.00008,
            'lstm_basic': 0.0001,
            'rule_based_risk': 0.00001
        }
        
        # Default budget limits by user tier
        self.default_budgets = {
            'free': {
                'monthly_limit': 5.0,
                'daily_limit': 0.50,
                'per_request_limit': 0.10
            },
            'premium': {
                'monthly_limit': 100.0,
                'daily_limit': 10.0,
                'per_request_limit': 2.0
            },
            'enterprise': {
                'monthly_limit': 1000.0,
                'daily_limit': 100.0,
                'per_request_limit': 10.0
            }
        }
        
        logger.info("Cost tracker initialized")
    
    async def initialize(self):
        """Initialize cost tracker"""
        if not self.redis:
            # Initialize Redis if not provided
            self.redis = redis.from_url("redis://localhost:6379")
        
        await self.db.initialize()
        logger.info("Cost tracker ready")
    
    async def calculate_cost(
        self,
        agent_type: str,
        tokens_used: int,
        custom_rate: Optional[float] = None
    ) -> float:
        """
        Calculate cost for agent usage
        
        Args:
            agent_type: Type of agent used
            tokens_used: Number of tokens processed
            custom_rate: Optional custom rate override
            
        Returns:
            Cost in USD
        """
        if custom_rate is not None:
            rate_per_1k = custom_rate
        else:
            rate_per_1k = self.agent_costs.get(agent_type, 0.001)  # Default fallback rate
        
        cost = (tokens_used / 1000.0) * rate_per_1k
        
        logger.debug(f"Cost calculation: {agent_type} {tokens_used} tokens = ${cost:.6f}")
        
        return cost
    
    async def check_user_budget(
        self,
        user_id: str,
        estimated_cost: float,
        budget_type: str = "per_request"
    ) -> bool:
        """
        Check if user can afford the estimated cost
        
        Args:
            user_id: User identifier
            estimated_cost: Estimated cost in USD
            budget_type: Type of budget to check ('per_request', 'daily', 'monthly')
            
        Returns:
            True if user can afford the cost
        """
        try:
            # Get user budget
            user_budget = await self.get_user_budget(user_id)
            
            # Check per-request limit
            if budget_type == "per_request":
                if estimated_cost > user_budget.per_request_limit:
                    logger.warning(
                        f"User {user_id} exceeds per-request limit: "
                        f"${estimated_cost:.4f} > ${user_budget.per_request_limit:.4f}"
                    )
                    return False
            
            # Check daily limit
            elif budget_type == "daily":
                daily_usage = await self.get_daily_usage(user_id)
                if (daily_usage + estimated_cost) > user_budget.daily_limit:
                    logger.warning(
                        f"User {user_id} would exceed daily limit: "
                        f"${daily_usage + estimated_cost:.4f} > ${user_budget.daily_limit:.4f}"
                    )
                    return False
            
            # Check monthly limit
            elif budget_type == "monthly":
                monthly_usage = await self.get_monthly_usage(user_id)
                if (monthly_usage + estimated_cost) > user_budget.monthly_limit:
                    logger.warning(
                        f"User {user_id} would exceed monthly limit: "
                        f"${monthly_usage + estimated_cost:.4f} > ${user_budget.monthly_limit:.4f}"
                    )
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Budget check failed for user {user_id}: {str(e)}")
            return False
    
    async def record_cost(
        self,
        user_id: str,
        agent_type: str,
        cost_usd: float,
        tokens_used: int,
        analysis_id: str
    ) -> bool:
        """
        Record a cost event
        
        Args:
            user_id: User identifier
            agent_type: Type of agent used
            cost_usd: Cost in USD
            tokens_used: Number of tokens used
            analysis_id: Analysis identifier
            
        Returns:
            True if recorded successfully
        """
        try:
            cost_event = CostEvent(
                user_id=user_id,
                agent_type=agent_type,
                cost_usd=cost_usd,
                tokens_used=tokens_used,
                timestamp=datetime.utcnow(),
                analysis_id=analysis_id
            )
            
            # Store in database
            await self.db.record_cost_event(cost_event)
            
            # Update Redis counters for fast access
            await self._update_redis_counters(cost_event)
            
            logger.debug(f"Recorded cost: User {user_id}, ${cost_usd:.6f}, {agent_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record cost: {str(e)}")
            return False
    
    async def get_user_budget(self, user_id: str) -> UserBudget:
        """Get user budget configuration"""
        try:
            # Try to get from database first
            budget_data = await self.db.get_user_budget(user_id)
            
            if budget_data:
                return UserBudget(**budget_data)
            
            # Get user tier to determine default budget
            user_tier = await self.db.get_user_tier(user_id) or 'free'
            default_limits = self.default_budgets[user_tier]
            
            # Create default budget
            budget = UserBudget(
                user_id=user_id,
                monthly_limit=default_limits['monthly_limit'],
                daily_limit=default_limits['daily_limit'],
                per_request_limit=default_limits['per_request_limit'],
                user_tier=user_tier,
                auto_downgrade=True
            )
            
            # Save default budget to database
            await self.db.save_user_budget(budget)
            
            return budget
            
        except Exception as e:
            logger.error(f"Failed to get user budget: {str(e)}")
            # Return minimal free tier budget as fallback
            return UserBudget(
                user_id=user_id,
                monthly_limit=5.0,
                daily_limit=0.50,
                per_request_limit=0.10,
                user_tier='free'
            )
    
    async def get_daily_usage(self, user_id: str) -> float:
        """Get user's daily cost usage"""
        try:
            # Try Redis first for speed
            today = datetime.utcnow().date()
            redis_key = f"daily_cost:{user_id}:{today}"
            
            cached_usage = await self.redis.get(redis_key)
            if cached_usage:
                return float(cached_usage)
            
            # Fall back to database
            usage = await self.db.get_daily_usage(user_id, today)
            
            # Cache for future requests
            await self.redis.setex(redis_key, 3600, str(usage))  # Cache for 1 hour
            
            return usage
            
        except Exception as e:
            logger.error(f"Failed to get daily usage: {str(e)}")
            return 0.0
    
    async def get_monthly_usage(self, user_id: str) -> float:
        """Get user's monthly cost usage"""
        try:
            # Try Redis first
            current_month = datetime.utcnow().strftime('%Y-%m')
            redis_key = f"monthly_cost:{user_id}:{current_month}"
            
            cached_usage = await self.redis.get(redis_key)
            if cached_usage:
                return float(cached_usage)
            
            # Fall back to database
            usage = await self.db.get_monthly_usage(user_id, current_month)
            
            # Cache for future requests
            await self.redis.setex(redis_key, 1800, str(usage))  # Cache for 30 minutes
            
            return usage
            
        except Exception as e:
            logger.error(f"Failed to get monthly usage: {str(e)}")
            return 0.0
    
    async def get_cost_breakdown(
        self,
        user_id: str,
        period: str = "monthly"
    ) -> Dict[str, Any]:
        """
        Get detailed cost breakdown by agent type
        
        Args:
            user_id: User identifier
            period: 'daily', 'weekly', or 'monthly'
            
        Returns:
            Cost breakdown by agent type
        """
        try:
            return await self.db.get_cost_breakdown(user_id, period)
            
        except Exception as e:
            logger.error(f"Failed to get cost breakdown: {str(e)}")
            return {}
    
    async def suggest_cost_optimization(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Suggest ways to optimize costs for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        try:
            # Get user's usage patterns
            breakdown = await self.get_cost_breakdown(user_id, "monthly")
            user_budget = await self.get_user_budget(user_id)
            monthly_usage = await self.get_monthly_usage(user_id)
            
            # Check if user is approaching limits
            usage_ratio = monthly_usage / user_budget.monthly_limit
            
            if usage_ratio > 0.8:
                suggestions.append({
                    'type': 'budget_warning',
                    'severity': 'high',
                    'message': f"You've used {usage_ratio*100:.1f}% of your monthly budget",
                    'action': 'Consider upgrading your plan or reducing usage'
                })
            
            # Analyze expensive agent usage
            if breakdown.get('gpt-4-turbo', 0) > monthly_usage * 0.5:
                suggestions.append({
                    'type': 'agent_optimization',
                    'severity': 'medium',
                    'message': "GPT-4 Turbo is your largest cost driver",
                    'action': 'Consider using Llama 2-7B for explanations instead'
                })
            
            # Suggest tier upgrade if beneficial
            if user_budget.user_tier == 'free' and monthly_usage > 3.0:
                suggestions.append({
                    'type': 'tier_upgrade',
                    'severity': 'low',
                    'message': "Premium tier might offer better value",
                    'action': 'Upgrade to Premium for higher limits and advanced features'
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate cost suggestions: {str(e)}")
            return []
    
    async def enforce_budget_limits(
        self,
        user_id: str,
        requested_agents: List[str]
    ) -> List[str]:
        """
        Enforce budget limits by downgrading to cheaper agents if needed
        
        Args:
            user_id: User identifier
            requested_agents: List of requested agent types
            
        Returns:
            List of approved/downgraded agent types
        """
        try:
            user_budget = await self.get_user_budget(user_id)
            
            if not user_budget.auto_downgrade:
                return requested_agents
            
            # Check if user is near limits
            daily_usage = await self.get_daily_usage(user_id)
            daily_remaining = user_budget.daily_limit - daily_usage
            
            # Define downgrade mappings
            downgrades = {
                'gpt-4-turbo': 'strategy_lite',
                'llama-7b': 'finbert_explainer',
                'finbert': 'finbert_lite'
            }
            
            approved_agents = []
            
            for agent in requested_agents:
                # Estimate cost for this agent
                estimated_cost = self.agent_costs.get(agent, 0.001) * 1000  # Assume 1K tokens
                
                if estimated_cost <= daily_remaining:
                    approved_agents.append(agent)
                    daily_remaining -= estimated_cost
                else:
                    # Try to downgrade
                    downgrade_agent = downgrades.get(agent, agent)
                    downgrade_cost = self.agent_costs.get(downgrade_agent, 0.001) * 1000
                    
                    if downgrade_cost <= daily_remaining:
                        approved_agents.append(downgrade_agent)
                        daily_remaining -= downgrade_cost
                        
                        logger.info(f"Downgraded {agent} to {downgrade_agent} for user {user_id}")
                    else:
                        logger.warning(f"Skipped {agent} due to budget limits for user {user_id}")
            
            return approved_agents
            
        except Exception as e:
            logger.error(f"Budget enforcement failed: {str(e)}")
            return requested_agents  # Return original list on error
    
    async def _update_redis_counters(self, cost_event: CostEvent):
        """Update Redis counters for fast access"""
        try:
            today = cost_event.timestamp.date()
            current_month = cost_event.timestamp.strftime('%Y-%m')
            
            # Update daily counter
            daily_key = f"daily_cost:{cost_event.user_id}:{today}"
            await self.redis.incrbyfloat(daily_key, cost_event.cost_usd)
            await self.redis.expire(daily_key, 86400 * 2)  # Expire after 2 days
            
            # Update monthly counter
            monthly_key = f"monthly_cost:{cost_event.user_id}:{current_month}"
            await self.redis.incrbyfloat(monthly_key, cost_event.cost_usd)
            await self.redis.expire(monthly_key, 86400 * 35)  # Expire after 35 days
            
            # Update agent-specific counters
            agent_key = f"agent_cost:{cost_event.user_id}:{cost_event.agent_type}:{current_month}"
            await self.redis.incrbyfloat(agent_key, cost_event.cost_usd)
            await self.redis.expire(agent_key, 86400 * 35)
            
        except Exception as e:
            logger.error(f"Failed to update Redis counters: {str(e)}")
    
    async def get_system_costs(self, period: str = "daily") -> Dict[str, Any]:
        """Get system-wide cost statistics"""
        try:
            return await self.db.get_system_cost_stats(period)
        except Exception as e:
            logger.error(f"Failed to get system costs: {str(e)}")
            return {}
    
    async def shutdown(self):
        """Shutdown cost tracker"""
        if self.redis:
            await self.redis.close()
        if self.db:
            await self.db.close()
        logger.info("Cost tracker shutdown complete")

# Example usage
async def test_cost_tracker():
    """Test cost tracker functionality"""
    
    tracker = CostTracker()
    await tracker.initialize()
    
    user_id = "test_user_123"
    
    # Test cost calculation
    cost = await tracker.calculate_cost("gpt-4-turbo", 1000)
    print(f"GPT-4 Turbo cost for 1K tokens: ${cost:.6f}")
    
    # Test budget check
    can_afford = await tracker.check_user_budget(user_id, cost)
    print(f"Can afford: {can_afford}")
    
    # Record some costs
    await tracker.record_cost(user_id, "gpt-4-turbo", cost, 1000, "test_analysis_1")
    await tracker.record_cost(user_id, "finbert", 0.0001, 500, "test_analysis_2")
    
    # Get usage
    daily_usage = await tracker.get_daily_usage(user_id)
    monthly_usage = await tracker.get_monthly_usage(user_id)
    
    print(f"Daily usage: ${daily_usage:.6f}")
    print(f"Monthly usage: ${monthly_usage:.6f}")
    
    # Get cost breakdown
    breakdown = await tracker.get_cost_breakdown(user_id)
    print(f"Cost breakdown: {breakdown}")
    
    # Get optimization suggestions
    suggestions = await tracker.suggest_cost_optimization(user_id)
    print(f"Optimization suggestions: {suggestions}")
    
    await tracker.shutdown()

if __name__ == "__main__":
    asyncio.run(test_cost_tracker())

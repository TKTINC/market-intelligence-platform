# services/agent-orchestration/src/orchestrator.py
import asyncio
import logging
import time
import json
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import redis.asyncio as redis

from agents.base_agent import BaseAgent
from agents.finbert_agent import FinBertAgent
from agents.gpt4_strategy_agent import GPT4StrategyAgent
from agents.llama_explanation_agent import LlamaExplanationAgent
from agents.tft_price_agent import TFTAgent
from agents.risk_analysis_agent import RiskAnalysisAgent
from circuit_breaker import CircuitBreaker
from cost_tracker import CostTracker
from database import DatabaseManager
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class WorkflowResult:
    """Result of agent workflow execution"""
    analysis_id: str
    agent_outputs: Dict[str, Any]
    total_cost: float
    duration_ms: int
    fallbacks_used: List[str]
    user_tier: str

@dataclass
class UserPreferences:
    """User agent preferences"""
    explanation_model: str = 'llama-7b'
    strategy_model: str = 'gpt-4-turbo'
    latency_tolerance: int = 300  # milliseconds
    user_tier: str = 'free'
    budget_limit: float = 50.0
    auto_fallback: bool = True

class AgentOrchestrator:
    """
    Central orchestrator for coordinating multiple AI agents
    Handles routing, fallbacks, cost tracking, and performance monitoring
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.db = DatabaseManager()
        self.cost_tracker = CostTracker()
        
        # Initialize agents
        self.agents: Dict[str, BaseAgent] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Routing configuration
        self.routing_table = {
            'news_analysis': ['sentiment', 'explanation'],
            'options_recommendation': ['options_strategy', 'risk_analysis'],
            'price_prediction': ['price_forecast', 'explanation'],
            'portfolio_analysis': ['sentiment', 'risk_analysis', 'explanation'],
            'comprehensive_analysis': ['sentiment', 'price_forecast', 'options_strategy', 'explanation']
        }
        
        # Agent configurations
        self.agent_configs = {
            'sentiment': {
                'agent_class': FinBertAgent,
                'cost_per_1k_tokens': 0.0001,
                'max_latency_ms': 50,
                'fallback': 'finbert_lite'
            },
            'options_strategy': {
                'agent_class': GPT4StrategyAgent,
                'cost_per_1k_tokens': 0.03,
                'max_latency_ms': 2000,
                'fallback': 'strategy_lite'
            },
            'explanation': {
                'agent_class': LlamaExplanationAgent,
                'cost_per_1k_tokens': 0.0004,
                'max_latency_ms': 500,
                'fallback': 'finbert_explainer'
            },
            'price_forecast': {
                'agent_class': TFTAgent,
                'cost_per_1k_tokens': 0.0002,
                'max_latency_ms': 100,
                'fallback': 'lstm_basic'
            },
            'risk_analysis': {
                'agent_class': RiskAnalysisAgent,
                'cost_per_1k_tokens': 0.0001,
                'max_latency_ms': 200,
                'fallback': 'rule_based_risk'
            }
        }
        
        # Performance tracking
        self.performance_metrics = {}
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'fallbacks_triggered': 0,
            'total_cost': 0.0,
            'agent_usage_count': {}
        }
    
    async def initialize(self):
        """Initialize all agents and circuit breakers"""
        logger.info("Initializing Agent Orchestrator...")
        
        try:
            # Initialize database connection
            await self.db.initialize()
            
            # Initialize agents
            for agent_name, config in self.agent_configs.items():
                try:
                    agent_class = config['agent_class']
                    agent = agent_class()
                    await agent.initialize()
                    
                    self.agents[agent_name] = agent
                    
                    # Initialize circuit breaker for each agent
                    self.circuit_breakers[agent_name] = CircuitBreaker(
                        failure_threshold=5,
                        recovery_timeout=60,
                        name=agent_name
                    )
                    
                    logger.info(f"Initialized agent: {agent_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize agent {agent_name}: {str(e)}")
                    # Continue with other agents
            
            # Start background tasks
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._performance_aggregator())
            
            logger.info(f"Agent Orchestrator initialized with {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {str(e)}")
            raise
    
    async def execute_workflow(
        self,
        request_type: str,
        user_id: str,
        payload: Dict[str, Any],
        user_preferences: Optional[UserPreferences] = None
    ) -> WorkflowResult:
        """Execute a complete workflow through appropriate agents"""
        
        start_time = time.time()
        analysis_id = str(uuid.uuid4())
        fallbacks_used = []
        
        try:
            # Get user preferences
            if not user_preferences:
                user_preferences = await self._get_user_preferences(user_id)
            
            # Check budget
            if not await self.cost_tracker.check_user_budget(user_id, estimated_cost=5.0):
                raise Exception("User budget exceeded")
            
            # Determine required agents
            required_agents = self._route_request(request_type, user_preferences.user_tier)
            
            # Execute agents in parallel where possible
            agent_outputs = {}
            total_cost = 0.0
            
            # Group agents by dependencies
            independent_agents = ['sentiment', 'price_forecast']  # Can run in parallel
            dependent_agents = ['options_strategy', 'risk_analysis', 'explanation']  # May need previous results
            
            # Execute independent agents first
            independent_tasks = []
            for agent_name in required_agents:
                if agent_name in independent_agents and agent_name in self.agents:
                    task = self._execute_agent(agent_name, payload, user_preferences, analysis_id)
                    independent_tasks.append((agent_name, task))
            
            # Wait for independent agents
            for agent_name, task in independent_tasks:
                try:
                    result = await task
                    agent_outputs[agent_name] = result['output']
                    total_cost += result['cost']
                    
                    if result.get('fallback_used'):
                        fallbacks_used.append(f"{agent_name}->{result['fallback_agent']}")
                        
                except Exception as e:
                    logger.error(f"Agent {agent_name} failed: {str(e)}")
                    # Try fallback
                    fallback_result = await self._execute_fallback(agent_name, payload, user_preferences)
                    if fallback_result:
                        agent_outputs[agent_name] = fallback_result['output']
                        total_cost += fallback_result['cost']
                        fallbacks_used.append(f"{agent_name}->fallback")
            
            # Execute dependent agents with previous results
            enhanced_payload = {**payload, 'previous_results': agent_outputs}
            
            for agent_name in required_agents:
                if agent_name in dependent_agents and agent_name in self.agents:
                    try:
                        result = await self._execute_agent(agent_name, enhanced_payload, user_preferences, analysis_id)
                        agent_outputs[agent_name] = result['output']
                        total_cost += result['cost']
                        
                        if result.get('fallback_used'):
                            fallbacks_used.append(f"{agent_name}->{result['fallback_agent']}")
                            
                    except Exception as e:
                        logger.error(f"Dependent agent {agent_name} failed: {str(e)}")
                        # Try fallback
                        fallback_result = await self._execute_fallback(agent_name, enhanced_payload, user_preferences)
                        if fallback_result:
                            agent_outputs[agent_name] = fallback_result['output']
                            total_cost += fallback_result['cost']
                            fallbacks_used.append(f"{agent_name}->fallback")
            
            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Log to database
            await self._log_workflow_execution(
                analysis_id, user_id, request_type, agent_outputs, 
                total_cost, duration_ms, fallbacks_used
            )
            
            # Update statistics
            await self._update_stats(True, total_cost, len(fallbacks_used), required_agents)
            
            return WorkflowResult(
                analysis_id=analysis_id,
                agent_outputs=agent_outputs,
                total_cost=total_cost,
                duration_ms=duration_ms,
                fallbacks_used=fallbacks_used,
                user_tier=user_preferences.user_tier
            )
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            await self._update_stats(False, 0.0, 0, [])
            logger.error(f"Workflow execution failed: {str(e)}")
            raise
    
    def _route_request(self, request_type: str, user_tier: str) -> List[str]:
        """Determine which agents should handle the request"""
        
        base_agents = self.routing_table.get(request_type, ['sentiment'])
        
        # Adjust for user tier
        if user_tier == 'free':
            # Use faster, cheaper models for free tier
            agent_mapping = {
                'options_strategy': 'strategy_lite',
                'explanation': 'finbert_explainer'
            }
            return [agent_mapping.get(agent, agent) for agent in base_agents]
        else:
            # Premium tier gets all advanced models
            return base_agents
    
    async def _execute_agent(
        self,
        agent_name: str,
        payload: Dict[str, Any],
        user_preferences: UserPreferences,
        analysis_id: str
    ) -> Dict[str, Any]:
        """Execute a single agent with circuit breaker protection"""
        
        if agent_name not in self.agents:
            raise Exception(f"Agent {agent_name} not available")
        
        agent = self.agents[agent_name]
        circuit_breaker = self.circuit_breakers[agent_name]
        
        try:
            # Execute with circuit breaker
            result = await circuit_breaker.call(
                agent.process,
                payload,
                timeout=user_preferences.latency_tolerance / 1000.0
            )
            
            # Calculate cost
            cost = await self._calculate_agent_cost(agent_name, result)
            
            # Log agent performance
            await self._log_agent_performance(
                agent_name, user_preferences, payload, result, 
                analysis_id, cost
            )
            
            return {
                'output': result,
                'cost': cost,
                'agent': agent_name,
                'fallback_used': False
            }
            
        except Exception as e:
            logger.warning(f"Agent {agent_name} failed with circuit breaker: {str(e)}")
            raise
    
    async def _execute_fallback(
        self,
        agent_name: str,
        payload: Dict[str, Any],
        user_preferences: UserPreferences
    ) -> Optional[Dict[str, Any]]:
        """Execute fallback agent when primary fails"""
        
        fallback_name = self.agent_configs[agent_name].get('fallback')
        if not fallback_name:
            return None
        
        try:
            # Use simplified processing for fallback
            fallback_result = await self._get_fallback_result(fallback_name, payload)
            
            cost = 0.001  # Minimal cost for fallback
            
            return {
                'output': fallback_result,
                'cost': cost,
                'agent': fallback_name,
                'fallback_used': True
            }
            
        except Exception as e:
            logger.error(f"Fallback {fallback_name} also failed: {str(e)}")
            return None
    
    async def _get_fallback_result(self, fallback_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback result using simple rules or cached responses"""
        
        fallback_responses = {
            'finbert_lite': {
                'sentiment': 'neutral',
                'confidence': 0.6,
                'source': 'fallback_rule_based'
            },
            'strategy_lite': {
                'strategy': 'HOLD',
                'reasoning': 'Conservative fallback recommendation',
                'risk_level': 'low',
                'source': 'fallback_conservative'
            },
            'finbert_explainer': {
                'explanation': 'Analysis temporarily unavailable. Using conservative approach.',
                'confidence': 0.5,
                'source': 'fallback_explanation'
            },
            'lstm_basic': {
                'prediction': payload.get('current_price', 100.0),
                'confidence': 0.3,
                'direction': 'neutral',
                'source': 'fallback_no_change'
            },
            'rule_based_risk': {
                'risk_score': 0.5,
                'risk_level': 'medium',
                'recommendations': ['Monitor position size', 'Set stop losses'],
                'source': 'fallback_conservative_risk'
            }
        }
        
        return fallback_responses.get(fallback_name, {
            'result': 'Service temporarily unavailable',
            'source': 'fallback_unavailable'
        })
    
    async def _calculate_agent_cost(self, agent_name: str, result: Dict[str, Any]) -> float:
        """Calculate cost for agent execution"""
        
        config = self.agent_configs[agent_name]
        base_cost = config['cost_per_1k_tokens']
        
        # Estimate tokens based on result size
        result_text = json.dumps(result)
        estimated_tokens = len(result_text) / 4  # Rough approximation
        
        return (estimated_tokens / 1000) * base_cost
    
    async def _log_agent_performance(
        self,
        agent_name: str,
        user_preferences: UserPreferences,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        analysis_id: str,
        cost: float
    ):
        """Log agent performance to database"""
        
        try:
            input_hash = hashlib.sha256(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
            output_hash = hashlib.sha256(json.dumps(output_data, sort_keys=True).encode()).hexdigest()
            
            await self.db.log_agent_performance(
                agent_type=agent_name,
                analysis_id=analysis_id,
                inference_time_ms=50,  # Will be measured properly in production
                input_hash=input_hash,
                output_hash=output_hash,
                tokens_used=int(cost * 1000 / self.agent_configs[agent_name]['cost_per_1k_tokens']),
                cost_usd=cost
            )
            
        except Exception as e:
            logger.error(f"Failed to log agent performance: {str(e)}")
    
    async def _log_workflow_execution(
        self,
        analysis_id: str,
        user_id: str,
        request_type: str,
        outputs: Dict[str, Any],
        cost: float,
        duration_ms: int,
        fallbacks: List[str]
    ):
        """Log complete workflow execution"""
        
        try:
            await self.db.log_workflow_execution(
                analysis_id=analysis_id,
                user_id=user_id,
                request_type=request_type,
                agent_outputs=outputs,
                total_cost=cost,
                duration_ms=duration_ms,
                fallbacks_used=fallbacks
            )
            
        except Exception as e:
            logger.error(f"Failed to log workflow execution: {str(e)}")
    
    async def _get_user_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences from database"""
        
        try:
            prefs = await self.db.get_user_llm_settings(user_id)
            if prefs:
                return UserPreferences(**prefs)
            else:
                # Return defaults for new users
                return UserPreferences()
                
        except Exception as e:
            logger.error(f"Failed to get user preferences: {str(e)}")
            return UserPreferences()
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user agent preferences"""
        
        try:
            await self.db.update_user_llm_settings(user_id, preferences)
            logger.info(f"Updated preferences for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to update user preferences: {str(e)}")
            raise
    
    async def get_agent_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all agents"""
        
        health_status = {}
        
        for agent_name, agent in self.agents.items():
            try:
                circuit_breaker = self.circuit_breakers[agent_name]
                
                health_status[agent_name] = {
                    'status': 'healthy' if circuit_breaker.state == 'CLOSED' else 'degraded',
                    'circuit_breaker_state': circuit_breaker.state,
                    'failure_count': circuit_breaker.failure_count,
                    'avg_latency_ms': await self._get_agent_avg_latency(agent_name),
                    'success_rate': await self._get_agent_success_rate(agent_name),
                    'queue_depth': await self._get_agent_queue_depth(agent_name)
                }
                
            except Exception as e:
                health_status[agent_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return health_status
    
    async def get_available_agents(self) -> List[str]:
        """Get list of available agent names"""
        return list(self.agents.keys())
    
    async def get_agent_performance(self, agent_type: str) -> Dict[str, Any]:
        """Get performance metrics for specific agent"""
        
        try:
            return await self.db.get_agent_performance_metrics(agent_type)
        except Exception as e:
            logger.error(f"Failed to get performance for {agent_type}: {str(e)}")
            return {}
    
    async def test_agent(self, agent_type: str, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test specific agent with sample data"""
        
        if agent_type not in self.agents:
            raise Exception(f"Agent {agent_type} not available")
        
        try:
            agent = self.agents[agent_type]
            result = await agent.process(test_data)
            
            return {
                'agent': agent_type,
                'test_data': test_data,
                'result': result,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'agent': agent_type,
                'test_data': test_data,
                'error': str(e),
                'status': 'failed'
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics"""
        return dict(self.stats)
    
    async def _update_stats(self, success: bool, cost: float, fallbacks: int, agents: List[str]):
        """Update internal statistics"""
        
        self.stats['total_requests'] += 1
        if success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
        
        self.stats['fallbacks_triggered'] += fallbacks
        self.stats['total_cost'] += cost
        
        for agent in agents:
            self.stats['agent_usage_count'][agent] = self.stats['agent_usage_count'].get(agent, 0) + 1
    
    async def _get_agent_avg_latency(self, agent_name: str) -> float:
        """Get average latency for agent from Redis metrics"""
        try:
            latency_key = f"agent_latency:{agent_name}"
            latency_data = await self.redis.lrange(latency_key, 0, 99)  # Last 100 measurements
            if latency_data:
                latencies = [float(x) for x in latency_data]
                return sum(latencies) / len(latencies)
            return 0.0
        except:
            return 0.0
    
    async def _get_agent_success_rate(self, agent_name: str) -> float:
        """Get success rate for agent"""
        try:
            success_key = f"agent_success:{agent_name}"
            total_key = f"agent_total:{agent_name}"
            
            success_count = await self.redis.get(success_key) or 0
            total_count = await self.redis.get(total_key) or 0
            
            if int(total_count) > 0:
                return float(success_count) / float(total_count)
            return 0.0
        except:
            return 0.0
    
    async def _get_agent_queue_depth(self, agent_name: str) -> int:
        """Get current queue depth for agent"""
        try:
            queue_key = f"agent_queue:{agent_name}"
            depth = await self.redis.llen(queue_key)
            return depth or 0
        except:
            return 0
    
    async def _health_monitor(self):
        """Background task to monitor agent health"""
        while True:
            try:
                for agent_name in self.agents:
                    # Check if agent is responsive
                    try:
                        await asyncio.wait_for(
                            self.agents[agent_name].health_check(),
                            timeout=5.0
                        )
                        # Reset circuit breaker on successful health check
                        self.circuit_breakers[agent_name].on_success()
                        
                    except:
                        # Mark as failed
                        self.circuit_breakers[agent_name].on_failure()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitor error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _performance_aggregator(self):
        """Background task to aggregate performance metrics"""
        while True:
            try:
                # Aggregate and store performance metrics
                for agent_name in self.agents:
                    avg_latency = await self._get_agent_avg_latency(agent_name)
                    success_rate = await self._get_agent_success_rate(agent_name)
                    
                    # Store in database for historical tracking
                    await self.db.store_performance_snapshot(
                        agent_name, avg_latency, success_rate
                    )
                
                await asyncio.sleep(300)  # Aggregate every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance aggregator error: {str(e)}")
                await asyncio.sleep(600)
    
    async def shutdown(self):
        """Shutdown orchestrator and cleanup resources"""
        logger.info("Shutting down Agent Orchestrator...")
        
        # Shutdown all agents
        for agent_name, agent in self.agents.items():
            try:
                await agent.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down agent {agent_name}: {str(e)}")
        
        # Close database connections
        if self.db:
            await self.db.close()
        
        logger.info("Agent Orchestrator shutdown complete")

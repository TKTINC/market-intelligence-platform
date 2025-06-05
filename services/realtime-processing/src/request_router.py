"""
Intelligent Request Router for optimizing agent selection and execution
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass 
class RoutingPlan:
    request_id: str
    agents_to_call: List[str]
    parallel_execution: bool
    timeout_ms: int
    fallback_strategy: str
    priority_order: List[str]
    estimated_cost: float
    estimated_time_ms: int

@dataclass
class AgentCapability:
    agent_name: str
    capabilities: List[str]
    avg_response_time_ms: int
    accuracy_score: float
    cost_per_request: float
    max_concurrent: int
    current_load: int

class IntelligentRequestRouter:
    def __init__(self):
        # Agent capabilities and performance
        self.agent_capabilities = {
            "sentiment": AgentCapability(
                agent_name="sentiment",
                capabilities=["sentiment_analysis", "news_analysis", "social_sentiment"],
                avg_response_time_ms=800,
                accuracy_score=0.85,
                cost_per_request=0.02,
                max_concurrent=20,
                current_load=0
            ),
            "llama_explanation": AgentCapability(
                agent_name="llama_explanation",
                capabilities=["explanations", "analysis_summary", "educational_content"],
                avg_response_time_ms=2500,
                accuracy_score=0.78,
                cost_per_request=0.15,
                max_concurrent=5,
                current_load=0
            ),
            "gpt4_strategy": AgentCapability(
                agent_name="gpt4_strategy", 
                capabilities=["options_strategies", "risk_analysis", "strategy_validation"],
                avg_response_time_ms=3200,
                accuracy_score=0.92,
                cost_per_request=0.45,
                max_concurrent=8,
                current_load=0
            ),
            "tft_forecasting": AgentCapability(
                agent_name="tft_forecasting",
                capabilities=["price_forecasting", "volatility_prediction", "regime_detection"],
                avg_response_time_ms=1800,
                accuracy_score=0.72,
                cost_per_request=0.08,
                max_concurrent=12,
                current_load=0
            )
        }
        
        # Routing strategies
        self.routing_strategies = {
            "speed_optimized": self._speed_optimized_routing,
            "accuracy_optimized": self._accuracy_optimized_routing,
            "cost_optimized": self._cost_optimized_routing,
            "balanced": self._balanced_routing
        }
        
        # User tier configurations
        self.tier_configs = {
            "free": {
                "max_agents": 2,
                "max_cost": 0.25,
                "timeout_ms": 10000,
                "parallel_execution": False
            },
            "basic": {
                "max_agents": 3,
                "max_cost": 1.00,
                "timeout_ms": 15000,
                "parallel_execution": True
            },
            "premium": {
                "max_agents": 4,
                "max_cost": 2.50,
                "timeout_ms": 20000,
                "parallel_execution": True
            },
            "enterprise": {
                "max_agents": 4,
                "max_cost": 5.00,
                "timeout_ms": 30000,
                "parallel_execution": True
            }
        }
    
    async def create_routing_plan(
        self,
        request: Any,
        user_info: Dict[str, Any]
    ) -> RoutingPlan:
        """Create optimized routing plan for request"""
        
        try:
            # Determine required capabilities
            required_capabilities = self._analyze_required_capabilities(request)
            
            # Get user tier configuration
            user_tier = user_info.get("tier", "free")
            tier_config = self.tier_configs[user_tier]
            
            # Select routing strategy
            routing_strategy = self._select_routing_strategy(request, user_tier)
            
            # Generate routing plan
            routing_plan = await routing_strategy(
                required_capabilities, tier_config, request
            )
            
            logger.info(f"Created routing plan: {routing_plan.agents_to_call} for request {routing_plan.request_id}")
            
            return routing_plan
            
        except Exception as e:
            logger.error(f"Routing plan creation failed: {e}")
            # Return fallback plan
            return self._create_fallback_plan(request)
    
    def _analyze_required_capabilities(self, request: Any) -> List[str]:
        """Analyze request to determine required capabilities"""
        
        required = []
        
        # Check analysis type
        analysis_type = getattr(request, 'analysis_type', 'standard')
        
        if analysis_type in ["comprehensive", "custom"]:
            required.extend([
                "sentiment_analysis", 
                "price_forecasting", 
                "explanations"
            ])
            
            if getattr(request, 'include_strategies', True):
                required.append("options_strategies")
                
        elif analysis_type == "standard":
            required.extend([
                "sentiment_analysis",
                "price_forecasting" 
            ])
            
            if getattr(request, 'include_explanations', True):
                required.append("explanations")
                
        elif analysis_type == "quick":
            required.extend([
                "sentiment_analysis"
            ])
        
        # Check specific agent requests
        agents_requested = getattr(request, 'agents_requested', None)
        if agents_requested:
            for agent in agents_requested:
                if agent in self.agent_capabilities:
                    required.extend(self.agent_capabilities[agent].capabilities)
        
        return list(set(required))  # Remove duplicates
    
    def _select_routing_strategy(self, request: Any, user_tier: str) -> callable:
        """Select optimal routing strategy"""
        
        # Priority-based strategy selection
        priority = getattr(request, 'priority', 'normal')
        
        if priority == "urgent":
            return self.routing_strategies["speed_optimized"]
        elif priority == "high":
            return self.routing_strategies["balanced"]
        elif user_tier in ["premium", "enterprise"]:
            return self.routing_strategies["accuracy_optimized"] 
        elif user_tier == "free":
            return self.routing_strategies["cost_optimized"]
        else:
            return self.routing_strategies["balanced"]
    
    async def _speed_optimized_routing(
        self,
        required_capabilities: List[str],
        tier_config: Dict[str, Any],
        request: Any
    ) -> RoutingPlan:
        """Speed-optimized routing strategy"""
        
        # Select fastest agents for each capability
        selected_agents = []
        total_cost = 0
        total_time = 0
        
        for capability in required_capabilities:
            # Find fastest agent for this capability
            fastest_agent = min(
                [a for a in self.agent_capabilities.values() 
                 if capability in a.capabilities and a.current_load < a.max_concurrent],
                key=lambda x: x.avg_response_time_ms,
                default=None
            )
            
            if fastest_agent and fastest_agent.agent_name not in selected_agents:
                selected_agents.append(fastest_agent.agent_name)
                total_cost += fastest_agent.cost_per_request
                total_time = max(total_time, fastest_agent.avg_response_time_ms)
        
        # Limit by tier configuration
        selected_agents = selected_agents[:tier_config["max_agents"]]
        
        return RoutingPlan(
            request_id=getattr(request, 'request_id', str(id(request))),
            agents_to_call=selected_agents,
            parallel_execution=True,  # Speed optimization
            timeout_ms=tier_config["timeout_ms"],
            fallback_strategy="fast_fallback",
            priority_order=selected_agents,
            estimated_cost=min(total_cost, tier_config["max_cost"]),
            estimated_time_ms=total_time
        )
    
    async def _accuracy_optimized_routing(
        self,
        required_capabilities: List[str],
        tier_config: Dict[str, Any],
        request: Any
    ) -> RoutingPlan:
        """Accuracy-optimized routing strategy"""
        
        # Select most accurate agents
        selected_agents = []
        total_cost = 0
        total_time = 0
        
        for capability in required_capabilities:
            # Find most accurate agent
            best_agent = max(
                [a for a in self.agent_capabilities.values() 
                 if capability in a.capabilities and a.current_load < a.max_concurrent],
                key=lambda x: x.accuracy_score,
                default=None
            )
            
            if best_agent and best_agent.agent_name not in selected_agents:
                selected_agents.append(best_agent.agent_name)
                total_cost += best_agent.cost_per_request
                total_time += best_agent.avg_response_time_ms  # Sequential for accuracy
        
        # Limit by tier configuration  
        selected_agents = selected_agents[:tier_config["max_agents"]]
        
        return RoutingPlan(
            request_id=getattr(request, 'request_id', str(id(request))),
            agents_to_call=selected_agents,
            parallel_execution=tier_config["parallel_execution"],
            timeout_ms=tier_config["timeout_ms"],
            fallback_strategy="accuracy_fallback",
            priority_order=selected_agents,
            estimated_cost=min(total_cost, tier_config["max_cost"]),
            estimated_time_ms=total_time if not tier_config["parallel_execution"] else max([
                self.agent_capabilities[a].avg_response_time_ms for a in selected_agents
            ])
        )
    
    async def _cost_optimized_routing(
        self,
        required_capabilities: List[str],
        tier_config: Dict[str, Any],
        request: Any
    ) -> RoutingPlan:
        """Cost-optimized routing strategy"""
        
        # Select cheapest agents that meet requirements
        selected_agents = []
        total_cost = 0
        
        # Sort capabilities by cost to prioritize cheap ones
        capability_agents = {}
        for capability in required_capabilities:
            agents = [a for a in self.agent_capabilities.values() 
                     if capability in a.capabilities and a.current_load < a.max_concurrent]
            if agents:
                capability_agents[capability] = min(agents, key=lambda x: x.cost_per_request)
        
        # Select agents within budget
        for capability, agent in capability_agents.items():
            if (agent.agent_name not in selected_agents and 
                total_cost + agent.cost_per_request <= tier_config["max_cost"]):
                selected_agents.append(agent.agent_name)
                total_cost += agent.cost_per_request
        
        return RoutingPlan(
            request_id=getattr(request, 'request_id', str(id(request))),
            agents_to_call=selected_agents,
            parallel_execution=tier_config["parallel_execution"],
            timeout_ms=tier_config["timeout_ms"],
            fallback_strategy="cost_fallback",
            priority_order=selected_agents,
            estimated_cost=total_cost,
            estimated_time_ms=sum([
                self.agent_capabilities[a].avg_response_time_ms for a in selected_agents
            ]) if not tier_config["parallel_execution"] else max([
                self.agent_capabilities[a].avg_response_time_ms for a in selected_agents
            ])
        )
    
    async def _balanced_routing(
        self,
        required_capabilities: List[str],
        tier_config: Dict[str, Any],
        request: Any
    ) -> RoutingPlan:
        """Balanced routing strategy considering speed, accuracy, and cost"""
        
        # Score agents based on balanced criteria
        agent_scores = {}
        
        for agent_name, agent in self.agent_capabilities.items():
            if agent.current_load >= agent.max_concurrent:
                continue
                
            # Calculate composite score (normalized)
            speed_score = 1 / (agent.avg_response_time_ms / 1000)  # Higher is better
            accuracy_score = agent.accuracy_score  # Higher is better
            cost_score = 1 / (agent.cost_per_request + 0.01)  # Higher is better (lower cost)
            
            # Weighted composite score
            composite_score = (speed_score * 0.3 + accuracy_score * 0.4 + cost_score * 0.3)
            agent_scores[agent_name] = composite_score
        
        # Select agents based on required capabilities and scores
        selected_agents = []
        total_cost = 0
        
        for capability in required_capabilities:
            # Find best-scoring agent for this capability
            candidate_agents = [
                (name, score) for name, score in agent_scores.items()
                if capability in self.agent_capabilities[name].capabilities
                and name not in selected_agents
            ]
            
            if candidate_agents:
                best_agent_name = max(candidate_agents, key=lambda x: x[1])[0]
                best_agent = self.agent_capabilities[best_agent_name]
                
                if total_cost + best_agent.cost_per_request <= tier_config["max_cost"]:
                    selected_agents.append(best_agent_name)
                    total_cost += best_agent.cost_per_request
        
        # Limit by tier configuration
        selected_agents = selected_agents[:tier_config["max_agents"]]
        
        return RoutingPlan(
            request_id=getattr(request, 'request_id', str(id(request))),
            agents_to_call=selected_agents,
            parallel_execution=tier_config["parallel_execution"],
            timeout_ms=tier_config["timeout_ms"],
            fallback_strategy="balanced_fallback",
            priority_order=selected_agents,
            estimated_cost=total_cost,
            estimated_time_ms=max([
                self.agent_capabilities[a].avg_response_time_ms for a in selected_agents
            ]) if tier_config["parallel_execution"] else sum([
                self.agent_capabilities[a].avg_response_time_ms for a in selected_agents
            ])
        )
    
    def _create_fallback_plan(self, request: Any) -> RoutingPlan:
        """Create fallback routing plan"""
        
        return RoutingPlan(
            request_id=getattr(request, 'request_id', str(id(request))),
            agents_to_call=["sentiment"],  # Minimal fallback
            parallel_execution=False,
            timeout_ms=5000,
            fallback_strategy="minimal",
            priority_order=["sentiment"],
            estimated_cost=0.02,
            estimated_time_ms=800
        )
    
    async def update_agent_performance(
        self,
        agent_name: str,
        response_time_ms: int,
        success: bool,
        cost: float
    ):
        """Update agent performance metrics"""
        
        try:
            if agent_name in self.agent_capabilities:
                agent = self.agent_capabilities[agent_name]
                
                # Update moving average of response time
                alpha = 0.1  # Smoothing factor
                agent.avg_response_time_ms = int(
                    alpha * response_time_ms + (1 - alpha) * agent.avg_response_time_ms
                )
                
                # Update accuracy score based on success
                if success:
                    agent.accuracy_score = min(0.99, agent.accuracy_score + 0.01)
                else:
                    agent.accuracy_score = max(0.01, agent.accuracy_score - 0.05)
                
                # Update cost
                agent.cost_per_request = alpha * cost + (1 - alpha) * agent.cost_per_request
                
        except Exception as e:
            logger.error(f"Agent performance update failed: {e}")
    
    async def update_agent_load(self, agent_name: str, load_change: int):
        """Update agent current load"""
        
        try:
            if agent_name in self.agent_capabilities:
                agent = self.agent_capabilities[agent_name]
                agent.current_load = max(0, agent.current_load + load_change)
                
        except Exception as e:
            logger.error(f"Agent load update failed: {e}")
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        
        return {
            "agent_capabilities": {
                name: {
                    "avg_response_time_ms": agent.avg_response_time_ms,
                    "accuracy_score": agent.accuracy_score,
                    "cost_per_request": agent.cost_per_request,
                    "current_load": agent.current_load,
                    "max_concurrent": agent.max_concurrent,
                    "utilization": agent.current_load / agent.max_concurrent
                }
                for name, agent in self.agent_capabilities.items()
            },
            "tier_configs": self.tier_configs,
            "routing_strategies": list(self.routing_strategies.keys())
        }

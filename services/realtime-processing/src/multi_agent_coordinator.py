"""
Multi-Agent Coordinator for orchestrating all MIP agents
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AgentEndpoint:
    name: str
    url: str
    health_endpoint: str
    timeout: int
    retry_count: int
    last_healthy: Optional[datetime] = None
    consecutive_failures: int = 0

@dataclass
class RoutingPlan:
    request_id: str
    agents_to_call: List[str]
    parallel_execution: bool
    timeout_ms: int
    fallback_strategy: str

class MultiAgentCoordinator:
    def __init__(self):
        # Agent configurations
        self.agents = {
            "sentiment": AgentEndpoint(
                name="sentiment",
                url="http://finbert-sentiment-service:8005",
                health_endpoint="/health",
                timeout=10,
                retry_count=2
            ),
            "llama_explanation": AgentEndpoint(
                name="llama_explanation", 
                url="http://llama-explanation-service:8006",
                health_endpoint="/health",
                timeout=30,
                retry_count=1
            ),
            "gpt4_strategy": AgentEndpoint(
                name="gpt4_strategy",
                url="http://gpt4-strategy-service:8007",
                health_endpoint="/health", 
                timeout=20,
                retry_count=2
            ),
            "tft_forecasting": AgentEndpoint(
                name="tft_forecasting",
                url="http://tft-forecasting-service:8008",
                health_endpoint="/health",
                timeout=15,
                retry_count=2
            )
        }
        
        # HTTP session for agent communication
        self.session = None
        
        # Circuit breaker states
        self.circuit_breakers = {agent: False for agent in self.agents.keys()}
        
        # Load balancing and failover
        self.agent_load = {agent: 0 for agent in self.agents.keys()}
        self.max_concurrent_requests = 50
        
    async def initialize_agents(self):
        """Initialize HTTP session and test agent connections"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
            
            # Test all agent connections
            for agent_name, agent in self.agents.items():
                try:
                    await self._check_agent_health(agent_name)
                    logger.info(f"Agent {agent_name} is healthy")
                except Exception as e:
                    logger.warning(f"Agent {agent_name} health check failed: {e}")
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise
    
    async def execute_coordinated_request(
        self,
        routing_plan: RoutingPlan,
        market_data: Dict[str, Any],
        request_id: str
    ) -> Dict[str, Any]:
        """Execute coordinated request across multiple agents"""
        
        try:
            agent_responses = {}
            
            if routing_plan.parallel_execution:
                # Execute agents in parallel
                tasks = []
                for agent_name in routing_plan.agents_to_call:
                    if self._is_agent_available(agent_name):
                        task = self._call_agent(agent_name, market_data, request_id)
                        tasks.append((agent_name, task))
                
                # Wait for all tasks with timeout
                results = await asyncio.gather(
                    *[task for _, task in tasks],
                    return_exceptions=True
                )
                
                # Process results
                for i, (agent_name, _) in enumerate(tasks):
                    result = results[i]
                    if isinstance(result, Exception):
                        logger.error(f"Agent {agent_name} failed: {result}")
                        self._handle_agent_failure(agent_name)
                        agent_responses[agent_name] = {"error": str(result)}
                    else:
                        agent_responses[agent_name] = result
                        self._handle_agent_success(agent_name)
            
            else:
                # Execute agents sequentially
                for agent_name in routing_plan.agents_to_call:
                    if self._is_agent_available(agent_name):
                        try:
                            response = await self._call_agent(agent_name, market_data, request_id)
                            agent_responses[agent_name] = response
                            self._handle_agent_success(agent_name)
                            
                            # Pass response to next agent if needed
                            if agent_name == "tft_forecasting" and "gpt4_strategy" in routing_plan.agents_to_call:
                                market_data["price_forecasts"] = response
                                
                        except Exception as e:
                            logger.error(f"Sequential agent {agent_name} failed: {e}")
                            self._handle_agent_failure(agent_name)
                            agent_responses[agent_name] = {"error": str(e)}
            
            return agent_responses
            
        except Exception as e:
            logger.error(f"Coordinated request execution failed: {e}")
            raise
    
    async def _call_agent(
        self,
        agent_name: str,
        market_data: Dict[str, Any],
        request_id: str
    ) -> Dict[str, Any]:
        """Call a specific agent with market data"""
        
        agent = self.agents[agent_name]
        
        try:
            # Increment agent load
            self.agent_load[agent_name] += 1
            
            # Prepare agent-specific request
            request_payload = self._prepare_agent_request(agent_name, market_data, request_id)
            
            # Make HTTP request
            async with self.session.post(
                f"{agent.url}{self._get_agent_endpoint(agent_name)}",
                json=request_payload,
                headers={"Authorization": "Bearer internal_service_token"},
                timeout=agent.timeout
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Agent {agent_name} call failed: {e}")
            raise
            
        finally:
            # Decrement agent load
            self.agent_load[agent_name] = max(0, self.agent_load[agent_name] - 1)
    
    def _prepare_agent_request(
        self,
        agent_name: str,
        market_data: Dict[str, Any],
        request_id: str
    ) -> Dict[str, Any]:
        """Prepare agent-specific request payload"""
        
        base_request = {
            "request_id": request_id,
            "symbols": market_data.get("symbols", []),
            "market_context": market_data.get("market_context", {})
        }
        
        if agent_name == "sentiment":
            return {
                **base_request,
                "news_sources": ["reuters", "bloomberg", "cnbc"],
                "analysis_depth": "comprehensive",
                "include_social_sentiment": True
            }
            
        elif agent_name == "llama_explanation":
            return {
                **base_request,
                "explanation_type": "market_analysis",
                "max_length": 500,
                "include_technical_analysis": True,
                "sentiment_data": market_data.get("sentiment_analysis"),
                "forecast_data": market_data.get("price_forecasts")
            }
            
        elif agent_name == "gpt4_strategy":
            return {
                **base_request,
                "user_intent": "Generate comprehensive options strategies",
                "portfolio_context": market_data.get("portfolio_context", {}),
                "risk_preferences": {"risk_tolerance": "medium"},
                "max_cost_usd": 1.0,
                "priority": "normal"
            }
            
        elif agent_name == "tft_forecasting":
            return {
                **base_request,
                "forecast_horizons": [1, 5, 10, 21],
                "include_options_greeks": True,
                "risk_adjustment": True,
                "confidence_intervals": [0.68, 0.95]
            }
        
        return base_request
    
    def _get_agent_endpoint(self, agent_name: str) -> str:
        """Get the appropriate endpoint for each agent"""
        
        endpoints = {
            "sentiment": "/sentiment/analyze",
            "llama_explanation": "/explanation/generate",
            "gpt4_strategy": "/strategy/generate", 
            "tft_forecasting": "/forecast/generate"
        }
        
        return endpoints.get(agent_name, "/")
    
    async def check_all_agents_health(self) -> Dict[str, Any]:
        """Check health of all agents"""
        
        health_results = {}
        overall_healthy = True
        
        for agent_name in self.agents.keys():
            try:
                health_status = await self._check_agent_health(agent_name)
                health_results[agent_name] = {
                    "status": "healthy",
                    "response_time_ms": health_status,
                    "last_checked": datetime.utcnow().isoformat(),
                    "load": self.agent_load[agent_name],
                    "circuit_breaker": self.circuit_breakers[agent_name]
                }
            except Exception as e:
                health_results[agent_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_checked": datetime.utcnow().isoformat(),
                    "load": self.agent_load[agent_name],
                    "circuit_breaker": self.circuit_breakers[agent_name]
                }
                overall_healthy = False
        
        health_results["overall_healthy"] = overall_healthy
        health_results["total_load"] = sum(self.agent_load.values())
        
        return health_results
    
    async def _check_agent_health(self, agent_name: str) -> int:
        """Check health of a specific agent"""
        
        agent = self.agents[agent_name]
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.session.get(
                f"{agent.url}{agent.health_endpoint}",
                timeout=5
            ) as response:
                
                if response.status == 200:
                    end_time = asyncio.get_event_loop().time()
                    response_time_ms = int((end_time - start_time) * 1000)
                    
                    agent.last_healthy = datetime.utcnow()
                    agent.consecutive_failures = 0
                    
                    return response_time_ms
                else:
                    raise Exception(f"HTTP {response.status}")
                    
        except Exception as e:
            agent.consecutive_failures += 1
            raise Exception(f"Health check failed: {e}")
    
    def _is_agent_available(self, agent_name: str) -> bool:
        """Check if agent is available (not circuit broken)"""
        
        if self.circuit_breakers[agent_name]:
            return False
            
        agent = self.agents[agent_name]
        
        # Check if agent has too many consecutive failures
        if agent.consecutive_failures >= 3:
            self.circuit_breakers[agent_name] = True
            logger.warning(f"Circuit breaker activated for agent {agent_name}")
            return False
            
        # Check load
        if self.agent_load[agent_name] >= self.max_concurrent_requests:
            logger.warning(f"Agent {agent_name} at maximum load")
            return False
            
        return True
    
    def _handle_agent_success(self, agent_name: str):
        """Handle successful agent response"""
        
        agent = self.agents[agent_name]
        agent.consecutive_failures = 0
        agent.last_healthy = datetime.utcnow()
        
        # Reset circuit breaker if it was activated
        if self.circuit_breakers[agent_name]:
            self.circuit_breakers[agent_name] = False
            logger.info(f"Circuit breaker reset for agent {agent_name}")
    
    def _handle_agent_failure(self, agent_name: str):
        """Handle agent failure"""
        
        agent = self.agents[agent_name]
        agent.consecutive_failures += 1
        
        # Activate circuit breaker if too many failures
        if agent.consecutive_failures >= 3 and not self.circuit_breakers[agent_name]:
            self.circuit_breakers[agent_name] = True
            logger.error(f"Circuit breaker activated for agent {agent_name} after {agent.consecutive_failures} failures")
    
    async def get_detailed_agent_status(self) -> Dict[str, Any]:
        """Get detailed status of all agents"""
        
        status = await self.check_all_agents_health()
        
        # Add additional metrics
        for agent_name, agent in self.agents.items():
            if agent_name in status:
                status[agent_name].update({
                    "url": agent.url,
                    "timeout": agent.timeout,
                    "retry_count": agent.retry_count,
                    "consecutive_failures": agent.consecutive_failures,
                    "last_healthy": agent.last_healthy.isoformat() if agent.last_healthy else None
                })
        
        return status
    
    async def restart_agent(self, agent_name: str) -> Dict[str, Any]:
        """Restart a specific agent (simulate restart by resetting state)"""
        
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        try:
            # Reset agent state
            agent = self.agents[agent_name]
            agent.consecutive_failures = 0
            self.circuit_breakers[agent_name] = False
            self.agent_load[agent_name] = 0
            
            # Test health
            response_time = await self._check_agent_health(agent_name)
            
            return {
                "agent_name": agent_name,
                "status": "restarted",
                "health_check_ms": response_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Agent restart failed for {agent_name}: {e}")
            raise
    
    async def identify_unhealthy_agents(self) -> List[str]:
        """Identify agents that are currently unhealthy"""
        
        unhealthy_agents = []
        
        for agent_name in self.agents.keys():
            try:
                await self._check_agent_health(agent_name)
            except Exception:
                unhealthy_agents.append(agent_name)
        
        return unhealthy_agents
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

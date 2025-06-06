"""
Agent Orchestrator for coordinating all MIP AI agents
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import time

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    def __init__(self):
        # Agent service configurations
        self.agents = {
            "sentiment": {
                "url": "http://finbert-sentiment-service:8005",
                "endpoint": "/sentiment/analyze",
                "timeout": 10,
                "cost_per_request": 0.02,
                "max_symbols": 10
            },
            "forecasting": {
                "url": "http://tft-forecasting-service:8008", 
                "endpoint": "/forecast/generate",
                "timeout": 15,
                "cost_per_request": 0.08,
                "max_symbols": 5
            },
            "strategy": {
                "url": "http://gpt4-strategy-service:8007",
                "endpoint": "/strategy/generate",
                "timeout": 20,
                "cost_per_request": 0.45,
                "max_symbols": 3
            },
            "explanation": {
                "url": "http://llama-explanation-service:8006",
                "endpoint": "/explanation/generate",
                "timeout": 30,
                "cost_per_request": 0.15,
                "max_symbols": 5
            }
        }
        
        # Real-time processing service
        self.realtime_service = {
            "url": "http://realtime-processing-service:8008",
            "endpoint": "/intelligence/unified"
        }
        
        # HTTP session for agent communication
        self.session = None
        
        # Performance tracking
        self.agent_performance = {agent: {"calls": 0, "errors": 0, "avg_time": 0} 
                                 for agent in self.agents.keys()}
        
        # User tier limits
        self.tier_limits = {
            "free": {"max_cost": 0.25, "agents": ["sentiment"]},
            "basic": {"max_cost": 1.00, "agents": ["sentiment", "forecasting"]},
            "premium": {"max_cost": 2.50, "agents": ["sentiment", "forecasting", "strategy"]},
            "enterprise": {"max_cost": 5.00, "agents": ["sentiment", "forecasting", "strategy", "explanation"]}
        }
        
    async def initialize(self):
        """Initialize the agent orchestrator"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
            
            # Test connections to all agents
            for agent_name, config in self.agents.items():
                try:
                    await self._test_agent_connection(agent_name)
                    logger.info(f"Agent {agent_name} connection verified")
                except Exception as e:
                    logger.warning(f"Agent {agent_name} connection failed: {e}")
            
            logger.info("Agent orchestrator initialized")
            
        except Exception as e:
            logger.error(f"Agent orchestrator initialization failed: {e}")
            raise
    
    async def close(self):
        """Close the agent orchestrator"""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> str:
        """Check health of agent orchestrator"""
        try:
            if not self.session:
                return "unhealthy - no session"
            
            # Test connection to real-time service
            async with self.session.get(f"{self.realtime_service['url']}/health") as response:
                if response.status == 200:
                    return "healthy"
                else:
                    return f"degraded - realtime service status {response.status}"
                    
        except Exception as e:
            logger.error(f"Agent orchestrator health check failed: {e}")
            return "unhealthy"
    
    async def execute_analysis(
        self,
        request_id: str,
        user_id: str,
        symbols: List[str],
        agents: List[str],
        analysis_depth: str = "standard",
        include_explanations: bool = True,
        max_cost_usd: float = 1.0,
        user_tier: str = "free"
    ) -> Dict[str, Any]:
        """Execute coordinated agent analysis"""
        
        try:
            start_time = time.time()
            
            # Validate user tier and costs
            available_agents = self._validate_user_access(agents, user_tier, max_cost_usd)
            
            if not available_agents:
                raise Exception("No agents available for user tier")
            
            # Use real-time processing service for coordination
            request_payload = {
                "user_id": user_id,
                "symbols": symbols,
                "analysis_type": analysis_depth,
                "agents_requested": available_agents,
                "include_explanations": include_explanations,
                "include_strategies": "strategy" in available_agents,
                "include_forecasts": "forecasting" in available_agents,
                "max_cost_usd": max_cost_usd,
                "real_time_data": True
            }
            
            # Call real-time processing service
            async with self.session.post(
                f"{self.realtime_service['url']}{self.realtime_service['endpoint']}",
                json=request_payload,
                headers={"Authorization": "Bearer internal_service_token"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Process and standardize the response
                    processed_result = self._process_unified_response(result, available_agents)
                    
                    # Update performance metrics
                    processing_time = time.time() - start_time
                    await self._update_performance_metrics(available_agents, processing_time, True)
                    
                    return processed_result
                    
                else:
                    error_text = await response.text()
                    raise Exception(f"Real-time service error {response.status}: {error_text}")
            
        except Exception as e:
            logger.error(f"Agent analysis execution failed: {e}")
            
            # Fallback to direct agent calls
            return await self._execute_direct_analysis(
                request_id, user_id, symbols, available_agents, analysis_depth, max_cost_usd
            )
    
    async def _execute_direct_analysis(
        self,
        request_id: str,
        user_id: str,
        symbols: List[str],
        agents: List[str],
        analysis_depth: str,
        max_cost_usd: float
    ) -> Dict[str, Any]:
        """Execute analysis by calling agents directly"""
        
        try:
            results = {}
            total_cost = 0.0
            successful_agents = []
            
            # Execute agents in parallel
            tasks = []
            for agent_name in agents:
                if agent_name in self.agents:
                    task = self._call_agent_direct(agent_name, symbols, analysis_depth)
                    tasks.append((agent_name, task))
            
            # Wait for all agent calls
            if tasks:
                agent_results = await asyncio.gather(
                    *[task for _, task in tasks], return_exceptions=True
                )
                
                # Process results
                for i, (agent_name, _) in enumerate(tasks):
                    result = agent_results[i]
                    
                    if isinstance(result, Exception):
                        logger.error(f"Direct agent call failed for {agent_name}: {result}")
                        await self._update_performance_metrics([agent_name], 0, False)
                    else:
                        results[agent_name] = result
                        successful_agents.append(agent_name)
                        total_cost += self.agents[agent_name]["cost_per_request"]
                        await self._update_performance_metrics([agent_name], 0, True)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(results)
            
            return {
                "agents_used": successful_agents,
                "total_cost": total_cost,
                "sentiment_analysis": results.get("sentiment"),
                "price_forecasts": results.get("forecasting"),
                "strategy_recommendations": results.get("strategy"),
                "explanations": results.get("explanation"),
                "overall_confidence": overall_confidence,
                "analysis_method": "direct"
            }
            
        except Exception as e:
            logger.error(f"Direct agent analysis failed: {e}")
            raise
    
    async def _call_agent_direct(
        self, 
        agent_name: str, 
        symbols: List[str], 
        analysis_depth: str
    ) -> Dict[str, Any]:
        """Call an agent directly"""
        
        agent_config = self.agents[agent_name]
        
        try:
            # Prepare agent-specific request
            request_payload = self._prepare_agent_request(agent_name, symbols, analysis_depth)
            
            # Make HTTP request
            async with self.session.post(
                f"{agent_config['url']}{agent_config['endpoint']}",
                json=request_payload,
                timeout=agent_config["timeout"]
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Agent {agent_name} error {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Direct agent call failed for {agent_name}: {e}")
            raise
    
    def _prepare_agent_request(
        self, 
        agent_name: str, 
        symbols: List[str], 
        analysis_depth: str
    ) -> Dict[str, Any]:
        """Prepare agent-specific request payload"""
        
        base_request = {
            "symbols": symbols[:self.agents[agent_name]["max_symbols"]],
            "analysis_depth": analysis_depth
        }
        
        if agent_name == "sentiment":
            return {
                **base_request,
                "news_sources": ["reuters", "bloomberg", "cnbc"],
                "include_social_sentiment": True
            }
            
        elif agent_name == "forecasting":
            return {
                **base_request,
                "forecast_horizons": [1, 5, 10, 21],
                "include_options_greeks": True,
                "confidence_intervals": [0.68, 0.95]
            }
            
        elif agent_name == "strategy":
            return {
                **base_request,
                "user_intent": "Generate options strategies",
                "risk_tolerance": "medium",
                "max_cost_usd": 1.0
            }
            
        elif agent_name == "explanation":
            return {
                **base_request,
                "explanation_type": "market_analysis",
                "max_length": 500,
                "include_technical_analysis": True
            }
        
        return base_request
    
    def _validate_user_access(
        self, 
        requested_agents: List[str], 
        user_tier: str, 
        max_cost_usd: float
    ) -> List[str]:
        """Validate user can access requested agents within cost limits"""
        
        tier_config = self.tier_limits.get(user_tier, self.tier_limits["free"])
        available_agents = []
        total_cost = 0.0
        
        for agent_name in requested_agents:
            if agent_name in tier_config["agents"]:
                agent_cost = self.agents[agent_name]["cost_per_request"]
                
                if total_cost + agent_cost <= min(max_cost_usd, tier_config["max_cost"]):
                    available_agents.append(agent_name)
                    total_cost += agent_cost
        
        return available_agents
    
    def _process_unified_response(
        self, 
        unified_result: Dict[str, Any], 
        requested_agents: List[str]
    ) -> Dict[str, Any]:
        """Process and standardize unified response from real-time service"""
        
        try:
            # Extract data from unified response
            processed = {
                "agents_used": unified_result.get("agents_used", []),
                "total_cost": self._calculate_total_cost(unified_result.get("agents_used", [])),
                "overall_confidence": unified_result.get("confidence_score", 0.5),
                "analysis_method": "unified"
            }
            
            # Extract agent-specific results
            if "sentiment_analysis" in unified_result:
                processed["sentiment_analysis"] = unified_result["sentiment_analysis"]
            
            if "price_forecasts" in unified_result:
                processed["price_forecasts"] = unified_result["price_forecasts"]
            
            if "strategy_recommendations" in unified_result:
                processed["strategy_recommendations"] = unified_result["strategy_recommendations"]
            
            if "explanations" in unified_result:
                processed["explanations"] = unified_result["explanations"]
            
            return processed
            
        except Exception as e:
            logger.error(f"Unified response processing failed: {e}")
            return {
                "agents_used": [],
                "total_cost": 0.0,
                "overall_confidence": 0.0,
                "analysis_method": "unified",
                "error": str(e)
            }
    
    def _calculate_total_cost(self, agents_used: List[str]) -> float:
        """Calculate total cost for agents used"""
        return sum(self.agents[agent]["cost_per_request"] for agent in agents_used if agent in self.agents)
    
    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence from agent results"""
        
        if not results:
            return 0.0
        
        confidences = []
        
        for agent_name, result in results.items():
            if isinstance(result, dict):
                confidence = result.get("confidence_score", 0.5)
                confidences.append(confidence)
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    async def _update_performance_metrics(
        self, 
        agents: List[str], 
        processing_time: float, 
        success: bool
    ):
        """Update performance metrics for agents"""
        
        try:
            for agent_name in agents:
                if agent_name in self.agent_performance:
                    metrics = self.agent_performance[agent_name]
                    metrics["calls"] += 1
                    
                    if success:
                        # Update average time (exponential moving average)
                        alpha = 0.1
                        metrics["avg_time"] = alpha * processing_time + (1 - alpha) * metrics["avg_time"]
                    else:
                        metrics["errors"] += 1
                        
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    async def _test_agent_connection(self, agent_name: str):
        """Test connection to a specific agent"""
        
        agent_config = self.agents[agent_name]
        
        try:
            async with self.session.get(
                f"{agent_config['url']}/health",
                timeout=5
            ) as response:
                if response.status != 200:
                    raise Exception(f"Health check failed with status {response.status}")
                    
        except Exception as e:
            raise Exception(f"Connection test failed: {e}")
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        
        try:
            agent_status = {}
            
            for agent_name, config in self.agents.items():
                try:
                    start_time = time.time()
                    await self._test_agent_connection(agent_name)
                    response_time = (time.time() - start_time) * 1000
                    
                    metrics = self.agent_performance[agent_name]
                    
                    agent_status[agent_name] = {
                        "status": "healthy",
                        "url": config["url"],
                        "response_time_ms": round(response_time, 1),
                        "cost_per_request": config["cost_per_request"],
                        "max_symbols": config["max_symbols"],
                        "total_calls": metrics["calls"],
                        "error_count": metrics["errors"],
                        "avg_time_ms": round(metrics["avg_time"] * 1000, 1),
                        "success_rate": (metrics["calls"] - metrics["errors"]) / max(1, metrics["calls"])
                    }
                    
                except Exception as e:
                    agent_status[agent_name] = {
                        "status": "unhealthy",
                        "error": str(e),
                        "url": config["url"]
                    }
            
            return agent_status
            
        except Exception as e:
            logger.error(f"Agent status retrieval failed: {e}")
            return {"error": str(e)}
    
    async def get_detailed_agent_status(self) -> Dict[str, Any]:
        """Get detailed status including real-time service"""
        
        try:
            basic_status = await self.get_agent_status()
            
            # Add real-time service status
            try:
                async with self.session.get(f"{self.realtime_service['url']}/health") as response:
                    if response.status == 200:
                        rt_health = await response.json()
                        basic_status["realtime_processing"] = {
                            "status": "healthy",
                            "url": self.realtime_service["url"],
                            "components": rt_health.get("components", {}),
                            "active_connections": rt_health.get("active_connections", 0)
                        }
                    else:
                        basic_status["realtime_processing"] = {
                            "status": "degraded",
                            "url": self.realtime_service["url"],
                            "http_status": response.status
                        }
            except Exception as e:
                basic_status["realtime_processing"] = {
                    "status": "unhealthy",
                    "url": self.realtime_service["url"],
                    "error": str(e)
                }
            
            # Add tier configurations
            basic_status["tier_configurations"] = self.tier_limits
            
            return basic_status
            
        except Exception as e:
            logger.error(f"Detailed agent status retrieval failed: {e}")
            return {"error": str(e)}
    
    async def clear_cache(self):
        """Clear any cached data"""
        try:
            # Reset performance metrics
            self.agent_performance = {agent: {"calls": 0, "errors": 0, "avg_time": 0} 
                                     for agent in self.agents.keys()}
            
            # Call cache clear on real-time service
            try:
                async with self.session.delete(
                    f"{self.realtime_service['url']}/cache/clear",
                    headers={"Authorization": "Bearer internal_service_token"}
                ) as response:
                    if response.status == 200:
                        logger.info("Real-time service cache cleared")
            except Exception as e:
                logger.warning(f"Real-time service cache clear failed: {e}")
                
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            raise

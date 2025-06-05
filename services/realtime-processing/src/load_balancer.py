"""
Agent Load Balancer with Circuit Breakers and Health Monitoring
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Circuit breaker tripped
    HALF_OPEN = "half_open" # Testing if service recovered

@dataclass
class AgentMetrics:
    agent_name: str
    current_load: int
    max_concurrent: int
    avg_response_time_ms: float
    success_rate: float
    error_count_1min: int
    error_count_5min: int
    total_requests: int
    last_request_time: datetime
    last_success_time: datetime
    health_score: float

@dataclass
class CircuitBreakerState:
    state: CircuitState
    failure_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    timeout_duration: int  # seconds
    next_attempt_time: Optional[datetime]
    half_open_requests: int
    max_half_open_requests: int = 3

@dataclass
class LoadBalancingStrategy:
    name: str
    weight_response_time: float
    weight_success_rate: float
    weight_current_load: float
    weight_health_score: float

class AgentLoadBalancer:
    def __init__(self):
        # Agent metrics tracking
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        
        # Circuit breaker states
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
        # Load balancing strategies
        self.strategies = {
            "round_robin": LoadBalancingStrategy("round_robin", 0, 0, 0, 0),
            "least_connections": LoadBalancingStrategy("least_connections", 0.1, 0.1, 0.7, 0.1),
            "response_time": LoadBalancingStrategy("response_time", 0.7, 0.2, 0.05, 0.05),
            "adaptive": LoadBalancingStrategy("adaptive", 0.3, 0.3, 0.2, 0.2),
            "health_based": LoadBalancingStrategy("health_based", 0.2, 0.2, 0.1, 0.5)
        }
        
        # Configuration
        self.config = {
            "circuit_breaker_threshold": 5,  # failures before opening
            "circuit_breaker_timeout": 60,   # seconds
            "health_check_interval": 30,     # seconds
            "metrics_window": 300,           # 5 minutes
            "max_response_time_threshold": 10000,  # 10 seconds
            "min_success_rate_threshold": 0.8,     # 80%
        }
        
        # Request routing history
        self.routing_history = []
        self.round_robin_index = 0
        
        # Performance tracking
        self.performance_metrics = {
            "total_requests_routed": 0,
            "total_agent_failures": 0,
            "average_routing_time_ms": 0.0,
            "circuit_breaker_activations": 0
        }
        
        # Background tasks
        self.background_tasks = []
        
    async def start(self):
        """Start the load balancer"""
        try:
            # Initialize agent metrics
            agent_names = ["sentiment", "llama_explanation", "gpt4_strategy", "tft_forecasting"]
            
            for agent_name in agent_names:
                self.agent_metrics[agent_name] = AgentMetrics(
                    agent_name=agent_name,
                    current_load=0,
                    max_concurrent=self._get_agent_max_concurrent(agent_name),
                    avg_response_time_ms=self._get_agent_baseline_response_time(agent_name),
                    success_rate=1.0,
                    error_count_1min=0,
                    error_count_5min=0,
                    total_requests=0,
                    last_request_time=datetime.utcnow(),
                    last_success_time=datetime.utcnow(),
                    health_score=1.0
                )
                
                self.circuit_breakers[agent_name] = CircuitBreakerState(
                    state=CircuitState.CLOSED,
                    failure_count=0,
                    last_failure_time=None,
                    last_success_time=datetime.utcnow(),
                    timeout_duration=self.config["circuit_breaker_timeout"],
                    next_attempt_time=None,
                    half_open_requests=0
                )
            
            # Start background tasks
            self.background_tasks = [
                asyncio.create_task(self._health_monitor()),
                asyncio.create_task(self._metrics_collector()),
                asyncio.create_task(self._circuit_breaker_manager())
            ]
            
            logger.info("Agent load balancer started successfully")
            
        except Exception as e:
            logger.error(f"Load balancer startup failed: {e}")
            raise
    
    async def stop(self):
        """Stop the load balancer"""
        try:
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            logger.info("Agent load balancer stopped")
            
        except Exception as e:
            logger.error(f"Load balancer shutdown error: {e}")
    
    def select_best_agent(
        self, 
        available_agents: List[str],
        strategy: str = "adaptive"
    ) -> Optional[str]:
        """Select the best agent based on load balancing strategy"""
        
        try:
            start_time = time.time()
            
            # Filter out agents with open circuit breakers
            healthy_agents = [
                agent for agent in available_agents
                if self._is_agent_available(agent)
            ]
            
            if not healthy_agents:
                logger.warning("No healthy agents available")
                return None
            
            # Apply load balancing strategy
            if strategy == "round_robin":
                selected_agent = self._round_robin_selection(healthy_agents)
            elif strategy == "least_connections":
                selected_agent = self._least_connections_selection(healthy_agents)
            elif strategy == "response_time":
                selected_agent = self._response_time_selection(healthy_agents)
            elif strategy == "health_based":
                selected_agent = self._health_based_selection(healthy_agents)
            else:  # adaptive
                selected_agent = self._adaptive_selection(healthy_agents)
            
            # Update metrics
            if selected_agent:
                self.performance_metrics["total_requests_routed"] += 1
                routing_time_ms = (time.time() - start_time) * 1000
                self._update_routing_time_metric(routing_time_ms)
                
                # Record routing decision
                self.routing_history.append({
                    "timestamp": datetime.utcnow(),
                    "selected_agent": selected_agent,
                    "available_agents": healthy_agents,
                    "strategy": strategy,
                    "routing_time_ms": routing_time_ms
                })
                
                # Trim history
                if len(self.routing_history) > 1000:
                    self.routing_history = self.routing_history[-500:]
            
            return selected_agent
            
        except Exception as e:
            logger.error(f"Agent selection failed: {e}")
            return healthy_agents[0] if healthy_agents else None
    
    def _is_agent_available(self, agent_name: str) -> bool:
        """Check if agent is available for requests"""
        
        try:
            if agent_name not in self.agent_metrics:
                return False
            
            metrics = self.agent_metrics[agent_name]
            circuit_breaker = self.circuit_breakers[agent_name]
            
            # Check circuit breaker state
            if circuit_breaker.state == CircuitState.OPEN:
                # Check if timeout has passed
                if (circuit_breaker.next_attempt_time and 
                    datetime.utcnow() >= circuit_breaker.next_attempt_time):
                    circuit_breaker.state = CircuitState.HALF_OPEN
                    circuit_breaker.half_open_requests = 0
                    logger.info(f"Circuit breaker for {agent_name} moved to HALF_OPEN")
                else:
                    return False
            
            # Check if agent is at capacity
            if metrics.current_load >= metrics.max_concurrent:
                return False
            
            # Check health score threshold
            if metrics.health_score < 0.3:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Agent availability check failed for {agent_name}: {e}")
            return False
    
    def _round_robin_selection(self, agents: List[str]) -> str:
        """Simple round-robin selection"""
        
        if not agents:
            return None
        
        selected_agent = agents[self.round_robin_index % len(agents)]
        self.round_robin_index += 1
        
        return selected_agent
    
    def _least_connections_selection(self, agents: List[str]) -> str:
        """Select agent with least current connections"""
        
        if not agents:
            return None
        
        return min(agents, key=lambda agent: self.agent_metrics[agent].current_load)
    
    def _response_time_selection(self, agents: List[str]) -> str:
        """Select agent with best response time"""
        
        if not agents:
            return None
        
        return min(agents, key=lambda agent: self.agent_metrics[agent].avg_response_time_ms)
    
    def _health_based_selection(self, agents: List[str]) -> str:
        """Select agent with best health score"""
        
        if not agents:
            return None
        
        return max(agents, key=lambda agent: self.agent_metrics[agent].health_score)
    
    def _adaptive_selection(self, agents: List[str]) -> str:
        """Adaptive selection based on multiple factors"""
        
        if not agents:
            return None
        
        strategy = self.strategies["adaptive"]
        agent_scores = {}
        
        for agent in agents:
            metrics = self.agent_metrics[agent]
            
            # Normalize metrics (0-1 scale, higher is better)
            response_time_score = 1.0 / (1.0 + metrics.avg_response_time_ms / 1000)
            success_rate_score = metrics.success_rate
            load_score = 1.0 - (metrics.current_load / max(1, metrics.max_concurrent))
            health_score = metrics.health_score
            
            # Calculate weighted score
            composite_score = (
                strategy.weight_response_time * response_time_score +
                strategy.weight_success_rate * success_rate_score +
                strategy.weight_current_load * load_score +
                strategy.weight_health_score * health_score
            )
            
            agent_scores[agent] = composite_score
        
        # Select agent with highest score
        best_agent = max(agent_scores.keys(), key=lambda agent: agent_scores[agent])
        
        logger.debug(f"Adaptive selection scores: {agent_scores}, selected: {best_agent}")
        
        return best_agent
    
    async def record_request_start(self, agent_name: str) -> str:
        """Record the start of a request to an agent"""
        
        try:
            if agent_name in self.agent_metrics:
                metrics = self.agent_metrics[agent_name]
                metrics.current_load += 1
                metrics.total_requests += 1
                metrics.last_request_time = datetime.utcnow()
                
                # Generate request ID for tracking
                request_id = f"{agent_name}_{int(time.time()*1000)}"
                
                return request_id
            
        except Exception as e:
            logger.error(f"Request start recording failed for {agent_name}: {e}")
        
        return f"{agent_name}_unknown"
    
    async def record_request_end(
        self, 
        agent_name: str, 
        request_id: str, 
        success: bool, 
        response_time_ms: int,
        error_message: Optional[str] = None
    ):
        """Record the completion of a request"""
        
        try:
            if agent_name not in self.agent_metrics:
                return
            
            metrics = self.agent_metrics[agent_name]
            circuit_breaker = self.circuit_breakers[agent_name]
            
            # Update load
            metrics.current_load = max(0, metrics.current_load - 1)
            
            # Update response time (exponential moving average)
            alpha = 0.1
            metrics.avg_response_time_ms = (
                alpha * response_time_ms + 
                (1 - alpha) * metrics.avg_response_time_ms
            )
            
            current_time = datetime.utcnow()
            
            if success:
                # Handle successful request
                metrics.last_success_time = current_time
                circuit_breaker.last_success_time = current_time
                
                # Reset circuit breaker if in half-open state
                if circuit_breaker.state == CircuitState.HALF_OPEN:
                    circuit_breaker.half_open_requests += 1
                    
                    if circuit_breaker.half_open_requests >= circuit_breaker.max_half_open_requests:
                        circuit_breaker.state = CircuitState.CLOSED
                        circuit_breaker.failure_count = 0
                        logger.info(f"Circuit breaker for {agent_name} closed after successful requests")
                
                elif circuit_breaker.state == CircuitState.CLOSED:
                    # Reset failure count on success
                    circuit_breaker.failure_count = max(0, circuit_breaker.failure_count - 1)
                
            else:
                # Handle failed request
                circuit_breaker.failure_count += 1
                circuit_breaker.last_failure_time = current_time
                metrics.error_count_1min += 1
                metrics.error_count_5min += 1
                
                self.performance_metrics["total_agent_failures"] += 1
                
                # Check if circuit breaker should open
                if (circuit_breaker.failure_count >= self.config["circuit_breaker_threshold"] and
                    circuit_breaker.state == CircuitState.CLOSED):
                    
                    circuit_breaker.state = CircuitState.OPEN
                    circuit_breaker.next_attempt_time = (
                        current_time + timedelta(seconds=circuit_breaker.timeout_duration)
                    )
                    
                    self.performance_metrics["circuit_breaker_activations"] += 1
                    
                    logger.warning(f"Circuit breaker OPENED for {agent_name} after {circuit_breaker.failure_count} failures")
                
                # If in half-open state and request failed, go back to open
                elif circuit_breaker.state == CircuitState.HALF_OPEN:
                    circuit_breaker.state = CircuitState.OPEN
                    circuit_breaker.next_attempt_time = (
                        current_time + timedelta(seconds=circuit_breaker.timeout_duration)
                    )
                    logger.warning(f"Circuit breaker back to OPEN for {agent_name} after half-open failure")
            
            # Update success rate (sliding window)
            await self._update_success_rate(agent_name)
            
            # Update health score
            await self._update_health_score(agent_name)
            
        except Exception as e:
            logger.error(f"Request end recording failed for {agent_name}: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive load balancer status"""
        
        try:
            agent_status = {}
            
            for agent_name, metrics in self.agent_metrics.items():
                circuit_breaker = self.circuit_breakers[agent_name]
                
                agent_status[agent_name] = {
                    "current_load": metrics.current_load,
                    "max_concurrent": metrics.max_concurrent,
                    "utilization": metrics.current_load / max(1, metrics.max_concurrent),
                    "avg_response_time_ms": round(metrics.avg_response_time_ms, 1),
                    "success_rate": round(metrics.success_rate, 3),
                    "health_score": round(metrics.health_score, 3),
                    "total_requests": metrics.total_requests,
                    "error_count_1min": metrics.error_count_1min,
                    "error_count_5min": metrics.error_count_5min,
                    "last_request": metrics.last_request_time.isoformat() if metrics.last_request_time else None,
                    "last_success": metrics.last_success_time.isoformat() if metrics.last_success_time else None,
                    "circuit_breaker": {
                        "state": circuit_breaker.state.value,
                        "failure_count": circuit_breaker.failure_count,
                        "last_failure": circuit_breaker.last_failure_time.isoformat() if circuit_breaker.last_failure_time else None,
                        "next_attempt": circuit_breaker.next_attempt_time.isoformat() if circuit_breaker.next_attempt_time else None
                    }
                }
            
            return {
                "agents": agent_status,
                "performance_metrics": self.performance_metrics,
                "load_balancing_strategies": list(self.strategies.keys()),
                "active_strategy": "adaptive",
                "total_healthy_agents": sum(1 for agent in self.agent_metrics.keys() if self._is_agent_available(agent)),
                "routing_history_size": len(self.routing_history)
            }
            
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return {"error": str(e)}
    
    def _get_agent_max_concurrent(self, agent_name: str) -> int:
        """Get maximum concurrent requests for agent"""
        
        max_concurrent_map = {
            "sentiment": 20,
            "llama_explanation": 5,
            "gpt4_strategy": 8,
            "tft_forecasting": 12
        }
        
        return max_concurrent_map.get(agent_name, 10)
    
    def _get_agent_baseline_response_time(self, agent_name: str) -> float:
        """Get baseline response time for agent"""
        
        baseline_times = {
            "sentiment": 800.0,
            "llama_explanation": 2500.0,
            "gpt4_strategy": 3200.0,
            "tft_forecasting": 1800.0
        }
        
        return baseline_times.get(agent_name, 1000.0)
    
    def _update_routing_time_metric(self, routing_time_ms: float):
        """Update average routing time metric"""
        
        alpha = 0.1
        current_avg = self.performance_metrics["average_routing_time_ms"]
        self.performance_metrics["average_routing_time_ms"] = (
            alpha * routing_time_ms + (1 - alpha) * current_avg
        )
    
    async def _update_success_rate(self, agent_name: str):
        """Update success rate for agent"""
        
        try:
            # Calculate success rate based on recent request history
            metrics = self.agent_metrics[agent_name]
            
            # Simple sliding window success rate calculation
            # In production, implement more sophisticated metrics
            recent_window = 100  # Last 100 requests
            
            if metrics.total_requests > 0:
                # Estimate success rate based on error counts
                recent_errors = metrics.error_count_5min
                recent_requests = min(metrics.total_requests, recent_window)
                
                success_rate = max(0, (recent_requests - recent_errors) / recent_requests)
                metrics.success_rate = success_rate
            
        except Exception as e:
            logger.error(f"Success rate update failed for {agent_name}: {e}")
    
    async def _update_health_score(self, agent_name: str):
        """Update health score for agent"""
        
        try:
            metrics = self.agent_metrics[agent_name]
            circuit_breaker = self.circuit_breakers[agent_name]
            
            # Calculate health score based on multiple factors
            health_factors = []
            
            # Success rate factor (0-1)
            health_factors.append(metrics.success_rate)
            
            # Response time factor (0-1, lower is better)
            max_acceptable_time = self.config["max_response_time_threshold"]
            response_time_factor = max(0, 1 - (metrics.avg_response_time_ms / max_acceptable_time))
            health_factors.append(response_time_factor)
            
            # Circuit breaker factor
            if circuit_breaker.state == CircuitState.CLOSED:
                circuit_factor = 1.0
            elif circuit_breaker.state == CircuitState.HALF_OPEN:
                circuit_factor = 0.5
            else:  # OPEN
                circuit_factor = 0.0
            health_factors.append(circuit_factor)
            
            # Load factor (0-1, lower load is better)
            load_factor = 1.0 - (metrics.current_load / max(1, metrics.max_concurrent))
            health_factors.append(load_factor)
            
            # Calculate weighted health score
            weights = [0.4, 0.3, 0.2, 0.1]  # success_rate, response_time, circuit_breaker, load
            metrics.health_score = sum(factor * weight for factor, weight in zip(health_factors, weights))
            
        except Exception as e:
            logger.error(f"Health score update failed for {agent_name}: {e}")
    
    async def _health_monitor(self):
        """Background task to monitor agent health"""
        
        while True:
            try:
                await asyncio.sleep(self.config["health_check_interval"])
                
                current_time = datetime.utcnow()
                
                for agent_name in self.agent_metrics.keys():
                    metrics = self.agent_metrics[agent_name]
                    
                    # Check if agent has been idle too long
                    if (current_time - metrics.last_request_time).total_seconds() > 300:  # 5 minutes
                        # Gradually improve health score for idle agents
                        metrics.health_score = min(1.0, metrics.health_score + 0.01)
                    
                    # Reset short-term error counters
                    time_since_last_error = (current_time - (metrics.last_request_time or current_time)).total_seconds()
                    
                    if time_since_last_error > 60:  # 1 minute
                        metrics.error_count_1min = 0
                    
                    if time_since_last_error > 300:  # 5 minutes
                        metrics.error_count_5min = 0
                
                logger.debug("Health monitoring cycle completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _metrics_collector(self):
        """Background task to collect and aggregate metrics"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute
                
                # Update success rates
                for agent_name in self.agent_metrics.keys():
                    await self._update_success_rate(agent_name)
                    await self._update_health_score(agent_name)
                
                # Log summary metrics
                total_load = sum(metrics.current_load for metrics in self.agent_metrics.values())
                avg_health = statistics.mean(metrics.health_score for metrics in self.agent_metrics.values())
                
                logger.info(f"Load balancer metrics - Total load: {total_load}, "
                          f"Avg health: {avg_health:.2f}, "
                          f"Requests routed: {self.performance_metrics['total_requests_routed']}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(60)
    
    async def _circuit_breaker_manager(self):
        """Background task to manage circuit breaker states"""
        
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                current_time = datetime.utcnow()
                
                for agent_name, circuit_breaker in self.circuit_breakers.items():
                    if circuit_breaker.state == CircuitState.OPEN:
                        # Check if it's time to try half-open
                        if (circuit_breaker.next_attempt_time and 
                            current_time >= circuit_breaker.next_attempt_time):
                            
                            circuit_breaker.state = CircuitState.HALF_OPEN
                            circuit_breaker.half_open_requests = 0
                            
                            logger.info(f"Circuit breaker for {agent_name} moved to HALF_OPEN for testing")
                    
                    elif circuit_breaker.state == CircuitState.HALF_OPEN:
                        # Check for timeout in half-open state
                        if (circuit_breaker.next_attempt_time and 
                            current_time >= circuit_breaker.next_attempt_time + timedelta(seconds=30)):
                            
                            # If no requests processed in half-open, go back to closed
                            if circuit_breaker.half_open_requests == 0:
                                circuit_breaker.state = CircuitState.CLOSED
                                circuit_breaker.failure_count = 0
                                logger.info(f"Circuit breaker for {agent_name} closed due to timeout in half-open")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Circuit breaker manager error: {e}")
                await asyncio.sleep(30)

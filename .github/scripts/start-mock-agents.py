#!/usr/bin/env python3
"""
=============================================================================
MOCK AGENTS STARTUP SCRIPT
Starts mock versions of all MIP Platform agents for testing
=============================================================================
"""

import os
import sys
import json
import time
import random
import asyncio
import argparse
import logging
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import aiohttp
from aiohttp import web
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    name: str
    port: int
    response_time_range: tuple
    accuracy_range: tuple
    endpoints: List[str]
    special_features: Dict[str, Any]

class MockAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.app = web.Application()
        self.setup_routes()
        self.request_count = 0
        self.start_time = time.time()
        self.health_status = "healthy"
        
    def setup_routes(self):
        """Setup HTTP routes for the mock agent."""
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/ready', self.readiness_check)
        self.app.router.add_get('/metrics', self.metrics)
        self.app.router.add_post('/process', self.process_request)
        
        # Agent-specific endpoints
        for endpoint in self.config.endpoints:
            self.app.router.add_post(f'/{endpoint}', self.process_request)
    
    async def health_check(self, request):
        """Health check endpoint."""
        return web.json_response({
            "status": self.health_status,
            "agent": self.config.name,
            "uptime_seconds": time.time() - self.start_time,
            "version": "mock-1.0.0"
        })
    
    async def readiness_check(self, request):
        """Readiness check endpoint."""
        # Simulate some startup time
        if time.time() - self.start_time < 5:
            return web.json_response(
                {"status": "not_ready", "agent": self.config.name},
                status=503
            )
        
        return web.json_response({
            "status": "ready",
            "agent": self.config.name,
            "requests_processed": self.request_count
        })
    
    async def metrics(self, request):
        """Prometheus-style metrics endpoint."""
        uptime = time.time() - self.start_time
        metrics = f"""# HELP agent_requests_total Total number of requests processed
# TYPE agent_requests_total counter
agent_requests_total{{agent="{self.config.name}"}} {self.request_count}

# HELP agent_uptime_seconds Agent uptime in seconds
# TYPE agent_uptime_seconds gauge
agent_uptime_seconds{{agent="{self.config.name}"}} {uptime:.2f}

# HELP agent_response_time_ms Average response time in milliseconds
# TYPE agent_response_time_ms gauge
agent_response_time_ms{{agent="{self.config.name}"}} {random.uniform(*self.config.response_time_range)}

# HELP agent_accuracy_score Current accuracy score
# TYPE agent_accuracy_score gauge
agent_accuracy_score{{agent="{self.config.name}"}} {random.uniform(*self.config.accuracy_range)}
"""
        return web.Response(text=metrics, content_type='text/plain')
    
    async def process_request(self, request):
        """Process agent-specific requests."""
        self.request_count += 1
        
        # Simulate processing time
        processing_time = random.uniform(*self.config.response_time_range) / 1000
        await asyncio.sleep(processing_time)
        
        # Get request data
        try:
            data = await request.json()
        except:
            data = {}
        
        # Generate agent-specific response
        response = self.generate_response(data, request.path)
        
        return web.json_response(response)
    
    def generate_response(self, data: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Generate agent-specific mock responses."""
        base_response = {
            "agent": self.config.name,
            "timestamp": time.time(),
            "processing_time_ms": random.uniform(*self.config.response_time_range),
            "request_id": f"req_{self.request_count}_{int(time.time())}"
        }
        
        if self.config.name == "finbert":
            return {
                **base_response,
                "sentiment": random.choice(["positive", "negative", "neutral"]),
                "confidence": random.uniform(0.7, 0.95),
                "scores": {
                    "positive": random.uniform(0.1, 0.9),
                    "negative": random.uniform(0.1, 0.9),
                    "neutral": random.uniform(0.1, 0.9)
                },
                "text_length": len(data.get("text", "")),
                "model_version": "finbert-financial-sentiment"
            }
        
        elif self.config.name == "llama":
            return {
                **base_response,
                "reasoning": "Based on the analysis of the provided information, considering market dynamics and financial principles...",
                "confidence": random.uniform(0.75, 0.90),
                "key_factors": [
                    "Market volatility",
                    "Economic indicators", 
                    "Sector performance",
                    "Company fundamentals"
                ],
                "risk_assessment": random.choice(["low", "medium", "high"]),
                "recommendation": random.choice(["buy", "hold", "sell"]),
                "model_version": "llama-3.1-8b-financial"
            }
        
        elif self.config.name == "gpt4":
            return {
                **base_response,
                "analysis": "Comprehensive market analysis indicates several key trends affecting the financial landscape...",
                "confidence": random.uniform(0.85, 0.95),
                "insights": [
                    "Strong earnings growth in tech sector",
                    "Federal Reserve policy impacts",
                    "Global economic uncertainties",
                    "ESG investing trends"
                ],
                "market_outlook": random.choice(["bullish", "bearish", "neutral"]),
                "time_horizon": random.choice(["short-term", "medium-term", "long-term"]),
                "model_version": "gpt-4-turbo"
            }
        
        elif self.config.name == "tft":
            forecast_periods = random.randint(5, 30)
            base_value = random.uniform(100, 500)
            
            forecast = []
            for i in range(forecast_periods):
                # Generate realistic price movement
                change = random.gauss(0.001, 0.02)  # 0.1% drift, 2% volatility
                base_value *= (1 + change)
                forecast.append(round(base_value, 2))
            
            return {
                **base_response,
                "forecast": forecast,
                "confidence_intervals": {
                    "upper_95": [f * 1.05 for f in forecast],
                    "lower_95": [f * 0.95 for f in forecast]
                },
                "forecast_horizon_days": forecast_periods,
                "model_accuracy": random.uniform(0.70, 0.85),
                "features_used": ["price", "volume", "volatility", "sentiment", "macro_indicators"],
                "model_version": "tft-pytorch-v2.1"
            }
        
        elif self.config.name == "orchestrator":
            return {
                **base_response,
                "coordination_status": "success",
                "agents_contacted": random.randint(2, 4),
                "consensus_score": random.uniform(0.6, 0.9),
                "final_recommendation": {
                    "action": random.choice(["buy", "sell", "hold"]),
                    "confidence": random.uniform(0.7, 0.9),
                    "risk_level": random.choice(["low", "medium", "high"])
                },
                "agent_responses": {
                    "finbert": {"sentiment": "positive", "weight": 0.3},
                    "llama": {"reasoning_score": 0.8, "weight": 0.3},
                    "gpt4": {"analysis_quality": 0.9, "weight": 0.25},
                    "tft": {"forecast_accuracy": 0.75, "weight": 0.15}
                }
            }
        
        else:
            return {
                **base_response,
                "result": "processed",
                "status": "success"
            }

class MockAgentManager:
    def __init__(self):
        self.agents: Dict[str, MockAgent] = {}
        self.agent_processes: Dict[str, subprocess.Popen] = {}
        self.running = False
        
        # Define agent configurations
        self.agent_configs = {
            "finbert": AgentConfig(
                name="finbert",
                port=8001,
                response_time_range=(100, 800),
                accuracy_range=(0.85, 0.95),
                endpoints=["analyze", "sentiment", "batch_process"],
                special_features={"supports_batch": True, "max_text_length": 512}
            ),
            "llama": AgentConfig(
                name="llama",
                port=8002,
                response_time_range=(1000, 3000),
                accuracy_range=(0.80, 0.90),
                endpoints=["reason", "generate", "analyze"],
                special_features={"context_window": 8192, "supports_streaming": True}
            ),
            "gpt4": AgentConfig(
                name="gpt4",
                port=8003,
                response_time_range=(500, 1500),
                accuracy_range=(0.90, 0.98),
                endpoints=["analyze", "chat", "complete"],
                special_features={"api_based": True, "rate_limited": True}
            ),
            "tft": AgentConfig(
                name="tft",
                port=8004,
                response_time_range=(2000, 5000),
                accuracy_range=(0.75, 0.85),
                endpoints=["forecast", "predict", "train"],
                special_features={"time_series": True, "supports_multivariate": True}
            ),
            "orchestrator": AgentConfig(
                name="orchestrator",
                port=8005,
                response_time_range=(800, 2000),
                accuracy_range=(0.80, 0.92),
                endpoints=["coordinate", "consensus", "execute"],
                special_features={"multi_agent": True, "workflow_manager": True}
            )
        }
    
    async def start_agent(self, agent_name: str) -> bool:
        """Start a single mock agent."""
        if agent_name not in self.agent_configs:
            logger.error(f"Unknown agent: {agent_name}")
            return False
        
        if agent_name in self.agents:
            logger.warning(f"Agent {agent_name} already running")
            return True
        
        config = self.agent_configs[agent_name]
        agent = MockAgent(config)
        
        try:
            # Start the web server
            runner = web.AppRunner(agent.app)
            await runner.setup()
            
            site = web.TCPSite(runner, '0.0.0.0', config.port)
            await site.start()
            
            self.agents[agent_name] = agent
            logger.info(f"✓ Started {agent_name} agent on port {config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {agent_name} agent: {e}")
            return False
    
    async def start_all_agents(self, agent_list: Optional[List[str]] = None) -> Dict[str, bool]:
        """Start all or specified mock agents."""
        if agent_list is None:
            agent_list = list(self.agent_configs.keys())
        
        results = {}
        
        for agent_name in agent_list:
            success = await self.start_agent(agent_name)
            results[agent_name] = success
            
            if success:
                # Small delay between starting agents
                await asyncio.sleep(1)
        
        return results
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Check health of all running agents."""
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for agent_name, agent in self.agents.items():
                config = self.agent_configs[agent_name]
                url = f"http://localhost:{config.port}/health"
                
                try:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            data = await response.json()
                            results[agent_name] = {
                                "status": "healthy",
                                "uptime": data.get("uptime_seconds", 0),
                                "port": config.port
                            }
                        else:
                            results[agent_name] = {
                                "status": "unhealthy",
                                "status_code": response.status,
                                "port": config.port
                            }
                            
                except asyncio.TimeoutError:
                    results[agent_name] = {
                        "status": "timeout",
                        "port": config.port
                    }
                except Exception as e:
                    results[agent_name] = {
                        "status": "error",
                        "error": str(e),
                        "port": config.port
                    }
        
        return results
    
    async def stop_all_agents(self):
        """Stop all running agents."""
        logger.info("Stopping all mock agents...")
        
        for agent_name in list(self.agents.keys()):
            try:
                # The web server will be stopped when the process exits
                del self.agents[agent_name]
                logger.info(f"✓ Stopped {agent_name} agent")
            except Exception as e:
                logger.error(f"Error stopping {agent_name}: {e}")
        
        self.running = False
    
    async def run_interactive_mode(self):
        """Run in interactive mode with status updates."""
        self.running = True
        
        try:
            while self.running:
                # Periodic health check
                await asyncio.sleep(30)
                
                if self.agents:
                    health_status = await self.health_check_all()
                    healthy_count = sum(1 for status in health_status.values() 
                                      if status["status"] == "healthy")
                    total_count = len(health_status)
                    
                    logger.info(f"Agent Status: {healthy_count}/{total_count} healthy")
                    
                    # Log any unhealthy agents
                    for agent_name, status in health_status.items():
                        if status["status"] != "healthy":
                            logger.warning(f"  {agent_name}: {status['status']}")
                
        except asyncio.CancelledError:
            logger.info("Interactive mode cancelled")
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
        finally:
            await self.stop_all_agents()

def setup_signal_handlers(manager: MockAgentManager):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        manager.running = False
        
        # Create a new event loop if none exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if not loop.is_closed():
            loop.create_task(manager.stop_all_agents())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    parser = argparse.ArgumentParser(description='Start mock agents for MIP Platform testing')
    parser.add_argument('--agents', '-a', nargs='+', 
                       choices=['finbert', 'llama', 'gpt4', 'tft', 'orchestrator'],
                       help='Specific agents to start (default: all)')
    parser.add_argument('--check-health', action='store_true',
                       help='Check health of running agents and exit')
    parser.add_argument('--daemon', '-d', action='store_true',
                       help='Run in daemon mode (no interactive output)')
    parser.add_argument('--timeout', '-t', type=int, default=300,
                       help='Run for specified seconds then exit (default: run indefinitely)')
    parser.add_argument('--output', '-o', help='Output status to JSON file')
    
    args = parser.parse_args()
    
    manager = MockAgentManager()
    
    try:
        if args.check_health:
            # Just check health and exit
            health_status = await manager.health_check_all()
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(health_status, f, indent=2)
            else:
                print(json.dumps(health_status, indent=2))
            
            # Exit with appropriate code
            healthy_count = sum(1 for status in health_status.values() 
                              if status["status"] == "healthy")
            sys.exit(0 if healthy_count == len(health_status) else 1)
        
        # Setup signal handlers
        setup_signal_handlers(manager)
        
        # Start agents
        logger.info("Starting mock agents...")
        start_results = await manager.start_all_agents(args.agents)
        
        # Report startup results
        successful_starts = sum(1 for success in start_results.values() if success)
        total_agents = len(start_results)
        
        logger.info(f"Startup complete: {successful_starts}/{total_agents} agents started")
        
        if successful_starts == 0:
            logger.error("No agents started successfully")
            sys.exit(1)
        
        for agent_name, success in start_results.items():
            status = "✓" if success else "✗"
            port = manager.agent_configs[agent_name].port
            logger.info(f"  {status} {agent_name} (port {port})")
        
        # Wait for agents to be ready
        logger.info("Waiting for agents to be ready...")
        await asyncio.sleep(5)
        
        # Initial health check
        health_status = await manager.health_check_all()
        
        if args.output:
            status_data = {
                "startup_results": start_results,
                "health_status": health_status,
                "timestamp": time.time()
            }
            with open(args.output, 'w') as f:
                json.dump(status_data, f, indent=2)
        
        if args.daemon:
            # Daemon mode - run for specified time or indefinitely
            if args.timeout > 0:
                logger.info(f"Running in daemon mode for {args.timeout} seconds...")
                await asyncio.sleep(args.timeout)
            else:
                logger.info("Running in daemon mode indefinitely...")
                while manager.running:
                    await asyncio.sleep(10)
        else:
            # Interactive mode
            logger.info("Agents started successfully!")
            logger.info("Press Ctrl+C to stop all agents")
            logger.info("Agent endpoints:")
            
            for agent_name in start_results:
                if start_results[agent_name]:
                    port = manager.agent_configs[agent_name].port
                    logger.info(f"  {agent_name}: http://localhost:{port}")
            
            await manager.run_interactive_mode()
    
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        if manager.agents:
            await manager.stop_all_agents()
        logger.info("Mock agents stopped")

if __name__ == "__main__":
    asyncio.run(main())

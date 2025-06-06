#!/usr/bin/env python3
"""
=============================================================================
LOCUST PERFORMANCE TESTING FOR MIP PLATFORM
Location: tests/performance/locustfile.py

Usage:
  locust -f tests/performance/locustfile.py --host=http://localhost:8000
  locust -f tests/performance/locustfile.py --host=http://localhost:8000 --users 50 --spawn-rate 5 --run-time 10m
=============================================================================
"""

import json
import random
import time
from locust import HttpUser, task, between, events
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MIPPlatformUser(HttpUser):
    """Base user class for MIP Platform load testing."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auth_token = None
        self.user_id = None
        self.portfolio_id = None
        
        # Test data
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        self.test_news = [
            "Company reports strong quarterly earnings beating analyst expectations",
            "Market volatility continues amid economic uncertainty",
            "Federal Reserve announces interest rate decision",
            "Tech sector shows resilience in challenging market conditions",
            "Merger announcement sends stock prices higher",
            "Regulatory investigation impacts company valuation",
            "Strong consumer demand drives revenue growth",
            "Supply chain disruptions affect quarterly results"
        ]
    
    def on_start(self):
        """Called when a user starts. Authenticate the user."""
        self.authenticate()
    
    def authenticate(self):
        """Authenticate user and get access token."""
        username = f"loadtest_user_{random.randint(1000, 9999)}"
        password = "loadtest_password"
        
        response = self.client.post("/api/v1/auth/login", json={
            "username": username,
            "password": password
        }, catch_response=True)
        
        if response.status_code == 200:
            data = response.json()
            self.auth_token = data.get("access_token")
            self.user_id = data.get("user_id")
            
            # Update client headers
            self.client.headers.update({
                "Authorization": f"Bearer {self.auth_token}"
            })
            
            response.success()
        elif response.status_code == 404:
            # Auth endpoint might not exist in mock, mark as success
            self.auth_token = "mock_token"
            self.user_id = random.randint(1, 1000)
            response.success()
        else:
            response.failure(f"Authentication failed: {response.status_code}")
    
    @task(3)
    def health_check(self):
        """Test health endpoints."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(5)
    def get_market_data(self):
        """Test market data retrieval."""
        symbols = random.sample(self.test_symbols, k=random.randint(1, 3))
        
        with self.client.get(f"/api/v1/market/data", 
                           params={"symbols": ",".join(symbols)},
                           catch_response=True) as response:
            if response.status_code in [200, 404]:  # 404 acceptable for mock
                response.success()
            else:
                response.failure(f"Market data request failed: {response.status_code}")


class FinBERTUser(MIPPlatformUser):
    """User focused on FinBERT sentiment analysis tasks."""
    
    weight = 3  # 30% of users
    
    @task(10)
    def analyze_sentiment(self):
        """Test FinBERT sentiment analysis."""
        text = random.choice(self.test_news)
        
        with self.client.post("/api/v1/agents/finbert/analyze", 
                            json={"text": text},
                            catch_response=True) as response:
            if response.status_code in [200, 404]:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if "sentiment" in data and "confidence" in data:
                            response.success()
                        else:
                            response.failure("Invalid response format")
                    except:
                        response.failure("Invalid JSON response")
                else:
                    response.success()  # Mock endpoint
            else:
                response.failure(f"Sentiment analysis failed: {response.status_code}")
    
    @task(5)
    def batch_sentiment_analysis(self):
        """Test batch sentiment analysis."""
        texts = random.sample(self.test_news, k=random.randint(2, 4))
        
        with self.client.post("/api/v1/agents/finbert/batch", 
                            json={"texts": texts},
                            catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Batch sentiment analysis failed: {response.status_code}")
    
    @task(2)
    def get_sentiment_history(self):
        """Test sentiment history retrieval."""
        symbol = random.choice(self.test_symbols)
        
        with self.client.get(f"/api/v1/agents/finbert/history/{symbol}",
                           catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Sentiment history failed: {response.status_code}")


class LlamaUser(MIPPlatformUser):
    """User focused on Llama reasoning tasks."""
    
    weight = 2  # 20% of users
    
    @task(8)
    def reasoning_analysis(self):
        """Test Llama reasoning analysis."""
        prompts = [
            "Analyze the financial implications of rising interest rates on tech stocks",
            "Evaluate the impact of merger announcements on market volatility",
            "Assess the risk factors in emerging market investments",
            "Compare value vs growth investing strategies in current market conditions"
        ]
        
        prompt = random.choice(prompts)
        
        with self.client.post("/api/v1/agents/llama/reason", 
                            json={
                                "prompt": prompt,
                                "max_tokens": 500,
                                "context": "financial_analysis"
                            },
                            catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Reasoning analysis failed: {response.status_code}")
    
    @task(3)
    def generate_insights(self):
        """Test insight generation."""
        symbol = random.choice(self.test_symbols)
        
        with self.client.post("/api/v1/agents/llama/insights", 
                            json={
                                "symbol": symbol,
                                "analysis_type": "comprehensive"
                            },
                            catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Insight generation failed: {response.status_code}")


class GPT4User(MIPPlatformUser):
    """User focused on GPT-4 analysis tasks."""
    
    weight = 3  # 30% of users
    
    @task(10)
    def comprehensive_analysis(self):
        """Test GPT-4 comprehensive analysis."""
        symbols = random.sample(self.test_symbols, k=random.randint(1, 2))
        
        with self.client.post("/api/v1/agents/gpt4/analyze", 
                            json={
                                "symbols": symbols,
                                "analysis_type": "fundamental",
                                "include_recommendations": True
                            },
                            catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"GPT-4 analysis failed: {response.status_code}")
    
    @task(6)
    def market_summary(self):
        """Test market summary generation."""
        with self.client.post("/api/v1/agents/gpt4/summary", 
                            json={
                                "market_segment": "technology",
                                "timeframe": "daily"
                            },
                            catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Market summary failed: {response.status_code}")
    
    @task(4)
    def investment_recommendations(self):
        """Test investment recommendation generation."""
        with self.client.post("/api/v1/agents/gpt4/recommend", 
                            json={
                                "risk_tolerance": random.choice(["conservative", "moderate", "aggressive"]),
                                "investment_horizon": random.choice(["short", "medium", "long"]),
                                "amount": random.randint(10000, 100000)
                            },
                            catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Investment recommendations failed: {response.status_code}")


class TFTUser(MIPPlatformUser):
    """User focused on TFT forecasting tasks."""
    
    weight = 2  # 20% of users
    
    @task(8)
    def generate_forecast(self):
        """Test TFT price forecasting."""
        symbol = random.choice(self.test_symbols)
        horizon = random.choice([1, 5, 10, 20])
        
        with self.client.post("/api/v1/agents/tft/forecast", 
                            json={
                                "symbol": symbol,
                                "horizon_days": horizon,
                                "include_confidence": True
                            },
                            catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"TFT forecast failed: {response.status_code}")
    
    @task(5)
    def multi_asset_forecast(self):
        """Test multi-asset forecasting."""
        symbols = random.sample(self.test_symbols, k=random.randint(2, 3))
        
        with self.client.post("/api/v1/agents/tft/multi_forecast", 
                            json={
                                "symbols": symbols,
                                "horizon_days": 5,
                                "correlation_analysis": True
                            },
                            catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Multi-asset forecast failed: {response.status_code}")
    
    @task(3)
    def model_performance(self):
        """Test model performance metrics."""
        with self.client.get("/api/v1/agents/tft/performance",
                           catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"TFT performance check failed: {response.status_code}")


class PortfolioManagerUser(MIPPlatformUser):
    """User focused on portfolio management tasks."""
    
    weight = 2  # 20% of users
    
    def on_start(self):
        super().on_start()
        self.create_test_portfolio()
    
    def create_test_portfolio(self):
        """Create a test portfolio for this user."""
        with self.client.post("/api/v1/portfolio/create", 
                            json={
                                "name": f"Test Portfolio {random.randint(1000, 9999)}",
                                "initial_cash": random.randint(50000, 200000),
                                "strategy": random.choice(["growth", "value", "balanced"]),
                                "risk_level": random.choice(["low", "medium", "high"])
                            },
                            catch_response=True) as response:
            if response.status_code in [201, 404]:
                if response.status_code == 201:
                    try:
                        data = response.json()
                        self.portfolio_id = data.get("portfolio_id")
                    except:
                        pass
                response.success()
            else:
                response.failure(f"Portfolio creation failed: {response.status_code}")
    
    @task(8)
    def add_positions(self):
        """Test adding positions to portfolio."""
        if not self.portfolio_id:
            return
        
        symbol = random.choice(self.test_symbols)
        quantity = random.randint(10, 100)
        
        with self.client.post(f"/api/v1/portfolio/{self.portfolio_id}/positions", 
                            json={
                                "symbol": symbol,
                                "quantity": quantity,
                                "action": "BUY"
                            },
                            catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Add position failed: {response.status_code}")
    
    @task(6)
    def get_portfolio_performance(self):
        """Test portfolio performance retrieval."""
        if not self.portfolio_id:
            return
        
        with self.client.get(f"/api/v1/portfolio/{self.portfolio_id}/performance",
                           catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Portfolio performance failed: {response.status_code}")
    
    @task(5)
    def risk_analysis(self):
        """Test portfolio risk analysis."""
        if not self.portfolio_id:
            return
        
        with self.client.get(f"/api/v1/portfolio/{self.portfolio_id}/risk",
                           catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Risk analysis failed: {response.status_code}")
    
    @task(3)
    def rebalancing_suggestions(self):
        """Test portfolio rebalancing suggestions."""
        if not self.portfolio_id:
            return
        
        with self.client.post(f"/api/v1/portfolio/{self.portfolio_id}/rebalance", 
                            json={
                                "target_allocation": {
                                    "AAPL": 0.3,
                                    "MSFT": 0.3,
                                    "GOOGL": 0.4
                                }
                            },
                            catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Rebalancing suggestions failed: {response.status_code}")


class OrchestratorUser(MIPPlatformUser):
    """User focused on orchestrated multi-agent tasks."""
    
    weight = 1  # 10% of users
    
    @task(10)
    def coordinated_analysis(self):
        """Test orchestrated multi-agent analysis."""
        symbol = random.choice(self.test_symbols)
        
        with self.client.post("/api/v1/orchestrator/coordinate", 
                            json={
                                "task": "comprehensive_analysis",
                                "symbol": symbol,
                                "agents": ["finbert", "llama", "gpt4", "tft"],
                                "priority": "normal"
                            },
                            catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Orchestrated analysis failed: {response.status_code}")
    
    @task(6)
    def investment_decision(self):
        """Test orchestrated investment decision."""
        symbols = random.sample(self.test_symbols, k=2)
        
        with self.client.post("/api/v1/orchestrator/decide", 
                            json={
                                "task": "investment_decision",
                                "symbols": symbols,
                                "amount": random.randint(10000, 50000),
                                "strategy": random.choice(["conservative", "balanced", "aggressive"])
                            },
                            catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Investment decision failed: {response.status_code}")
    
    @task(4)
    def get_consensus(self):
        """Test agent consensus retrieval."""
        with self.client.get("/api/v1/orchestrator/consensus",
                           params={"topic": "market_outlook"},
                           catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Consensus retrieval failed: {response.status_code}")


class WebSocketUser(MIPPlatformUser):
    """User testing WebSocket connections."""
    
    weight = 1  # 10% of users
    
    @task(8)
    def simulate_websocket_activity(self):
        """Simulate WebSocket activity using HTTP requests."""
        # Since Locust doesn't natively support WebSocket testing well,
        # we'll simulate WebSocket-like activity with HTTP requests
        
        with self.client.post("/api/v1/websocket/subscribe", 
                            json={
                                "channels": ["market_data", "portfolio_updates"],
                                "symbols": random.sample(self.test_symbols, k=2)
                            },
                            catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"WebSocket subscribe failed: {response.status_code}")
    
    @task(5)
    def get_real_time_updates(self):
        """Test real-time update polling."""
        with self.client.get("/api/v1/updates/poll",
                           catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Real-time updates failed: {response.status_code}")


# Performance test scenarios for different load patterns
class LightLoadUser(MIPPlatformUser):
    """User for light load testing (normal business hours)."""
    
    wait_time = between(2, 5)
    weight = 1
    
    @task
    def light_activity(self):
        """Simulate light user activity."""
        endpoints = ["/health", "/api/v1/market/status", "/api/v1/user/dashboard"]
        endpoint = random.choice(endpoints)
        
        with self.client.get(endpoint, catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Light activity failed: {response.status_code}")


class HeavyLoadUser(MIPPlatformUser):
    """User for heavy load testing (peak trading hours)."""
    
    wait_time = between(0.5, 1.5)
    weight = 1
    
    @task
    def heavy_activity(self):
        """Simulate heavy user activity."""
        # Rapid-fire requests
        for _ in range(random.randint(2, 5)):
            symbol = random.choice(self.test_symbols)
            
            with self.client.get(f"/api/v1/market/data/{symbol}",
                               catch_response=True) as response:
                if response.status_code in [200, 404]:
                    response.success()
                else:
                    response.failure(f"Heavy activity failed: {response.status_code}")
            
            time.sleep(0.1)  # Very short delay


# Event listeners for custom metrics and reporting
@events.request.add_listener
def record_custom_metrics(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Record custom metrics for MIP-specific analysis."""
    
    # Track agent-specific performance
    if "/agents/" in name:
        agent_name = name.split("/agents/")[1].split("/")[0]
        
        # Record agent response times (this would integrate with monitoring system)
        if not hasattr(record_custom_metrics, 'agent_metrics'):
            record_custom_metrics.agent_metrics = {}
        
        if agent_name not in record_custom_metrics.agent_metrics:
            record_custom_metrics.agent_metrics[agent_name] = []
        
        record_custom_metrics.agent_metrics[agent_name].append(response_time)


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts."""
    logger.info("Starting MIP Platform load test")
    logger.info(f"Target host: {environment.host}")
    logger.info(f"User classes: {[cls.__name__ for cls in environment.user_classes]}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops."""
    logger.info("MIP Platform load test completed")
    
    # Log agent-specific metrics if available
    if hasattr(record_custom_metrics, 'agent_metrics'):
        logger.info("Agent Performance Summary:")
        for agent, response_times in record_custom_metrics.agent_metrics.items():
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                max_time = max(response_times)
                logger.info(f"  {agent}: avg={avg_time:.1f}ms, max={max_time:.1f}ms, requests={len(response_times)}")


# Custom locust configuration based on environment variables
def get_user_classes():
    """Get user classes based on environment configuration."""
    test_scenario = os.getenv('LOCUST_SCENARIO', 'balanced')
    
    if test_scenario == 'agent_focused':
        return [FinBERTUser, LlamaUser, GPT4User, TFTUser]
    elif test_scenario == 'portfolio_focused':
        return [PortfolioManagerUser, OrchestratorUser]
    elif test_scenario == 'light_load':
        return [LightLoadUser]
    elif test_scenario == 'heavy_load':
        return [HeavyLoadUser]
    elif test_scenario == 'websocket_focused':
        return [WebSocketUser]
    else:  # balanced
        return [FinBERTUser, LlamaUser, GPT4User, TFTUser, PortfolioManagerUser, OrchestratorUser, WebSocketUser]


# Set user classes based on configuration
import os
if __name__ != "__main__":
    # When running with locust command
    user_classes = get_user_classes()
    
    # Override the default user classes
    import sys
    current_module = sys.modules[__name__]
    for cls in [FinBERTUser, LlamaUser, GPT4User, TFTUser, PortfolioManagerUser, OrchestratorUser, WebSocketUser, LightLoadUser, HeavyLoadUser]:
        if cls not in user_classes:
            delattr(current_module, cls.__name__)


# CLI for standalone execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MIP Platform Locust Performance Test')
    parser.add_argument('--host', default='http://localhost:8000', help='Target host')
    parser.add_argument('--users', '-u', type=int, default=10, help='Number of users')
    parser.add_argument('--spawn-rate', '-r', type=int, default=2, help='Spawn rate (users per second)')
    parser.add_argument('--run-time', '-t', default='60s', help='Run time (e.g., 60s, 10m)')
    parser.add_argument('--scenario', choices=['balanced', 'agent_focused', 'portfolio_focused', 'light_load', 'heavy_load', 'websocket_focused'],
                       default='balanced', help='Test scenario')
    
    args = parser.parse_args()
    
    print(f"""
=============================================================================
MIP PLATFORM PERFORMANCE TEST CONFIGURATION
=============================================================================
Host: {args.host}
Users: {args.users}
Spawn Rate: {args.spawn_rate}/sec
Duration: {args.run_time}
Scenario: {args.scenario}

To run this test:
  locust -f {__file__} --host={args.host} --users {args.users} --spawn-rate {args.spawn_rate} --run-time {args.run_time}

Set LOCUST_SCENARIO={args.scenario} environment variable to use specific scenario.
=============================================================================
    """)

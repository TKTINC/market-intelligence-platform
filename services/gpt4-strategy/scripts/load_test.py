"""
Load Testing for GPT-4 Strategy Service using Locust
"""

from locust import HttpUser, task, between
import json
import random
import time

class StrategyUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup user session"""
        self.headers = {
            "Authorization": "Bearer test_token",
            "Content-Type": "application/json"
        }
        
        self.user_id = f"load_test_user_{random.randint(1000, 9999)}"
        
        # Sample market contexts for variety
        self.market_contexts = [
            {
                "symbol": "SPY",
                "current_price": 450.0,
                "vix": 18.5,
                "trend": "bullish"
            },
            {
                "symbol": "QQQ",
                "current_price": 380.0,
                "vix": 22.3,
                "trend": "neutral"
            },
            {
                "symbol": "IWM",
                "current_price": 200.0,
                "vix": 25.1,
                "trend": "bearish"
            }
        ]
        
        # Sample user intents
        self.user_intents = [
            "Generate a conservative income strategy",
            "I want a bullish strategy with limited risk",
            "Create a volatility play for earnings",
            "Need a hedging strategy for my portfolio",
            "Generate a neutral strategy for sideways market"
        ]
    
    @task(5)
    def generate_strategy(self):
        """Test strategy generation endpoint"""
        
        request_data = {
            "user_id": self.user_id,
            "market_context": random.choice(self.market_contexts),
            "user_intent": random.choice(self.user_intents),
            "portfolio_context": {
                "cash_available": random.randint(5000, 50000),
                "existing_positions": []
            },
            "risk_preferences": {
                "risk_tolerance": random.choice(["low", "medium", "high"]),
                "max_loss": random.randint(500, 5000)
            },
            "max_cost_usd": 0.50
        }
        
        with self.client.post(
            "/strategy/generate",
            json=request_data,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                # Rate limiting is expected under load
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(2)
    def batch_generate(self):
        """Test batch strategy generation"""
        
        batch_size = random.randint(2, 3)
        requests = []
        
        for _ in range(batch_size):
            requests.append({
                "user_id": self.user_id,
                "market_context": random.choice(self.market_contexts),
                "user_intent": random.choice(self.user_intents),
                "max_cost_usd": 0.30
            })
        
        batch_request = {
            "requests": requests,
            "batch_priority": "normal"
        }
        
        with self.client.post(
            "/strategy/batch",
            json=batch_request,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 202, 429]:
                response.success()
            else:
                response.failure(f"Batch request failed: {response.status_code}")
    
    @task(1)
    def check_health(self):
        """Test health endpoint"""
        
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure("Health check failed")
    
    @task(1)
    def get_usage(self):
        """Test usage statistics endpoint"""
        
        with self.client.get(
            f"/user/{self.user_id}/usage",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:  # 404 is OK for new users
                response.success()
            else:
                response.failure(f"Usage check failed: {response.status_code}")
    
    @task(1)
    def get_metrics(self):
        """Test metrics endpoint"""
        
        with self.client.get(
            "/metrics",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics check failed: {response.status_code}")

class AdminUser(HttpUser):
    """Simulate admin users checking system status"""
    wait_time = between(5, 10)
    weight = 1  # Lower weight than regular users
    
    def on_start(self):
        self.headers = {
            "Authorization": "Bearer admin_token",
            "Content-Type": "application/json"
        }
    
    @task
    def admin_metrics(self):
        """Admin checking detailed metrics"""
        
        with self.client.get(
            "/metrics",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure("Admin metrics failed")

# Performance benchmarks
class BenchmarkUser(HttpUser):
    """User for performance benchmarking"""
    wait_time = between(0.1, 0.5)  # Aggressive timing
    
    def on_start(self):
        self.headers = {
            "Authorization": "Bearer benchmark_token",
            "Content-Type": "application/json"
        }
    
    @task
    def benchmark_strategy_generation(self):
        """Benchmark strategy generation performance"""
        
        start_time = time.time()
        
        request_data = {
            "user_id": "benchmark_user",
            "market_context": {
                "symbol": "SPY",
                "current_price": 450.0,
                "vix": 20.0,
                "trend": "neutral"
            },
            "user_intent": "Generate covered call strategy",
            "max_cost_usd": 0.25  # Lower cost for faster processing
        }
        
        with self.client.post(
            "/strategy/generate",
            json=request_data,
            headers=self.headers,
            catch_response=True
        ) as response:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                if response_time < 2000:  # Under 2 seconds
                    response.success()
                else:
                    response.failure(f"Slow response: {response_time:.0f}ms")
            elif response.status_code == 429:
                response.success()  # Rate limiting is acceptable
            else:
                response.failure(f"Benchmark failed: {response.status_code}")

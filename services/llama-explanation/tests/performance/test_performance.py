# services/llama-explanation/tests/performance/test_performance.py
import asyncio
import time
import statistics
import json
from typing import List, Dict, Any
import httpx
import pytest
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np

class LlamaPerformanceTester:
    """Comprehensive performance testing for Llama service"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
        self.results: List[Dict[str, Any]] = []
    
    async def test_single_request_latency(self, num_requests: int = 100) -> Dict[str, Any]:
        """Test single request latency"""
        print(f"Testing single request latency with {num_requests} requests...")
        
        latencies = []
        token_counts = []
        confidence_scores = []
        
        test_context = {
            "analysis_type": "sentiment",
            "symbol": "AAPL",
            "sentiment_score": 0.75,
            "current_price": 150.0,
            "iv_rank": 65
        }
        
        for i in range(num_requests):
            start_time = time.time()
            
            try:
                response = await self.client.post(
                    f"{self.base_url}/explain",
                    json={
                        "context": test_context,
                        "max_tokens": 200,
                        "temperature": 0.1,
                        "priority": "normal"
                    }
                )
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms
                
                if response.status_code == 200:
                    data = response.json()
                    latencies.append(latency)
                    token_counts.append(data.get('tokens_used', 0))
                    confidence_scores.append(data.get('confidence_score', 0.0))
                else:
                    print(f"Request {i+1} failed with status {response.status_code}")
                
            except Exception as e:
                print(f"Request {i+1} failed with error: {str(e)}")
            
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        if not latencies:
            return {"error": "No successful requests"}
        
        return {
            "test_type": "single_request_latency",
            "num_requests": len(latencies),
            "latency_stats": {
                "mean_ms": statistics.mean(latencies),
                "median_ms": statistics.median(latencies),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            "token_stats": {
                "mean_tokens": statistics.mean(token_counts),
                "total_tokens": sum(token_counts)
            },
            "confidence_stats": {
                "mean_confidence": statistics.mean(confidence_scores),
                "min_confidence": min(confidence_scores),
                "max_confidence": max(confidence_scores)
            },
            "throughput": {
                "requests_per_second": len(latencies) / (max(latencies) / 1000),
                "tokens_per_second": sum(token_counts) / (sum(latencies) / 1000)
            }
        }
    
    async def test_concurrent_requests(
        self, 
        concurrent_users: int = 10, 
        requests_per_user: int = 5
    ) -> Dict[str, Any]:
        """Test concurrent request handling"""
        print(f"Testing concurrent requests: {concurrent_users} users, {requests_per_user} requests each...")
        
        async def user_simulation(user_id: int) -> List[Dict[str, Any]]:
            """Simulate a single user making multiple requests"""
            user_results = []
            
            for req_id in range(requests_per_user):
                start_time = time.time()
                
                try:
                    response = await self.client.post(
                        f"{self.base_url}/explain",
                        json={
                            "context": {
                                "analysis_type": "price_prediction",
                                "symbol": f"TEST{user_id}",
                                "current_price": 100.0 + user_id
                            },
                            "max_tokens": 150,
                            "temperature": 0.1,
                            "priority": "normal"
                        }
                    )
                    
                    end_time = time.time()
                    latency = (end_time - start_time) * 1000
                    
                    if response.status_code == 200:
                        data = response.json()
                        user_results.append({
                            "user_id": user_id,
                            "request_id": req_id,
                            "latency_ms": latency,
                            "tokens_used": data.get('tokens_used', 0),
                            "success": True
                        })
                    else:
                        user_results.append({
                            "user_id": user_id,
                            "request_id": req_id,
                            "latency_ms": latency,
                            "success": False,
                            "status_code": response.status_code
                        })
                        
                except Exception as e:
                    user_results.append({
                        "user_id": user_id,
                        "request_id": req_id,
                        "success": False,
                        "error": str(e)
                    })
            
            return user_results
        
        # Run concurrent user simulations
        start_time = time.time()
        tasks = [user_simulation(user_id) for user_id in range(concurrent_users)]
        all_results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Flatten results
        flat_results = [result for user_results in all_results for result in user_results]
        successful_results = [r for r in flat_results if r.get('success', False)]
        
        if not successful_results:
            return {"error": "No successful concurrent requests"}
        
        latencies = [r['latency_ms'] for r in successful_results]
        total_tokens = sum(r.get('tokens_used', 0) for r in successful_results)
        
        return {
            "test_type": "concurrent_requests",
            "concurrent_users": concurrent_users,
            "requests_per_user": requests_per_user,
            "total_requests": len(flat_results),
            "successful_requests": len(successful_results),
            "success_rate": len(successful_requests) / len(flat_results),
            "total_time_seconds": total_time,
            "latency_stats": {
                "mean_ms": statistics.mean(latencies),
                "median_ms": statistics.median(latencies),
                "p95_ms": np.percentile(latencies, 95),
                "max_ms": max(latencies),
                "min_ms": min(latencies)
            },
            "throughput": {
                "requests_per_second": len(successful_results) / total_time,
                "tokens_per_second": total_tokens / total_time
            }
        }
    
    async def test_queue_behavior(self, queue_load: int = 50) -> Dict[str, Any]:
        """Test queue handling under load"""
        print(f"Testing queue behavior with {queue_load} simultaneous requests...")
        
        start_time = time.time()
        
        # Submit many requests simultaneously
        tasks = []
        for i in range(queue_load):
            task = self.client.post(
                f"{self.base_url}/explain",
                json={
                    "context": {
                        "analysis_type": "comprehensive",
                        "symbol": f"QUEUE{i}",
                        "current_price": 100.0 + i
                    },
                    "max_tokens": 100,
                    "priority": "normal" if i % 2 == 0 else "high"
                }
            )
            tasks.append(task)
        
        # Wait for all to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        successful = 0
        timeouts = 0
        errors = 0
        latencies = []
        
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                errors += 1
            elif hasattr(response, 'status_code'):
                if response.status_code == 200:
                    successful += 1
                    # Estimate latency (not precise for concurrent)
                    latencies.append(total_time * 1000 / queue_load)
                elif response.status_code == 408:  # Timeout
                    timeouts += 1
                else:
                    errors += 1
            else:
                errors += 1
        
        return {
            "test_type": "queue_behavior",
            "queue_load": queue_load,
            "total_time_seconds": total_time,
            "successful_requests": successful,
            "timeout_requests": timeouts,
            "error_requests": errors,
            "success_rate": successful / queue_load,
            "throughput_rps": successful / total_time,
            "average_latency_ms": statistics.mean(latencies) if latencies else 0
        }
    
    async def test_batch_processing(self, batch_sizes: List[int] = [1, 5, 10]) -> Dict[str, Any]:
        """Test batch processing performance"""
        print("Testing batch processing performance...")
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # Create batch request
            batch_requests = [
                {
                    "context": {
                        "analysis_type": "sentiment",
                        "symbol": f"BATCH{i}",
                        "sentiment_score": 0.5 + (i * 0.1)
                    },
                    "max_tokens": 150
                }
                for i in range(batch_size)
            ]
            
            start_time = time.time()
            
            try:
                response = await self.client.post(
                    f"{self.base_url}/explain/batch",
                    json=batch_requests
                )
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    successful = len([r for r in results if 'error' not in r])
                    
                    batch_results[batch_size] = {
                        "batch_size": batch_size,
                        "total_latency_ms": latency,
                        "latency_per_item_ms": latency / batch_size,
                        "successful_items": successful,
                        "success_rate": successful / batch_size,
                        "throughput_items_per_second": batch_size / (latency / 1000)
                    }
                else:
                    batch_results[batch_size] = {
                        "batch_size": batch_size,
                        "error": f"HTTP {response.status_code}"
                    }
                    
            except Exception as e:
                batch_results[batch_size] = {
                    "batch_size": batch_size,
                    "error": str(e)
                }
        
        return {
            "test_type": "batch_processing",
            "batch_results": batch_results
        }
    
    async def test_memory_usage(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Test memory usage over time"""
        print(f"Testing memory usage over {duration_minutes} minutes...")
        
        memory_samples = []
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        # Generate continuous load
        async def continuous_load():
            while time.time() < end_time:
                try:
                    await self.client.post(
                        f"{self.base_url}/explain",
                        json={
                            "context": {
                                "analysis_type": "options_strategy",
                                "symbol": "MEMORY_TEST",
                                "iv_rank": 50
                            },
                            "max_tokens": 200
                        }
                    )
                except:
                    pass  # Ignore errors for continuous load
                
                await asyncio.sleep(1.0)  # Request every second
        
        # Monitor memory usage
        async def monitor_memory():
            while time.time() < end_time:
                try:
                    response = await self.client.get(f"{self.base_url}/admin/stats")
                    if response.status_code == 200:
                        data = response.json()
                        memory_samples.append({
                            "timestamp": time.time(),
                            "system_memory": data.get('system', {}).get('memory_percent', 0),
                            "gpu_memory": data.get('gpu', {}).get('memory_allocated_gb', 0)
                        })
                except:
                    pass
                
                await asyncio.sleep(10.0)  # Sample every 10 seconds
        
        # Run both tasks
        await asyncio.gather(continuous_load(), monitor_memory())
        
        if not memory_samples:
            return {"error": "No memory samples collected"}
        
        system_memory = [s['system_memory'] for s in memory_samples]
        gpu_memory = [s['gpu_memory'] for s in memory_samples]
        
        return {
            "test_type": "memory_usage",
            "duration_minutes": duration_minutes,
            "samples_collected": len(memory_samples),
            "system_memory_stats": {
                "mean_percent": statistics.mean(system_memory),
                "max_percent": max(system_memory),
                "min_percent": min(system_memory)
            },
            "gpu_memory_stats": {
                "mean_gb": statistics.mean(gpu_memory),
                "max_gb": max(gpu_memory),
                "min_gb": min(gpu_memory)
            }
        }
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite"""
        print("Running comprehensive Llama performance test suite...")
        
        # Check service health first
        try:
            health_response = await self.client.get(f"{self.base_url}/health")
            if health_response.status_code != 200:
                return {"error": "Service not healthy"}
        except Exception as e:
            return {"error": f"Cannot connect to service: {str(e)}"}
        
        test_results = {
            "test_timestamp": time.time(),
            "service_url": self.base_url,
            "tests": {}
        }
        
        # Run all tests
        tests = [
            ("single_request_latency", self.test_single_request_latency(50)),
            ("concurrent_requests", self.test_concurrent_requests(5, 3)),
            ("queue_behavior", self.test_queue_behavior(20)),
            ("batch_processing", self.test_batch_processing([1, 3, 5]))
        ]
        
        for test_name, test_coro in tests:
            print(f"\nRunning {test_name}...")
            try:
                result = await test_coro
                test_results["tests"][test_name] = result
                print(f"✅ {test_name} completed")
            except Exception as e:
                test_results["tests"][test_name] = {"error": str(e)}
                print(f"❌ {test_name} failed: {str(e)}")
        
        # Generate summary
        test_results["summary"] = self._generate_summary(test_results["tests"])
        
        return test_results
    
    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance test summary"""
        summary = {
            "overall_status": "passed",
            "key_metrics": {},
            "recommendations": []
        }
        
        # Extract key metrics
        if "single_request_latency" in test_results:
            latency_test = test_results["single_request_latency"]
            if "latency_stats" in latency_test:
                mean_latency = latency_test["latency_stats"]["mean_ms"]
                p95_latency = latency_test["latency_stats"]["p95_ms"]
                
                summary["key_metrics"]["mean_latency_ms"] = mean_latency
                summary["key_metrics"]["p95_latency_ms"] = p95_latency
                
                # Check against targets
                if mean_latency > 300:  # Target: 210ms
                    summary["overall_status"] = "warning"
                    summary["recommendations"].append("Mean latency exceeds 300ms target")
                
                if p95_latency > 500:  # Target: <500ms
                    summary["overall_status"] = "warning"
                    summary["recommendations"].append("P95 latency exceeds 500ms target")
        
        if "concurrent_requests" in test_results:
            concurrent_test = test_results["concurrent_requests"]
            if "success_rate" in concurrent_test:
                success_rate = concurrent_test["success_rate"]
                summary["key_metrics"]["concurrent_success_rate"] = success_rate
                
                if success_rate < 0.95:  # Target: >95%
                    summary["overall_status"] = "failed"
                    summary["recommendations"].append("Concurrent request success rate below 95%")
        
        if not summary["recommendations"]:
            summary["recommendations"].append("All performance targets met")
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filename: str = "performance_results.json"):
        """Save test results to file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable performance report"""
        report = []
        report.append("=== Llama Performance Test Report ===\n")
        
        if "single_request_latency" in results["tests"]:
            latency_test = results["tests"]["single_request_latency"]
            if "latency_stats" in latency_test:
                stats = latency_test["latency_stats"]
                report.append("Single Request Latency:")
                report.append(f"  Mean: {stats['mean_ms']:.1f}ms")
                report.append(f"  P95:  {stats['p95_ms']:.1f}ms")
                report.append(f"  P99:  {stats['p99_ms']:.1f}ms\n")
        
        if "concurrent_requests" in results["tests"]:
            concurrent_test = results["tests"]["concurrent_requests"]
            report.append("Concurrent Requests:")
            report.append(f"  Success Rate: {concurrent_test.get('success_rate', 0)*100:.1f}%")
            report.append(f"  Throughput: {concurrent_test.get('throughput', {}).get('requests_per_second', 0):.1f} req/s\n")
        
        # Summary
        summary = results.get("summary", {})
        report.append(f"Overall Status: {summary.get('overall_status', 'unknown').upper()}")
        
        recommendations = summary.get("recommendations", [])
        if recommendations:
            report.append("\nRecommendations:")
            for rec in recommendations:
                report.append(f"  • {rec}")
        
        return "\n".join(report)

# Command-line interface for performance testing
async def main():
    """Main function for running performance tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Llama Performance Testing")
    parser.add_argument("--url", default="http://localhost:8000", help="Service URL")
    parser.add_argument("--test", choices=["single", "concurrent", "queue", "batch", "all"], 
                       default="all", help="Test type to run")
    parser.add_argument("--output", default="performance_results.json", help="Output file")
    parser.add_argument("--requests", type=int, default=50, help="Number of requests for single test")
    parser.add_argument("--users", type=int, default=5, help="Concurrent users")
    
    args = parser.parse_args()
    
    tester = LlamaPerformanceTester(args.url)
    
    try:
        if args.test == "all":
            results = await tester.run_comprehensive_test()
        elif args.test == "single":
            results = await tester.test_single_request_latency(args.requests)
        elif args.test == "concurrent":
            results = await tester.test_concurrent_requests(args.users, 5)
        elif args.test == "queue":
            results = await tester.test_queue_behavior(30)
        elif args.test == "batch":
            results = await tester.test_batch_processing([1, 3, 5, 10])
        
        # Save results
        tester.save_results(results, args.output)
        
        # Print report
        if args.test == "all":
            report = tester.generate_report(results)
            print("\n" + report)
        else:
            print(f"\nTest Results: {json.dumps(results, indent=2)}")
            
    finally:
        await tester.client.aclose()

if __name__ == "__main__":
    asyncio.run(main())

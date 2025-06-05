#!/usr/bin/env python3
"""
TFT Model Benchmarking Script
"""

import asyncio
import aiohttp
import time
import statistics
import json
import argparse
from typing import List, Dict, Any
from datetime import datetime

class TFTBenchmark:
    def __init__(self, base_url: str = "http://localhost:8007"):
        self.base_url = base_url
        self.session = None
        
        # Test data
        self.test_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]
        self.test_horizons = [1, 5, 10, 21]
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def benchmark_single_forecast(self, symbol: str, horizons: List[int]) -> Dict[str, Any]:
        """Benchmark a single forecast request"""
        
        url = f"{self.base_url}/forecast/generate"
        payload = {
            "user_id": "benchmark_user",
            "symbol": symbol,
            "forecast_horizons": horizons,
            "include_options_greeks": True,
            "risk_adjustment": True
        }
        
        headers = {"Authorization": "Bearer benchmark_token"}
        
        start_time = time.time()
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                end_time = time.time()
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "response_time": (end_time - start_time) * 1000,  # ms
                        "symbol": symbol,
                        "horizons": horizons,
                        "forecast_id": result.get("forecast_id"),
                        "processing_time": result.get("processing_time_ms", 0),
                        "confidence": result.get("forecasts", {}).get("1d", {}).get("confidence", 0),
                        "error": None
                    }
                else:
                    return {
                        "success": False,
                        "response_time": (end_time - start_time) * 1000,
                        "symbol": symbol,
                        "horizons": horizons,
                        "error": f"HTTP {response.status}: {await response.text()}"
                    }
                    
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "response_time": (end_time - start_time) * 1000,
                "symbol": symbol,
                "horizons": horizons,
                "error": str(e)
            }
    
    async def benchmark_batch_forecast(self, symbols: List[str], horizons: List[int]) -> Dict[str, Any]:
        """Benchmark batch forecast request"""
        
        url = f"{self.base_url}/forecast/batch"
        payload = {
            "symbols": symbols,
            "forecast_horizons": horizons
        }
        
        headers = {"Authorization": "Bearer benchmark_token"}
        
        start_time = time.time()
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                end_time = time.time()
                
                if response.status == 200:
                    results = await response.json()
                    successful = sum(1 for r in results if r.get("forecasts"))
                    
                    return {
                        "success": True,
                        "response_time": (end_time - start_time) * 1000,
                        "symbols": symbols,
                        "horizons": horizons,
                        "total_symbols": len(symbols),
                        "successful_forecasts": successful,
                        "success_rate": successful / len(symbols),
                        "error": None
                    }
                else:
                    return {
                        "success": False,
                        "response_time": (end_time - start_time) * 1000,
                        "symbols": symbols,
                        "horizons": horizons,
                        "error": f"HTTP {response.status}: {await response.text()}"
                    }
                    
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "response_time": (end_time - start_time) * 1000,
                "symbols": symbols,
                "horizons": horizons,
                "error": str(e)
            }
    
    async def benchmark_concurrent_requests(self, num_requests: int = 10) -> List[Dict[str, Any]]:
        """Benchmark concurrent requests"""
        
        print(f"🔄 Running {num_requests} concurrent forecast requests...")
        
        tasks = []
        for i in range(num_requests):
            symbol = self.test_symbols[i % len(self.test_symbols)]
            horizons = self.test_horizons
            task = self.benchmark_single_forecast(symbol, horizons)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "response_time": 0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def benchmark_load_test(self, duration_seconds: int = 60, requests_per_second: int = 5) -> Dict[str, Any]:
        """Run sustained load test"""
        
        print(f"🔥 Running load test for {duration_seconds}s at {requests_per_second} RPS...")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        results = []
        request_count = 0
        
        while time.time() < end_time:
            # Calculate requests to send this second
            current_second = int(time.time() - start_time)
            target_requests = (current_second + 1) * requests_per_second
            
            if request_count < target_requests:
                requests_to_send = min(requests_per_second, target_requests - request_count)
                
                # Send requests
                tasks = []
                for _ in range(requests_to_send):
                    symbol = self.test_symbols[request_count % len(self.test_symbols)]
                    task = self.benchmark_single_forecast(symbol, [1, 5])
                    tasks.append(task)
                    request_count += 1
                
                # Wait for completion
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend([r for r in batch_results if not isinstance(r, Exception)])
            
            # Wait until next second
            await asyncio.sleep(max(0, 1 - (time.time() % 1)))
        
        return {
            "total_requests": len(results),
            "successful_requests": sum(1 for r in results if r.get("success", False)),
            "failed_requests": sum(1 for r in results if not r.get("success", False)),
            "duration_seconds": duration_seconds,
            "actual_rps": len(results) / duration_seconds,
            "results": results
        }
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze benchmark results"""
        
        if not results:
            return {"error": "No results to analyze"}
        
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]
        
        if not successful_results:
            return {
                "total_requests": len(results),
                "success_rate": 0.0,
                "error_rate": 1.0,
                "errors": [r.get("error", "Unknown error") for r in failed_results]
            }
        
        response_times = [r["response_time"] for r in successful_results]
        processing_times = [r.get("processing_time", 0) for r in successful_results if r.get("processing_time")]
        
        analysis = {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "success_rate": len(successful_results) / len(results),
            "error_rate": len(failed_results) / len(results),
            
            "response_time_stats": {
                "min": min(response_times),
                "max": max(response_times),
                "mean": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "p95": self._percentile(response_times, 95),
                "p99": self._percentile(response_times, 99)
            } if response_times else {},
            
            "processing_time_stats": {
                "min": min(processing_times),
                "max": max(processing_times),
                "mean": statistics.mean(processing_times),
                "median": statistics.median(processing_times)
            } if processing_times else {},
            
            "errors": list(set([r.get("error") for r in failed_results if r.get("error")]))
        }
        
        return analysis
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = int(index)
            upper = lower + 1
            weight = index - lower
            return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        
        print("🚀 Starting TFT Forecasting Service Benchmark")
        print("=" * 60)
        
        benchmark_results = {}
        
        # 1. Single request benchmark
        print("\n📊 Single Request Benchmark")
        single_results = []
        for symbol in self.test_symbols:
            result = await self.benchmark_single_forecast(symbol, self.test_horizons)
            single_results.append(result)
            status = "✅" if result["success"] else "❌"
            print(f"  {status} {symbol}: {result['response_time']:.0f}ms")
        
        benchmark_results["single_requests"] = self.analyze_results(single_results)
        
        # 2. Batch request benchmark
        print("\n📦 Batch Request Benchmark")
        batch_result = await self.benchmark_batch_forecast(self.test_symbols, [1, 5, 10])
        status = "✅" if batch_result["success"] else "❌"
        print(f"  {status} Batch ({len(self.test_symbols)} symbols): {batch_result['response_time']:.0f}ms")
        
        benchmark_results["batch_requests"] = batch_result
        
        # 3. Concurrent requests benchmark
        print("\n🔄 Concurrent Requests Benchmark")
        concurrent_results = await self.benchmark_concurrent_requests(10)
        concurrent_analysis = self.analyze_results(concurrent_results)
        
        print(f"  📈 {concurrent_analysis['successful_requests']}/{concurrent_analysis['total_requests']} successful")
        print(f"  ⏱️  Avg response time: {concurrent_analysis['response_time_stats'].get('mean', 0):.0f}ms")
        print(f"  📊 P95 response time: {concurrent_analysis['response_time_stats'].get('p95', 0):.0f}ms")
        
        benchmark_results["concurrent_requests"] = concurrent_analysis
        
        # 4. Load test (optional - shorter duration for demo)
        print("\n🔥 Load Test (30 seconds)")
        load_results = await self.benchmark_load_test(30, 3)  # 30s at 3 RPS
        load_analysis = self.analyze_results(load_results["results"])
        
        print(f"  📈 {load_results['successful_requests']}/{load_results['total_requests']} successful")
        print(f"  📊 Actual RPS: {load_results['actual_rps']:.1f}")
        print(f"  ⏱️  Avg response time: {load_analysis['response_time_stats'].get('mean', 0):.0f}ms")
        
        benchmark_results["load_test"] = {
            **load_results,
            "analysis": load_analysis
        }
        
        return benchmark_results

async def main():
    parser = argparse.ArgumentParser(description="TFT Forecasting Service Benchmark")
    parser.add_argument("--url", default="http://localhost:8007", help="Service URL")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--duration", type=int, default=60, help="Load test duration (seconds)")
    parser.add_argument("--rps", type=int, default=5, help="Requests per second for load test")
    
    args = parser.parse_args()
    
    async with TFTBenchmark(args.url) as benchmark:
        results = await benchmark.run_full_benchmark()
        
        print("\n" + "=" * 60)
        print("📋 BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Summary statistics
        single_stats = results["single_requests"]["response_time_stats"]
        print(f"Single Request Performance:")
        print(f"  • Average: {single_stats.get('mean', 0):.0f}ms")
        print(f"  • P95: {single_stats.get('p95', 0):.0f}ms")
        print(f"  • Success Rate: {results['single_requests']['success_rate']:.1%}")
        
        concurrent_stats = results["concurrent_requests"]["response_time_stats"]
        print(f"\nConcurrent Performance:")
        print(f"  • Average: {concurrent_stats.get('mean', 0):.0f}ms")
        print(f"  • P95: {concurrent_stats.get('p95', 0):.0f}ms")
        print(f"  • Success Rate: {results['concurrent_requests']['success_rate']:.1%}")
        
        load_stats = results["load_test"]["analysis"]["response_time_stats"]
        print(f"\nLoad Test Performance:")
        print(f"  • Sustained RPS: {results['load_test']['actual_rps']:.1f}")
        print(f"  • Average: {load_stats.get('mean', 0):.0f}ms")
        print(f"  • Success Rate: {results['load_test']['analysis']['success_rate']:.1%}")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())

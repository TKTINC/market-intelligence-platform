# services/llama-explanation/scripts/verify_deployment.py
"""
Comprehensive deployment verification script for Llama 2-7B service
Tests all functionality after deployment to ensure service is working correctly
"""

import asyncio
import time
import json
import sys
from typing import Dict, Any, List
import httpx
import argparse

class LlamaDeploymentVerifier:
    """Comprehensive deployment verification for Llama service"""
    
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=timeout)
        self.verification_results = {}
    
    async def verify_health_endpoint(self) -> Dict[str, Any]:
        """Verify health endpoint is working"""
        print("🔍 Verifying health endpoint...")
        
        try:
            response = await self.client.get(f"{self.base_url}/health")
            
            if response.status_code != 200:
                return {
                    "status": "failed",
                    "error": f"Health endpoint returned {response.status_code}"
                }
            
            health_data = response.json()
            
            # Check required fields
            required_fields = ["status", "model_loaded", "gpu_available", "memory_usage"]
            missing_fields = [field for field in required_fields if field not in health_data]
            
            if missing_fields:
                return {
                    "status": "failed",
                    "error": f"Missing fields in health response: {missing_fields}"
                }
            
            # Check service status
            if health_data["status"] not in ["healthy", "degraded"]:
                return {
                    "status": "failed", 
                    "error": f"Service status is '{health_data['status']}'"
                }
            
            return {
                "status": "passed",
                "service_status": health_data["status"],
                "model_loaded": health_data["model_loaded"],
                "gpu_available": health_data["gpu_available"],
                "uptime_seconds": health_data.get("uptime_seconds", 0)
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Health check failed: {str(e)}"
            }
    
    async def verify_explanation_generation(self) -> Dict[str, Any]:
        """Verify explanation generation functionality"""
        print("🔍 Verifying explanation generation...")
        
        test_cases = [
            {
                "name": "sentiment_analysis",
                "context": {
                    "analysis_type": "sentiment",
                    "symbol": "AAPL",
                    "sentiment_score": 0.75,
                    "current_price": 150.0
                },
                "max_tokens": 200,
                "temperature": 0.1
            },
            {
                "name": "price_prediction",
                "context": {
                    "analysis_type": "price_prediction", 
                    "symbol": "TSLA",
                    "prediction": {"direction": "up", "confidence": 0.8},
                    "current_price": 250.0
                },
                "max_tokens": 180,
                "temperature": 0.15
            },
            {
                "name": "options_strategy",
                "context": {
                    "analysis_type": "options_strategy",
                    "symbol": "NVDA",
                    "strategy": {"strategy": "IRON_CONDOR", "max_profit": 500},
                    "iv_rank": 85
                },
                "max_tokens": 220,
                "temperature": 0.1
            }
        ]
        
        results = {}
        
        for test_case in test_cases:
            print(f"  Testing {test_case['name']}...")
            
            try:
                start_time = time.time()
                
                response = await self.client.post(
                    f"{self.base_url}/explain",
                    json=test_case
                )
                
                latency = (time.time() - start_time) * 1000
                
                if response.status_code != 200:
                    results[test_case['name']] = {
                        "status": "failed",
                        "error": f"HTTP {response.status_code}: {response.text}"
                    }
                    continue
                
                data = response.json()
                
                # Validate response structure
                required_fields = ["explanation", "tokens_used", "processing_time_ms", "confidence_score"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    results[test_case['name']] = {
                        "status": "failed",
                        "error": f"Missing response fields: {missing_fields}"
                    }
                    continue
                
                # Validate response content
                explanation = data["explanation"]
                tokens_used = data["tokens_used"]
                confidence_score = data["confidence_score"]
                
                if not explanation or len(explanation.strip()) < 50:
                    results[test_case['name']] = {
                        "status": "failed",
                        "error": "Explanation too short or empty"
                    }
                    continue
                
                if tokens_used <= 0:
                    results[test_case['name']] = {
                        "status": "failed",
                        "error": "Invalid tokens_used value"
                    }
                    continue
                
                if not (0.0 <= confidence_score <= 1.0):
                    results[test_case['name']] = {
                        "status": "failed",
                        "error": f"Invalid confidence_score: {confidence_score}"
                    }
                    continue
                
                # Check latency
                if latency > 2000:  # 2 second warning threshold
                    status = "warning"
                    note = f"High latency: {latency:.1f}ms"
                else:
                    status = "passed"
                    note = f"Latency: {latency:.1f}ms"
                
                results[test_case['name']] = {
                    "status": status,
                    "latency_ms": latency,
                    "tokens_used": tokens_used,
                    "confidence_score": confidence_score,
                    "explanation_length": len(explanation),
                    "note": note
                }
                
            except Exception as e:
                results[test_case['name']] = {
                    "status": "failed",
                    "error": f"Request failed: {str(e)}"
                }
        
        # Overall status
        all_passed = all(r.get("status") == "passed" for r in results.values())
        any_failed = any(r.get("status") == "failed" for r in results.values())
        
        overall_status = "passed" if all_passed else ("failed" if any_failed else "warning")
        
        return {
            "status": overall_status,
            "test_cases": results,
            "summary": {
                "total_tests": len(test_cases),
                "passed": len([r for r in results.values() if r.get("status") == "passed"]),
                "failed": len([r for r in results.values() if r.get("status") == "failed"]),
                "warnings": len([r for r in results.values() if r.get("status") == "warning"])
            }
        }
    
    async def verify_batch_processing(self) -> Dict[str, Any]:
        """Verify batch processing functionality"""
        print("🔍 Verifying batch processing...")
        
        try:
            batch_requests = [
                {
                    "context": {
                        "analysis_type": "sentiment",
                        "symbol": f"BATCH{i}",
                        "sentiment_score": 0.5 + (i * 0.1)
                    },
                    "max_tokens": 150,
                    "temperature": 0.1
                }
                for i in range(3)
            ]
            
            start_time = time.time()
            
            response = await self.client.post(
                f"{self.base_url}/explain/batch",
                json=batch_requests
            )
            
            latency = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return {
                    "status": "failed",
                    "error": f"Batch endpoint returned {response.status_code}"
                }
            
            data = response.json()
            
            if "results" not in data:
                return {
                    "status": "failed",
                    "error": "Missing 'results' in batch response"
                }
            
            results = data["results"]
            
            if len(results) != len(batch_requests):
                return {
                    "status": "failed",
                    "error": f"Expected {len(batch_requests)} results, got {len(results)}"
                }
            
            # Check each result
            successful_results = 0
            for i, result in enumerate(results):
                if isinstance(result, dict) and "explanation" in result:
                    successful_results += 1
            
            success_rate = successful_results / len(results)
            
            if success_rate < 0.8:  # 80% success threshold
                return {
                    "status": "failed",
                    "error": f"Batch success rate too low: {success_rate:.1%}"
                }
            
            return {
                "status": "passed",
                "batch_size": len(batch_requests),
                "successful_results": successful_results,
                "success_rate": success_rate,
                "total_latency_ms": latency,
                "latency_per_item_ms": latency / len(batch_requests)
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Batch processing test failed: {str(e)}"
            }
    
    async def verify_concurrent_requests(self) -> Dict[str, Any]:
        """Verify concurrent request handling"""
        print("🔍 Verifying concurrent request handling...")
        
        try:
            # Submit 5 concurrent requests
            concurrent_requests = 5
            
            tasks = []
            for i in range(concurrent_requests):
                task = self.client.post(
                    f"{self.base_url}/explain",
                    json={
                        "context": {
                            "analysis_type": "comprehensive",
                            "symbol": f"CONC{i}",
                            "current_price": 100.0 + i
                        },
                        "max_tokens": 150,
                        "priority": "normal"
                    }
                )
                tasks.append(task)
            
            start_time = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            successful_responses = 0
            errors = 0
            
            for response in responses:
                if isinstance(response, Exception):
                    errors += 1
                elif hasattr(response, 'status_code') and response.status_code == 200:
                    successful_responses += 1
                else:
                    errors += 1
            
            success_rate = successful_responses / concurrent_requests
            
            if success_rate < 0.8:  # 80% success threshold
                return {
                    "status": "failed",
                    "error": f"Concurrent request success rate too low: {success_rate:.1%}"
                }
            
            return {
                "status": "passed",
                "concurrent_requests": concurrent_requests,
                "successful_responses": successful_responses,
                "errors": errors,
                "success_rate": success_rate,
                "total_time_seconds": total_time,
                "average_latency_ms": (total_time * 1000) / concurrent_requests
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Concurrent request test failed: {str(e)}"
            }
    
    async def verify_error_handling(self) -> Dict[str, Any]:
        """Verify error handling"""
        print("🔍 Verifying error handling...")
        
        test_cases = [
            {
                "name": "invalid_json",
                "request": "invalid json",
                "expected_status": 422
            },
            {
                "name": "missing_context",
                "request": {"max_tokens": 100},
                "expected_status": 422
            },
            {
                "name": "invalid_max_tokens",
                "request": {
                    "context": {"test": "data"},
                    "max_tokens": 10000  # Exceeds limit
                },
                "expected_status": 422
            },
            {
                "name": "invalid_temperature",
                "request": {
                    "context": {"test": "data"},
                    "temperature": 2.0  # Exceeds limit
                },
                "expected_status": 422
            }
        ]
        
        results = {}
        
        for test_case in test_cases:
            try:
                if test_case["name"] == "invalid_json":
                    # Send invalid JSON
                    response = await self.client.post(
                        f"{self.base_url}/explain",
                        content=test_case["request"],
                        headers={"Content-Type": "application/json"}
                    )
                else:
                    response = await self.client.post(
                        f"{self.base_url}/explain",
                        json=test_case["request"]
                    )
                
                if response.status_code == test_case["expected_status"]:
                    results[test_case["name"]] = {"status": "passed"}
                else:
                    results[test_case["name"]] = {
                        "status": "failed",
                        "error": f"Expected {test_case['expected_status']}, got {response.status_code}"
                    }
                    
            except Exception as e:
                results[test_case["name"]] = {
                    "status": "failed", 
                    "error": f"Exception: {str(e)}"
                }
        
        all_passed = all(r.get("status") == "passed" for r in results.values())
        
        return {
            "status": "passed" if all_passed else "failed",
            "test_cases": results
        }
    
    async def verify_monitoring_endpoints(self) -> Dict[str, Any]:
        """Verify monitoring and admin endpoints"""
        print("🔍 Verifying monitoring endpoints...")
        
        endpoints = [
            {"path": "/status", "name": "status"},
            {"path": "/metrics", "name": "metrics"},
            {"path": "/admin/stats", "name": "admin_stats"}
        ]
        
        results = {}
        
        for endpoint in endpoints:
            try:
                response = await self.client.get(f"{self.base_url}{endpoint['path']}")
                
                if response.status_code == 200:
                    results[endpoint['name']] = {
                        "status": "passed",
                        "response_size": len(response.content)
                    }
                else:
                    results[endpoint['name']] = {
                        "status": "failed",
                        "error": f"HTTP {response.status_code}"
                    }
                    
            except Exception as e:
                results[endpoint['name']] = {
                    "status": "failed",
                    "error": f"Request failed: {str(e)}"
                }
        
        all_passed = all(r.get("status") == "passed" for r in results.values())
        
        return {
            "status": "passed" if all_passed else "failed",
            "endpoints": results
        }
    
    async def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run all verification tests"""
        print("🚀 Starting comprehensive Llama deployment verification...\n")
        
        verification_tests = [
            ("health_endpoint", self.verify_health_endpoint),
            ("explanation_generation", self.verify_explanation_generation),
            ("batch_processing", self.verify_batch_processing),
            ("concurrent_requests", self.verify_concurrent_requests),
            ("error_handling", self.verify_error_handling),
            ("monitoring_endpoints", self.verify_monitoring_endpoints)
        ]
        
        results = {
            "verification_timestamp": time.time(),
            "service_url": self.base_url,
            "tests": {},
            "summary": {}
        }
        
        passed_tests = 0
        failed_tests = 0
        warning_tests = 0
        
        for test_name, test_func in verification_tests:
            print(f"Running {test_name}...")
            
            try:
                test_result = await test_func()
                results["tests"][test_name] = test_result
                
                status = test_result.get("status", "unknown")
                if status == "passed":
                    passed_tests += 1
                    print(f"✅ {test_name} - PASSED")
                elif status == "warning":
                    warning_tests += 1
                    print(f"⚠️  {test_name} - WARNING")
                else:
                    failed_tests += 1
                    print(f"❌ {test_name} - FAILED")
                    if "error" in test_result:
                        print(f"   Error: {test_result['error']}")
                        
            except Exception as e:
                failed_tests += 1
                results["tests"][test_name] = {
                    "status": "failed",
                    "error": f"Test execution failed: {str(e)}"
                }
                print(f"❌ {test_name} - FAILED (Exception: {str(e)})")
            
            print()  # Empty line for readability
        
        # Generate summary
        total_tests = len(verification_tests)
        overall_status = "passed" if failed_tests == 0 else "failed"
        
        results["summary"] = {
            "overall_status": overall_status,
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "warnings": warning_tests,
            "success_rate": passed_tests / total_tests
        }
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable verification report"""
        lines = []
        lines.append("=" * 60)
        lines.append("   LLAMA 2-7B DEPLOYMENT VERIFICATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        summary = results.get("summary", {})
        lines.append(f"Overall Status: {summary.get('overall_status', 'unknown').upper()}")
        lines.append(f"Service URL: {results.get('service_url', 'unknown')}")
        lines.append(f"Tests Passed: {summary.get('passed', 0)}/{summary.get('total_tests', 0)}")
        lines.append(f"Success Rate: {summary.get('success_rate', 0):.1%}")
        lines.append("")
        
        # Test details
        for test_name, test_result in results.get("tests", {}).items():
            status = test_result.get("status", "unknown")
            icon = "✅" if status == "passed" else ("⚠️" if status == "warning" else "❌")
            
            lines.append(f"{icon} {test_name.replace('_', ' ').title()}")
            
            if status == "failed" and "error" in test_result:
                lines.append(f"   Error: {test_result['error']}")
            elif status == "passed" and test_name == "explanation_generation":
                summary_data = test_result.get("summary", {})
                lines.append(f"   Tests: {summary_data.get('passed', 0)}/{summary_data.get('total_tests', 0)} passed")
            
            lines.append("")
        
        # Recommendations
        lines.append("Recommendations:")
        if summary.get("overall_status") == "passed":
            lines.append("• ✅ Service is ready for production use")
        else:
            lines.append("• ❌ Service has issues that need to be resolved")
            lines.append("• Review failed tests and fix underlying issues")
            lines.append("• Re-run verification after fixes")
        
        return "\n".join(lines)
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()

async def main():
    """Main verification function"""
    parser = argparse.ArgumentParser(description="Llama Deployment Verification")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Llama service URL")
    parser.add_argument("--output", default="verification_results.json",
                       help="Output file for detailed results")
    parser.add_argument("--timeout", type=float, default=60.0,
                       help="Request timeout in seconds")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")
    
    args = parser.parse_args()
    
    verifier = LlamaDeploymentVerifier(args.url, args.timeout)
    
    try:
        # Run verification
        results = await verifier.run_comprehensive_verification()
        
        # Save detailed results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        if not args.quiet:
            print(f"Detailed results saved to {args.output}")
        
        # Generate and display report
        report = verifier.generate_report(results)
        print(report)
        
        # Exit with appropriate code
        overall_status = results.get("summary", {}).get("overall_status", "failed")
        sys.exit(0 if overall_status == "passed" else 1)
        
    except KeyboardInterrupt:
        print("\nVerification interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Verification failed: {str(e)}")
        sys.exit(1)
    finally:
        await verifier.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

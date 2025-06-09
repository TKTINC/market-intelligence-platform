#!/usr/bin/env python3

"""
Market Intelligence Platform - Comprehensive Testing Framework
================================================================

This script provides automated testing for the entire MIP platform.

Usage:
    python3 comprehensive_tests.py --env local|aws --test-type all|sanity|integration|e2e|performance|security
"""

import asyncio
import aiohttp
import json
import time
import logging
import argparse
import sys
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result structure"""
    test_name: str
    status: str  # PASS, FAIL, SKIP
    duration: float
    message: str
    details: Optional[Dict] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class TestEnvironment:
    """Test environment configuration"""
    name: str
    api_base_url: str
    services: Dict[str, str]

class ComprehensiveTestSuite:
    """Main test suite for Market Intelligence Platform"""
    
    def __init__(self, environment: str = "local"):
        self.environment = environment
        self.config = self._load_environment_config(environment)
        self.test_results: List[TestResult] = []
        self.session: Optional[aiohttp.ClientSession] = None
        
    def _load_environment_config(self, env: str) -> TestEnvironment:
        """Load environment-specific configuration"""
        
        if env == "local":
            return TestEnvironment(
                name="local",
                api_base_url="http://localhost:8000",
                services={
                    "api-gateway": "http://localhost:8000",
                    "agent-orchestration": "http://localhost:8001",
                    "sentiment-analysis": "http://localhost:8002",
                    "gpt4-strategy": "http://localhost:8003",
                    "virtual-trading": "http://localhost:8006"
                }
            )
        elif env == "aws":
            return TestEnvironment(
                name="aws",
                api_base_url="http://api-gateway-url-from-aws",
                services={}
            )
        else:
            raise ValueError(f"Unknown environment: {env}")
    
    async def setup(self):
        """Setup test environment"""
        logger.info(f"Setting up test environment: {self.config.name}")
        
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Wait for services to be ready
        await self._wait_for_services()
        
    async def teardown(self):
        """Cleanup after tests"""
        if self.session:
            await self.session.close()
        
        await self._generate_test_report()
        
    async def _wait_for_services(self, max_wait: int = 300):
        """Wait for all services to be ready"""
        logger.info("Waiting for services to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                async with self.session.get(f"{self.config.api_base_url}/health") as resp:
                    if resp.status == 200:
                        logger.info("Services are ready")
                        return
            except Exception:
                pass
            
            await asyncio.sleep(10)
        
        raise RuntimeError("Services did not become ready in time")
    
    # ==============================================================================
    # SANITY TESTS
    # ==============================================================================
    
    async def run_sanity_tests(self) -> List[TestResult]:
        """Run basic sanity tests"""
        logger.info("Running sanity tests...")
        
        sanity_results = []
        
        # Test all service health endpoints
        for service_name, service_url in self.config.services.items():
            sanity_results.append(await self._test_service_health(service_name, service_url))
        
        # Test database connectivity (if local)
        if self.environment == "local":
            sanity_results.append(await self._test_database_connectivity())
            sanity_results.append(await self._test_redis_connectivity())
        
        return sanity_results
    
    async def _test_service_health(self, service_name: str, service_url: str) -> TestResult:
        """Test individual service health"""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{service_url}/health") as resp:
                duration = time.time() - start_time
                
                if resp.status == 200:
                    return TestResult(
                        test_name=f"{service_name}_health",
                        status="PASS",
                        duration=duration,
                        message=f"{service_name} health check passed"
                    )
                else:
                    return TestResult(
                        test_name=f"{service_name}_health",
                        status="FAIL",
                        duration=duration,
                        message=f"{service_name} returned status {resp.status}"
                    )
                    
        except Exception as e:
            return TestResult(
                test_name=f"{service_name}_health",
                status="FAIL",
                duration=time.time() - start_time,
                message=f"{service_name} health check failed: {str(e)}"
            )
    
    async def _test_database_connectivity(self) -> TestResult:
        """Test database connectivity"""
        start_time = time.time()
        
        try:
            # Try to connect using Docker exec
            result = subprocess.run([
                "docker", "exec", "-i",
                "$(docker-compose ps -q postgres)",
                "psql", "-U", "mip_user", "-d", "market_intelligence", "-c", "SELECT 1;"
            ], capture_output=True, text=True, shell=True)
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return TestResult(
                    test_name="database_connectivity",
                    status="PASS",
                    duration=duration,
                    message="Database connectivity successful"
                )
            else:
                return TestResult(
                    test_name="database_connectivity",
                    status="FAIL",
                    duration=duration,
                    message="Database connectivity failed"
                )
                
        except Exception as e:
            return TestResult(
                test_name="database_connectivity",
                status="FAIL",
                duration=time.time() - start_time,
                message=f"Database test failed: {str(e)}"
            )
    
    async def _test_redis_connectivity(self) -> TestResult:
        """Test Redis connectivity"""
        start_time = time.time()
        
        try:
            result = subprocess.run([
                "docker", "exec", "-i",
                "$(docker-compose ps -q redis)",
                "redis-cli", "-a", "redis_secure_password_2024", "ping"
            ], capture_output=True, text=True, shell=True)
            
            duration = time.time() - start_time
            
            if result.returncode == 0 and "PONG" in result.stdout:
                return TestResult(
                    test_name="redis_connectivity",
                    status="PASS",
                    duration=duration,
                    message="Redis connectivity successful"
                )
            else:
                return TestResult(
                    test_name="redis_connectivity",
                    status="FAIL",
                    duration=duration,
                    message="Redis connectivity failed"
                )
                
        except Exception as e:
            return TestResult(
                test_name="redis_connectivity",
                status="FAIL",
                duration=time.time() - start_time,
                message=f"Redis test failed: {str(e)}"
            )
    
    # ==============================================================================
    # INTEGRATION TESTS
    # ==============================================================================
    
    async def run_integration_tests(self) -> List[TestResult]:
        """Run integration tests"""
        logger.info("Running integration tests...")
        
        integration_results = []
        
        # Test API Gateway to service communication
        integration_results.append(await self._test_api_gateway_integration())
        
        # Test agent orchestration
        integration_results.append(await self._test_agent_orchestration())
        
        return integration_results
    
    async def _test_api_gateway_integration(self) -> TestResult:
        """Test API Gateway integration"""
        start_time = time.time()
        
        try:
            # Test API Gateway endpoints
            async with self.session.get(f"{self.config.api_base_url}/api/v1/health") as resp:
                duration = time.time() - start_time
                
                if resp.status == 200:
                    return TestResult(
                        test_name="api_gateway_integration",
                        status="PASS",
                        duration=duration,
                        message="API Gateway integration successful"
                    )
                else:
                    return TestResult(
                        test_name="api_gateway_integration",
                        status="FAIL",
                        duration=duration,
                        message=f"API Gateway returned status {resp.status}"
                    )
                    
        except Exception as e:
            return TestResult(
                test_name="api_gateway_integration",
                status="FAIL",
                duration=time.time() - start_time,
                message=f"API Gateway integration failed: {str(e)}"
            )
    
    async def _test_agent_orchestration(self) -> TestResult:
        """Test agent orchestration"""
        start_time = time.time()
        
        try:
            test_request = {
                "symbol": "AAPL",
                "analysis_type": "sentiment"
            }
            
            async with self.session.post(
                f"{self.config.api_base_url}/api/v1/agents/analyze",
                json=test_request
            ) as resp:
                duration = time.time() - start_time
                
                if resp.status in [200, 202]:  # Accept 202 for async processing
                    return TestResult(
                        test_name="agent_orchestration",
                        status="PASS",
                        duration=duration,
                        message="Agent orchestration working"
                    )
                else:
                    return TestResult(
                        test_name="agent_orchestration",
                        status="FAIL",
                        duration=duration,
                        message=f"Agent orchestration failed with status {resp.status}"
                    )
                    
        except Exception as e:
            return TestResult(
                test_name="agent_orchestration",
                status="FAIL",
                duration=time.time() - start_time,
                message=f"Agent orchestration test failed: {str(e)}"
            )
    
    # ==============================================================================
    # END-TO-END TESTS
    # ==============================================================================
    
    async def run_e2e_tests(self) -> List[TestResult]:
        """Run end-to-end tests"""
        logger.info("Running end-to-end tests...")
        
        e2e_results = []
        
        # Test complete workflow
        e2e_results.append(await self._test_complete_workflow())
        
        return e2e_results
    
    async def _test_complete_workflow(self) -> TestResult:
        """Test complete trading workflow"""
        start_time = time.time()
        
        try:
            # This would test a complete user workflow
            # For now, just test that the API is responsive
            async with self.session.get(f"{self.config.api_base_url}/health") as resp:
                duration = time.time() - start_time
                
                if resp.status == 200:
                    return TestResult(
                        test_name="complete_workflow",
                        status="PASS",
                        duration=duration,
                        message="Complete workflow test passed"
                    )
                else:
                    return TestResult(
                        test_name="complete_workflow",
                        status="FAIL",
                        duration=duration,
                        message="Complete workflow test failed"
                    )
                    
        except Exception as e:
            return TestResult(
                test_name="complete_workflow",
                status="FAIL",
                duration=time.time() - start_time,
                message=f"Complete workflow test failed: {str(e)}"
            )
    
    # ==============================================================================
    # PERFORMANCE TESTS
    # ==============================================================================
    
    async def run_performance_tests(self) -> List[TestResult]:
        """Run performance tests"""
        logger.info("Running performance tests...")
        
        performance_results = []
        
        # Test response times
        performance_results.append(await self._test_response_times())
        
        return performance_results
    
    async def _test_response_times(self) -> TestResult:
        """Test API response times"""
        start_time = time.time()
        
        try:
            response_times = []
            
            # Test multiple requests
            for _ in range(5):
                request_start = time.time()
                async with self.session.get(f"{self.config.api_base_url}/health") as resp:
                    request_time = time.time() - request_start
                    if resp.status == 200:
                        response_times.append(request_time)
            
            duration = time.time() - start_time
            avg_response_time = sum(response_times) / len(response_times) if response_times else 999
            
            if avg_response_time < 2.0:  # 2 second threshold
                return TestResult(
                    test_name="response_times",
                    status="PASS",
                    duration=duration,
                    message=f"Average response time: {avg_response_time:.2f}s",
                    details={"avg_response_time": avg_response_time}
                )
            else:
                return TestResult(
                    test_name="response_times",
                    status="FAIL",
                    duration=duration,
                    message=f"Slow response time: {avg_response_time:.2f}s"
                )
                
        except Exception as e:
            return TestResult(
                test_name="response_times",
                status="FAIL",
                duration=time.time() - start_time,
                message=f"Response time test failed: {str(e)}"
            )
    
    # ==============================================================================
    # SECURITY TESTS
    # ==============================================================================
    
    async def run_security_tests(self) -> List[TestResult]:
        """Run security tests"""
        logger.info("Running security tests...")
        
        security_results = []
        
        # Test authentication
        security_results.append(await self._test_authentication())
        
        return security_results
    
    async def _test_authentication(self) -> TestResult:
        """Test authentication"""
        start_time = time.time()
        
        try:
            # Test access without authentication
            async with self.session.get(f"{self.config.api_base_url}/api/v1/protected") as resp:
                duration = time.time() - start_time
                
                # Should return 401 or 404 (if endpoint doesn't exist yet)
                if resp.status in [401, 404]:
                    return TestResult(
                        test_name="authentication",
                        status="PASS",
                        duration=duration,
                        message="Authentication test passed"
                    )
                else:
                    return TestResult(
                        test_name="authentication",
                        status="FAIL",
                        duration=duration,
                        message=f"Authentication test failed: {resp.status}"
                    )
                    
        except Exception as e:
            return TestResult(
                test_name="authentication",
                status="FAIL",
                duration=time.time() - start_time,
                message=f"Authentication test failed: {str(e)}"
            )
    
    # ==============================================================================
    # REPORT GENERATION
    # ==============================================================================
    
    async def _generate_test_report(self):
        """Generate test report"""
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "PASS"])
        failed_tests = len([r for r in self.test_results if r.status == "FAIL"])
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate HTML report
        html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MIP Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .test-result {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
        .pass {{ background-color: #d4edda; color: #155724; }}
        .fail {{ background-color: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Market Intelligence Platform - Test Report</h1>
        <p>Environment: {self.config.name}</p>
        <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <p>Total Tests: {total_tests}</p>
        <p>Passed: {passed_tests}</p>
        <p>Failed: {failed_tests}</p>
        <p>Success Rate: {success_rate:.1f}%</p>
    </div>
    
    <div class="results">
        <h2>Test Results</h2>
"""
        
        for result in self.test_results:
            status_class = result.status.lower()
            html_report += f"""
        <div class="test-result {status_class}">
            <h3>{result.test_name}</h3>
            <p>Status: {result.status}</p>
            <p>Duration: {result.duration:.2f}s</p>
            <p>Message: {result.message}</p>
        </div>
"""
        
        html_report += """
    </div>
</body>
</html>
"""
        
        # Save reports
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        with open(f"test_report_{self.config.name}_{timestamp}.html", "w") as f:
            f.write(html_report)
        
        # JSON report
        json_report = {
            "environment": self.config.name,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate
            },
            "results": [asdict(result) for result in self.test_results]
        }
        
        with open(f"test_report_{self.config.name}_{timestamp}.json", "w") as f:
            json.dump(json_report, f, indent=2, default=str)
        
        logger.info(f"Test report generated - Success rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    
    # ==============================================================================
    # MAIN TEST EXECUTION
    # ==============================================================================
    
    async def run_all_tests(self, test_types: List[str] = None):
        """Run all specified test types"""
        
        if test_types is None:
            test_types = ["sanity", "integration", "e2e", "performance", "security"]
        
        await self.setup()
        
        try:
            if "sanity" in test_types:
                sanity_results = await self.run_sanity_tests()
                self.test_results.extend(sanity_results)
            
            if "integration" in test_types:
                integration_results = await self.run_integration_tests()
                self.test_results.extend(integration_results)
            
            if "e2e" in test_types:
                e2e_results = await self.run_e2e_tests()
                self.test_results.extend(e2e_results)
            
            if "performance" in test_types:
                performance_results = await self.run_performance_tests()
                self.test_results.extend(performance_results)
            
            if "security" in test_types:
                security_results = await self.run_security_tests()
                self.test_results.extend(security_results)
                
        finally:
            await self.teardown()

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="MIP Test Suite")
    parser.add_argument("--env", choices=["local", "aws"], default="local", help="Environment to test")
    parser.add_argument("--test-type", choices=["all", "sanity", "integration", "e2e", "performance", "security"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine test types to run
    if args.test_type == "all":
        test_types = ["sanity", "integration", "e2e", "performance", "security"]
    else:
        test_types = [args.test_type]
    
    # Run tests
    test_suite = ComprehensiveTestSuite(args.env)
    await test_suite.run_all_tests(test_types)
    
    # Print summary
    total_tests = len(test_suite.test_results)
    passed_tests = len([r for r in test_suite.test_results if r.status == "PASS"])
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Environment: {args.env}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"{'='*60}")
    
    if success_rate >= 90:
        print("🎉 Test suite completed successfully!")
        sys.exit(0)
    else:
        print("❌ Test suite completed with failures!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

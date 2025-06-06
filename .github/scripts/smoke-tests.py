#!/usr/bin/env python3
"""
=============================================================================
SMOKE TESTS SCRIPT
Basic health and functionality validation for MIP Platform deployments
=============================================================================
"""

import asyncio
import aiohttp
import json
import sys
import time
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    test_name: str
    status: str  # 'pass', 'fail', 'error', 'skip'
    duration_ms: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class SmokeTestSuite:
    environment: str
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    skipped_tests: int
    total_duration_ms: float
    overall_status: str
    test_results: List[TestResult]

class MIPSmokeTests:
    def __init__(self, environment: str):
        self.environment = environment
        self.session: Optional[aiohttp.ClientSession] = None
        self.start_time = time.time()
        
        # Environment-specific configurations
        self.configs = {
            'development': {
                'gateway_url': 'http://localhost:8000',
                'dashboard_url': 'http://localhost:3000',
                'timeout': 10,
                'expect_auth': False,
                'expect_https': False
            },
            'staging': {
                'gateway_url': os.getenv('STAGING_GATEWAY_URL', 'https://mip-staging.example.com'),
                'dashboard_url': os.getenv('STAGING_DASHBOARD_URL', 'https://dashboard-staging.example.com'),
                'timeout': 15,
                'expect_auth': True,
                'expect_https': True
            },
            'production': {
                'gateway_url': os.getenv('PROD_GATEWAY_URL', 'https://mip.example.com'),
                'dashboard_url': os.getenv('PROD_DASHBOARD_URL', 'https://dashboard.mip.example.com'),
                'timeout': 20,
                'expect_auth': True,
                'expect_https': True
            }
        }
        
        if environment not in self.configs:
            raise ValueError(f"Unknown environment: {environment}")
        
        self.config = self.configs[environment]
        self.test_results: List[TestResult] = []
    
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(ssl=False if self.environment == 'development' else True)
        timeout = aiohttp.ClientTimeout(total=self.config['timeout'])
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'MIP-SmokeTest/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and record the result."""
        start_time = time.time()
        
        try:
            logger.info(f"Running test: {test_name}")
            details = await test_func()
            
            duration_ms = (time.time() - start_time) * 1000
            result = TestResult(
                test_name=test_name,
                status='pass',
                duration_ms=duration_ms,
                details=details
            )
            
            logger.info(f"✓ {test_name} - PASSED ({duration_ms:.1f}ms)")
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = TestResult(
                test_name=test_name,
                status='error',
                duration_ms=duration_ms,
                details={},
                error_message=str(e)
            )
            
            logger.error(f"✗ {test_name} - ERROR: {e}")
        
        self.test_results.append(result)
        return result
    
    async def test_gateway_health(self) -> Dict[str, Any]:
        """Test gateway health endpoint."""
        url = f"{self.config['gateway_url']}/health"
        
        async with self.session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Health check failed with status {response.status}")
            
            data = await response.json()
            
            return {
                'status_code': response.status,
                'response_data': data,
                'has_version': 'version' in data,
                'has_uptime': 'uptime' in data or 'uptime_seconds' in data
            }
    
    async def test_gateway_readiness(self) -> Dict[str, Any]:
        """Test gateway readiness endpoint."""
        url = f"{self.config['gateway_url']}/ready"
        
        async with self.session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Readiness check failed with status {response.status}")
            
            data = await response.json()
            
            return {
                'status_code': response.status,
                'response_data': data,
                'services_ready': data.get('services_ready', 0),
                'database_connected': data.get('database', {}).get('status') == 'connected'
            }
    
    async def test_authentication_endpoint(self) -> Dict[str, Any]:
        """Test authentication endpoint."""
        url = f"{self.config['gateway_url']}/api/v1/auth/login"
        
        # Test with invalid credentials (should return 401)
        invalid_payload = {
            'username': 'test_invalid_user',
            'password': 'invalid_password'
        }
        
        async with self.session.post(url, json=invalid_payload) as response:
            if response.status not in [401, 422]:  # 401 Unauthorized or 422 Validation Error
                raise Exception(f"Auth endpoint returned unexpected status {response.status}")
            
            return {
                'status_code': response.status,
                'endpoint_accessible': True,
                'returns_error_for_invalid_creds': response.status == 401
            }
    
    async def test_api_rate_limiting(self) -> Dict[str, Any]:
        """Test API rate limiting."""
        url = f"{self.config['gateway_url']}/api/v1/health"
        
        # Make multiple rapid requests
        start_time = time.time()
        responses = []
        
        for i in range(10):
            try:
                async with self.session.get(url) as response:
                    responses.append({
                        'status': response.status,
                        'headers': dict(response.headers)
                    })
            except Exception as e:
                responses.append({
                    'error': str(e)
                })
        
        duration = time.time() - start_time
        
        # Check for rate limiting headers
        rate_limit_headers = []
        for resp in responses:
            if isinstance(resp, dict) and 'headers' in resp:
                headers = resp['headers']
                for header in ['X-RateLimit-Limit', 'X-RateLimit-Remaining', 'X-RateLimit-Reset']:
                    if header.lower() in [h.lower() for h in headers.keys()]:
                        rate_limit_headers.append(header)
        
        return {
            'total_requests': len(responses),
            'successful_requests': sum(1 for r in responses if r.get('status') == 200),
            'duration_seconds': duration,
            'rate_limit_headers_present': len(rate_limit_headers) > 0,
            'avg_response_time_ms': (duration / len(responses)) * 1000
        }
    
    async def test_websocket_connection(self) -> Dict[str, Any]:
        """Test WebSocket connection."""
        if self.config['gateway_url'].startswith('https'):
            ws_url = self.config['gateway_url'].replace('https', 'wss') + '/ws'
        else:
            ws_url = self.config['gateway_url'].replace('http', 'ws') + '/ws'
        
        try:
            async with self.session.ws_connect(ws_url) as ws:
                # Send a test message
                test_message = {'type': 'ping', 'timestamp': time.time()}
                await ws.send_str(json.dumps(test_message))
                
                # Wait for response with timeout
                try:
                    async with asyncio.timeout(5):
                        msg = await ws.receive()
                        
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            response_data = json.loads(msg.data)
                            return {
                                'connection_successful': True,
                                'message_exchange': True,
                                'response_data': response_data
                            }
                        else:
                            return {
                                'connection_successful': True,
                                'message_exchange': False,
                                'msg_type': str(msg.type)
                            }
                
                except asyncio.TimeoutError:
                    return {
                        'connection_successful': True,
                        'message_exchange': False,
                        'error': 'Response timeout'
                    }
        
        except Exception as e:
            # WebSocket might not be available in all environments
            return {
                'connection_successful': False,
                'error': str(e),
                'note': 'WebSocket may not be available in this environment'
            }
    
    async def test_dashboard_accessibility(self) -> Dict[str, Any]:
        """Test dashboard accessibility."""
        url = self.config['dashboard_url']
        
        async with self.session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Dashboard not accessible, status {response.status}")
            
            content = await response.text()
            
            # Basic checks for a React app
            has_react_div = 'id="root"' in content or 'id="app"' in content
            has_js_bundle = '.js' in content
            has_css = '.css' in content or '<style' in content
            
            return {
                'status_code': response.status,
                'content_length': len(content),
                'appears_to_be_react_app': has_react_div,
                'has_javascript': has_js_bundle,
                'has_css': has_css,
                'content_type': response.headers.get('content-type', '')
            }
    
    async def test_api_endpoints_basic(self) -> Dict[str, Any]:
        """Test basic API endpoints."""
        base_url = self.config['gateway_url']
        endpoints_to_test = [
            '/api/v1/health',
            '/api/v1/info',
            '/api/v1/version',
            '/api/docs',  # OpenAPI docs
            '/metrics'    # Prometheus metrics
        ]
        
        results = {}
        
        for endpoint in endpoints_to_test:
            url = f"{base_url}{endpoint}"
            
            try:
                async with self.session.get(url) as response:
                    results[endpoint] = {
                        'status_code': response.status,
                        'accessible': 200 <= response.status < 400,
                        'content_type': response.headers.get('content-type', ''),
                        'content_length': response.headers.get('content-length', 0)
                    }
            except Exception as e:
                results[endpoint] = {
                    'accessible': False,
                    'error': str(e)
                }
        
        accessible_count = sum(1 for r in results.values() if r.get('accessible', False))
        
        return {
            'endpoints_tested': len(endpoints_to_test),
            'accessible_endpoints': accessible_count,
            'accessibility_rate': accessible_count / len(endpoints_to_test),
            'endpoint_results': results
        }
    
    async def test_database_connectivity(self) -> Dict[str, Any]:
        """Test database connectivity through API."""
        url = f"{self.config['gateway_url']}/api/v1/health/database"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'database_accessible': True,
                        'status_data': data,
                        'connection_pool_info': data.get('connection_pool', {})
                    }
                else:
                    return {
                        'database_accessible': False,
                        'status_code': response.status
                    }
        
        except Exception as e:
            # Endpoint might not exist, try alternative
            try:
                url = f"{self.config['gateway_url']}/ready"
                async with self.session.get(url) as response:
                    data = await response.json()
                    db_status = data.get('database', {}).get('status')
                    
                    return {
                        'database_accessible': db_status == 'connected',
                        'status_from_ready_endpoint': True,
                        'database_status': db_status
                    }
            
            except Exception:
                return {
                    'database_accessible': False,
                    'error': str(e),
                    'note': 'Could not determine database status'
                }
    
    async def test_ssl_certificate(self) -> Dict[str, Any]:
        """Test SSL certificate validity (for staging/production)."""
        if not self.config['expect_https']:
            return {
                'ssl_test_skipped': True,
                'reason': 'HTTPS not expected in this environment'
            }
        
        url = self.config['gateway_url']
        
        try:
            # Make request and check SSL
            async with self.session.get(url) as response:
                return {
                    'ssl_valid': True,
                    'status_code': response.status,
                    'ssl_info': 'Certificate appears valid (no SSL errors)'
                }
        
        except aiohttp.ClientSSLError as e:
            return {
                'ssl_valid': False,
                'ssl_error': str(e)
            }
        except Exception as e:
            return {
                'ssl_valid': False,
                'error': str(e)
            }
    
    async def test_cors_headers(self) -> Dict[str, Any]:
        """Test CORS headers."""
        url = f"{self.config['gateway_url']}/api/v1/health"
        
        # Make an OPTIONS request
        async with self.session.options(url) as response:
            headers = response.headers
            
            cors_headers = {
                'Access-Control-Allow-Origin': headers.get('Access-Control-Allow-Origin'),
                'Access-Control-Allow-Methods': headers.get('Access-Control-Allow-Methods'),
                'Access-Control-Allow-Headers': headers.get('Access-Control-Allow-Headers'),
                'Access-Control-Max-Age': headers.get('Access-Control-Max-Age')
            }
            
            has_cors = any(value is not None for value in cors_headers.values())
            
            return {
                'cors_enabled': has_cors,
                'cors_headers': cors_headers,
                'options_status_code': response.status
            }
    
    async def run_all_tests(self) -> SmokeTestSuite:
        """Run all smoke tests."""
        logger.info(f"Starting smoke tests for {self.environment} environment")
        logger.info(f"Gateway URL: {self.config['gateway_url']}")
        logger.info(f"Dashboard URL: {self.config['dashboard_url']}")
        
        # Define test suite
        tests = [
            ("Gateway Health Check", self.test_gateway_health),
            ("Gateway Readiness Check", self.test_gateway_readiness),
            ("Authentication Endpoint", self.test_authentication_endpoint),
            ("API Rate Limiting", self.test_api_rate_limiting),
            ("WebSocket Connection", self.test_websocket_connection),
            ("Dashboard Accessibility", self.test_dashboard_accessibility),
            ("API Endpoints Basic", self.test_api_endpoints_basic),
            ("Database Connectivity", self.test_database_connectivity),
            ("SSL Certificate", self.test_ssl_certificate),
            ("CORS Headers", self.test_cors_headers)
        ]
        
        # Run all tests
        for test_name, test_func in tests:
            await self.run_test(test_name, test_func)
        
        # Calculate summary statistics
        total_duration = (time.time() - self.start_time) * 1000
        
        passed = sum(1 for r in self.test_results if r.status == 'pass')
        failed = sum(1 for r in self.test_results if r.status == 'fail')
        errors = sum(1 for r in self.test_results if r.status == 'error')
        skipped = sum(1 for r in self.test_results if r.status == 'skip')
        
        # Determine overall status
        if errors > 0:
            overall_status = 'error'
        elif failed > 0:
            overall_status = 'fail'
        elif passed == len(self.test_results):
            overall_status = 'pass'
        else:
            overall_status = 'partial'
        
        suite_result = SmokeTestSuite(
            environment=self.environment,
            timestamp=datetime.utcnow().isoformat(),
            total_tests=len(self.test_results),
            passed_tests=passed,
            failed_tests=failed,
            error_tests=errors,
            skipped_tests=skipped,
            total_duration_ms=total_duration,
            overall_status=overall_status,
            test_results=self.test_results
        )
        
        return suite_result

async def main():
    parser = argparse.ArgumentParser(description='Run smoke tests for MIP Platform')
    parser.add_argument('environment', choices=['development', 'staging', 'production'],
                       help='Environment to test')
    parser.add_argument('--output', '-o', help='Output file for test results (JSON)')
    parser.add_argument('--timeout', '-t', type=int, default=60,
                       help='Overall timeout in seconds (default: 60)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--fail-fast', action='store_true',
                       help='Stop on first test failure')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Run smoke tests with timeout
        async with asyncio.timeout(args.timeout):
            async with MIPSmokeTests(args.environment) as smoke_tests:
                results = await smoke_tests.run_all_tests()
        
        # Output results
        results_dict = asdict(results)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results_dict, f, indent=2)
            logger.info(f"Test results written to {args.output}")
        else:
            print(json.dumps(results_dict, indent=2))
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("SMOKE TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Environment: {results.environment}")
        logger.info(f"Total Tests: {results.total_tests}")
        logger.info(f"Passed: {results.passed_tests}")
        logger.info(f"Failed: {results.failed_tests}")
        logger.info(f"Errors: {results.error_tests}")
        logger.info(f"Duration: {results.total_duration_ms:.1f}ms")
        logger.info(f"Overall Status: {results.overall_status.upper()}")
        
        # Log failed/error tests
        if results.failed_tests > 0 or results.error_tests > 0:
            logger.info("\nFAILED/ERROR TESTS:")
            for test in results.test_results:
                if test.status in ['fail', 'error']:
                    logger.error(f"  {test.test_name}: {test.error_message}")
        
        # Exit with appropriate code
        if results.overall_status in ['fail', 'error']:
            logger.error("Smoke tests failed!")
            sys.exit(1)
        
        logger.info("All smoke tests passed!")
        
    except asyncio.TimeoutError:
        logger.error(f"Smoke tests timed out after {args.timeout} seconds")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running smoke tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

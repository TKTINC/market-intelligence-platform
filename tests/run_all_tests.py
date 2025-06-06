#!/usr/bin/env python3
"""
=============================================================================
COMPREHENSIVE TEST RUNNER FOR MIP PLATFORM
Location: tests/run_all_tests.py

Executes all agent tests, integration tests, and performance benchmarks
=============================================================================
"""

import asyncio
import subprocess
import json
import time
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import argparse
from datetime import datetime
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    test_name: str
    test_type: str
    status: str  # 'passed', 'failed', 'error', 'skipped'
    duration_seconds: float
    output_file: Optional[str]
    error_message: Optional[str]
    metrics: Optional[Dict[str, Any]]

@dataclass
class TestSuiteResult:
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    skipped_tests: int
    total_duration_seconds: float
    test_results: List[TestResult]
    overall_status: str

class ComprehensiveTestRunner:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "test-results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Test configurations
        self.test_configurations = {
            'agent_tests': {
                'finbert_benchmark': {
                    'script': 'src/agents/finbert/tests/benchmark_finbert.py',
                    'args': ['--iterations', '50', '--output', 'test-results/finbert-benchmark.json'],
                    'timeout': 300,
                    'required': True
                },
                'llama_reasoning': {
                    'script': 'src/agents/llama/tests/reasoning_quality_test.py',
                    'args': ['--output', 'test-results/llama-reasoning.json'],
                    'timeout': 600,
                    'required': True
                },
                'gpt4_rate_limits': {
                    'script': 'src/agents/gpt4/tests/rate_limit_test.py',
                    'args': ['--output', 'test-results/gpt4-rate-limits.json'],
                    'timeout': 300,
                    'required': True
                },
                'tft_forecast_accuracy': {
                    'script': 'src/agents/tft/tests/forecast_accuracy_test.py',
                    'args': ['--output', 'test-results/tft-forecast-accuracy.json'],
                    'timeout': 400,
                    'required': True
                }
            },
            'integration_tests': {
                'full_workflow': {
                    'script': 'tests/integration/test_full_workflow.py',
                    'args': ['--workflow', 'all', '--output', 'test-results/integration-workflow.json'],
                    'timeout': 600,
                    'required': True
                },
                'realtime_flow': {
                    'script': 'tests/integration/test_realtime_flow.py',
                    'args': ['--duration', '30', '--output', 'test-results/realtime-flow.json'],
                    'timeout': 120,
                    'required': True
                }
            },
            'performance_tests': {
                'locust_load_test': {
                    'script': 'locust',
                    'args': [
                        '-f', 'tests/performance/locustfile.py',
                        '--host', 'http://localhost:8000',
                        '--users', '20',
                        '--spawn-rate', '4',
                        '--run-time', '2m',
                        '--headless',
                        '--html', 'test-results/locust-report.html'
                    ],
                    'timeout': 180,
                    'required': False
                }
            },
            'cicd_scripts': {
                'security_score': {
                    'script': '.github/scripts/calculate-security-score.py',
                    'args': [
                        'test-data/bandit-report.json',
                        'test-data/safety-report.json', 
                        'test-data/semgrep-report.json',
                        '--output', 'test-results/security-score.json'
                    ],
                    'timeout': 60,
                    'required': False,
                    'setup_required': True
                },
                'smoke_tests': {
                    'script': '.github/scripts/smoke-tests.py',
                    'args': ['development', '--output', 'test-results/smoke-tests.json'],
                    'timeout': 120,
                    'required': False
                },
                'performance_validation': {
                    'script': '.github/scripts/validate-performance.py',
                    'args': ['development', '--duration', '30', '--output', 'test-results/performance-validation.json'],
                    'timeout': 60,
                    'required': False
                }
            }
        }
        
        # Environment setup
        self.environment_ready = False
        self.mock_services_started = False
    
    async def setup_test_environment(self) -> bool:
        """Setup test environment including mock services and test data."""
        logger.info("Setting up test environment...")
        
        try:
            # Create necessary directories
            (self.base_dir / "test-data").mkdir(exist_ok=True)
            
            # Start mock agents if needed
            if not await self.check_services_running():
                logger.info("Starting mock agents...")
                await self.start_mock_services()
            
            # Setup test data
            await self.setup_test_data()
            
            # Wait for services to be ready
            await asyncio.sleep(5)
            
            self.environment_ready = True
            logger.info("Test environment setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            return False
    
    async def check_services_running(self) -> bool:
        """Check if required services are already running."""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # Check if main gateway is running
                async with session.get('http://localhost:8000/health', timeout=5) as response:
                    if response.status == 200:
                        logger.info("Services already running")
                        return True
        except:
            pass
        
        return False
    
    async def start_mock_services(self) -> None:
        """Start mock services for testing."""
        try:
            # Start mock agents
            mock_script = self.base_dir / '.github/scripts/start-mock-agents.py'
            if mock_script.exists():
                logger.info("Starting mock agents...")
                
                process = await asyncio.create_subprocess_exec(
                    sys.executable, str(mock_script), '--daemon', '--timeout', '600',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Wait a bit for services to start
                await asyncio.sleep(10)
                
                # Check if process is still running
                if process.returncode is None:
                    self.mock_services_started = True
                    logger.info("Mock services started successfully")
                else:
                    stdout, stderr = await process.communicate()
                    logger.warning(f"Mock services may have failed: {stderr.decode()}")
            else:
                logger.warning("Mock agents script not found, tests may fail")
                
        except Exception as e:
            logger.warning(f"Failed to start mock services: {e}")
    
    async def setup_test_data(self) -> None:
        """Setup test data including downloading datasets."""
        try:
            # Download test data
            download_script = self.base_dir / '.github/scripts/download-test-data.py'
            if download_script.exists():
                logger.info("Setting up test data...")
                
                process = await asyncio.create_subprocess_exec(
                    sys.executable, str(download_script), 'all', '--mock-only',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                await process.wait()
                logger.info("Test data setup completed")
            
            # Create mock security scan results for security score test
            await self.create_mock_security_reports()
            
        except Exception as e:
            logger.warning(f"Failed to setup test data: {e}")
    
    async def create_mock_security_reports(self) -> None:
        """Create mock security reports for testing."""
        test_data_dir = self.base_dir / "test-data"
        
        # Mock Bandit report
        bandit_report = {
            "results": [
                {
                    "issue_severity": "MEDIUM",
                    "issue_confidence": "HIGH",
                    "filename": "src/example.py",
                    "test_name": "hardcoded_password_string"
                }
            ],
            "metrics": {"_totals": {"loc": 1000}}
        }
        
        # Mock Safety report
        safety_report = {
            "vulnerabilities": [
                {
                    "package_name": "requests",
                    "severity": "low",
                    "advisory": "CVE-2023-32681"
                }
            ]
        }
        
        # Mock Semgrep report
        semgrep_report = {
            "results": [
                {
                    "check_id": "python.lang.security.audit.dangerous-subprocess-use",
                    "extra": {"severity": "WARNING"}
                }
            ]
        }
        
        # Write mock reports
        with open(test_data_dir / "bandit-report.json", 'w') as f:
            json.dump(bandit_report, f)
        
        with open(test_data_dir / "safety-report.json", 'w') as f:
            json.dump(safety_report, f)
        
        with open(test_data_dir / "semgrep-report.json", 'w') as f:
            json.dump(semgrep_report, f)
    
    async def run_single_test(self, test_name: str, test_config: Dict[str, Any]) -> TestResult:
        """Run a single test and return results."""
        logger.info(f"Running test: {test_name}")
        
        start_time = time.time()
        script_path = self.base_dir / test_config['script']
        
        # Check if script exists
        if not script_path.exists() and test_config['script'] != 'locust':
            return TestResult(
                test_name=test_name,
                test_type="unknown",
                status="skipped",
                duration_seconds=0,
                output_file=None,
                error_message=f"Script not found: {script_path}",
                metrics=None
            )
        
        try:
            # Prepare command
            if test_config['script'] == 'locust':
                cmd = ['locust'] + test_config['args']
            else:
                cmd = [sys.executable, str(script_path)] + test_config['args']
            
            # Run test
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.base_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=test_config.get('timeout', 300)
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                
                return TestResult(
                    test_name=test_name,
                    test_type="timeout",
                    status="error",
                    duration_seconds=time.time() - start_time,
                    output_file=None,
                    error_message=f"Test timed out after {test_config.get('timeout', 300)} seconds",
                    metrics=None
                )
            
            duration = time.time() - start_time
            
            # Determine test result based on return code
            if process.returncode == 0:
                status = "passed"
                error_message = None
            else:
                status = "failed"
                error_message = stderr.decode() if stderr else "Test failed with no error message"
            
            # Try to extract output file and metrics
            output_file = None
            metrics = None
            
            for arg in test_config['args']:
                if arg.endswith('.json') and 'test-results' in arg:
                    output_file = arg
                    break
            
            # Try to load metrics from output file
            if output_file and Path(output_file).exists():
                try:
                    with open(output_file, 'r') as f:
                        metrics = json.load(f)
                except:
                    pass
            
            return TestResult(
                test_name=test_name,
                test_type=test_config.get('type', 'unknown'),
                status=status,
                duration_seconds=duration,
                output_file=output_file,
                error_message=error_message,
                metrics=metrics
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                test_type="error",
                status="error",
                duration_seconds=time.time() - start_time,
                output_file=None,
                error_message=str(e),
                metrics=None
            )
    
    async def run_test_suite(self, suite_name: str, tests: Dict[str, Dict[str, Any]], 
                           parallel: bool = False) -> TestSuiteResult:
        """Run a test suite and return aggregated results."""
        logger.info(f"Running test suite: {suite_name}")
        
        start_time = time.time()
        test_results = []
        
        if parallel and len(tests) > 1:
            # Run tests in parallel
            tasks = []
            for test_name, test_config in tests.items():
                if test_config.get('required', True) or not self.args.required_only:
                    task = self.run_single_test(test_name, test_config)
                    tasks.append(task)
            
            if tasks:
                test_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Convert exceptions to error results
                for i, result in enumerate(test_results):
                    if isinstance(result, Exception):
                        test_name = list(tests.keys())[i]
                        test_results[i] = TestResult(
                            test_name=test_name,
                            test_type="exception",
                            status="error",
                            duration_seconds=0,
                            output_file=None,
                            error_message=str(result),
                            metrics=None
                        )
        else:
            # Run tests sequentially
            for test_name, test_config in tests.items():
                if test_config.get('required', True) or not self.args.required_only:
                    result = await self.run_single_test(test_name, test_config)
                    test_results.append(result)
        
        # Calculate suite statistics
        total_duration = time.time() - start_time
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.status == "passed")
        failed_tests = sum(1 for r in test_results if r.status == "failed")
        error_tests = sum(1 for r in test_results if r.status == "error")
        skipped_tests = sum(1 for r in test_results if r.status == "skipped")
        
        # Determine overall status
        if error_tests > 0:
            overall_status = "error"
        elif failed_tests > 0:
            overall_status = "failed"
        elif passed_tests == total_tests:
            overall_status = "passed"
        else:
            overall_status = "partial"
        
        return TestSuiteResult(
            suite_name=suite_name,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            skipped_tests=skipped_tests,
            total_duration_seconds=total_duration,
            test_results=test_results,
            overall_status=overall_status
        )
    
    async def run_all_tests(self) -> Dict[str, TestSuiteResult]:
        """Run all test suites."""
        logger.info("Starting comprehensive test execution")
        
        # Setup environment
        if not await self.setup_test_environment():
            logger.error("Failed to setup test environment")
            return {}
        
        suite_results = {}
        
        # Run test suites in order
        test_order = ['agent_tests', 'integration_tests', 'performance_tests', 'cicd_scripts']
        
        for suite_name in test_order:
            if suite_name in self.test_configurations:
                if self.args.suite and suite_name not in self.args.suite:
                    logger.info(f"Skipping suite: {suite_name}")
                    continue
                
                tests = self.test_configurations[suite_name]
                
                # Run tests in parallel for agent tests, sequentially for others
                parallel = suite_name == 'agent_tests'
                
                suite_result = await self.run_test_suite(suite_name, tests, parallel)
                suite_results[suite_name] = suite_result
                
                # Log suite summary
                logger.info(f"Suite {suite_name} completed: {suite_result.passed_tests}/{suite_result.total_tests} passed")
                
                # Stop on first failure if requested
                if self.args.fail_fast and suite_result.overall_status in ['failed', 'error']:
                    logger.error(f"Stopping execution due to failure in {suite_name}")
                    break
        
        return suite_results
    
    def generate_comprehensive_report(self, suite_results: Dict[str, TestSuiteResult]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        
        # Calculate overall statistics
        total_tests = sum(suite.total_tests for suite in suite_results.values())
        total_passed = sum(suite.passed_tests for suite in suite_results.values())
        total_failed = sum(suite.failed_tests for suite in suite_results.values())
        total_errors = sum(suite.error_tests for suite in suite_results.values())
        total_skipped = sum(suite.skipped_tests for suite in suite_results.values())
        total_duration = sum(suite.total_duration_seconds for suite in suite_results.values())
        
        # Calculate success rate
        success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        # Determine overall status
        if total_errors > 0:
            overall_status = "error"
        elif total_failed > 0:
            overall_status = "failed"
        elif total_passed == total_tests:
            overall_status = "passed"
        else:
            overall_status = "partial"
        
        # Generate recommendations
        recommendations = []
        
        if success_rate < 0.9:
            recommendations.append(f"Test success rate ({success_rate:.1%}) is below target. Review and fix failing tests.")
        
        for suite_name, suite in suite_results.items():
            if suite.failed_tests > 0:
                failed_test_names = [test.test_name for test in suite.test_results if test.status == "failed"]
                recommendations.append(f"Failed tests in {suite_name}: {', '.join(failed_test_names)}")
            
            if suite.error_tests > 0:
                error_test_names = [test.test_name for test in suite.test_results if test.status == "error"]
                recommendations.append(f"Error tests in {suite_name}: {', '.join(error_test_names)}")
        
        if total_duration > 1800:  # 30 minutes
            recommendations.append("Total test execution time is high. Consider optimizing slow tests or running them in parallel.")
        
        if not recommendations:
            recommendations.append("All tests completed successfully within acceptable time limits.")
        
        return {
            'execution_summary': {
                'timestamp': datetime.utcnow().isoformat(),
                'total_duration_seconds': total_duration,
                'overall_status': overall_status,
                'success_rate': success_rate
            },
            'test_statistics': {
                'total_tests': total_tests,
                'passed_tests': total_passed,
                'failed_tests': total_failed,
                'error_tests': total_errors,
                'skipped_tests': total_skipped
            },
            'suite_results': {name: asdict(suite) for name, suite in suite_results.items()},
            'recommendations': recommendations,
            'test_artifacts': {
                'results_directory': str(self.results_dir),
                'output_files': [
                    str(file) for file in self.results_dir.glob('*.json')
                    if file.exists()
                ]
            }
        }
    
    def cleanup_test_environment(self):
        """Cleanup test environment and stop services."""
        logger.info("Cleaning up test environment...")
        
        if self.mock_services_started:
            # Try to stop mock services gracefully
            try:
                # This would normally send a shutdown signal to mock services
                logger.info("Stopping mock services...")
            except Exception as e:
                logger.warning(f"Failed to stop mock services: {e}")

async def main():
    parser = argparse.ArgumentParser(description='Comprehensive MIP Platform Test Runner')
    parser.add_argument('--suite', '-s', nargs='+',
                       choices=['agent_tests', 'integration_tests', 'performance_tests', 'cicd_scripts'],
                       help='Specific test suites to run (default: all)')
    parser.add_argument('--required-only', action='store_true',
                       help='Run only required tests')
    parser.add_argument('--fail-fast', action='store_true',
                       help='Stop on first test suite failure')
    parser.add_argument('--output', '-o', default='test-results/comprehensive-report.json',
                       help='Output file for comprehensive report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--base-dir', default='.',
                       help='Base directory for test execution')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize test runner
        runner = ComprehensiveTestRunner(args.base_dir)
        runner.args = args  # Store args for access in methods
        
        # Run all tests
        suite_results = await runner.run_all_tests()
        
        if not suite_results:
            logger.error("No tests were executed")
            sys.exit(1)
        
        # Generate comprehensive report
        report = runner.generate_comprehensive_report(suite_results)
        
        # Save report
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive report saved to {output_path}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE TEST EXECUTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Overall Status: {report['execution_summary']['overall_status'].upper()}")
        logger.info(f"Total Tests: {report['test_statistics']['total_tests']}")
        logger.info(f"Passed: {report['test_statistics']['passed_tests']}")
        logger.info(f"Failed: {report['test_statistics']['failed_tests']}")
        logger.info(f"Errors: {report['test_statistics']['error_tests']}")
        logger.info(f"Skipped: {report['test_statistics']['skipped_tests']}")
        logger.info(f"Success Rate: {report['execution_summary']['success_rate']:.1%}")
        logger.info(f"Total Duration: {report['execution_summary']['total_duration_seconds']:.1f} seconds")
        
        # Print suite breakdown
        logger.info("\nSuite Breakdown:")
        for suite_name, suite_data in report['suite_results'].items():
            status = suite_data['overall_status'].upper()
            logger.info(f"  {suite_name}: {status} ({suite_data['passed_tests']}/{suite_data['total_tests']} passed)")
        
        # Print recommendations
        if report['recommendations']:
            logger.info("\nRecommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                logger.info(f"  {i}. {rec}")
        
        # Cleanup
        runner.cleanup_test_environment()
        
        # Exit with appropriate code
        overall_status = report['execution_summary']['overall_status']
        sys.exit(0 if overall_status == 'passed' else 1)
        
    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

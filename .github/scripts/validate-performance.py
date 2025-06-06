#!/usr/bin/env python3
"""
=============================================================================
PERFORMANCE VALIDATION SCRIPT
Validates performance SLAs and benchmarks for MIP Platform
=============================================================================
"""

import asyncio
import aiohttp
import json
import time
import statistics
import sys
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PerformanceSLA:
    max_response_time_ms: float
    min_throughput_rps: float
    max_error_rate_percent: float
    min_availability_percent: float
    max_memory_usage_mb: Optional[float] = None
    max_cpu_usage_percent: Optional[float] = None

@dataclass
class PerformanceMetrics:
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    throughput_rps: float
    error_rate_percent: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    test_duration_seconds: float

@dataclass
class AgentPerformanceResult:
    agent_name: str
    sla: PerformanceSLA
    metrics: PerformanceMetrics
    sla_compliance: Dict[str, bool]
    overall_compliance: bool
    recommendations: List[str]

@dataclass
class SystemPerformanceResult:
    environment: str
    timestamp: str
    overall_compliance: bool
    agent_results: Dict[str, AgentPerformanceResult]
    system_metrics: Dict[str, Any]
    recommendations: List[str]

class PerformanceValidator:
    def __init__(self, environment: str):
        self.environment = environment
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Environment-specific configurations
        self.configs = {
            'development': {
                'gateway_url': 'http://localhost:8000',
                'concurrent_users': 5,
                'test_duration_seconds': 30,
                'ramp_up_seconds': 5
            },
            'staging': {
                'gateway_url': os.getenv('STAGING_GATEWAY_URL', 'https://mip-staging.example.com'),
                'concurrent_users': 20,
                'test_duration_seconds': 120,
                'ramp_up_seconds': 10
            },
            'production': {
                'gateway_url': os.getenv('PROD_GATEWAY_URL', 'https://mip.example.com'),
                'concurrent_users': 50,
                'test_duration_seconds': 300,
                'ramp_up_seconds': 30
            }
        }
        
        if environment not in self.configs:
            raise ValueError(f"Unknown environment: {environment}")
        
        self.config = self.configs[environment]
        
        # Performance SLAs for each agent
        self.agent_slas = {
            'finbert': PerformanceSLA(
                max_response_time_ms=2000,
                min_throughput_rps=10,
                max_error_rate_percent=1.0,
                min_availability_percent=99.5,
                max_memory_usage_mb=4000,
                max_cpu_usage_percent=80
            ),
            'llama': PerformanceSLA(
                max_response_time_ms=5000,
                min_throughput_rps=5,
                max_error_rate_percent=2.0,
                min_availability_percent=99.0,
                max_memory_usage_mb=16000,
                max_cpu_usage_percent=90
            ),
            'gpt4': PerformanceSLA(
                max_response_time_ms=3000,
                min_throughput_rps=15,
                max_error_rate_percent=0.5,
                min_availability_percent=99.9,
                max_memory_usage_mb=2000,
                max_cpu_usage_percent=70
            ),
            'tft': PerformanceSLA(
                max_response_time_ms=10000,
                min_throughput_rps=3,
                max_error_rate_percent=1.5,
                min_availability_percent=99.0,
                max_memory_usage_mb=8000,
                max_cpu_usage_percent=85
            ),
            'orchestrator': PerformanceSLA(
                max_response_time_ms=8000,
                min_throughput_rps=8,
                max_error_rate_percent=1.0,
                min_availability_percent=99.5,
                max_memory_usage_mb=4000,
                max_cpu_usage_percent=75
            )
        }
        
        # Test payloads for each agent
        self.test_payloads = {
            'finbert': {
                'text': 'The company reported strong quarterly earnings with revenue growth of 15% year-over-year, exceeding analyst expectations.'
            },
            'llama': {
                'prompt': 'Analyze the financial implications of a merger between two tech companies with market caps of $100B and $50B respectively.',
                'max_tokens': 500
            },
            'gpt4': {
                'messages': [
                    {'role': 'user', 'content': 'Provide a brief analysis of current market volatility and its impact on tech stocks.'}
                ],
                'max_tokens': 300
            },
            'tft': {
                'symbol': 'AAPL',
                'horizon_days': 5,
                'features': ['price', 'volume', 'volatility']
            },
            'orchestrator': {
                'task': 'market_analysis',
                'symbol': 'MSFT',
                'analysis_type': 'comprehensive'
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=30)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'MIP-PerformanceValidator/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def make_request(self, agent_name: str, endpoint: str) -> Tuple[float, bool, Optional[str]]:
        """Make a single request to an agent and measure performance."""
        url = f"{self.config['gateway_url']}/api/v1/{agent_name}/{endpoint}"
        payload = self.test_payloads.get(agent_name, {})
        
        start_time = time.time()
        
        try:
            async with self.session.post(url, json=payload) as response:
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Read response to ensure complete request
                await response.read()
                
                success = 200 <= response.status < 300
                error_msg = None if success else f"HTTP {response.status}"
                
                return response_time, success, error_msg
        
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            return response_time, False, "Request timeout"
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return response_time, False, str(e)
    
    async def load_test_agent(self, agent_name: str, endpoint: str = "process") -> PerformanceMetrics:
        """Run load test against a specific agent."""
        logger.info(f"Load testing {agent_name} agent...")
        
        concurrent_users = self.config['concurrent_users']
        test_duration = self.config['test_duration_seconds']
        ramp_up = self.config['ramp_up_seconds']
        
        # Results storage
        response_times = []
        successes = 0
        failures = 0
        errors = []
        
        # Test execution
        test_start_time = time.time()
        end_time = test_start_time + test_duration
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def make_single_request():
            async with semaphore:
                response_time, success, error_msg = await self.make_request(agent_name, endpoint)
                
                response_times.append(response_time)
                
                if success:
                    nonlocal successes
                    successes += 1
                else:
                    nonlocal failures
                    failures += 1
                    if error_msg:
                        errors.append(error_msg)
        
        # Generate load with gradual ramp-up
        tasks = []
        
        while time.time() < end_time:
            # Calculate current load based on ramp-up
            elapsed = time.time() - test_start_time
            if elapsed < ramp_up:
                current_concurrency = int((elapsed / ramp_up) * concurrent_users)
            else:
                current_concurrency = concurrent_users
            
            # Launch requests up to current concurrency
            while len([t for t in tasks if not t.done()]) < current_concurrency and time.time() < end_time:
                task = asyncio.create_task(make_single_request())
                tasks.append(task)
                
                # Small delay to spread requests
                await asyncio.sleep(0.1)
            
            # Clean up completed tasks
            tasks = [t for t in tasks if not t.done()]
            
            await asyncio.sleep(0.5)  # Check interval
        
        # Wait for remaining tasks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        actual_duration = time.time() - test_start_time
        
        # Calculate metrics
        if not response_times:
            raise Exception(f"No successful requests for {agent_name}")
        
        total_requests = len(response_times)
        throughput = total_requests / actual_duration
        error_rate = (failures / total_requests) * 100 if total_requests > 0 else 0
        
        metrics = PerformanceMetrics(
            avg_response_time_ms=statistics.mean(response_times),
            p95_response_time_ms=statistics.quantiles(response_times, n=20)[18],  # 95th percentile
            p99_response_time_ms=statistics.quantiles(response_times, n=100)[98],  # 99th percentile
            min_response_time_ms=min(response_times),
            max_response_time_ms=max(response_times),
            throughput_rps=throughput,
            error_rate_percent=error_rate,
            total_requests=total_requests,
            successful_requests=successes,
            failed_requests=failures,
            test_duration_seconds=actual_duration
        )
        
        logger.info(f"{agent_name} test completed: {total_requests} requests in {actual_duration:.1f}s")
        logger.info(f"  Throughput: {throughput:.1f} RPS")
        logger.info(f"  Avg Response Time: {metrics.avg_response_time_ms:.1f}ms")
        logger.info(f"  Error Rate: {error_rate:.1f}%")
        
        return metrics
    
    def check_sla_compliance(self, agent_name: str, metrics: PerformanceMetrics) -> Tuple[Dict[str, bool], bool]:
        """Check SLA compliance for an agent."""
        sla = self.agent_slas[agent_name]
        
        compliance = {
            'response_time': metrics.p95_response_time_ms <= sla.max_response_time_ms,
            'throughput': metrics.throughput_rps >= sla.min_throughput_rps,
            'error_rate': metrics.error_rate_percent <= sla.max_error_rate_percent,
            'availability': ((metrics.successful_requests / metrics.total_requests) * 100) >= sla.min_availability_percent
        }
        
        overall_compliance = all(compliance.values())
        
        return compliance, overall_compliance
    
    def generate_recommendations(self, agent_name: str, metrics: PerformanceMetrics, 
                               compliance: Dict[str, bool]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        sla = self.agent_slas[agent_name]
        
        if not compliance['response_time']:
            recommendations.append(
                f"Response time SLA violation: P95 {metrics.p95_response_time_ms:.1f}ms > {sla.max_response_time_ms}ms. "
                f"Consider optimizing {agent_name} model inference or scaling up resources."
            )
        
        if not compliance['throughput']:
            recommendations.append(
                f"Throughput SLA violation: {metrics.throughput_rps:.1f} RPS < {sla.min_throughput_rps} RPS. "
                f"Consider horizontal scaling or optimizing {agent_name} agent."
            )
        
        if not compliance['error_rate']:
            recommendations.append(
                f"Error rate SLA violation: {metrics.error_rate_percent:.1f}% > {sla.max_error_rate_percent}%. "
                f"Investigate and fix errors in {agent_name} agent."
            )
        
        if not compliance['availability']:
            availability = (metrics.successful_requests / metrics.total_requests) * 100
            recommendations.append(
                f"Availability SLA violation: {availability:.1f}% < {sla.min_availability_percent}%. "
                f"Improve {agent_name} agent reliability and error handling."
            )
        
        # Performance optimization suggestions
        if metrics.p99_response_time_ms > metrics.p95_response_time_ms * 2:
            recommendations.append(
                f"High response time variance in {agent_name}. P99: {metrics.p99_response_time_ms:.1f}ms, "
                f"P95: {metrics.p95_response_time_ms:.1f}ms. Consider request batching or caching."
            )
        
        if metrics.error_rate_percent > 0 and compliance['error_rate']:
            recommendations.append(
                f"Consider implementing circuit breaker pattern for {agent_name} to handle failures gracefully."
            )
        
        return recommendations
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics."""
        try:
            # Try to get Prometheus metrics
            metrics_url = f"{self.config['gateway_url']}/metrics"
            
            async with self.session.get(metrics_url) as response:
                if response.status == 200:
                    metrics_text = await response.text()
                    
                    # Parse some key metrics (simplified)
                    system_metrics = {
                        'metrics_endpoint_available': True,
                        'metrics_size_bytes': len(metrics_text.encode()),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    # Look for specific metrics in the response
                    if 'agent_requests_total' in metrics_text:
                        system_metrics['agent_metrics_available'] = True
                    
                    if 'process_resident_memory_bytes' in metrics_text:
                        system_metrics['memory_metrics_available'] = True
                    
                    return system_metrics
        
        except Exception as e:
            logger.warning(f"Could not fetch system metrics: {e}")
        
        # Fallback system metrics
        return {
            'metrics_endpoint_available': False,
            'timestamp': datetime.utcnow().isoformat(),
            'environment': self.environment,
            'test_config': self.config
        }
    
    async def validate_agent_performance(self, agent_name: str) -> AgentPerformanceResult:
        """Validate performance for a single agent."""
        logger.info(f"Validating performance for {agent_name} agent...")
        
        try:
            # Run load test
            metrics = await self.load_test_agent(agent_name)
            
            # Check SLA compliance
            compliance, overall_compliance = self.check_sla_compliance(agent_name, metrics)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(agent_name, metrics, compliance)
            
            result = AgentPerformanceResult(
                agent_name=agent_name,
                sla=self.agent_slas[agent_name],
                metrics=metrics,
                sla_compliance=compliance,
                overall_compliance=overall_compliance,
                recommendations=recommendations
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error validating {agent_name} performance: {e}")
            
            # Return a failed result
            dummy_metrics = PerformanceMetrics(
                avg_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                min_response_time_ms=0,
                max_response_time_ms=0,
                throughput_rps=0,
                error_rate_percent=100,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                test_duration_seconds=0
            )
            
            return AgentPerformanceResult(
                agent_name=agent_name,
                sla=self.agent_slas[agent_name],
                metrics=dummy_metrics,
                sla_compliance={k: False for k in ['response_time', 'throughput', 'error_rate', 'availability']},
                overall_compliance=False,
                recommendations=[f"Failed to test {agent_name}: {str(e)}"]
            )
    
    async def validate_all_agents(self, agents: Optional[List[str]] = None) -> SystemPerformanceResult:
        """Validate performance for all specified agents."""
        if agents is None:
            agents = list(self.agent_slas.keys())
        
        logger.info(f"Starting performance validation for {len(agents)} agents in {self.environment}")
        
        # Validate each agent
        agent_results = {}
        
        for agent_name in agents:
            if agent_name in self.agent_slas:
                result = await self.validate_agent_performance(agent_name)
                agent_results[agent_name] = result
            else:
                logger.warning(f"No SLA defined for agent: {agent_name}")
        
        # Get system metrics
        system_metrics = await self.get_system_metrics()
        
        # Determine overall compliance
        overall_compliance = all(result.overall_compliance for result in agent_results.values())
        
        # Generate system-wide recommendations
        system_recommendations = []
        
        non_compliant_agents = [name for name, result in agent_results.items() 
                               if not result.overall_compliance]
        
        if non_compliant_agents:
            system_recommendations.append(
                f"SLA violations detected in {len(non_compliant_agents)} agents: {', '.join(non_compliant_agents)}. "
                f"Consider reviewing resource allocation and scaling strategies."
            )
        
        # Performance patterns
        high_latency_agents = [name for name, result in agent_results.items() 
                              if result.metrics.p95_response_time_ms > 3000]
        
        if high_latency_agents:
            system_recommendations.append(
                f"High latency detected in: {', '.join(high_latency_agents)}. "
                f"Consider implementing response caching or request optimization."
            )
        
        low_throughput_agents = [name for name, result in agent_results.items() 
                                if result.metrics.throughput_rps < 5]
        
        if low_throughput_agents:
            system_recommendations.append(
                f"Low throughput detected in: {', '.join(low_throughput_agents)}. "
                f"Consider horizontal scaling or load balancing improvements."
            )
        
        if not system_recommendations and overall_compliance:
            system_recommendations.append("All agents meeting performance SLAs. System performing within acceptable parameters.")
        
        result = SystemPerformanceResult(
            environment=self.environment,
            timestamp=datetime.utcnow().isoformat(),
            overall_compliance=overall_compliance,
            agent_results=agent_results,
            system_metrics=system_metrics,
            recommendations=system_recommendations
        )
        
        return result

async def main():
    parser = argparse.ArgumentParser(description='Validate MIP Platform performance')
    parser.add_argument('environment', choices=['development', 'staging', 'production'],
                       help='Environment to validate')
    parser.add_argument('--agents', '-a', nargs='+',
                       choices=['finbert', 'llama', 'gpt4', 'tft', 'orchestrator'],
                       help='Specific agents to validate (default: all)')
    parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    parser.add_argument('--duration', '-d', type=int,
                       help='Override test duration in seconds')
    parser.add_argument('--concurrency', '-c', type=int,
                       help='Override concurrent users count')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--fail-fast', action='store_true',
                       help='Exit on first SLA violation')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        async with PerformanceValidator(args.environment) as validator:
            # Apply overrides
            if args.duration:
                validator.config['test_duration_seconds'] = args.duration
            if args.concurrency:
                validator.config['concurrent_users'] = args.concurrency
            
            # Run validation
            results = await validator.validate_all_agents(args.agents)
        
        # Output results
        results_dict = asdict(results)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results_dict, f, indent=2)
            logger.info(f"Performance results written to {args.output}")
        else:
            print(json.dumps(results_dict, indent=2))
        
        # Log summary
        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE VALIDATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Environment: {results.environment}")
        logger.info(f"Overall Compliance: {'✓ PASS' if results.overall_compliance else '✗ FAIL'}")
        logger.info(f"Agents Tested: {len(results.agent_results)}")
        
        for agent_name, agent_result in results.agent_results.items():
            status = "✓ PASS" if agent_result.overall_compliance else "✗ FAIL"
            logger.info(f"  {agent_name:12} {status} - "
                       f"{agent_result.metrics.throughput_rps:.1f} RPS, "
                       f"P95: {agent_result.metrics.p95_response_time_ms:.1f}ms, "
                       f"Error: {agent_result.metrics.error_rate_percent:.1f}%")
        
        # Log recommendations
        if results.recommendations:
            logger.info("\nSYSTEM RECOMMENDATIONS:")
            for i, rec in enumerate(results.recommendations, 1):
                logger.info(f"  {i}. {rec}")
        
        # Log agent-specific recommendations
        for agent_name, agent_result in results.agent_results.items():
            if agent_result.recommendations:
                logger.info(f"\n{agent_name.upper()} RECOMMENDATIONS:")
                for i, rec in enumerate(agent_result.recommendations, 1):
                    logger.info(f"  {i}. {rec}")
        
        # Exit with appropriate code
        if not results.overall_compliance:
            logger.error("Performance validation failed!")
            if args.fail_fast:
                sys.exit(1)
        
        logger.info("Performance validation completed!")
        
    except Exception as e:
        logger.error(f"Error during performance validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

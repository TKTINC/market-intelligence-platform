#!/usr/bin/env python3
"""
=============================================================================
INTEGRATION TEST SUITE - FULL WORKFLOW
Location: tests/integration/test_full_workflow.py
=============================================================================
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse
import sys
import os
from datetime import datetime
import numpy as np
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WorkflowStep:
    name: str
    endpoint: str
    method: str
    payload: Optional[Dict[str, Any]]
    expected_status: int
    timeout: float
    dependencies: List[str]

@dataclass
class WorkflowResult:
    step_name: str
    success: bool
    execution_time_seconds: float
    response_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    status_code: Optional[int]

@dataclass
class IntegrationTestResult:
    workflow_name: str
    total_steps: int
    successful_steps: int
    failed_steps: int
    total_execution_time_seconds: float
    workflow_results: List[WorkflowResult]
    all_steps_successful: bool
    performance_acceptable: bool
    recommendations: List[str]

class IntegrationTestSuite:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.test_data = {}  # Store data between workflow steps
        
        # Workflow definitions
        self.workflows = {
            'end_to_end_market_analysis': self.define_market_analysis_workflow(),
            'portfolio_management_flow': self.define_portfolio_workflow(),
            'real_time_trading_flow': self.define_trading_workflow(),
            'multi_user_concurrent_flow': self.define_concurrent_workflow()
        }
        
        # Performance thresholds
        self.thresholds = {
            'max_total_workflow_time': 60.0,  # seconds
            'max_step_time': 15.0,  # seconds
            'min_success_rate': 0.95,
            'max_error_rate': 0.05
        }
    
    def define_market_analysis_workflow(self) -> List[WorkflowStep]:
        """Define end-to-end market analysis workflow."""
        return [
            WorkflowStep(
                name="user_authentication",
                endpoint="/api/v1/auth/login",
                method="POST",
                payload={
                    "username": "test_analyst",
                    "password": "test_password"
                },
                expected_status=200,
                timeout=5.0,
                dependencies=[]
            ),
            WorkflowStep(
                name="market_data_ingestion",
                endpoint="/api/v1/data/ingest",
                method="POST",
                payload={
                    "symbols": ["AAPL", "MSFT", "GOOGL"],
                    "data_sources": ["price", "volume", "news"],
                    "timeframe": "1d"
                },
                expected_status=202,
                timeout=10.0,
                dependencies=["user_authentication"]
            ),
            WorkflowStep(
                name="sentiment_analysis",
                endpoint="/api/v1/agents/finbert/analyze",
                method="POST",
                payload={
                    "texts": [
                        "Apple reports strong quarterly earnings beating expectations",
                        "Microsoft cloud revenue shows impressive growth trajectory",
                        "Google faces regulatory challenges in multiple markets"
                    ]
                },
                expected_status=200,
                timeout=8.0,
                dependencies=["user_authentication"]
            ),
            WorkflowStep(
                name="market_forecasting",
                endpoint="/api/v1/agents/tft/forecast",
                method="POST",
                payload={
                    "symbols": ["AAPL", "MSFT", "GOOGL"],
                    "horizon_days": 5,
                    "include_confidence": True
                },
                expected_status=200,
                timeout=15.0,
                dependencies=["market_data_ingestion"]
            ),
            WorkflowStep(
                name="reasoning_analysis",
                endpoint="/api/v1/agents/llama/reason",
                method="POST",
                payload={
                    "prompt": "Analyze the market implications of tech stocks showing strong earnings growth while facing regulatory pressure",
                    "context": "technology sector analysis",
                    "max_tokens": 500
                },
                expected_status=200,
                timeout=12.0,
                dependencies=["sentiment_analysis"]
            ),
            WorkflowStep(
                name="comprehensive_analysis",
                endpoint="/api/v1/agents/gpt4/analyze",
                method="POST",
                payload={
                    "query": "Provide investment recommendations for tech portfolio based on recent earnings and market conditions",
                    "include_risk_assessment": True
                },
                expected_status=200,
                timeout=10.0,
                dependencies=["reasoning_analysis", "market_forecasting"]
            ),
            WorkflowStep(
                name="orchestrated_decision",
                endpoint="/api/v1/orchestrator/coordinate",
                method="POST",
                payload={
                    "task": "investment_decision",
                    "symbols": ["AAPL", "MSFT", "GOOGL"],
                    "strategy": "growth_focused",
                    "risk_tolerance": "moderate"
                },
                expected_status=200,
                timeout=20.0,
                dependencies=["comprehensive_analysis"]
            ),
            WorkflowStep(
                name="real_time_updates",
                endpoint="/api/v1/websocket/connect",
                method="GET",
                payload=None,
                expected_status=101,  # WebSocket upgrade
                timeout=5.0,
                dependencies=["orchestrated_decision"]
            )
        ]
    
    def define_portfolio_workflow(self) -> List[WorkflowStep]:
        """Define portfolio management workflow."""
        return [
            WorkflowStep(
                name="user_authentication",
                endpoint="/api/v1/auth/login",
                method="POST",
                payload={"username": "portfolio_manager", "password": "secure_pass"},
                expected_status=200,
                timeout=5.0,
                dependencies=[]
            ),
            WorkflowStep(
                name="create_portfolio",
                endpoint="/api/v1/portfolio/create",
                method="POST",
                payload={
                    "name": "Tech Growth Portfolio",
                    "initial_cash": 100000,
                    "strategy": "growth",
                    "risk_level": "moderate"
                },
                expected_status=201,
                timeout=5.0,
                dependencies=["user_authentication"]
            ),
            WorkflowStep(
                name="add_positions",
                endpoint="/api/v1/portfolio/{portfolio_id}/positions",
                method="POST",
                payload={
                    "trades": [
                        {"symbol": "AAPL", "quantity": 100, "action": "BUY"},
                        {"symbol": "MSFT", "quantity": 150, "action": "BUY"},
                        {"symbol": "GOOGL", "quantity": 50, "action": "BUY"}
                    ]
                },
                expected_status=200,
                timeout=8.0,
                dependencies=["create_portfolio"]
            ),
            WorkflowStep(
                name="risk_assessment",
                endpoint="/api/v1/portfolio/{portfolio_id}/risk",
                method="GET",
                payload=None,
                expected_status=200,
                timeout=10.0,
                dependencies=["add_positions"]
            ),
            WorkflowStep(
                name="performance_analytics",
                endpoint="/api/v1/portfolio/{portfolio_id}/performance",
                method="GET",
                payload=None,
                expected_status=200,
                timeout=8.0,
                dependencies=["add_positions"]
            ),
            WorkflowStep(
                name="rebalancing_suggestions",
                endpoint="/api/v1/portfolio/{portfolio_id}/rebalance",
                method="POST",
                payload={
                    "target_allocation": {
                        "AAPL": 0.35,
                        "MSFT": 0.40,
                        "GOOGL": 0.25
                    }
                },
                expected_status=200,
                timeout=12.0,
                dependencies=["performance_analytics", "risk_assessment"]
            )
        ]
    
    def define_trading_workflow(self) -> List[WorkflowStep]:
        """Define real-time trading workflow."""
        return [
            WorkflowStep(
                name="user_authentication",
                endpoint="/api/v1/auth/login",
                method="POST",
                payload={"username": "trader", "password": "trading_pass"},
                expected_status=200,
                timeout=5.0,
                dependencies=[]
            ),
            WorkflowStep(
                name="market_data_stream",
                endpoint="/api/v1/market/stream",
                method="POST",
                payload={"symbols": ["TSLA", "NVDA"], "stream_type": "real_time"},
                expected_status=200,
                timeout=8.0,
                dependencies=["user_authentication"]
            ),
            WorkflowStep(
                name="generate_signals",
                endpoint="/api/v1/signals/generate",
                method="POST",
                payload={
                    "symbols": ["TSLA", "NVDA"],
                    "strategy": "momentum",
                    "timeframe": "5m"
                },
                expected_status=200,
                timeout=10.0,
                dependencies=["market_data_stream"]
            ),
            WorkflowStep(
                name="validate_signals",
                endpoint="/api/v1/signals/validate",
                method="POST",
                payload={
                    "signal_ids": [],  # Will be populated from previous step
                    "validation_rules": ["risk_check", "position_size", "correlation"]
                },
                expected_status=200,
                timeout=8.0,
                dependencies=["generate_signals"]
            ),
            WorkflowStep(
                name="execute_trades",
                endpoint="/api/v1/trading/execute",
                method="POST",
                payload={
                    "orders": [],  # Will be populated from validated signals
                    "execution_type": "virtual",
                    "risk_limits": {"max_position_size": 0.1}
                },
                expected_status=200,
                timeout=12.0,
                dependencies=["validate_signals"]
            ),
            WorkflowStep(
                name="monitor_positions",
                endpoint="/api/v1/positions/monitor",
                method="GET",
                payload=None,
                expected_status=200,
                timeout=5.0,
                dependencies=["execute_trades"]
            )
        ]
    
    def define_concurrent_workflow(self) -> List[WorkflowStep]:
        """Define workflow for testing concurrent access."""
        return [
            WorkflowStep(
                name="multiple_user_auth",
                endpoint="/api/v1/auth/login",
                method="POST",
                payload={"username": "concurrent_user_{user_id}", "password": "test_pass"},
                expected_status=200,
                timeout=5.0,
                dependencies=[]
            ),
            WorkflowStep(
                name="concurrent_data_requests",
                endpoint="/api/v1/data/fetch",
                method="POST",
                payload={"symbols": ["SPY", "QQQ"], "concurrent": True},
                expected_status=200,
                timeout=10.0,
                dependencies=["multiple_user_auth"]
            ),
            WorkflowStep(
                name="parallel_analysis",
                endpoint="/api/v1/analysis/parallel",
                method="POST",
                payload={"analysis_type": "multi_asset", "parallel_jobs": 5},
                expected_status=200,
                timeout=15.0,
                dependencies=["concurrent_data_requests"]
            )
        ]
    
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=100)
        timeout = aiohttp.ClientTimeout(total=30)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'Content-Type': 'application/json'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def execute_workflow_step(self, step: WorkflowStep) -> WorkflowResult:
        """Execute a single workflow step."""
        logger.info(f"Executing step: {step.name}")
        
        start_time = time.time()
        
        try:
            # Replace placeholders in endpoint with actual values
            endpoint = step.endpoint
            if "{portfolio_id}" in endpoint and "portfolio_id" in self.test_data:
                endpoint = endpoint.replace("{portfolio_id}", str(self.test_data["portfolio_id"]))
            
            url = f"{self.base_url}{endpoint}"
            
            # Prepare payload
            payload = step.payload
            if payload and "concurrent_user_{user_id}" in str(payload):
                # Handle concurrent user testing
                payload = json.loads(json.dumps(payload).replace("{user_id}", "1"))
            
            # Make HTTP request
            if step.method == "GET":
                async with self.session.get(url, timeout=step.timeout) as response:
                    response_data = await response.json() if response.content_type == 'application/json' else await response.text()
                    status_code = response.status
            elif step.method == "POST":
                async with self.session.post(url, json=payload, timeout=step.timeout) as response:
                    response_data = await response.json() if response.content_type == 'application/json' else await response.text()
                    status_code = response.status
            elif step.method == "PUT":
                async with self.session.put(url, json=payload, timeout=step.timeout) as response:
                    response_data = await response.json() if response.content_type == 'application/json' else await response.text()
                    status_code = response.status
            else:
                raise ValueError(f"Unsupported HTTP method: {step.method}")
            
            execution_time = time.time() - start_time
            
            # Check if response meets expectations
            success = status_code == step.expected_status
            
            # Store relevant data for subsequent steps
            if success and isinstance(response_data, dict):
                if step.name == "create_portfolio" and "portfolio_id" in response_data:
                    self.test_data["portfolio_id"] = response_data["portfolio_id"]
                elif step.name == "user_authentication" and "access_token" in response_data:
                    self.test_data["auth_token"] = response_data["access_token"]
                    # Update session headers with auth token
                    self.session.headers.update({"Authorization": f"Bearer {response_data['access_token']}"})
                elif step.name == "generate_signals" and "signals" in response_data:
                    self.test_data["signal_ids"] = [s["id"] for s in response_data["signals"]]
            
            return WorkflowResult(
                step_name=step.name,
                success=success,
                execution_time_seconds=execution_time,
                response_data=response_data if success else None,
                error_message=None if success else f"Expected status {step.expected_status}, got {status_code}",
                status_code=status_code
            )
        
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return WorkflowResult(
                step_name=step.name,
                success=False,
                execution_time_seconds=execution_time,
                response_data=None,
                error_message=f"Request timed out after {step.timeout} seconds",
                status_code=None
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return WorkflowResult(
                step_name=step.name,
                success=False,
                execution_time_seconds=execution_time,
                response_data=None,
                error_message=str(e),
                status_code=None
            )
    
    async def execute_workflow(self, workflow_name: str) -> IntegrationTestResult:
        """Execute a complete workflow."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        logger.info(f"Starting workflow: {workflow_name}")
        
        workflow_steps = self.workflows[workflow_name]
        workflow_results = []
        
        start_time = time.time()
        
        # Execute steps in dependency order
        executed_steps = set()
        
        while len(executed_steps) < len(workflow_steps):
            # Find steps that can be executed (dependencies satisfied)
            ready_steps = []
            for step in workflow_steps:
                if step.name not in executed_steps:
                    if all(dep in executed_steps for dep in step.dependencies):
                        ready_steps.append(step)
            
            if not ready_steps:
                # No more steps can be executed - check for circular dependencies
                remaining_steps = [s.name for s in workflow_steps if s.name not in executed_steps]
                error_msg = f"Cannot execute remaining steps due to unmet dependencies: {remaining_steps}"
                logger.error(error_msg)
                
                # Create failed results for remaining steps
                for step in workflow_steps:
                    if step.name not in executed_steps:
                        workflow_results.append(WorkflowResult(
                            step_name=step.name,
                            success=False,
                            execution_time_seconds=0.0,
                            response_data=None,
                            error_message="Dependency not satisfied",
                            status_code=None
                        ))
                break
            
            # Execute ready steps (can be done in parallel if no data dependencies)
            step_tasks = []
            for step in ready_steps:
                step_tasks.append(self.execute_workflow_step(step))
            
            # Wait for all ready steps to complete
            step_results = await asyncio.gather(*step_tasks, return_exceptions=True)
            
            for i, result in enumerate(step_results):
                if isinstance(result, Exception):
                    # Handle execution exception
                    workflow_results.append(WorkflowResult(
                        step_name=ready_steps[i].name,
                        success=False,
                        execution_time_seconds=0.0,
                        response_data=None,
                        error_message=str(result),
                        status_code=None
                    ))
                else:
                    workflow_results.append(result)
                
                executed_steps.add(ready_steps[i].name)
        
        total_execution_time = time.time() - start_time
        
        # Calculate results
        successful_steps = sum(1 for r in workflow_results if r.success)
        failed_steps = len(workflow_results) - successful_steps
        all_steps_successful = failed_steps == 0
        performance_acceptable = total_execution_time <= self.thresholds['max_total_workflow_time']
        
        # Generate recommendations
        recommendations = self.generate_workflow_recommendations(
            workflow_results, total_execution_time, workflow_name
        )
        
        result = IntegrationTestResult(
            workflow_name=workflow_name,
            total_steps=len(workflow_steps),
            successful_steps=successful_steps,
            failed_steps=failed_steps,
            total_execution_time_seconds=total_execution_time,
            workflow_results=workflow_results,
            all_steps_successful=all_steps_successful,
            performance_acceptable=performance_acceptable,
            recommendations=recommendations
        )
        
        logger.info(f"Workflow {workflow_name} completed: {successful_steps}/{len(workflow_steps)} steps successful")
        
        return result
    
    def generate_workflow_recommendations(self, results: List[WorkflowResult], 
                                        total_time: float, workflow_name: str) -> List[str]:
        """Generate recommendations based on workflow results."""
        recommendations = []
        
        # Failed steps
        failed_steps = [r for r in results if not r.success]
        if failed_steps:
            recommendations.append(
                f"Failed steps in {workflow_name}: {', '.join(r.step_name for r in failed_steps)}. "
                f"Review error messages and fix underlying issues."
            )
        
        # Performance issues
        slow_steps = [r for r in results if r.execution_time_seconds > self.thresholds['max_step_time']]
        if slow_steps:
            recommendations.append(
                f"Slow steps detected: {', '.join(r.step_name for r in slow_steps)}. "
                f"Consider optimization or timeout adjustments."
            )
        
        # Overall performance
        if total_time > self.thresholds['max_total_workflow_time']:
            recommendations.append(
                f"Total workflow time ({total_time:.1f}s) exceeds threshold "
                f"({self.thresholds['max_total_workflow_time']}s). Consider parallel execution improvements."
            )
        
        # Specific workflow recommendations
        if workflow_name == "end_to_end_market_analysis":
            auth_result = next((r for r in results if r.step_name == "user_authentication"), None)
            if auth_result and not auth_result.success:
                recommendations.append("Authentication issues detected. Verify auth service configuration.")
        
        if workflow_name == "portfolio_management_flow":
            portfolio_steps = [r for r in results if "portfolio" in r.step_name and not r.success]
            if portfolio_steps:
                recommendations.append("Portfolio management issues detected. Check database connectivity.")
        
        if not recommendations:
            recommendations.append(f"Workflow {workflow_name} executed successfully within performance thresholds.")
        
        return recommendations
    
    async def test_multi_user_concurrent_access(self, concurrent_users: int = 10) -> Dict[str, Any]:
        """Test concurrent access by multiple users."""
        logger.info(f"Testing concurrent access with {concurrent_users} users")
        
        async def simulate_user_session(user_id: int) -> Dict[str, Any]:
            """Simulate a user session."""
            session_start = time.time()
            
            # Create dedicated session for this user
            async with aiohttp.ClientSession() as user_session:
                actions = ['login', 'fetch_data', 'analyze', 'logout']
                action_results = []
                
                for action in actions:
                    action_start = time.time()
                    
                    try:
                        if action == 'login':
                            async with user_session.post(
                                f"{self.base_url}/api/v1/auth/login",
                                json={"username": f"user_{user_id}", "password": "test_pass"}
                            ) as response:
                                success = response.status == 200
                                if success:
                                    auth_data = await response.json()
                                    user_session.headers.update({
                                        "Authorization": f"Bearer {auth_data.get('access_token', '')}"
                                    })
                        
                        elif action == 'fetch_data':
                            async with user_session.get(f"{self.base_url}/api/v1/data/market") as response:
                                success = response.status == 200
                        
                        elif action == 'analyze':
                            async with user_session.post(
                                f"{self.base_url}/api/v1/agents/finbert/analyze",
                                json={"text": f"Market analysis request from user {user_id}"}
                            ) as response:
                                success = response.status == 200
                        
                        elif action == 'logout':
                            async with user_session.post(f"{self.base_url}/api/v1/auth/logout") as response:
                                success = response.status in [200, 204]
                        
                        else:
                            success = True
                    
                    except Exception as e:
                        success = False
                        logger.debug(f"User {user_id} action {action} failed: {e}")
                    
                    action_end = time.time()
                    
                    action_results.append({
                        'action': action,
                        'duration_seconds': action_end - action_start,
                        'success': success
                    })
                    
                    # Small delay between actions
                    await asyncio.sleep(0.1)
            
            session_end = time.time()
            
            return {
                'user_id': user_id,
                'session_duration_seconds': session_end - session_start,
                'actions': action_results,
                'all_actions_successful': all(a['success'] for a in action_results)
            }
        
        # Run concurrent user sessions
        user_tasks = [simulate_user_session(i) for i in range(concurrent_users)]
        concurrent_results = await asyncio.gather(*user_tasks, return_exceptions=True)
        
        # Process results
        successful_sessions = []
        failed_sessions = []
        
        for result in concurrent_results:
            if isinstance(result, Exception):
                failed_sessions.append({'error': str(result)})
            else:
                if result['all_actions_successful']:
                    successful_sessions.append(result)
                else:
                    failed_sessions.append(result)
        
        return {
            'concurrent_users': concurrent_users,
            'successful_sessions': len(successful_sessions),
            'failed_sessions': len(failed_sessions),
            'success_rate': len(successful_sessions) / concurrent_users,
            'avg_session_duration': np.mean([s['session_duration_seconds'] for s in successful_sessions]) if successful_sessions else 0,
            'concurrency_test_passed': len(successful_sessions) / concurrent_users >= 0.8,
            'detailed_results': successful_sessions + failed_sessions
        }
    
    async def run_all_workflows(self) -> Dict[str, IntegrationTestResult]:
        """Run all defined workflows."""
        logger.info("Starting integration test suite - running all workflows")
        
        workflow_results = {}
        
        for workflow_name in self.workflows.keys():
            try:
                # Reset test data between workflows
                self.test_data = {}
                
                # Execute workflow
                result = await self.execute_workflow(workflow_name)
                workflow_results[workflow_name] = result
                
                # Small delay between workflows
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to execute workflow {workflow_name}: {e}")
                workflow_results[workflow_name] = IntegrationTestResult(
                    workflow_name=workflow_name,
                    total_steps=0,
                    successful_steps=0,
                    failed_steps=1,
                    total_execution_time_seconds=0.0,
                    workflow_results=[],
                    all_steps_successful=False,
                    performance_acceptable=False,
                    recommendations=[f"Workflow execution failed: {str(e)}"]
                )
        
        return workflow_results

async def main():
    parser = argparse.ArgumentParser(description='Integration Test Suite - Full Workflow')
    parser.add_argument('--base-url', '-u', default='http://localhost:8000',
                       help='Base URL for API endpoints (default: http://localhost:8000)')
    parser.add_argument('--workflow', '-w', 
                       choices=['end_to_end_market_analysis', 'portfolio_management_flow', 
                               'real_time_trading_flow', 'multi_user_concurrent_flow', 'all'],
                       default='all',
                       help='Specific workflow to run (default: all)')
    parser.add_argument('--concurrent-users', '-c', type=int, default=10,
                       help='Number of concurrent users for concurrency test (default: 10)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for results (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        async with IntegrationTestSuite(args.base_url) as test_suite:
            if args.workflow == 'all':
                # Run all workflows
                workflow_results = await test_suite.run_all_workflows()
                
                # Also run concurrent user test
                concurrent_results = await test_suite.test_multi_user_concurrent_access(args.concurrent_users)
                
                # Combine results
                final_results = {
                    'workflows': {name: asdict(result) for name, result in workflow_results.items()},
                    'concurrent_access': concurrent_results,
                    'overall_summary': {
                        'total_workflows': len(workflow_results),
                        'successful_workflows': sum(1 for r in workflow_results.values() if r.all_steps_successful),
                        'concurrent_test_passed': concurrent_results['concurrency_test_passed']
                    }
                }
            
            elif args.workflow == 'multi_user_concurrent_flow':
                # Run only concurrent test
                concurrent_results = await test_suite.test_multi_user_concurrent_access(args.concurrent_users)
                final_results = {'concurrent_access': concurrent_results}
            
            else:
                # Run specific workflow
                result = await test_suite.execute_workflow(args.workflow)
                final_results = {'workflow': asdict(result)}
            
            # Output results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(final_results, f, indent=2)
                logger.info(f"Results written to {args.output}")
            else:
                print(json.dumps(final_results, indent=2))
            
            # Log summary
            if 'workflows' in final_results:
                logger.info("\n" + "="*60)
                logger.info("INTEGRATION TEST SUMMARY")
                logger.info("="*60)
                
                for name, result in final_results['workflows'].items():
                    status = "✓ PASS" if result['all_steps_successful'] else "✗ FAIL"
                    logger.info(f"{name}: {status} ({result['successful_steps']}/{result['total_steps']} steps)")
                
                if 'concurrent_access' in final_results:
                    concurrent_status = "✓ PASS" if final_results['concurrent_access']['concurrency_test_passed'] else "✗ FAIL"
                    logger.info(f"Concurrent Access: {concurrent_status}")
            
            # Exit with appropriate code
            if 'workflows' in final_results:
                all_passed = all(r['all_steps_successful'] for r in final_results['workflows'].values())
                concurrent_passed = final_results.get('concurrent_access', {}).get('concurrency_test_passed', True)
                sys.exit(0 if all_passed and concurrent_passed else 1)
            else:
                sys.exit(0)
    
    except Exception as e:
        logger.error(f"Integration test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Monitoring stack testing script
"""

import asyncio
import aiohttp
import time
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringStackTester:
    """Test the monitoring stack"""
    
    def __init__(self):
        self.base_urls = {
            'prometheus': 'http://localhost:9090',
            'grafana': 'http://localhost:3000',
            'alertmanager': 'http://localhost:9093'
        }
        
    async def test_service_health(self):
        """Test health of all monitoring services"""
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for service, url in self.base_urls.items():
                try:
                    health_endpoint = f"{url}/-/healthy" if service != 'grafana' else f"{url}/api/health"
                    async with session.get(health_endpoint) as resp:
                        results[service] = resp.status == 200
                except Exception as e:
                    logger.error(f"{service} health check failed: {e}")
                    results[service] = False
                    
        return results
        
    async def run_comprehensive_test(self):
        """Run comprehensive monitoring stack test"""
        
        logger.info("🚀 Starting monitoring stack test...")
        
        results = {
            'timestamp': time.time(),
            'service_health': {},
            'overall_status': 'unknown'
        }
        
        # Test service health
        logger.info("🔍 Testing service health...")
        results['service_health'] = await self.test_service_health()
        
        # Determine overall status
        all_services_healthy = all(results['service_health'].values())
        results['overall_status'] = 'healthy' if all_services_healthy else 'unhealthy'
        
        return results
        
    def print_test_results(self, results):
        """Print formatted test results"""
        
        print("\n" + "="*60)
        print("🧪 MONITORING STACK TEST RESULTS")
        print("="*60)
        
        # Service Health
        print("\n🏥 Service Health:")
        for service, healthy in results['service_health'].items():
            status = "✅ Healthy" if healthy else "❌ Unhealthy"
            print(f"   {service.capitalize()}: {status}")
            
        # Overall Status
        status_emoji = '🟢' if results['overall_status'] == 'healthy' else '🔴'
        print(f"\n{status_emoji} Overall Status: {results['overall_status'].upper()}")
        
        print("="*60)

async def main():
    """Main test execution"""
    
    tester = MonitoringStackTester()
    results = await tester.run_comprehensive_test()
    tester.print_test_results(results)
    
    # Save results to file
    with open('monitoring_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n📄 Test results saved to: monitoring_test_results.json")

if __name__ == "__main__":
    asyncio.run(main())

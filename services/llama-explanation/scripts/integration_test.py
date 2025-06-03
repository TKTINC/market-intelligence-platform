# services/llama-explanation/scripts/integration_test.py
"""
Integration test between Llama service and Agent Orchestration
Tests the complete workflow from orchestrator to Llama and back
"""

import asyncio
import json
import time
from typing import Dict, Any
import httpx

async def test_orchestrator_integration():
    """Test integration with agent orchestrator"""
    
    # URLs (adjust as needed)
    orchestrator_url = "http://localhost:8000"  # Agent orchestration service
    llama_url = "http://localhost:8001"         # Llama service
    
    print("🔗 Testing Orchestrator → Llama Integration")
    print("-" * 50)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        
        # Test 1: Direct Llama call
        print("1. Testing direct Llama service...")
        
        llama_request = {
            "context": {
                "analysis_type": "sentiment",
                "symbol": "AAPL",
                "sentiment_score": 0.8,
                "current_price": 150.0
            },
            "max_tokens": 200,
            "temperature": 0.1
        }
        
        try:
            response = await client.post(f"{llama_url}/explain", json=llama_request)
            if response.status_code == 200:
                print("✅ Direct Llama call successful")
                llama_result = response.json()
                print(f"   Generated {llama_result['tokens_used']} tokens")
            else:
                print(f"❌ Direct Llama call failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Direct Llama call failed: {e}")
            return False
        
        # Test 2: Orchestrator → Llama via agent routing
        print("\n2. Testing Orchestrator → Llama routing...")
        
        orchestrator_request = {
            "analysis_type": "comprehensive_analysis",
            "data": {
                "symbol": "TSLA",
                "current_price": 250.0,
                "sentiment_score": 0.75,
                "iv_rank": 65
            },
            "preferences": {
                "explanation_model": "llama-7b",
                "latency_tolerance": 2000
            }
        }
        
        try:
            response = await client.post(
                f"{orchestrator_url}/analyze", 
                json=orchestrator_request
            )
            
            if response.status_code == 200:
                print("✅ Orchestrator routing successful")
                orchestrator_result = response.json()
                
                # Check if Llama was used for explanation
                if "explanation" in orchestrator_result.get("results", {}):
                    explanation_result = orchestrator_result["results"]["explanation"]
                    print("✅ Llama explanation found in results")
                    print(f"   Explanation length: {len(explanation_result.get('explanation', ''))}")
                else:
                    print("⚠️  No explanation found in orchestrator results")
                
            else:
                print(f"❌ Orchestrator routing failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Orchestrator routing failed: {e}")
            return False
        
        # Test 3: Performance comparison
        print("\n3. Testing performance comparison...")
        
        # Time direct Llama call
        start_time = time.time()
        await client.post(f"{llama_url}/explain", json=llama_request)
        direct_time = (time.time() - start_time) * 1000
        
        # Time orchestrated call
        start_time = time.time()
        await client.post(f"{orchestrator_url}/analyze", json=orchestrator_request)
        orchestrated_time = (time.time() - start_time) * 1000
        
        print(f"   Direct Llama: {direct_time:.1f}ms")
        print(f"   Via Orchestrator: {orchestrated_time:.1f}ms")
        print(f"   Overhead: {orchestrated_time - direct_time:.1f}ms")
        
        if orchestrated_time < direct_time * 3:  # Max 3x overhead
            print("✅ Performance overhead acceptable")
        else:
            print("⚠️  High performance overhead")
        
        print("\n🎉 Integration test completed successfully!")
        return True

if __name__ == "__main__":
    success = asyncio.run(test_orchestrator_integration())
    exit(0 if success else 1)

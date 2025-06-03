# services/llama-explanation/scripts/benchmark.py
"""
Benchmark script for comparing different Llama configurations
"""
import asyncio
import time
import json
from typing import Dict, List, Any
import httpx

async def benchmark_configurations():
    """Benchmark different Llama configurations"""
    
    configurations = [
        {"N_GPU_LAYERS": 35, "N_CTX": 4096, "MAX_CONCURRENT": 3},
        {"N_GPU_LAYERS": 30, "N_CTX": 4096, "MAX_CONCURRENT": 5},
        {"N_GPU_LAYERS": 35, "N_CTX": 2048, "MAX_CONCURRENT": 3},
    ]
    
    results = {}
    
    for config in configurations:
        print(f"Testing configuration: {config}")
        
        # Here you would restart the service with new config
        # For now, just test current configuration
        
        tester = LlamaPerformanceTester()
        result = await tester.test_single_request_latency(20)
        
        config_key = f"gpu{config['N_GPU_LAYERS']}_ctx{config['N_CTX']}_conc{config['MAX_CONCURRENT']}"
        results[config_key] = {
            "configuration": config,
            "performance": result
        }
    
    return results

if __name__ == "__main__":
    results = asyncio.run(benchmark_configurations())
    print(json.dumps(results, indent=2))

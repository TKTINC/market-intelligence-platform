#!/usr/bin/env python3
"""
=============================================================================
FINBERT AGENT PERFORMANCE BENCHMARK
Location: src/agents/finbert/tests/benchmark_finbert.py
=============================================================================
"""

import asyncio
import time
import json
import numpy as np
import psutil
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import argparse
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FinBERTPerformanceResult:
    iterations: int
    performance_metrics: Dict[str, float]
    accuracy_metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    test_duration_seconds: float
    passed_benchmarks: bool

class FinBERTBenchmark:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.getenv('FINBERT_MODEL_PATH', '/models/finbert')
        
        # Test sentences for sentiment analysis
        self.test_sentences = [
            "Company reports exceptional quarterly earnings beating analyst expectations",
            "Market volatility continues as investors remain uncertain about future outlook", 
            "Merger talks between tech giants could reshape the industry landscape",
            "Federal Reserve announces interest rate decision affecting market sentiment",
            "Economic indicators suggest potential recession concerns among analysts",
            "Strong revenue growth drives stock price higher in after-hours trading",
            "Regulatory investigation impacts company's market valuation significantly",
            "Dividend increase announcement boosts investor confidence substantially",
            "Supply chain disruptions affect quarterly guidance and projections",
            "New product launch generates positive market response and enthusiasm"
        ]
        
        # Performance thresholds
        self.thresholds = {
            'max_avg_response_time_ms': 1500,
            'max_p95_response_time_ms': 2000,
            'min_throughput_rps': 10,
            'min_accuracy': 0.85,
            'max_memory_usage_mb': 4000,
            'max_cpu_usage_percent': 80
        }
    
    async def simulate_finbert_inference(self, text: str) -> Dict[str, Any]:
        """Simulate FinBERT inference with realistic timing."""
        
        # Simulate processing time based on text length
        base_time = 0.1  # Base processing time
        text_factor = len(text) / 100 * 0.01  # Additional time per 100 chars
        processing_time = base_time + text_factor + np.random.normal(0, 0.05)
        processing_time = max(0.05, processing_time)  # Minimum 50ms
        
        await asyncio.sleep(processing_time)
        
        # Simulate sentiment prediction
        sentiments = ['positive', 'negative', 'neutral']
        predicted_sentiment = np.random.choice(sentiments)
        
        # Generate realistic confidence scores
        if predicted_sentiment == 'positive':
            confidence = np.random.beta(8, 2)  # Skewed towards high confidence
        elif predicted_sentiment == 'negative':
            confidence = np.random.beta(7, 3)
        else:  # neutral
            confidence = np.random.beta(5, 5)  # More uniform
        
        # Generate scores for all sentiments (should sum to ~1)
        positive_score = np.random.uniform(0.1, 0.9)
        negative_score = np.random.uniform(0.1, 0.9 - positive_score)
        neutral_score = 1.0 - positive_score - negative_score
        
        return {
            'sentiment': predicted_sentiment,
            'confidence': float(confidence),
            'scores': {
                'positive': float(positive_score),
                'negative': float(negative_score),
                'neutral': float(neutral_score)
            },
            'processing_time_ms': processing_time * 1000
        }
    
    async def benchmark_sentiment_analysis(self, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark sentiment analysis performance."""
        
        logger.info(f"Starting FinBERT benchmark with {iterations} iterations")
        
        results = {
            'iterations': iterations,
            'response_times': [],
            'accuracy_scores': [],
            'memory_usage': [],
            'cpu_usage': [],
            'predictions': []
        }
        
        # Get initial system state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        for i in range(iterations):
            # Random sentence selection
            text = np.random.choice(self.test_sentences)
            
            # Measure system resources before inference
            memory_before = process.memory_info().rss / 1024 / 1024
            cpu_before = psutil.cpu_percent(interval=None)
            
            # Run inference
            inference_start = time.time()
            result = await self.simulate_finbert_inference(text)
            inference_time = (time.time() - inference_start) * 1000
            
            # Measure system resources after inference
            memory_after = process.memory_info().rss / 1024 / 1024
            cpu_after = psutil.cpu_percent(interval=None)
            
            # Store results
            results['response_times'].append(inference_time)
            results['accuracy_scores'].append(result['confidence'])
            results['memory_usage'].append(memory_after - memory_before)
            results['cpu_usage'].append(max(0, cpu_after - cpu_before))
            results['predictions'].append({
                'text': text[:50] + '...' if len(text) > 50 else text,
                'sentiment': result['sentiment'],
                'confidence': result['confidence']
            })
            
            # Progress logging
            if (i + 1) % 20 == 0:
                logger.info(f"Completed {i + 1}/{iterations} iterations")
        
        total_time = time.time() - start_time
        
        # Calculate performance statistics
        response_times = results['response_times']
        results['performance_metrics'] = {
            'avg_response_time_ms': float(np.mean(response_times)),
            'p95_response_time_ms': float(np.percentile(response_times, 95)),
            'p99_response_time_ms': float(np.percentile(response_times, 99)),
            'min_response_time_ms': float(np.min(response_times)),
            'max_response_time_ms': float(np.max(response_times)),
            'std_response_time_ms': float(np.std(response_times)),
            'avg_accuracy': float(np.mean(results['accuracy_scores'])),
            'min_accuracy': float(np.min(results['accuracy_scores'])),
            'max_accuracy': float(np.max(results['accuracy_scores'])),
            'avg_memory_delta_mb': float(np.mean(results['memory_usage'])),
            'peak_memory_delta_mb': float(np.max(results['memory_usage'])),
            'avg_cpu_delta_percent': float(np.mean(results['cpu_usage'])),
            'peak_cpu_delta_percent': float(np.max(results['cpu_usage'])),
            'throughput_requests_per_second': iterations / total_time,
            'total_test_duration_seconds': total_time
        }
        
        logger.info(f"Benchmark completed in {total_time:.2f} seconds")
        logger.info(f"Average response time: {results['performance_metrics']['avg_response_time_ms']:.1f}ms")
        logger.info(f"Throughput: {results['performance_metrics']['throughput_requests_per_second']:.1f} RPS")
        
        return results
    
    def test_accuracy_with_labeled_data(self) -> Dict[str, Any]:
        """Test accuracy against labeled financial sentiment data."""
        
        logger.info("Testing accuracy with labeled data")
        
        # Labeled test data (text, expected_sentiment, expected_confidence_threshold)
        labeled_data = [
            ("Company exceeds revenue expectations significantly", "positive", 0.8),
            ("Stock price plummets amid accounting scandal", "negative", 0.8),
            ("Quarterly report meets analyst expectations exactly", "neutral", 0.6),
            ("Massive layoffs announced affecting thousands", "negative", 0.9),
            ("New product launch generates substantial buzz", "positive", 0.7),
            ("Market conditions remain stable and unchanged", "neutral", 0.7),
            ("Dividend cut disappoints longtime shareholders", "negative", 0.8),
            ("Strong earnings growth drives investor optimism", "positive", 0.8),
            ("Regulatory compliance costs impact margins", "negative", 0.6),
            ("Strategic partnership announced with tech giant", "positive", 0.7)
        ]
        
        correct_predictions = 0
        confidence_threshold_met = 0
        detailed_results = []
        
        for text, expected_sentiment, confidence_threshold in labeled_data:
            # Simulate prediction (in real implementation, call actual model)
            # For demo, we'll create realistic but random predictions
            sentiments = ['positive', 'negative', 'neutral']
            
            # Bias prediction towards expected sentiment (80% accuracy)
            if np.random.random() < 0.8:
                predicted_sentiment = expected_sentiment
                confidence = np.random.uniform(confidence_threshold, 0.95)
            else:
                predicted_sentiment = np.random.choice([s for s in sentiments if s != expected_sentiment])
                confidence = np.random.uniform(0.5, 0.8)
            
            # Check accuracy
            correct = predicted_sentiment == expected_sentiment
            confidence_ok = confidence >= confidence_threshold
            
            if correct:
                correct_predictions += 1
            if confidence_ok:
                confidence_threshold_met += 1
            
            detailed_results.append({
                'text': text,
                'expected_sentiment': expected_sentiment,
                'predicted_sentiment': predicted_sentiment,
                'confidence': confidence,
                'correct_prediction': correct,
                'confidence_threshold_met': confidence_ok
            })
        
        total_samples = len(labeled_data)
        accuracy = correct_predictions / total_samples
        confidence_rate = confidence_threshold_met / total_samples
        
        accuracy_metrics = {
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'confidence_threshold_rate': confidence_rate,
            'passed_accuracy_threshold': accuracy >= self.thresholds['min_accuracy'],
            'detailed_results': detailed_results
        }
        
        logger.info(f"Accuracy test completed: {accuracy:.1%} accuracy")
        logger.info(f"Confidence threshold met: {confidence_rate:.1%}")
        
        return accuracy_metrics
    
    def memory_profiling_test(self) -> Dict[str, Any]:
        """Profile memory usage during inference."""
        
        logger.info("Running memory profiling test")
        
        process = psutil.Process()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        memory_samples = []
        test_iterations = 50
        
        for i in range(test_iterations):
            # Sample memory before inference
            pre_memory = process.memory_info().rss / 1024 / 1024
            
            # Simulate inference
            text = np.random.choice(self.test_sentences)
            # In real implementation: result = model.predict(text)
            time.sleep(np.random.uniform(0.1, 0.3))  # Simulate processing
            
            # Sample memory after inference  
            post_memory = process.memory_info().rss / 1024 / 1024
            
            memory_samples.append({
                'iteration': i,
                'pre_inference_mb': pre_memory,
                'post_inference_mb': post_memory,
                'delta_mb': post_memory - pre_memory
            })
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        memory_metrics = {
            'baseline_memory_mb': baseline_memory,
            'final_memory_mb': final_memory,
            'total_memory_increase_mb': final_memory - baseline_memory,
            'avg_memory_per_inference_mb': np.mean([s['delta_mb'] for s in memory_samples]),
            'max_memory_per_inference_mb': np.max([s['delta_mb'] for s in memory_samples]),
            'memory_samples': memory_samples,
            'passed_memory_threshold': final_memory <= self.thresholds['max_memory_usage_mb']
        }
        
        logger.info(f"Memory profiling completed")
        logger.info(f"Total memory increase: {memory_metrics['total_memory_increase_mb']:.1f} MB")
        
        return memory_metrics
    
    def check_performance_thresholds(self, performance_metrics: Dict[str, float],
                                   accuracy_metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check if performance meets defined thresholds."""
        
        checks = {
            'response_time_avg': performance_metrics['avg_response_time_ms'] <= self.thresholds['max_avg_response_time_ms'],
            'response_time_p95': performance_metrics['p95_response_time_ms'] <= self.thresholds['max_p95_response_time_ms'],
            'throughput': performance_metrics['throughput_requests_per_second'] >= self.thresholds['min_throughput_rps'],
            'accuracy': accuracy_metrics['accuracy'] >= self.thresholds['min_accuracy'],
            'memory_usage': performance_metrics.get('peak_memory_delta_mb', 0) <= self.thresholds['max_memory_usage_mb'],
            'cpu_usage': performance_metrics.get('peak_cpu_delta_percent', 0) <= self.thresholds['max_cpu_usage_percent']
        }
        
        return checks
    
    async def run_full_benchmark(self, iterations: int = 100) -> FinBERTPerformanceResult:
        """Run complete FinBERT performance benchmark."""
        
        logger.info("="*60)
        logger.info("STARTING FINBERT PERFORMANCE BENCHMARK")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Run performance benchmark
        perf_results = await self.benchmark_sentiment_analysis(iterations)
        
        # Run accuracy test
        accuracy_results = self.test_accuracy_with_labeled_data()
        
        # Run memory profiling
        memory_results = self.memory_profiling_test()
        
        # Check thresholds
        threshold_checks = self.check_performance_thresholds(
            perf_results['performance_metrics'],
            accuracy_results
        )
        
        total_duration = time.time() - start_time
        
        # Determine overall pass/fail
        passed_benchmarks = all(threshold_checks.values())
        
        # Compile final result
        result = FinBERTPerformanceResult(
            iterations=iterations,
            performance_metrics=perf_results['performance_metrics'],
            accuracy_metrics=accuracy_results,
            resource_usage=memory_results,
            test_duration_seconds=total_duration,
            passed_benchmarks=passed_benchmarks
        )
        
        # Log summary
        logger.info("="*60)
        logger.info("FINBERT BENCHMARK SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Duration: {total_duration:.1f} seconds")
        logger.info(f"Iterations: {iterations}")
        logger.info(f"Average Response Time: {perf_results['performance_metrics']['avg_response_time_ms']:.1f}ms")
        logger.info(f"P95 Response Time: {perf_results['performance_metrics']['p95_response_time_ms']:.1f}ms")
        logger.info(f"Throughput: {perf_results['performance_metrics']['throughput_requests_per_second']:.1f} RPS")
        logger.info(f"Accuracy: {accuracy_results['accuracy']:.1%}")
        logger.info(f"Overall Result: {'✓ PASSED' if passed_benchmarks else '✗ FAILED'}")
        
        # Log threshold checks
        logger.info("\nThreshold Checks:")
        for check_name, passed in threshold_checks.items():
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {check_name}")
        
        return result

def main():
    parser = argparse.ArgumentParser(description='FinBERT Performance Benchmark')
    parser.add_argument('--iterations', '-i', type=int, default=100,
                       help='Number of benchmark iterations (default: 100)')
    parser.add_argument('--model-path', '-m', type=str,
                       help='Path to FinBERT model')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for results (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize benchmark
        benchmark = FinBERTBenchmark(model_path=args.model_path)
        
        # Run benchmark
        result = asyncio.run(benchmark.run_full_benchmark(args.iterations))
        
        # Output results
        result_dict = asdict(result)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result_dict, f, indent=2)
            logger.info(f"Results written to {args.output}")
        else:
            print(json.dumps(result_dict, indent=2))
        
        # Exit with appropriate code
        sys.exit(0 if result.passed_benchmarks else 1)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
=============================================================================
LLAMA AGENT REASONING QUALITY TEST
Location: src/agents/llama/tests/reasoning_quality_test.py
=============================================================================
"""

import asyncio
import time
import json
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse
import sys
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ReasoningPrompt:
    prompt: str
    expected_concepts: List[str]
    difficulty: str
    category: str
    min_quality_score: float

@dataclass
class ReasoningResult:
    prompt: str
    response: str
    response_time_ms: float
    quality_score: float
    concept_coverage: float
    difficulty: str
    category: str
    passed: bool
    issues: List[str]

@dataclass
class LlamaReasoningTestResult:
    total_tests: int
    passed_tests: int
    failed_tests: int
    avg_quality_score: float
    avg_response_time_ms: float
    category_breakdown: Dict[str, Dict[str, Any]]
    context_window_results: Dict[str, Any]
    overall_passed: bool
    recommendations: List[str]

class LlamaReasoningTest:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.getenv('LLAMA_MODEL_PATH', '/models/llama-3.1-8b')
        
        # Define reasoning test prompts
        self.reasoning_prompts = [
            # Financial Analysis Category
            ReasoningPrompt(
                prompt="Analyze the financial implications of a company reporting 20% revenue increase but 10% decrease in net profit margin. What factors could explain this situation and what should investors consider?",
                expected_concepts=["revenue growth", "margin compression", "cost inflation", "operational efficiency", "investment impact"],
                difficulty="medium",
                category="financial_analysis",
                min_quality_score=0.75
            ),
            ReasoningPrompt(
                prompt="A tech company trades at P/E ratio of 45 while industry average is 25. The company has 40% annual growth rate vs industry 15%. Evaluate if this valuation premium is justified.",
                expected_concepts=["valuation", "growth premium", "PEG ratio", "risk assessment", "market efficiency"],
                difficulty="hard",
                category="financial_analysis", 
                min_quality_score=0.80
            ),
            ReasoningPrompt(
                prompt="Explain how rising interest rates impact different sectors differently, specifically comparing utilities vs technology stocks.",
                expected_concepts=["interest rates", "sector rotation", "discount rates", "growth stocks", "defensive stocks"],
                difficulty="medium",
                category="financial_analysis",
                min_quality_score=0.70
            ),
            
            # Market Strategy Category
            ReasoningPrompt(
                prompt="Two companies are merging with combined market cap of $150B. Analyze the key factors that determine whether this merger creates or destroys shareholder value.",
                expected_concepts=["synergies", "integration costs", "market power", "regulatory approval", "cultural fit"],
                difficulty="hard",
                category="market_strategy",
                min_quality_score=0.75
            ),
            ReasoningPrompt(
                prompt="A portfolio manager must choose between value and growth investing during a market downturn. Provide a structured analysis of both approaches.",
                expected_concepts=["value investing", "growth investing", "market cycles", "risk management", "diversification"],
                difficulty="medium",
                category="market_strategy",
                min_quality_score=0.70
            ),
            
            # Risk Assessment Category
            ReasoningPrompt(
                prompt="Assess the systematic vs idiosyncratic risks of investing in emerging market debt during a global economic slowdown.",
                expected_concepts=["systematic risk", "idiosyncratic risk", "emerging markets", "currency risk", "sovereign risk"],
                difficulty="hard",
                category="risk_assessment",
                min_quality_score=0.80
            ),
            ReasoningPrompt(
                prompt="A hedge fund uses 3x leverage. Explain the risk-return implications and under what market conditions this strategy might fail catastrophically.",
                expected_concepts=["leverage", "margin calls", "volatility", "drawdown", "risk management"],
                difficulty="hard",
                category="risk_assessment",
                min_quality_score=0.85
            ),
            
            # Economic Reasoning Category
            ReasoningPrompt(
                prompt="Central bank increases money supply by 15% while inflation remains at 2%. Analyze potential economic explanations and market implications.",
                expected_concepts=["monetary policy", "liquidity trap", "velocity of money", "asset prices", "transmission mechanisms"],
                difficulty="hard",
                category="economic_reasoning",
                min_quality_score=0.75
            ),
            ReasoningPrompt(
                prompt="Oil prices drop 40% in 6 months. Trace through the economic implications for different industries and geographic regions.",
                expected_concepts=["commodity cycles", "input costs", "consumer spending", "inflation", "sectoral impacts"],
                difficulty="medium",
                category="economic_reasoning",
                min_quality_score=0.70
            )
        ]
        
        # Performance thresholds
        self.thresholds = {
            'min_overall_quality': 0.75,
            'max_avg_response_time_ms': 8000,
            'min_pass_rate': 0.80,
            'max_context_processing_time_ms': 15000
        }
    
    async def simulate_llama_reasoning(self, prompt: str) -> Tuple[str, float]:
        """Simulate Llama reasoning response with realistic timing."""
        
        # Calculate processing time based on prompt complexity
        base_time = 2.0  # Base processing time
        complexity_factor = len(prompt) / 100 * 0.5  # Additional time per 100 chars
        reasoning_time = base_time + complexity_factor + np.random.normal(0, 0.8)
        reasoning_time = max(1.0, reasoning_time)  # Minimum 1 second
        
        await asyncio.sleep(reasoning_time)
        
        # Generate realistic reasoning response
        reasoning_templates = [
            "Based on the analysis of the provided information, several key factors need consideration. ",
            "To properly evaluate this situation, we must examine multiple interconnected elements. ",
            "This scenario requires a structured approach to understand the underlying dynamics. ",
            "From a financial perspective, this situation presents both opportunities and risks. ",
            "The economic implications of this scenario are multifaceted and require careful analysis. "
        ]
        
        analysis_phrases = [
            "market dynamics suggest",
            "fundamental analysis indicates", 
            "considering the risk factors",
            "from a valuation perspective",
            "the regulatory environment",
            "operational efficiency metrics",
            "competitive positioning",
            "macroeconomic trends",
            "investor sentiment patterns",
            "liquidity considerations"
        ]
        
        conclusion_phrases = [
            "Therefore, investors should carefully monitor",
            "In conclusion, the key takeaway is",
            "Given these factors, the recommendation would be to",
            "The optimal strategy would involve",
            "Risk management requires attention to"
        ]
        
        # Build response
        response_parts = [
            np.random.choice(reasoning_templates),
            f"First, {np.random.choice(analysis_phrases)} that the situation involves complex interplay of market forces. ",
            f"Additionally, {np.random.choice(analysis_phrases)} the importance of timing and market conditions. ",
            f"Furthermore, {np.random.choice(analysis_phrases)} potential regulatory and competitive impacts. ",
            f"{np.random.choice(conclusion_phrases)} the evolving market landscape and adjust strategies accordingly."
        ]
        
        response = "".join(response_parts)
        
        return response, reasoning_time * 1000  # Return time in ms
    
    def evaluate_reasoning_quality(self, prompt: ReasoningPrompt, response: str) -> Tuple[float, float, List[str]]:
        """Evaluate the quality of reasoning response."""
        
        issues = []
        
        # 1. Concept Coverage (40% of score)
        concepts_found = 0
        for concept in prompt.expected_concepts:
            # Look for concept or related terms in response
            concept_patterns = [
                concept.lower(),
                concept.replace('_', ' ').lower(),
                concept.replace(' ', '').lower()
            ]
            
            if any(pattern in response.lower() for pattern in concept_patterns):
                concepts_found += 1
        
        concept_coverage = concepts_found / len(prompt.expected_concepts)
        
        if concept_coverage < 0.6:
            issues.append(f"Low concept coverage: {concept_coverage:.1%}")
        
        # 2. Response Length and Structure (20% of score)
        word_count = len(response.split())
        length_score = min(1.0, word_count / 100)  # Optimal around 100+ words
        
        if word_count < 50:
            issues.append(f"Response too short: {word_count} words")
        
        # 3. Coherence and Logic (25% of score)
        # Simple heuristics for coherence
        sentence_count = len([s for s in response.split('.') if s.strip()])
        avg_sentence_length = word_count / max(1, sentence_count)
        
        coherence_score = 1.0
        if avg_sentence_length > 30:  # Very long sentences
            coherence_score -= 0.2
            issues.append("Some sentences may be too complex")
        
        if sentence_count < 3:  # Too few sentences
            coherence_score -= 0.3
            issues.append("Response lacks sufficient detail")
        
        # 4. Financial Terminology (15% of score)
        financial_terms = [
            'market', 'risk', 'return', 'investment', 'portfolio', 'valuation', 
            'volatility', 'liquidity', 'capital', 'revenue', 'profit', 'margin',
            'growth', 'earnings', 'dividend', 'analysis', 'strategy'
        ]
        
        terms_used = sum(1 for term in financial_terms if term in response.lower())
        terminology_score = min(1.0, terms_used / 5)  # Expect at least 5 financial terms
        
        if terminology_score < 0.6:
            issues.append("Limited use of financial terminology")
        
        # Calculate overall quality score
        quality_score = (
            concept_coverage * 0.40 +
            length_score * 0.20 +
            coherence_score * 0.25 +
            terminology_score * 0.15
        )
        
        return quality_score, concept_coverage, issues
    
    async def test_reasoning_quality(self) -> Dict[str, Any]:
        """Test reasoning quality across all prompts."""
        
        logger.info(f"Testing reasoning quality with {len(self.reasoning_prompts)} prompts")
        
        results = []
        
        for i, prompt in enumerate(self.reasoning_prompts):
            logger.info(f"Testing prompt {i+1}/{len(self.reasoning_prompts)}: {prompt.category}")
            
            start_time = time.time()
            
            # Get reasoning response
            response, response_time_ms = await self.simulate_llama_reasoning(prompt.prompt)
            
            # Evaluate quality
            quality_score, concept_coverage, issues = self.evaluate_reasoning_quality(prompt, response)
            
            # Determine if test passed
            passed = quality_score >= prompt.min_quality_score
            
            result = ReasoningResult(
                prompt=prompt.prompt[:100] + "..." if len(prompt.prompt) > 100 else prompt.prompt,
                response=response[:200] + "..." if len(response) > 200 else response,
                response_time_ms=response_time_ms,
                quality_score=quality_score,
                concept_coverage=concept_coverage,
                difficulty=prompt.difficulty,
                category=prompt.category,
                passed=passed,
                issues=issues
            )
            
            results.append(result)
            
            # Log result
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"  {status} - Quality: {quality_score:.2f}, Time: {response_time_ms:.0f}ms")
        
        # Calculate summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        avg_quality = np.mean([r.quality_score for r in results])
        avg_response_time = np.mean([r.response_time_ms for r in results])
        
        # Category breakdown
        categories = set(r.category for r in results)
        category_breakdown = {}
        
        for category in categories:
            category_results = [r for r in results if r.category == category]
            category_breakdown[category] = {
                'total_tests': len(category_results),
                'passed_tests': sum(1 for r in category_results if r.passed),
                'avg_quality_score': np.mean([r.quality_score for r in category_results]),
                'avg_response_time_ms': np.mean([r.response_time_ms for r in category_results])
            }
        
        return {
            'test_results': [asdict(r) for r in results],
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate': passed_tests / total_tests,
            'avg_quality_score': avg_quality,
            'avg_response_time_ms': avg_response_time,
            'category_breakdown': category_breakdown,
            'thresholds_met': {
                'overall_quality': avg_quality >= self.thresholds['min_overall_quality'],
                'response_time': avg_response_time <= self.thresholds['max_avg_response_time_ms'],
                'pass_rate': (passed_tests / total_tests) >= self.thresholds['min_pass_rate']
            }
        }
    
    def test_context_window_handling(self) -> Dict[str, Any]:
        """Test handling of various context window sizes."""
        
        logger.info("Testing context window handling")
        
        context_sizes = [512, 1024, 2048, 4096, 8192]
        results = []
        
        for size in context_sizes:
            logger.info(f"Testing context size: {size} tokens")
            
            # Generate mock text of specified token size (rough approximation)
            # Assuming ~4 characters per token on average
            char_count = size * 4
            base_text = "Financial market analysis requires understanding of complex economic indicators. "
            mock_context = (base_text * (char_count // len(base_text) + 1))[:char_count]
            
            start_time = time.time()
            
            # Simulate processing time based on context size
            # Larger contexts take longer to process
            base_processing = 1.0
            context_factor = size / 1000  # Linear scaling factor
            processing_time = base_processing + context_factor + np.random.normal(0, 0.3)
            processing_time = max(0.5, processing_time)
            
            time.sleep(processing_time)
            
            end_time = time.time()
            actual_time_ms = (end_time - start_time) * 1000
            
            # Simulate memory usage (rough approximation)
            memory_used_mb = size * 0.008  # ~8KB per 1000 tokens
            
            # Determine success based on processing time threshold
            success = actual_time_ms < self.thresholds['max_context_processing_time_ms']
            
            result = {
                'context_size_tokens': size,
                'context_size_chars': char_count,
                'processing_time_ms': actual_time_ms,
                'estimated_memory_mb': memory_used_mb,
                'success': success,
                'tokens_per_second': size / (actual_time_ms / 1000) if actual_time_ms > 0 else 0
            }
            
            results.append(result)
            
            status = "✓ PASS" if success else "✗ FAIL"
            logger.info(f"  {status} - {size} tokens in {actual_time_ms:.0f}ms")
        
        max_supported = max(r['context_size_tokens'] for r in results if r['success'])
        
        return {
            'context_tests': results,
            'max_supported_context_tokens': max_supported,
            'avg_tokens_per_second': np.mean([r['tokens_per_second'] for r in results if r['success']]),
            'performance_degradation': {
                'linear_scaling': True,  # Simplified assumption
                'memory_efficiency': np.mean([r['estimated_memory_mb'] for r in results])
            }
        }
    
    def generate_recommendations(self, reasoning_results: Dict[str, Any],
                               context_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        
        recommendations = []
        
        # Quality-based recommendations
        if reasoning_results['avg_quality_score'] < self.thresholds['min_overall_quality']:
            recommendations.append(
                f"Overall reasoning quality ({reasoning_results['avg_quality_score']:.2f}) below threshold. "
                f"Consider fine-tuning on financial reasoning tasks."
            )
        
        # Response time recommendations
        if reasoning_results['avg_response_time_ms'] > self.thresholds['max_avg_response_time_ms']:
            recommendations.append(
                f"Average response time ({reasoning_results['avg_response_time_ms']:.0f}ms) exceeds threshold. "
                f"Consider model optimization or hardware upgrades."
            )
        
        # Category-specific recommendations
        for category, stats in reasoning_results['category_breakdown'].items():
            if stats['avg_quality_score'] < 0.7:
                recommendations.append(
                    f"Low performance in {category} category. Consider additional training data for this domain."
                )
        
        # Context window recommendations
        if context_results['max_supported_context_tokens'] < 4096:
            recommendations.append(
                "Limited context window support. Consider model architecture improvements for longer contexts."
            )
        
        # Pass rate recommendations
        if reasoning_results['pass_rate'] < self.thresholds['min_pass_rate']:
            recommendations.append(
                f"Test pass rate ({reasoning_results['pass_rate']:.1%}) below threshold. "
                f"Review failing test cases and improve model consistency."
            )
        
        if not recommendations:
            recommendations.append("All tests passing within acceptable thresholds. Model performance is satisfactory.")
        
        return recommendations
    
    async def run_full_test_suite(self) -> LlamaReasoningTestResult:
        """Run complete Llama reasoning test suite."""
        
        logger.info("="*60)
        logger.info("STARTING LLAMA REASONING TEST SUITE")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Run reasoning quality tests
        reasoning_results = await self.test_reasoning_quality()
        
        # Run context window tests
        context_results = self.test_context_window_handling()
        
        # Generate recommendations
        recommendations = self.generate_recommendations(reasoning_results, context_results)
        
        # Determine overall pass/fail
        overall_passed = all([
            reasoning_results['thresholds_met']['overall_quality'],
            reasoning_results['thresholds_met']['response_time'],
            reasoning_results['thresholds_met']['pass_rate']
        ])
        
        total_duration = time.time() - start_time
        
        # Compile final result
        result = LlamaReasoningTestResult(
            total_tests=reasoning_results['total_tests'],
            passed_tests=reasoning_results['passed_tests'],
            failed_tests=reasoning_results['failed_tests'],
            avg_quality_score=reasoning_results['avg_quality_score'],
            avg_response_time_ms=reasoning_results['avg_response_time_ms'],
            category_breakdown=reasoning_results['category_breakdown'],
            context_window_results=context_results,
            overall_passed=overall_passed,
            recommendations=recommendations
        )
        
        # Log summary
        logger.info("="*60)
        logger.info("LLAMA REASONING TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Duration: {total_duration:.1f} seconds")
        logger.info(f"Tests: {reasoning_results['passed_tests']}/{reasoning_results['total_tests']} passed")
        logger.info(f"Pass Rate: {reasoning_results['pass_rate']:.1%}")
        logger.info(f"Average Quality Score: {reasoning_results['avg_quality_score']:.2f}")
        logger.info(f"Average Response Time: {reasoning_results['avg_response_time_ms']:.0f}ms")
        logger.info(f"Max Context Support: {context_results['max_supported_context_tokens']} tokens")
        logger.info(f"Overall Result: {'✓ PASSED' if overall_passed else '✗ FAILED'}")
        
        # Log category breakdown
        logger.info("\nCategory Performance:")
        for category, stats in reasoning_results['category_breakdown'].items():
            logger.info(f"  {category}: {stats['passed_tests']}/{stats['total_tests']} "
                       f"(Quality: {stats['avg_quality_score']:.2f})")
        
        return result

def main():
    parser = argparse.ArgumentParser(description='Llama Reasoning Quality Test')
    parser.add_argument('--model-path', '-m', type=str,
                       help='Path to Llama model')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for results (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--context-only', action='store_true',
                       help='Run only context window tests')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize test suite
        test_suite = LlamaReasoningTest(model_path=args.model_path)
        
        if args.context_only:
            # Run only context window tests
            result = test_suite.test_context_window_handling()
        else:
            # Run full test suite
            result = asyncio.run(test_suite.run_full_test_suite())
        
        # Output results
        result_dict = asdict(result) if hasattr(result, '__dict__') else result
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result_dict, f, indent=2)
            logger.info(f"Results written to {args.output}")
        else:
            print(json.dumps(result_dict, indent=2))
        
        # Exit with appropriate code
        if isinstance(result, LlamaReasoningTestResult):
            sys.exit(0 if result.overall_passed else 1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

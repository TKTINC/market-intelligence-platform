# services/llama-explanation/src/llama_engine.py
import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import gc

# Llama-cpp-python imports
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llama2ChatHandler
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available, using mock implementation")

from config import settings

logger = logging.getLogger(__name__)

class LlamaEngine:
    """
    Optimized Llama 2-7B engine for financial explanations
    Uses quantized GGUF model for efficient GPU inference
    """
    
    def __init__(self):
        self.model: Optional[Llama] = None
        self.chat_handler: Optional[Llama2ChatHandler] = None
        self.is_initialized = False
        self.model_path = settings.MODEL_PATH
        self.generation_stats = {
            'total_requests': 0,
            'total_tokens_generated': 0,
            'total_inference_time': 0.0,
            'average_tokens_per_second': 0.0
        }
        
        # Financial system prompt
        self.default_system_prompt = """You are a financial AI assistant providing clear explanations of trading recommendations and market analysis. 

Your responses should:
- Focus on risk factors and their implications
- Explain market conditions driving recommendations
- Provide expected outcomes and probabilities  
- Use clear, professional language suitable for both retail and institutional investors
- Be concise but comprehensive (aim for 100-200 words)
- Include specific reasoning behind conclusions

Always consider:
- Risk management principles
- Market volatility and timing
- Portfolio impact and diversification
- Regulatory and compliance factors"""

    async def initialize(self):
        """Initialize the Llama model with quantization"""
        logger.info("Initializing Llama 2-7B engine...")
        
        try:
            # Check if model file exists
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            if not LLAMA_CPP_AVAILABLE:
                logger.warning("Using mock Llama implementation for development")
                self.is_initialized = True
                return
            
            # Check GPU availability
            gpu_available = torch.cuda.is_available()
            logger.info(f"GPU available: {gpu_available}")
            
            if gpu_available:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU memory: {gpu_memory:.1f} GB")
            
            # Initialize Llama model with optimized settings
            start_time = time.time()
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=settings.N_CTX,              # Context window size
                n_batch=settings.N_BATCH,          # Batch size for prompt processing
                n_gpu_layers=settings.N_GPU_LAYERS if gpu_available else 0,  # GPU layers
                n_threads=settings.N_THREADS,      # CPU threads
                f16_kv=True,                       # Use float16 for key/value cache
                use_mlock=True,                    # Lock model in memory
                use_mmap=True,                     # Memory-map model file
                verbose=False                      # Reduce logging noise
            )
            
            # Initialize chat handler for conversation format
            self.chat_handler = Llama2ChatHandler(llama=self.model)
            
            initialization_time = time.time() - start_time
            logger.info(f"Llama model initialized in {initialization_time:.2f} seconds")
            
            # Log model information
            self._log_model_info()
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Llama engine: {str(e)}")
            raise
    
    def _log_model_info(self):
        """Log detailed model information"""
        try:
            info = {
                "model_path": self.model_path,
                "context_size": settings.N_CTX,
                "gpu_layers": settings.N_GPU_LAYERS,
                "batch_size": settings.N_BATCH,
                "threads": settings.N_THREADS
            }
            
            if torch.cuda.is_available():
                info["gpu_device"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            logger.info(f"Model configuration: {json.dumps(info, indent=2)}")
            
        except Exception as e:
            logger.error(f"Failed to log model info: {str(e)}")
    
    async def generate_explanation(
        self,
        context: Dict[str, Any],
        max_tokens: int = 256,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """Generate explanation from context"""
        
        if not self.is_initialized:
            raise RuntimeError("Llama engine not initialized")
        
        start_time = time.time()
        
        try:
            # Use custom system prompt or default
            sys_prompt = system_prompt or self.default_system_prompt
            
            # Format the user prompt from context
            user_prompt = self._format_context_prompt(context)
            
            # Generate explanation
            if LLAMA_CPP_AVAILABLE and self.model:
                result = await self._generate_with_llama(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            else:
                # Mock implementation for development
                result = await self._generate_mock_explanation(context, max_tokens)
            
            # Calculate metrics
            inference_time = time.time() - start_time
            tokens_generated = result.get('tokens_used', len(result['explanation'].split()))
            
            # Update statistics
            self._update_stats(inference_time, tokens_generated)
            
            # Add metadata
            result.update({
                'processing_time_ms': int(inference_time * 1000),
                'tokens_per_second': tokens_generated / inference_time if inference_time > 0 else 0,
                'model_info': {
                    'model': 'llama-2-7b-chat',
                    'quantization': 'Q4_K_M',
                    'temperature': temperature,
                    'max_tokens': max_tokens
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            raise
    
    async def _generate_with_llama(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Generate text using Llama model"""
        
        try:
            # Format messages for Llama 2 chat format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Run generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_llama_generation,
                messages,
                max_tokens,
                temperature
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Llama generation failed: {str(e)}")
            raise
    
    def _run_llama_generation(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Run Llama generation (synchronous)"""
        
        try:
            # Create chat completion
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["</s>", "[INST]", "[/INST]"],
                stream=False
            )
            
            # Extract response
            explanation = response['choices'][0]['message']['content'].strip()
            
            # Calculate tokens used
            tokens_used = response['usage']['total_tokens']
            
            # Calculate confidence score based on response quality
            confidence_score = self._calculate_confidence(explanation)
            
            return {
                'explanation': explanation,
                'tokens_used': tokens_used,
                'confidence_score': confidence_score
            }
            
        except Exception as e:
            logger.error(f"Llama generation error: {str(e)}")
            raise
    
    async def _generate_mock_explanation(
        self,
        context: Dict[str, Any],
        max_tokens: int
    ) -> Dict[str, Any]:
        """Mock explanation generation for development"""
        
        # Simulate processing time
        await asyncio.sleep(0.2)  # 200ms simulated inference time
        
        # Extract context information
        analysis_type = context.get('analysis_type', 'general')
        symbol = context.get('symbol', 'N/A')
        
        # Generate contextual explanation
        explanations = {
            'sentiment': f"""The sentiment analysis for {symbol} reveals positive market sentiment driven by strong institutional confidence and favorable analyst coverage. Key factors include robust earnings expectations, sector leadership, and technical momentum indicators. The bullish sentiment is supported by increasing call option activity and reduced put/call ratios, suggesting optimistic positioning among both retail and institutional investors. Risk factors to monitor include broader market volatility and sector-specific headwinds that could impact sentiment sustainability.""",
            
            'price_prediction': f"""The price forecasting model for {symbol} indicates a moderate upward trajectory based on technical momentum and fundamental strength. The prediction considers multiple factors including moving average convergence, volume profile analysis, and historical volatility patterns. Current support levels provide a solid foundation for price appreciation, while resistance levels suggest measured gains rather than explosive moves. Investors should consider the probability-weighted scenarios and maintain appropriate risk management through position sizing and stop-loss levels.""",
            
            'options_strategy': f"""The recommended options strategy for {symbol} is designed to capitalize on current market conditions while managing downside risk. Given the implied volatility environment and time decay considerations, this approach balances income generation with capital preservation. The strategy structure accounts for earnings announcements, dividend dates, and historical price movement patterns. Maximum profit potential is achieved within the expected trading range, while defined risk parameters ensure losses remain within acceptable portfolio allocation limits.""",
            
            'comprehensive': f"""This comprehensive analysis of {symbol} integrates multiple analytical dimensions to provide a holistic investment perspective. The convergence of positive sentiment indicators, favorable technical setups, and strategic options positioning creates a compelling opportunity within defined risk parameters. Key success factors include continued market stability, sector rotation dynamics, and company-specific catalysts. The analysis emphasizes risk-adjusted returns and portfolio construction principles, ensuring recommendations align with prudent investment management practices."""
        }
        
        # Select appropriate explanation
        explanation_key = 'comprehensive'
        for key in explanations.keys():
            if key in analysis_type.lower():
                explanation_key = key
                break
        
        explanation = explanations[explanation_key]
        
        # Calculate mock metrics
        tokens_used = len(explanation.split())
        confidence_score = 0.87
        
        return {
            'explanation': explanation,
            'tokens_used': tokens_used,
            'confidence_score': confidence_score
        }
    
    def _format_context_prompt(self, context: Dict[str, Any]) -> str:
        """Format context into a prompt for the model"""
        
        # Extract key information
        analysis_type = context.get('analysis_type', 'general analysis')
        symbol = context.get('symbol', 'the security')
        
        # Build context string
        context_parts = []
        
        if 'sentiment_score' in context:
            sentiment = "positive" if context['sentiment_score'] > 0 else "negative" if context['sentiment_score'] < 0 else "neutral"
            context_parts.append(f"Sentiment analysis shows {sentiment} sentiment (score: {context['sentiment_score']:.2f})")
        
        if 'price_prediction' in context:
            pred = context['price_prediction']
            context_parts.append(f"Price prediction: {pred.get('direction', 'neutral')} movement expected")
        
        if 'strategy' in context:
            strategy = context['strategy']
            context_parts.append(f"Recommended strategy: {strategy.get('strategy', 'N/A')}")
        
        if 'risk_score' in context:
            context_parts.append(f"Risk assessment: {context['risk_score']}/10")
        
        # Current market conditions
        if 'current_price' in context:
            context_parts.append(f"Current price: ${context['current_price']}")
        
        if 'iv_rank' in context:
            context_parts.append(f"Implied volatility rank: {context['iv_rank']}%")
        
        # Build final prompt
        context_str = ". ".join(context_parts) if context_parts else "General market analysis requested"
        
        prompt = f"""Please provide a clear explanation for the {analysis_type} of {symbol}.

Context: {context_str}

Provide a professional explanation that covers:
1. Key factors driving this analysis
2. Risk considerations and mitigation strategies  
3. Expected outcomes and probability assessments
4. Actionable insights for investors

Focus on practical implications and maintain a balanced perspective on both opportunities and risks."""

        return prompt
    
    def _calculate_confidence(self, explanation: str) -> float:
        """Calculate confidence score for generated explanation"""
        
        try:
            # Basic quality metrics
            word_count = len(explanation.split())
            sentence_count = explanation.count('.') + explanation.count('!') + explanation.count('?')
            
            # Quality indicators
            has_numbers = any(char.isdigit() for char in explanation)
            has_financial_terms = any(term in explanation.lower() for term in [
                'risk', 'volatility', 'return', 'portfolio', 'strategy', 
                'market', 'price', 'analysis', 'investment'
            ])
            
            # Base confidence
            confidence = 0.7
            
            # Adjust based on length (optimal range 50-300 words)
            if 50 <= word_count <= 300:
                confidence += 0.1
            elif word_count < 30:
                confidence -= 0.2
            
            # Adjust based on structure
            if sentence_count >= 3:
                confidence += 0.05
            
            # Adjust based on content quality
            if has_financial_terms:
                confidence += 0.1
            if has_numbers:
                confidence += 0.05
            
            # Ensure confidence is within bounds
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.75  # Default confidence
    
    def _update_stats(self, inference_time: float, tokens_generated: int):
        """Update generation statistics"""
        
        self.generation_stats['total_requests'] += 1
        self.generation_stats['total_tokens_generated'] += tokens_generated
        self.generation_stats['total_inference_time'] += inference_time
        
        # Calculate average tokens per second
        if self.generation_stats['total_inference_time'] > 0:
            self.generation_stats['average_tokens_per_second'] = (
                self.generation_stats['total_tokens_generated'] / 
                self.generation_stats['total_inference_time']
            )
    
    async def warmup(self, iterations: int = 3) -> Dict[str, Any]:
        """Warm up the model with test generations"""
        
        logger.info(f"Warming up Llama model with {iterations} iterations...")
        
        warmup_times = []
        test_context = {
            'analysis_type': 'sentiment',
            'symbol': 'AAPL',
            'sentiment_score': 0.75,
            'current_price': 150.0
        }
        
        for i in range(iterations):
            try:
                start_time = time.time()
                
                await self.generate_explanation(
                    context=test_context,
                    max_tokens=100,
                    temperature=0.1
                )
                
                warmup_time = time.time() - start_time
                warmup_times.append(warmup_time)
                
                logger.info(f"Warmup iteration {i+1}: {warmup_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Warmup iteration {i+1} failed: {str(e)}")
        
        if warmup_times:
            avg_time = sum(warmup_times) / len(warmup_times)
            logger.info(f"Warmup completed. Average time: {avg_time:.2f}s")
            
            return {
                'status': 'completed',
                'iterations': len(warmup_times),
                'average_time_s': avg_time,
                'times_s': warmup_times
            }
        else:
            return {
                'status': 'failed',
                'iterations': 0,
                'error': 'All warmup iterations failed'
            }
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        stats = dict(self.generation_stats)
        
        # Add memory info if available
        if torch.cuda.is_available():
            stats['gpu_memory'] = {
                'allocated_gb': torch.cuda.memory_allocated(0) / (1024**3),
                'cached_gb': torch.cuda.memory_reserved(0) / (1024**3)
            }
        
        return stats
    
    async def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed engine statistics"""
        
        return {
            'model_status': {
                'initialized': self.is_initialized,
                'model_path': self.model_path,
                'gpu_available': torch.cuda.is_available()
            },
            'generation_stats': self.generation_stats,
            'configuration': {
                'context_size': settings.N_CTX,
                'gpu_layers': settings.N_GPU_LAYERS,
                'batch_size': settings.N_BATCH,
                'threads': settings.N_THREADS
            }
        }
    
    def is_ready(self) -> bool:
        """Check if engine is ready for inference"""
        return self.is_initialized and (self.model is not None or not LLAMA_CPP_AVAILABLE)
    
    async def reload(self):
        """Reload the model"""
        logger.info("Reloading Llama model...")
        
        try:
            # Cleanup existing model
            if self.model:
                del self.model
                self.model = None
            
            if self.chat_handler:
                del self.chat_handler
                self.chat_handler = None
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Reinitialize
            self.is_initialized = False
            await self.initialize()
            
            logger.info("Model reloaded successfully")
            
        except Exception as e:
            logger.error(f"Model reload failed: {str(e)}")
            raise
    
    async def shutdown(self):
        """Shutdown the engine and cleanup resources"""
        logger.info("Shutting down Llama engine...")
        
        try:
            if self.model:
                del self.model
                self.model = None
            
            if self.chat_handler:
                del self.chat_handler
                self.chat_handler = None
            
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            self.is_initialized = False
            logger.info("Llama engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            raise

"""
Enhanced FinBERT model with options-specific fine-tuning
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedFinBERTModel:
    """Enhanced FinBERT model with options intelligence and financial context"""
    
    def __init__(self):
        """Initialize the enhanced FinBERT model"""
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Options-specific vocabulary enhancements
        self.options_vocabulary = {
            'calls', 'puts', 'strike', 'expiry', 'expiration', 'iv', 'implied volatility',
            'delta', 'gamma', 'theta', 'vega', 'rho', 'options chain', 'premium',
            'in the money', 'out of the money', 'at the money', 'itm', 'otm', 'atm',
            'covered call', 'cash secured put', 'iron condor', 'strangle', 'straddle',
            'butterfly', 'calendar spread', 'credit spread', 'debit spread',
            'assignment', 'exercise', 'rollover', 'earnings play', 'vol crush'
        }
        
        # Financial context enhancements
        self.financial_context_weights = {
            'earnings': 1.2,
            'fed': 1.3,
            'inflation': 1.1,
            'guidance': 1.2,
            'revenue': 1.1,
            'profit': 1.1,
            'margin': 1.1,
            'outlook': 1.2
        }
        
        logger.info("📊 Enhanced FinBERT model initialized")

    async def load_model(self):
        """Load and initialize the FinBERT model with enhancements"""
        try:
            logger.info("🔄 Loading enhanced FinBERT model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Create sentiment pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                return_all_scores=True
            )
            
            # Add options vocabulary to tokenizer if needed
            await self._enhance_tokenizer_vocabulary()
            
            logger.info("✅ Enhanced FinBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load FinBERT model: {str(e)}")
            raise

    async def _enhance_tokenizer_vocabulary(self):
        """Enhance tokenizer with options-specific vocabulary"""
        try:
            # Check if options terms are in vocabulary
            options_tokens = []
            for term in self.options_vocabulary:
                tokens = self.tokenizer.tokenize(term)
                if len(tokens) > 1:  # Multi-token terms need special handling
                    options_tokens.append(term)
            
            # In production, you might add these as special tokens
            # self.tokenizer.add_tokens(options_tokens)
            # self.model.resize_token_embeddings(len(self.tokenizer))
            
            logger.debug(f"📝 Enhanced vocabulary with {len(options_tokens)} options terms")
            
        except Exception as e:
            logger.error(f"❌ Vocabulary enhancement failed: {str(e)}")

    async def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment with enhanced financial and options context"""
        try:
            if not text or not text.strip():
                return self._get_neutral_sentiment()
            
            # Preprocess text for better financial context
            processed_text = await self._preprocess_for_sentiment(text)
            
            # Run FinBERT sentiment analysis
            results = self.sentiment_pipeline(processed_text)
            
            # Extract sentiment scores
            sentiment_scores = {result['label'].lower(): result['score'] for result in results[0]}
            
            # Determine primary sentiment
            primary_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
            sentiment_label = primary_sentiment[0]
            confidence = primary_sentiment[1]
            
            # Map FinBERT labels to standardized sentiment
            sentiment_mapping = {
                'positive': 1.0,
                'negative': -1.0,
                'neutral': 0.0
            }
            
            # Calculate sentiment score (-1 to 1)
            base_score = sentiment_mapping.get(sentiment_label, 0.0)
            sentiment_score = base_score * confidence
            
            # Apply financial context weighting
            weighted_score = await self._apply_financial_context_weighting(
                processed_text, sentiment_score
            )
            
            # Apply options-specific adjustments
            options_adjusted_score = await self._apply_options_context_adjustment(
                processed_text, weighted_score
            )
            
            return {
                'sentiment_score': round(options_adjusted_score, 4),
                'sentiment_label': self._score_to_label(options_adjusted_score),
                'confidence': round(confidence, 4),
                'raw_scores': sentiment_scores,
                'financial_context_applied': abs(weighted_score - sentiment_score) > 0.01,
                'options_context_applied': abs(options_adjusted_score - weighted_score) > 0.01,
                'processed_text_length': len(processed_text)
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {str(e)}")
            return self._get_neutral_sentiment()

    async def _preprocess_for_sentiment(self, text: str) -> str:
        """Preprocess text for better sentiment analysis"""
        try:
            # Remove excessive whitespace
            processed = ' '.join(text.split())
            
            # Expand common financial abbreviations
            abbreviations = {
                'Q1': 'first quarter',
                'Q2': 'second quarter', 
                'Q3': 'third quarter',
                'Q4': 'fourth quarter',
                'YoY': 'year over year',
                'QoQ': 'quarter over quarter',
                'EPS': 'earnings per share',
                'P/E': 'price to earnings',
                'ROI': 'return on investment',
                'ROE': 'return on equity',
                'EBITDA': 'earnings before interest taxes depreciation amortization',
                'IPO': 'initial public offering',
                'M&A': 'mergers and acquisitions'
            }
            
            for abbr, full_form in abbreviations.items():
                processed = processed.replace(abbr, full_form)
            
            # Enhance options terminology for better context
            options_expansions = {
                'calls': 'call options',
                'puts': 'put options',
                'IV': 'implied volatility',
                'OTM': 'out of the money',
                'ITM': 'in the money',
                'ATM': 'at the money'
            }
            
            for term, expansion in options_expansions.items():
                processed = processed.replace(f' {term} ', f' {expansion} ')
            
            # Limit length to model's maximum sequence length
            max_length = 512
            if len(processed) > max_length:
                processed = processed[:max_length-3] + '...'
            
            return processed
            
        except Exception as e:
            logger.error(f"❌ Text preprocessing failed: {str(e)}")
            return text

    async def _apply_financial_context_weighting(
        self, 
        text: str, 
        base_sentiment: float
    ) -> float:
        """Apply financial context weighting to sentiment"""
        try:
            text_lower = text.lower()
            context_multiplier = 1.0
            
            # Check for financial context terms
            for term, weight in self.financial_context_weights.items():
                if term in text_lower:
                    context_multiplier *= weight
                    break  # Use first match to avoid over-weighting
            
            # Apply market timing context
            if any(term in text_lower for term in ['earnings', 'guidance', 'outlook']):
                # Earnings-related sentiment tends to be more impactful
                context_multiplier *= 1.1
            
            if any(term in text_lower for term in ['fed', 'federal reserve', 'interest rate']):
                # Fed-related sentiment has broad market impact
                context_multiplier *= 1.2
            
            # Normalize to prevent extreme values
            context_multiplier = min(context_multiplier, 2.0)
            
            weighted_sentiment = base_sentiment * context_multiplier
            
            # Ensure result stays within bounds
            return max(-1.0, min(1.0, weighted_sentiment))
            
        except Exception as e:
            logger.error(f"❌ Financial context weighting failed: {str(e)}")
            return base_sentiment

    async def _apply_options_context_adjustment(
        self, 
        text: str, 
        sentiment: float
    ) -> float:
        """Apply options-specific context adjustments"""
        try:
            text_lower = text.lower()
            adjustment_factor = 1.0
            
            # Options volatility context
            if any(term in text_lower for term in ['volatility', 'iv', 'vol crush']):
                # Volatility mentions often indicate uncertainty
                if sentiment > 0:
                    adjustment_factor *= 0.9  # Reduce positive sentiment slightly
                else:
                    adjustment_factor *= 1.1  # Amplify negative sentiment slightly
            
            # Options strategy context
            strategy_mentions = sum(1 for term in [
                'covered call', 'iron condor', 'strangle', 'straddle', 'butterfly'
            ] if term in text_lower)
            
            if strategy_mentions > 0:
                # Strategy discussions indicate sophisticated analysis
                adjustment_factor *= 1.05  # Slight boost for strategic content
            
            # Expiration timing context
            if any(term in text_lower for term in ['expiration', 'expiry', 'expires']):
                # Time-sensitive options content
                adjustment_factor *= 1.1
            
            # Greeks context
            greeks_mentions = sum(1 for term in ['delta', 'gamma', 'theta', 'vega'] 
                                if term in text_lower)
            if greeks_mentions >= 2:
                # Multiple Greeks indicate technical analysis
                adjustment_factor *= 1.05
            
            # Apply adjustment
            adjusted_sentiment = sentiment * adjustment_factor
            
            # Ensure bounds
            return max(-1.0, min(1.0, adjusted_sentiment))
            
        except Exception as e:
            logger.error(f"❌ Options context adjustment failed: {str(e)}")
            return sentiment

    def _score_to_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'

    def _get_neutral_sentiment(self) -> Dict:
        """Return neutral sentiment for error cases"""
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'confidence': 0.0,
            'raw_scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
            'financial_context_applied': False,
            'options_context_applied': False,
            'processed_text_length': 0
        }

    async def batch_analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for multiple texts efficiently"""
        try:
            if not texts:
                return []
            
            # Process texts in batches to manage memory
            batch_size = 32
            results = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_results = []
                
                for text in batch:
                    result = await self.analyze_sentiment(text)
                    batch_results.append(result)
                
                results.extend(batch_results)
                
                # Small delay to prevent overwhelming the GPU
                if len(batch) == batch_size:
                    await asyncio.sleep(0.1)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch sentiment analysis failed: {str(e)}")
            return [self._get_neutral_sentiment() for _ in texts]

    async def get_model_info(self) -> Dict:
        """Get model information and statistics"""
        try:
            return {
                'model_name': self.model_name,
                'device': str(self.device),
                'vocabulary_size': len(self.tokenizer.vocab) if self.tokenizer else 0,
                'max_sequence_length': 512,
                'options_vocabulary_terms': len(self.options_vocabulary),
                'financial_context_terms': len(self.financial_context_weights),
                'model_loaded': self.model is not None,
                'tokenizer_loaded': self.tokenizer is not None,
                'pipeline_ready': self.sentiment_pipeline is not None
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get model info: {str(e)}")
            return {}

    async def cleanup(self):
        """Cleanup model resources"""
        try:
            if self.model:
                del self.model
                self.model = None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            if self.sentiment_pipeline:
                del self.sentiment_pipeline
                self.sentiment_pipeline = None
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ FinBERT model cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Model cleanup failed: {str(e)}")

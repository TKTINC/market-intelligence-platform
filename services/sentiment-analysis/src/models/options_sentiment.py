"""
Options-specific sentiment analysis with trading intelligence
"""

import logging
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)

class OptionsSentimentAnalyzer:
    """Advanced options sentiment analysis with market context"""
    
    def __init__(self):
        """Initialize the options sentiment analyzer"""
        self.options_patterns = self._load_options_patterns()
        self.strategy_sentiment_mapping = self._load_strategy_sentiment_mapping()
        self.volatility_sentiment_keywords = self._load_volatility_keywords()
        self.direction_bias_keywords = self._load_direction_bias_keywords()
        
        # Sentiment weights for different content types
        self.content_type_weights = {
            'earnings_analysis': 1.3,
            'technical_analysis': 1.1,
            'options_flow_analysis': 1.4,
            'strategy_discussion': 1.2,
            'market_commentary': 1.0
        }
        
        logger.info("📊 Options sentiment analyzer initialized")

    async def load_model(self):
        """Load options sentiment models and configurations"""
        try:
            logger.info("🔄 Loading options sentiment models...")
            
            # In production, load trained models for options sentiment
            # For now, we'll use rule-based and pattern-based analysis
            
            await self._initialize_sentiment_models()
            
            logger.info("✅ Options sentiment models loaded")
            
        except Exception as e:
            logger.error(f"❌ Failed to load options sentiment models: {str(e)}")
            raise

    async def _initialize_sentiment_models(self):
        """Initialize sentiment analysis models"""
        # Placeholder for model initialization
        # In production, load pre-trained models for:
        # - Options strategy sentiment classification
        # - Volatility expectation prediction
        # - Directional bias detection
        pass

    async def analyze_options_sentiment(
        self, 
        text: str, 
        entities: Dict, 
        base_sentiment: Dict
    ) -> Dict:
        """Analyze options-specific sentiment with entity context"""
        try:
            # Extract options-specific sentiment signals
            options_bias = await self._analyze_options_bias(text, entities)
            volatility_expectation = await self._analyze_volatility_expectation(text, entities)
            strategy_sentiment = await self._analyze_strategy_sentiment(text, entities)
            
            # Calculate relevance score
            relevance_score = await self._calculate_options_relevance(text, entities)
            
            # Adjust base sentiment with options context
            adjusted_sentiment = await self._adjust_sentiment_with_options_context(
                base_sentiment, options_bias, volatility_expectation
            )
            
            return {
                'options_bias': options_bias,
                'volatility_expectation': volatility_expectation,
                'strategy_sentiment': strategy_sentiment,
                'relevance_score': relevance_score,
                'adjusted_sentiment_score': adjusted_sentiment['score'],
                'sentiment_label': adjusted_sentiment['label'],
                'confidence': adjusted_sentiment['confidence']
            }
            
        except Exception as e:
            logger.error(f"❌ Options sentiment analysis failed: {str(e)}")
            return self._get_neutral_options_sentiment()

    async def analyze_social_sentiment(
        self, 
        text: str, 
        entities: Dict, 
        base_sentiment: Dict,
        engagement_metrics: Dict
    ) -> Dict:
        """Analyze social media sentiment with engagement weighting"""
        try:
            # Get base options sentiment
            options_sentiment = await self.analyze_options_sentiment(text, entities, base_sentiment)
            
            # Calculate social amplification factor
            amplification_factor = await self._calculate_social_amplification(
                engagement_metrics, len(text)
            )
            
            # Calculate engagement score
            engagement_score = await self._calculate_engagement_score(engagement_metrics)
            
            # Calculate viral potential
            viral_potential = await self._calculate_viral_potential(
                engagement_metrics, options_sentiment['relevance_score']
            )
            
            # Adjust sentiment based on social context
            adjusted_score = base_sentiment['sentiment_score'] * amplification_factor
            
            return {
                **options_sentiment,
                'adjusted_sentiment_score': max(-1.0, min(1.0, adjusted_score)),
                'amplification_factor': amplification_factor,
                'engagement_score': engagement_score,
                'viral_potential': viral_potential,
                'social_context_applied': True
            }
            
        except Exception as e:
            logger.error(f"❌ Social sentiment analysis failed: {str(e)}")
            return self._get_neutral_options_sentiment()

    async def analyze_comprehensive_options_sentiment(
        self, 
        text: str, 
        entities: Dict,
        market_context: Dict
    ) -> Dict:
        """Comprehensive options sentiment analysis for forum content"""
        try:
            # Analyze multiple dimensions of options sentiment
            options_sentiment_score = await self._calculate_options_sentiment_score(text, entities)
            volatility_sentiment = await self._analyze_volatility_sentiment(text, entities)
            direction_bias = await self._analyze_detailed_direction_bias(text, entities)
            strategy_sentiment = await self._analyze_comprehensive_strategy_sentiment(text, entities)
            
            # Market context analysis
            iv_sentiment = await self._analyze_iv_sentiment(text, market_context)
            theta_sentiment = await self._analyze_theta_sentiment(text, entities)
            earnings_bias = await self._analyze_earnings_proximity_bias(text, market_context)
            
            # Author and source credibility analysis
            author_credibility = await self._assess_author_credibility(text, entities)
            source_reliability = await self._assess_source_reliability(text)
            
            # Calculate overall confidence
            confidence = await self._calculate_comprehensive_confidence(
                entities, author_credibility, source_reliability
            )
            
            return {
                'options_sentiment_score': options_sentiment_score,
                'volatility_sentiment': volatility_sentiment,
                'direction_bias': direction_bias,
                'strategy_sentiment': strategy_sentiment,
                'iv_sentiment': iv_sentiment,
                'theta_sentiment': theta_sentiment,
                'earnings_bias': earnings_bias,
                'author_credibility': author_credibility,
                'source_reliability': source_reliability,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Comprehensive options sentiment analysis failed: {str(e)}")
            return self._get_neutral_comprehensive_sentiment()

    async def _analyze_options_bias(self, text: str, entities: Dict) -> float:
        """Analyze overall options bias in the content"""
        try:
            text_lower = text.lower()
            bias_score = 0.0
            
            # Check for bullish options indicators
            bullish_indicators = [
                'buying calls', 'call volume', 'call sweep', 'bullish spread',
                'upside potential', 'breakout above', 'resistance break'
            ]
            
            # Check for bearish options indicators
            bearish_indicators = [
                'buying puts', 'put volume', 'put sweep', 'bearish spread',
                'downside risk', 'breakdown below', 'support break'
            ]
            
            # Count indicators
            bullish_count = sum(1 for indicator in bullish_indicators if indicator in text_lower)
            bearish_count = sum(1 for indicator in bearish_indicators if indicator in text_lower)
            
            # Calculate bias based on indicator balance
            if bullish_count > bearish_count:
                bias_score = min(0.8, (bullish_count - bearish_count) * 0.2)
            elif bearish_count > bullish_count:
                bias_score = max(-0.8, -(bearish_count - bullish_count) * 0.2)
            
            # Adjust based on entities
            if entities.get('call_mentions', 0) > entities.get('put_mentions', 0):
                bias_score += 0.1
            elif entities.get('put_mentions', 0) > entities.get('call_mentions', 0):
                bias_score -= 0.1
            
            return round(max(-1.0, min(1.0, bias_score)), 3)
            
        except Exception as e:
            logger.error(f"❌ Options bias analysis failed: {str(e)}")
            return 0.0

    async def _analyze_volatility_expectation(self, text: str, entities: Dict) -> Dict:
        """Analyze volatility expectations from options content"""
        try:
            text_lower = text.lower()
            vol_expectation = {
                'direction': 'neutral',
                'magnitude': 0.0,
                'confidence': 0.0
            }
            
            # High volatility keywords
            high_vol_keywords = [
                'vol spike', 'volatility crush', 'iv expansion', 'big move expected',
                'earnings volatility', 'vol pop', 'high iv', 'volatility surge'
            ]
            
            # Low volatility keywords
            low_vol_keywords = [
                'vol crush', 'iv contraction', 'low volatility', 'sideways movement',
                'range bound', 'vol collapse', 'quiet market'
            ]
            
            # Count volatility indicators
            high_vol_count = sum(1 for keyword in high_vol_keywords if keyword in text_lower)
            low_vol_count = sum(1 for keyword in low_vol_keywords if keyword in text_lower)
            
            if high_vol_count > low_vol_count:
                vol_expectation['direction'] = 'increasing'
                vol_expectation['magnitude'] = min(1.0, high_vol_count * 0.3)
                vol_expectation['confidence'] = min(0.9, high_vol_count * 0.2)
            elif low_vol_count > high_vol_count:
                vol_expectation['direction'] = 'decreasing'
                vol_expectation['magnitude'] = min(1.0, low_vol_count * 0.3)
                vol_expectation['confidence'] = min(0.9, low_vol_count * 0.2)
            
            # Check for specific IV mentions in entities
            if entities.get('iv_mentions'):
                vol_expectation['confidence'] = min(1.0, vol_expectation['confidence'] + 0.2)
            
            return vol_expectation
            
        except Exception as e:
            logger.error(f"❌ Volatility expectation analysis failed: {str(e)}")
            return {'direction': 'neutral', 'magnitude': 0.0, 'confidence': 0.0}

    async def _analyze_strategy_sentiment(self, text: str, entities: Dict) -> Dict:
        """Analyze sentiment around options strategies"""
        try:
            strategies_mentioned = entities.get('strategies', [])
            strategy_sentiment = {}
            
            for strategy in strategies_mentioned:
                sentiment_score = await self._get_strategy_sentiment_score(text, strategy)
                strategy_sentiment[strategy] = {
                    'sentiment_score': sentiment_score,
                    'confidence': 0.7 if abs(sentiment_score) > 0.1 else 0.3
                }
            
            # Calculate overall strategy sentiment
            if strategy_sentiment:
                avg_sentiment = np.mean([s['sentiment_score'] for s in strategy_sentiment.values()])
                avg_confidence = np.mean([s['confidence'] for s in strategy_sentiment.values()])
            else:
                avg_sentiment = 0.0
                avg_confidence = 0.0
            
            return {
                'individual_strategies': strategy_sentiment,
                'overall_sentiment': round(avg_sentiment, 3),
                'overall_confidence': round(avg_confidence, 3),
                'strategies_count': len(strategies_mentioned)
            }
            
        except Exception as e:
            logger.error(f"❌ Strategy sentiment analysis failed: {str(e)}")
            return {'individual_strategies': {}, 'overall_sentiment': 0.0, 'overall_confidence': 0.0, 'strategies_count': 0}

    async def _get_strategy_sentiment_score(self, text: str, strategy: str) -> float:
        """Get sentiment score for a specific strategy"""
        try:
            strategy_lower = strategy.lower()
            text_lower = text.lower()
            
            # Find strategy mentions and surrounding context
            strategy_contexts = []
            start_idx = 0
            
            while True:
                idx = text_lower.find(strategy_lower, start_idx)
                if idx == -1:
                    break
                
                # Extract context around strategy mention
                context_start = max(0, idx - 50)
                context_end = min(len(text), idx + len(strategy) + 50)
                context = text_lower[context_start:context_end]
                strategy_contexts.append(context)
                
                start_idx = idx + 1
            
            if not strategy_contexts:
                return 0.0
            
            # Analyze sentiment in each context
            sentiment_scores = []
            for context in strategy_contexts:
                score = await self._analyze_context_sentiment(context)
                sentiment_scores.append(score)
            
            return np.mean(sentiment_scores) if sentiment_scores else 0.0
            
        except Exception as e:
            logger.error(f"❌ Strategy sentiment scoring failed: {str(e)}")
            return 0.0

    async def _analyze_context_sentiment(self, context: str) -> float:
        """Analyze sentiment in a given context"""
        try:
            positive_words = [
                'profit', 'gain', 'successful', 'winning', 'profitable', 'good',
                'excellent', 'optimal', 'perfect', 'ideal', 'best', 'recommend'
            ]
            
            negative_words = [
                'loss', 'losing', 'bad', 'poor', 'terrible', 'avoid', 'risky',
                'dangerous', 'failed', 'unsuccessful', 'worst', 'mistake'
            ]
            
            positive_count = sum(1 for word in positive_words if word in context)
            negative_count = sum(1 for word in negative_words if word in context)
            
            if positive_count > negative_count:
                return min(0.8, (positive_count - negative_count) * 0.2)
            elif negative_count > positive_count:
                return max(-0.8, -(negative_count - positive_count) * 0.2)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"❌ Context sentiment analysis failed: {str(e)}")
            return 0.0

    async def _calculate_options_relevance(self, text: str, entities: Dict) -> float:
        """Calculate how relevant the content is to options trading"""
        try:
            relevance_score = 0.0
            
            # Base relevance from entities
            symbols_count = len(entities.get('symbols', []))
            strategies_count = len(entities.get('strategies', []))
            strikes_count = len(entities.get('strikes', []))
            
            relevance_score += min(0.3, symbols_count * 0.1)
            relevance_score += min(0.4, strategies_count * 0.2)
            relevance_score += min(0.2, strikes_count * 0.05)
            
            # Additional relevance from options terminology
            options_terms = [
                'options', 'calls', 'puts', 'strike', 'expiry', 'premium',
                'delta', 'gamma', 'theta', 'vega', 'implied volatility'
            ]
            
            text_lower = text.lower()
            term_matches = sum(1 for term in options_terms if term in text_lower)
            relevance_score += min(0.3, term_matches * 0.05)
            
            # Boost for detailed options analysis
            if any(phrase in text_lower for phrase in [
                'options chain', 'options flow', 'unusual activity', 'options volume'
            ]):
                relevance_score += 0.2
            
            return min(1.0, relevance_score)
            
        except Exception as e:
            logger.error(f"❌ Options relevance calculation failed: {str(e)}")
            return 0.0

    async def _adjust_sentiment_with_options_context(
        self, 
        base_sentiment: Dict, 
        options_bias: float, 
        volatility_expectation: Dict
    ) -> Dict:
        """Adjust base sentiment with options-specific context"""
        try:
            base_score = base_sentiment.get('sentiment_score', 0.0)
            base_confidence = base_sentiment.get('confidence', 0.0)
            
            # Apply options bias adjustment
            bias_adjustment = options_bias * 0.3  # Moderate influence
            adjusted_score = base_score + bias_adjustment
            
            # Apply volatility context
            vol_direction = volatility_expectation.get('direction', 'neutral')
            vol_magnitude = volatility_expectation.get('magnitude', 0.0)
            
            if vol_direction == 'increasing' and vol_magnitude > 0.5:
                # High volatility expectation can amplify sentiment
                adjusted_score *= (1.0 + vol_magnitude * 0.2)
            elif vol_direction == 'decreasing':
                # Low volatility expectation can dampen sentiment
                adjusted_score *= (1.0 - vol_magnitude * 0.1)
            
            # Ensure bounds
            adjusted_score = max(-1.0, min(1.0, adjusted_score))
            
            # Adjust confidence based on options context strength
            options_context_strength = abs(options_bias) + volatility_expectation.get('confidence', 0.0)
            confidence_boost = min(0.2, options_context_strength * 0.1)
            adjusted_confidence = min(1.0, base_confidence + confidence_boost)
            
            # Determine new label
            if adjusted_score > 0.1:
                label = 'positive'
            elif adjusted_score < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'score': round(adjusted_score, 4),
                'label': label,
                'confidence': round(adjusted_confidence, 4)
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment adjustment failed: {str(e)}")
            return {
                'score': base_sentiment.get('sentiment_score', 0.0),
                'label': base_sentiment.get('sentiment_label', 'neutral'),
                'confidence': base_sentiment.get('confidence', 0.0)
            }

    async def _calculate_social_amplification(
        self, 
        engagement_metrics: Dict, 
        text_length: int
    ) -> float:
        """Calculate social media amplification factor"""
        try:
            # Base amplification
            amplification = 1.0
            
            # Engagement-based amplification
            likes = engagement_metrics.get('likes', 0)
            retweets = engagement_metrics.get('retweets', 0)
            replies = engagement_metrics.get('replies', 0)
            
            # Calculate engagement score
            engagement_score = likes + (retweets * 3) + (replies * 2)
            
            # Apply amplification based on engagement
            if engagement_score > 1000:
                amplification *= 1.5
            elif engagement_score > 100:
                amplification *= 1.2
            elif engagement_score > 10:
                amplification *= 1.1
            
            # Content length factor
            if text_length > 200:  # Longer, more detailed content
                amplification *= 1.1
            elif text_length < 50:  # Short content might be less reliable
                amplification *= 0.9
            
            return min(2.0, amplification)  # Cap at 2x amplification
            
        except Exception as e:
            logger.error(f"❌ Social amplification calculation failed: {str(e)}")
            return 1.0

    async def _calculate_engagement_score(self, engagement_metrics: Dict) -> float:
        """Calculate normalized engagement score"""
        try:
            likes = engagement_metrics.get('likes', 0)
            retweets = engagement_metrics.get('retweets', 0)
            replies = engagement_metrics.get('replies', 0)
            views = engagement_metrics.get('views', 1)  # Avoid division by zero
            
            # Calculate engagement rate
            total_engagement = likes + retweets + replies
            engagement_rate = total_engagement / max(views, 1)
            
            # Normalize to 0-1 scale
            normalized_score = min(1.0, engagement_rate * 100)
            
            return round(normalized_score, 4)
            
        except Exception as e:
            logger.error(f"❌ Engagement score calculation failed: {str(e)}")
            return 0.0

    async def _calculate_viral_potential(
        self, 
        engagement_metrics: Dict, 
        options_relevance: float
    ) -> float:
        """Calculate viral potential of the content"""
        try:
            # Base viral potential from engagement velocity
            retweets = engagement_metrics.get('retweets', 0)
            likes = engagement_metrics.get('likes', 0)
            time_posted = engagement_metrics.get('time_posted_hours_ago', 24)
            
            # Calculate engagement velocity (engagement per hour)
            engagement_velocity = (retweets * 2 + likes) / max(time_posted, 1)
            
            # Viral potential based on velocity
            if engagement_velocity > 100:
                viral_potential = 0.9
            elif engagement_velocity > 50:
                viral_potential = 0.7
            elif engagement_velocity > 10:
                viral_potential = 0.5
            elif engagement_velocity > 1:
                viral_potential = 0.3
            else:
                viral_potential = 0.1
            
            # Boost for options-relevant content
            viral_potential *= (1.0 + options_relevance * 0.5)
            
            return min(1.0, viral_potential)
            
        except Exception as e:
            logger.error(f"❌ Viral potential calculation failed: {str(e)}")
            return 0.0

    # Additional comprehensive analysis methods...

    async def _calculate_options_sentiment_score(self, text: str, entities: Dict) -> float:
        """Calculate comprehensive options sentiment score"""
        try:
            # Implementation for detailed options sentiment scoring
            # This would include analysis of specific options terminology,
            # strategy implications, and market context
            return 0.0  # Placeholder
        except Exception as e:
            logger.error(f"❌ Options sentiment score calculation failed: {str(e)}")
            return 0.0

    async def _analyze_volatility_sentiment(self, text: str, entities: Dict) -> float:
        """Analyze sentiment specifically around volatility"""
        # Implementation details...
        return 0.0

    async def _analyze_detailed_direction_bias(self, text: str, entities: Dict) -> str:
        """Analyze detailed directional bias"""
        # Implementation details...
        return 'neutral'

    async def _analyze_comprehensive_strategy_sentiment(self, text: str, entities: Dict) -> Dict:
        """Comprehensive strategy sentiment analysis"""
        # Implementation details...
        return {}

    async def _analyze_iv_sentiment(self, text: str, market_context: Dict) -> float:
        """Analyze implied volatility sentiment"""
        # Implementation details...
        return 0.0

    async def _analyze_theta_sentiment(self, text: str, entities: Dict) -> float:
        """Analyze time decay sentiment"""
        # Implementation details...
        return 0.0

    async def _analyze_earnings_proximity_bias(self, text: str, market_context: Dict) -> float:
        """Analyze bias related to earnings proximity"""
        # Implementation details...
        return 0.0

    async def _assess_author_credibility(self, text: str, entities: Dict) -> float:
        """Assess author credibility based on content analysis"""
        # Implementation details...
        return 0.5

    async def _assess_source_reliability(self, text: str) -> float:
        """Assess source reliability"""
        # Implementation details...
        return 0.5

    async def _calculate_comprehensive_confidence(
        self, 
        entities: Dict, 
        author_credibility: float, 
        source_reliability: float
    ) -> float:
        """Calculate comprehensive confidence score"""
        # Implementation details...
        return 0.5

    def _load_options_patterns(self) -> Dict:
        """Load options-specific patterns for analysis"""
        return {
            'call_patterns': [r'\bcalls?\b', r'\bcall options?\b'],
            'put_patterns': [r'\bputs?\b', r'\bput options?\b'],
            'strategy_patterns': [
                r'\bcovered call\b', r'\biron condor\b', r'\bstrangle\b',
                r'\bstraddle\b', r'\bbutterfly\b', r'\bcredit spread\b'
            ]
        }

    def _load_strategy_sentiment_mapping(self) -> Dict:
        """Load strategy-specific sentiment mappings"""
        return {
            'covered_call': {'bias': 'neutral_to_bullish', 'volatility': 'low'},
            'iron_condor': {'bias': 'neutral', 'volatility': 'low'},
            'strangle': {'bias': 'neutral', 'volatility': 'high'},
            'straddle': {'bias': 'neutral', 'volatility': 'high'}
        }

    def _load_volatility_keywords(self) -> Dict:
        """Load volatility-related keywords"""
        return {
            'high_vol': ['spike', 'surge', 'explosion', 'crush', 'pop'],
            'low_vol': ['collapse', 'contraction', 'calm', 'quiet', 'stable']
        }

    def _load_direction_bias_keywords(self) -> Dict:
        """Load directional bias keywords"""
        return {
            'bullish': ['bull', 'up', 'rise', 'rally', 'breakout', 'moon'],
            'bearish': ['bear', 'down', 'fall', 'crash', 'breakdown', 'dump']
        }

    def _get_neutral_options_sentiment(self) -> Dict:
        """Return neutral options sentiment"""
        return {
            'options_bias': 0.0,
            'volatility_expectation': {'direction': 'neutral', 'magnitude': 0.0, 'confidence': 0.0},
            'strategy_sentiment': {'individual_strategies': {}, 'overall_sentiment': 0.0, 'overall_confidence': 0.0, 'strategies_count': 0},
            'relevance_score': 0.0,
            'adjusted_sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'confidence': 0.0
        }

    def _get_neutral_comprehensive_sentiment(self) -> Dict:
        """Return neutral comprehensive sentiment"""
        return {
            'options_sentiment_score': 0.0,
            'volatility_sentiment': 0.0,
            'direction_bias': 'neutral',
            'strategy_sentiment': {},
            'iv_sentiment': 0.0,
            'theta_sentiment': 0.0,
            'earnings_bias': 0.0,
            'author_credibility': 0.5,
            'source_reliability': 0.5,
            'confidence': 0.0
        }

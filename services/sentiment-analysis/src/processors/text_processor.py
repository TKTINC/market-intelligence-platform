"""
Enhanced text processing for financial and options content
"""

import logging
import re
import string
from typing import Dict, List, Optional, Tuple
import asyncio
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

class EnhancedTextProcessor:
    """Enhanced text processor for financial and options content"""
    
    def __init__(self):
        """Initialize the enhanced text processor"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.warning(f"⚠️  NLTK download warning: {str(e)}")
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Financial-specific stop words to remove
        self.financial_stop_words = {
            'stock', 'market', 'trading', 'trader', 'invest', 'investor',
            'price', 'share', 'equity', 'security'
        }
        
        # Important financial terms to preserve
        self.preserve_terms = {
            'bull', 'bear', 'bullish', 'bearish', 'volatile', 'volatility',
            'calls', 'puts', 'option', 'options', 'strike', 'expiry',
            'delta', 'gamma', 'theta', 'vega', 'premium'
        }
        
        # Social media specific patterns
        self.social_patterns = {
            'hashtags': re.compile(r'#\w+'),
            'mentions': re.compile(r'@\w+'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'tickers': re.compile(r'\$[A-Z]{1,5}'),
            'emojis': re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+')
        }
        
        # Financial abbreviation expansions
        self.abbreviation_expansions = {
            'DD': 'due diligence',
            'YOLO': 'you only live once',
            'FOMO': 'fear of missing out',
            'FUD': 'fear uncertainty doubt',
            'HODLing': 'holding',
            'ATH': 'all time high',
            'ATL': 'all time low',
            'TA': 'technical analysis',
            'FA': 'fundamental analysis'
        }
        
        logger.info("📝 Enhanced text processor initialized")

    async def preprocess_news_content(
        self, 
        title: str, 
        content: str, 
        source: str
    ) -> Optional[str]:
        """Preprocess news content for sentiment analysis"""
        try:
            if not title and not content:
                return None
            
            # Combine title and content with appropriate weighting
            combined_text = ""
            if title:
                # Title gets more weight in news analysis
                combined_text += f"{title}. {title}. "  # Duplicate for emphasis
            
            if content:
                # Clean and truncate content
                cleaned_content = await self._clean_news_content(content)
                combined_text += cleaned_content
            
            # Basic preprocessing
            processed_text = await self._basic_preprocessing(combined_text)
            
            # News-specific processing
            processed_text = await self._news_specific_processing(processed_text, source)
            
            # Validate minimum length
            if len(processed_text.strip()) < 20:
                return None
            
            return processed_text
            
        except Exception as e:
            logger.error(f"❌ News content preprocessing failed: {str(e)}")
            return None

    async def preprocess_social_content(
        self, 
        text: str, 
        platform: str,
        engagement_metrics: Dict
    ) -> Optional[str]:
        """Preprocess social media content"""
        try:
            if not text or len(text.strip()) < 10:
                return None
            
            # Platform-specific preprocessing
            if platform.lower() in ['twitter', 'x']:
                processed_text = await self._preprocess_twitter_content(text)
            elif platform.lower() == 'reddit':
                processed_text = await self._preprocess_reddit_content(text)
            else:
                processed_text = await self._preprocess_generic_social_content(text)
            
            # Apply engagement-based filtering
            if not await self._passes_engagement_filter(processed_text, engagement_metrics):
                return None
            
            # Basic preprocessing
            processed_text = await self._basic_preprocessing(processed_text)
            
            # Validate result
            if len(processed_text.strip()) < 15:
                return None
            
            return processed_text
            
        except Exception as e:
            logger.error(f"❌ Social content preprocessing failed: {str(e)}")
            return None

    async def preprocess_options_content(
        self, 
        text: str, 
        title: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Optional[str]:
        """Preprocess options-specific content from forums and analysis"""
        try:
            if not text:
                return None
            
            # Combine title and text for options content
            combined_text = ""
            if title:
                combined_text += f"{title}. "
            combined_text += text
            
            # Options-specific cleaning
            processed_text = await self._clean_options_content(combined_text)
            
            # Expand options-specific abbreviations
            processed_text = await self._expand_options_abbreviations(processed_text)
            
            # Basic preprocessing
            processed_text = await self._basic_preprocessing(processed_text)
            
            # Options-specific enhancements
            processed_text = await self._enhance_options_terminology(processed_text)
            
            # Validate length and content quality
            if len(processed_text.strip()) < 25:
                return None
            
            return processed_text
            
        except Exception as e:
            logger.error(f"❌ Options content preprocessing failed: {str(e)}")
            return None

    async def calculate_financial_relevance(self, text: str) -> float:
        """Calculate how relevant the text is to financial markets"""
        try:
            if not text:
                return 0.0
            
            text_lower = text.lower()
            relevance_score = 0.0
            
            # Financial keywords
            financial_keywords = {
                'stock', 'stocks', 'market', 'trading', 'investment', 'investor',
                'earnings', 'revenue', 'profit', 'loss', 'dividend', 'yield',
                'bull', 'bear', 'rally', 'crash', 'volatility', 'volume'
            }
            
            # Options keywords
            options_keywords = {
                'options', 'calls', 'puts', 'strike', 'expiry', 'premium',
                'delta', 'gamma', 'theta', 'vega', 'iv', 'implied volatility'
            }
            
            # Economic keywords
            economic_keywords = {
                'fed', 'federal reserve', 'interest rate', 'inflation', 'gdp',
                'unemployment', 'economic', 'economy', 'fiscal', 'monetary'
            }
            
            # Count keyword matches
            financial_matches = sum(1 for keyword in financial_keywords if keyword in text_lower)
            options_matches = sum(1 for keyword in options_keywords if keyword in text_lower)
            economic_matches = sum(1 for keyword in economic_keywords if keyword in text_lower)
            
            # Calculate relevance components
            relevance_score += min(0.4, financial_matches * 0.05)
            relevance_score += min(0.4, options_matches * 0.08)  # Options get higher weight
            relevance_score += min(0.3, economic_matches * 0.06)
            
            # Check for ticker symbols
            ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')
            tickers = ticker_pattern.findall(text)
            ticker_score = min(0.3, len(tickers) * 0.1)
            relevance_score += ticker_score
            
            # Check for financial numbers (prices, percentages)
            number_patterns = [
                r'\$\d+\.?\d*',  # Dollar amounts
                r'\d+\.?\d*%',   # Percentages
                r'\d+\.?\d*\s*(basis|bp|bps)',  # Basis points
            ]
            
            number_matches = sum(len(re.findall(pattern, text)) for pattern in number_patterns)
            relevance_score += min(0.2, number_matches * 0.05)
            
            return min(1.0, relevance_score)
            
        except Exception as e:
            logger.error(f"❌ Financial relevance calculation failed: {str(e)}")
            return 0.0

    async def _clean_news_content(self, content: str) -> str:
        """Clean news article content"""
        try:
            # Remove HTML tags
            content = re.sub(r'<[^>]+>', '', content)
            
            # Remove URLs
            content = re.sub(r'http[s]?://\S+', '', content)
            
            # Remove email addresses
            content = re.sub(r'\S+@\S+', '', content)
            
            # Remove excessive whitespace
            content = re.sub(r'\s+', ' ', content)
            
            # Remove common news disclaimers and boilerplate
            disclaimer_patterns = [
                r'This article was originally published.*',
                r'For more information.*',
                r'To read more.*',
                r'Subscribe to.*',
                r'Follow us on.*'
            ]
            
            for pattern in disclaimer_patterns:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE)
            
            # Limit length for processing efficiency
            max_length = 2000
            if len(content) > max_length:
                # Try to cut at sentence boundary
                sentences = sent_tokenize(content[:max_length])
                if sentences:
                    content = ' '.join(sentences[:-1])  # Remove potentially incomplete last sentence
                else:
                    content = content[:max_length]
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"❌ News content cleaning failed: {str(e)}")
            return content

    async def _preprocess_twitter_content(self, text: str) -> str:
        """Preprocess Twitter/X content"""
        try:
            # Extract and preserve ticker symbols
            tickers = self.social_patterns['tickers'].findall(text)
            ticker_text = ' '.join(tickers)
            
            # Remove URLs but preserve ticker information
            text = self.social_patterns['urls'].sub('', text)
            
            # Process hashtags - extract meaningful ones
            hashtags = self.social_patterns['hashtags'].findall(text)
            financial_hashtags = [tag for tag in hashtags if self._is_financial_hashtag(tag)]
            hashtag_text = ' '.join(tag[1:] for tag in financial_hashtags)  # Remove #
            
            # Remove hashtags and mentions from main text
            text = self.social_patterns['hashtags'].sub('', text)
            text = self.social_patterns['mentions'].sub('', text)
            
            # Handle emojis - some have financial meaning
            emoji_sentiment = self._extract_emoji_sentiment(text)
            text = self.social_patterns['emojis'].sub('', text)
            
            # Combine processed components
            processed_parts = [text, ticker_text, hashtag_text]
            if emoji_sentiment:
                processed_parts.append(emoji_sentiment)
            
            processed_text = ' '.join(part for part in processed_parts if part.strip())
            
            return processed_text.strip()
            
        except Exception as e:
            logger.error(f"❌ Twitter preprocessing failed: {str(e)}")
            return text

    async def _preprocess_reddit_content(self, text: str) -> str:
        """Preprocess Reddit content"""
        try:
            # Reddit-specific patterns
            # Remove Reddit formatting
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
            text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
            text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
            
            # Remove quotes (> text)
            text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)
            
            # Remove URLs
            text = re.sub(r'http[s]?://\S+', '', text)
            text = re.sub(r'www\.\S+', '', text)
            
            # Remove subreddit references
            text = re.sub(r'r/\w+', '', text)
            
            # Remove user references but preserve context
            text = re.sub(r'u/\w+', 'someone', text)
            
            # Handle Reddit slang
            reddit_slang = {
                'hodl': 'hold',
                'stonks': 'stocks',
                'tendies': 'profits',
                'diamond hands': 'strong holding',
                'paper hands': 'weak holding',
                'apes': 'retail investors',
                'moon': 'significant increase',
                'rocket': 'rapid increase'
            }
            
            for slang, replacement in reddit_slang.items():
                text = re.sub(rf'\b{slang}\b', replacement, text, flags=re.IGNORECASE)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"❌ Reddit preprocessing failed: {str(e)}")
            return text

    async def _preprocess_generic_social_content(self, text: str) -> str:
        """Preprocess generic social media content"""
        try:
            # Remove URLs
            text = re.sub(r'http[s]?://\S+', '', text)
            
            # Remove excessive punctuation
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            
            # Remove excessive capitalization
            text = re.sub(r'\b[A-Z]{3,}\b', lambda m: m.group().lower(), text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"❌ Generic social preprocessing failed: {str(e)}")
            return text

    async def _basic_preprocessing(self, text: str) -> str:
        """Apply basic text preprocessing"""
        try:
            # Convert to lowercase but preserve important terms
            preserved_terms = []
            for term in self.preserve_terms:
                if term in text:
                    preserved_terms.append(term)
            
            # Convert to lowercase
            text = text.lower()
            
            # Restore preserved terms
            for term in preserved_terms:
                text = text.replace(term.lower(), term)
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove leading/trailing whitespace
            text = text.strip()
            
            # Expand contractions
            contractions = {
                "won't": "will not",
                "can't": "cannot",
                "n't": " not",
                "'re": " are",
                "'ve": " have",
                "'ll": " will",
                "'d": " would"
            }
            
            for contraction, expansion in contractions.items():
                text = text.replace(contraction, expansion)
            
            return text
            
        except Exception as e:
            logger.error(f"❌ Basic preprocessing failed: {str(e)}")
            return text

    async def _news_specific_processing(self, text: str, source: str) -> str:
        """Apply news-specific processing"""
        try:
            # Source-specific adjustments
            if source and 'reuters' in source.lower():
                # Reuters tends to be factual, boost neutral language
                pass
            elif source and any(term in source.lower() for term in ['marketwatch', 'bloomberg', 'cnbc']):
                # Financial news sources, enhance financial terminology
                pass
            
            # Remove common news phrases that don't add sentiment value
            news_noise = [
                'according to reports',
                'sources familiar with the matter',
                'company spokesperson said',
                'in a statement',
                'during a conference call'
            ]
            
            for phrase in news_noise:
                text = text.replace(phrase, '')
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"❌ News-specific processing failed: {str(e)}")
            return text

    async def _clean_options_content(self, text: str) -> str:
        """Clean options-specific content"""
        try:
            # Remove common forum noise
            forum_noise = [
                'not financial advice',
                'do your own dd',
                'this is not investment advice',
                'obligatory rocket emojis'
            ]
            
            text_lower = text.lower()
            for noise in forum_noise:
                text_lower = text_lower.replace(noise, '')
            
            # Restore case for preserved terms
            for term in self.preserve_terms:
                if term.lower() in text_lower:
                    text_lower = text_lower.replace(term.lower(), term)
            
            return text_lower.strip()
            
        except Exception as e:
            logger.error(f"❌ Options content cleaning failed: {str(e)}")
            return text

    async def _expand_options_abbreviations(self, text: str) -> str:
        """Expand options-specific abbreviations"""
        try:
            # Options-specific abbreviations
            options_abbreviations = {
                'CC': 'covered call',
                'CSP': 'cash secured put',
                'IC': 'iron condor',
                'DTE': 'days to expiration',
                'OTM': 'out of the money',
                'ITM': 'in the money',
                'ATM': 'at the money'
            }
            
            # Apply general abbreviations
            for abbr, expansion in self.abbreviation_expansions.items():
                text = re.sub(rf'\b{abbr}\b', expansion, text, flags=re.IGNORECASE)
            
            # Apply options abbreviations
            for abbr, expansion in options_abbreviations.items():
                text = re.sub(rf'\b{abbr}\b', expansion, text, flags=re.IGNORECASE)
            
            return text
            
        except Exception as e:
            logger.error(f"❌ Options abbreviation expansion failed: {str(e)}")
            return text

    async def _enhance_options_terminology(self, text: str) -> str:
        """Enhance options terminology for better analysis"""
        try:
            # Add context to options terms
            enhancements = {
                r'\bcalls\b': 'call options',
                r'\bputs\b': 'put options',
                r'\biv\b': 'implied volatility',
                r'\bgreeks\b': 'options greeks'
            }
            
            for pattern, replacement in enhancements.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
            return text
            
        except Exception as e:
            logger.error(f"❌ Options terminology enhancement failed: {str(e)}")
            return text

    async def _passes_engagement_filter(
        self, 
        text: str, 
        engagement_metrics: Dict
    ) -> bool:
        """Check if content passes engagement filtering"""
        try:
            # Minimum engagement thresholds
            min_likes = 5
            min_total_engagement = 10
            
            likes = engagement_metrics.get('likes', 0)
            retweets = engagement_metrics.get('retweets', 0)
            replies = engagement_metrics.get('replies', 0)
            
            total_engagement = likes + retweets + replies
            
            # Pass if meets minimum engagement OR has high financial relevance
            if total_engagement >= min_total_engagement or likes >= min_likes:
                return True
            
            # Check financial relevance as alternative filter
            financial_relevance = await self.calculate_financial_relevance(text)
            return financial_relevance >= 0.7
            
        except Exception as e:
            logger.error(f"❌ Engagement filtering failed: {str(e)}")
            return True  # Default to allowing content

    def _is_financial_hashtag(self, hashtag: str) -> bool:
        """Check if hashtag is financially relevant"""
        financial_hashtags = {
            '#stocks', '#trading', '#options', '#investing', '#market',
            '#earnings', '#fed', '#crypto', '#forex', '#commodities'
        }
        return hashtag.lower() in financial_hashtags

    def _extract_emoji_sentiment(self, text: str) -> Optional[str]:
        """Extract sentiment from emojis in financial context"""
        # Simple emoji sentiment mapping
        positive_emojis = ['🚀', '📈', '💎', '🔥', '💰', '🤑']
        negative_emojis = ['📉', '💸', '😭', '🔴', '⬇️']
        
        for emoji in positive_emojis:
            if emoji in text:
                return 'positive sentiment'
        
        for emoji in negative_emojis:
            if emoji in text:
                return 'negative sentiment'
        
        return None

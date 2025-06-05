"""
Response Aggregator for combining multi-agent responses into unified intelligence
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class AgentResponse:
    agent_name: str
    response_data: Dict[str, Any]
    confidence_score: float
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None

@dataclass
class AggregationWeights:
    sentiment_weight: float = 0.25
    forecasting_weight: float = 0.30
    strategy_weight: float = 0.30
    explanation_weight: float = 0.15

class ResponseAggregator:
    def __init__(self):
        # Aggregation strategies
        self.aggregation_strategies = {
            "quick": self._quick_aggregation,
            "standard": self._standard_aggregation,
            "comprehensive": self._comprehensive_aggregation,
            "custom": self._custom_aggregation
        }
        
        # Agent response schemas
        self.response_schemas = {
            "sentiment": ["sentiment_score", "sentiment_label", "news_sentiment", "social_sentiment"],
            "tft_forecasting": ["price_predictions", "volatility_forecast", "confidence_intervals", "greeks"],
            "gpt4_strategy": ["recommended_strategies", "risk_analysis", "entry_exit_points", "portfolio_allocation"],
            "llama_explanation": ["market_analysis", "explanation_text", "key_factors", "risk_assessment"]
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "min_confidence": 0.3,
            "max_response_time_ms": 30000,
            "min_agents_for_comprehensive": 3
        }
        
    async def aggregate_agent_responses(
        self,
        agent_responses: Dict[str, Any],
        request: Any,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate responses from multiple agents into unified intelligence"""
        
        try:
            # Parse agent responses
            parsed_responses = await self._parse_agent_responses(agent_responses)
            
            # Validate response quality
            validated_responses = await self._validate_responses(parsed_responses)
            
            # Select aggregation strategy
            analysis_type = getattr(request, 'analysis_type', 'standard')
            aggregation_strategy = self.aggregation_strategies.get(
                analysis_type, self._standard_aggregation
            )
            
            # Aggregate responses
            unified_response = await aggregation_strategy(
                validated_responses, request, market_data
            )
            
            # Calculate overall confidence
            unified_response["confidence_score"] = await self._calculate_overall_confidence(
                validated_responses
            )
            
            # Generate alerts if needed
            unified_response["alerts"] = await self._generate_alerts(
                unified_response, validated_responses, market_data
            )
            
            # Add metadata
            unified_response["aggregation_metadata"] = {
                "agents_used": list(agent_responses.keys()),
                "successful_agents": len(validated_responses),
                "aggregation_strategy": analysis_type,
                "processing_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Successfully aggregated responses from {len(validated_responses)} agents")
            
            return unified_response
            
        except Exception as e:
            logger.error(f"Response aggregation failed: {e}")
            return await self._create_fallback_response(agent_responses, market_data)
    
    async def _parse_agent_responses(
        self, 
        agent_responses: Dict[str, Any]
    ) -> List[AgentResponse]:
        """Parse and validate individual agent responses"""
        
        parsed_responses = []
        
        for agent_name, response_data in agent_responses.items():
            try:
                if isinstance(response_data, dict) and "error" not in response_data:
                    # Extract standard fields
                    confidence = response_data.get("confidence_score", 0.5)
                    processing_time = response_data.get("processing_time_ms", 0)
                    
                    parsed_response = AgentResponse(
                        agent_name=agent_name,
                        response_data=response_data,
                        confidence_score=confidence,
                        processing_time_ms=processing_time,
                        success=True
                    )
                    
                    parsed_responses.append(parsed_response)
                    
                else:
                    # Handle error response
                    error_msg = response_data.get("error", "Unknown error") if isinstance(response_data, dict) else str(response_data)
                    
                    parsed_response = AgentResponse(
                        agent_name=agent_name,
                        response_data={},
                        confidence_score=0.0,
                        processing_time_ms=0,
                        success=False,
                        error_message=error_msg
                    )
                    
                    parsed_responses.append(parsed_response)
                    
            except Exception as e:
                logger.error(f"Failed to parse response from {agent_name}: {e}")
                
        return parsed_responses
    
    async def _validate_responses(
        self, 
        parsed_responses: List[AgentResponse]
    ) -> List[AgentResponse]:
        """Validate response quality and filter out poor responses"""
        
        validated_responses = []
        
        for response in parsed_responses:
            if not response.success:
                logger.warning(f"Skipping failed response from {response.agent_name}: {response.error_message}")
                continue
                
            # Check confidence threshold
            if response.confidence_score < self.quality_thresholds["min_confidence"]:
                logger.warning(f"Low confidence response from {response.agent_name}: {response.confidence_score}")
                continue
                
            # Check response time threshold
            if response.processing_time_ms > self.quality_thresholds["max_response_time_ms"]:
                logger.warning(f"Slow response from {response.agent_name}: {response.processing_time_ms}ms")
                # Don't skip, but note the issue
                
            # Validate response schema
            if await self._validate_response_schema(response):
                validated_responses.append(response)
            else:
                logger.warning(f"Invalid response schema from {response.agent_name}")
                
        return validated_responses
    
    async def _validate_response_schema(self, response: AgentResponse) -> bool:
        """Validate that response contains expected fields"""
        
        try:
            expected_fields = self.response_schemas.get(response.agent_name, [])
            response_data = response.response_data
            
            # Check if at least 50% of expected fields are present
            present_fields = sum(1 for field in expected_fields if field in response_data)
            
            if len(expected_fields) == 0:
                return True  # No schema defined, accept
                
            return present_fields / len(expected_fields) >= 0.5
            
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False
    
    async def _quick_aggregation(
        self,
        responses: List[AgentResponse],
        request: Any,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Quick aggregation strategy for fast responses"""
        
        aggregated = {
            "market_context": market_data.get("market_context", {}),
            "symbols": getattr(request, 'symbols', [])
        }
        
        # Get sentiment if available
        sentiment_response = next((r for r in responses if r.agent_name == "sentiment"), None)
        if sentiment_response:
            aggregated["sentiment_analysis"] = {
                "overall_sentiment": sentiment_response.response_data.get("sentiment_label", "neutral"),
                "sentiment_score": sentiment_response.response_data.get("sentiment_score", 0.0),
                "confidence": sentiment_response.confidence_score
            }
        
        # Get quick forecast if available
        forecast_response = next((r for r in responses if r.agent_name == "tft_forecasting"), None)
        if forecast_response:
            predictions = forecast_response.response_data.get("price_predictions", {})
            aggregated["price_forecasts"] = {
                "short_term": predictions.get("1_day", {}),
                "confidence": forecast_response.confidence_score
            }
        
        return aggregated
    
    async def _standard_aggregation(
        self,
        responses: List[AgentResponse],
        request: Any,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Standard aggregation strategy with balanced coverage"""
        
        aggregated = {
            "market_context": market_data.get("market_context", {}),
            "symbols": getattr(request, 'symbols', [])
        }
        
        # Aggregate sentiment analysis
        sentiment_response = next((r for r in responses if r.agent_name == "sentiment"), None)
        if sentiment_response:
            aggregated["sentiment_analysis"] = await self._aggregate_sentiment(sentiment_response)
        
        # Aggregate price forecasts
        forecast_response = next((r for r in responses if r.agent_name == "tft_forecasting"), None)
        if forecast_response:
            aggregated["price_forecasts"] = await self._aggregate_forecasts(forecast_response)
        
        # Aggregate strategy recommendations
        strategy_response = next((r for r in responses if r.agent_name == "gpt4_strategy"), None)
        if strategy_response:
            aggregated["strategy_recommendations"] = await self._aggregate_strategies(strategy_response)
        
        # Aggregate explanations
        explanation_response = next((r for r in responses if r.agent_name == "llama_explanation"), None)
        if explanation_response:
            aggregated["explanations"] = await self._aggregate_explanations(explanation_response)
        
        return aggregated
    
    async def _comprehensive_aggregation(
        self,
        responses: List[AgentResponse],
        request: Any,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive aggregation with cross-agent validation and enrichment"""
        
        # Start with standard aggregation
        aggregated = await self._standard_aggregation(responses, request, market_data)
        
        # Cross-agent validation and enrichment
        aggregated = await self._cross_validate_responses(aggregated, responses)
        
        # Generate comprehensive insights
        aggregated["comprehensive_insights"] = await self._generate_comprehensive_insights(
            responses, market_data
        )
        
        # Risk assessment
        aggregated["risk_assessment"] = await self._aggregate_risk_assessment(responses)
        
        # Market regime analysis
        aggregated["market_regime"] = await self._analyze_market_regime(responses, market_data)
        
        return aggregated
    
    async def _custom_aggregation(
        self,
        responses: List[AgentResponse],
        request: Any,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Custom aggregation based on specific request parameters"""
        
        # Use standard as base
        aggregated = await self._standard_aggregation(responses, request, market_data)
        
        # Apply custom weights if provided
        custom_weights = getattr(request, 'aggregation_weights', None)
        if custom_weights:
            aggregated = await self._apply_custom_weights(aggregated, responses, custom_weights)
        
        return aggregated
    
    async def _aggregate_sentiment(self, sentiment_response: AgentResponse) -> Dict[str, Any]:
        """Aggregate sentiment analysis data"""
        
        data = sentiment_response.response_data
        
        return {
            "overall_sentiment": data.get("sentiment_label", "neutral"),
            "sentiment_score": data.get("sentiment_score", 0.0),
            "news_sentiment": data.get("news_sentiment", {}),
            "social_sentiment": data.get("social_sentiment", {}),
            "sentiment_trend": data.get("sentiment_trend", "stable"),
            "confidence": sentiment_response.confidence_score,
            "sources_analyzed": data.get("sources_count", 0)
        }
    
    async def _aggregate_forecasts(self, forecast_response: AgentResponse) -> Dict[str, Any]:
        """Aggregate price forecasting data"""
        
        data = forecast_response.response_data
        predictions = data.get("price_predictions", {})
        
        return {
            "horizons": {
                "1_day": predictions.get("1_day", {}),
                "5_day": predictions.get("5_day", {}),
                "10_day": predictions.get("10_day", {}),
                "21_day": predictions.get("21_day", {})
            },
            "volatility_forecast": data.get("volatility_forecast", {}),
            "confidence_intervals": data.get("confidence_intervals", {}),
            "greeks": data.get("greeks", {}),
            "model_confidence": forecast_response.confidence_score,
            "regime_detected": data.get("market_regime", "normal")
        }
    
    async def _aggregate_strategies(self, strategy_response: AgentResponse) -> Dict[str, Any]:
        """Aggregate strategy recommendations"""
        
        data = strategy_response.response_data
        
        return {
            "recommended_strategies": data.get("recommended_strategies", []),
            "risk_analysis": data.get("risk_analysis", {}),
            "entry_exit_points": data.get("entry_exit_points", {}),
            "portfolio_allocation": data.get("portfolio_allocation", {}),
            "expected_returns": data.get("expected_returns", {}),
            "max_risk": data.get("max_risk", {}),
            "strategy_confidence": strategy_response.confidence_score,
            "reasoning": data.get("strategy_reasoning", "")
        }
    
    async def _aggregate_explanations(self, explanation_response: AgentResponse) -> Dict[str, Any]:
        """Aggregate explanations and analysis"""
        
        data = explanation_response.response_data
        
        return {
            "market_analysis": data.get("market_analysis", ""),
            "key_factors": data.get("key_factors", []),
            "technical_analysis": data.get("technical_analysis", ""),
            "fundamental_factors": data.get("fundamental_factors", []),
            "risk_factors": data.get("risk_factors", []),
            "explanation_confidence": explanation_response.confidence_score,
            "summary": data.get("explanation_text", "")
        }
    
    async def _cross_validate_responses(
        self,
        aggregated: Dict[str, Any],
        responses: List[AgentResponse]
    ) -> Dict[str, Any]:
        """Cross-validate responses between agents for consistency"""
        
        try:
            validation_results = {}
            
            # Validate sentiment vs forecast consistency
            if "sentiment_analysis" in aggregated and "price_forecasts" in aggregated:
                sentiment_score = aggregated["sentiment_analysis"].get("sentiment_score", 0)
                forecast_trend = self._extract_forecast_trend(aggregated["price_forecasts"])
                
                consistency = await self._check_sentiment_forecast_consistency(
                    sentiment_score, forecast_trend
                )
                validation_results["sentiment_forecast_consistency"] = consistency
            
            # Validate strategy vs risk consistency
            if "strategy_recommendations" in aggregated:
                risk_consistency = await self._validate_strategy_risk_consistency(
                    aggregated["strategy_recommendations"]
                )
                validation_results["strategy_risk_consistency"] = risk_consistency
            
            aggregated["cross_validation"] = validation_results
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            
        return aggregated
    
    async def _generate_comprehensive_insights(
        self,
        responses: List[AgentResponse],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive market insights from all agent data"""
        
        insights = {
            "market_outlook": "neutral",
            "conviction_level": "medium",
            "key_drivers": [],
            "opportunities": [],
            "risks": [],
            "time_horizon_analysis": {}
        }
        
        try:
            # Analyze agent consensus
            agent_sentiments = []
            for response in responses:
                if response.agent_name == "sentiment":
                    sentiment_score = response.response_data.get("sentiment_score", 0)
                    agent_sentiments.append(sentiment_score)
                elif response.agent_name == "tft_forecasting":
                    # Convert forecast to sentiment-like score
                    predictions = response.response_data.get("price_predictions", {})
                    if predictions:
                        trend_score = self._forecast_to_sentiment_score(predictions)
                        agent_sentiments.append(trend_score)
            
            # Calculate consensus
            if agent_sentiments:
                consensus_score = np.mean(agent_sentiments)
                insights["market_outlook"] = self._score_to_outlook(consensus_score)
                insights["conviction_level"] = self._calculate_conviction(agent_sentiments)
            
            # Extract key drivers from all agents
            for response in responses:
                if response.agent_name == "llama_explanation":
                    key_factors = response.response_data.get("key_factors", [])
                    insights["key_drivers"].extend(key_factors[:3])  # Top 3
                elif response.agent_name == "sentiment":
                    news_sentiment = response.response_data.get("news_sentiment", {})
                    if "key_topics" in news_sentiment:
                        insights["key_drivers"].extend(news_sentiment["key_topics"][:2])
            
        except Exception as e:
            logger.error(f"Comprehensive insights generation failed: {e}")
            
        return insights
    
    async def _aggregate_risk_assessment(
        self, 
        responses: List[AgentResponse]
    ) -> Dict[str, Any]:
        """Aggregate risk assessment from multiple agents"""
        
        risk_assessment = {
            "overall_risk": "medium",
            "risk_factors": [],
            "volatility_outlook": "normal",
            "downside_protection": {},
            "scenario_analysis": {}
        }
        
        try:
            risk_scores = []
            
            for response in responses:
                if response.agent_name == "gpt4_strategy":
                    strategy_risk = response.response_data.get("risk_analysis", {})
                    if "risk_level" in strategy_risk:
                        risk_scores.append(self._risk_level_to_score(strategy_risk["risk_level"]))
                    
                    risk_factors = strategy_risk.get("risk_factors", [])
                    risk_assessment["risk_factors"].extend(risk_factors)
                
                elif response.agent_name == "tft_forecasting":
                    volatility = response.response_data.get("volatility_forecast", {})
                    if volatility:
                        vol_score = volatility.get("expected_volatility", 0.2)
                        risk_assessment["volatility_outlook"] = self._volatility_to_outlook(vol_score)
                
                elif response.agent_name == "llama_explanation":
                    explanation_risks = response.response_data.get("risk_factors", [])
                    risk_assessment["risk_factors"].extend(explanation_risks)
            
            # Calculate overall risk
            if risk_scores:
                avg_risk_score = np.mean(risk_scores)
                risk_assessment["overall_risk"] = self._score_to_risk_level(avg_risk_score)
            
        except Exception as e:
            logger.error(f"Risk assessment aggregation failed: {e}")
            
        return risk_assessment
    
    async def _analyze_market_regime(
        self,
        responses: List[AgentResponse],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze current market regime from agent responses"""
        
        regime_analysis = {
            "current_regime": "normal",
            "regime_confidence": 0.5,
            "regime_indicators": {},
            "regime_persistence": "medium"
        }
        
        try:
            # Get TFT regime detection
            tft_response = next((r for r in responses if r.agent_name == "tft_forecasting"), None)
            if tft_response:
                regime_data = tft_response.response_data.get("market_regime", {})
                if regime_data:
                    regime_analysis["current_regime"] = regime_data.get("regime", "normal")
                    regime_analysis["regime_confidence"] = regime_data.get("confidence", 0.5)
                    regime_analysis["regime_indicators"] = regime_data.get("indicators", {})
            
            # Supplement with sentiment regime indicators
            sentiment_response = next((r for r in responses if r.agent_name == "sentiment"), None)
            if sentiment_response:
                sentiment_score = sentiment_response.response_data.get("sentiment_score", 0)
                if sentiment_score > 0.7:
                    regime_analysis["regime_indicators"]["sentiment_regime"] = "bullish"
                elif sentiment_score < -0.3:
                    regime_analysis["regime_indicators"]["sentiment_regime"] = "bearish"
                else:
                    regime_analysis["regime_indicators"]["sentiment_regime"] = "neutral"
            
        except Exception as e:
            logger.error(f"Market regime analysis failed: {e}")
            
        return regime_analysis
    
    async def _calculate_overall_confidence(
        self, 
        responses: List[AgentResponse]
    ) -> float:
        """Calculate overall confidence score from agent responses"""
        
        try:
            if not responses:
                return 0.0
            
            # Weight confidences by agent importance and quality
            weights = AggregationWeights()
            weighted_scores = []
            
            for response in responses:
                agent_weight = getattr(weights, f"{response.agent_name}_weight", 0.25)
                
                # Adjust weight based on response quality
                quality_multiplier = 1.0
                if response.processing_time_ms > 10000:  # Slow response
                    quality_multiplier *= 0.8
                if len(response.response_data) < 3:  # Sparse response
                    quality_multiplier *= 0.9
                    
                weighted_score = response.confidence_score * agent_weight * quality_multiplier
                weighted_scores.append(weighted_score)
            
            # Calculate weighted average
            total_confidence = sum(weighted_scores)
            total_weight = sum(getattr(weights, f"{r.agent_name}_weight", 0.25) for r in responses)
            
            if total_weight > 0:
                return min(0.99, max(0.01, total_confidence / total_weight))
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    async def _generate_alerts(
        self,
        unified_response: Dict[str, Any],
        responses: List[AgentResponse],
        market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate alerts based on aggregated intelligence"""
        
        alerts = []
        
        try:
            # High volatility alert
            if "price_forecasts" in unified_response:
                volatility = unified_response["price_forecasts"].get("volatility_forecast", {})
                expected_vol = volatility.get("expected_volatility", 0)
                
                if expected_vol > 0.4:  # 40% volatility threshold
                    alerts.append({
                        "type": "high_volatility",
                        "severity": "warning",
                        "message": f"High volatility expected: {expected_vol:.1%}",
                        "recommendation": "Consider protective strategies"
                    })
            
            # Sentiment divergence alert
            if "sentiment_analysis" in unified_response and "price_forecasts" in unified_response:
                sentiment_score = unified_response["sentiment_analysis"].get("sentiment_score", 0)
                forecast_trend = self._extract_forecast_trend(unified_response["price_forecasts"])
                
                if abs(sentiment_score - forecast_trend) > 0.5:
                    alerts.append({
                        "type": "sentiment_divergence",
                        "severity": "info",
                        "message": "Sentiment and price forecast divergence detected",
                        "recommendation": "Monitor for potential reversal signals"
                    })
            
            # Low confidence alert
            overall_confidence = unified_response.get("confidence_score", 0.5)
            if overall_confidence < 0.4:
                alerts.append({
                    "type": "low_confidence",
                    "severity": "warning",
                    "message": f"Low analysis confidence: {overall_confidence:.1%}",
                    "recommendation": "Gather additional data before making decisions"
                })
            
            # Strategy risk alert
            if "strategy_recommendations" in unified_response:
                strategies = unified_response["strategy_recommendations"]
                max_risk = strategies.get("max_risk", {}).get("total_risk", 0)
                
                if max_risk > 0.15:  # 15% maximum risk threshold
                    alerts.append({
                        "type": "high_strategy_risk",
                        "severity": "warning",
                        "message": f"High strategy risk detected: {max_risk:.1%}",
                        "recommendation": "Consider risk reduction measures"
                    })
            
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
            
        return alerts
    
    async def _create_fallback_response(
        self,
        agent_responses: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create fallback response when aggregation fails"""
        
        return {
            "market_context": market_data.get("market_context", {}),
            "symbols": market_data.get("symbols", []),
            "sentiment_analysis": {"overall_sentiment": "neutral", "confidence": 0.1},
            "price_forecasts": {"status": "unavailable"},
            "strategy_recommendations": {"status": "unavailable"},
            "explanations": {"summary": "Analysis temporarily unavailable"},
            "confidence_score": 0.1,
            "alerts": [{
                "type": "system_error",
                "severity": "error",
                "message": "Analysis system experiencing issues",
                "recommendation": "Please try again in a few moments"
            }],
            "aggregation_metadata": {
                "agents_used": list(agent_responses.keys()),
                "successful_agents": 0,
                "aggregation_strategy": "fallback",
                "processing_timestamp": datetime.utcnow().isoformat()
            }
        }
    
    # Helper methods
    def _extract_forecast_trend(self, forecasts: Dict[str, Any]) -> float:
        """Extract trend direction from price forecasts"""
        try:
            horizons = forecasts.get("horizons", {})
            if "5_day" in horizons:
                current_price = horizons["5_day"].get("current_price", 100)
                predicted_price = horizons["5_day"].get("predicted_price", 100)
                return (predicted_price - current_price) / current_price
        except:
            pass
        return 0.0
    
    def _forecast_to_sentiment_score(self, predictions: Dict[str, Any]) -> float:
        """Convert forecast predictions to sentiment-like score"""
        try:
            # Use 5-day forecast as representative
            if "5_day" in predictions:
                trend = self._extract_forecast_trend({"horizons": {"5_day": predictions["5_day"]}})
                return max(-1, min(1, trend * 5))  # Scale and clamp
        except:
            pass
        return 0.0
    
    def _score_to_outlook(self, score: float) -> str:
        """Convert numeric score to market outlook"""
        if score > 0.3:
            return "bullish"
        elif score < -0.3:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_conviction(self, scores: List[float]) -> str:
        """Calculate conviction level from score variance"""
        if len(scores) < 2:
            return "low"
        
        variance = np.var(scores)
        if variance < 0.1:
            return "high"
        elif variance < 0.3:
            return "medium"
        else:
            return "low"
    
    def _risk_level_to_score(self, risk_level: str) -> float:
        """Convert risk level to numeric score"""
        mapping = {"low": 0.2, "medium": 0.5, "high": 0.8, "very_high": 0.95}
        return mapping.get(risk_level.lower(), 0.5)
    
    def _score_to_risk_level(self, score: float) -> str:
        """Convert numeric score to risk level"""
        if score > 0.75:
            return "high"
        elif score > 0.5:
            return "medium"
        else:
            return "low"
    
    def _volatility_to_outlook(self, volatility: float) -> str:
        """Convert volatility to outlook"""
        if volatility > 0.4:
            return "high"
        elif volatility > 0.25:
            return "elevated"
        else:
            return "normal"
    
    async def _check_sentiment_forecast_consistency(
        self, 
        sentiment_score: float, 
        forecast_trend: float
    ) -> Dict[str, Any]:
        """Check consistency between sentiment and forecast"""
        
        # Both should point in same direction
        sentiment_direction = "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral"
        forecast_direction = "positive" if forecast_trend > 0.02 else "negative" if forecast_trend < -0.02 else "neutral"
        
        consistent = sentiment_direction == forecast_direction or "neutral" in [sentiment_direction, forecast_direction]
        
        return {
            "consistent": consistent,
            "sentiment_direction": sentiment_direction,
            "forecast_direction": forecast_direction,
            "divergence_magnitude": abs(sentiment_score - forecast_trend)
        }
    
    async def _validate_strategy_risk_consistency(
        self, 
        strategies: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate strategy recommendations for risk consistency"""
        
        try:
            recommendations = strategies.get("recommended_strategies", [])
            risk_analysis = strategies.get("risk_analysis", {})
            
            # Check if recommended strategies match stated risk tolerance
            high_risk_strategies = sum(1 for s in recommendations if s.get("risk_level", "medium") == "high")
            total_strategies = len(recommendations)
            
            risk_ratio = high_risk_strategies / max(1, total_strategies)
            stated_risk = risk_analysis.get("risk_level", "medium")
            
            expected_risk_ratio = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(stated_risk, 0.5)
            
            consistent = abs(risk_ratio - expected_risk_ratio) < 0.3
            
            return {
                "consistent": consistent,
                "risk_ratio": risk_ratio,
                "expected_ratio": expected_risk_ratio,
                "stated_risk_level": stated_risk
            }
            
        except Exception as e:
            logger.error(f"Strategy risk validation failed: {e}")
            return {"consistent": True, "error": str(e)}
    
    async def _apply_custom_weights(
        self,
        aggregated: Dict[str, Any],
        responses: List[AgentResponse],
        custom_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Apply custom weighting to aggregated results"""
        
        try:
            # Re-weight confidence scores
            total_weight = 0
            weighted_confidence = 0
            
            for response in responses:
                weight = custom_weights.get(response.agent_name, 0.25)
                weighted_confidence += response.confidence_score * weight
                total_weight += weight
            
            if total_weight > 0:
                aggregated["confidence_score"] = weighted_confidence / total_weight
            
            # Apply weights to specific sections if needed
            if "sentiment_weight" in custom_weights and "sentiment_analysis" in aggregated:
                sentiment_weight = custom_weights["sentiment_weight"]
                aggregated["sentiment_analysis"]["adjusted_weight"] = sentiment_weight
            
        except Exception as e:
            logger.error(f"Custom weight application failed: {e}")
            
        return aggregated

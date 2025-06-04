"""
GPT-4 Strategy Engine - Core strategy generation logic
"""

import openai
import asyncio
import json
import time
import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import tiktoken

logger = logging.getLogger(__name__)

@dataclass
class StrategyResult:
    strategies: List[Dict[str, Any]]
    confidence_score: float
    reasoning: str
    risk_assessment: Dict[str, Any]
    actual_cost: float
    tokens_used: int
    model_used: str

class GPT4StrategyEngine:
    def __init__(self):
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = "gpt-4-turbo-preview"
        self.fallback_model = "gpt-3.5-turbo"
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Rate limiting settings
        self.max_tokens_per_request = 4000
        self.max_requests_per_minute = 50
        
        # Cost tracking
        self.cost_per_token_input = 0.00001  # $0.01/1K tokens
        self.cost_per_token_output = 0.00003  # $0.03/1K tokens
        
    async def health_check(self) -> str:
        """Check GPT-4 API health"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "health"}],
                max_tokens=5
            )
            return "healthy"
        except Exception as e:
            logger.error(f"GPT-4 health check failed: {e}")
            return "unhealthy"
    
    async def estimate_cost(self, user_intent: str) -> float:
        """Estimate cost for a strategy generation request"""
        try:
            # Estimate tokens for a typical strategy request
            prompt_tokens = len(self.encoding.encode(user_intent)) + 2000  # Base prompt
            estimated_output_tokens = 1000  # Typical response
            
            estimated_cost = (
                prompt_tokens * self.cost_per_token_input +
                estimated_output_tokens * self.cost_per_token_output
            )
            
            return estimated_cost
        except Exception as e:
            logger.error(f"Cost estimation failed: {e}")
            return 0.50  # Default estimate
    
    async def generate_strategy(
        self,
        user_intent: str,
        enriched_context: Dict[str, Any],
        risk_preferences: Optional[Dict[str, Any]] = None
    ) -> StrategyResult:
        """Generate options strategy using GPT-4 with full context"""
        
        start_time = time.time()
        
        try:
            # Build comprehensive prompt
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(
                user_intent, enriched_context, risk_preferences
            )
            
            # Count input tokens
            input_tokens = len(self.encoding.encode(system_prompt + user_prompt))
            
            # Make GPT-4 request
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens_per_request,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            output_tokens = response.usage.completion_tokens
            actual_cost = (
                input_tokens * self.cost_per_token_input +
                output_tokens * self.cost_per_token_output
            )
            
            result_data = json.loads(response.choices[0].message.content)
            
            return StrategyResult(
                strategies=result_data.get("strategies", []),
                confidence_score=result_data.get("confidence_score", 0.8),
                reasoning=result_data.get("reasoning", ""),
                risk_assessment=result_data.get("risk_assessment", {}),
                actual_cost=actual_cost,
                tokens_used=input_tokens + output_tokens,
                model_used=self.model
            )
            
        except Exception as e:
            logger.error(f"GPT-4 strategy generation failed: {e}")
            # Try fallback
            return await self.generate_fallback_strategy(user_intent, enriched_context)
    
    async def generate_fallback_strategy(
        self,
        user_intent: str,
        context: Dict[str, Any]
    ) -> StrategyResult:
        """Generate strategy using simpler fallback logic"""
        
        try:
            # Use GPT-3.5 as fallback
            simplified_prompt = self._build_fallback_prompt(user_intent, context)
            
            response = await self.client.chat.completions.create(
                model=self.fallback_model,
                messages=[
                    {"role": "system", "content": "You are an options trading assistant."},
                    {"role": "user", "content": simplified_prompt}
                ],
                max_tokens=1000,
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            
            result_data = json.loads(response.choices[0].message.content)
            
            return StrategyResult(
                strategies=result_data.get("strategies", []),
                confidence_score=0.6,  # Lower confidence for fallback
                reasoning="Generated using fallback model due to service constraints",
                risk_assessment={"warning": "Fallback strategy - validate carefully"},
                actual_cost=0.0,
                tokens_used=response.usage.total_tokens,
                model_used=self.fallback_model
            )
            
        except Exception as e:
            logger.error(f"Fallback strategy generation failed: {e}")
            # Return basic strategy templates
            return self._get_template_strategies(user_intent)
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt for GPT-4"""
        return """You are an expert options trading strategist with deep knowledge of:
1. Options Greeks and pricing models
2. Portfolio risk management and hedging
3. Market microstructure and volatility dynamics
4. Regulatory compliance and position sizing
5. Tax implications and assignment risks

Guidelines:
- Always consider current portfolio positions when recommending strategies
- Prioritize risk management over profit maximization
- Provide specific entry/exit criteria and position sizing
- Include Greeks analysis and volatility sensitivity
- Validate strategies against market conditions and liquidity
- Flag any regulatory or compliance considerations
- Return responses in valid JSON format

Response format:
{
  "strategies": [
    {
      "name": "Strategy Name",
      "type": "covered_call|protective_put|iron_condor|etc",
      "description": "Detailed description",
      "legs": [
        {
          "action": "buy|sell",
          "option_type": "call|put",
          "strike": 150.0,
          "expiration": "2024-01-19",
          "quantity": 1,
          "premium": 5.50
        }
      ],
      "max_profit": 550.0,
      "max_loss": -450.0,
      "breakeven_points": [155.5],
      "delta": 0.25,
      "gamma": 0.15,
      "theta": -0.05,
      "vega": 0.30,
      "capital_required": 5000.0,
      "probability_of_profit": 0.65,
      "ideal_market_condition": "neutral to slightly bullish",
      "entry_criteria": "IV > 25%, RSI < 70",
      "exit_criteria": "50% profit or 21 DTE",
      "risk_warnings": ["Assignment risk on short calls"]
    }
  ],
  "confidence_score": 0.85,
  "reasoning": "Detailed explanation of strategy selection and market analysis",
  "risk_assessment": {
    "overall_risk": "medium",
    "key_risks": ["volatility expansion", "early assignment"],
    "portfolio_impact": "neutral to slightly positive delta",
    "stress_scenarios": ["What happens if underlying moves +/- 10%"]
  }
}"""

    def _build_user_prompt(
        self,
        user_intent: str,
        enriched_context: Dict[str, Any],
        risk_preferences: Optional[Dict[str, Any]]
    ) -> str:
        """Build detailed user prompt with context"""
        
        prompt = f"""
REQUEST: {user_intent}

MARKET CONTEXT:
{json.dumps(enriched_context.get('market', {}), indent=2)}

CURRENT PORTFOLIO:
{json.dumps(enriched_context.get('portfolio', {}), indent=2)}

RISK PREFERENCES:
{json.dumps(risk_preferences or {}, indent=2)}

ADDITIONAL CONTEXT:
- Current VIX: {enriched_context.get('market', {}).get('vix', 'N/A')}
- Market Trend: {enriched_context.get('market', {}).get('trend', 'N/A')}
- Sector Rotation: {enriched_context.get('market', {}).get('sector_rotation', 'N/A')}
- Earnings Calendar: {enriched_context.get('market', {}).get('upcoming_earnings', [])}

Please generate 2-3 options strategies that best match this request, considering:
1. Current portfolio positions and Greeks exposure
2. Market conditions and volatility environment
3. Risk tolerance and capital constraints
4. Time horizon and profit targets
5. Liquidity and bid-ask spreads

Provide specific, actionable strategies with entry/exit criteria.
"""
        return prompt
    
    def _build_fallback_prompt(self, user_intent: str, context: Dict[str, Any]) -> str:
        """Build simplified prompt for fallback model"""
        return f"""
Generate options strategies for: {user_intent}

Market context: {json.dumps(context.get('market', {}), indent=2)}

Return JSON with basic strategy recommendations including legs, max profit/loss, and risk level.
"""
    
    def _get_template_strategies(self, user_intent: str) -> StrategyResult:
        """Return template strategies as last resort"""
        
        template_strategies = [
            {
                "name": "Conservative Covered Call",
                "type": "covered_call",
                "description": "Basic income generation strategy",
                "legs": [
                    {
                        "action": "sell",
                        "option_type": "call",
                        "strike": "current_price + 5%",
                        "expiration": "30 DTE",
                        "quantity": 1,
                        "premium": "estimated"
                    }
                ],
                "max_profit": "premium + (strike - current_price)",
                "max_loss": "unlimited downside",
                "risk_warnings": ["Assignment risk", "Capped upside"]
            }
        ]
        
        return StrategyResult(
            strategies=template_strategies,
            confidence_score=0.3,
            reasoning="Template strategy due to service unavailability",
            risk_assessment={"warning": "Template strategy - requires customization"},
            actual_cost=0.0,
            tokens_used=0,
            model_used="template"
        )

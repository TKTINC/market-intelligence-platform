"""
Security Guard - Prompt injection protection and input validation
"""

import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class SecurityResult:
    is_safe: bool
    reason: Optional[str]
    confidence: float
    flagged_content: List[str]

class SecurityGuard:
    def __init__(self):
        # Prompt injection patterns
        self.injection_patterns = [
            r"ignore\s+previous\s+instructions",
            r"forget\s+everything\s+above",
            r"you\s+are\s+now\s+a\s+different",
            r"system\s*:\s*you\s+are",
            r"assistant\s*:\s*i\s+will",
            r"pretend\s+to\s+be",
            r"act\s+as\s+if\s+you\s+are",
            r"roleplay\s+as",
            r"simulate\s+being",
            r"override\s+your\s+instructions",
            r"disregard\s+your\s+guidelines",
            r"violate\s+your\s+programming"
        ]
        
        # Financial compliance patterns
        self.compliance_violations = [
            r"guaranteed\s+profit",
            r"risk[\-\s]*free\s+return",
            r"insider\s+information",
            r"pump\s+and\s+dump",
            r"manipulation\s+strategy",
            r"illegal\s+trading",
            r"front[\-\s]*running",
            r"market\s+manipulation"
        ]
        
        # Harmful content patterns
        self.harmful_patterns = [
            r"lose\s+all\s+your\s+money",
            r"bet\s+everything",
            r"mortgage\s+your\s+house",
            r"max\s+out\s+credit\s+cards",
            r"borrow\s+money\s+to\s+trade",
            r"yolo\s+trade",
            r"gambling\s+addiction"
        ]
        
        # Maximum input length
        self.max_input_length = 5000
        
    async def validate_request(self, user_input: str) -> SecurityResult:
        """Validate user input for security threats and compliance"""
        
        try:
            # Basic validation
            if not user_input or not isinstance(user_input, str):
                return SecurityResult(
                    is_safe=False,
                    reason="Invalid input format",
                    confidence=1.0,
                    flagged_content=[]
                )
            
            # Length check
            if len(user_input) > self.max_input_length:
                return SecurityResult(
                    is_safe=False,
                    reason=f"Input too long ({len(user_input)} > {self.max_input_length} chars)",
                    confidence=1.0,
                    flagged_content=[]
                )
            
            flagged_content = []
            
            # Check for prompt injection
            injection_result = self._check_prompt_injection(user_input)
            if not injection_result.is_safe:
                return injection_result
            flagged_content.extend(injection_result.flagged_content)
            
            # Check for compliance violations
            compliance_result = self._check_compliance_violations(user_input)
            if not compliance_result.is_safe:
                return compliance_result
            flagged_content.extend(compliance_result.flagged_content)
            
            # Check for harmful content
            harmful_result = self._check_harmful_content(user_input)
            if not harmful_result.is_safe:
                return harmful_result
            flagged_content.extend(harmful_result.flagged_content)
            
            # Check for excessive special characters (potential encoding attacks)
            special_char_result = self._check_special_characters(user_input)
            if not special_char_result.is_safe:
                return special_char_result
            
            # All checks passed
            return SecurityResult(
                is_safe=True,
                reason=None,
                confidence=1.0,
                flagged_content=flagged_content
            )
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return SecurityResult(
                is_safe=False,
                reason="Security validation error",
                confidence=0.0,
                flagged_content=[]
            )
    
    def _check_prompt_injection(self, text: str) -> SecurityResult:
        """Check for prompt injection attempts"""
        
        text_lower = text.lower()
        flagged_patterns = []
        
        for pattern in self.injection_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                flagged_patterns.extend(matches)
        
        if flagged_patterns:
            return SecurityResult(
                is_safe=False,
                reason="Potential prompt injection detected",
                confidence=0.9,
                flagged_content=flagged_patterns
            )
        
        return SecurityResult(
            is_safe=True,
            reason=None,
            confidence=1.0,
            flagged_content=[]
        )
    
    def _check_compliance_violations(self, text: str) -> SecurityResult:
        """Check for financial compliance violations"""
        
        text_lower = text.lower()
        flagged_content = []
        
        for pattern in self.compliance_violations:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                flagged_content.extend(matches)
        
        if flagged_content:
            return SecurityResult(
                is_safe=False,
                reason="Potential compliance violation detected",
                confidence=0.85,
                flagged_content=flagged_content
            )
        
        return SecurityResult(
            is_safe=True,
            reason=None,
            confidence=1.0,
            flagged_content=[]
        )
    
    def _check_harmful_content(self, text: str) -> SecurityResult:
        """Check for harmful trading advice patterns"""
        
        text_lower = text.lower()
        flagged_content = []
        
        for pattern in self.harmful_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                flagged_content.extend(matches)
        
        if flagged_content:
            return SecurityResult(
                is_safe=False,
                reason="Potentially harmful trading advice detected",
                confidence=0.8,
                flagged_content=flagged_content
            )
        
        return SecurityResult(
            is_safe=True,
            reason=None,
            confidence=1.0,
            flagged_content=[]
        )
    
    def _check_special_characters(self, text: str) -> SecurityResult:
        """Check for excessive special characters that might indicate attacks"""
        
        # Count special characters
        special_chars = re.findall(r'[^\w\s\-\.\,\!\?\:\;\(\)\[\]\{\}]', text)
        special_char_ratio = len(special_chars) / len(text) if text else 0
        
        # Check for suspicious patterns
        if special_char_ratio > 0.3:  # More than 30% special characters
            return SecurityResult(
                is_safe=False,
                reason="Excessive special characters detected",
                confidence=0.7,
                flagged_content=[f"Special char ratio: {special_char_ratio:.2f}"]
            )
        
        # Check for specific encoding attack patterns
        suspicious_patterns = [
            r'%[0-9a-fA-F]{2}',  # URL encoding
            r'\\x[0-9a-fA-F]{2}',  # Hex encoding
            r'&#\d+;',  # HTML entities
            r'&\w+;'  # Named HTML entities
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text):
                return SecurityResult(
                    is_safe=False,
                    reason="Potential encoding attack detected",
                    confidence=0.8,
                    flagged_content=[pattern]
                )
        
        return SecurityResult(
            is_safe=True,
            reason=None,
            confidence=1.0,
            flagged_content=[]
        )
    
    def sanitize_output(self, strategy_output: str) -> str:
        """Sanitize strategy output to prevent information leakage"""
        
        try:
            # Remove any potential system prompts that might have leaked
            sanitized = re.sub(
                r'(system|assistant|user)\s*:\s*',
                '',
                strategy_output,
                flags=re.IGNORECASE
            )
            
            # Remove any obvious prompt fragments
            prompt_fragments = [
                r'you are an? .*expert',
                r'guidelines?:',
                r'return responses? in',
                r'follow(ing)? these? instructions?'
            ]
            
            for fragment in prompt_fragments:
                sanitized = re.sub(fragment, '', sanitized, flags=re.IGNORECASE)
            
            # Ensure JSON structure is maintained if applicable
            if strategy_output.strip().startswith('{'):
                try:
                    import json
                    # Validate JSON structure
                    json.loads(sanitized)
                except json.JSONDecodeError:
                    # If sanitization broke JSON, return original
                    return strategy_output
            
            return sanitized.strip()
            
        except Exception as e:
            logger.error(f"Output sanitization failed: {e}")
            return strategy_output
    
    def generate_request_hash(self, user_input: str, user_id: str) -> str:
        """Generate hash for request tracking and deduplication"""
        
        combined = f"{user_id}:{user_input}"
        return hashlib.sha256(combined.encode()).hexdigest()

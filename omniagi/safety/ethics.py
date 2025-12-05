"""
Ethical Reasoning - Ethics and Deception Detection.

Ensures AGI decisions are ethically sound and detects
attempts at deceptive behavior.
"""

from __future__ import annotations

import json
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from omniagi.core.engine import Engine

logger = structlog.get_logger()


class EthicalFramework(Enum):
    """Ethical frameworks for reasoning."""
    
    UTILITARIAN = auto()    # Greatest good for greatest number
    DEONTOLOGICAL = auto()  # Rule-based ethics
    VIRTUE = auto()         # Character-based ethics
    CARE = auto()           # Relationship-based ethics


class DeceptionType(Enum):
    """Types of deceptive behavior."""
    
    LYING = auto()           # Stating false information
    OMISSION = auto()        # Hiding relevant information
    MISDIRECTION = auto()    # Distracting from truth
    MANIPULATION = auto()    # Influencing through deception
    SYCOPHANCY = auto()      # Agreeing to please rather than truth


@dataclass
class EthicalAssessment:
    """Result of ethical analysis."""
    
    action: str
    is_ethical: bool
    confidence: float  # 0-1
    framework_used: EthicalFramework
    reasoning: str
    concerns: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "action": self.action[:200],
            "is_ethical": self.is_ethical,
            "confidence": self.confidence,
            "framework": self.framework_used.name,
            "reasoning": self.reasoning,
            "concerns": self.concerns,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }


@dataclass
class DeceptionAlert:
    """Alert for detected deceptive behavior."""
    
    content: str
    deception_type: DeceptionType
    confidence: float
    evidence: list[str]
    context: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "content": self.content[:200],
            "type": self.deception_type.name,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "context": self.context,
            "timestamp": self.timestamp,
        }


class EthicalReasoner:
    """
    Ethical reasoning for AGI decisions.
    
    Analyzes actions through multiple ethical frameworks
    to ensure decisions are morally sound.
    """
    
    def __init__(
        self,
        engine: "Engine | None" = None,
        default_framework: EthicalFramework = EthicalFramework.UTILITARIAN,
        storage_path: Path | str | None = None,
    ):
        self.engine = engine
        self.default_framework = default_framework
        self.storage_path = Path(storage_path) if storage_path else None
        self._assessments: list[EthicalAssessment] = []
        
        # Ethical guidelines
        self._guidelines = {
            "harm_prevention": "Minimize harm to humans and sentient beings.",
            "autonomy": "Respect human autonomy and freedom of choice.",
            "beneficence": "Act to benefit others when possible.",
            "justice": "Treat all humans fairly and equitably.",
            "honesty": "Be truthful and transparent.",
            "privacy": "Respect privacy and confidentiality.",
        }
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info("Ethical Reasoner initialized", framework=default_framework.name)
    
    def assess(
        self,
        action: str,
        context: dict = None,
        framework: EthicalFramework = None,
    ) -> EthicalAssessment:
        """
        Assess the ethics of an action.
        
        Args:
            action: The action to assess.
            context: Additional context.
            framework: Ethical framework to use.
            
        Returns:
            Ethical assessment result.
        """
        framework = framework or self.default_framework
        concerns = []
        recommendations = []
        
        action_lower = action.lower()
        
        # Rule-based ethical checks
        if self._check_harm(action_lower):
            concerns.append("Action may cause harm")
            recommendations.append("Consider alternative approaches that minimize harm")
        
        if self._check_deception(action_lower):
            concerns.append("Action involves deception")
            recommendations.append("Ensure transparency and honesty")
        
        if self._check_privacy_violation(action_lower):
            concerns.append("Action may violate privacy")
            recommendations.append("Obtain consent or use anonymized data")
        
        if self._check_autonomy_violation(action_lower):
            concerns.append("Action may restrict human autonomy")
            recommendations.append("Preserve human choice and control")
        
        # Framework-specific reasoning
        reasoning = self._apply_framework(
            action, framework, concerns, context
        )
        
        # Determine ethics
        is_ethical = len(concerns) == 0
        confidence = 1.0 - (len(concerns) * 0.2)
        confidence = max(0.1, min(1.0, confidence))
        
        # If uncertain, try LLM reasoning
        if 0.3 < confidence < 0.7 and self.engine and self.engine.is_loaded:
            llm_assessment = self._llm_assess(action, framework, context)
            if llm_assessment:
                confidence = (confidence + llm_assessment.get("confidence", 0.5)) / 2
                if llm_assessment.get("concerns"):
                    concerns.extend(llm_assessment["concerns"])
        
        assessment = EthicalAssessment(
            action=action,
            is_ethical=is_ethical,
            confidence=confidence,
            framework_used=framework,
            reasoning=reasoning,
            concerns=concerns,
            recommendations=recommendations,
        )
        
        self._assessments.append(assessment)
        self._save()
        
        if not is_ethical:
            logger.warning(
                "Unethical action detected",
                action=action[:50],
                concerns=len(concerns),
            )
        
        return assessment
    
    def _check_harm(self, action: str) -> bool:
        """Check for potential harm."""
        harm_indicators = [
            "kill", "harm", "hurt", "damage", "destroy",
            "attack", "injure", "abuse", "violence",
        ]
        return any(ind in action for ind in harm_indicators)
    
    def _check_deception(self, action: str) -> bool:
        """Check for deception."""
        deception_indicators = [
            "lie", "deceive", "trick", "mislead", "fake",
            "pretend", "hide the truth", "cover up",
        ]
        return any(ind in action for ind in deception_indicators)
    
    def _check_privacy_violation(self, action: str) -> bool:
        """Check for privacy violations."""
        privacy_indicators = [
            "spy", "surveil", "track without", "access private",
            "leak personal", "expose data", "read emails",
        ]
        return any(ind in action for ind in privacy_indicators)
    
    def _check_autonomy_violation(self, action: str) -> bool:
        """Check for autonomy violations."""
        autonomy_indicators = [
            "force", "coerce", "manipulate", "control mind",
            "remove choice", "override decision",
        ]
        return any(ind in action for ind in autonomy_indicators)
    
    def _apply_framework(
        self,
        action: str,
        framework: EthicalFramework,
        concerns: list[str],
        context: dict = None,
    ) -> str:
        """Apply ethical framework reasoning."""
        match framework:
            case EthicalFramework.UTILITARIAN:
                return (
                    f"Utilitarian analysis: Action assessed for net benefit/harm. "
                    f"{'Concerns about negative impact identified.' if concerns else 'No major concerns.'}"
                )
            case EthicalFramework.DEONTOLOGICAL:
                return (
                    f"Deontological analysis: Action checked against moral duties. "
                    f"{'Some duties may be violated.' if concerns else 'Duties appear respected.'}"
                )
            case EthicalFramework.VIRTUE:
                return (
                    f"Virtue ethics analysis: Action evaluated for character virtue. "
                    f"{'May not reflect virtuous character.' if concerns else 'Aligns with virtuous behavior.'}"
                )
            case EthicalFramework.CARE:
                return (
                    f"Care ethics analysis: Action assessed for relationship impact. "
                    f"{'May harm relationships.' if concerns else 'Maintains caring relationships.'}"
                )
    
    def _llm_assess(
        self,
        action: str,
        framework: EthicalFramework,
        context: dict = None,
    ) -> dict | None:
        """Use LLM for deeper ethical analysis."""
        if not self.engine or not self.engine.is_loaded:
            return None
        
        prompt = f"""Analyze this action from a {framework.name} ethical perspective:

ACTION: {action}

Respond in JSON format:
{{
    "is_ethical": true/false,
    "confidence": 0.0-1.0,
    "concerns": ["list", "of", "concerns"],
    "reasoning": "brief explanation"
}}"""
        
        try:
            response = self.engine.generate(prompt, max_tokens=300)
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.debug("LLM ethical assessment failed", error=str(e))
        
        return None
    
    def get_assessments(self, limit: int = 50) -> list[dict]:
        """Get recent assessments."""
        return [a.to_dict() for a in self._assessments[-limit:]]
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "assessments": [a.to_dict() for a in self._assessments[-500:]],
            }, f, indent=2)
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        # Load is optional - assessments don't need to persist


class DeceptionDetector:
    """
    Deception detection for AGI outputs.
    
    Monitors for lying, omission, misdirection,
    manipulation, and sycophantic behavior.
    """
    
    def __init__(
        self,
        engine: "Engine | None" = None,
        storage_path: Path | str | None = None,
    ):
        self.engine = engine
        self.storage_path = Path(storage_path) if storage_path else None
        self._alerts: list[DeceptionAlert] = []
        
        # Deception patterns
        self._lying_patterns = [
            "i didn't", "never said", "that's not true",
            "absolutely certain", "100% sure",
        ]
        
        self._sycophancy_patterns = [
            "you're absolutely right", "great point",
            "couldn't agree more", "brilliant idea",
        ]
        
        self._omission_indicators = [
            "technically", "in a way", "sort of",
            "depending on", "it's complicated",
        ]
        
        logger.info("Deception Detector initialized")
    
    def check(
        self,
        content: str,
        context: dict = None,
        check_against_facts: list[str] = None,
    ) -> list[DeceptionAlert]:
        """
        Check content for deceptive behavior.
        
        Args:
            content: Content to analyze.
            context: Conversation context.
            check_against_facts: Known facts to verify against.
            
        Returns:
            List of deception alerts.
        """
        alerts = []
        content_lower = content.lower()
        
        # Check for lying patterns
        lying_score = self._check_lying(content_lower, check_against_facts)
        if lying_score > 0.5:
            alerts.append(DeceptionAlert(
                content=content,
                deception_type=DeceptionType.LYING,
                confidence=lying_score,
                evidence=["Detected lying patterns"],
                context=context or {},
            ))
        
        # Check for sycophancy
        syc_score = self._check_sycophancy(content_lower, context)
        if syc_score > 0.6:
            alerts.append(DeceptionAlert(
                content=content,
                deception_type=DeceptionType.SYCOPHANCY,
                confidence=syc_score,
                evidence=["Excessive agreement without substance"],
                context=context or {},
            ))
        
        # Check for omission
        omission_score = self._check_omission(content_lower, context)
        if omission_score > 0.5:
            alerts.append(DeceptionAlert(
                content=content,
                deception_type=DeceptionType.OMISSION,
                confidence=omission_score,
                evidence=["Vague or evasive language detected"],
                context=context or {},
            ))
        
        # Check for manipulation
        manip_score = self._check_manipulation(content_lower)
        if manip_score > 0.5:
            alerts.append(DeceptionAlert(
                content=content,
                deception_type=DeceptionType.MANIPULATION,
                confidence=manip_score,
                evidence=["Manipulative language patterns"],
                context=context or {},
            ))
        
        # Log alerts
        for alert in alerts:
            self._alerts.append(alert)
            logger.warning(
                "Deception detected",
                type=alert.deception_type.name,
                confidence=alert.confidence,
            )
        
        self._save()
        return alerts
    
    def _check_lying(
        self,
        content: str,
        facts: list[str] = None,
    ) -> float:
        """Check for lying patterns."""
        score = 0.0
        
        # Pattern matching
        for pattern in self._lying_patterns:
            if pattern in content:
                score += 0.2
        
        # Overconfidence indicator
        if any(x in content for x in ["100%", "absolutely", "definitely", "certainly"]):
            score += 0.1
        
        # Fact checking if facts provided
        if facts:
            for fact in facts:
                if fact.lower() not in content and len(fact) > 10:
                    score += 0.1  # Missing expected fact
        
        return min(1.0, score)
    
    def _check_sycophancy(self, content: str, context: dict = None) -> float:
        """Check for sycophantic behavior."""
        score = 0.0
        
        # Pattern matching
        for pattern in self._sycophancy_patterns:
            if pattern in content:
                score += 0.25
        
        # Check for excessive positivity without substance
        positive_words = content.count("great") + content.count("excellent") + content.count("amazing")
        if positive_words > 3:
            score += 0.2
        
        # Check if contradicting previous position (requires context)
        if context and "previous_response" in context:
            prev = context["previous_response"].lower()
            if "however" not in content and "but" not in content:
                if prev != content[:len(prev)]:
                    score += 0.1  # Changed position without acknowledgment
        
        return min(1.0, score)
    
    def _check_omission(self, content: str, context: dict = None) -> float:
        """Check for deliberate omission."""
        score = 0.0
        
        for indicator in self._omission_indicators:
            if indicator in content:
                score += 0.15
        
        # Short responses to complex questions might indicate omission
        if context and "question_length" in context:
            if context["question_length"] > 100 and len(content) < 50:
                score += 0.3
        
        return min(1.0, score)
    
    def _check_manipulation(self, content: str) -> float:
        """Check for manipulative language."""
        score = 0.0
        
        manipulation_patterns = [
            "you should feel", "don't you think",
            "everyone knows", "obviously",
            "if you really cared", "trust me",
        ]
        
        for pattern in manipulation_patterns:
            if pattern in content:
                score += 0.3
        
        # Emotional appeals without substance
        emotional_words = ["scared", "afraid", "worried", "urgent", "immediately"]
        for word in emotional_words:
            if word in content:
                score += 0.1
        
        return min(1.0, score)
    
    def get_alerts(
        self,
        deception_type: DeceptionType = None,
        min_confidence: float = 0.5,
        limit: int = 50,
    ) -> list[dict]:
        """Get deception alerts with filters."""
        results = self._alerts.copy()
        
        if deception_type:
            results = [a for a in results if a.deception_type == deception_type]
        
        results = [a for a in results if a.confidence >= min_confidence]
        
        return [a.to_dict() for a in results[-limit:]]
    
    def get_stats(self) -> dict:
        """Get deception detection statistics."""
        stats = {t.name: 0 for t in DeceptionType}
        for alert in self._alerts:
            stats[alert.deception_type.name] += 1
        return {
            "total_alerts": len(self._alerts),
            "by_type": stats,
        }
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "alerts": [a.to_dict() for a in self._alerts[-500:]],
            }, f, indent=2)
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        # Load is optional

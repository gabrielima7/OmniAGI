"""
Common Sense Reasoning Module.

Implements intuitive physics, social reasoning, and
implicit world knowledge for AGI-level understanding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PhysicalProperty(Enum):
    """Physical properties of objects."""
    SOLID = auto()
    LIQUID = auto()
    GAS = auto()
    HEAVY = auto()
    LIGHT = auto()
    FRAGILE = auto()
    ELASTIC = auto()
    RIGID = auto()


class SocialRelation(Enum):
    """Types of social relationships."""
    FRIEND = auto()
    FAMILY = auto()
    COLLEAGUE = auto()
    STRANGER = auto()
    AUTHORITY = auto()
    SUBORDINATE = auto()


@dataclass
class CommonSenseEntity:
    """An entity with common sense properties."""
    name: str
    category: str
    properties: Set[PhysicalProperty] = field(default_factory=set)
    typical_uses: List[str] = field(default_factory=list)
    typical_locations: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)


@dataclass
class IntuitivePrediction:
    """A prediction about the physical world."""
    scenario: str
    prediction: str
    confidence: float
    reasoning: str


class IntuitivePhysics:
    """
    Intuitive physics engine for common sense reasoning.
    
    Implements naive physics understanding without
    explicit simulation.
    """
    
    def __init__(self):
        # Physical rules (implicit knowledge)
        self.rules = {
            "gravity": "Unsupported objects fall down",
            "solidity": "Two solid objects cannot occupy the same space",
            "containment": "Liquids take the shape of their container",
            "support": "Objects need support to stay in place",
            "inertia": "Objects in motion tend to stay in motion",
            "collision": "Objects collide when their paths intersect",
        }
        
        # Object knowledge
        self.object_properties: Dict[str, Set[str]] = {
            "ball": {"round", "rolls", "bounces"},
            "cup": {"container", "holds_liquid", "fragile"},
            "book": {"flat", "stackable", "contains_info"},
            "water": {"liquid", "flows", "wets"},
            "table": {"flat_surface", "supports", "furniture"},
            "chair": {"supports_person", "furniture", "movable"},
        }
    
    def predict(self, scenario: str) -> IntuitivePrediction:
        """
        Make a prediction about a physical scenario.
        
        Args:
            scenario: Description of the scenario
            
        Returns:
            IntuitivePrediction with result
        """
        scenario_lower = scenario.lower()
        
        # Check for common patterns
        if "drop" in scenario_lower or "fall" in scenario_lower:
            return IntuitivePrediction(
                scenario=scenario,
                prediction="The object will fall to the ground",
                confidence=0.95,
                reasoning="Gravity causes unsupported objects to fall",
            )
        
        if "pour" in scenario_lower and "water" in scenario_lower:
            return IntuitivePrediction(
                scenario=scenario,
                prediction="The water will flow downward and spread",
                confidence=0.9,
                reasoning="Liquids flow down due to gravity and take container shape",
            )
        
        if "push" in scenario_lower:
            return IntuitivePrediction(
                scenario=scenario,
                prediction="The object will move in the direction of the push",
                confidence=0.85,
                reasoning="Force causes acceleration in the direction of force",
            )
        
        if "stack" in scenario_lower:
            return IntuitivePrediction(
                scenario=scenario,
                prediction="Objects can be stacked if stable base exists",
                confidence=0.8,
                reasoning="Support and balance determine stacking stability",
            )
        
        return IntuitivePrediction(
            scenario=scenario,
            prediction="Cannot make confident prediction",
            confidence=0.3,
            reasoning="Scenario not well understood",
        )
    
    def check_possibility(self, action: str) -> Tuple[bool, str]:
        """
        Check if an action is physically possible.
        
        Returns:
            (is_possible, explanation)
        """
        action_lower = action.lower()
        
        # Impossible actions
        impossible = [
            ("walk through wall", "Solid objects cannot occupy same space"),
            ("lift truck", "Object too heavy for human strength"),
            ("fly without", "Humans cannot fly unaided"),
        ]
        
        for pattern, reason in impossible:
            if pattern in action_lower:
                return False, reason
        
        return True, "Action appears physically possible"


class SocialReasoning:
    """
    Social reasoning and theory of mind.
    
    Understands social norms, emotions, and intentions.
    """
    
    def __init__(self):
        # Social norms
        self.norms = {
            "greeting": "People greet when meeting",
            "politeness": "Requests should be polite",
            "personal_space": "Maintain appropriate distance",
            "turn_taking": "Conversations involve turn-taking",
            "reciprocity": "Favors are often reciprocated",
        }
        
        # Emotion-cause mappings
        self.emotion_causes = {
            "happy": ["receiving_gift", "achieving_goal", "meeting_friend"],
            "sad": ["losing_something", "failure", "separation"],
            "angry": ["unfairness", "obstruction", "insult"],
            "afraid": ["threat", "uncertainty", "danger"],
            "surprised": ["unexpected_event", "new_information"],
        }
    
    def infer_emotion(self, situation: str) -> Tuple[str, float]:
        """
        Infer likely emotion from a situation.
        
        Returns:
            (emotion, confidence)
        """
        situation_lower = situation.lower()
        
        for emotion, causes in self.emotion_causes.items():
            for cause in causes:
                if cause.replace("_", " ") in situation_lower:
                    return emotion, 0.8
        
        # Check for positive/negative indicators
        positive = ["win", "success", "love", "happy", "gift", "celebrate"]
        negative = ["lose", "fail", "hate", "sad", "hurt", "reject"]
        
        for word in positive:
            if word in situation_lower:
                return "happy", 0.6
        
        for word in negative:
            if word in situation_lower:
                return "sad", 0.6
        
        return "neutral", 0.3
    
    def predict_intention(self, action: str, context: str = "") -> str:
        """Predict the intention behind an action."""
        action_lower = action.lower()
        
        intention_patterns = {
            "give": "to help or show generosity",
            "ask": "to obtain information or assistance",
            "hide": "to conceal something or avoid detection",
            "run": "to escape or arrive quickly",
            "smile": "to express friendliness or happiness",
            "cry": "to express sadness or seek comfort",
        }
        
        for pattern, intention in intention_patterns.items():
            if pattern in action_lower:
                return intention
        
        return "unclear intention"
    
    def check_social_appropriateness(self, action: str, context: str) -> Tuple[bool, str]:
        """Check if an action is socially appropriate."""
        action_lower = action.lower()
        context_lower = context.lower()
        
        # Inappropriate in formal contexts
        if "formal" in context_lower:
            informal = ["joke", "casual", "slang"]
            for word in informal:
                if word in action_lower:
                    return False, "Too informal for formal context"
        
        # Generally inappropriate
        inappropriate = ["insult", "interrupt rudely", "ignore"]
        for word in inappropriate:
            if word in action_lower:
                return False, f"Action '{word}' is generally inappropriate"
        
        return True, "Action appears socially appropriate"


class CommonSenseKnowledgeBase:
    """
    Knowledge base for common sense facts.
    
    Contains implicit world knowledge.
    """
    
    def __init__(self):
        # Temporal knowledge
        self.temporal = {
            "morning": {"time": "6-12", "activities": ["wake", "breakfast", "commute"]},
            "afternoon": {"time": "12-18", "activities": ["work", "lunch", "meetings"]},
            "evening": {"time": "18-22", "activities": ["dinner", "relax", "family"]},
            "night": {"time": "22-6", "activities": ["sleep", "rest"]},
        }
        
        # Causal knowledge
        self.causes = {
            "rain": ["wet ground", "people use umbrellas", "indoor activities"],
            "fire": ["heat", "light", "smoke", "danger"],
            "eating": ["less hungry", "more energy", "satisfaction"],
            "exercise": ["tired", "healthier", "stronger"],
        }
        
        # Object affordances
        self.affordances = {
            "knife": ["cut", "slice", "spread"],
            "chair": ["sit", "stand_on", "move"],
            "door": ["open", "close", "lock"],
            "computer": ["type", "browse", "compute"],
            "phone": ["call", "text", "browse"],
        }
        
        # Typical sequences
        self.scripts = {
            "restaurant": ["enter", "wait", "seated", "order", "eat", "pay", "leave"],
            "shopping": ["enter", "browse", "select", "pay", "leave"],
            "cooking": ["gather_ingredients", "prepare", "cook", "serve", "eat"],
        }
    
    def get_typical_sequence(self, activity: str) -> List[str]:
        """Get typical sequence of events for an activity."""
        for script, steps in self.scripts.items():
            if script in activity.lower():
                return steps
        return []
    
    def get_effects(self, cause: str) -> List[str]:
        """Get typical effects of a cause."""
        for c, effects in self.causes.items():
            if c in cause.lower():
                return effects
        return []
    
    def get_affordances(self, obj: str) -> List[str]:
        """Get what can be done with an object."""
        for o, actions in self.affordances.items():
            if o in obj.lower():
                return actions
        return []


class CommonSenseReasoner:
    """
    Complete common sense reasoning system.
    
    Integrates intuitive physics, social reasoning,
    and world knowledge.
    """
    
    def __init__(self):
        self.physics = IntuitivePhysics()
        self.social = SocialReasoning()
        self.knowledge = CommonSenseKnowledgeBase()
    
    def reason(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Perform common sense reasoning on a query.
        
        Returns comprehensive analysis.
        """
        result = {
            "query": query,
            "physical_prediction": None,
            "social_inference": None,
            "knowledge_applied": [],
            "confidence": 0.0,
        }
        
        # Physical reasoning
        if any(w in query.lower() for w in ["what happens", "will", "if"]):
            pred = self.physics.predict(query)
            result["physical_prediction"] = {
                "prediction": pred.prediction,
                "confidence": pred.confidence,
                "reasoning": pred.reasoning,
            }
            result["confidence"] = max(result["confidence"], pred.confidence)
        
        # Social reasoning
        if any(w in query.lower() for w in ["feel", "think", "want", "why"]):
            emotion, conf = self.social.infer_emotion(query)
            intention = self.social.predict_intention(query, context)
            result["social_inference"] = {
                "emotion": emotion,
                "intention": intention,
                "confidence": conf,
            }
            result["confidence"] = max(result["confidence"], conf)
        
        # Apply world knowledge
        effects = self.knowledge.get_effects(query)
        if effects:
            result["knowledge_applied"].append({
                "type": "causal",
                "effects": effects,
            })
        
        sequence = self.knowledge.get_typical_sequence(query)
        if sequence:
            result["knowledge_applied"].append({
                "type": "script",
                "sequence": sequence,
            })
        
        if result["knowledge_applied"]:
            result["confidence"] = max(result["confidence"], 0.7)
        
        return result

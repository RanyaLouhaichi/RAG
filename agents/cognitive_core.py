# agents/cognitive_core.py
# The cognitive foundation that transforms your agents into sophisticated reasoners
# This file contains the enhanced belief system and cognitive capabilities

from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import math
import logging

class ConfidenceLevel(Enum):
    """Represents different levels of confidence in beliefs"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    CERTAIN = 1.0

class InformationSource(Enum):
    """Types of information sources with different credibility patterns"""
    DIRECT_OBSERVATION = "direct_observation"  # Agent's own experience
    TRUSTED_AGENT = "trusted_agent"           # From known reliable agents
    EXTERNAL_API = "external_api"             # From external systems
    USER_INPUT = "user_input"                 # From human users
    INFERRED = "inferred"                     # Derived through reasoning
    HISTORICAL_DATA = "historical_data"       # From past records

@dataclass
class ConfidentBelief:
    """
    A sophisticated belief that tracks not just what we believe,
    but how confident we are and why we believe it.
    
    This is the foundation of intelligent reasoning - understanding
    the quality and reliability of our own knowledge.
    """
    value: Any                              # The actual belief content
    confidence: float                       # How confident we are (0.0 to 1.0)
    source: InformationSource              # Where this belief came from
    created_at: datetime                   # When we learned this
    last_updated: datetime                 # When we last confirmed/updated this
    evidence_count: int = 0                # How many pieces of evidence support this
    contradiction_count: int = 0           # How many contradictions we've seen
    temporal_decay_rate: float = 0.01      # How quickly confidence fades over time
    context_tags: List[str] = field(default_factory=list)  # What contexts this applies to
    
    def __post_init__(self):
        """Initialize computed properties after creation"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_updated is None:
            self.last_updated = self.created_at
    
    def current_confidence(self) -> float:
        """
        Calculate current confidence considering temporal decay.
        
        This mimics how human confidence in information naturally
        decreases over time unless refreshed.
        """
        time_elapsed = datetime.now() - self.last_updated
        days_elapsed = time_elapsed.total_seconds() / (24 * 3600)
        
        # Apply exponential decay - information becomes less reliable over time
        decay_factor = math.exp(-self.temporal_decay_rate * days_elapsed)
        base_confidence = self.confidence * decay_factor
        
        # Factor in evidence vs contradictions - more evidence increases confidence
        evidence_factor = self.evidence_count / (self.evidence_count + self.contradiction_count + 1)
        
        # Combine factors with some minimum confidence to avoid complete certainty loss
        final_confidence = max(0.05, base_confidence * (0.5 + 0.5 * evidence_factor))
        
        return min(1.0, final_confidence)
    
    def add_supporting_evidence(self, source: InformationSource = None):
        """
        Strengthen this belief with additional supporting evidence.
        
        This is how agents learn to trust information more when
        multiple sources confirm the same thing.
        """
        self.evidence_count += 1
        self.last_updated = datetime.now()
        
        # Boost confidence based on evidence accumulation with diminishing returns
        confidence_boost = 0.1 * (1.0 - self.confidence)
        self.confidence = min(1.0, self.confidence + confidence_boost)
    
    def add_contradiction(self, conflicting_value: Any, source: InformationSource):
        """
        Record when we encounter information that contradicts this belief.
        
        Sophisticated reasoners don't ignore contradictions - they
        factor them into their confidence assessments.
        """
        self.contradiction_count += 1
        self.last_updated = datetime.now()
        
        # Reduce confidence based on contradictions - proportional to current confidence
        confidence_reduction = 0.2 * self.confidence
        self.confidence = max(0.05, self.confidence - confidence_reduction)
    
    def is_stale(self, max_age_days: int = 30) -> bool:
        """Check if this belief might be outdated and needs refreshing"""
        age = datetime.now() - self.last_updated
        return age.total_seconds() > (max_age_days * 24 * 3600)
    
    def is_reliable(self, threshold: float = 0.6) -> bool:
        """Check if this belief is reliable enough for confident decision-making"""
        return self.current_confidence() >= threshold

class CognitiveBeliefSystem:
    """
    An advanced belief management system that enables sophisticated reasoning
    about knowledge, uncertainty, and information quality.
    
    This replaces simple dictionary storage with intelligent knowledge management
    that mirrors how human experts reason about what they know.
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.beliefs: Dict[str, ConfidentBelief] = {}
        
        # Track how reliable different sources have been historically
        # These values will be learned and adjusted over time based on experience
        self.source_reliability: Dict[str, float] = {
            "self": 0.8,                    # Agent's own observations
            "trusted_agents": 0.7,          # Other agents in the system
            "external_apis": 0.6,           # External data sources
            "user_input": 0.5,             # Human input (can be uncertain)
            "inferred": 0.4                 # Derived conclusions
        }
        
        # Track relationships between beliefs for more sophisticated reasoning
        self.belief_networks: Dict[str, List[str]] = {}
        
        # Initialize logging for cognitive processes
        self.logger = logging.getLogger(f"CognitiveBeliefs-{agent_name}")
    
    def believe(self, key: str, value: Any, confidence: float = None, 
                source: InformationSource = InformationSource.DIRECT_OBSERVATION,
                context_tags: List[str] = None) -> ConfidentBelief:
        """
        Form a new belief or update an existing one with sophisticated integration.
        
        This is how agents learn - not by replacing old knowledge, but by
        intelligently integrating new information with existing understanding.
        """
        current_time = datetime.now()
        context_tags = context_tags or []
        
        # Determine confidence if not explicitly provided
        if confidence is None:
            confidence = self._estimate_confidence(source, value)
        
        # Check if we already have a belief about this topic
        if key in self.beliefs:
            existing_belief = self.beliefs[key]
            
            # If the new information matches existing belief, strengthen it
            if self._values_match(existing_belief.value, value):
                existing_belief.add_supporting_evidence(source)
                existing_belief.context_tags.extend(context_tags)
                self.logger.debug(f"Strengthened belief '{key}' with supporting evidence")
                return existing_belief
            
            # If it contradicts, we need to reason about which to trust
            else:
                self.logger.debug(f"Belief conflict detected for '{key}' - resolving...")
                return self._resolve_belief_conflict(key, existing_belief, value, 
                                                   confidence, source, context_tags)
        
        # Create new belief when we don't have existing knowledge
        new_belief = ConfidentBelief(
            value=value,
            confidence=confidence,
            source=source,
            created_at=current_time,
            last_updated=current_time,
            context_tags=context_tags
        )
        
        self.beliefs[key] = new_belief
        self.logger.debug(f"Formed new belief '{key}' with confidence {confidence:.2f}")
        return new_belief
    
    def _estimate_confidence(self, source: InformationSource, value: Any) -> float:
        """
        Estimate appropriate confidence level based on source and content characteristics.
        
        This is how agents develop intuition about information quality.
        """
        # Base confidence levels for different information sources
        base_confidence = {
            InformationSource.DIRECT_OBSERVATION: 0.8,
            InformationSource.TRUSTED_AGENT: 0.7,
            InformationSource.EXTERNAL_API: 0.6,
            InformationSource.USER_INPUT: 0.5,
            InformationSource.INFERRED: 0.4,
            InformationSource.HISTORICAL_DATA: 0.5
        }.get(source, 0.5)
        
        # Adjust confidence based on characteristics of the value itself
        if isinstance(value, (int, float)) and value != 0:
            # Numerical values with specific numbers often more reliable than vague estimates
            base_confidence += 0.1
        elif isinstance(value, str) and len(value) > 100:
            # Detailed text descriptions might indicate more thorough analysis
            base_confidence += 0.05
        elif isinstance(value, list) and len(value) > 3:
            # Multiple data points suggest more comprehensive information
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _values_match(self, value1: Any, value2: Any, tolerance: float = 0.1) -> bool:
        """
        Determine if two values are essentially the same, accounting for
        reasonable variations in numerical data or minor text differences.
        
        This sophisticated matching helps agents recognize when different
        sources are providing consistent information.
        """
        if type(value1) != type(value2):
            return False
        
        # Special handling for numerical values with tolerance for minor variations
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            if value1 == 0 and value2 == 0:
                return True
            relative_diff = abs(value1 - value2) / max(abs(value1), abs(value2), 1)
            return relative_diff <= tolerance
        
        # For lists, check if they're substantially similar
        if isinstance(value1, list) and isinstance(value2, list):
            if len(value1) == len(value2):
                return value1 == value2
            # Consider lists similar if they have significant overlap
            if len(value1) > 0 and len(value2) > 0:
                overlap = len(set(str(v) for v in value1) & set(str(v) for v in value2))
                return overlap / max(len(value1), len(value2)) > 0.7
        
        # For strings and other types, require exact match
        return value1 == value2
    
    def _resolve_belief_conflict(self, key: str, existing_belief: ConfidentBelief,
                                new_value: Any, new_confidence: float,
                                new_source: InformationSource, 
                                context_tags: List[str]) -> ConfidentBelief:
        """
        Intelligently handle conflicting information - a key aspect of sophisticated reasoning.
        
        Instead of just overwriting or ignoring conflicts, we reason about
        which information to trust based on confidence, recency, and source reliability.
        """
        current_confidence = existing_belief.current_confidence()
        
        # Record the contradiction in existing belief for future reference
        existing_belief.add_contradiction(new_value, new_source)
        
        self.logger.debug(f"Resolving conflict for '{key}': existing confidence {current_confidence:.2f} vs new {new_confidence:.2f}")
        
        # Decide which belief to keep based on confidence and other factors
        if new_confidence > current_confidence * 1.2:  # Significantly more confident
            # Replace with new belief but preserve some history about the conflict
            new_belief = ConfidentBelief(
                value=new_value,
                confidence=new_confidence,
                source=new_source,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                contradiction_count=1,  # Note there was conflicting info
                context_tags=context_tags
            )
            self.beliefs[key] = new_belief
            self.logger.debug(f"Replaced belief '{key}' with more confident information")
            return new_belief
        
        else:
            # Keep existing belief but note we've seen conflicting information
            # This uncertainty will affect future decision-making appropriately
            self.logger.debug(f"Kept existing belief '{key}' despite conflicting information")
            return existing_belief
    
    def get_belief(self, key: str) -> Optional[ConfidentBelief]:
        """Retrieve a belief if it exists"""
        return self.beliefs.get(key)
    
    def get_confident_beliefs(self, threshold: float = 0.6) -> Dict[str, ConfidentBelief]:
        """Get only beliefs that meet a confidence threshold - useful for decision-making"""
        return {
            key: belief for key, belief in self.beliefs.items()
            if belief.current_confidence() >= threshold
        }
    
    def get_uncertain_beliefs(self, threshold: float = 0.4) -> Dict[str, ConfidentBelief]:
        """Identify beliefs that might need verification or updating"""
        return {
            key: belief for key, belief in self.beliefs.items()
            if belief.current_confidence() <= threshold
        }
    
    def assess_knowledge_quality(self) -> Dict[str, Any]:
        """
        Analyze the overall quality of the agent's knowledge base.
        
        This enables meta-cognitive awareness - understanding the
        reliability of one's own knowledge.
        """
        if not self.beliefs:
            return {"status": "no_beliefs", "confidence": 0.0}
        
        confidences = [belief.current_confidence() for belief in self.beliefs.values()]
        avg_confidence = sum(confidences) / len(confidences)
        
        reliable_count = len(self.get_confident_beliefs(0.7))
        uncertain_count = len(self.get_uncertain_beliefs(0.4))
        
        return {
            "average_confidence": avg_confidence,
            "total_beliefs": len(self.beliefs),
            "reliable_beliefs": reliable_count,
            "uncertain_beliefs": uncertain_count,
            "knowledge_stability": reliable_count / len(self.beliefs),
            "needs_verification": uncertain_count > len(self.beliefs) * 0.3
        }
    
    def suggest_information_needs(self) -> List[str]:
        """
        Identify what information the agent should seek to improve its knowledge.
        
        This enables proactive learning - agents actively seeking information
        to fill gaps in their understanding.
        """
        suggestions = []
        
        # Find beliefs that need verification due to low confidence
        uncertain_beliefs = self.get_uncertain_beliefs(0.5)
        if uncertain_beliefs:
            suggestions.append(f"Verify uncertain beliefs: {list(uncertain_beliefs.keys())[:3]}")
        
        # Find stale beliefs that might need updating
        stale_beliefs = [key for key, belief in self.beliefs.items() if belief.is_stale()]
        if stale_beliefs:
            suggestions.append(f"Update stale information: {stale_beliefs[:3]}")
        
        # Suggest areas where more evidence would help build confidence
        low_evidence = [
            key for key, belief in self.beliefs.items() 
            if belief.evidence_count < 2 and belief.current_confidence() < 0.8
        ]
        if low_evidence:
            suggestions.append(f"Gather more evidence for: {low_evidence[:3]}")
        
        return suggestions

class EnhancedMentalState:
    """
    Your evolved mental state that combines sophisticated belief management
    with your existing capabilities and obligations model.
    
    This replaces your current MentalState while maintaining full compatibility
    with your existing agent code.
    """
    
    def __init__(self, agent_name: str):
        # Your existing mental state components - these remain exactly the same
        self.capabilities: List[str] = []
        self.obligations: List[str] = []
        self.decisions: List[Dict[str, Any]] = []
        
        # Enhanced cognitive components that add sophisticated reasoning
        self.belief_system = CognitiveBeliefSystem(agent_name)
        self.reasoning_context: Dict[str, Any] = {}
        
        # Configurable confidence thresholds for different types of decisions
        self.confidence_thresholds = {
            "action": 0.6,        # Minimum confidence to take action
            "collaboration": 0.4,  # When to ask for help from other agents
            "certainty": 0.8      # When to express high confidence in responses
        }
        
        # Compatibility layer - your existing code can still access beliefs as a dict
        self.beliefs = CognitiveBeliefProxy(self.belief_system)
    
    def update_belief(self, key: str, value: Any, **kwargs) -> ConfidentBelief:
        """Convenient wrapper for belief updates that works with your existing code"""
        return self.belief_system.believe(key, value, **kwargs)
    
    def get_reliable_facts(self) -> Dict[str, Any]:
        """Get facts the agent is confident about - useful for decision-making"""
        confident_beliefs = self.belief_system.get_confident_beliefs(
            self.confidence_thresholds["certainty"]
        )
        return {key: belief.value for key, belief in confident_beliefs.items()}
    
    def should_seek_help(self, topic: str) -> bool:
        """Determine if agent should request collaboration on this topic"""
        belief = self.belief_system.get_belief(topic)
        if not belief:
            return True  # Don't know anything about this topic
        
        return belief.current_confidence() < self.confidence_thresholds["collaboration"]
    
    def express_uncertainty(self, topic: str) -> Optional[str]:
        """Generate appropriate uncertainty expressions for responses"""
        belief = self.belief_system.get_belief(topic)
        if not belief:
            return "I don't have information about this topic."
        
        confidence = belief.current_confidence()
        if confidence < 0.3:
            return f"I'm quite uncertain about {topic}. My confidence is only {confidence:.1%}."
        elif confidence < 0.6:
            return f"I have some information about {topic}, but I'm not very confident ({confidence:.1%})."
        
        return None  # Confident enough not to express uncertainty

class CognitiveBeliefProxy:
    """
    A proxy that makes the enhanced belief system compatible with your existing code.
    
    This allows your existing agents to continue using beliefs like a dictionary
    while gaining all the cognitive enhancements behind the scenes.
    """
    
    def __init__(self, belief_system: CognitiveBeliefSystem):
        self._belief_system = belief_system
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access: belief_system['key']"""
        belief = self._belief_system.get_belief(key)
        return belief.value if belief else None
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style assignment: belief_system['key'] = value"""
        self._belief_system.believe(key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style get method with default"""
        belief = self._belief_system.get_belief(key)
        return belief.value if belief else default
    
    def update(self, other: Dict[str, Any]):
        """Dictionary-style update method for compatibility with existing code"""
        for key, value in other.items():
            self._belief_system.believe(key, value)
    
    def keys(self):
        """Return keys like a dictionary"""
        return self._belief_system.beliefs.keys()
    
    def values(self):
        """Return values like a dictionary"""
        return [belief.value for belief in self._belief_system.beliefs.values()]
    
    def items(self):
        """Return items like a dictionary"""
        return [(key, belief.value) for key, belief in self._belief_system.beliefs.items()]
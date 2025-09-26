from enum import Enum
import logging
import uuid
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import redis

class AgentCapability(Enum):
    GENERATE_RESPONSE = "generate_response"
    MAINTAIN_CONVERSATION = "maintain_conversation"
    COORDINATE_AGENTS = "coordinate_agents"
    RETRIEVE_DATA = "retrieve_data"
    RANK_CONTENT = "rank_content"
    PROVIDE_RECOMMENDATIONS = "provide_recommendations"
    GENERATE_ARTICLE = "generate_article"  
    EVALUATE_ARTICLE = "evaluate_article"
    PROCESS_FEEDBACK = "process_feedback"

class ConfidentBelief:
    def __init__(self, value: Any, confidence: float = 0.5, source: str = "", 
                 timestamp: Optional[datetime] = None, decay_rate: float = 0.1):
        self.value = value
        self.confidence = max(0.0, min(1.0, confidence))  
        self.source = source
        self.timestamp = timestamp or datetime.now()
        self.decay_rate = decay_rate
        self.access_count = 0
        self.last_accessed = self.timestamp

    def get_current_confidence(self) -> float:
        if self.decay_rate == 0:
            return self.confidence
        hours_passed = (datetime.now() - self.timestamp).total_seconds() / 3600
        decayed_confidence = self.confidence * (1 - self.decay_rate * hours_passed / 24)
        return max(0.0, decayed_confidence)

    def access(self):
        self.access_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "decay_rate": self.decay_rate,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfidentBelief':
        belief = cls(
            value=data["value"],
            confidence=data["confidence"],
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            decay_rate=data.get("decay_rate", 0.1)
        )
        belief.access_count = data.get("access_count", 0)
        belief.last_accessed = datetime.fromisoformat(data.get("last_accessed", belief.timestamp.isoformat()))
        return belief

class CompetencyModel:
    def __init__(self):
        self.competencies = {}
        self.performance_history = []

    def add_competency(self, task_type: str, success_rate: float, conditions: Dict[str, Any] = None):
        self.competencies[task_type] = {
            "success_rate": success_rate,
            "conditions": conditions or {},
            "last_updated": datetime.now(),
            "usage_count": self.competencies.get(task_type, {}).get("usage_count", 0) + 1
        }

    def get_competency(self, task_type: str) -> Optional[Dict[str, Any]]:
        return self.competencies.get(task_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "competencies": {k: {**v, "last_updated": v["last_updated"].isoformat()} 
                           for k, v in self.competencies.items()},
            "performance_history": self.performance_history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompetencyModel':
        model = cls()
        competencies_data = data.get("competencies", {})
        for task_type, comp_data in competencies_data.items():
            comp_data_copy = comp_data.copy()
            comp_data_copy["last_updated"] = datetime.fromisoformat(comp_data_copy["last_updated"])
            model.competencies[task_type] = comp_data_copy
        model.performance_history = data.get("performance_history", [])
        return model

class EnhancedMentalState:
    def __init__(self, agent_id: str, redis_client: Optional[redis.Redis] = None):
        self.agent_id = agent_id
        self.redis_client = redis_client
        self.capabilities: List[AgentCapability] = []
        self.obligations: List[str] = []
        self.beliefs: Dict[str, ConfidentBelief] = {}
        self.decisions: List[Dict[str, Any]] = []
        self.competency_model = CompetencyModel()
        self.reflection_patterns: List[Dict[str, Any]] = []
        self.collaborative_requests: List[Dict[str, Any]] = []
        self.vector_memory = None
        self.MemoryType = None
        self._initialize_vector_memory(redis_client)
        
    def _initialize_vector_memory(self, redis_client: Optional[redis.Redis]):
        try:
            from orchestrator.memory.vector_memory_manager import VectorMemoryManager, MemoryType # type: ignore
            if redis_client:
                self.vector_memory = VectorMemoryManager(redis_client)
                self.MemoryType = MemoryType
                logging.info(f"✅ Vector memory enabled for agent {self.agent_id}")
            else:
                logging.warning(f"⚠️ No Redis client provided for agent {self.agent_id}")
        except ImportError as e:
            logging.warning(f"⚠️ Vector memory not available for agent {self.agent_id}: Import failed - {e}")
        except Exception as e:
            logging.error(f"❌ Vector memory initialization failed for agent {self.agent_id}: {e}")
        
    def add_belief(self, key: str, value: Any, confidence: float = 0.5, source: str = ""):
        self.beliefs[key] = ConfidentBelief(value, confidence, source)
        if self.vector_memory and self.MemoryType:
            try:
                content = f"Belief: {key} = {value}"
                metadata = {"key": key, "source": source}
                self.vector_memory.store_memory(
                    agent_id=self.agent_id,
                    memory_type=self.MemoryType.BELIEF,
                    content=content,
                    metadata=metadata,
                    confidence=confidence
                )
            except Exception as e:
                logging.error(f"Failed to store belief in semantic memory: {e}")
        self._persist_if_possible()

    def get_belief(self, key: str) -> Optional[Any]:
        if key in self.beliefs:
            self.beliefs[key].access()
            return self.beliefs[key].value
        return None

    def get_belief_confidence(self, key: str) -> float:
        if key in self.beliefs:
            return self.beliefs[key].get_current_confidence()
        return 0.0

    def update_belief_confidence(self, key: str, new_confidence: float, source: str = ""):
        if key in self.beliefs:
            self.beliefs[key].confidence = max(0.0, min(1.0, new_confidence))
            self.beliefs[key].source = source or self.beliefs[key].source
            self.beliefs[key].timestamp = datetime.now()
            self._persist_if_possible()

    def add_decision(self, decision: Dict[str, Any]):
        decision_entry = {
            **decision,
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id
        }
        self.decisions.append(decision_entry)
        if self.vector_memory and self.MemoryType:
            try:
                content = f"Decision: {decision.get('action', 'unknown')} - {decision.get('reasoning', '')}"
                metadata = {
                    "action": decision.get("action"),
                    "confidence": decision.get("confidence", 0.5),
                    "outcome": decision.get("outcome")
                }
                self.vector_memory.store_memory(
                    agent_id=self.agent_id,
                    memory_type=self.MemoryType.DECISION,
                    content=content,
                    metadata=metadata,
                    confidence=decision.get("confidence", 0.5)
                )
            except Exception as e:
                logging.error(f"Failed to store decision in semantic memory: {e}")
        if len(self.decisions) > 100:
            self.decisions = self.decisions[-100:]
        self._persist_if_possible()
    def request_collaboration(self, agent_type: str, reasoning_type: str, context: Dict[str, Any]):
        request = {
            "agent_type": agent_type,
            "reasoning_type": reasoning_type,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "requester_id": self.agent_id,
            "confidence_threshold": 0.6
        }
        self.collaborative_requests.append(request)
        if self.vector_memory and self.MemoryType:
            try:
                content = f"Collaboration request: {agent_type} for {reasoning_type}"
                self.vector_memory.store_memory(
                    agent_id=self.agent_id,
                    memory_type=self.MemoryType.COLLABORATION,
                    content=content,
                    metadata=request,
                    confidence=0.6
                )
            except Exception as e:
                logging.error(f"Failed to store collaboration in semantic memory: {e}")
        if self.redis_client:
            try:
                self.redis_client.publish(f"collaboration:{agent_type}", json.dumps(request, default=str))
            except Exception as e:
                logging.error(f"Failed to publish collaboration request: {e}")

    def should_request_help(self, task_type: str, current_confidence: float = 0.5) -> bool:
        if current_confidence < 0.6:
            return True
        if hasattr(self, 'recall_similar_experiences'):
            similar_experiences = self.recall_similar_experiences(f"collaboration {task_type}", max_results=3)
            successful_collaborations = [exp for exp in similar_experiences 
                                       if "collaboration" in exp.content and exp.confidence > 0.7]
            if len(successful_collaborations) >= 2:
                return True
        return False
    def add_reflection(self, reflection: Dict[str, Any]):
        reflection_entry = {
            **reflection,
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id
        }
        self.reflection_patterns.append(reflection_entry)
        if self.vector_memory and self.MemoryType:
            try:
                content = f"Reflection: {reflection.get('lessons_learned', '')} - {reflection.get('performance_notes', '')}"
                metadata = {
                    "success": reflection.get("success"),
                    "operation": reflection.get("operation"),
                    "quality_score": reflection.get("quality_score")
                }
                self.vector_memory.store_memory(
                    agent_id=self.agent_id,
                    memory_type=self.MemoryType.REFLECTION,
                    content=content,
                    metadata=metadata,
                    confidence=0.8
                )
            except Exception as e:
                logging.error(f"Failed to store reflection in semantic memory: {e}")
        if len(self.reflection_patterns) > 50:
            self.reflection_patterns = self.reflection_patterns[-50:]
        self._persist_if_possible()

    def add_experience(self, experience_description: str, outcome: str, 
                      confidence: float = 0.7, metadata: Dict[str, Any] = None):
        if self.vector_memory and self.MemoryType:
            try:
                content = f"Experience: {experience_description} -> {outcome}"
                self.vector_memory.store_memory(
                    agent_id=self.agent_id,
                    memory_type=self.MemoryType.EXPERIENCE,
                    content=content,
                    metadata=metadata or {},
                    confidence=confidence
                )
                logging.info(f"✅ Stored experience for agent {self.agent_id}: {experience_description[:50]}...")
            except Exception as e:
                logging.error(f"Failed to store experience in semantic memory: {e}")

    def recall_similar_experiences(self, query: str, max_results: int = 5) -> List[Any]:
        if not self.vector_memory:
            return []
        try:
            return self.vector_memory.search_memories(
                query=query,
                agent_id=self.agent_id,
                max_results=max_results
            )
        except Exception as e:
            logging.error(f"Failed to recall experiences: {e}")
            return []

    def get_semantic_insights(self, query: str) -> Dict[str, Any]:
        if not self.vector_memory:
            return {"insights": [], "related_memories": []}
        try:
            memories = self.vector_memory.search_memories(
                query=query,
                agent_id=self.agent_id,
                max_results=10
            )
            insights = {
                "beliefs": [m for m in memories if m.memory_type.value == "belief"],
                "decisions": [m for m in memories if m.memory_type.value == "decision"],
                "reflections": [m for m in memories if m.memory_type.value == "reflection"],
                "experiences": [m for m in memories if m.memory_type.value == "experience"]
            }
            return {
                "query": query,
                "total_relevant_memories": len(memories),
                "insights": insights,
                "summary": self._generate_insight_summary(insights)
            }
        except Exception as e:
            logging.error(f"Failed to get semantic insights: {e}")
            return {"insights": [], "related_memories": []}

    def _generate_insight_summary(self, insights: Dict[str, List]) -> str:
        summary_parts = []
        if insights["beliefs"]:
            summary_parts.append(f"Found {len(insights['beliefs'])} relevant beliefs")
        if insights["decisions"]:
            summary_parts.append(f"Found {len(insights['decisions'])} related decisions")
        if insights["reflections"]:
            summary_parts.append(f"Found {len(insights['reflections'])} reflections")
        if insights["experiences"]:
            summary_parts.append(f"Found {len(insights['experiences'])} similar experiences")
        return "; ".join(summary_parts) if summary_parts else "No relevant insights found"

    def request_collaboration(self, agent_type: str, reasoning_type: str, context: Dict[str, Any]):
        request = {
            "agent_type": agent_type,
            "reasoning_type": reasoning_type,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "requester_id": self.agent_id
        }
        self.collaborative_requests.append(request)
        if self.vector_memory and self.MemoryType:
            try:
                content = f"Collaboration request: {agent_type} for {reasoning_type}"
                self.vector_memory.store_memory(
                    agent_id=self.agent_id,
                    memory_type=self.MemoryType.COLLABORATION,
                    content=content,
                    metadata=request,
                    confidence=0.6
                )
            except Exception as e:
                logging.error(f"Failed to store collaboration in semantic memory: {e}")
        if self.redis_client:
            try:
                self.redis_client.publish(f"collaboration:{agent_type}", json.dumps(request, default=str))
            except Exception as e:
                logging.error(f"Failed to publish collaboration request: {e}")

    def _persist_if_possible(self):
        if self.redis_client:
            try:
                state_data = self.to_dict()
                self.redis_client.set(f"mental_state:{self.agent_id}", 
                                    json.dumps(state_data, default=str))
                self.redis_client.expire(f"mental_state:{self.agent_id}", 3600)
            except Exception as e:
                logging.error(f"Failed to persist mental state for {self.agent_id}: {e}")

    def load_from_redis(self):
        if self.redis_client:
            try:
                data = self.redis_client.get(f"mental_state:{self.agent_id}")
                if data:
                    state_data = json.loads(data)
                    self.from_dict(state_data)
            except Exception as e:
                logging.error(f"Failed to load mental state for {self.agent_id}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        beliefs_dict = {}
        for k, v in self.beliefs.items():
            if hasattr(v, 'to_dict'):
                beliefs_dict[k] = v.to_dict()
            else:
                beliefs_dict[k] = v
        return {
            "agent_id": self.agent_id,
            "capabilities": [cap.value for cap in self.capabilities],
            "obligations": self.obligations,
            "beliefs": beliefs_dict,
            "decisions": self.decisions,
            "competency_model": self.competency_model.to_dict() if self.competency_model else {},
            "reflection_patterns": self.reflection_patterns,
            "collaborative_requests": self.collaborative_requests
        }

    def from_dict(self, data: Dict[str, Any]):
        self.agent_id = data.get("agent_id", self.agent_id)
        self.capabilities = [AgentCapability(cap) for cap in data.get("capabilities", [])]
        self.obligations = data.get("obligations", [])
        beliefs_data = data.get("beliefs", {})
        self.beliefs = {k: ConfidentBelief.from_dict(v) for k, v in beliefs_data.items()}
        self.decisions = data.get("decisions", [])
        competency_data = data.get("competency_model", {})
        self.competency_model = CompetencyModel.from_dict(competency_data)
        self.reflection_patterns = data.get("reflection_patterns", [])
        self.collaborative_requests = data.get("collaborative_requests", [])

class BaseAgent:
    OBJECTIVE = "Base agent objective"

    def __init__(self, name: str, redis_client: Optional[redis.Redis] = None):
        self.name = name
        self.agent_id = f"{name}_{str(uuid.uuid4())[:8]}"
        self.redis_client = redis_client
        self.mental_state = EnhancedMentalState(self.agent_id, redis_client)
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        if redis_client:
            self.mental_state.load_from_redis()
        if self.mental_state.vector_memory:
            self.logger.info(f"✅ Agent {self.name} initialized with semantic memory")
        else:
            self.logger.warning(f"⚠️ Agent {self.name} initialized without semantic memory")

    def _ensure_model_manager(self):
        if not hasattr(self, 'model_manager'):
            from orchestrator.core.model_manager import ModelManager # type: ignore
            self.model_manager = ModelManager(redis_client=self.redis_client)
            self.logger.warning(f"⚠️ {self.name} created its own ModelManager - should use shared instance")

    def log(self, message: str) -> None:
        self.logger.info(f"{self.name} - {message}")

    def _perceive(self, input_data: Dict[str, Any]) -> None:
        self.log("Perceiving input data")
        for key, value in input_data.items():
            confidence = 0.9 if value else 0.1
            self.mental_state.add_belief(f"input_{key}", value, confidence, "perception")

    def _act(self) -> Dict[str, Any]:
        self.log("Acting based on current beliefs and competencies")
        decision = {
            "action": "base_action",
            "reasoning": "base reasoning",
            "confidence": 0.5,
            "beliefs_used": list(self.mental_state.beliefs.keys())
        }
        self.mental_state.add_decision(decision)
        return {}

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        self.log("Reflecting on action result")
        success = action_result.get("workflow_status", "failure") == "success"
        reflection = {
            "action_taken": "base_action",
            "result": action_result,
            "success": success,
            "lessons_learned": "Base reflection pattern",
            "confidence_adjustments": {}
        }
        self.mental_state.add_reflection(reflection)
        task_type = action_result.get("task_type", "general")
        current_success_rate = self.mental_state.competency_model.get_competency(task_type)
        if current_success_rate:
            old_rate = current_success_rate["success_rate"]
            new_rate = (old_rate + (1.0 if success else 0.0)) / 2
        else:
            new_rate = 1.0 if success else 0.0
        self.mental_state.competency_model.add_competency(task_type, new_rate)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self._perceive(input_data)
        action_result = self._act()
        self._rethink(action_result)
        return action_result

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(input_data)

    def get_mental_state_summary(self) -> Dict[str, Any]:
        beliefs_summary = {}
        for key, belief in self.mental_state.beliefs.items():
            beliefs_summary[key] = {
                "value": belief.value,
                "confidence": belief.get_current_confidence(),
                "source": belief.source,
                "access_count": belief.access_count
            }
        return {
            "agent_id": self.agent_id,
            "capabilities": [cap.value for cap in self.mental_state.capabilities],
            "obligations": self.mental_state.obligations,
            "beliefs_count": len(self.mental_state.beliefs),
            "beliefs_summary": beliefs_summary,
            "decisions_count": len(self.mental_state.decisions),
            "reflections_count": len(self.mental_state.reflection_patterns),
            "competencies": self.mental_state.competency_model.competencies,
            "has_semantic_memory": self.mental_state.vector_memory is not None
        }

    def get_semantic_context(self, query: str) -> str:
        if self.mental_state.vector_memory:
            insights = self.mental_state.get_semantic_insights(query)
            context_parts = []
            for memory_type, memories in insights["insights"].items():
                if memories:
                    context_parts.append(f"{memory_type.title()}: {len(memories)} relevant memories")
                    for memory in memories[:2]:
                        context_parts.append(f"  - {memory.content[:100]}...")
            return "\n".join(context_parts) if context_parts else "No relevant context found"
        return "Semantic memory not available"

class MentalState:
    def __init__(self):
        self.capabilities: List[AgentCapability] = []
        self.obligations: List[str] = []
        self.beliefs: Dict[str, Any] = {}
        self.decisions: List[Dict[str, Any]] = []
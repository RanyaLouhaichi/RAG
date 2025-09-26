import redis # type: ignore
import json
import numpy as np
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer # type: ignore
import logging
from dataclasses import dataclass
from enum import Enum

class MemoryType(Enum):
    """Types of memories that can be stored"""
    BELIEF = "belief"
    DECISION = "decision"
    REFLECTION = "reflection"
    EXPERIENCE = "experience"
    COLLABORATION = "collaboration"
    INSIGHT = "insight"

@dataclass
class SemanticMemory:
    """A memory with semantic understanding"""
    id: str
    agent_id: str
    memory_type: MemoryType
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    vector: Optional[List[float]] = None
    confidence: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None

class VectorMemoryManager:
    """Manages semantic memory using vector embeddings"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.logger = logging.getLogger("VectorMemoryManager")
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384  
            self.logger.info("Initialized SentenceTransformer for embeddings")
        except Exception as e:
            self.logger.error(f"Failed to initialize SentenceTransformer: {e}")
            raise
        self.max_memories_per_agent = 1000
        self.similarity_threshold = 0.5  
        self.memory_decay_days = 30
        self._initialize_vector_indices()
        
    def _initialize_vector_indices(self):
        """Initialize Redis vector search indices"""
        try:
            self.logger.info("Vector indices initialized (using Redis hash simulation)")
        except Exception as e:
            self.logger.warning(f"Vector index initialization failed: {e}")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text"""
        try:
            embedding = self.encoder.encode(text)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * self.embedding_dim

    def _calculate_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            similarity = dot_product / (norm_v1 * norm_v2)
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0

    def store_memory(self, agent_id: str, memory_type: MemoryType, content: str,
                    metadata: Dict[str, Any] = None, confidence: float = 0.5) -> str:
        """Store a memory with semantic embedding"""
        try:
            memory_id = f"memory:{agent_id}:{memory_type.value}:{hashlib.md5(content.encode()).hexdigest()[:8]}"
            vector = self._generate_embedding(content)
            memory = SemanticMemory(
                id=memory_id,
                agent_id=agent_id,
                memory_type=memory_type,
                content=content,
                metadata=metadata or {},
                timestamp=datetime.now(),
                vector=vector,
                confidence=confidence
            )
            memory_data = {
                "id": memory.id,
                "agent_id": memory.agent_id,
                "memory_type": memory.memory_type.value,
                "content": memory.content,
                "metadata": json.dumps(memory.metadata),
                "timestamp": memory.timestamp.isoformat(),
                "vector": json.dumps(memory.vector),
                "confidence": memory.confidence,
                "access_count": memory.access_count
            }
            self.redis_client.hset(memory_id, mapping=memory_data)
            self.redis_client.expire(memory_id, int(timedelta(days=self.memory_decay_days).total_seconds()))
            agent_memories_key = f"agent_memories:{agent_id}"
            self.redis_client.sadd(agent_memories_key, memory_id)
            self.redis_client.expire(agent_memories_key, int(timedelta(days=self.memory_decay_days).total_seconds()))
            type_memories_key = f"type_memories:{memory_type.value}"
            self.redis_client.sadd(type_memories_key, memory_id)
            self.redis_client.expire(type_memories_key, int(timedelta(days=self.memory_decay_days).total_seconds()))
            self.logger.info(f"Stored memory {memory_id} for agent {agent_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            return ""

    def search_memories(self, query: str, agent_id: Optional[str] = None,
                       memory_type: Optional[MemoryType] = None,
                       max_results: int = 10) -> List[SemanticMemory]:
        """Search memories by semantic similarity"""
        try:
            query_vector = self._generate_embedding(query)
            candidates = self._get_candidate_memories(agent_id, memory_type)
            scored_memories = []
            for memory_id in candidates:
                memory = self._load_memory(memory_id)
                if memory and memory.vector:
                    similarity = self._calculate_similarity(query_vector, memory.vector)
                    if similarity >= self.similarity_threshold:
                        scored_memories.append((similarity, memory))
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            results = [memory for _, memory in scored_memories[:max_results]]
            for memory in results:
                self._update_access_stats(memory.id)
            
            self.logger.info(f"Found {len(results)} relevant memories for query: {query[:50]}")
            return results
            
        except Exception as e:
            self.logger.error(f"Memory search failed: {e}")
            return []

    def _get_candidate_memories(self, agent_id: Optional[str] = None,
                              memory_type: Optional[MemoryType] = None) -> List[str]:
        """Get candidate memory IDs for search"""
        try:
            if agent_id and memory_type:
                agent_key = f"agent_memories:{agent_id}"
                type_key = f"type_memories:{memory_type.value}"
                temp_key = f"temp_intersection:{agent_id}:{memory_type.value}"
                
                self.redis_client.sinterstore(temp_key, agent_key, type_key)
                candidates = list(self.redis_client.smembers(temp_key))
                self.redis_client.delete(temp_key)
                
            elif agent_id:
                agent_key = f"agent_memories:{agent_id}"
                candidates = list(self.redis_client.smembers(agent_key))
                
            elif memory_type:
                type_key = f"type_memories:{memory_type.value}"
                candidates = list(self.redis_client.smembers(type_key))
                
            else:
                candidates = list(self.redis_client.keys("memory:*"))[:1000]  
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Failed to get candidate memories: {e}")
            return []

    def _load_memory(self, memory_id: str) -> Optional[SemanticMemory]:
        """Load memory from Redis"""
        try:
            data = self.redis_client.hgetall(memory_id)
            if not data:
                return None
            
            return SemanticMemory(
                id=data["id"],
                agent_id=data["agent_id"],
                memory_type=MemoryType(data["memory_type"]),
                content=data["content"],
                metadata=json.loads(data.get("metadata", "{}")),
                timestamp=datetime.fromisoformat(data["timestamp"]),
                vector=json.loads(data.get("vector", "[]")),
                confidence=float(data.get("confidence", 0.5)),
                access_count=int(data.get("access_count", 0)),
                last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load memory {memory_id}: {e}")
            return None

    def _update_access_stats(self, memory_id: str):
        """Update memory access statistics"""
        try:
            current_count = self.redis_client.hget(memory_id, "access_count") or "0"
            new_count = int(current_count) + 1
            
            self.redis_client.hset(memory_id, mapping={
                "access_count": new_count,
                "last_accessed": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Failed to update access stats for {memory_id}: {e}")

    def find_related_memories(self, memory_id: str, max_results: int = 5) -> List[SemanticMemory]:
        """Find memories related to a specific memory"""
        try:
            source_memory = self._load_memory(memory_id)
            if not source_memory:
                return []
            return self.search_memories(
                query=source_memory.content,
                agent_id=None,  
                max_results=max_results + 1  
            )[1:]  
            
        except Exception as e:
            self.logger.error(f"Failed to find related memories: {e}")
            return []

    def get_agent_memory_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get summary of agent's memories"""
        try:
            agent_key = f"agent_memories:{agent_id}"
            memory_ids = list(self.redis_client.smembers(agent_key))
            memories = []
            type_counts = {}
            total_confidence = 0
            
            for memory_id in memory_ids:
                memory = self._load_memory(memory_id)
                if memory:
                    memories.append(memory)
                    type_counts[memory.memory_type.value] = type_counts.get(memory.memory_type.value, 0) + 1
                    total_confidence += memory.confidence
            avg_confidence = total_confidence / len(memories) if memories else 0
            recent_memories = sorted(memories, key=lambda m: m.timestamp, reverse=True)[:5]
            most_accessed = sorted(memories, key=lambda m: m.access_count, reverse=True)[:3]
            
            return {
                "agent_id": agent_id,
                "total_memories": len(memories),
                "memory_types": type_counts,
                "average_confidence": avg_confidence,
                "recent_memories": [{"id": m.id, "content": m.content[:100], "type": m.memory_type.value} for m in recent_memories],
                "most_accessed": [{"id": m.id, "content": m.content[:100], "access_count": m.access_count} for m in most_accessed]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory summary for {agent_id}: {e}")
            return {}

    def consolidate_memories(self, agent_id: str, similarity_threshold: float = 0.9):
        """Consolidate similar memories to prevent redundancy"""
        try:
            agent_key = f"agent_memories:{agent_id}"
            memory_ids = list(self.redis_client.smembers(agent_key))
            
            memories = []
            for memory_id in memory_ids:
                memory = self._load_memory(memory_id)
                if memory:
                    memories.append(memory)
            consolidated_count = 0
            for i, memory1 in enumerate(memories):
                for j, memory2 in enumerate(memories[i+1:], i+1):
                    if memory1.memory_type == memory2.memory_type:
                        similarity = self._calculate_similarity(memory1.vector, memory2.vector)
                        
                        if similarity >= similarity_threshold:
                            if memory1.confidence >= memory2.confidence:
                                self._delete_memory(memory2.id)
                                consolidated_count += 1
                            else:
                                self._delete_memory(memory1.id)
                                consolidated_count += 1
                            break
            
            self.logger.info(f"Consolidated {consolidated_count} memories for agent {agent_id}")
            return consolidated_count
            
        except Exception as e:
            self.logger.error(f"Memory consolidation failed for {agent_id}: {e}")
            return 0

    def _delete_memory(self, memory_id: str):
        """Delete a memory and clean up indices"""
        try:
            memory_data = self.redis_client.hgetall(memory_id)
            if memory_data:
                agent_id = memory_data.get("agent_id")
                memory_type = memory_data.get("memory_type")
                if agent_id:
                    self.redis_client.srem(f"agent_memories:{agent_id}", memory_id)
                if memory_type:
                    self.redis_client.srem(f"type_memories:{memory_type}", memory_id)
            self.redis_client.delete(memory_id)   
        except Exception as e:
            self.logger.error(f"Failed to delete memory {memory_id}: {e}")

    def get_memory_insights(self) -> Dict[str, Any]:
        """Get system-wide memory insights"""
        try:
            memory_keys = self.redis_client.keys("memory:*")
            agent_counts = {}
            type_counts = {}
            total_memories = len(memory_keys)
            
            for memory_key in memory_keys[:100]: 
                memory = self._load_memory(memory_key)
                if memory:
                    agent_counts[memory.agent_id] = agent_counts.get(memory.agent_id, 0) + 1
                    type_counts[memory.memory_type.value] = type_counts.get(memory.memory_type.value, 0) + 1
            return {
                "total_memories": total_memories,
                "memories_by_agent": agent_counts,
                "memories_by_type": type_counts,
                "memory_indices": {
                    "agent_indices": len(self.redis_client.keys("agent_memories:*")),
                    "type_indices": len(self.redis_client.keys("type_memories:*"))
                }
            }    
        except Exception as e:
            self.logger.error(f"Failed to get memory insights: {e}")
            return {}
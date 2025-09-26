# VectorMemoryManager: Semantic Experience Memory

## Overview

`VectorMemoryManager` provides a semantic memory layer for agents and workflows, enabling storage and retrieval of experiences, beliefs, and knowledge using vector embeddings. It supports multiple memory types (episodic, semantic, experience) and integrates with Redis and vector databases for scalable, high-performance memory.

## Key Features

- **Memory Types:** Supports EXPERIENCE, BELIEF, and other memory types for flexible agent cognition.
- **Embedding Storage:** Stores content and metadata as vector embeddings for similarity search and retrieval.
- **Semantic Search:** Enables agents to recall relevant experiences, patterns, or knowledge based on context or queries.
- **Integration:** Used for storing workflow checkpoints, completions, and agent experiences for meta-learning and analytics.

## Example Usage

```python
vm = VectorMemoryManager(redis_client)
vm.store_memory(
    agent_id="workflow_general_orchestration",
    memory_type=MemoryType.EXPERIENCE,
    content="Workflow completed successfully.",
    metadata={"workflow_id": "wf_123"},
    confidence=0.9
)
results = vm.search_memories(query="workflow completed", agent_id="workflow_system")
```

## Advanced Capabilities

- **Confidence Scoring:** Associates confidence levels with each memory for adaptive retrieval.
- **Metadata Indexing:** Stores rich metadata for advanced filtering and analytics.
- **Agent Experience Replay:** Supports experience-driven learning and workflow optimization.

## Integration

VectorMemoryManager is used by the persistent workflow state manager, agents, and orchestrator for semantic checkpointing, experience replay, and research analytics.

---

# JurixSharedMemory: Distributed Agent Memory

## Overview

`JurixSharedMemory` is the shared memory abstraction for all agents and orchestrator components in JURIX. It provides a unified interface for storing, retrieving, and synchronizing data across agents, workflows, and sessions, backed by Redis for high performance and durability.

## Key Responsibilities

- **Conversation Management:** Stores and retrieves conversation histories for chat and workflow sessions.
- **Agent Coordination:** Enables agents to share intermediate results, articles, recommendations, and context.
- **Caching & Deduplication:** Provides fast caching for expensive computations, ticket data, and workflow outputs.
- **Versioning & Expiry:** Supports versioned storage and automatic expiry for transient or session-specific data.

## Main Features

- `store(key, value)`: Store any serializable object under a unique key.
- `retrieve(key)`: Retrieve stored objects by key.
- `get_conversation(conversation_id)`: Fetch conversation history for a session.
- `set_conversation(conversation_id, history)`: Update conversation history.
- `redis_client`: Direct access to the underlying Redis client for advanced operations.

## Example Usage

```python
memory = JurixSharedMemory()
memory.store("article_draft:TICKET-123", article_data)
history = memory.get_conversation("conv_abc")
```

## Integration

JurixSharedMemory is injected into all agents and orchestrator components, ensuring seamless data sharing and state management across the multi-agent system.

---

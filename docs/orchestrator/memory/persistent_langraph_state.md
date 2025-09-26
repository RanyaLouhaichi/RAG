# Persistent LangGraph Workflow State

## Overview

`LangGraphRedisManager` provides robust, persistent state management for all LangGraph-powered workflows in JURIX. It leverages Redis for fast, durable storage of workflow state, checkpoints, and performance metrics, enabling resumable, auditable, and scalable orchestration.

## Core Concepts

- **PersistentWorkflowState:** A rich TypedDict capturing all relevant workflow context, agent outputs, history, and performance data.
- **Workflow Types:** Supports multiple workflow types (orchestration, productivity, article generation, recommendations) with type-safe state management.
- **Checkpointing:** Periodically saves workflow state and node history, supporting recovery, auditing, and incremental learning.
- **Semantic Memory Integration:** Stores workflow checkpoints and completions in vector memory for experience-driven learning and analytics.
- **Performance Metrics:** Tracks execution times, retry counts, error history, and global workflow statistics for monitoring and optimization.

## Key Features

- **State Creation & Loading:** Creates new workflow states with unique IDs and loads existing states from Redis.
- **Checkpointing:** After each node execution, checkpoints are stored with node results and execution metadata.
- **Resumption & Retry:** Supports resuming paused or failed workflows, with retry limits and error tracking.
- **Completion Handling:** Marks workflows as completed, stores results, and updates global performance metrics.
- **Cleanup:** Periodically removes old or completed workflows to maintain system hygiene.

## Example Usage

```python
manager = LangGraphRedisManager()
state = manager.create_workflow_state(WorkflowType.GENERAL_ORCHESTRATION, initial_state)
manager.checkpoint_workflow(state, current_node="recommendation_agent", node_result=result)
manager.complete_workflow(state, final_result)
```

## Advanced Features

- **Semantic Checkpoint Storage:** Integrates with vector memory to store workflow experiences for future retrieval and meta-learning.
- **Global Insights:** Aggregates workflow performance, success rates, and recent patterns for system analytics and research.
- **Error Handling:** Captures and logs all errors, supporting robust debugging and reliability.

## Integration

Used by the orchestrator to persist all workflow executions, enabling advanced monitoring, analytics, and research metrics for thesis and production deployments.

---

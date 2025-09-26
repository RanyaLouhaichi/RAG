# JurixState: Unified Orchestration State

## Overview

`JurixState` is the central state object used throughout the orchestration workflows in the JURIX platform. It encapsulates all relevant context, intermediate results, and agent outputs as the workflow progresses through various nodes (agents) in the LangGraph-powered orchestration engine.

## Key Responsibilities

- **Context Propagation:** Maintains the user query, intent classification, conversation history, and project context across all workflow steps.
- **Agent Interoperability:** Serves as the shared data structure for all agents (retrieval, recommendation, chat, productivity, etc.), enabling seamless data exchange and collaborative reasoning.
- **Workflow Tracking:** Tracks workflow status, current node, collaboration metadata, and model usage for advanced monitoring and analytics.
- **Extensibility:** Designed to be easily extended with new fields as new workflow types or agent capabilities are introduced.

## Main Fields

- `query`: The original user query or task prompt.
- `intent`: Output of the intent classification agent, used for routing.
- `conversation_id`: Unique identifier for the conversation/session.
- `conversation_history`: List of previous user/agent exchanges.
- `articles`, `recommendations`, `tickets`: Core outputs from retrieval and recommendation agents.
- `workflow_status`, `workflow_stage`: Status and stage tracking for robust orchestration.
- `collaboration_metadata`, `collaboration_trace`: Rich metadata for multi-agent collaboration and traceability.
- `metrics`, `predictions`, `visualization_data`: Used in productivity and predictive workflows for analytics and dashboarding.
- `model_usage_summary`, `langsmith_metrics`: Integrated model usage and LangSmith tracing for research and thesis metrics.

## Usage Pattern

JurixState is passed as input and output to each node in the workflow graph. Each agent reads from and writes to the state, enabling both sequential and collaborative agent reasoning. The state is also used for logging, monitoring, and post-hoc analysis.

## Example

```python
from orchestrator.graph.state import JurixState

state = JurixState(
    query="How can we improve sprint velocity?",
    intent={},
    conversation_id="abc123",
    articles=[],
    recommendations=[],
    tickets=[],
    status="pending",
    # ... other fields ...
)
```

## Extending JurixState

To add new fields, simply extend the TypedDict definition in `state.py`. All agents and orchestrator nodes will automatically propagate the new fields as long as they use `.copy()` and dictionary-style access.

---

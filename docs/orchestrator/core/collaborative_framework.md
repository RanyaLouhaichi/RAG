# Collaborative Framework

The `CollaborativeFramework` is a key component of the orchestrator responsible for enabling and managing collaboration between different agents. It analyzes the needs of a primary agent and coordinates with other agents to provide the necessary support, such as data analysis, content generation, or context enrichment.

## Key Responsibilities:

- **Capability Mapping:** Maintains a map of which agents can fulfill specific collaboration needs.
- **Needs Analysis:** Analyzes the results and mental state of a primary agent to identify when collaboration is required.
- **Orchestration:** Manages the process of collaboration, including selecting the appropriate collaborating agent, preparing the context, and merging the results.
- **Shared Model Management:** Ensures that all agents within the framework use a shared `ModelManager` for consistent and efficient language model access.
- **Performance Tracking:** Monitors the performance of collaborations and provides insights into collaboration patterns.

## Collaboration Needs

The framework defines the following `CollaborationNeed` enums:

- `DATA_ANALYSIS`
- `CONTENT_GENERATION`
- `VALIDATION`
- `CONTEXT_ENRICHMENT`
- `STRATEGIC_REASONING`

## Methods:

- `__init__(...)`: Initializes the framework with a Redis client, an agent registry, and a shared model manager.
- `coordinate_agents(...)`: The main method for coordinating a collaboration. It runs the primary agent, analyzes its needs, and orchestrates the collaboration.
- `_analyze_collaboration_needs(...)`: Analyzes the primary agent's output and mental state to determine if collaboration is needed.
- `_orchestrate_collaboration_fixed(...)`: Manages the collaboration process, including selecting agents and merging results.
- `_prepare_collaboration_context_fixed(...)`: Prepares the context for the collaborating agent.
- `_merge_results_fixed(...)`: Merges the results from the collaborating agent back into the primary agent's results.
- `get_collaboration_insights()`: Provides insights into collaboration patterns and performance.

## Usage:

The `CollaborativeFramework` is used by the main orchestrator to handle complex tasks that require the expertise of multiple agents. By enabling agents to work together, it significantly enhances the overall capabilities and problem-solving abilities of the system.

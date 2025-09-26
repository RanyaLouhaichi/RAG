# Chat Agent MCP Server

The `ChatAgentMCPServer` is an MCP (Multi-Agent Communication Protocol) server that exposes the capabilities of the `ChatAgent`. It enables other agents to interact with the chat agent to generate conversational responses, manage conversation context, and coordinate with other agents.

## Key Responsibilities:

- **Tool Registration:** Registers tools for generating responses, maintaining context, and coordinating with other agents.
- **Resource Registration:** Provides access to conversation templates, the agent's mental state, and semantic memory.
- **Prompt Registration:** Exposes prompts for common conversational scenarios like Agile consultations and sprint analysis.

## Registered Tools:

- **`generate_response`**:
    - **Description:** Generates an intelligent conversational response based on a prompt and context.
    - **Input:** `prompt` (string), `session_id` (string), and optional context like `articles`, `recommendations`, `tickets`, `intent`, and `predictions`.
    - **Output:** A dictionary containing the `response`, `articles_used`, and `workflow_status`.

- **`maintain_context`**:
    - **Description:** Maintains and retrieves the conversation context for a given session.
    - **Input:** `session_id` (string).
    - **Output:** A dictionary with the `session_id`, conversation `history`, and a `context_maintained` flag.

- **`coordinate_agents`**:
    - **Description:** Coordinates with other agents to obtain enhanced responses.
    - **Input:** `collaboration_type` (string), `target_agent` (string), and `context` (object).
    - **Output:** A dictionary confirming the collaboration request.

## Registered Resources:

- **`chat://templates/conversations`**:
    - **Name:** Conversation Templates
    - **Description:** Provides pre-defined templates for common conversational scenarios.

- **`chat://state/mental`**:
    - **Name:** Mental State
    - **Description:** Exposes the current mental state and beliefs of the chat agent.

- **`chat://memory/semantic`**:
    - **Name:** Semantic Memory
    - **Description:** Provides access to the chat agent's semantic memory context and insights.

## Registered Prompts:

- **`agile_consultation`**: A prompt for seeking expert advice on Agile methodologies.
- **`sprint_analysis`**: A prompt for analyzing sprint performance and getting improvement recommendations.

## Usage:

This server acts as the communication endpoint for the `ChatAgent`. It allows the orchestrator and other agents to leverage the chat agent's conversational and coordination abilities within the multi-agent system.

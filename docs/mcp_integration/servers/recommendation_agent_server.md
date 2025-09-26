# Recommendation Agent MCP Server

The `RecommendationAgentMCPServer` is an MCP (Multi-Agent Communication Protocol) server that exposes the capabilities of the `RecommendationAgent`. It provides tools for generating, enhancing, and validating strategic recommendations based on project data and user prompts.

## Key Responsibilities:

- **Tool Registration:** Registers tools for generating, enhancing, and validating recommendations.
- **Resource Registration:** Provides access to recommendation templates and quality metrics.
- **Prompt Registration:** Exposes prompts for generating recommendations in specific areas like process optimization and risk mitigation.

## Registered Tools:

- **`generate_recommendations`**:
    - **Description:** Generates intelligent, context-aware recommendations.
    - **Input:** `prompt` (string), and optional context like `session_id`, `project`, `tickets`, `articles`, `workflow_type`, and `predictions`.
    - **Output:** A dictionary containing a list of `recommendations`.

- **`enhance_recommendations`**:
    - **Description:** Enhances existing recommendations with additional context.
    - **Input:** `recommendations` (array), and optional `context`, `project`, `tickets`, and `articles`.
    - **Output:** A dictionary with the original and enhanced recommendation counts, and the list of enhanced recommendations.

- **`validate_recommendations`**:
    - **Description:** Validates the quality of a list of recommendations.
    - **Input:** `recommendations` (array), and optional `project` and `project_context`.
    - **Output:** A dictionary with validation results, including an average quality score and a flag indicating if improvement is needed.

## Registered Resources:

- **`recommendations://templates/strategic`**:
    - **Name:** Strategic Recommendation Templates
    - **Description:** Provides templates for different types of strategic recommendations, such as process improvement and technical debt management.

- **`recommendations://metrics/quality`**:
    - **Name:** Quality Metrics
    - **Description:** Exposes quality metrics and statistics about the generated recommendations.

## Registered Prompts:

- **`process_optimization`**: A prompt to generate recommendations for process optimization.
- **`risk_mitigation`**: A prompt to generate recommendations for mitigating project risks.

## Usage:

This server allows other agents to leverage the `RecommendationAgent`'s ability to provide strategic advice. It is a key component for turning data and analysis into actionable insights for the user.

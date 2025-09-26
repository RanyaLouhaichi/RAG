# Article Generator MCP Server

## Key Responsibilities:

- **Tool Registration:** Registers tools for generating and refining articles.
- **Resource Registration:** Provides access to article templates and quality checklists.
- **Prompt Registration:** Exposes prompts for generating different types of articles.

## Registered Tools:

- **`generate_article`**: 
    - **Description:** Generates a knowledge article from a resolved Jira ticket.
    - **Input:** `ticket_id` (string), `refinement_suggestion` (optional string).
    - **Output:** A dictionary containing the generated `article`, `workflow_status`, and `collaboration_applied` flag.

- **`refine_article`**: 
    - **Description:** Refines an existing article based on feedback.
    - **Input:** `article` (object), `refinement_type` (enum: "technical", "clarity", "completeness", "general").
    - **Output:** A dictionary containing the refined article and related information.

## Registered Resources:

- **`articles://templates/knowledge`**: 
    - **Name:** Knowledge Article Templates
    - **Description:** Provides templates for different types of articles, such as bug resolutions, feature implementations, and troubleshooting guides.

- **`articles://quality/checklist`**: 
    - **Name:** Article Quality Checklist
    - **Description:** Contains quality criteria for knowledge articles, covering structure, content, and metadata.

## Registered Prompts:

- **`generate_kb_article`**: A prompt to generate a comprehensive knowledge base article for a given ticket.
- **`troubleshooting_article`**: A prompt to create a troubleshooting guide for a specific error or component.

## Usage:

This server is started as part of the MCP infrastructure and allows other agents, such as the `ChatAgent`, to request article generation services. It plays a crucial role in automating the creation of documentation from resolved issues.

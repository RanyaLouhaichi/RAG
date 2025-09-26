# Retrieval Agent MCP Server

The `RetrievalAgentMCPServer` is an MCP (Multi-Agent Communication Protocol) server that exposes the capabilities of the `RetrievalAgent`. It provides tools for searching, ranking, and processing articles from a knowledge base.

## Key Responsibilities:

- **Tool Registration:** Registers tools for searching articles, ranking them by relevance, and extracting keywords.
- **Resource Registration:** Provides access to the status of the search index and pre-configured search templates.
- **Prompt Registration:** Exposes prompts for finding documentation and searching for troubleshooting guides.

## Registered Tools:

- **`search_articles`**:
    - **Description:** Searches for relevant Confluence articles based on a query.
    - **Input:** `query` (string), and optional `session_id` and `purpose`.
    - **Output:** A dictionary containing a list of `articles`, `retrieval_quality` metrics, and `workflow_status`.

- **`rank_articles`**:
    - **Description:** Ranks a list of articles by their relevance to a given query.
    - **Input:** `articles` (array) and `query` (string).
    - **Output:** A dictionary with the `ranked_articles`, the `top_article`, and the `ranking_method` used.

- **`extract_keywords`**:
    - **Description:** Extracts keywords from a given text to improve search queries.
    - **Input:** `text` (string).
    - **Output:** A dictionary containing the extracted `keywords`.

## Registered Resources:

- **`retrieval://index/status`**:
    - **Name:** Search Index Status
    - **Description:** Provides the current status of the search index, including the document count and health.

- **`retrieval://templates/searches`**:
    - **Name:** Search Templates
    - **Description:** Offers pre-configured search templates for common topics like Agile best practices and technical documentation.

## Registered Prompts:

- **`find_documentation`**: A prompt to find relevant documentation for a specific topic.
- **`troubleshooting_search`**: A prompt to search for troubleshooting guides and solutions for a particular error.

## Usage:

This server allows other agents to leverage the `RetrievalAgent`'s ability to search and retrieve information from the knowledge base. It is a fundamental component for any agent that needs to access and reason about existing documentation.

# Confluence MCP Server

The `ConfluenceMCPServer` class is an MCP (Multi-Agent Communication Protocol) server that exposes Confluence data and operations. It simulates a Confluence environment, providing tools for searching, creating, updating, and managing articles.

## Key Responsibilities:

- **Tool Registration:** Registers a comprehensive set of tools for interacting with Confluence articles.
- **Resource Registration:** Provides access to Confluence-related resources like spaces, templates, and writing guidelines.
- **Data Simulation:** Simulates a Confluence instance with mock articles for development and testing purposes.

## Registered Tools:

- **`search_articles`**: Searches for articles by keywords, with optional filtering by space and a limit on results.
- **`get_article`**: Retrieves a specific article by its ID.
- **`create_article`**: Creates a new article with a title, content, space, and optional tags.
- **`update_article`**: Updates an existing article's title, content, or tags.
- **`get_related_articles`**: Finds articles related to a given topic or another article.
- **`get_article_templates`**: Retrieves available article templates, with an option to specify a template type.

## Registered Resources:

- **`confluence://spaces/list`**: 
    - **Name:** Confluence Spaces
    - **Description:** Provides a list of available Confluence spaces.

- **`confluence://templates/article`**: 
    - **Name:** Article Templates
    - **Description:** Contains pre-defined templates for different types of articles.

- **`confluence://guidelines/writing`**: 
    - **Name:** Writing Guidelines
    - **Description:** Offers guidelines on writing style and formatting for Confluence articles.

## Usage:

This server is designed to be used within the MCP framework, allowing other agents to access and manipulate Confluence data. By simulating the Confluence API, it enables the development and testing of agents that rely on knowledge base interactions without requiring a live Confluence instance.

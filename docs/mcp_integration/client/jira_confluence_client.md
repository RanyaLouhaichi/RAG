# Jira Confluence MCP Client

The `JiraConfluenceMCPClient` class provides a client interface for interacting with Jira and Confluence data through the MCP (Multi-Agent Communication Protocol). It manages connections to the Jira and Confluence MCP servers and exposes methods for accessing their functionalities.

## Key Responsibilities:

- **Connection Management:** Establishes and terminates connections to the Jira and Confluence MCP servers.
- **Jira Operations:** 
    - Fetches available Jira projects.
    - Retrieves tickets for a specific project, with optional date filtering.
    - Searches for Jira tickets using JQL (Jira Query Language).
    - Analyzes project metrics over a specified period.
- **Confluence Operations:**
    - Searches for Confluence articles with optional space and limit parameters.
    - Creates new articles in a specified Confluence space.
    - Retrieves article templates.
- **Resource Access:** Fetches JQL query templates from the Jira server and writing guidelines from the Confluence server.

## Methods:

- `__init__(self, redis_client: redis.Redis)`: Initializes the client with a Redis client instance.
- `connect_to_jira_server(self)`: Connects to the Jira MCP server.
- `disconnect(self)`: Disconnects from all connected MCP servers.
- `get_available_projects(self) -> List[str]`: Retrieves a list of available Jira project keys.
- `get_project_tickets(self, project_key: str, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]`: Fetches tickets for a given project.
- `search_jira(self, jql: str, max_results: int = 50) -> List[Dict[str, Any]]`: Performs a JQL search in Jira.
- `analyze_project_metrics(self, project_key: str, period_days: int = 30) -> Dict[str, Any]`: Analyzes and returns metrics for a Jira project.
- `connect_to_confluence_server(self)`: Connects to the Confluence MCP server.
- `search_confluence_articles(self, query: str, space: str = None, limit: int = 10) -> List[Dict[str, Any]]`: Searches for articles in Confluence.
- `create_confluence_article(self, title: str, content: str, space: str, tags: List[str] = None) -> Dict[str, Any]`: Creates a new article in Confluence.
- `get_article_template(self, template_type: str = None) -> Dict[str, Any]`: Retrieves Confluence article templates.
- `get_jira_jql_templates(self) -> Dict[str, str]`: Fetches JQL query templates.
- `get_confluence_writing_guidelines(self) -> Dict[str, Any]`: Retrieves Confluence writing guidelines.

## Usage:

This client is used by agents or other components that need to interact with Jira and Confluence. It abstracts the underlying MCP communication, providing a straightforward async API for data retrieval and manipulation.

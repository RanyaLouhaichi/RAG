# MCP Client

The `mcp_client.py` module defines two classes, `MCPClient` and `MCPAgentClient`, that facilitate communication between different agents in the system using the MCP (Multi-Agent Communication Protocol).

## MCPClient

The `MCPClient` class provides a low-level interface for managing connections to MCP servers and making calls to their tools, resources, and prompts.

### Key Responsibilities:

- **Connection Management:** Establishes, maintains, and closes connections to multiple MCP servers.
- **Tool Execution:** Calls tools on remote servers with specified arguments.
- **Resource Access:** Reads resources from remote servers.
- **Prompt Retrieval:** Fetches prompts from remote servers.
- **Capability Discovery:** Discovers the available tools, resources, and prompts of a server.
- **Logging:** Logs tool calls to Redis for analytics.

### Methods:

- `__init__(self, redis_client: redis.Redis)`: Initializes the client with a Redis client instance.
- `connect_to_server(self, server_name: str, server_command: List[str]) -> ClientSession`: Connects to a specified MCP server.
- `call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any`: Calls a tool on a connected server.
- `read_resource(self, server_name: str, resource_uri: str) -> Any`: Reads a resource from a server.
- `get_prompt(self, server_name: str, prompt_name: str, arguments: Dict[str, Any]) -> Any`: Retrieves a prompt from a server.
- `discover_capabilities(self, server_name: str) -> Dict[str, Any]`: Discovers the capabilities of a server.
- `_log_tool_call(...)`: Logs tool call details to Redis.
- `close_all(self)`: Closes all active connections.

## MCPAgentClient

The `MCPAgentClient` class provides a high-level, agent-oriented interface on top of `MCPClient`. It simplifies common agent operations by abstracting the details of server and tool names.

### Key Responsibilities:

- **Simplified Agent Interactions:** Offers methods for common agent tasks like generating responses, retrieving data, getting recommendations, and making predictions.
- **Abstraction:** Hides the complexity of direct `MCPClient` calls.

### Methods:

- `__init__(self, mcp_client: MCPClient)`: Initializes the client with an `MCPClient` instance.
- `generate_response(...)`: Generates a response using the Chat Agent.
- `retrieve_tickets(...)`: Retrieves tickets using the Jira Data Agent.
- `get_recommendations(...)`: Gets recommendations from the Recommendation Agent.
- `predict_sprint_completion(...)`: Predicts sprint completion using the Predictive Agent.
- `search_articles(...)`: Searches for articles using the Retrieval Agent.
- `generate_dashboard(...)`: Generates a dashboard using the Dashboard Agent.
- `generate_article(...)`: Generates an article using the Article Generator.

## Usage:

The `MCPClient` is used for general-purpose MCP communication, while the `MCPAgentClient` is tailored for streamlined agent-to-agent interactions. These clients are essential for the collaborative functioning of the multi-agent system.

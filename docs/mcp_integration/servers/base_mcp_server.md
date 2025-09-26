# Base MCP Server

The `BaseMCPServer` class provides a foundational structure for creating MCP (Multi-Agent Communication Protocol) servers that expose the capabilities of different agents. It handles the core MCP functionalities, allowing subclasses to focus on implementing agent-specific tools, resources, and prompts.

## Key Responsibilities:

- **Server Initialization:** Initializes an MCP server with a given agent name, agent instance, and Redis client.
- **Handler Setup:** Sets up handlers for standard MCP requests, including `list_tools`, `call_tool`, `list_resources`, `read_resource`, `list_prompts`, and `get_prompt`.
- **Tool and Resource Management:** Provides registries for tools and resources, along with methods to register them.
- **Usage Logging:** Logs tool usage to Redis for analytics and monitoring.
- **Prompt Handling:** Manages agent-specific prompts and fills them with provided arguments.

## Methods:

- `__init__(self, agent_name: str, agent_instance: Any, redis_client: redis.Redis)`: Initializes the base server.
- `_setup_handlers(self)`: Sets up the MCP protocol handlers.
- `_register_tools(self)`: A placeholder method to be overridden by subclasses for registering tools.
- `_register_resources(self)`: A placeholder method to be overridden by subclasses for registering resources.
- `_get_agent_prompts(self) -> List[Dict[str, Any]]`: A placeholder method for subclasses to provide agent-specific prompts.
- `_log_tool_usage(...)`: Logs tool usage details to Redis.
- `_fill_prompt_template(...)`: Fills a prompt template with given arguments.
- `register_tool(...)`: Registers a new tool with its description, schema, and handler.
- `register_resource(...)`: Registers a new resource with its URI, name, description, and content or handler.
- `start(self)`: Starts the MCP server and waits for requests.

## Subclassing:

To create a new agent server, you should inherit from `BaseMCPServer` and override the following methods:

- `_register_tools()`: To register the specific tools your agent provides.
- `_register_resources()`: To register any resources your agent exposes.
- `_get_agent_prompts()`: To provide a list of specialized prompts for your agent.

This base class significantly simplifies the process of creating new MCP servers, promoting consistency and reusability across the multi-agent system.

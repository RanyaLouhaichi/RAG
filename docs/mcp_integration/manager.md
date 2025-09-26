# MCP Manager

The `MCPManager` class is responsible for managing the lifecycle of MCP (Multi-Agent Communication Protocol) servers and client connections within the system. It orchestrates the creation, startup, and shutdown of various agent servers, and handles client connections to these servers.

## Key Responsibilities:

- **Server Management:** Creates and manages instances of different MCP servers, such as `ChatAgentMCPServer`, `JiraDataAgentMCPServer`, etc.
- **Client Connections:** Manages client connections to the MCP servers using `MCPClient` and `MCPAgentClient`.
- **Lifecycle Orchestration:** Provides methods to start and stop individual or all MCP servers.
- **Capability Discovery:** Discovers the capabilities of all connected servers.
- **System Status:** Provides a status overview of the MCP system, including server status, client connections, and Redis connectivity.

## Methods:

- `__init__(self, orchestrator)`: Initializes the `MCPManager` with a reference to the main orchestrator, a Redis client, and server configurations.
- `create_servers(self)`: Creates instances of all configured MCP servers.
- `start_server(self, server_name: str)`: Starts a specific MCP server by name.
- `start_all_servers(self)`: Starts all configured MCP servers concurrently.
- `connect_clients(self)`: Connects the MCP client to all registered servers.
- `discover_all_capabilities(self) -> Dict[str, Any]`: Gathers and returns the capabilities of all connected servers.
- `get_mcp_status(self) -> Dict[str, Any]`: Returns a dictionary containing the current status of the MCP system.
- `shutdown(self)`: Shuts down the entire MCP system by closing all client connections.

## Usage:

The `MCPManager` is typically instantiated by the main orchestrator. It is used to initialize and manage the communication infrastructure for the multi-agent system.

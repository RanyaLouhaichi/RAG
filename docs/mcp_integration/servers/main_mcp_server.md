# MCP Servers Main Entry Point

The `main.py` script in the `mcp_integration/servers` directory serves as the main entry point for launching the standalone MCP (Multi-Agent Communication Protocol) servers, such as the `JiraMCPServer` and `ConfluenceMCPServer`.

## Functionality

This script reads a command-line argument to determine which server to start. It then imports and runs the `main` function from the corresponding server module.

## Usage

To run a specific MCP server, you can use the following command:

```bash
python -m mcp_integration.servers [server_name]
```

Where `[server_name]` can be:

- `jira`: To start the `JiraMCPServer`.
- `confluence`: To start the `ConfluenceMCPServer`.

If no server name is provided or an unknown name is given, the script will print a usage message and exit.

## Example

To start the Jira MCP server, you would run:

```bash
python -m mcp_integration.servers jira
```

This script is essential for managing and running the different MCP servers that provide data and services to the multi-agent system.

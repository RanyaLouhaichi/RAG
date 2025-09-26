# MCP Server Launcher Documentation

## Overview

The `MCPServerLauncher` class is used to start and stop the MCP (Multi-Agent Communication Protocol) servers. It provides a convenient way to manage the lifecycle of the MCP servers, which are used for communication between the different agents in the system.

## `MCPServerLauncher` Class

### Methods

- **`start_server(name: str, module: str)`**: Starts an MCP server as a subprocess. The `name` parameter is a friendly name for the server, and the `module` parameter is the Python module to run.

- **`start_all_servers()`**: Starts all the MCP servers defined in the `servers_to_start` dictionary. By default, this includes the `jira` and `confluence` servers.

- **`stop_all_servers()`**: Stops all the running MCP servers.

## Usage

The `MCPServerLauncher` can be used to start and stop the MCP servers from the command line. For example:

```bash
python -m mcp_integration.launch_servers
```

This will start all the MCP servers and keep them running until the script is interrupted.

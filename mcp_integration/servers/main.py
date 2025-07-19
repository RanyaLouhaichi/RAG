"""Entry points for MCP servers"""
import sys
import asyncio
import logging

logging.basicConfig(level=logging.INFO)

if len(sys.argv) > 1:
    server_name = sys.argv[1]
    
    if server_name == "jira":
        from .jira_mcp_server import main
        asyncio.run(main())
    elif server_name == "confluence":
        from .confluence_mcp_server import main
        asyncio.run(main())
    else:
        print(f"Unknown server: {server_name}")
        sys.exit(1)
else:
    print("Usage: python -m mcp_integration.servers [jira|confluence]")
    sys.exit(1)
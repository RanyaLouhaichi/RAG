#!/usr/bin/env python
import asyncio
import sys
from mcp_integration.servers.confluence_mcp_server import main

if __name__ == "__main__":
    # Add the same structure to confluence_mcp_server.py
    asyncio.run(main())
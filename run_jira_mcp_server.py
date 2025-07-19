#!/usr/bin/env python
"""
Jira MCP Server - Exposes Jira API as MCP tools
"""

import json
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from integrations.jira_api_client import JiraAPIClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jira-mcp-server")

# Create server
server = Server("jira-mcp-server")

# Initialize Jira client
jira_client = JiraAPIClient()
if jira_client.test_connection():
    logger.info("✅ Connected to Jira")
else:
    logger.error("❌ Failed to connect to Jira")

@server.tool()
async def get_projects():
    """Get all available Jira projects"""
    projects = jira_client.get_projects()
    return json.dumps({
        "projects": [
            {"key": p.get("key"), "name": p.get("name"), "id": p.get("id")}
            for p in projects
        ],
        "count": len(projects)
    }, indent=2)

@server.tool()
async def get_project_tickets(project_key: str, start_date: str = None, end_date: str = None):
    """Get all tickets for a specific project"""
    tickets = jira_client.get_all_issues_for_project(project_key, start_date, end_date)
    return json.dumps({
        "project": project_key,
        "tickets": tickets,
        "count": len(tickets)
    }, indent=2, default=str)

@server.tool()
async def get_ticket(ticket_key: str):
    """Get details of a specific ticket"""
    ticket = jira_client.get_issue(ticket_key)
    return json.dumps(ticket, indent=2, default=str)

@server.tool()
async def search_tickets(jql: str, max_results: int = 50):
    """Search tickets using JQL"""
    result = jira_client.search_issues(jql, max_results=max_results)
    return json.dumps({
        "jql": jql,
        "tickets": result.get("issues", []),
        "total": result.get("total", 0)
    }, indent=2, default=str)

def main():
    """Run the server"""
    logger.info("Starting Jira MCP Server...")
    stdio_server(server)

if __name__ == "__main__":
    main()
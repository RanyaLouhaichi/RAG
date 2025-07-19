# mcp_integration/servers/jira_mcp_server.py
import asyncio
import json
import logging
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from integrations.jira_api_client import JiraAPIClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("JiraMCPServer")

# Initialize Jira client globally
jira_client = None

def init_jira_client():
    """Initialize Jira client"""
    global jira_client
    try:
        jira_client = JiraAPIClient()
        if jira_client.test_connection():
            logger.info("✅ Connected to Jira via MCP Server")
        else:
            logger.error("❌ Failed to connect to Jira")
            jira_client = None
    except Exception as e:
        logger.error(f"❌ Error initializing Jira client: {e}")
        jira_client = None

# Create server instance
server = Server("jira-mcp-server")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all available Jira tools"""
    return [
        types.Tool(
            name="get_projects",
            description="Get all available Jira projects",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="get_project_tickets",
            description="Get all tickets for a specific project",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_key": {
                        "type": "string",
                        "description": "Jira project key (e.g., PROJ, ABDERA)"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (optional)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional)"
                    }
                },
                "required": ["project_key"]
            }
        ),
        types.Tool(
            name="get_ticket",
            description="Get details of a specific ticket",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticket_key": {
                        "type": "string",
                        "description": "Jira ticket key (e.g., PROJ-123)"
                    }
                },
                "required": ["ticket_key"]
            }
        ),
        types.Tool(
            name="search_tickets",
            description="Search tickets using JQL (Jira Query Language)",
            inputSchema={
                "type": "object",
                "properties": {
                    "jql": {
                        "type": "string",
                        "description": "JQL query string"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 50)"
                    }
                },
                "required": ["jql"]
            }
        ),
        types.Tool(
            name="analyze_project_metrics",
            description="Analyze project metrics like velocity, cycle time, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_key": {
                        "type": "string",
                        "description": "Jira project key"
                    },
                    "period_days": {
                        "type": "integer",
                        "description": "Number of days to analyze (default: 30)"
                    }
                },
                "required": ["project_key"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Execute a Jira tool"""
    global jira_client
    
    # Ensure Jira client is initialized
    if not jira_client:
        init_jira_client()
        if not jira_client:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "Failed to initialize Jira client"})
            )]
    
    try:
        result = None
        
        if name == "get_projects":
            projects = jira_client.get_projects()
            result = {
                "projects": [
                    {
                        "key": p.get("key"),
                        "name": p.get("name"),
                        "id": p.get("id")
                    }
                    for p in projects
                ],
                "count": len(projects)
            }
        
        elif name == "get_project_tickets":
            project_key = arguments["project_key"]
            start_date = arguments.get("start_date")
            end_date = arguments.get("end_date")
            
            tickets = jira_client.get_all_issues_for_project(
                project_key, start_date, end_date
            )
            
            result = {
                "project": project_key,
                "tickets": tickets,
                "count": len(tickets),
                "date_range": {
                    "start": start_date or "all",
                    "end": end_date or "all"
                }
            }
        
        elif name == "get_ticket":
            ticket_key = arguments["ticket_key"]
            ticket = jira_client.get_issue(ticket_key)
            result = ticket
        
        elif name == "search_tickets":
            jql = arguments["jql"]
            max_results = arguments.get("max_results", 50)
            
            search_result = jira_client.search_issues(jql, max_results=max_results)
            
            result = {
                "jql": jql,
                "tickets": search_result.get("issues", []),
                "total": search_result.get("total", 0),
                "max_results": max_results
            }
        
        elif name == "analyze_project_metrics":
            project_key = arguments["project_key"]
            period_days = arguments.get("period_days", 30)
            
            # Get tickets
            tickets = jira_client.get_all_issues_for_project(project_key)
            
            # Calculate metrics
            metrics = calculate_project_metrics(tickets, period_days)
            
            result = {
                "project": project_key,
                "period_days": period_days,
                "metrics": metrics
            }
        
        else:
            raise ValueError(f"Unknown tool: {name}")
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]
        
    except Exception as e:
        logger.error(f"Tool execution failed: {name} - {str(e)}")
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]

@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """List available Jira resources"""
    return [
        types.Resource(
            uri="jira://config/status_mappings",
            name="Status Mappings",
            description="Jira status category mappings",
            mimeType="application/json"
        ),
        types.Resource(
            uri="jira://templates/jql_queries",
            name="JQL Query Templates",
            description="Common JQL query templates",
            mimeType="application/json"
        ),
        types.Resource(
            uri="jira://metadata/field_definitions",
            name="Field Definitions",
            description="Jira field definitions and custom fields",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a Jira resource"""
    if uri == "jira://config/status_mappings":
        return json.dumps({
            "done": ["Done", "Closed", "Resolved", "Complete", "Fixed"],
            "in_progress": ["In Progress", "In Development", "In Review", "Testing"],
            "todo": ["To Do", "Open", "New", "Backlog", "Ready"],
            "blocked": ["Blocked", "On Hold", "Waiting", "Impediment"]
        }, indent=2)
    
    elif uri == "jira://templates/jql_queries":
        return json.dumps({
            "sprint_tickets": "project = {project} AND sprint in openSprints()",
            "my_open_tickets": "assignee = currentUser() AND status not in (Done, Closed)",
            "recent_bugs": "project = {project} AND issuetype = Bug AND created >= -7d",
            "overdue_tickets": "project = {project} AND duedate < now() AND status not in (Done, Closed)",
            "high_priority": "project = {project} AND priority in (Highest, High) AND status not in (Done, Closed)"
        }, indent=2)
    
    elif uri == "jira://metadata/field_definitions":
        return json.dumps({
            "standard_fields": {
                "summary": "Issue summary",
                "description": "Issue description",
                "status": "Current status",
                "assignee": "Assigned user",
                "reporter": "Issue reporter",
                "priority": "Issue priority",
                "issuetype": "Type of issue"
            },
            "custom_fields": {
                "customfield_10010": "Story Points",
                "customfield_10011": "Epic Link",
                "customfield_10012": "Sprint"
            }
        }, indent=2)
    
    else:
        raise ValueError(f"Unknown resource: {uri}")

@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    """List available prompts"""
    return [
        types.Prompt(
            name="sprint_analysis",
            description="Analyze current sprint progress",
            arguments=[
                types.PromptArgument(
                    name="project",
                    description="Project key",
                    required=True
                )
            ]
        ),
        types.Prompt(
            name="team_workload",
            description="Analyze team workload distribution",
            arguments=[
                types.PromptArgument(
                    name="project",
                    description="Project key",
                    required=True
                )
            ]
        )
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
    """Get a specific prompt"""
    if name == "sprint_analysis":
        project = arguments.get("project", "PROJECT")
        return types.GetPromptResult(
            description=f"Sprint analysis for project {project}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Analyze the current sprint for project {project}. Include completion percentage, velocity, blockers, and recommendations."
                    )
                )
            ]
        )
    
    elif name == "team_workload":
        project = arguments.get("project", "PROJECT")
        return types.GetPromptResult(
            description=f"Team workload analysis for project {project}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Analyze team workload distribution for project {project}. Identify overloaded team members and suggest rebalancing."
                    )
                )
            ]
        )
    
    else:
        raise ValueError(f"Unknown prompt: {name}")

def calculate_project_metrics(tickets: List[Dict[str, Any]], period_days: int) -> Dict[str, Any]:
    """Calculate project metrics"""
    cutoff_date = datetime.now() - timedelta(days=period_days)
    
    metrics = {
        "total_tickets": len(tickets),
        "status_distribution": {},
        "velocity": 0,
        "average_cycle_time": 0,
        "assignee_distribution": {},
        "issue_type_distribution": {}
    }
    
    cycle_times = []
    completed_in_period = 0
    
    for ticket in tickets:
        fields = ticket.get("fields", {})
        
        # Status distribution
        status = fields.get("status", {}).get("name", "Unknown")
        metrics["status_distribution"][status] = metrics["status_distribution"].get(status, 0) + 1
        
        # Issue type distribution
        issue_type = fields.get("issuetype", {}).get("name", "Unknown")
        metrics["issue_type_distribution"][issue_type] = metrics["issue_type_distribution"].get(issue_type, 0) + 1
        
        # Assignee distribution
        assignee = fields.get("assignee", {}).get("displayName", "Unassigned") if fields.get("assignee") else "Unassigned"
        metrics["assignee_distribution"][assignee] = metrics["assignee_distribution"].get(assignee, 0) + 1
        
        # Check if completed in period
        resolution_date = fields.get("resolutiondate")
        if resolution_date:
            res_date = datetime.fromisoformat(resolution_date.replace("Z", "+00:00"))
            if res_date.replace(tzinfo=None) > cutoff_date:
                completed_in_period += 1
            
            # Calculate cycle time
            created_date = datetime.fromisoformat(fields.get("created").replace("Z", "+00:00"))
            cycle_time = (res_date - created_date).days
            if cycle_time >= 0:
                cycle_times.append(cycle_time)
    
    # Calculate velocity (tickets completed per week)
    weeks_in_period = period_days / 7
    metrics["velocity"] = round(completed_in_period / weeks_in_period, 1) if weeks_in_period > 0 else 0
    
    # Calculate average cycle time
    if cycle_times:
        metrics["average_cycle_time"] = round(sum(cycle_times) / len(cycle_times), 1)
    
    return metrics

async def main():
    """Main entry point for the MCP server"""
    # Initialize Jira client
    init_jira_client()
    
    logger.info("Starting Jira MCP Server v1.9.4...")
    
    # Run the server using stdio
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)
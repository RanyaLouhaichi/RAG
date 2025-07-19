from datetime import datetime
import json
from mcp_integration.servers.base_mcp_server import BaseMCPServer
from typing import Dict, Any, List
import asyncio

class JiraDataAgentMCPServer(BaseMCPServer):
    """MCP Server for Jira Data Agent"""
    
    def _register_tools(self):
        """Register Jira data agent tools"""
        
        # Tool: Retrieve Tickets
        async def retrieve_tickets(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Retrieve Jira tickets"""
            result = self.agent.run({
                "project_id": arguments.get("project_id", "PROJ123"),
                "time_range": arguments.get("time_range", {}),
                "analysis_depth": arguments.get("analysis_depth", "basic")
            })
            
            return {
                "tickets": result.get("tickets", []),
                "metadata": result.get("metadata", {}),
                "ticket_count": len(result.get("tickets", [])),
                "workflow_status": result.get("workflow_status", "success")
            }
        
        self.register_tool(
            name="retrieve_tickets",
            description="Retrieve and analyze Jira tickets for a project",
            input_schema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "time_range": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string"},
                            "end": {"type": "string"}
                        }
                    },
                    "analysis_depth": {"type": "string", "enum": ["basic", "enhanced"]}
                }
            },
            handler=retrieve_tickets
        )
        
        # Tool: Analyze Project Data
        async def analyze_project(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Analyze project data patterns"""
            project_id = arguments.get("project_id")
            
            # First retrieve tickets
            tickets_result = await retrieve_tickets({"project_id": project_id})
            tickets = tickets_result.get("tickets", [])
            
            # Analyze patterns
            status_distribution = {}
            assignee_workload = {}
            
            for ticket in tickets:
                fields = ticket.get("fields", {})
                
                # Status analysis
                status = fields.get("status", {}).get("name", "Unknown")
                status_distribution[status] = status_distribution.get(status, 0) + 1
                
                # Assignee analysis
                assignee_info = fields.get("assignee")
                if assignee_info:
                    assignee = assignee_info.get("displayName", "Unassigned")
                    assignee_workload[assignee] = assignee_workload.get(assignee, 0) + 1
            
            return {
                "project_id": project_id,
                "total_tickets": len(tickets),
                "status_distribution": status_distribution,
                "assignee_workload": assignee_workload,
                "has_bottlenecks": any(count > len(tickets) * 0.3 for count in status_distribution.values())
            }
        
        self.register_tool(
            name="analyze_project",
            description="Analyze project data patterns and identify issues",
            input_schema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"}
                },
                "required": ["project_id"]
            },
            handler=analyze_project
        )
        
        # Tool: Monitor Changes
        async def monitor_changes(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Monitor Jira data changes"""
            project_id = arguments.get("project_id")
            
            # Check for updates
            has_updates = self.redis_client.exists(f"updates:{project_id}")
            
            if has_updates:
                # Get latest data
                tickets_result = await retrieve_tickets({"project_id": project_id})
                
                return {
                    "has_changes": True,
                    "project_id": project_id,
                    "ticket_count": tickets_result.get("ticket_count", 0),
                    "last_check": datetime.now().isoformat()
                }
            
            return {
                "has_changes": False,
                "project_id": project_id,
                "last_check": datetime.now().isoformat()
            }
        
        self.register_tool(
            name="monitor_changes",
            description="Monitor Jira data for changes",
            input_schema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"}
                },
                "required": ["project_id"]
            },
            handler=monitor_changes
        )
    
    def _register_resources(self):
        """Register Jira data resources"""
        
        # Resource: Available Projects
        async def get_projects():
            """Get available projects"""
            if hasattr(self.agent, 'available_projects'):
                return {
                    "projects": self.agent.available_projects,
                    "using_api": self.agent.use_real_api
                }
            return {"projects": ["PROJ123"], "using_api": False}
        
        self.register_resource(
            uri="jira://projects/available",
            name="Available Projects",
            description="List of available Jira projects",
            handler=get_projects
        )
        
        # Resource: Cached Data Summary
        async def get_cache_summary():
            """Get summary of cached data"""
            cache_keys = self.redis_client.keys("jira_raw_data:*")
            summaries = []
            
            for key in cache_keys[:10]:  # Limit to 10
                metadata_key = f"{key}:metadata"
                metadata = self.redis_client.get(metadata_key)
                if metadata:
                    summaries.append(json.loads(metadata))
            
            return {
                "total_cached_datasets": len(cache_keys),
                "recent_caches": summaries
            }
        
        self.register_resource(
            uri="jira://cache/summary",
            name="Cache Summary",
            description="Summary of cached Jira data",
            handler=get_cache_summary
        )
        
        # Resource: Performance Metrics
        async def get_performance():
            """Get performance metrics"""
            return self.agent.get_performance_metrics()
        
        self.register_resource(
            uri="jira://metrics/performance",
            name="Performance Metrics",
            description="Agent performance metrics",
            handler=get_performance
        )
    
    def _get_agent_prompts(self) -> List[Dict[str, Any]]:
        """Get Jira data agent prompts"""
        return [
            {
                "name": "project_summary",
                "description": "Generate a project summary",
                "template": "Provide a comprehensive summary of project {project_id} including ticket distribution, team workload, and potential bottlenecks.",
                "arguments": [
                    {"name": "project_id", "description": "Jira project ID", "required": True}
                ]
            },
            {
                "name": "sprint_data_analysis",
                "description": "Analyze sprint data",
                "template": "Analyze the sprint data for {project_id} between {start_date} and {end_date}. Focus on {metrics}.",
                "arguments": [
                    {"name": "project_id", "description": "Jira project ID", "required": True},
                    {"name": "start_date", "description": "Start date", "required": True},
                    {"name": "end_date", "description": "End date", "required": True},
                    {"name": "metrics", "description": "Metrics to focus on", "required": False}
                ]
            }
        ]
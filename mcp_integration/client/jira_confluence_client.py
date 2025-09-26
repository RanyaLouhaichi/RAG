import asyncio
import json
import logging
import subprocess
import sys
from typing import Dict, Any, List, Optional, Tuple
import redis
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class JiraConfluenceMCPClient:
    """MCP Client for accessing Jira and Confluence data"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger("JiraConfluenceMCPClient")
        self.jira_session: Optional[ClientSession] = None
        self.confluence_session: Optional[ClientSession] = None
        self._jira_context = None
        self._confluence_context = None
        self._jira_process = None  # Store subprocess reference
    
    async def connect_to_jira_server(self):
        """Connect to Jira MCP server"""
        try:
            server_params = StdioServerParameters(
                command=sys.executable,  # Use the current Python interpreter
                args=["-m", "mcp_integration.servers.jira_mcp_server"],
                env=None
            )
            
            self.logger.info(f"Starting Jira MCP server with command: {server_params.command} {' '.join(server_params.args)}")
            
            self._jira_context = stdio_client(server_params)
            read_stream, write_stream = await self._jira_context.__aenter__()
            
            self.jira_session = ClientSession(read_stream, write_stream)
            await self.jira_session.__aenter__()
            
            result = await self.jira_session.initialize()
            
            self.logger.info(f"Connected to Jira MCP Server: {result}")
            
            tools_response = await self.jira_session.list_tools()
            
            tools = []
            if hasattr(tools_response, 'tools'):
                tools = tools_response.tools
            elif isinstance(tools_response, list):
                tools = tools_response
            
            tool_names = []
            for tool in tools:
                if hasattr(tool, 'name'):
                    tool_names.append(tool.name)
                elif isinstance(tool, dict):
                    tool_names.append(tool.get('name', 'unknown'))
                elif isinstance(tool, tuple) and len(tool) > 0:
                    tool_names.append(str(tool[0]))
                else:
                    tool_names.append('unknown')
            
            self.logger.info(f"Available Jira tools: {tool_names}")
            
            return True
                    
        except Exception as e:
            self.logger.error(f"Failed to connect to Jira MCP: {e}", exc_info=True)
            raise
    
    async def disconnect(self):
        """Disconnect from MCP servers"""
        try:
            if self.jira_session:
                await self.jira_session.__aexit__(None, None, None)
                self.jira_session = None
            
            if self.confluence_session:
                await self.confluence_session.__aexit__(None, None, None)
                self.confluence_session = None
            
            if self._jira_context:
                await self._jira_context.__aexit__(None, None, None)
                self._jira_context = None
                
            if self._confluence_context:
                await self._confluence_context.__aexit__(None, None, None)
                self._confluence_context = None
                
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
    
    async def get_available_projects(self) -> List[str]:
        """Get available Jira projects"""
        if not self.jira_session:
            raise RuntimeError("Not connected to Jira MCP server")
        
        try:
            result = await self.jira_session.call_tool("get_projects", {})
            
            if hasattr(result, 'content') and result.content:
                projects_data = json.loads(result.content[0].text)
            elif isinstance(result, dict):
                projects_data = result
            elif isinstance(result, str):
                projects_data = json.loads(result)
            else:
                projects_data = json.loads(str(result))
            
            projects = projects_data.get("projects", [])
            return [p["key"] for p in projects]
            
        except Exception as e:
            self.logger.error(f"Failed to get projects: {e}")
            return []
    
    async def get_project_tickets(self, project_key: str, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """Get tickets for a project"""
        if not self.jira_session:
            raise RuntimeError("Not connected to Jira MCP server")
        
        try:
            args = {"project_key": project_key}
            if start_date:
                args["start_date"] = start_date
            if end_date:
                args["end_date"] = end_date
            
            result = await self.jira_session.call_tool("get_project_tickets", args)
            
            if hasattr(result, 'content') and result.content:
                data = json.loads(result.content[0].text)
            elif isinstance(result, dict):
                data = result
            elif isinstance(result, str):
                data = json.loads(result)
            else:
                data = json.loads(str(result))
            
            return data.get("tickets", [])
            
        except Exception as e:
            self.logger.error(f"Failed to get project tickets: {e}")
            return []
    
    async def search_jira(self, jql: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search Jira using JQL"""
        if not self.jira_session:
            raise RuntimeError("Not connected to Jira MCP server")
        
        try:
            result = await self.jira_session.call_tool("search_tickets", {
                "jql": jql,
                "max_results": max_results
            })
            
            if hasattr(result, 'content') and result.content:
                data = json.loads(result.content[0].text)
            elif isinstance(result, dict):
                data = result
            elif isinstance(result, str):
                data = json.loads(result)
            else:
                data = json.loads(str(result))
                
            return data.get("tickets", [])
            
        except Exception as e:
            self.logger.error(f"Failed to search Jira: {e}")
            return []
    
    async def analyze_project_metrics(self, project_key: str, period_days: int = 30) -> Dict[str, Any]:
        """Get project metrics analysis"""
        if not self.jira_session:
            raise RuntimeError("Not connected to Jira MCP server")
        
        try:
            result = await self.jira_session.call_tool("analyze_project_metrics", {
                "project_key": project_key,
                "period_days": period_days
            })
            
            if hasattr(result, 'content') and result.content:
                data = json.loads(result.content[0].text)
            elif isinstance(result, dict):
                data = result
            elif isinstance(result, str):
                data = json.loads(result)
            else:
                data = json.loads(str(result))
                
            return data.get("metrics", {})
            
        except Exception as e:
            self.logger.error(f"Failed to analyze project metrics: {e}")
            return {}
    
    async def connect_to_confluence_server(self):
        """Connect to Confluence MCP server"""
        try:
            server_params = StdioServerParameters(
                command="python",
                args=["-m", "mcp_integration.servers.confluence_mcp_server"],
                env=None
            )
            
            self._confluence_context = stdio_client(server_params)
            read_stream, write_stream = await self._confluence_context.__aenter__()
            
            self.confluence_session = ClientSession(read_stream, write_stream)
            await self.confluence_session.__aenter__()
            
            await self.confluence_session.initialize()
            
            self.logger.info("Connected to Confluence MCP Server")
                    
        except Exception as e:
            self.logger.error(f"Failed to connect to Confluence MCP: {e}")
            raise

    async def search_confluence_articles(self, query: str, space: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search Confluence articles"""
        if not self.confluence_session:
            raise RuntimeError("Not connected to Confluence MCP server")
        
        args = {"query": query, "limit": limit}
        if space:
            args["space"] = space
        
        result = await self.confluence_session.call_tool("search_articles", args)
        data = json.loads(result.content[0].text)
        return data.get("results", [])
    
    async def create_confluence_article(self, title: str, content: str, space: str, tags: List[str] = None) -> Dict[str, Any]:
        """Create a new Confluence article"""
        if not self.confluence_session:
            raise RuntimeError("Not connected to Confluence MCP server")
        
        args = {
            "title": title,
            "content": content,
            "space": space
        }
        if tags:
            args["tags"] = tags
        
        result = await self.confluence_session.call_tool("create_article", args)
        return json.loads(result.content[0].text)
    
    async def get_article_template(self, template_type: str = None) -> Dict[str, Any]:
        """Get Confluence article templates"""
        if not self.confluence_session:
            raise RuntimeError("Not connected to Confluence MCP server")
        
        args = {}
        if template_type:
            args["template_type"] = template_type
        
        result = await self.confluence_session.call_tool("get_article_templates", args)
        return json.loads(result.content[0].text)
    
    async def get_jira_jql_templates(self) -> Dict[str, str]:
        """Get JQL query templates"""
        if not self.jira_session:
            raise RuntimeError("Not connected to Jira MCP server")
        
        result = await self.jira_session.read_resource("jira://templates/jql_queries")
        return json.loads(result.contents[0].text)
    
    async def get_confluence_writing_guidelines(self) -> Dict[str, Any]:
        """Get Confluence writing guidelines"""
        if not self.confluence_session:
            raise RuntimeError("Not connected to Confluence MCP server")
        
        result = await self.confluence_session.read_resource("confluence://guidelines/writing")
        return json.loads(result.contents[0].text)
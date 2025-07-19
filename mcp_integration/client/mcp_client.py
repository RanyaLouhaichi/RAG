import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import redis
from datetime import datetime

class MCPClient:
    """MCP Client for inter-agent communication"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger("MCPClient")
        self.sessions: Dict[str, ClientSession] = {}
        self._contexts: Dict[str, Any] = {}  # Store context managers
        
    async def connect_to_server(self, server_name: str, server_command: List[str]) -> ClientSession:
        """Connect to an MCP server"""
        try:
            if server_name in self.sessions:
                return self.sessions[server_name]
            
            # Create server parameters
            server_params = StdioServerParameters(
                command=server_command[0],
                args=server_command[1:] if len(server_command) > 1 else [],
                env=None
            )
            
            # Create client context
            context = stdio_client(server_params)
            read_stream, write_stream = await context.__aenter__()
            
            # Store context for cleanup
            self._contexts[server_name] = context
            
            # Create session
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            
            # Initialize session
            await session.initialize()
            
            # Store session
            self.sessions[server_name] = session
            self.logger.info(f"Connected to MCP server: {server_name}")
            
            return session
                    
        except Exception as e:
            self.logger.error(f"Failed to connect to {server_name}: {e}")
            raise
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on a specific server"""
        try:
            session = self.sessions.get(server_name)
            if not session:
                raise ValueError(f"Not connected to server: {server_name}")
            
            # Call tool
            result = await session.call_tool(tool_name, arguments)
            
            # Log tool call
            self._log_tool_call(server_name, tool_name, arguments, result)
            
            # Parse result
            if hasattr(result, 'content') and result.content:
                return json.loads(result.content[0].text)
            return result
            
        except Exception as e:
            self.logger.error(f"Tool call failed: {server_name}.{tool_name} - {e}")
            raise
    
    async def read_resource(self, server_name: str, resource_uri: str) -> Any:
        """Read a resource from a server"""
        try:
            session = self.sessions.get(server_name)
            if not session:
                raise ValueError(f"Not connected to server: {server_name}")
            
            # Read resource
            result = await session.read_resource(resource_uri)
            
            # Parse result
            if hasattr(result, 'contents') and result.contents:
                return json.loads(result.contents[0].text)
            return result
            
        except Exception as e:
            self.logger.error(f"Resource read failed: {server_name}.{resource_uri} - {e}")
            raise
    
    async def get_prompt(self, server_name: str, prompt_name: str, arguments: Dict[str, Any]) -> Any:
        """Get a prompt from a server"""
        try:
            session = self.sessions.get(server_name)
            if not session:
                raise ValueError(f"Not connected to server: {server_name}")
            
            # Get prompt
            result = await session.get_prompt(prompt_name, arguments)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prompt retrieval failed: {server_name}.{prompt_name} - {e}")
            raise
    
    async def discover_capabilities(self, server_name: str) -> Dict[str, Any]:
        """Discover capabilities of a server"""
        try:
            session = self.sessions.get(server_name)
            if not session:
                raise ValueError(f"Not connected to server: {server_name}")
            
            # List tools
            tools = await session.list_tools()
            
            # List resources  
            resources = await session.list_resources()
            
            # List prompts
            prompts = await session.list_prompts()
            
            return {
                "server": server_name,
                "tools": [{"name": t.name, "description": t.description} for t in tools],
                "resources": [{"uri": r.uri, "name": r.name} for r in resources],
                "prompts": [{"name": p.name, "description": p.description} for p in prompts]
            }
            
        except Exception as e:
            self.logger.error(f"Capability discovery failed: {server_name} - {e}")
            raise
    
    def _log_tool_call(self, server_name: str, tool_name: str, arguments: Dict[str, Any], result: Any):
        """Log tool call for analytics"""
        log_entry = {
            "server": server_name,
            "tool": tool_name,
            "arguments": arguments,
            "result_size": len(str(result)),
            "timestamp": datetime.now().isoformat()
        }
        
        log_key = f"mcp_client_calls:{datetime.now().strftime('%Y%m%d')}"
        self.redis_client.lpush(log_key, json.dumps(log_entry))
        self.redis_client.expire(log_key, 86400 * 7)
    
    async def close_all(self):
        """Close all connections"""
        for server_name in list(self.sessions.keys()):
            try:
                # Close session
                session = self.sessions[server_name]
                await session.__aexit__(None, None, None)
                
                # Close context
                if server_name in self._contexts:
                    context = self._contexts[server_name]
                    await context.__aexit__(None, None, None)
                    del self._contexts[server_name]
                
                del self.sessions[server_name]
                self.logger.info(f"Closed connection to {server_name}")
            except Exception as e:
                self.logger.error(f"Error closing {server_name}: {e}")

class MCPAgentClient:
    """High-level MCP client for agent operations"""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.logger = logging.getLogger("MCPAgentClient")
    
    async def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate response using Chat Agent"""
        result = await self.mcp_client.call_tool(
            "chat_agent",
            "generate_response",
            {
                "prompt": prompt,
                "session_id": context.get("session_id") if context else None,
                "articles": context.get("articles", []) if context else [],
                "recommendations": context.get("recommendations", []) if context else [],
                "predictions": context.get("predictions", {}) if context else {}
            }
        )
        return result.get("response", "")
    
    async def retrieve_tickets(self, project_id: str, time_range: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """Retrieve tickets using Jira Data Agent"""
        result = await self.mcp_client.call_tool(
            "jira_data_agent",
            "retrieve_tickets",
            {
                "project_id": project_id,
                "time_range": time_range or {},
                "analysis_depth": "enhanced"
            }
        )
        return result.get("tickets", [])
    
    async def get_recommendations(self, prompt: str, project: str, tickets: List[Dict[str, Any]] = None) -> List[str]:
        """Get recommendations using Recommendation Agent"""
        result = await self.mcp_client.call_tool(
            "recommendation_agent",
            "generate_recommendations",
            {
                "prompt": prompt,
                "project": project,
                "tickets": tickets or []
            }
        )
        return result.get("recommendations", [])
    
    async def predict_sprint_completion(self, tickets: List[Dict[str, Any]], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Predict sprint completion using Predictive Agent"""
        result = await self.mcp_client.call_tool(
            "predictive_agent",
            "predict_sprint_completion",
            {
                "tickets": tickets,
                "metrics": metrics,
                "historical_data": {}
            }
        )
        return result
    
    async def search_articles(self, query: str) -> List[Dict[str, Any]]:
        """Search articles using Retrieval Agent"""
        result = await self.mcp_client.call_tool(
            "retrieval_agent",
            "search_articles",
            {"query": query}
        )
        return result.get("articles", [])
    
    async def generate_dashboard(self, project_id: str, tickets: List[Dict[str, Any]], 
                               predictions: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate dashboard using Dashboard Agent"""
        result = await self.mcp_client.call_tool(
            "dashboard_agent",
            "generate_dashboard",
            {
                "project_id": project_id,
                "tickets": tickets,
                "predictions": predictions or {}
            }
        )
        return result
    
    async def generate_article(self, ticket_id: str) -> Dict[str, Any]:
        """Generate article using Article Generator"""
        result = await self.mcp_client.call_tool(
            "article_generator",
            "generate_article",
            {"ticket_id": ticket_id}
        )
        return result.get("article", {})
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    Tool, TextContent, ImageContent, EmbeddedResource,
    ListToolsResult, CallToolResult, 
    ListResourcesResult, ReadResourceResult,
    ListPromptsResult, GetPromptResult,
    PromptMessage, PromptArgument
)
import redis
from datetime import datetime

class BaseMCPServer:
    """Base class for MCP servers that expose agent capabilities"""
    
    def __init__(self, agent_name: str, agent_instance: Any, redis_client: redis.Redis):
        self.agent_name = agent_name
        self.agent = agent_instance
        self.redis_client = redis_client
        self.server = Server(f"{agent_name}_mcp_server")
        self.logger = logging.getLogger(f"MCP_{agent_name}")
        
        # Tool registry
        self.tools: Dict[str, Tool] = {}
        self.tool_handlers: Dict[str, Callable] = {}
        
        # Resource registry
        self.resources: Dict[str, Any] = {}
        
        # Setup handlers
        self._setup_handlers()
        self._register_tools()
        self._register_resources()
        
    def _setup_handlers(self):
        """Setup MCP protocol handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List all available tools from this agent"""
            return list(self.tools.values())
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
            """Execute a tool and return results"""
            if name not in self.tool_handlers:
                raise ValueError(f"Unknown tool: {name}")
            
            try:
                # Log tool usage
                self._log_tool_usage(name, arguments)
                
                # Execute tool
                result = await self.tool_handlers[name](arguments)
                
                # Return as MCP content
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
                
            except Exception as e:
                self.logger.error(f"Tool execution failed: {name} - {str(e)}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.list_resources()
        async def handle_list_resources() -> list[Any]:
            """List available resources"""
            resources = []
            for uri, resource in self.resources.items():
                resources.append({
                    "uri": uri,
                    "name": resource.get("name", uri),
                    "description": resource.get("description", ""),
                    "mimeType": resource.get("mimeType", "application/json")
                })
            return resources
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read a specific resource"""
            if uri not in self.resources:
                raise ValueError(f"Unknown resource: {uri}")
            
            resource = self.resources[uri]
            if "handler" in resource:
                content = await resource["handler"]()
            else:
                content = resource.get("content", {})
            
            return json.dumps(content, indent=2)
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> list[Any]:
            """List available prompts"""
            # Each agent can provide specialized prompts
            return self._get_agent_prompts()
        
        @self.server.get_prompt()
        async def handle_get_prompt(name: str, arguments: dict) -> Any:
            """Get a specific prompt with arguments filled"""
            prompts = {p["name"]: p for p in self._get_agent_prompts()}
            if name not in prompts:
                raise ValueError(f"Unknown prompt: {name}")
            
            prompt = prompts[name]
            # Fill in the prompt template with arguments
            filled_prompt = self._fill_prompt_template(prompt, arguments)
            
            return {
                "messages": [
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=filled_prompt)
                    )
                ]
            }
    
    def _register_tools(self):
        """Register agent-specific tools - override in subclasses"""
        pass
    
    def _register_resources(self):
        """Register agent-specific resources - override in subclasses"""
        pass
    
    def _get_agent_prompts(self) -> List[Dict[str, Any]]:
        """Get agent-specific prompts - override in subclasses"""
        return []
    
    def _log_tool_usage(self, tool_name: str, arguments: Dict[str, Any]):
        """Log tool usage for analytics"""
        usage_log = {
            "agent": self.agent_name,
            "tool": tool_name,
            "arguments": arguments,
            "timestamp": datetime.now().isoformat()
        }
        
        log_key = f"mcp_tool_usage:{self.agent_name}:{datetime.now().strftime('%Y%m%d')}"
        self.redis_client.lpush(log_key, json.dumps(usage_log))
        self.redis_client.expire(log_key, 86400 * 7)  # Keep for 7 days
    
    def _fill_prompt_template(self, prompt: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        """Fill prompt template with arguments"""
        template = prompt.get("template", "")
        for key, value in arguments.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template
    
    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], handler: Callable):
        """Register a new tool"""
        self.tools[name] = Tool(
            name=name,
            description=description,
            inputSchema=input_schema
        )
        self.tool_handlers[name] = handler
    
    def register_resource(self, uri: str, name: str, description: str, 
                         handler: Optional[Callable] = None, content: Optional[Any] = None):
        """Register a new resource"""
        self.resources[uri] = {
            "name": name,
            "description": description,
            "handler": handler,
            "content": content,
            "mimeType": "application/json"
        }
    
    async def start(self):
        """Start the MCP server"""
        self.logger.info(f"Starting MCP server for {self.agent_name}")
        # Initialize with stdio transport
        async with self.server.run_stdio():
            await asyncio.Event().wait()
import asyncio
import os
import sys
from typing import Dict, Any, List, Optional
import logging
import redis
from mcp_integration.servers.chat_agent_server import ChatAgentMCPServer
from mcp_integration.servers.jira_data_agent_server import JiraDataAgentMCPServer
from mcp_integration.servers.recommendation_agent_server import RecommendationAgentMCPServer
from mcp_integration.servers.predictive_agent_server import PredictiveAgentMCPServer
from mcp_integration.servers.dashboard_agent_server import DashboardAgentMCPServer
from mcp_integration.servers.retrieval_agent_server import RetrievalAgentMCPServer
from mcp_integration.servers.article_generator_server import ArticleGeneratorMCPServer
from mcp_integration.client.mcp_client import MCPClient, MCPAgentClient

class MCPManager:
    """Manages MCP servers and client connections"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.redis_client = orchestrator.shared_memory.redis_client
        self.logger = logging.getLogger("MCPManager")
        self.servers = {}
        self.mcp_client = MCPClient(self.redis_client)
        self.agent_client = MCPAgentClient(self.mcp_client)
        self.server_configs = {
            "chat_agent": {
                "class": ChatAgentMCPServer,
                "agent": orchestrator.chat_agent,
                "port": 8001
            },
            "jira_data_agent": {
                "class": JiraDataAgentMCPServer,
                "agent": orchestrator.jira_data_agent,
                "port": 8002
            },
            "recommendation_agent": {
                "class": RecommendationAgentMCPServer,
                "agent": orchestrator.recommendation_agent,
                "port": 8003
            },
            "predictive_agent": {
                "class": PredictiveAgentMCPServer,
                "agent": orchestrator.predictive_analysis_agent,
                "port": 8004
            },
            "dashboard_agent": {
                "class": DashboardAgentMCPServer,
                "agent": orchestrator.productivity_dashboard_agent,
                "port": 8005
            },
            "retrieval_agent": {
                "class": RetrievalAgentMCPServer,
                "agent": orchestrator.retrieval_agent,
                "port": 8006
            },
            "article_generator": {
                "class": ArticleGeneratorMCPServer,
                "agent": orchestrator.jira_article_generator,
                "port": 8007
            }
        }
    
    def create_servers(self):
        """Create MCP server instances"""
        for name, config in self.server_configs.items():
            try:
                server = config["class"](
                    agent_name=name,
                    agent_instance=config["agent"],
                    redis_client=self.redis_client
                )
                self.servers[name] = server
                self.logger.info(f"Created MCP server: {name}")
            except Exception as e:
                self.logger.error(f"Failed to create server {name}: {e}")
    
    async def start_server(self, server_name: str):
        """Start a specific MCP server"""
        if server_name not in self.servers:
            self.logger.error(f"Unknown server: {server_name}")
            return
        
        server = self.servers[server_name]
        self.logger.info(f"Starting MCP server: {server_name}")
        
        try:
            await server.start()
        except Exception as e:
            self.logger.error(f"Failed to start server {server_name}: {e}")
    
    async def start_all_servers(self):
        """Start all MCP servers"""
        tasks = []
        for server_name in self.servers:
            task = asyncio.create_task(self.start_server(server_name))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def connect_clients(self):
        """Connect MCP client to all servers"""
        for server_name, config in self.server_configs.items():
            try:
                server_command = [sys.executable, "-m", f"mcp_servers.{server_name}"]
                
                await self.mcp_client.connect_to_server(server_name, server_command)
                self.logger.info(f"Connected to server: {server_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to connect to {server_name}: {e}")
    
    async def discover_all_capabilities(self) -> Dict[str, Any]:
        """Discover capabilities of all connected servers"""
        capabilities = {}
        
        for server_name in self.mcp_client.sessions:
            try:
                caps = await self.mcp_client.discover_capabilities(server_name)
                capabilities[server_name] = caps
            except Exception as e:
                self.logger.error(f"Failed to discover capabilities for {server_name}: {e}")
        
        return capabilities
    
    def get_mcp_status(self) -> Dict[str, Any]:
        """Get MCP system status"""
        return {
            "servers": {
                name: {
                    "created": name in self.servers,
                    "connected": name in self.mcp_client.sessions
                }
                for name in self.server_configs
            },
            "total_servers": len(self.servers),
            "connected_clients": len(self.mcp_client.sessions),
            "redis_connected": self.redis_client.ping()
        }
    
    async def shutdown(self):
        """Shutdown MCP system"""
        await self.mcp_client.close_all()
        self.logger.info("MCP system shutdown complete")
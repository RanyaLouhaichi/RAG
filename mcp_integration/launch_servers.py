# mcp_integration/launch_servers.py
import asyncio
import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCPServerLauncher")

class MCPServerLauncher:
    def __init__(self):
        self.servers = {}
        
    def start_server(self, name: str, module: str):
        """Start an MCP server as a subprocess"""
        try:
            cmd = [sys.executable, "-m", module]
            logger.info(f"Starting {name} with command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.servers[name] = process
            logger.info(f"Started {name} server (PID: {process.pid})")
            
            # Give it time to start
            time.sleep(2)
            
            # Check if it's still running
            if process.poll() is not None:
                stderr = process.stderr.read()
                raise Exception(f"Server {name} failed to start: {stderr}")
                
            return process
            
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            raise
    
    def start_all_servers(self):
        """Start all MCP servers"""
        servers_to_start = {
            "jira": "mcp_integration.servers.jira_mcp_server",
            "confluence": "mcp_integration.servers.confluence_mcp_server"
        }
        
        for name, module in servers_to_start.items():
            self.start_server(name, module)
            
    def stop_all_servers(self):
        """Stop all running servers"""
        for name, process in self.servers.items():
            if process.poll() is None:
                logger.info(f"Stopping {name} server")
                process.terminate()
                process.wait(timeout=5)

if __name__ == "__main__":
    launcher = MCPServerLauncher()
    try:
        launcher.start_all_servers()
        logger.info("All MCP servers started. Press Ctrl+C to stop.")
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down servers...")
        launcher.stop_all_servers()
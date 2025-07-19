# test_mcp_connection.py
import asyncio
import json
import logging
import sys
from mcp_integration.client.jira_confluence_client import JiraConfluenceMCPClient
import redis

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("MCP_TEST")

async def test_mcp_connection():
    """Test MCP connection and data retrieval"""
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    print("=" * 60)
    print("Testing MCP Connection")
    print("=" * 60)
    
    try:
        # Create MCP client
        mcp_client = JiraConfluenceMCPClient(redis_client)
        print("✅ Created MCP client")
        
        # Connect to Jira server
        print("\n1. Connecting to Jira MCP server...")
        await mcp_client.connect_to_jira_server()
        print("✅ Connected to Jira MCP server")
        
        # Get available projects
        print("\n2. Getting available projects...")
        projects = await mcp_client.get_available_projects()
        print(f"✅ Found projects: {projects}")
        
        # Test getting tickets for each project
        for project in projects[:1]:  # Test first project only
            print(f"\n3. Getting tickets for project {project}...")
            tickets = await mcp_client.get_project_tickets(project)
            print(f"✅ Found {len(tickets)} tickets in {project}")
            
            # Show first ticket as sample
            if tickets:
                first_ticket = tickets[0]
                print(f"\nSample ticket:")
                print(f"  Key: {first_ticket.get('key')}")
                print(f"  Summary: {first_ticket.get('fields', {}).get('summary')}")
                print(f"  Status: {first_ticket.get('fields', {}).get('status', {}).get('name')}")
        
        # Test search functionality
        print("\n4. Testing search...")
        search_results = await mcp_client.search_jira("project = ACE", max_results=5)
        print(f"✅ Search returned {len(search_results)} results")
        
        # Disconnect
        print("\n5. Disconnecting...")
        await mcp_client.disconnect()
        print("✅ Disconnected successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_sync_test():
    """Test synchronous wrapper"""
    print("\n" + "=" * 60)
    print("Testing Synchronous Wrapper")
    print("=" * 60)
    
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    # This simulates what JiraDataAgent does
    try:
        import concurrent.futures
        
        def run_in_thread():
            # Create new event loop in thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Create and connect client
                mcp_client = JiraConfluenceMCPClient(redis_client)
                loop.run_until_complete(mcp_client.connect_to_jira_server())
                
                # Get projects
                projects = loop.run_until_complete(mcp_client.get_available_projects())
                print(f"✅ Projects from thread: {projects}")
                
                # Get tickets
                if projects:
                    tickets = loop.run_until_complete(
                        mcp_client.get_project_tickets(projects[0])
                    )
                    print(f"✅ Got {len(tickets)} tickets from thread")
                    return tickets
                
                return []
                
            finally:
                loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            result = future.result(timeout=30)
            print(f"✅ Thread execution successful: {len(result)} tickets")
            
    except Exception as e:
        print(f"❌ Thread execution failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    # First test async
    success = await test_mcp_connection()
    
    if success:
        print("\n✅ Async test passed!")
    else:
        print("\n❌ Async test failed!")
    
    # Then test sync wrapper
    run_sync_test()

if __name__ == "__main__":
    # Check if MCP server is running
    print("Make sure the Jira MCP server is running!")
    print("Run: python -m mcp_integration.servers.jira_mcp_server")
    print("")
    
    asyncio.run(main())
#!/usr/bin/env python
"""Debug script to check orchestrator initialization"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.core.orchestrator import orchestrator # type: ignore

print("=" * 60)
print("🔍 Debugging Orchestrator Initialization")
print("=" * 60)

# Check JiraDataAgent
jira_agent = orchestrator.jira_data_agent
print(f"\n📊 JiraDataAgent Status:")
print(f"   - Using Real API: {getattr(jira_agent, 'use_real_api', False)}")
print(f"   - Using MCP: {getattr(jira_agent, 'use_mcp', False)}")
print(f"   - Available Projects: {getattr(jira_agent, 'available_projects', [])}")

# Check if Jira client exists
if hasattr(jira_agent, 'jira_client'):
    print(f"   - Jira Client Initialized: ✅")
    
    # Test connection
    print(f"\n🔌 Testing Jira Client Connection...")
    if jira_agent.jira_client.test_connection():
        print(f"   ✅ Connected to Jira!")
        
        # Get projects directly
        projects = jira_agent.jira_client.get_projects()
        print(f"   📁 Found {len(projects)} projects:")
        for p in projects[:5]:
            print(f"      - {p['key']}: {p['name']}")
    else:
        print(f"   ❌ Connection failed!")
else:
    print(f"   - Jira Client Initialized: ❌")
    print(f"   - Using mock data!")

# Test a simple query
print(f"\n🧪 Testing a simple workflow...")
try:
    result = orchestrator.run_workflow("Hello", "test-123")
    print(f"   ✅ Workflow completed")
    print(f"   Response: {result.get('response', 'No response')[:100]}...")
except Exception as e:
    print(f"   ❌ Workflow failed: {e}")
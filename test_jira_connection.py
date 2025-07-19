#!/usr/bin/env python
"""Test script to verify Jira connection"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.jira_api_client import JiraAPIClient
from integrations.api_config import APIConfig

def test_connection():
    print("=" * 60)
    print("🔧 Testing Jira Connection")
    print("=" * 60)
    
    # Show configuration
    print(f"📍 Jira URL: {APIConfig.JIRA_URL}")
    print(f"👤 Username: {APIConfig.JIRA_USERNAME}")
    print(f"🔑 Password: {'*' * len(APIConfig.JIRA_PASSWORD)}")
    print()
    
    # Create client
    client = JiraAPIClient()
    
    # Test connection
    print("🔌 Testing connection...")
    if client.test_connection():
        print("✅ Connection successful!")
        
        # Get projects
        print("\n📁 Fetching projects...")
        projects = client.get_projects()
        
        if projects:
            print(f"✅ Found {len(projects)} projects:")
            for project in projects:
                print(f"   - {project['key']}: {project['name']}")
                
            # Test getting issues from first project
            if projects:
                first_project = projects[0]['key']
                print(f"\n📋 Testing issue retrieval for project {first_project}...")
                
                try:
                    issues = client.get_all_issues_for_project(first_project)
                    print(f"✅ Found {len(issues)} issues in {first_project}")
                except Exception as e:
                    print(f"❌ Error getting issues: {e}")
        else:
            print("❌ No projects found!")
    else:
        print("❌ Connection failed!")
        print("\nPossible issues:")
        print("1. Is Jira running at http://localhost:2990/jira?")
        print("2. Are the credentials correct? (admin/admin)")
        print("3. Is there a firewall blocking the connection?")

if __name__ == "__main__":
    test_connection()
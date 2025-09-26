#!/usr/bin/env python3
"""Test Jira API connection"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.api_config import APIConfig
from integrations.jira_api_client import JiraAPIClient

def test_jira_connection():
    print("🧪 Testing Jira Connection...")
    print("=" * 60)
    APIConfig.log_config()
    print("=" * 60)
    client = JiraAPIClient()
    

    if client.test_connection():
        print("\n✅ Connection successful!")
        print("\n📋 Fetching projects...")
        projects = client.get_projects()
        
        if projects:
            print(f"\nFound {len(projects)} projects:")
            for i, project in enumerate(projects[:5]):  
                print(f"  {i+1}. {project['key']} - {project['name']}")
            
            if len(projects) > 5:
                print(f"  ... and {len(projects) - 5} more")

            if projects:
                test_project = projects[0]['key']
                print(f"\n📝 Testing issue retrieval for project: {test_project}")
                
                issues = client.get_all_issues_for_project(test_project)
                print(f"Found {len(issues)} issues in {test_project}")
                
                if issues:
                    print(f"\nFirst issue:")
                    issue = issues[0]
                    print(f"  Key: {issue['key']}")
                    print(f"  Summary: {issue['fields']['summary']}")
                    print(f"  Status: {issue['fields']['status']['name']}")
        else:
            print("\n⚠️ No projects found!")
    else:
        print("\n❌ Connection failed!")
        print("Please check your credentials and Jira URL in .env file")

if __name__ == "__main__":
    test_jira_connection()
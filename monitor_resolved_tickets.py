"""Monitor Jira for resolved tickets and auto-generate articles"""
import time
from datetime import datetime, timedelta
from orchestrator.core.orchestrator import orchestrator # type: ignore
from integrations.jira_api_client import JiraAPIClient

def monitor_resolved_tickets():
    print("🔍 Monitoring Jira for resolved tickets...")
    
    client = JiraAPIClient()
    processed_tickets = set()
    
    # Load previously processed tickets from file
    try:
        with open("processed_tickets.txt", "r") as f:
            processed_tickets = set(line.strip() for line in f)
    except:
        pass
    
    while True:
        try:
            # Get all projects
            projects = client.get_projects()
            
            for project in projects:
                project_key = project['key']
                print(f"\n📋 Checking project: {project_key}")
                
                # JQL to find resolved tickets in last 24 hours
                jql = f"project = {project_key} AND status = Done AND resolved >= -24h ORDER BY resolved DESC"
                
                result = client.search_issues(jql)
                issues = result.get("issues", [])
                
                for ticket in issues:
                    ticket_key = ticket["key"]
                    
                    if ticket_key not in processed_tickets:
                        print(f"\n✅ Found newly resolved ticket: {ticket_key}")
                        print(f"   Summary: {ticket['fields']['summary']}")
                        print(f"   Resolved by: {ticket['fields'].get('assignee', {}).get('displayName', 'Unknown')}")
                        
                        # Generate article
                        print(f"   📝 Generating article...")
                        try:
                            state = orchestrator.run_jira_workflow(ticket_key, project_id=project_key)
                            
                            if state.get("workflow_status") == "success":
                                print(f"   ✅ Article generated successfully!")
                                print(f"   Title: {state.get('article', {}).get('title', 'N/A')}")
                                
                                # Mark as processed
                                processed_tickets.add(ticket_key)
                                with open("processed_tickets.txt", "a") as f:
                                    f.write(f"{ticket_key}\n")
                            else:
                                print(f"   ❌ Article generation failed")
                                
                        except Exception as e:
                            print(f"   ❌ Error generating article: {e}")
            
            print(f"\n💤 Sleeping for 5 minutes...")
            time.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_resolved_tickets()
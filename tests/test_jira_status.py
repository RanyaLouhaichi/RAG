# test_jira_status.py
from orchestrator.core.orchestrator import orchestrator # type: ignore

# Test direct Jira data retrieval
jira_agent = orchestrator.jira_data_agent

# Get raw data
result = jira_agent.run({
    "project_id": "ABDERA",
    "analysis_depth": "enhanced"
})

tickets = result.get("tickets", [])
print(f"Total tickets: {len(tickets)}")

# Analyze status distribution
status_counts = {}
for ticket in tickets[:30]:  # First 10 tickets
    fields = ticket.get("fields", {})
    status = fields.get("status", {}).get("name", "Unknown")
    status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\nTicket: {ticket.get('key')}")
    print(f"  Status: {status}")
    print(f"  Summary: {fields.get('summary', 'No summary')[:50]}")
    print(f"  Created: {fields.get('created', 'Unknown')}")
    print(f"  Updated: {fields.get('updated', 'Unknown')}")
    print(f"  Resolution: {fields.get('resolution', 'Not resolved')}")

print(f"\nStatus distribution: {status_counts}")
import requests
import json
from datetime import datetime

# Test the webhook with detailed logging
webhook_url = "http://localhost:5000/webhook/jira-ticket-resolved"

print(f"Testing webhook at {webhook_url}")
print("=" * 50)

payload = {
    "webhookEvent": "jira:issue_updated",
    "issue": {
        "key": "ABDERA-TEST-001",
        "fields": {
            "project": {"key": "ABDERA"},
            "summary": "Test ticket for webhook debugging",
            "status": {"name": "Done"},
            "resolutiondate": datetime.now().isoformat()
        }
    },
    "changelog": {
        "items": [{
            "field": "status",
            "fromString": "In Progress",
            "toString": "Done"
        }]
    }
}

try:
    print("Sending webhook...")
    response = requests.post(webhook_url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 202:
        print("\n✅ Webhook accepted! Check webhook_activity.log for details")
    else:
        print("\n❌ Webhook failed!")
        
except Exception as e:
    print(f"Error: {e}")
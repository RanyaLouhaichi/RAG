# Create check_result.py
import requests
import json
import time

ticket_id = "ABDERA-TEST-001"

print(f"Checking status for {ticket_id}...")

# Wait a bit for processing
for i in range(10):
    time.sleep(2)
    
    response = requests.get(f"http://localhost:5000/webhook/status/{ticket_id}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nStatus: {result.get('status')}")
        print(f"Timestamp: {result.get('timestamp')}")
        
        article = result.get('article', {})
        if article:
            print(f"Article Title: {article.get('title')}")
            print(f"Article Status: {article.get('status')}")
            print(f"\nArticle Preview:")
            print("-" * 50)
            print(article.get('content', 'No content')[:500])
            print("-" * 50)
        break
    else:
        print(f"Attempt {i+1}: Still processing...")

# Also check the dashboard
print(f"\nCheck the dashboard at: http://localhost:5000/webhook/dashboard")
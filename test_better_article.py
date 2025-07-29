# test_better_article.py
import requests
import json

# First, clear any existing article
ticket_id = "OBSERVE-14"

response = requests.post(
    f"http://localhost:5001/api/article/generate/{ticket_id}",
    json = {
    "projectKey": "OBSERVE",
    "summary": "User activity logs do not show timestamps",
    "description": "The user activity log page lists actions performed by users, but does not display the date and time for each action, making it difficult to track when changes occurred.",
    "type": "Bug",
    "status": "Done"
},


    headers={"Content-Type": "application/json"}
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    article = response.json().get("article", {})
    print("\nGenerated Article:")
    print("="*50)
    print(article.get("content", "No content"))
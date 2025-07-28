# test_better_article.py
import requests
import json

# First, clear any existing article
ticket_id = "OBSERVE-2"

response = requests.post(
    f"http://localhost:5001/api/article/generate/{ticket_id}",
    json = {
    "projectKey": "OBSERVE",
    "summary": "Search function returns outdated results after data update",
    "description": "When new data is added or updated, the search results still show old values until a manual refresh. Users expect the search to reflect the latest data immediately.",
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
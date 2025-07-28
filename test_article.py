# test_article.py
import requests
import json

response = requests.post(
    "http://localhost:5001/api/article/generate/OBSERVE-9",
    json={},
    headers={"Content-Type": "application/json"}
)

print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
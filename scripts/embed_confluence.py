import json
from sentence_transformers import SentenceTransformer  # type: ignore
import chromadb  # type: ignore
from chromadb.config import Settings # type: ignore


# Load the JSON data
with open("data/confluence_articles.json", "r") as f:
    data = json.load(f)
    articles = data["articles"]

# Initialize the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Chroma client with a persistent database path
client = chromadb.PersistentClient(path="c:/Users/tlouh/Desktop/JURIX/chromadb_data")
collection = client.get_or_create_collection(name="confluence_articles")

# Embed and store articles
for article in articles:
    embedding = model.encode(article["content"]).tolist()
    # Convert tags list to a string
    tags_str = ", ".join(article["metadata"]["tags"]) if article["metadata"]["tags"] else ""
    collection.add(
        ids=[article["id"]],
        embeddings=[embedding],
        documents=[article["content"]],
        metadatas=[{"title": article["title"], "space": article["space"], "tags": tags_str}]
    )

print("Articles embedded and stored in Chroma DB.")
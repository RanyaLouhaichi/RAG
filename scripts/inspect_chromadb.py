import chromadb  # type: ignore

# Initialize Chroma client
client = chromadb.PersistentClient(path="c:/Users/tlouh/Desktop/JURIX/chromadb_data")

# List all collections
collections = client.list_collections()
print("Collections in the database:")
for collection in collections:
    print(f"- {collection.name}")

# Try to access 'confluence_articles'
try:
    collection = client.get_collection(name="confluence_articles")
    print("\n'confluence_articles' collection exists.")

    # Retrieve all documents
    documents = collection.get()
    print(f"Documents in 'confluence_articles':")
    for i, doc in enumerate(documents["documents"]):
        print(f"  {i+1}. {doc[:100]}...")  # Affiche les 100 premiers caractères
except ValueError:
    print("\n'confluence_articles' collection does not exist.")

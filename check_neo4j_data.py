#!/usr/bin/env python3
"""Check what data is in the RAG system"""

import os
from dotenv import load_dotenv
from orchestrator.rag.enhanced_rag_pipeline import EnhancedRAGPipeline # type: ignore
from neo4j import GraphDatabase
import chromadb

load_dotenv()

def check_rag_data():
    # Initialize pipeline
    pipeline = EnhancedRAGPipeline(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "jurix_neo4j_password"),
        confluence_url=os.getenv("CONFLUENCE_URL"),
        confluence_user=os.getenv("CONFLUENCE_USERNAME"),
        confluence_password=os.getenv("CONFLUENCE_PASSWORD")
    )
    
    print("\n=== NEO4J DATA ===")
    with pipeline.neo4j_manager.driver.session() as session:
        # Check documents
        result = session.run("MATCH (d:Document) RETURN d.id, d.title, d.content LIMIT 5")
        docs = list(result)
        print(f"Documents: {len(docs)}")
        for doc in docs:
            print(f"  - {doc['d.id']}: {doc['d.title']}")
            print(f"    Content preview: {doc['d.content'][:100]}...")
        
        # Check chunks
        result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
        chunk_count = result.single()["count"]
        print(f"\nChunks: {chunk_count}")
        
        # Check tickets
        result = session.run("MATCH (t:Ticket) RETURN count(t) as count")
        ticket_count = result.single()["count"]
        print(f"Tickets: {ticket_count}")
        
        # Check relationships
        result = session.run("MATCH ()-[r]->() RETURN type(r), count(r) as count")
        print("\nRelationships:")
        for record in result:
            print(f"  - {record['type(r)']}: {record['count']}")
    
    print("\n=== CHROMADB DATA ===")
    # Check ChromaDB collections
    print(f"Document collection: {pipeline.doc_collection.count()} documents")
    print(f"Chunk collection: {pipeline.chunk_collection.count()} chunks")
    
    # Sample from collections
    if pipeline.doc_collection.count() > 0:
        sample = pipeline.doc_collection.get(limit=2)
        print("\nSample documents:")
        for i in range(len(sample['ids'])):
            print(f"  - ID: {sample['ids'][i]}")
            print(f"    Metadata: {sample['metadatas'][i]}")
    
    if pipeline.chunk_collection.count() > 0:
        sample = pipeline.chunk_collection.get(limit=2)
        print("\nSample chunks:")
        for i in range(len(sample['ids'])):
            print(f"  - ID: {sample['ids'][i]}")
            print(f"    Quality score: {sample['metadatas'][i].get('quality_score', 'N/A')}")

if __name__ == "__main__":
    check_rag_data()
#!/usr/bin/env python3
"""Setup script for enhanced RAG system"""

import os
import sys
import subprocess
import time
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_Setup")

def check_docker():
    """Check if Docker is installed and running"""
    try:
        subprocess.run(['docker', '--version'], check=True, capture_output=True)
        subprocess.run(['docker', 'ps'], check=True, capture_output=True)
        logger.info("✅ Docker is installed and running")
        return True
    except subprocess.CalledProcessError:
       logger.error("❌ Docker is not installed or not running")
       return False

def start_neo4j():
   """Start Neo4j using docker-compose"""
   logger.info("Starting Neo4j...")
   try:
       subprocess.run(['docker-compose', '-f', 'docker-compose-neo4j.yml', 'up', '-d'], check=True)
       logger.info("⏳ Waiting for Neo4j to start (30 seconds)...")
       time.sleep(30)
       
       # Test Neo4j connection
       from neo4j import GraphDatabase
       uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
       auth = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "jurix_neo4j_password"))
       
       driver = GraphDatabase.driver(uri, auth=auth)
       with driver.session() as session:
           result = session.run("RETURN 1 as test")
           if result.single()["test"] == 1:
               logger.info("✅ Neo4j is running and accessible")
               driver.close()
               return True
       
   except Exception as e:
       logger.error(f"❌ Failed to start Neo4j: {e}")
       return False

def install_dependencies():
   """Install Python dependencies"""
   logger.info("Installing RAG dependencies...")
   try:
       subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements_rag.txt'], check=True)
       logger.info("✅ Dependencies installed")
       return True
   except subprocess.CalledProcessError as e:
       logger.error(f"❌ Failed to install dependencies: {e}")
       return False

def test_imports():
   """Test that all required imports work"""
   logger.info("Testing imports...")
   try:
       import neo4j
       import docling
       from sentence_transformers import SentenceTransformer
       import chromadb
       import tiktoken
       logger.info("✅ All imports successful")
       return True
   except ImportError as e:
       logger.error(f"❌ Import failed: {e}")
       return False

def initialize_rag_system():
   """Initialize the RAG system components"""
   logger.info("Initializing RAG system...")
   
   try:
       from orchestrator.rag.enhanced_rag_pipeline import EnhancedRAGPipeline # type: ignore
       
       # Initialize pipeline
       pipeline = EnhancedRAGPipeline(
           neo4j_uri=os.getenv("NEO4J_URI"),
           neo4j_user=os.getenv("NEO4J_USERNAME"),
           neo4j_password=os.getenv("NEO4J_PASSWORD"),
           confluence_url=os.getenv("CONFLUENCE_URL"),
           confluence_user=os.getenv("CONFLUENCE_USERNAME"),
           confluence_password=os.getenv("CONFLUENCE_PASSWORD")
       )
       
       logger.info("✅ RAG pipeline initialized successfully")
       return pipeline
       
   except Exception as e:
       logger.error(f"❌ Failed to initialize RAG system: {e}")
       return None

def main():
   """Main setup function"""
   logger.info("🚀 Starting RAG Enhancement Setup")
   
   # Load environment variables
   load_dotenv()
   
   # Check prerequisites
   if not check_docker():
       logger.error("Please install and start Docker first")
       return False
   
   # Install dependencies
   if not install_dependencies():
       return False
   
   # Test imports
   if not test_imports():
       return False
   
   # Start Neo4j
   if not start_neo4j():
       return False
   
   # Initialize RAG system
   pipeline = initialize_rag_system()
   if not pipeline:
       return False
   
   logger.info("✨ RAG Enhancement setup complete!")
   logger.info("\nNext steps:")
   logger.info("1. Run ingestion: python ingest_confluence.py <SPACE_KEY>")
   logger.info("2. Update your orchestrator to use EnhancedRetrievalAgent")
   logger.info("3. Neo4j Browser available at: http://localhost:7474")
   
   return True

if __name__ == "__main__":
   success = main()
   sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""Script to ingest Confluence spaces into the enhanced RAG system"""

import os
import sys
import logging
import argparse
from dotenv import load_dotenv
from datetime import datetime

from orchestrator.rag.enhanced_rag_pipeline import EnhancedRAGPipeline # type: ignore
from agents.jira_data_agent import JiraDataAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ConfluenceIngestion")

def ingest_confluence_space(space_key: str, limit: int = 100):
    """Ingest a Confluence space"""
    logger.info(f"Starting ingestion for space: {space_key}")
    
    # Initialize pipeline
    pipeline = EnhancedRAGPipeline(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USERNAME"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        confluence_url=os.getenv("CONFLUENCE_URL"),
        confluence_user=os.getenv("CONFLUENCE_USERNAME"),
        confluence_password=os.getenv("CONFLUENCE_PASSWORD")
    )
    
    # Ingest space
    pipeline.ingest_confluence_space(space_key, limit)
    
    logger.info(f"Completed ingestion for space: {space_key}")

def sync_jira_tickets(project_key: str):
    """Sync Jira tickets to knowledge graph"""
    logger.info(f"Syncing Jira tickets for project: {project_key}")
    
    # Initialize Jira agent
    import redis
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    jira_agent = JiraDataAgent(redis_client=redis_client)
    
    # Get tickets
    result = jira_agent.run({'project_id': project_key})
    tickets = result.get('tickets', [])
    
    if tickets:
        # Initialize pipeline
        pipeline = EnhancedRAGPipeline(
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USERNAME"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            confluence_url=os.getenv("CONFLUENCE_URL"),
            confluence_user=os.getenv("CONFLUENCE_USERNAME"),
            confluence_password=os.getenv("CONFLUENCE_PASSWORD")
        )
        
        # Add tickets to graph
        pipeline.add_jira_tickets(tickets)
        pipeline.update_document_impact_scores()
        
        logger.info(f"Synced {len(tickets)} tickets from {project_key}")
    else:
        logger.warning(f"No tickets found for project: {project_key}")

def main():
    parser = argparse.ArgumentParser(description='Ingest Confluence spaces and Jira tickets')
    parser.add_argument('--space', type=str, help='Confluence space key to ingest')
    parser.add_argument('--project', type=str, help='Jira project key to sync')
    parser.add_argument('--limit', type=int, default=100, help='Limit number of documents to ingest')
    parser.add_argument('--all-spaces', action='store_true', help='Ingest all available spaces')
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    if args.all_spaces:
        # Get all spaces
        from integrations.docling.confluence_extractor import ConfluenceDoclingExtractor
        extractor = ConfluenceDoclingExtractor(
            os.getenv("CONFLUENCE_URL"),
            os.getenv("CONFLUENCE_USERNAME"),
            os.getenv("CONFLUENCE_PASSWORD")
        )
        
        spaces = extractor.get_all_spaces()
        logger.info(f"Found {len(spaces)} spaces")
        
        for space in spaces:
            try:
                ingest_confluence_space(space['key'], args.limit)
            except Exception as e:
                logger.error(f"Failed to ingest space {space['key']}: {e}")
                continue
    
    elif args.space:
        ingest_confluence_space(args.space, args.limit)
    
    if args.project:
        sync_jira_tickets(args.project)
    
    if not args.space and not args.project and not args.all_spaces:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
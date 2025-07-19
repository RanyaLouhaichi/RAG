#!/usr/bin/env python3
"""Test collaborative retrieval with enhanced RAG"""

import logging
from orchestrator.core.orchestrator import orchestrator  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestCollaboration")

def test_recommendation_with_enhanced_retrieval():
    """Test that recommendation agent gets articles from enhanced retrieval"""
    
    query = "Give me recommendations for improving our onboarding process For Onboarding Project and share the relevant articles from confluence."
    
    logger.info(f"\nTesting collaborative query: {query}")
    print(f"\n=== Testing collaborative query: {query}\n")
    
    result = orchestrator.run_workflow(query)
    
    # Check results
    articles = result.get('articles', [])
    recommendations = result.get('recommendations', [])
    collaboration_metadata = result.get('collaboration_metadata', {})
    
    # PRINT summary
    print("\n📊 Results:")
    print(f"Articles found: {len(articles)}")
    print(f"Recommendations: {len(recommendations)}")
    print(f"Collaboration occurred: {collaboration_metadata.get('is_collaborative', False)}")
    print(f"Collaborating agents: {collaboration_metadata.get('collaborating_agents', [])}")
    
    # PRINT details for first 3 articles
    for i, article in enumerate(articles[:3]):
        print(f"\n📄 Article {i+1}: {article.get('title', 'Untitled')}")
        print(f"   Sources: {article.get('sources', [])}")
        print(f"   From graph: {article.get('from_knowledge_graph', False)}")
        print(f"   Score breakdown: {article.get('score_breakdown', {})}")
    
    # PRINT collaboration trace
    if 'collaboration_trace' in result:
        print("\n🤝 Collaboration Trace:")
        for trace in result['collaboration_trace']:
            print(f"   {trace['node']}: {trace.get('articles_count', 0)} articles")

if __name__ == "__main__":
    test_recommendation_with_enhanced_retrieval()

#!/usr/bin/env python3
# Test script to validate your hybrid architecture implementation

import sys
import os
sys.path.append(os.path.abspath('.'))

from orchestrator.core.orchestrator import run_workflow # type: ignore
import logging

def test_hybrid_recommendations():
    """Test collaborative recommendation generation"""
    print("🧪 Testing Hybrid Architecture - Collaborative Recommendations")
    print("=" * 60)
    
    # Test query that should trigger collaboration
    test_query = "Our team velocity seems low, what recommendations do you have for improving productivity?"
    
    print(f"Query: {test_query}")
    print("\nRunning workflow with hybrid architecture...")
    
    result = run_workflow(test_query)
    
    print(f"\nResults:")
    print(f"Intent: {result.get('intent', {}).get('intent', 'Unknown')}")
    print(f"Response: {result.get('response', 'No response')[:200]}...")
    print(f"Recommendations: {len(result.get('recommendations', []))}")
    
    # Check if collaboration occurred
    collab_info = result.get('collaboration_info', {})
    if collab_info:
        print(f"\n🤝 Collaboration Details:")
        print(f"Primary Agent: {collab_info.get('primary_agent', 'Unknown')}")
        print(f"Collaborating Agents: {collab_info.get('collaborating_agents', [])}")
        print(f"Collaboration Types: {collab_info.get('collaboration_types', [])}")
    else:
        print("\n❌ No collaboration detected")
    
    return result

def test_hybrid_data_analysis():
    """Test collaborative data analysis"""
    print("\n🧪 Testing Hybrid Architecture - Collaborative Data Analysis")
    print("=" * 60)
    
    test_query = "Show me detailed analytics for project PROJ123 with forecasting"
    
    print(f"Query: {test_query}")
    print("\nRunning workflow with hybrid architecture...")
    
    result = run_workflow(test_query)
    
    print(f"\nResults:")
    print(f"Tickets retrieved: {len(result.get('tickets', []))}")
    print(f"Additional metrics: {bool(result.get('additional_metrics'))}")
    
    collab_info = result.get('collaboration_info', {})
    if collab_info:
        print(f"\n🤝 Collaboration Details:")
        print(f"Collaborating Agents: {collab_info.get('collaborating_agents', [])}")
    
    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Starting Hybrid Architecture Tests")
    print("=" * 60)
    
    try:
        # Test collaborative recommendations
        rec_result = test_hybrid_recommendations()
        
        # Test collaborative data analysis  
        data_result = test_hybrid_data_analysis()
        
        print("\n✅ Hybrid Architecture Tests Completed!")
        print("Check the collaboration_info in results to see agent coordination")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
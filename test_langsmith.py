#!/usr/bin/env python3
"""Test script to verify LangSmith integration"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:5001"

def test_langsmith_integration():
    """Run comprehensive LangSmith integration test"""
    
    print("=" * 60)
    print("LANGSMITH INTEGRATION TEST FOR JURIX THESIS")
    print("=" * 60)
    
    # Test 1: Check if LangSmith is enabled
    print("\n1. Testing LangSmith metrics endpoint...")
    response = requests.get(f"{BASE_URL}/api/langsmith-metrics")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ LangSmith Enabled: {data['langsmith_enabled']}")
        print(f"📊 Dashboard URL: {data.get('dashboard_url', 'N/A')}")
    else:
        print(f"❌ Failed to get metrics: {response.status_code}")
    
    # Test 2: Run a traced workflow
    print("\n2. Running traced workflow test...")
    test_queries = [
        "What are the recommendations for improving sprint velocity in project OBSERVE?",
        "Analyze the productivity metrics for our OBSERVE team",
        "Predict if we will complete the current sprint for the OBSERVE project",
        "Find articles about agile best practices"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Test {i}: {query[:50]}...")
        
        response = requests.post(
            f"{BASE_URL}/api/langsmith-test",
            json={"query": query}
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data.get("result", {})
            metrics = data.get("metrics", {})
            
            print(f"   ✅ Workflow traced successfully")
            print(f"   📈 Agents used: {result.get('agents_used', 0)}")
            print(f"   🤝 Collaborations: {result.get('collaboration_count', 0)}")
            print(f"   ⚡ Total runs: {metrics.get('total_runs', 0)}")
            
            # Show thesis metrics
            thesis_metrics = metrics.get("thesis_metrics", {})
            if thesis_metrics:
                print(f"\n   📊 THESIS METRICS:")
                print(f"      - System Complexity: {thesis_metrics.get('system_complexity', 0)}")
                print(f"      - System Reliability: {thesis_metrics.get('system_reliability', 0):.2%}")
                print(f"      - Collaboration Density: {thesis_metrics.get('collaboration_density', 0):.2f}")
                print(f"      - Avg Response Time: {thesis_metrics.get('avg_response_time', 0):.3f}s")
        else:
            print(f"   ❌ Test failed: {response.json().get('error', 'Unknown error')}")
        
        time.sleep(2)  # Avoid rate limiting
    
    # Test 3: Get final metrics summary
    print("\n3. Getting final metrics summary...")
    response = requests.get(f"{BASE_URL}/api/langsmith-metrics")
    if response.status_code == 200:
        data = response.json()
        metrics = data.get("metrics", {})
        
        print("\n📊 FINAL LANGSMITH METRICS FOR THESIS:")
        print("-" * 40)
        
        # Agent statistics
        agent_stats = metrics.get("agent_statistics", {})
        print(f"Agents:")
        print(f"  - Total runs: {agent_stats.get('total_agent_runs', 0)}")
        print(f"  - Unique agents: {agent_stats.get('unique_agents', 0)}")
        if agent_stats.get("most_active_agents"):
            print(f"  - Most active: {agent_stats['most_active_agents'][:3]}")
        
        # Workflow statistics
        workflow_stats = metrics.get("workflow_statistics", {})
        print(f"\nWorkflows:")
        print(f"  - Total runs: {workflow_stats.get('total_workflow_runs', 0)}")
        print(f"  - Types: {list(workflow_stats.get('workflow_breakdown', {}).keys())}")
        
        # Performance metrics
        perf_metrics = metrics.get("performance_metrics", {})
        print(f"\nPerformance:")
        print(f"  - Avg latency: {perf_metrics.get('average_latency_seconds', 0):.3f}s")
        print(f"  - Error rate: {perf_metrics.get('error_rate', 0):.2%}")
        
        # Collaboration insights
        collab_insights = metrics.get("collaboration_insights", {})
        print(f"\nCollaboration:")
        print(f"  - Total collaborations: {collab_insights.get('total_collaborations', 0)}")
        print(f"  - Unique pairs: {collab_insights.get('unique_collaboration_pairs', 0)}")
        
        # Export location
        print(f"\n📁 Metrics exported to: {data.get('export_file', 'N/A')}")
        print(f"🔗 View in LangSmith: {data.get('dashboard_url', 'N/A')}")
        
    print("\n" + "=" * 60)
    print("TEST COMPLETE - Check LangSmith Dashboard")
    print("=" * 60)

if __name__ == "__main__":
    test_langsmith_integration()
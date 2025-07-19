# main.py (updated to show model usage)
import asyncio
import json
import sys
import os
from orchestrator.core.orchestrator import orchestrator # type: ignore
import uuid

def interactive_workflow():
    """Interactive command-line interface"""
    print("\n🤖 JURIX AI System with Dynamic Model Intelligence")
    print("=" * 60)
    # Check if we're using real API
    if hasattr(orchestrator.jira_data_agent, 'use_real_api') and orchestrator.jira_data_agent.use_real_api:
        print("📋 Connected to real Jira API!")
    print("Type your questions or commands:")
    print("  'dashboard <project>' - Generate productivity dashboard")
    print("  'article <ticket>' - Generate article from ticket")
    print("  'stats' - Show model performance statistics")
    print("  'api' - Start API server")
    print("  'mcp status' - Show MCP system status")
    print("  'mcp capabilities' - Show MCP server capabilities")
    print("  'mcp tool <server> <tool>' - Call an MCP tool")
    print("  'quit' - Exit")
    print("=" * 60)
    
    conversation_id = str(uuid.uuid4())
    
    # Access model manager for stats
    model_manager = orchestrator.chat_agent.model_manager
    
    while True:
        try:
            query = input("\n🎯 Your question > ").strip()
            
            if query.lower() == "quit":
                print("👋 Goodbye!")
                break
                
            elif query.lower() == "stats":
                print("\n📊 Dynamic Model Performance by Agent:")
                stats = orchestrator.shared_model_manager.get_agent_performance_stats()
                
                print(f"\n⏱️ Session Info:")
                print(f"  Duration: {stats['session_info']['duration']:.1f}s")
                print(f"  Total Requests: {stats['session_info']['total_requests']}")
                print(f"  Model Switches: {stats['session_info']['model_switches']}")

                print(f"\n📊 Detailed Model Switching:")
                for agent, data in summary.get("agent_model_usage", {}).items():
                    if len(data["models_used"]) > 1:
                        print(f"  {agent} switched models: {data['models_used']}")
                        print(f"    Likely reason: Exploration or fallback after quality issue")
                
                # Show available models
                print(f"\n🎮 Available Models: {orchestrator.shared_model_manager.available_models}")
                
                print(f"\n🤖 Agent Performance:")
                for agent_name, agent_stats in stats['by_agent'].items():
                    if agent_stats['total_requests'] > 0 or agent_stats['model_performance']:
                        print(f"\n  {agent_name}:")
                        print(f"    Session Requests: {agent_stats['total_requests']}")
                        print(f"    Models Used: {agent_stats['models_used']}")
                        if agent_stats.get('preferred_model'):
                            print(f"    Preferred Model: {agent_stats['preferred_model']}")
                        
                        if agent_stats['model_performance']:
                            print(f"    Model Performance:")
                            for model, perf in agent_stats['model_performance'].items():
                                print(f"      {model}: {perf['uses']} uses, "
                                    f"{perf['success_rate']:.0%} success, "
                                    f"{perf['avg_quality']:.2f} quality, "
                                    f"{perf['avg_time']:.2f}s avg")
                        print(f"    Avg Response Time: {agent_stats['avg_response_time']:.2f}s")    
            elif query.lower() == "api":
                print("Starting API server...")
                from api import app
                app.run(debug=True, host='0.0.0.0', port=5000)
                
            elif query.lower().startswith("dashboard"):
                parts = query.split()
                project_id = parts[1] if len(parts) > 1 else "PROJ123"
                
                print(f"⏳ Generating dashboard for {project_id}...")
                state = orchestrator.run_productivity_workflow(
                    project_id,
                    {
                        "start": "2025-05-01T00:00:00Z",
                        "end": "2025-05-17T23:59:59Z"
                    }
                )
                
                print(f"\n📊 Dashboard generated!")
                print(f"Dashboard ID: {state.get('dashboard_id', 'N/A')}")
                print(f"Metrics: {bool(state.get('metrics'))}")
                print(f"Report: {state.get('report', '')[:200]}...")

            elif query.lower() == "mcp status":
                status = orchestrator.get_mcp_status()
                print("\n🔌 MCP System Status:")
                print(f"  Servers: {status['total_servers']}")
                print(f"  Connected: {status['connected_clients']}")
                for server, info in status['servers'].items():
                    print(f"  - {server}: {'✅' if info['connected'] else '❌'}")
            
            elif query.lower() == "mcp capabilities":
                print("\n🔍 Discovering MCP capabilities...")
                capabilities = asyncio.run(orchestrator.get_mcp_capabilities())
                for server, caps in capabilities.items():
                    print(f"\n📦 {server}:")
                    print(f"  Tools: {len(caps['tools'])}")
                    for tool in caps['tools']:
                        print(f"    - {tool['name']}: {tool['description']}")
            
            elif query.lower().startswith("mcp tool"):
                parts = query.split()
                if len(parts) >= 4:
                    server = parts[2]
                    tool = parts[3]
                    args = {} if len(parts) <= 4 else json.loads(' '.join(parts[4:]))
                    
                    result = asyncio.run(
                        orchestrator.mcp_manager.mcp_client.call_tool(server, tool, args)
                    )
                    print(f"\n✅ Tool result: {json.dumps(result, indent=2)}")
                
            elif query.lower().startswith("article"):
                parts = query.split()
                ticket_id = parts[1] if len(parts) > 1 else "TICKET-001"
                
                print(f"⏳ Generating article for {ticket_id}...")
                state = orchestrator.run_jira_workflow(ticket_id)
                
                print(f"\n📄 Article generation status: {state.get('workflow_status')}")
                if state.get('article'):
                    print(f"Title: {state['article'].get('title', 'N/A')}")
                    print(f"Stage: {state.get('workflow_stage', 'N/A')}")
            
            # In main.py, add this option:
            elif query.lower() == "workflow":
                # Show last workflow details
                print("\n📊 Last Workflow Model Usage:")
                # Get recent workflow from Redis
                workflow_keys = orchestrator.shared_model_manager.redis_client.keys("workflow_models:*")
                if workflow_keys:
                    # Get the most recent one
                    latest_key = sorted(workflow_keys)[-1]
                    workflow_id = latest_key.split(":")[-1]
                    summary = orchestrator.shared_model_manager.get_workflow_summary(workflow_id)
                    
                    print(f"\nWorkflow ID: {workflow_id}")
                    for agent_name, usage in summary.get("agent_model_usage", {}).items():
                        print(f"\n  {agent_name}:")
                        print(f"    Total Calls: {usage['total_calls']}")
                        print(f"    Models Used: {usage['models_used']}")
                        for model, count in usage['model_counts'].items():
                            print(f"      {model}: {count} times")
                            
            else:
                # Regular chat workflow
                print(f"\n💭 Processing with dynamic model selection...")
                final_state = orchestrator.run_workflow(query, conversation_id)
                
                print(f"\n🤖 Response:")
                print(final_state.get("response", "No response generated"))
                
                # Show model usage from logs
                print(f"\n🧠 Model Intelligence Active - Check logs for model selection details")

                # Show model usage for THIS workflow
                if final_state.get("model_usage_summary"):
                    summary = final_state["model_usage_summary"]
                    print(f"\n🎯 Model Usage in This Workflow:")
                    for agent_name, usage in summary.get("agent_model_usage", {}).items():
                        print(f"  {agent_name}:")
                        print(f"    Models: {usage['models_used']}")
                        for model, count in usage.get('model_counts', {}).items():
                            print(f"      {model}: {count} calls")
                
                # Show additional info if available
                if final_state.get("articles"):
                    print(f"\n📚 Used {len(final_state['articles'])} articles")
                if final_state.get("recommendations"):
                    print(f"\n💡 Generated {len(final_state['recommendations'])} recommendations")
                if final_state.get("collaboration_metadata"):
                    print(f"\n🤝 Collaboration occurred with: {final_state['collaboration_metadata'].get('collaborating_agents', [])}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def run_single_query(query: str):
    """Run a single query"""
    try:
        conversation_id = str(uuid.uuid4())
        print(f"\n💭 Processing with dynamic model selection...")
        final_state = orchestrator.run_workflow(query, conversation_id)
        
        print(f"Query: {query}")
        print("━" * 60)
        print(f"Response: {final_state.get('response', 'No response generated')}")
        
        # Show quick stats
        model_manager = orchestrator.chat_agent.model_manager
        stats = model_manager.get_live_performance_stats()
        print(f"\n🧠 Models Used Today: {list(stats['model_usage_today'].keys())}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "api":
            from api import app
            app.run(debug=True, host='0.0.0.0', port=5000)
        else:
            # Run single query
            query = " ".join(sys.argv[1:])
            run_single_query(query)
    else:
        # Run interactive mode
        interactive_workflow()
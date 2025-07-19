import sys
import os
import logging
import redis
import json
from datetime import datetime

# Add the JURIX root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from orchestrator.core.orchestrator import run_workflow # type: ignore
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
import uuid

def setup_logging():
    """Setup enhanced logging for Redis integration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('workflow_debug.log'),
            logging.StreamHandler()
        ]
    )

def test_redis_connection():
    """Test Redis connection before starting the workflow"""
    try:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()
        print("✅ Redis connection successful!")
        return True
    except redis.ConnectionError:
        print("❌ Redis connection failed!")
        print("Please make sure Redis is running:")
        print("  Docker: docker start redis-stack")
        print("  Native: redis-server")
        return False
    except Exception as e:
        print(f"❌ Redis error: {e}")
        return False

def display_collaboration_details(final_state: dict):
    """Display detailed collaboration information in a beautiful format"""
    print("\n" + "🤝" + " COLLABORATION ANALYSIS " + "🤝".center(50, "="))
    
    # Get collaboration metadata
    collab_meta = final_state.get("collaboration_metadata", {})
    final_collab = final_state.get("final_collaboration_summary", {})
    
    # Use the most complete metadata available
    collaboration_data = collab_meta if collab_meta else final_collab
    
    if not collaboration_data:
        print("   ℹ️  No collaboration occurred for this query")
        print("   📋 This was handled by a single agent independently")
        return
    
    # Display collaboration overview
    primary_agent = collaboration_data.get("primary_agent", "unknown")
    collaborating_agents = collaboration_data.get("collaborating_agents", [])
    collaboration_types = collaboration_data.get("collaboration_types", [])
    quality = collaboration_data.get("collaboration_quality", 0)
    
    print(f"   🎯 PRIMARY AGENT: {primary_agent}")
    
    if collaborating_agents:
        print(f"   🤖 COLLABORATING AGENTS: {len(collaborating_agents)} agents helped")
        for i, agent in enumerate(collaborating_agents, 1):
            print(f"      {i}. {agent}")
    else:
        print("   🤖 COLLABORATING AGENTS: None (single agent execution)")
    
    if collaboration_types:
        print(f"   🔧 COLLABORATION TYPES:")
        for i, collab_type in enumerate(collaboration_types, 1):
            type_description = get_collaboration_type_description(collab_type)
            print(f"      {i}. {collab_type}: {type_description}")
    
    # Display collaboration quality with visual indicator
    quality_percentage = quality * 100
    quality_bar = "█" * int(quality_percentage / 10) + "░" * (10 - int(quality_percentage / 10))
    quality_emoji = "🟢" if quality > 0.8 else "🟡" if quality > 0.5 else "🔴"
    
    print(f"   📊 COLLABORATION QUALITY: {quality_emoji} {quality_percentage:.1f}%")
    print(f"      [{quality_bar}] {quality:.3f}")
    
    # Display collaboration statistics
    successful = collaboration_data.get("successful_collaborations", 0)
    total_attempts = collaboration_data.get("total_collaboration_attempts", 0)
    
    if total_attempts > 0:
        success_rate = (successful / total_attempts) * 100
        print(f"   📈 SUCCESS RATE: {successful}/{total_attempts} ({success_rate:.1f}%)")
    
    # Display timing information
    start_time = collaboration_data.get("start_time")
    end_time = collaboration_data.get("end_time")
    
    if start_time and end_time:
        try:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            duration = (end_dt - start_dt).total_seconds()
            print(f"   ⏱️  COLLABORATION DURATION: {duration:.3f} seconds")
        except:
            pass
    
    # Display collaboration workflow
    print(f"\n   🔄 COLLABORATION WORKFLOW:")
    if collaborating_agents:
        print(f"      1. {primary_agent} assessed the query")
        print(f"      2. Identified need for: {', '.join(collaboration_types)}")
        for i, agent in enumerate(collaborating_agents, 3):
            print(f"      {i}. Collaborated with {agent}")
        print(f"      {len(collaborating_agents) + 3}. Synthesized enhanced response")
    else:
        print(f"      1. {primary_agent} handled the query independently")
        print(f"      2. No collaboration needed")
    
    # Display additional insights
    if collaboration_data.get("workflow_completed"):
        print(f"   ✅ WORKFLOW STATUS: Successfully completed")
    
    final_agent = collaboration_data.get("final_agent")
    if final_agent:
        print(f"   🏁 FINAL RESPONSE BY: {final_agent}")

def get_collaboration_type_description(collab_type: str) -> str:
    """Get human-readable description of collaboration types"""
    descriptions = {
        "context_enrichment": "Added relevant articles and background information",
        "data_analysis": "Provided data insights and metrics analysis", 
        "strategic_reasoning": "Contributed strategic recommendations and planning",
        "validation": "Performed quality review and suggestions",
        "content_generation": "Enhanced response content and formatting"
    }
    return descriptions.get(collab_type, "Provided specialized assistance")

def display_enhanced_results(final_state: dict):
    """Display comprehensive results including collaboration details"""
    
    # Extract and display response
    if "response" in final_state:
        response = final_state["response"]
    else:
        response = "No response generated"
    
    # Display main results
    print(f"\n🎯 QUERY INTENT: {final_state.get('intent', {}).get('intent', 'Unknown')}")
    print(f"🤖 AI RESPONSE:")
    print(f"   {response}")
    
    # Display additional information if available
    articles = final_state.get('articles', [])
    if articles:
        print(f"\n📚 ARTICLES USED: {len(articles)} articles")
        for i, article in enumerate(articles[:3], 1):
            title = article.get('title', 'Untitled')
            print(f"   {i}. {title}")
        if len(articles) > 3:
            print(f"   ... and {len(articles) - 3} more")
            
    recommendations = final_state.get('recommendations', [])
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS: {len(recommendations)} suggestions")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"   {i}. {rec[:80]}{'...' if len(rec) > 80 else ''}")
        if len(recommendations) > 3:
            print(f"   ... and {len(recommendations) - 3} more")
    
    tickets = final_state.get('tickets', [])
    if tickets:
        print(f"\n🎫 JIRA TICKETS: {len(tickets)} tickets analyzed")
        for i, ticket in enumerate(tickets[:3], 1):
            key = ticket.get('key', 'Unknown')
            summary = ticket.get('fields', {}).get('summary', 'No summary')
            print(f"   {i}. {key}: {summary[:60]}{'...' if len(summary) > 60 else ''}")
        if len(tickets) > 3:
            print(f"   ... and {len(tickets) - 3} more")
    
    # Display collaboration details
    display_collaboration_details(final_state)

def interactive_workflow():
    """Enhanced interactive workflow with collaboration visualization"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger("Main")
    
    # Test Redis connection
    if not test_redis_connection():
        return
    
    # Initialize shared memory
    try:
        shared_memory = JurixSharedMemory()
        logger.info("Shared memory initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize shared memory: {e}")
        return
    
    conversation_id = str(uuid.uuid4())
    logger.info(f"Starting interactive workflow with conversation ID: {conversation_id}")
    
    print("\n🤖 JURIX Collaborative AI System")
    print("=" * 60)
    print("Ask questions and see how agents collaborate to help you!")
    print("\nSpecial commands:")
    print("  'stats' - Redis memory statistics")
    print("  'mental' - Agent mental states") 
    print("  'memory' - Semantic memory analysis")
    print("  'workflows' - Workflow intelligence")
    print("  'hybrid' - Collaboration system status")
    print("  'testcollab' - Test collaboration system")
    print("  'debug' - Debug workflow execution")
    print("  'search <query>' - Search semantic memories")
    print("  'clear' - Clear conversation history")
    print("  'quit' - Exit")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n🎯 Your question > ").strip()
            
            if query.lower() == "quit":
                print("👋 Goodbye!")
                break
                
            elif query.lower() == "stats":
                # Show Redis statistics
                stats = shared_memory.get_memory_stats()
                print(f"\n📊 Redis Memory Statistics:")
                print(f"   Used Memory: {stats.get('used_memory_human', 'N/A')}")
                print(f"   Total Keys: {stats.get('total_keys', 'N/A')}")
                print(f"   Connected Clients: {stats.get('connected_clients', 'N/A')}")
                continue
                
            elif query.lower() == "memory":
                # Show semantic memory statistics
                print("\n🧠 Semantic Memory Analysis:")
                try:
                    from orchestrator.memory.vector_memory_manager import VectorMemoryManager # type: ignore
                    
                    vector_memory = VectorMemoryManager(shared_memory.redis_client)
                    insights = vector_memory.get_memory_insights()
                    
                    print(f"   📊 Total Memories: {insights.get('total_memories', 0)}")
                    print(f"   🤖 Memories by Agent:")
                    for agent_id, count in insights.get('memories_by_agent', {}).items():
                        print(f"      {agent_id}: {count} memories")
                    
                    print(f"   📝 Memories by Type:")
                    for mem_type, count in insights.get('memories_by_type', {}).items():
                        print(f"      {mem_type}: {count} memories")
                        
                except Exception as e:
                    print(f"   ❌ Error retrieving memory stats: {e}")
                continue
                
            elif query.lower() == "workflows":
                # Show workflow performance and insights
                print("\n🔄 Workflow Intelligence:")
                try:
                    from orchestrator.memory.persistent_langraph_state import LangGraphRedisManager # type: ignore
                    
                    workflow_manager = LangGraphRedisManager(shared_memory.redis_client)
                    insights = workflow_manager.get_workflow_insights()
                    
                    if insights.get("global_performance"):
                        perf = insights["global_performance"]
                        print(f"   📊 Global Performance:")
                        print(f"      Total Workflows: {perf.get('total_workflows', 0)}")
                        print(f"      Successful: {perf.get('successful_workflows', 0)}")
                        print(f"      Failed: {perf.get('failed_workflows', 0)}")
                        print(f"      Avg Execution Time: {perf.get('avg_execution_time', 0):.2f}s")
                        
                        if perf.get('total_workflows', 0) > 0:
                            success_rate = (perf.get('successful_workflows', 0) / perf.get('total_workflows', 1)) * 100
                            print(f"      Success Rate: {success_rate:.1f}%")
                            
                except Exception as e:
                    print(f"   ❌ Error retrieving workflow stats: {e}")
                continue
                
            elif query.lower() == "mental":
                # Show agent mental states
                print("\n🧠 Agent Mental States:")
                try:
                    redis_client = shared_memory.redis_client
                    mental_state_keys = redis_client.keys("mental_state:*")
                    
                    if not mental_state_keys:
                        print("   No active agent mental states found")
                    else:
                        for key in mental_state_keys:
                            agent_data = redis_client.get(key)
                            if agent_data:
                                try:
                                    state = json.loads(agent_data)
                                    agent_id = state.get("agent_id", "unknown")
                                    beliefs_count = len(state.get("beliefs", {}))
                                    decisions_count = len(state.get("decisions", []))
                                    capabilities = state.get("capabilities", [])
                                    
                                    print(f"   🤖 Agent: {agent_id}")
                                    print(f"      Capabilities: {', '.join(capabilities)}")
                                    print(f"      Beliefs: {beliefs_count}")
                                    print(f"      Decisions: {decisions_count}")
                                    print()
                                    
                                except json.JSONDecodeError:
                                    print(f"   ⚠️ Could not parse state for {key}")
                                    
                except Exception as e:
                    print(f"   ❌ Error retrieving mental states: {e}")
                continue
                
            elif query.lower() == "hybrid":
                # Show hybrid architecture status
                print("\n🔄 Collaborative Architecture Status:")
                try:
                    from orchestrator.core.collaborative_framework import CollaborativeFramework # type: ignore
                    print("   ✅ Collaborative Framework: Available")
                    
                    # Show recent collaborations
                    recent_collabs = shared_memory.redis_client.lrange("collaboration_history", 0, 4)
                    if recent_collabs:
                        print(f"   📊 Recent Collaborations: {len(recent_collabs)}")
                        for i, collab_data in enumerate(recent_collabs, 1):
                            collab = json.loads(collab_data)
                            primary = collab.get("primary_agent", "unknown")
                            collaborators = collab.get("collaborating_agents", [])
                            print(f"      {i}. {primary} + {collaborators}")
                    else:
                        print("   📊 No recent collaborations found")
                        
                except Exception as e:
                    print(f"   ❌ Hybrid Architecture Error: {e}")
                continue
                
            elif query.lower() == "testcollab":
                # Test collaboration with different query types
                test_queries = [
                    "Give me productivity recommendations for project PROJ123",
                    "Find articles about Kubernetes deployment", 
                    "What is Agile methodology?"
                ]
                
                print(f"🧪 Testing collaboration with different query types...")
                
                for i, test_query in enumerate(test_queries, 1):
                    print(f"\n--- Test {i}: {test_query} ---")
                    final_state = run_workflow(test_query)
                    
                    collab_meta = final_state.get("collaboration_metadata", {})
                    collaborating_agents = collab_meta.get("collaborating_agents", [])
                    
                    if collaborating_agents:
                        print(f"✅ Collaboration detected: {collaborating_agents}")
                    else:
                        print(f"ℹ️  Single agent execution")
                continue
                
            elif query.lower() == "debug":
                # Debug workflow step by step
                test_query = "Give me recommendations for PROJ123"
                
                print("🔍 Debug workflow execution:")
                result = run_workflow(test_query)
                
                print("\n🔎 Collaboration metadata found:")
                collab_keys = [k for k in result.keys() if 'collab' in k.lower()]
                for key in collab_keys:
                    print(f"   {key}: {result[key]}")
                continue
                
            elif query.lower().startswith("search "):
                # Search semantic memory
                search_query = query[7:]  # Remove "search " prefix
                print(f"\n🔍 Searching memories for: '{search_query}'")
                try:
                    from orchestrator.memory.vector_memory_manager import VectorMemoryManager # type: ignore
                    
                    vector_memory = VectorMemoryManager(shared_memory.redis_client)
                    results = vector_memory.search_memories(search_query, max_results=5)
                    
                    if results:
                        print(f"   Found {len(results)} relevant memories:")
                        for i, memory in enumerate(results, 1):
                            print(f"   {i}. [{memory.memory_type.value}] {memory.content[:100]}...")
                            print(f"      Agent: {memory.agent_id} | Confidence: {memory.confidence:.2f}")
                            print()
                    else:
                        print("   No relevant memories found")
                        
                except Exception as e:
                    print(f"   ❌ Search error: {e}")
                continue
                
            elif query.lower() == "clear":
                # Clear conversation history
                shared_memory.redis_client.delete(f"conversation:{conversation_id}")
                print("🗑️ Conversation history cleared!")
                conversation_id = str(uuid.uuid4())  # New conversation ID
                continue
                
            if not query:
                continue
            
            # Process the query and show collaboration details
            print(f"\n⚡ Processing: '{query}'")
            print("━" * 60)
            
            logger.info(f"Processing query: {query}")
            final_state = run_workflow(query, conversation_id)
            logger.info(f"Final state: {final_state}")
            
            # Display comprehensive results with collaboration analysis
            display_enhanced_results(final_state)
            
            print("━" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
            
        except Exception as e:
            logger.error(f"Error in interactive workflow: {str(e)}")
            print(f"❌ Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")

def run_single_query(query: str):
    """Run a single query for testing with collaboration details"""
    setup_logging()
    
    if not test_redis_connection():
        return
    
    try:
        conversation_id = str(uuid.uuid4())
        final_state = run_workflow(query, conversation_id)
        
        print(f"Query: {query}")
        print("━" * 60)
        
        # Display comprehensive results
        display_enhanced_results(final_state)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        # Run single query mode
        query = " ".join(sys.argv[1:])
        run_single_query(query)
    else:
        # Run interactive mode
        interactive_workflow()
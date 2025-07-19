#!/usr/bin/env python3

import sys
import os
import redis
import json
from datetime import datetime

# Add the JURIX root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from agents.jira_data_agent import JiraDataAgent
from agents.base_agent import BaseAgent, AgentCapability

def test_redis_connection():
    """Test basic Redis connection"""
    print("🔍 Testing Redis connection...")
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        result = r.ping()
        print(f"✅ Redis ping successful: {result}")
        
        # Test basic operations
        r.set('test_key', 'Hello Redis!')
        value = r.get('test_key')
        print(f"✅ Redis set/get test: {value}")
        
        r.delete('test_key')
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def test_shared_memory():
    """Test enhanced shared memory"""
    print("\n🔍 Testing JurixSharedMemory...")
    try:
        shared_memory = JurixSharedMemory()
        
        # Test conversation storage
        session_id = "test_session_123"
        shared_memory.add_interaction(session_id, "user", "Hello Redis!")
        shared_memory.add_interaction(session_id, "assistant", "Hello! I'm using Redis now.")
        
        conversation = shared_memory.get_conversation(session_id)
        print(f"✅ Conversation test: {len(conversation)} messages stored")
        
        # Test data storage
        test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
        shared_memory.store("test_key", test_data)
        retrieved_data = shared_memory.get("test_key")
        print(f"✅ Data storage test: {retrieved_data}")
        
        # Test tickets storage
        tickets = [{"key": "TEST-1", "summary": "Test ticket"}]
        shared_memory.store_tickets("TEST_PROJECT", tickets)
        retrieved_tickets = shared_memory.get_tickets("TEST_PROJECT")
        print(f"✅ Tickets storage test: {len(retrieved_tickets)} tickets")
        
        # Test memory stats
        stats = shared_memory.get_memory_stats()
        print(f"✅ Memory stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Shared memory test failed: {e}")
        return False

def test_enhanced_agent():
    """Test enhanced base agent with Redis"""
    print("\n🔍 Testing Enhanced BaseAgent...")
    try:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        agent = BaseAgent("test_agent", redis_client)
        
        # Test belief system
        agent.mental_state.add_belief("test_belief", "test_value", 0.8, "test")
        belief_value = agent.mental_state.get_belief("test_belief")
        confidence = agent.mental_state.get_belief_confidence("test_belief")
        print(f"✅ Belief system test: value={belief_value}, confidence={confidence}")
        
        # Test decision making
        test_input = {"test_input": "value"}
        result = agent.process(test_input)
        print(f"✅ Agent processing test: {result}")
        
        # Test mental state summary
        summary = agent.get_mental_state_summary()
        print(f"✅ Mental state summary: {summary['beliefs_count']} beliefs, {summary['decisions_count']} decisions")
        
        return True
    except Exception as e:
        print(f"❌ Enhanced agent test failed: {e}")
        return False

def test_jira_data_agent():
    """Test Redis-enhanced JiraDataAgent"""
    print("\n🔍 Testing Redis-Enhanced JiraDataAgent...")
    try:
        agent = JiraDataAgent()
        
        # Test data retrieval
        input_data = {
            "project_id": "PROJ123",
            "time_range": {
                "start": "2025-05-01T00:00:00Z",
                "end": "2025-05-15T23:59:59Z"
            }
        }
        
        result = agent.run(input_data)
        print(f"✅ JiraDataAgent test: {result['workflow_status']}")
        print(f"   Tickets retrieved: {len(result.get('tickets', []))}")
        print(f"   Cache hit: {result.get('metadata', {}).get('cache_hit', False)}")
        
        # Test performance metrics
        metrics = agent.get_performance_metrics()
        print(f"✅ Performance metrics: {metrics}")
        
        # Test mental state
        mental_summary = agent.get_mental_state_summary()
        print(f"✅ Agent mental state: {mental_summary['beliefs_count']} beliefs")
        
        return True
    except Exception as e:
        print(f"❌ JiraDataAgent test failed: {e}")
        return False

def test_pub_sub():
    """Test Redis pub/sub for agent coordination"""
    print("\n🔍 Testing Redis Pub/Sub...")
    try:
        shared_memory = JurixSharedMemory()
        
        # Test event publishing
        test_event = {
            "event_type": "test_event",
            "timestamp": datetime.now().isoformat(),
            "data": "test_data"
        }
        
        shared_memory.publish_agent_event("test_channel", test_event)
        print("✅ Event published successfully")
        
        # Test subscription (brief test)
        pubsub = shared_memory.subscribe_to_events(["test_channel"])
        if pubsub:
            print("✅ Subscription created successfully")
            pubsub.close()
        
        return True
    except Exception as e:
        print(f"❌ Pub/Sub test failed: {e}")
        return False

def main():
    """Run all Redis integration tests"""
    print("🚀 Starting JURIX Redis Integration Tests\n")
    
    tests = [
        ("Redis Connection", test_redis_connection),
        ("Shared Memory", test_shared_memory),
        ("Enhanced Agent", test_enhanced_agent),
        ("Jira Data Agent", test_jira_data_agent),
        ("Pub/Sub", test_pub_sub)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your JURIX system is Redis-ready!")
        print("\nNext steps:")
        print("1. Run your main workflow: python main.py")
        print("2. Check Redis UI at: http://localhost:8001")
        print("3. Start implementing the advanced enhancements!")
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
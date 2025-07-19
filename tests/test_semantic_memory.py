#!/usr/bin/env python3

import sys
import os
import redis
import json

# Add the JURIX root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def simple_test():
    print("🔧 Simple Semantic Memory Test")
    print("=" * 40)
    
    # Test 1: Basic imports
    print("1. Testing imports...")
    try:
        import sentence_transformers
        print("   ✅ sentence_transformers imported")
    except ImportError as e:
        print(f"   ❌ sentence_transformers missing: {e}")
        print("   Run: pip install sentence-transformers")
        return False
    
    try:
        import numpy
        print("   ✅ numpy imported")
    except ImportError as e:
        print(f"   ❌ numpy missing: {e}")
        print("   Run: pip install numpy")
        return False
    
    # Test 2: Redis connection
    print("\n2. Testing Redis...")
    try:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()
        print("   ✅ Redis connected")
    except Exception as e:
        print(f"   ❌ Redis failed: {e}")
        return False
    
    # Test 3: Check if vector memory file exists
    print("\n3. Checking files...")
    vector_memory_path = "orchestrator/memory/vector_memory_manager.py"
    if os.path.exists(vector_memory_path):
        print(f"   ✅ {vector_memory_path} exists")
    else:
        print(f"   ❌ {vector_memory_path} missing")
        print("   You need to create this file with the VectorMemoryManager code")
        return False
    
    # Test 4: Try importing VectorMemoryManager
    print("\n4. Testing VectorMemoryManager import...")
    try:
        from orchestrator.memory.vector_memory_manager import VectorMemoryManager, MemoryType # type: ignore
        print("   ✅ VectorMemoryManager imported successfully")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        print("   Check the vector_memory_manager.py file for syntax errors")
        return False
    
    # Test 5: Create VectorMemoryManager instance
    print("\n5. Creating VectorMemoryManager...")
    try:
        vm = VectorMemoryManager(redis_client)
        print("   ✅ VectorMemoryManager created")
    except Exception as e:
        print(f"   ❌ Creation failed: {e}")
        return False
    
    # Test 6: Store a simple memory
    print("\n6. Storing test memory...")
    try:
        memory_id = vm.store_memory(
            agent_id="test_agent",
            memory_type=MemoryType.EXPERIENCE,
            content="This is a test memory about PROJ123 sprint planning and velocity optimization",
            metadata={"test": True, "project": "PROJ123"},
            confidence=0.9
        )
        if memory_id:
            print(f"   ✅ Memory stored: {memory_id}")
        else:
            print("   ❌ Memory storage returned empty ID")
            return False
    except Exception as e:
        print(f"   ❌ Storage failed: {e}")
        return False
    
    # Test 7: Search for the memory
    print("\n7. Testing search...")
    try:
        results = vm.search_memories("PROJ123", max_results=5)
        print(f"   ✅ Search completed: {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"      {i}. {result.content[:60]}...")
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
        return False
    
    # Test 8: Check memory insights
    print("\n8. Getting memory insights...")
    try:
        insights = vm.get_memory_insights()
        print(f"   ✅ Total memories: {insights.get('total_memories', 0)}")
        print(f"   ✅ Agent distribution: {insights.get('memories_by_agent', {})}")
    except Exception as e:
        print(f"   ❌ Insights failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Semantic memory is working.")
    return True

def test_agent_integration():
    print("\n" + "=" * 40)
    print("🤖 Testing Agent Integration")
    print("=" * 40)
    
    # Test 1: Import enhanced BaseAgent
    print("1. Testing BaseAgent import...")
    try:
        from agents.base_agent import BaseAgent
        print("   ✅ BaseAgent imported")
    except Exception as e:
        print(f"   ❌ BaseAgent import failed: {e}")
        return False
    
    # Test 2: Create enhanced agent
    print("\n2. Creating enhanced agent...")
    try:
        import redis
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        agent = BaseAgent("test_enhanced_agent", redis_client=redis_client)
        print("   ✅ Enhanced agent created")
    except Exception as e:
        print(f"   ❌ Agent creation failed: {e}")
        return False
    
    # Test 3: Check if agent has vector memory
    print("\n3. Checking agent semantic memory...")
    has_vector_memory = hasattr(agent.mental_state, 'vector_memory')
    vector_memory_available = has_vector_memory and agent.mental_state.vector_memory is not None
    
    print(f"   Has vector_memory attribute: {has_vector_memory}")
    print(f"   Vector memory available: {vector_memory_available}")
    
    if not vector_memory_available:
        print("   ❌ Agent doesn't have working semantic memory")
        return False
    else:
        print("   ✅ Agent has working semantic memory")
    
    # Test 4: Test agent adding experience
    print("\n4. Testing agent experience storage...")
    try:
        agent.mental_state.add_experience(
            experience_description="Test experience about PROJ123 optimization strategies",
            outcome="successful_test",
            confidence=0.9,
            metadata={"project": "PROJ123", "test": "agent_integration"}
        )
        print("   ✅ Agent stored experience successfully")
    except Exception as e:
        print(f"   ❌ Agent experience storage failed: {e}")
        return False
    
    # Test 5: Test agent memory search
    print("\n5. Testing agent memory search...")
    try:
        memories = agent.mental_state.recall_similar_experiences("PROJ123 optimization", max_results=3)
        print(f"   ✅ Agent found {len(memories)} similar experiences")
        for i, memory in enumerate(memories, 1):
            print(f"      {i}. {memory.content[:60]}...")
    except Exception as e:
        print(f"   ❌ Agent memory search failed: {e}")
        return False
    
    print("\n🎉 Agent integration test passed!")
    return True

if __name__ == "__main__":
    success1 = simple_test()
    if success1:
        success2 = test_agent_integration()
        if success1 and success2:
            print("\n🚀 Everything is working! Your semantic memory system is ready.")
            print("\nNext steps:")
            print("1. Run: python main.py")
            print("2. Try: 'advice on PROJ123'")
            print("3. Try: 'search PROJ123'")
        else:
            print("\n❌ Agent integration failed. Check the errors above.")
    else:
        print("\n❌ Basic setup failed. Fix the errors above first.")
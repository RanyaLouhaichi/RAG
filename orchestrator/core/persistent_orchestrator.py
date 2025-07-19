import sys
import os
from typing import Any, Dict
from langgraph.graph import StateGraph, END
import redis
import json
import logging
from datetime import datetime
import uuid

# Import existing components
from orchestrator.graph.state import JurixState # type: ignore
from orchestrator.core.intent_router import classify_intent# type: ignore
from agents.chat_agent import ChatAgent
from retrieval_agent_basic import RetrievalAgent
from agents.recommendation_agent import RecommendationAgent
from agents.jira_data_agent import JiraDataAgent
from orchestrator.memory.shared_memory import JurixSharedMemory# type: ignore
from orchestrator.memory.persistent_langraph_state import (# type: ignore
    LangGraphRedisManager, 
    WorkflowType, 
    PersistentWorkflowState
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class PersistentOrchestrator:
    """Enhanced orchestrator with Redis persistence and workflow intelligence"""
    
    def __init__(self):
        self.logger = logging.getLogger("PersistentOrchestrator")
        
        # Initialize Redis and shared memory
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.shared_memory = JurixSharedMemory(redis_url="redis://localhost:6379")
        
        # Initialize workflow manager
        self.workflow_manager = LangGraphRedisManager(self.redis_client)
        
        # Initialize agents with Redis support
        self.chat_agent = ChatAgent(self.shared_memory)
        self.retrieval_agent = RetrievalAgent(self.shared_memory)
        self.recommendation_agent = RecommendationAgent(self.shared_memory)
        self.jira_data_agent = JiraDataAgent(redis_client=self.redis_client)
        
        self.logger.info("PersistentOrchestrator initialized with Redis support")

    def _log_node_execution(self, state: PersistentWorkflowState, node_name: str, 
                           func, *args, **kwargs):
        """Wrapper to log and persist node execution"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Executing node: {node_name} for workflow {state['workflow_id']}")
            
            # Execute the node
            result = func(*args, **kwargs)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update performance metrics
            state["performance_metrics"]["node_execution_times"][node_name] = execution_time
            state["performance_metrics"]["agent_interactions"] += 1
            
            # Create checkpoint
            self.workflow_manager.checkpoint_workflow(state, node_name, result)
            
            self.logger.info(f"Node {node_name} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log error
            error_entry = {
                "node": node_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time
            }
            state["error_history"].append(error_entry)
            
            # Update workflow status
            state["workflow_status"] = "error"
            
            # Save error state
            self.workflow_manager.checkpoint_workflow(state, f"{node_name}_error", {"error": str(e)})
            
            self.logger.error(f"Node {node_name} failed after {execution_time:.2f}s: {e}")
            raise

    def classify_intent_node(self, state: PersistentWorkflowState) -> PersistentWorkflowState:
        """Enhanced intent classification with semantic context"""
        def _classify():
            # Get semantic context from previous similar queries
            semantic_context = self._get_semantic_context_for_query(state["query"])
            state["semantic_context"]["intent_classification"] = semantic_context
            
            # Perform classification
            intent_result = classify_intent(state["query"], state["conversation_history"])
            
            # Enhance intent with semantic insights
            if semantic_context.get("similar_intents"):
                intent_result["semantic_confidence"] = 0.8
                intent_result["similar_patterns"] = semantic_context["similar_intents"][:3]
            
            updated_state = state.copy()
            updated_state["intent"] = intent_result
            updated_state["project"] = intent_result.get("project")
            
            return updated_state
        
        return self._log_node_execution(state, "classify_intent", _classify)

    def jira_data_agent_node(self, state: PersistentWorkflowState) -> PersistentWorkflowState:
        """Enhanced Jira data retrieval with caching intelligence"""
        def _retrieve_data():
            project = state.get("project")
            if not project:
                self.logger.warning("No project specified for JiraDataAgent")
                return state.copy()
            
            # Check cache efficiency
            cache_hit = self._check_jira_cache_status(project)
            if cache_hit:
                state["performance_metrics"]["cache_hits"] += 1
            
            input_data = {
                "project_id": project,
                "time_range": {
                    "start": "2025-05-01T00:00:00Z",
                    "end": "2025-05-17T23:59:59Z"
                }
            }
            
            result = self.jira_data_agent.run(input_data)
            updated_state = state.copy()
            updated_state["tickets"] = result.get("tickets", [])
            
            # Store retrieval pattern in semantic memory
            self._store_data_retrieval_pattern(state, result)
            
            return updated_state
        
        return self._log_node_execution(state, "jira_data_agent", _retrieve_data)

    def recommendation_agent_node(self, state: PersistentWorkflowState) -> PersistentWorkflowState:
        """Enhanced recommendation generation with pattern learning"""
        def _generate_recommendations():
            # Get recommendations from past similar scenarios
            semantic_recommendations = self._get_semantic_recommendations(state)
            
            input_data = {
                "session_id": state["conversation_id"],
                "user_prompt": state["query"],
                "articles": state["articles"],
                "project": state["project"],
                "tickets": state["tickets"],
                "workflow_type": "persistent_orchestration",
                "intent": state["intent"],
                "semantic_context": semantic_recommendations
            }
            
            result = self.recommendation_agent.run(input_data)
            updated_state = state.copy()
            updated_state["recommendations"] = result.get("recommendations", [])
            updated_state["needs_context"] = result.get("needs_context", False)
            
            # Store recommendation pattern
            self._store_recommendation_pattern(state, result)
            
            return updated_state
        
        return self._log_node_execution(state, "recommendation_agent", _generate_recommendations)

    def retrieval_agent_node(self, state: PersistentWorkflowState) -> PersistentWorkflowState:
        """Enhanced article retrieval with semantic search"""
        def _retrieve_articles():
            input_data = {
                "session_id": state["conversation_id"],
                "user_prompt": state["query"],
                "intent": state["intent"]
            }
            
            result = self.retrieval_agent.run(input_data)
            updated_state = state.copy()
            updated_state["articles"] = result.get("articles", [])
            
            # Update shared memory for other agents
            self.shared_memory.store("articles", updated_state["articles"])
            
            return updated_state
        
        return self._log_node_execution(state, "retrieval_agent", _retrieve_articles)

    def chat_agent_node(self, state: PersistentWorkflowState) -> PersistentWorkflowState:
        """Enhanced chat response with conversation intelligence"""
        def _generate_response():
            # Get conversation patterns from semantic memory
            conversation_context = self._get_conversation_context(state)
            
            input_data = {
                "session_id": state["conversation_id"],
                "user_prompt": state["query"],
                "articles": state["articles"],
                "recommendations": state["recommendations"],
                "tickets": state["tickets"],
                "intent": state["intent"],
                "conversation_context": conversation_context
            }
            
            result = self.chat_agent.run(input_data)
            updated_state = state.copy()
            updated_state.update({
                "response": result.get("response", "No response generated"),
                "conversation_history": self.shared_memory.get_conversation(state["conversation_id"]),
                "articles_used": result.get("articles_used", []),
                "tickets": result.get("tickets", state["tickets"]),
                "workflow_status": result.get("workflow_status", "completed")
            })
            
            # Store conversation pattern
            self._store_conversation_pattern(state, result)
            
            return updated_state
        
        return self._log_node_execution(state, "chat_agent", _generate_response)

    def _get_semantic_context_for_query(self, query: str) -> Dict[str, Any]:
        """Get semantic context for query classification"""
        try:
            from orchestrator.memory.vector_memory_manager import VectorMemoryManager # type: ignore
            vm = VectorMemoryManager(self.redis_client)
            
            # Search for similar queries
            similar_memories = vm.search_memories(
                query=f"query classification {query}",
                max_results=5
            )
            
            return {
                "similar_intents": [m.metadata for m in similar_memories if "intent" in m.metadata],
                "confidence_boost": len(similar_memories) * 0.1
            }
        except Exception as e:
            self.logger.warning(f"Failed to get semantic context: {e}")
            return {}

    def _check_jira_cache_status(self, project: str) -> bool:
        """Check if Jira data is cached for the project"""
        try:
            cache_key = f"tickets:{project}"
            return self.redis_client.exists(cache_key)
        except Exception as e:
            self.logger.warning(f"Failed to check cache status: {e}")
            return False

    def _store_data_retrieval_pattern(self, state: PersistentWorkflowState, result: Dict[str, Any]):
        """Store data retrieval patterns in semantic memory"""
        try:
            if hasattr(self.jira_data_agent.mental_state, 'add_experience'):
                project = state.get("project", "unknown")
                ticket_count = len(result.get("tickets", []))
                cache_hit = result.get("metadata", {}).get("cache_hit", False)
                
                self.jira_data_agent.mental_state.add_experience(
                    experience_description=f"Retrieved {ticket_count} tickets for {project}",
                    outcome=f"cache_{'hit' if cache_hit else 'miss'}_workflow_context",
                    confidence=0.9,
                    metadata={
                        "project": project,
                        "ticket_count": ticket_count,
                        "cache_hit": cache_hit,
                        "workflow_id": state["workflow_id"]
                    }
                )
        except Exception as e:
            self.logger.warning(f"Failed to store data retrieval pattern: {e}")

    def _get_semantic_recommendations(self, state: PersistentWorkflowState) -> Dict[str, Any]:
        """Get semantic context for recommendations"""
        try:
            from orchestrator.memory.vector_memory_manager import VectorMemoryManager # type: ignore
            vm = VectorMemoryManager(self.redis_client)
            
            query = f"recommendations {state.get('project', '')} {state['query']}"
            similar_patterns = vm.search_memories(query, max_results=3)
            
            return {
                "similar_recommendations": [m.content for m in similar_patterns],
                "pattern_confidence": len(similar_patterns) * 0.2
            }
        except Exception as e:
            self.logger.warning(f"Failed to get semantic recommendations: {e}")
            return {}

    def _store_recommendation_pattern(self, state: PersistentWorkflowState, result: Dict[str, Any]):
        """Store recommendation patterns"""
        try:
            if hasattr(self.recommendation_agent.mental_state, 'add_experience'):
                recommendations = result.get("recommendations", [])
                project = state.get("project", "unknown")
                
                self.recommendation_agent.mental_state.add_experience(
                    experience_description=f"Generated {len(recommendations)} recommendations for {project}",
                    outcome="workflow_recommendation_success",
                    confidence=0.8,
                    metadata={
                        "project": project,
                        "recommendation_count": len(recommendations),
                        "query": state["query"][:100],
                        "workflow_id": state["workflow_id"]
                    }
                )
        except Exception as e:
            self.logger.warning(f"Failed to store recommendation pattern: {e}")

    def _get_conversation_context(self, state: PersistentWorkflowState) -> Dict[str, Any]:
        """Get conversation context from semantic memory"""
        try:
            from orchestrator.memory.vector_memory_manager import VectorMemoryManager # type: ignore
            vm = VectorMemoryManager(self.redis_client)
            
            # Search for similar conversations
            query = f"conversation {state['query']}"
            similar_conversations = vm.search_memories(query, max_results=3)
            
            return {
                "similar_conversations": [m.content[:100] for m in similar_conversations],
                "conversation_patterns": [m.metadata for m in similar_conversations]
            }
        except Exception as e:
            self.logger.warning(f"Failed to get conversation context: {e}")
            return {}

    def _store_conversation_pattern(self, state: PersistentWorkflowState, result: Dict[str, Any]):
        """Store conversation patterns"""
        try:
            if hasattr(self.chat_agent.mental_state, 'add_experience'):
                response = result.get("response", "")
                
                self.chat_agent.mental_state.add_experience(
                    experience_description=f"Handled {state['intent']['intent']} query: {state['query'][:100]}",
                    outcome=f"generated_{len(response)}_char_response",
                    confidence=0.9,
                    metadata={
                        "intent": state["intent"]["intent"],
                        "project": state.get("project"),
                        "response_length": len(response),
                        "workflow_id": state["workflow_id"],
                        "articles_used": len(result.get("articles_used", [])),
                        "recommendations_used": len(state.get("recommendations", []))
                    }
                )
        except Exception as e:
            self.logger.warning(f"Failed to store conversation pattern: {e}")

    def build_persistent_workflow(self):
        """Build the enhanced workflow with persistence"""
        workflow = StateGraph(PersistentWorkflowState)
        
        # Add nodes with persistence
        workflow.add_node("classify_intent", self.classify_intent_node)
        workflow.add_node("jira_data_agent", self.jira_data_agent_node)
        workflow.add_node("recommendation_agent", self.recommendation_agent_node)
        workflow.add_node("retrieval_agent", self.retrieval_agent_node)
        workflow.add_node("chat_agent", self.chat_agent_node)

        workflow.set_entry_point("classify_intent")

        def route(state: PersistentWorkflowState) -> str:
            intent = state["intent"]["intent"] if "intent" in state and "intent" in state["intent"] else "generic_question"
            needs_context = state.get("needs_context", False)
            
            self.logger.info(f"Routing workflow {state['workflow_id']} with intent: {intent}")
            
            if needs_context:
                return "chat_agent"
            
            routing = {
                "generic_question": "chat_agent",
                "follow_up": "chat_agent",
                "article_retrieval": "retrieval_agent",
                "recommendation": "jira_data_agent"
            }
            return routing.get(intent, "chat_agent")

        workflow.add_conditional_edges("classify_intent", route)
        workflow.add_edge("jira_data_agent", "recommendation_agent")
        workflow.add_edge("recommendation_agent", "chat_agent")
        workflow.add_edge("retrieval_agent", "chat_agent")
        workflow.add_edge("chat_agent", END)

        return workflow.compile()

    def run_persistent_workflow(self, query: str, conversation_id: str = None) -> PersistentWorkflowState:
        """Run workflow with full persistence and intelligence"""
        conversation_id = conversation_id or str(uuid.uuid4())
        
        # Create initial state
        initial_state = {
            "query": query,
            "intent": {},
            "conversation_id": conversation_id,
            "conversation_history": [],
            "articles": [],
            "recommendations": [],
            "tickets": [],
            "status": "pending",
            "response": "",
            "articles_used": [],
            "workflow_status": "in_progress",
            "next_agent": "",
            "project": None
        }
        
        # Create persistent workflow state
        persistent_state = self.workflow_manager.create_workflow_state(
            WorkflowType.GENERAL_ORCHESTRATION, 
            initial_state
        )
        
        # Build and run workflow
        workflow = self.build_persistent_workflow()
        
        try:
            final_state = None
            for event in workflow.stream(persistent_state):
                for node_name, node_state in event.items():
                    self.logger.info(f"Workflow {persistent_state['workflow_id']} processed node: {node_name}")
                    final_state = node_state
            
            # Complete workflow
            if final_state:
                self.workflow_manager.complete_workflow(final_state, {
                    "response": final_state.get("response", ""),
                    "articles_used": final_state.get("articles_used", []),
                    "recommendations": final_state.get("recommendations", [])
                })
            
            return final_state or persistent_state
            
        except Exception as e:
            self.logger.error(f"Workflow {persistent_state['workflow_id']} failed: {e}")
            
            # Mark as failed
            persistent_state["workflow_status"] = "failed"
            error_entry = {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "node": persistent_state.get("current_node", "unknown")
            }
            persistent_state["error_history"].append(error_entry)
            
            # Save failed state
            self.workflow_manager._save_workflow_state(persistent_state)
            
            return persistent_state

# Global instance for backward compatibility
persistent_orchestrator = PersistentOrchestrator()

def run_workflow(query: str, conversation_id: str = None) -> JurixState:
    """Backward compatible function that uses persistent orchestrator"""
    persistent_state = persistent_orchestrator.run_persistent_workflow(query, conversation_id)
    
    # Convert back to JurixState for compatibility
    return JurixState(
        query=persistent_state["query"],
        intent=persistent_state["intent"],
        conversation_id=persistent_state["conversation_id"],
        conversation_history=persistent_state["conversation_history"],
        articles=persistent_state["articles"],
        recommendations=persistent_state["recommendations"],
        status=persistent_state["status"],
        response=persistent_state["response"],
        articles_used=persistent_state["articles_used"],
        workflow_status=persistent_state["workflow_status"],
        next_agent=persistent_state["next_agent"],
        project=persistent_state["project"],
        tickets=persistent_state["tickets"]
    )
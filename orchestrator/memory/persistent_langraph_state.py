import redis
import json
import uuid
import logging
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime, timedelta
from enum import Enum
import hashlib

class WorkflowType(Enum):
    """Types of workflows in the system"""
    GENERAL_ORCHESTRATION = "general_orchestration"
    PRODUCTIVITY_ANALYSIS = "productivity_analysis"
    JIRA_ARTICLE_GENERATION = "jira_article_generation"
    RECOMMENDATION_GENERATION = "recommendation_generation"

class WorkflowStatus(Enum):
    """Status of workflow execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    RESUMED = "resumed"

class PersistentWorkflowState(TypedDict):
    """Enhanced state that persists across workflow executions"""
    # Original JurixState fields
    query: str
    intent: Dict[str, Any]
    conversation_id: str
    conversation_history: List[Dict[str, str]]
    articles: List[Dict[str, Any]]
    recommendations: List[str]
    tickets: List[Dict[str, Any]]
    status: str
    response: str
    articles_used: List[Dict[str, Any]]
    workflow_status: str
    next_agent: str
    project: Optional[str]
    
    # Enhanced persistence fields
    workflow_id: str
    workflow_type: str
    workflow_version: str
    created_at: str
    last_updated: str
    execution_count: int
    current_node: str
    node_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    semantic_context: Dict[str, Any]
    error_history: List[Dict[str, Any]]
    retry_count: int
    parent_workflow_id: Optional[str]
    child_workflow_ids: List[str]

class LangGraphRedisManager:
    """Manages persistent LangGraph workflows with Redis"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.logger = logging.getLogger("LangGraphRedisManager")
        
        # Configuration
        self.state_ttl = 86400 * 7  # 7 days
        self.checkpoint_interval = 30  # seconds
        self.max_retries = 3
        
        # Initialize Redis structures
        self._initialize_redis_structures()
        
        self.logger.info("LangGraphRedisManager initialized")

    def _initialize_redis_structures(self):
        """Initialize Redis data structures for workflow management"""
        try:
            # Create workflow indices
            self.redis_client.setnx("workflow_counter", 0)
            
            # Initialize performance tracking
            performance_key = "workflow_performance:global"
            if not self.redis_client.exists(performance_key):
                initial_perf = {
                    "total_workflows": 0,
                    "successful_workflows": 0,
                    "failed_workflows": 0,
                    "avg_execution_time": 0.0,
                    "last_updated": datetime.now().isoformat()
                }
                self.redis_client.set(performance_key, json.dumps(initial_perf))
            
            self.logger.info("Redis structures initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis structures: {e}")

    def create_workflow_state(self, workflow_type: WorkflowType, initial_state: Dict[str, Any]) -> PersistentWorkflowState:
        """Create a new persistent workflow state"""
        try:
            # Generate unique workflow ID
            workflow_counter = self.redis_client.incr("workflow_counter")
            workflow_id = f"workflow_{workflow_type.value}_{workflow_counter}_{str(uuid.uuid4())[:8]}"
            
            # Create enhanced state
            persistent_state = PersistentWorkflowState(
                # Copy original state
                **initial_state,
                
                # Add persistence fields
                workflow_id=workflow_id,
                workflow_type=workflow_type.value,
                workflow_version="1.0",
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                execution_count=0,
                current_node="start",
                node_history=[],
                performance_metrics={
                    "start_time": datetime.now().isoformat(),
                    "node_execution_times": {},
                    "memory_usage": {},
                    "cache_hits": 0,
                    "agent_interactions": 0
                },
                semantic_context={},
                error_history=[],
                retry_count=0,
                parent_workflow_id=None,
                child_workflow_ids=[]
            )
            
            # Store in Redis
            self._save_workflow_state(persistent_state)
            
            # Add to active workflows index
            self.redis_client.sadd("active_workflows", workflow_id)
            self.redis_client.sadd(f"workflows_by_type:{workflow_type.value}", workflow_id)
            
            self.logger.info(f"Created workflow state: {workflow_id}")
            return persistent_state
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow state: {e}")
            raise

    def load_workflow_state(self, workflow_id: str) -> Optional[PersistentWorkflowState]:
        """Load workflow state from Redis"""
        try:
            state_key = f"workflow_state:{workflow_id}"
            state_data = self.redis_client.get(state_key)
            
            if state_data:
                return json.loads(state_data)
            else:
                self.logger.warning(f"Workflow state not found: {workflow_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load workflow state {workflow_id}: {e}")
            return None

    def _save_workflow_state(self, state: PersistentWorkflowState):
        """Save workflow state to Redis"""
        try:
            workflow_id = state["workflow_id"]
            state_key = f"workflow_state:{workflow_id}"
            
            # Update timestamp
            state["last_updated"] = datetime.now().isoformat()
            
            # Store state
            self.redis_client.set(state_key, json.dumps(state, default=str), ex=self.state_ttl)
            
            # Update metadata
            metadata_key = f"workflow_metadata:{workflow_id}"
            metadata = {
                "workflow_type": state["workflow_type"],
                "status": state["workflow_status"],
                "current_node": state["current_node"],
                "last_updated": state["last_updated"],
                "execution_count": state["execution_count"]
            }
            self.redis_client.set(metadata_key, json.dumps(metadata), ex=self.state_ttl)
            
        except Exception as e:
            self.logger.error(f"Failed to save workflow state: {e}")

    def checkpoint_workflow(self, state: PersistentWorkflowState, current_node: str, 
                          node_result: Dict[str, Any] = None):
        """Create a checkpoint for the workflow"""
        try:
            # Update current node
            state["current_node"] = current_node
            state["execution_count"] += 1
            
            # Add to node history
            checkpoint_entry = {
                "node": current_node,
                "timestamp": datetime.now().isoformat(),
                "result": node_result or {},
                "execution_count": state["execution_count"]
            }
            state["node_history"].append(checkpoint_entry)
            
            # Limit history size
            if len(state["node_history"]) > 100:
                state["node_history"] = state["node_history"][-100:]
            
            # Save updated state
            self._save_workflow_state(state)
            
            # Store in semantic memory if available
            self._store_checkpoint_semantically(state, current_node, node_result)
            
            self.logger.info(f"Checkpointed workflow {state['workflow_id']} at node {current_node}")
            
        except Exception as e:
            self.logger.error(f"Failed to checkpoint workflow: {e}")

    def _store_checkpoint_semantically(self, state: PersistentWorkflowState, 
                                     current_node: str, node_result: Dict[str, Any]):
        """Store workflow checkpoint in semantic memory"""
        try:
            from orchestrator.memory.vector_memory_manager import VectorMemoryManager, MemoryType # type: ignore
            
            vm = VectorMemoryManager(self.redis_client)
            
            # Create semantic description of the checkpoint
            content = f"Workflow {state['workflow_type']} reached node {current_node}. "
            if node_result:
                if node_result.get("workflow_status") == "success":
                    content += f"Node executed successfully. "
                else:
                    content += f"Node had issues. "
            
            if state.get("query"):
                content += f"Original query: {state['query'][:100]}"
            
            metadata = {
                "workflow_id": state["workflow_id"],
                "workflow_type": state["workflow_type"],
                "node": current_node,
                "status": state["workflow_status"],
                "execution_count": state["execution_count"]
            }
            
            vm.store_memory(
                agent_id=f"workflow_{state['workflow_type']}",
                memory_type=MemoryType.EXPERIENCE,
                content=content,
                metadata=metadata,
                confidence=0.8
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store checkpoint semantically: {e}")

    def resume_workflow(self, workflow_id: str) -> Optional[PersistentWorkflowState]:
        """Resume a paused or failed workflow"""
        try:
            state = self.load_workflow_state(workflow_id)
            if not state:
                return None
            
            # Update status
            state["workflow_status"] = WorkflowStatus.RESUMED.value
            state["retry_count"] += 1
            
            # Check retry limit
            if state["retry_count"] > self.max_retries:
                state["workflow_status"] = WorkflowStatus.FAILED.value
                self.logger.warning(f"Workflow {workflow_id} exceeded retry limit")
                return state
            
            # Add resume entry to history
            resume_entry = {
                "action": "resume",
                "timestamp": datetime.now().isoformat(),
                "retry_count": state["retry_count"]
            }
            state["node_history"].append(resume_entry)
            
            # Save updated state
            self._save_workflow_state(state)
            
            self.logger.info(f"Resumed workflow {workflow_id} from node {state['current_node']}")
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to resume workflow {workflow_id}: {e}")
            return None

    def complete_workflow(self, state: PersistentWorkflowState, final_result: Dict[str, Any]):
        """Mark workflow as completed and store results"""
        try:
            # Update state
            state["workflow_status"] = WorkflowStatus.COMPLETED.value
            state["current_node"] = "END"
            
            # Calculate performance metrics
            start_time = datetime.fromisoformat(state["performance_metrics"]["start_time"])
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            state["performance_metrics"]["end_time"] = end_time.isoformat()
            state["performance_metrics"]["total_execution_time"] = execution_time
            state["performance_metrics"]["nodes_executed"] = len(state["node_history"])
            
            # Add completion entry
            completion_entry = {
                "action": "complete",
                "timestamp": end_time.isoformat(),
                "final_result": final_result,
                "execution_time": execution_time
            }
            state["node_history"].append(completion_entry)
            
            # Save final state
            self._save_workflow_state(state)
            
            # Remove from active workflows
            self.redis_client.srem("active_workflows", state["workflow_id"])
            
            # Update global performance metrics
            self._update_global_performance(state)
            
            # Store completion in semantic memory
            self._store_completion_semantically(state, final_result)
            
            self.logger.info(f"Completed workflow {state['workflow_id']} in {execution_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to complete workflow: {e}")

    def _update_global_performance(self, state: PersistentWorkflowState):
        """Update global workflow performance metrics"""
        try:
            performance_key = "workflow_performance:global"
            current_perf = self.redis_client.get(performance_key)
            
            if current_perf:
                perf_data = json.loads(current_perf)
            else:
                perf_data = {
                    "total_workflows": 0,
                    "successful_workflows": 0,
                    "failed_workflows": 0,
                    "avg_execution_time": 0.0
                }
            
            # Update counters
            perf_data["total_workflows"] += 1
            
            if state["workflow_status"] == WorkflowStatus.COMPLETED.value:
                perf_data["successful_workflows"] += 1
            else:
                perf_data["failed_workflows"] += 1
            
            # Update average execution time
            execution_time = state["performance_metrics"].get("total_execution_time", 0)
            current_avg = perf_data["avg_execution_time"]
            total_workflows = perf_data["total_workflows"]
            
            new_avg = ((current_avg * (total_workflows - 1)) + execution_time) / total_workflows
            perf_data["avg_execution_time"] = new_avg
            perf_data["last_updated"] = datetime.now().isoformat()
            
            # Store updated performance
            self.redis_client.set(performance_key, json.dumps(perf_data))
            
        except Exception as e:
            self.logger.error(f"Failed to update global performance: {e}")

    def _store_completion_semantically(self, state: PersistentWorkflowState, final_result: Dict[str, Any]):
        """Store workflow completion in semantic memory"""
        try:
            from orchestrator.memory.vector_memory_manager import VectorMemoryManager, MemoryType # type: ignore
            
            vm = VectorMemoryManager(self.redis_client)
            
            # Create semantic description of the completion
            execution_time = state["performance_metrics"].get("total_execution_time", 0)
            success = state["workflow_status"] == WorkflowStatus.COMPLETED.value
            
            content = f"Completed {state['workflow_type']} workflow in {execution_time:.1f}s. "
            content += f"Status: {'successful' if success else 'failed'}. "
            
            if state.get("query"):
                content += f"Original query: {state['query'][:100]}. "
            
            if final_result.get("response"):
                content += f"Generated response with {len(final_result['response'])} characters."
            
            metadata = {
                "workflow_id": state["workflow_id"],
                "workflow_type": state["workflow_type"],
                "success": success,
                "execution_time": execution_time,
                "nodes_executed": len(state["node_history"]),
                "query": state.get("query", "")[:100]
            }
            
            vm.store_memory(
                agent_id=f"workflow_system",
                memory_type=MemoryType.EXPERIENCE,
                content=content,
                metadata=metadata,
                confidence=0.9 if success else 0.6
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store completion semantically: {e}")

    def get_workflow_insights(self, workflow_type: Optional[WorkflowType] = None) -> Dict[str, Any]:
        """Get insights about workflow performance and patterns"""
        try:
            insights = {}
            
            # Global performance
            global_perf = self.redis_client.get("workflow_performance:global")
            if global_perf:
                insights["global_performance"] = json.loads(global_perf)
            
            # Active workflows
            active_workflows = list(self.redis_client.smembers("active_workflows"))
            insights["active_workflows_count"] = len(active_workflows)
            
            # Workflows by type
            if workflow_type:
                type_workflows = list(self.redis_client.smembers(f"workflows_by_type:{workflow_type.value}"))
                insights[f"{workflow_type.value}_workflows"] = len(type_workflows)
            else:
                for wf_type in WorkflowType:
                    type_workflows = list(self.redis_client.smembers(f"workflows_by_type:{wf_type.value}"))
                    insights[f"{wf_type.value}_workflows"] = len(type_workflows)
            
            # Recent workflow patterns (from semantic memory)
            try:
                from orchestrator.memory.vector_memory_manager import VectorMemoryManager # type: ignore
                vm = VectorMemoryManager(self.redis_client)
                
                recent_workflows = vm.search_memories(
                    query="workflow completed",
                    agent_id="workflow_system",
                    max_results=20
                )
                
                insights["recent_patterns"] = {
                    "total_recent": len(recent_workflows),
                    "success_rate": sum(1 for m in recent_workflows 
                                      if m.metadata.get("success", False)) / len(recent_workflows) * 100
                                    if recent_workflows else 0
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to get semantic insights: {e}")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow insights: {e}")
            return {}

    def cleanup_old_workflows(self, max_age_days: int = 7):
        """Clean up old completed workflows"""
        try:
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            
            # Get all workflow metadata keys
            metadata_keys = self.redis_client.keys("workflow_metadata:*")
            cleaned_count = 0
            
            for metadata_key in metadata_keys:
                metadata = self.redis_client.get(metadata_key)
                if metadata:
                    metadata_data = json.loads(metadata)
                    last_updated = datetime.fromisoformat(metadata_data["last_updated"])
                    
                    if last_updated < cutoff_time and metadata_data["status"] in ["completed", "failed"]:
                        # Extract workflow ID
                        workflow_id = metadata_key.split(":")[-1]
                        
                        # Delete workflow state and metadata
                        self.redis_client.delete(f"workflow_state:{workflow_id}")
                        self.redis_client.delete(metadata_key)
                        
                        # Remove from indices
                        self.redis_client.srem("active_workflows", workflow_id)
                        
                        cleaned_count += 1
            
            self.logger.info(f"Cleaned up {cleaned_count} old workflows")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old workflows: {e}")
            return 0
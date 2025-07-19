# orchestrator/core/productivity_workflow.py
# ENHANCED VERSION - Now with universal collaboration support!

import sys
import os
from typing import Dict, Optional, List, Any
from langgraph.graph import StateGraph, END # type: ignore
from orchestrator.graph.state import JurixState # type: ignore
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
import functools
import logging
import uuid
from datetime import datetime
from agents.jira_data_agent import JiraDataAgent
from agents.productivity_dashboard_agent import ProductivityDashboardAgent
from agents.recommendation_agent import RecommendationAgent

# NEW: Import the universal collaboration coordinator
from orchestrator.core.universal_collaboration_coordinator import UniversalCollaborationCoordinator # type: ignore

def log_aspect(func):
    @functools.wraps(func)
    def wrapper(state):
        logging.basicConfig(filename='productivity_workflow.log', level=logging.INFO)
        logging.info(f"Executing {func.__name__} with state: {state}")
        result = func(state)
        logging.info(f"Completed {func.__name__} with result: {result}")
        return result
    return wrapper

# Initialize components
shared_memory = JurixSharedMemory()
jira_data_agent = JiraDataAgent(redis_client=shared_memory.redis_client)
productivity_dashboard_agent = ProductivityDashboardAgent(redis_client=shared_memory.redis_client)
recommendation_agent = RecommendationAgent(shared_memory)

# NEW: Create agents registry for collaboration
agents_registry = {
    "jira_data_agent": jira_data_agent,
    "productivity_dashboard_agent": productivity_dashboard_agent,
    "recommendation_agent": recommendation_agent
}

# NEW: Initialize universal collaboration coordinator
collaboration_coordinator = UniversalCollaborationCoordinator(
    shared_memory.redis_client, 
    agents_registry
)

logging.info("🎭 Enhanced Productivity Workflow initialized with Universal Collaboration")

@log_aspect
def jira_data_agent_node(state: JurixState) -> JurixState:
    """ENHANCED: Now with collaboration support"""
    input_data = {
        "project_id": state["project_id"],
        "time_range": state["time_range"],
        "analysis_depth": "enhanced",  # Request enhanced analysis for better collaboration
        "workflow_context": "productivity_analysis"
    }
    
    # NEW: Use collaboration coordinator instead of direct agent call
    result = collaboration_coordinator.coordinate_workflow_step(
        primary_agent="jira_data_agent",
        input_data=input_data,
        workflow_id=state.get("conversation_id", "productivity_workflow"),
        step_name="data_retrieval"
    )
    
    updated_state = state.copy()
    updated_state["tickets"] = result.get("tickets", [])
    updated_state["workflow_status"] = result.get("workflow_status", "failure")
    updated_state["metadata"] = result.get("metadata", {})
    
    # NEW: Preserve collaboration metadata
    if result.get("collaboration_metadata"):
        updated_state["collaboration_metadata"] = result["collaboration_metadata"]
        logging.info(f"🤝 Data node: Collaboration metadata preserved: {result['collaboration_metadata']}")
    
    if updated_state["workflow_status"] == "failure":
        updated_state["error"] = "Failed to retrieve Jira ticket data"
    
    logging.info(f"Updated state in jira_data_agent_node: {updated_state}")
    return updated_state

@log_aspect
def recommendation_agent_node(state: JurixState) -> JurixState:
    """ENHANCED: Now with collaboration support"""
    project_id = state["project_id"]
    ticket_count = len(state["tickets"])
    prompt = f"Analyze productivity data for {project_id} with {ticket_count} tickets and provide recommendations for improving team efficiency"
    
    input_data = {
        "session_id": state["conversation_id"],
        "user_prompt": prompt,
        "project": project_id,
        "tickets": state["tickets"],
        "workflow_type": "productivity_analysis",
        "intent": {"intent": "recommendation", "project": project_id},
        "collaboration_context": "productivity_optimization"  # NEW: Collaboration context
    }
    
    # NEW: Use collaboration coordinator
    result = collaboration_coordinator.coordinate_workflow_step(
        primary_agent="recommendation_agent",
        input_data=input_data,
        workflow_id=state.get("conversation_id", "productivity_workflow"),
        step_name="recommendation_generation"
    )
    
    updated_state = state.copy()
    updated_state["recommendations"] = result.get("recommendations", [])
    updated_state["recommendation_status"] = result.get("workflow_status", "failure")
    
    # NEW: Merge collaboration metadata
    existing_collab = updated_state.get("collaboration_metadata", {})
    new_collab = result.get("collaboration_metadata", {})
    if new_collab:
        merged_collab = merge_collaboration_metadata(existing_collab, new_collab)
        updated_state["collaboration_metadata"] = merged_collab
        logging.info(f"🤝 Recommendation node: Merged collaboration metadata")
    
    # Fallback recommendations if agent fails
    if updated_state["recommendation_status"] == "failure" and updated_state["workflow_status"] == "success":
        updated_state["recommendations"] = [
            "Consider balanced workload distribution among team members",
            "Review tickets in bottleneck stages to identify common blockers",
            "Schedule regular process review meetings to address efficiency issues"
        ]
        logging.warning("Using default recommendations due to recommendation agent failure")
    
    logging.info(f"Updated state in recommendation_agent_node: {updated_state}")
    return updated_state

@log_aspect
def productivity_dashboard_agent_node(state: JurixState) -> JurixState:
    """ENHANCED: Now with collaboration support"""
    input_data = {
        "tickets": state["tickets"],
        "recommendations": state["recommendations"],
        "project_id": state["project_id"],
        "collaboration_context": "comprehensive_analysis",  # NEW: Collaboration context
        "primary_agent_result": {  # NEW: Provide context from previous steps
            "tickets": state["tickets"],
            "recommendations": state["recommendations"],
            "metadata": state.get("metadata", {})
        }
    }
    
    # NEW: Use collaboration coordinator
    result = collaboration_coordinator.coordinate_workflow_step(
        primary_agent="productivity_dashboard_agent",
        input_data=input_data,
        workflow_id=state.get("conversation_id", "productivity_workflow"),
        step_name="dashboard_generation"
    )
    
    updated_state = state.copy()
    updated_state["metrics"] = result.get("metrics", {})
    updated_state["visualization_data"] = result.get("visualization_data", {})
    updated_state["report"] = result.get("report", "")
    updated_state["workflow_status"] = result.get("workflow_status", "failure")
    
    # NEW: Final collaboration metadata merge
    existing_collab = updated_state.get("collaboration_metadata", {})
    new_collab = result.get("collaboration_metadata", {})
    if existing_collab or new_collab:
        final_collab = merge_collaboration_metadata(existing_collab, new_collab)
        final_collab["workflow_completed"] = True
        final_collab["final_agent"] = "productivity_dashboard_agent"
        final_collab["workflow_type"] = "productivity_analysis"
        updated_state["collaboration_metadata"] = final_collab
        updated_state["final_collaboration_summary"] = final_collab  # Backup
        logging.info(f"🎉 Final collaboration metadata: {final_collab}")
    
    # Store dashboard in shared memory
    dashboard_id = f"dashboard_{state['project_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shared_memory.store(dashboard_id, {
        "project_id": state["project_id"],
        "time_range": state["time_range"],
        "metrics": updated_state["metrics"],
        "visualization_data": updated_state["visualization_data"],
        "report": updated_state["report"],
        "recommendations": updated_state["recommendations"],
        "timestamp": datetime.now().isoformat(),
        "collaboration_metadata": updated_state.get("collaboration_metadata", {})
    })
    
    updated_state["dashboard_id"] = dashboard_id
    
    logging.info(f"Updated state in productivity_dashboard_agent_node: {updated_state}")
    return updated_state

def merge_collaboration_metadata(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to merge collaboration metadata"""
    if not existing:
        return new.copy() if new else {}
    if not new:
        return existing.copy()
    
    merged = existing.copy()
    
    # Merge agent lists
    existing_agents = set(merged.get("collaborating_agents", []))
    new_agents = set(new.get("collaborating_agents", []))
    merged["collaborating_agents"] = list(existing_agents | new_agents)
    
    # Merge collaboration types
    existing_types = set(merged.get("collaboration_types", []))
    new_types = set(new.get("collaboration_types", []))
    merged["collaboration_types"] = list(existing_types | new_types)
    
    # Update other metadata
    merged.update(new)
    merged["total_workflow_collaborations"] = len(merged["collaborating_agents"])
    
    return merged

def build_productivity_workflow():
    """Build enhanced workflow with collaboration"""
    workflow = StateGraph(JurixState)
    
    # Use enhanced collaborative nodes
    workflow.add_node("jira_data_agent", jira_data_agent_node)
    workflow.add_node("recommendation_agent", recommendation_agent_node)
    workflow.add_node("productivity_dashboard_agent", productivity_dashboard_agent_node)
    
    workflow.set_entry_point("jira_data_agent")
    
    workflow.add_edge("jira_data_agent", "recommendation_agent")
    workflow.add_edge("recommendation_agent", "productivity_dashboard_agent")
    workflow.add_edge("productivity_dashboard_agent", END)
    
    def handle_error(state: JurixState) -> str:
        if state["workflow_status"] == "failure":
            return END
        return "recommendation_agent"
    
    workflow.add_conditional_edges(
        "jira_data_agent",
        handle_error
    )
    
    return workflow.compile()

def run_productivity_workflow(project_id: str, time_range: Dict[str, str], conversation_id: str = None) -> JurixState:
    """ENHANCED: Run productivity workflow with universal collaboration"""
    conversation_id = conversation_id or str(uuid.uuid4())
    
    # Create enhanced state with collaboration support
    state = JurixState(
        query=f"Generate productivity analysis for {project_id}",
        intent={"intent": "productivity_analysis", "project": project_id},
        conversation_id=conversation_id,
        conversation_history=[],
        articles=[],
        recommendations=[],
        status="pending",
        response="",
        articles_used=[],
        workflow_status="",
        next_agent="",
        project=project_id,
        project_id=project_id,
        time_range=time_range,
        tickets=[],
        metrics={},
        visualization_data={},
        report="",
        metadata={},
        ticket_id="",
        article={},
        redundant=False,
        refinement_suggestion=None,
        approved=False,
        refinement_count=0,
        has_refined=False,
        iteration_count=0,
        workflow_stage="",
        recommendation_id=None,
        workflow_history=[],
        error=None,
        recommendation_status=None,
        dashboard_id=None,
        # NEW: Collaboration fields
        collaboration_metadata=None,
        final_collaboration_summary=None,
        collaboration_insights=None,
        collaboration_trace=None,
        collaborative_agents_used=None
    )
    
    logging.info(f"🚀 Starting ENHANCED productivity workflow with collaboration for {project_id}")
    logging.info(f"Initial state before stream: {state}")
    
    workflow = build_productivity_workflow()
    final_state = state
    
    # Track collaboration throughout workflow
    collaboration_trace = []
    
    for event in workflow.stream(state):
        for node_name, node_state in event.items():
            # Track collaboration metadata through each step
            collab_metadata = node_state.get("collaboration_metadata", {})
            if collab_metadata:
                collaboration_trace.append({
                    "node": node_name,
                    "collaboration": collab_metadata,
                    "timestamp": datetime.now().isoformat()
                })
                logging.info(f"🤝 {node_name} generated collaboration: {collab_metadata}")
            
            final_state = node_state
    
    # Ensure final state has collaboration metadata
    if not final_state.get("collaboration_metadata") and collaboration_trace:
        # Reconstruct from trace
        logging.info("🚨 Reconstructing collaboration metadata from trace")
        final_state = collaboration_coordinator.fix_missing_collaboration(final_state)
    
    # Add collaboration trace for debugging
    final_state["collaboration_trace"] = collaboration_trace
    
    logging.info(f"🎉 Productivity workflow completed with collaboration")
    logging.info(f"Final state after stream: {final_state}")
    
    return final_state

# NEW: Enhanced wrapper function for easy integration
def run_collaborative_productivity_workflow(project_id: str, time_range: Dict[str, str]) -> JurixState:
    """
    Enhanced wrapper that guarantees collaboration
    Use this function for maximum collaboration features
    """
    enhanced_workflow = collaboration_coordinator.integrate_with_existing_workflow(run_productivity_workflow)
    return enhanced_workflow(project_id, time_range)
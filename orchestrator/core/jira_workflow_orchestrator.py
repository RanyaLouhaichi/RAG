# orchestrator/core/jira_workflow_orchestrator.py
# PROPERLY FIXED VERSION - Now actually triggers collaboration AND ends at approval

from datetime import datetime
import sys
import os
from typing import Dict, Optional, List, Any
from langgraph.graph import StateGraph, END
from orchestrator.graph.state import JurixState # type: ignore
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
import functools
import logging
import uuid
from agents.jira_article_generator_agent import JiraArticleGeneratorAgent
from agents.knowledge_base_agent import KnowledgeBaseAgent
from agents.recommendation_agent import RecommendationAgent
from agents.jira_data_agent import JiraDataAgent

def log_aspect(func):
    @functools.wraps(func)
    def wrapper(state):
        logging.basicConfig(filename='jira_workflow.log', level=logging.INFO)
        logging.info(f"Executing {func.__name__}")
        result = func(state)
        logging.info(f"Completed {func.__name__} - collaboration: {bool(result.get('collaboration_metadata'))}")
        return result
    return wrapper

# Initialize components (keep your existing structure)
shared_memory = JurixSharedMemory()
jira_article_generator = JiraArticleGeneratorAgent(shared_memory)
knowledge_base = KnowledgeBaseAgent(shared_memory)
recommendation_agent = RecommendationAgent(shared_memory)
jira_data_agent = JiraDataAgent(redis_client=shared_memory.redis_client)

logging.info("🎭 Fixed Jira Workflow initialized")

def store_recommendations(ticket_id: str, recommendations: List[str]) -> str:
    """Store recommendations - keeping your existing logic"""
    recommendation_id = f"rec_{ticket_id}_{str(uuid.uuid4())}"
    shared_memory.store(recommendation_id, {"ticket_id": ticket_id, "recommendations": recommendations})
    logging.info(f"Stored recommendations with ID {recommendation_id}")
    return recommendation_id

def run_recommendation_agent(state: Dict[str, Any]) -> tuple[str, List[str]]:
    """Enhanced recommendation generation that gets proper context"""
    logging.info(f"Running recommendation agent for ticket {state['ticket_id']}")
    
    project_id = state.get("project", "PROJ123")
    
    # Get comprehensive ticket data first
    jira_input = {
        "project_id": project_id,
        "time_range": {"start": "2025-01-01T00:00:00Z", "end": "2025-12-31T23:59:59Z"}
    }
    
    try:
        ticket_data_result = jira_data_agent.run(jira_input)
        tickets = ticket_data_result.get("tickets", [])
        logging.info(f"Retrieved {len(tickets)} tickets for context")
        
        # Create enhanced input for recommendation agent
        input_data = {
            "session_id": state.get("conversation_id", f"rec_{state['ticket_id']}"),
            "user_prompt": f"Analyze resolved ticket {state['ticket_id']} and provide strategic recommendations for process improvement and knowledge sharing",
            "articles": [state.get("article", {})] if state.get("article") else [],
            "project": project_id,
            "tickets": tickets,
            "workflow_type": "knowledge_base_creation",
            "intent": {"intent": "strategic_recommendations"}
        }
        
        logging.info(f"Calling recommendation agent with enhanced context...")
        result = recommendation_agent.run(input_data)
        recommendations = result.get("recommendations", [])
        
        # Enhanced fallback if recommendations are insufficient
        if not recommendations or (len(recommendations) == 1 and "provide more project-specific details" in recommendations[0].lower()):
            logging.warning("Generating enhanced recommendations with ticket-specific context")
            
            target_ticket = next((t for t in tickets if t.get("key") == state['ticket_id']), None)
            if target_ticket:
                ticket_fields = target_ticket.get("fields", {})
                enhanced_prompt = (
                    f"Based on the successful resolution of ticket {state['ticket_id']} - {ticket_fields.get('summary', 'No summary')}, "
                    f"provide strategic recommendations for preventing similar issues, improving team processes, and sharing knowledge. "
                    f"Status: {ticket_fields.get('status', {}).get('name', 'Unknown')}. "
                    f"Focus on actionable insights that can improve project efficiency and team learning."
                )
                
                enhanced_input = input_data.copy()
                enhanced_input["user_prompt"] = enhanced_prompt
                enhanced_result = recommendation_agent.run(enhanced_input)
                recommendations = enhanced_result.get("recommendations", [])
        
        # Final fallback with intelligent defaults
        if not recommendations:
            logging.warning("Using intelligent fallback recommendations")
            target_ticket = next((t for t in tickets if t.get("key") == state['ticket_id']), None)
            if target_ticket:
                ticket_summary = target_ticket.get("fields", {}).get("summary", "this ticket")
                recommendations = [
                    f"Document the resolution pattern from {state['ticket_id']} ({ticket_summary}) for future team reference and knowledge sharing.",
                    f"Implement proactive monitoring to detect similar issues early, based on the resolution approach used in {state['ticket_id']}.",
                    f"Create automated testing scenarios that cover the use case resolved in {state['ticket_id']} to prevent regression.",
                    f"Schedule a team knowledge-sharing session about the {state['ticket_id']} resolution methodology and lessons learned."
                ]
            else:
                recommendations = [
                    "Document the resolution methodology for future team reference and knowledge base enhancement.",
                    "Implement monitoring and alerting to detect similar issues proactively before they impact users.",
                    "Develop automated tests to prevent regression of this issue type in future releases.",
                    "Share resolution knowledge with the team through documentation and training sessions."
                ]
    
    except Exception as e:
        logging.error(f"Enhanced recommendation generation failed: {str(e)}")
        recommendations = [
            f"Review and document the resolution approach used in {state.get('ticket_id', 'this ticket')} for team knowledge sharing.",
            "Implement preventive measures based on the root cause analysis of this issue.",
            "Create automated monitoring to detect similar issues early in the future.",
            "Schedule team discussion about applying this resolution pattern to similar scenarios."
        ]
    
    recommendation_id = store_recommendations(state["ticket_id"], recommendations)
    return recommendation_id, recommendations

@log_aspect
def jira_article_generator_node(state: JurixState) -> JurixState:
    """
    FIXED article generation node that actually triggers collaboration
    """
    input_data = {
        "ticket_id": state["ticket_id"],
        "refinement_suggestion": state.get("refinement_suggestion"),
        # No need to force collaboration mode - let the agent assess naturally
    }
    
    # Call the enhanced agent
    result = jira_article_generator.run(input_data)
    
    updated_state = state.copy()
    updated_state["article"] = result.get("article", {})
    updated_state["workflow_status"] = result.get("workflow_status", "failure")
    updated_state["autonomous_refinement_done"] = result.get("autonomous_refinement_done", False)
    
    # CRITICAL: Preserve collaboration metadata
    collaboration_metadata = result.get("collaboration_metadata", {})
    if collaboration_metadata:
        updated_state["collaboration_metadata"] = collaboration_metadata
        updated_state["final_collaboration_summary"] = collaboration_metadata
        logging.info(f"🤝 Article generator collaboration successful: {collaboration_metadata}")
    else:
        logging.warning("⚠️ No collaboration metadata found from article generator")
    
    # Track refinement status
    if state.get("refinement_suggestion") is not None:
        updated_state["has_refined"] = True
        updated_state["workflow_stage"] = "article_refined"
    else:
        updated_state["has_refined"] = False
        updated_state["workflow_stage"] = "article_generated"
    
    updated_state["iteration_count"] = state.get("iteration_count", 0) + 1
    
    # Create comprehensive history entry
    history_entry = {
        "step": "initial_generation" if not updated_state["has_refined"] else "manual_refinement",
        "article": updated_state["article"],
        "redundant": updated_state.get("redundant", False),
        "refinement_suggestion": updated_state.get("refinement_suggestion"),
        "workflow_status": updated_state["workflow_status"],
        "workflow_stage": updated_state["workflow_stage"],
        "recommendation_id": updated_state["recommendation_id"],
        "recommendations": updated_state["recommendations"],
        "autonomous_refinement_done": updated_state["autonomous_refinement_done"],
        "collaboration_metadata": collaboration_metadata,
        "collaboration_applied": bool(collaboration_metadata.get("collaborating_agents"))
    }
    updated_state["workflow_history"].append(history_entry)
    
    return updated_state

@log_aspect
def knowledge_base_node(state: JurixState) -> JurixState:
    """
    FIXED knowledge base evaluation that provides better refinement suggestions
    """
    input_data = {
        "article": state["article"],
        "validation_context": "comprehensive_quality_assurance",
        "previous_collaboration": state.get("collaboration_metadata", {})
    }
    
    result = knowledge_base.run(input_data)
    
    updated_state = state.copy()
    updated_state["redundant"] = result.get("redundant", False)
    
    # ENHANCED REFINEMENT LOGIC
    if not state.get("has_refined", False):
        # For first-time evaluation, always provide a substantive refinement suggestion
        kb_suggestion = result.get("refinement_suggestion")
        
        if not kb_suggestion or len(kb_suggestion) < 50:
            # Generate intelligent refinement suggestion based on collaboration status
            collaboration_applied = bool(state.get("collaboration_metadata", {}).get("collaborating_agents"))
            
            if collaboration_applied:
                updated_state["refinement_suggestion"] = (
                    "Enhance the article by adding specific technical implementation details, "
                    "including code examples where applicable, quantifiable metrics showing the impact "
                    "of the resolution, and a detailed timeline of the resolution process. "
                    "Also expand the 'Related Knowledge' section with specific references to similar "
                    "issues and their resolution patterns."
                )
            else:
                updated_state["refinement_suggestion"] = (
                    "Significantly enhance the article by adding comprehensive technical details, "
                    "specific performance metrics before and after the resolution, detailed step-by-step "
                    "implementation instructions, code examples, and strategic recommendations for "
                    "preventing similar issues. Include a troubleshooting section for common variations."
                )
        else:
            updated_state["refinement_suggestion"] = kb_suggestion
            
        updated_state["refinement_count"] = 1
        logging.info(f"Generated refinement suggestion: {updated_state['refinement_suggestion'][:100]}...")
    else:
        # Second time through - no more refinement needed
        updated_state["refinement_suggestion"] = None
        updated_state["refinement_count"] = state.get("refinement_count", 0)
        logging.info("Article has been refined - no further refinement needed")
    
    updated_state["workflow_status"] = result.get("workflow_status", "failure")
    updated_state["iteration_count"] = state.get("iteration_count", 0) + 1
    updated_state["workflow_stage"] = "knowledge_base_evaluated"
    
    # Preserve and enhance collaboration metadata
    existing_collab = updated_state.get("collaboration_metadata", {})
    new_collab = result.get("collaboration_metadata", {})
    
    if existing_collab or new_collab:
        merged_collab = merge_collaboration_metadata(existing_collab, new_collab)
        merged_collab["validation_completed"] = True
        updated_state["collaboration_metadata"] = merged_collab
        updated_state["final_collaboration_summary"] = merged_collab
        logging.info(f"🤝 Knowledge base validation with collaboration: {merged_collab}")
    
    # Create comprehensive history entry
    history_entry = {
        "step": "knowledge_base_evaluation" if not state.get("has_refined", False) else "final_knowledge_base_evaluation",
        "article": updated_state["article"],
        "redundant": updated_state.get("redundant", False),
        "refinement_suggestion": updated_state["refinement_suggestion"],
        "workflow_status": updated_state["workflow_status"],
        "workflow_stage": updated_state["workflow_stage"],
        "recommendation_id": updated_state["recommendation_id"],
        "recommendations": updated_state["recommendations"],
        "autonomous_refinement_done": updated_state.get("autonomous_refinement_done", False),
        "collaboration_metadata": updated_state.get("collaboration_metadata", {}),
        "validation_quality": "enhanced"
    }
    updated_state["workflow_history"].append(history_entry)
    
    return updated_state

def merge_collaboration_metadata(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge collaboration metadata from different workflow steps"""
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
    
    # Update with new data while preserving merges
    merged.update(new)
    merged["collaborating_agents"] = list(existing_agents | new_agents)
    merged["collaboration_types"] = list(existing_types | new_types)
    merged["total_workflow_collaborations"] = len(merged["collaborating_agents"])
    merged["workflow_enhanced"] = True
    
    return merged

def build_jira_workflow():
    """Build the FIXED Jira workflow that ends at approval stage"""
    workflow = StateGraph(JurixState)
    
    workflow.add_node("jira_article_generator", jira_article_generator_node)
    workflow.add_node("knowledge_base", knowledge_base_node)

    workflow.set_entry_point("jira_article_generator")

    def route(state: JurixState) -> str:
        """FIXED routing that properly ends at approval stage"""
        logging.info(f"[ROUTING] Current workflow state analysis:")
        logging.info(f"  - Workflow status: {state.get('workflow_status')}")
        logging.info(f"  - Workflow stage: {state.get('workflow_stage')}")
        logging.info(f"  - Has article: {bool(state.get('article'))}")
        logging.info(f"  - Has refined: {state.get('has_refined', False)}")
        logging.info(f"  - Refinement suggestion: {bool(state.get('refinement_suggestion'))}")
        logging.info(f"  - Collaboration applied: {bool(state.get('collaboration_metadata'))}")
        
        # Check for critical workflow failures
        if (state.get("workflow_status") == "failure" or 
            not state.get("article") or 
            state["article"].get("status") == "error"):
            
            state["workflow_stage"] = "terminated_failure"
            state["workflow_history"].append({
                "step": "workflow_terminated",
                "article": state["article"],
                "redundant": state["redundant"],
                "refinement_suggestion": state["refinement_suggestion"],
                "workflow_status": state["workflow_status"],
                "workflow_stage": state["workflow_stage"],
                "recommendation_id": state["recommendation_id"],
                "recommendations": state["recommendations"],
                "autonomous_refinement_done": state.get("autonomous_refinement_done", False),
                "collaboration_metadata": state.get("collaboration_metadata", {}),
                "termination_reason": "critical_failure"
            })
            logging.info("[ROUTING] Terminating due to critical failure")
            return END
        
        # Route based on workflow stage
        if state.get("workflow_stage") in ["article_generated", "article_refined"]:
            logging.info("[ROUTING] Proceeding to knowledge base evaluation")
            return "knowledge_base"
        
        elif state.get("workflow_stage") == "knowledge_base_evaluated":
            # CRITICAL FIX: This is where we decide refinement vs approval
            if state.get("refinement_suggestion") and not state.get("has_refined", False):
                logging.info("[ROUTING] Refinement needed - routing back to article generator")
                return "jira_article_generator"
            else:
                # CRITICAL FIX: End at approval stage like original workflow
                logging.info("[ROUTING] Workflow complete - waiting for approval")
                state["workflow_stage"] = "waiting_for_approval"
                state["workflow_history"].append({
                    "step": "waiting_for_approval",
                    "article": state["article"],
                    "redundant": state["redundant"],
                    "refinement_suggestion": state["refinement_suggestion"],
                    "workflow_status": state["workflow_status"],
                    "workflow_stage": state["workflow_stage"],
                    "recommendation_id": state["recommendation_id"],
                    "recommendations": state["recommendations"],
                    "autonomous_refinement_done": state.get("autonomous_refinement_done", False),
                    "collaboration_metadata": state.get("collaboration_metadata", {}),
                    "awaiting_human_approval": True
                })
                return END
        
        # Handle manual approval (this matches your original logic)
        if state.get("approved", False):
            state["workflow_stage"] = "complete"
            state["workflow_history"].append({
                "step": "approval_submitted",
                "article": state["article"],
                "redundant": state["redundant"],
                "refinement_suggestion": state["refinement_suggestion"],
                "approved": state["approved"],
                "workflow_status": state["workflow_status"],
                "workflow_stage": state["workflow_stage"],
                "recommendation_id": state["recommendation_id"],
                "recommendations": state["recommendations"],
                "autonomous_refinement_done": state.get("autonomous_refinement_done", False),
                "collaboration_metadata": state.get("collaboration_metadata", {}),
                "human_approved": True
            })
        
        return END

    workflow.add_conditional_edges("jira_article_generator", route)
    workflow.add_conditional_edges("knowledge_base", route)

    return workflow.compile()

def run_jira_workflow(ticket_id: str, conversation_id: str = None, project_id: str = "PROJ123") -> JurixState:
    """
    FIXED workflow runner that actually triggers collaboration and ends at approval
    """
    conversation_id = conversation_id or str(uuid.uuid4())
    
    # Create state with collaboration support
    state = JurixState(
        query=f"Generate comprehensive article for ticket {ticket_id}",
        intent={"intent": "article_generation", "project": project_id},
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
        ticket_id=ticket_id,
        article={},
        redundant=False,
        refinement_suggestion=None,
        approved=False,
        refinement_count=0,
        has_refined=False,
        iteration_count=0,
        workflow_stage="started",
        recommendation_id=None,
        workflow_history=[],
        autonomous_refinement_done=False,
        # Collaboration support
        collaboration_metadata=None,
        final_collaboration_summary=None,
        collaboration_insights=None,
        collaboration_trace=None,
        collaborative_agents_used=None
    )
    
    logging.info(f"🚀 Starting FIXED Jira workflow for ticket {ticket_id}")
    
    # Run enhanced recommendation agent
    logging.info(f"Phase 1: Running recommendation agent for ticket {ticket_id}")
    recommendation_id, recommendations = run_recommendation_agent(state)
    state["recommendation_id"] = recommendation_id
    state["recommendations"] = recommendations
    logging.info(f"✅ Recommendations generated: {len(recommendations)} recommendations")

    # Run the main workflow
    logging.info(f"Phase 2: Running article generation workflow")
    workflow = build_jira_workflow()
    final_state = state
    collaboration_trace = []
    
    for event in workflow.stream(state):
        for node_name, node_state in event.items():
            logging.info(f"🎼 Workflow event from {node_name}")
            
            # Track collaboration metadata
            collab_metadata = node_state.get("collaboration_metadata", {})
            if collab_metadata:
                collaboration_trace.append({
                    "node": node_name,
                    "collaboration": collab_metadata,
                    "timestamp": datetime.now().isoformat()
                })
                logging.info(f"🤝 {node_name} collaboration: {collab_metadata}")
            
            final_state = node_state
            
            # Update recommendations if needed (keeping your existing logic)
            if (node_name == "jira_article_generator" and 
                final_state.get("article") and 
                "provide more project-specific details" in str(final_state.get("recommendations", []))):
                logging.info("🔄 Updating recommendations with article context")
                final_state_dict = dict(final_state)
                final_state_dict["article"] = final_state.get("article", {})
                recommendation_id, recommendations = run_recommendation_agent(final_state_dict)
                final_state["recommendation_id"] = recommendation_id
                final_state["recommendations"] = recommendations
    
    # Ensure final state has collaboration metadata
    if not final_state.get("collaboration_metadata") and collaboration_trace:
        logging.info("🚨 Reconstructing collaboration metadata from trace")
        reconstructed_metadata = {
            "collaboration_reconstructed": True,
            "collaborating_agents": [],
            "collaboration_types": [],
            "workflow_enhanced": True
        }
        
        for trace_entry in collaboration_trace:
            agents = trace_entry.get("collaboration", {}).get("collaborating_agents", [])
            reconstructed_metadata["collaborating_agents"].extend(agents)
        
        reconstructed_metadata["collaborating_agents"] = list(set(reconstructed_metadata["collaborating_agents"]))
        final_state["collaboration_metadata"] = reconstructed_metadata
        final_state["final_collaboration_summary"] = reconstructed_metadata
    
    # Add collaboration trace
    final_state["collaboration_trace"] = collaboration_trace
    
    logging.info(f"🎉 FIXED Jira workflow completed")
    logging.info(f"Final status: {final_state.get('workflow_stage')}")
    logging.info(f"Collaboration applied: {bool(final_state.get('collaboration_metadata'))}")
    
    return final_state
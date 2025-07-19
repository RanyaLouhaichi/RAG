# orchestrator/core/universal_collaboration_coordinator.py
# UNIVERSAL COLLABORATION COORDINATOR - Works with ALL your workflows!

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import redis
from agents.base_agent import BaseAgent
from orchestrator.core.collaborative_framework import CollaborativeFramework, CollaborationNeed, CollaborationRequest # type: ignore

class UniversalCollaborationCoordinator:
    """
    Universal coordinator that can enhance ANY workflow with collaboration
    Works with LangGraph workflows, direct agent calls, and hybrid approaches
    """
    
    def __init__(self, redis_client: redis.Redis, agents_registry: Dict[str, BaseAgent]):
        self.redis_client = redis_client
        self.agents_registry = agents_registry
        self.logger = logging.getLogger("UniversalCollaborationCoordinator")
        
        # Initialize the core collaborative framework
        self.collaborative_framework = CollaborativeFramework(redis_client, agents_registry)
        
        # Track collaboration across all workflows
        self.workflow_collaborations = {}
        
        self.logger.info("🎭 Universal Collaboration Coordinator initialized for ALL workflows")

    def enhance_agent_call(self, agent_name: str, input_data: Dict[str, Any], 
                          workflow_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Universal method to enhance ANY agent call with collaboration
        
        Usage:
            # Instead of: result = agent.run(input_data)
            # Use: result = coordinator.enhance_agent_call("agent_name", input_data, workflow_context)
        """
        workflow_context = workflow_context or {}
        
        try:
            # Get the agent
            agent = self.agents_registry.get(agent_name)
            if not agent:
                self.logger.error(f"Agent {agent_name} not found in registry")
                return {"error": f"Agent {agent_name} not available", "workflow_status": "error"}
            
            self.logger.info(f"🎯 Enhancing {agent_name} call with collaboration intelligence")
            
            # Use asyncio to coordinate with collaborative framework
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                enhanced_result = loop.run_until_complete(
                    self.collaborative_framework.coordinate_agents(agent_name, input_data)
                )
            finally:
                loop.close()
            
            # Add workflow context to collaboration metadata
            if enhanced_result.get("collaboration_metadata"):
                enhanced_result["collaboration_metadata"]["workflow_context"] = workflow_context
            
            self.logger.info(f"✅ Enhanced {agent_name} with collaboration")
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"❌ Collaboration enhancement failed for {agent_name}: {e}")
            # Fallback to direct agent call
            agent = self.agents_registry.get(agent_name)
            if agent:
                return agent.run(input_data)
            return {"error": str(e), "workflow_status": "error"}

    def coordinate_workflow_step(self, primary_agent: str, input_data: Dict[str, Any],
                                workflow_id: str, step_name: str) -> Dict[str, Any]:
        """
        Coordinate a single workflow step with collaboration
        Perfect for integrating into existing LangGraph nodes
        """
        workflow_context = {
            "workflow_id": workflow_id,
            "step_name": step_name,
            "timestamp": datetime.now().isoformat()
        }
        
        # Track this step in workflow collaboration history
        if workflow_id not in self.workflow_collaborations:
            self.workflow_collaborations[workflow_id] = {
                "steps": [],
                "agents_used": set(),
                "total_collaborations": 0
            }
        
        result = self.enhance_agent_call(primary_agent, input_data, workflow_context)
        
        # Update workflow collaboration tracking
        workflow_collab = self.workflow_collaborations[workflow_id]
        workflow_collab["steps"].append({
            "step": step_name,
            "agent": primary_agent,
            "collaboration": bool(result.get("collaboration_metadata")),
            "timestamp": datetime.now().isoformat()
        })
        workflow_collab["agents_used"].add(primary_agent)
        
        if result.get("collaboration_metadata"):
            workflow_collab["total_collaborations"] += 1
            collaborating_agents = result["collaboration_metadata"].get("collaborating_agents", [])
            workflow_collab["agents_used"].update(collaborating_agents)
        
        return result

    def enhance_langgraph_node(self, node_function):
        """
        Decorator to enhance any LangGraph node with collaboration
        
        Usage:
        @coordinator.enhance_langgraph_node
        def my_node(state):
            # Your existing node code
            return updated_state
        """
        def enhanced_node(state):
            # Extract agent name from function name
            agent_name = node_function.__name__.replace("_node", "").replace("_", "_")
            
            # Map common node names to agents
            agent_mapping = {
                "jira_data_agent": "jira_data_agent",
                "recommendation_agent": "recommendation_agent", 
                "retrieval_agent": "retrieval_agent",
                "chat_agent": "chat_agent",
                "productivity_dashboard_agent": "productivity_dashboard_agent",
                "jira_article_generator": "jira_article_generator_agent",
                "knowledge_base": "knowledge_base_agent"
            }
            
            agent_name = agent_mapping.get(agent_name, agent_name)
            
            # Convert state to input_data format
            input_data = dict(state)
            
            # Get workflow context from state
            workflow_context = {
                "workflow_id": state.get("workflow_id", state.get("conversation_id", "unknown")),
                "step_name": node_function.__name__,
                "state_keys": list(state.keys())
            }
            
            # Enhance with collaboration
            result = self.enhance_agent_call(agent_name, input_data, workflow_context)
            
            # Merge result back into state
            updated_state = state.copy()
            updated_state.update(result)
            
            return updated_state
        
        return enhanced_node

    def fix_missing_collaboration(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix workflows that lost collaboration metadata during execution
        This is a rescue method for your existing workflows
        """
        if workflow_result.get("collaboration_metadata"):
            # Already has collaboration metadata
            return workflow_result
        
        self.logger.warning("🚨 Detecting missing collaboration metadata - attempting reconstruction")
        
        # Try to reconstruct from other evidence
        reconstructed_collab = {
            "collaboration_reconstructed": True,
            "reconstruction_timestamp": datetime.now().isoformat(),
            "collaborating_agents": [],
            "collaboration_types": [],
            "reconstruction_evidence": []
        }
        
        # Look for evidence of collaboration in the result
        if workflow_result.get("articles") and len(workflow_result["articles"]) > 0:
            # Evidence of retrieval agent collaboration
            reconstructed_collab["collaborating_agents"].append("retrieval_agent")
            reconstructed_collab["collaboration_types"].append("context_enrichment")
            reconstructed_collab["reconstruction_evidence"].append("articles_present")
        
        if workflow_result.get("recommendations") and len(workflow_result["recommendations"]) > 0:
            # Evidence of recommendation agent collaboration
            reconstructed_collab["collaborating_agents"].append("recommendation_agent")
            reconstructed_collab["collaboration_types"].append("strategic_reasoning")
            reconstructed_collab["reconstruction_evidence"].append("recommendations_present")
        
        if workflow_result.get("tickets") and len(workflow_result["tickets"]) > 0:
            # Evidence of data agent collaboration
            reconstructed_collab["collaborating_agents"].append("jira_data_agent")
            reconstructed_collab["collaboration_types"].append("data_analysis")
            reconstructed_collab["reconstruction_evidence"].append("tickets_present")
        
        if reconstructed_collab["collaborating_agents"]:
            workflow_result["collaboration_metadata"] = reconstructed_collab
            self.logger.info(f"🔧 Reconstructed collaboration metadata: {len(reconstructed_collab['collaborating_agents'])} agents")
        
        return workflow_result

    def create_collaborative_node_wrapper(self, agent_name: str, 
                                        input_transformer=None, 
                                        output_transformer=None):
        """
        Create a wrapper for any agent that adds collaboration to LangGraph nodes
        
        Usage:
        collaborative_jira_node = coordinator.create_collaborative_node_wrapper(
            "jira_data_agent",
            input_transformer=lambda state: {"project_id": state["project"], "time_range": state["time_range"]},
            output_transformer=lambda result, state: {"tickets": result.get("tickets", [])}
        )
        """
        def collaborative_node(state):
            # Transform state to agent input
            if input_transformer:
                agent_input = input_transformer(state)
            else:
                agent_input = dict(state)
            
            # Add workflow context
            workflow_context = {
                "workflow_id": state.get("workflow_id", state.get("conversation_id", "unknown")),
                "node_name": f"{agent_name}_collaborative_node",
                "original_state_keys": list(state.keys())
            }
            
            # Enhance with collaboration
            result = self.enhance_agent_call(agent_name, agent_input, workflow_context)
            
            # Transform result back to state format
            if output_transformer:
                state_updates = output_transformer(result, state)
            else:
                state_updates = result
            
            # Create updated state
            updated_state = state.copy()
            updated_state.update(state_updates)
            
            # Ensure collaboration metadata is preserved
            if result.get("collaboration_metadata"):
                updated_state["collaboration_metadata"] = result["collaboration_metadata"]
            
            return updated_state
        
        return collaborative_node

    def integrate_with_existing_workflow(self, workflow_function):
        """
        Wrapper to add collaboration to any existing workflow function
        
        Usage:
        enhanced_workflow = coordinator.integrate_with_existing_workflow(run_productivity_workflow)
        result = enhanced_workflow(project_id, time_range)
        """
        def enhanced_workflow(*args, **kwargs):
            # Run the original workflow
            result = workflow_function(*args, **kwargs)
            
            # Fix any missing collaboration metadata
            enhanced_result = self.fix_missing_collaboration(result)
            
            # Add universal collaboration summary
            if not enhanced_result.get("final_collaboration_summary"):
                enhanced_result["final_collaboration_summary"] = self._generate_workflow_summary(enhanced_result)
            
            return enhanced_result
        
        return enhanced_workflow

    def _generate_workflow_summary(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a collaboration summary for any workflow result"""
        summary = {
            "workflow_analyzed": True,
            "analysis_timestamp": datetime.now().isoformat(),
            "collaboration_detected": False,
            "agents_involved": [],
            "data_flow_analysis": {}
        }
        
        # Analyze the result to detect collaboration patterns
        if workflow_result.get("articles"):
            summary["agents_involved"].append("retrieval_agent")
            summary["data_flow_analysis"]["articles"] = len(workflow_result["articles"])
        
        if workflow_result.get("recommendations"):
            summary["agents_involved"].append("recommendation_agent")
            summary["data_flow_analysis"]["recommendations"] = len(workflow_result["recommendations"])
        
        if workflow_result.get("tickets"):
            summary["agents_involved"].append("jira_data_agent") 
            summary["data_flow_analysis"]["tickets"] = len(workflow_result["tickets"])
        
        if workflow_result.get("response"):
            summary["agents_involved"].append("chat_agent")
            summary["data_flow_analysis"]["response_length"] = len(workflow_result["response"])
        
        if workflow_result.get("metrics"):
            summary["agents_involved"].append("productivity_dashboard_agent")
            summary["data_flow_analysis"]["metrics"] = bool(workflow_result["metrics"])
        
        if len(summary["agents_involved"]) > 1:
            summary["collaboration_detected"] = True
            summary["collaboration_quality"] = min(len(summary["agents_involved"]) / 3.0, 1.0)
        
        return summary

    def get_workflow_collaboration_stats(self, workflow_id: str = None) -> Dict[str, Any]:
        """Get collaboration statistics for workflows"""
        if workflow_id and workflow_id in self.workflow_collaborations:
            return self.workflow_collaborations[workflow_id]
        
        # Return global stats
        total_workflows = len(self.workflow_collaborations)
        total_steps = sum(len(wf["steps"]) for wf in self.workflow_collaborations.values())
        total_collaborations = sum(wf["total_collaborations"] for wf in self.workflow_collaborations.values())
        
        return {
            "total_workflows_tracked": total_workflows,
            "total_steps_executed": total_steps,
            "total_collaborations": total_collaborations,
            "collaboration_rate": total_collaborations / total_steps if total_steps > 0 else 0,
            "recent_workflows": list(self.workflow_collaborations.keys())[-5:]
        }
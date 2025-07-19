from flask import Flask, request, jsonify
from orchestrator.core.jira_workflow_orchestrator import run_jira_workflow, shared_memory # type: ignore
from agents.jira_data_agent import JiraDataAgent

app = Flask(__name__)
jira_data_agent = JiraDataAgent()  

@app.route("/jira-workflow", methods=["POST"])
def trigger_jira_workflow():
    data = request.json
    ticket_id = data.get("ticket_id", "TICKET-001")
    project_id = data.get("project_id", "PROJ123")  
    
    if not ticket_id:
        return jsonify({"error": "ticket_id is required"}), 400
    
    final_state = run_jira_workflow(ticket_id, project_id=project_id)
    
    
    shared_memory.store(f"workflow_state_{ticket_id}", final_state)
    
    return jsonify({
        "ticket_id": ticket_id,
        "workflow_history": final_state.get("workflow_history", []),
        "current_state": {
            "article": final_state.get("article", {}),
            "redundant": final_state.get("redundant", False),
            "refinement_suggestion": final_state.get("refinement_suggestion"),
            "workflow_status": final_state.get("workflow_status", "unknown"),
            "workflow_stage": final_state.get("workflow_stage", "unknown"),
            "recommendation_id": final_state.get("recommendation_id"),
            "recommendations": final_state.get("recommendations", []),
            "autonomous_refinement_done": final_state.get("autonomous_refinement_done", False),
            "has_refined": final_state.get("has_refined", False)
        }
    })

@app.route("/jira-workflow/approve", methods=["POST"])
def approve_article():
    data = request.json
    ticket_id = data.get("ticket_id")
    if not ticket_id:
        return jsonify({"error": "ticket_id is required"}), 400

    
    current_state = shared_memory.get(f"workflow_state_{ticket_id}")
    if not current_state:
        return jsonify({"error": f"No workflow state found for ticket {ticket_id}"}), 404

   
    current_state["approved"] = True
    current_state["workflow_stage"] = "complete"
    current_state["workflow_history"].append({
        "step": "approval_submitted",
        "article": current_state["article"],
        "redundant": current_state.get("redundant", False),
        "refinement_suggestion": current_state.get("refinement_suggestion"),
        "approved": True,
        "workflow_status": "success",
        "workflow_stage": "complete",
        "recommendation_id": current_state.get("recommendation_id"),
        "recommendations": current_state.get("recommendations", []),
        "autonomous_refinement_done": current_state.get("autonomous_refinement_done", False)
    })

    shared_memory.store(f"workflow_state_{ticket_id}", current_state)

    return jsonify({
        "ticket_id": ticket_id,
        "workflow_history": current_state.get("workflow_history", []),
        "current_state": {
            "article": current_state.get("article", {}),
            "redundant": current_state.get("redundant", False),
            "refinement_suggestion": current_state.get("refinement_suggestion"),
            "workflow_status": "success",
            "workflow_stage": "complete",
            "recommendation_id": current_state.get("recommendation_id"),
            "recommendations": current_state.get("recommendations", []),
            "autonomous_refinement_done": current_state.get("autonomous_refinement_done", False),
            "has_refined": current_state.get("has_refined", False),
            "approved": True
        }
    })

@app.route("/recommendations/<ticket_id>", methods=["GET"])
def get_recommendations(ticket_id):
    """Endpoint to get recommendations for a specific ticket."""
    recommendation_id = request.args.get("recommendation_id")
    
    if recommendation_id:
        recommendations_data = shared_memory.get(recommendation_id)
        if recommendations_data:
            return jsonify(recommendations_data)
 
    current_state = shared_memory.get(f"workflow_state_{ticket_id}")
    if not current_state:
        return jsonify({"error": f"No workflow state found for ticket {ticket_id}"}), 404
    
    project_id = current_state.get("project", "PROJ123")
    
    from orchestrator.core.jira_workflow_orchestrator import run_recommendation_agent # type: ignore
    recommendation_id, recommendations = run_recommendation_agent({
        "ticket_id": ticket_id,
        "project": project_id,
        "conversation_id": f"rec_{ticket_id}",
        "articles": current_state.get("article", {})
    })
    
    return jsonify({
        "ticket_id": ticket_id,
        "recommendation_id": recommendation_id,
        "recommendations": recommendations
    })

if __name__ == "__main__":
    app.run(debug=True)
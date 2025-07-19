import json
from typing import Any, Dict
from flask import Flask, request, jsonify
from orchestrator.core.orchestrator import orchestrator # type: ignore
import logging
from datetime import datetime
from aria_api import register_aria_routes # type: ignore
from flask_socketio import SocketIO # type: ignore
# Add these imports at the top
from logging.handlers import RotatingFileHandler
import os

# Configure webhook-specific logging with UTF-8 encoding
if not os.path.exists('logs'):
    os.makedirs('logs')

webhook_logger = logging.getLogger('webhook')
webhook_logger.setLevel(logging.DEBUG)

# Create file handler with UTF-8 encoding
webhook_handler = RotatingFileHandler(
    'webhook_activity.log', 
    maxBytes=10485760, 
    backupCount=5,
    encoding='utf-8'  # Specify UTF-8 encoding
)
webhook_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
webhook_handler.setFormatter(formatter)

# Add handler to logger
webhook_logger.addHandler(webhook_handler)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
webhook_logger.addHandler(console_handler)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
logging.basicConfig(level=logging.INFO)

@app.route("/ask-orchestrator", methods=["POST"])
def ask_orchestrator():
    """General orchestration endpoint"""
    try:
        data = request.json
        query = data.get("query", "")
        conversation_id = data.get("conversation_id")
        
        result = orchestrator.run_workflow(query, conversation_id)
        
        return jsonify({
            "query": query,
            "response": result.get("response", ""),
            "articles": result.get("articles", []),
            "recommendations": result.get("recommendations", []),
            "predictions": result.get("predictions", {}),
            "collaboration_metadata": result.get("collaboration_metadata", {}),
            "workflow_status": result.get("workflow_status", "completed")
        })
    
    except Exception as e:
        logging.error(f"Error in ask-orchestrator: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/dashboard", methods=["POST"])
def dashboard():
    """Productivity dashboard endpoint"""
    try:
        data = request.get_json()
        project_id = data.get("project_id")  # e.g., "ABDERA" or "ACE"
        
        if not project_id:
            # Get first available project
            if orchestrator.jira_data_agent.use_real_api and orchestrator.jira_data_agent.available_projects:
                project_id = orchestrator.jira_data_agent.available_projects[0]
            else:
                return jsonify({"error": "No project specified and no projects available"}), 400
        
        # No time range - get all tickets
        state = orchestrator.run_productivity_workflow(project_id, {})
        
        response = {
            "project_id": state["project_id"],
            "tickets": state["tickets"],
            "metrics": state["metrics"],
            "visualization_data": state["visualization_data"],
            "recommendations": state["recommendations"],
            "report": state["report"],
            "predictions": state.get("predictions", {}),
            "workflow_status": state["workflow_status"],
            "dashboard_id": state.get("dashboard_id"),
            "collaboration_metadata": state.get("collaboration_metadata", {})
        }
        
        return jsonify(response), 200 if state["workflow_status"] == "success" else 500
    
    except Exception as e:
        logging.error(f"Error in dashboard endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predictions", methods=["POST"])
def predictions():
    """Predictive analysis endpoint"""
    try:
        data = request.get_json()
        project_id = data.get("project_id", "PROJ123")
        analysis_type = data.get("analysis_type", "comprehensive")
        
        state = orchestrator.run_predictive_workflow(project_id, analysis_type)
        
        response = {
            "project_id": project_id,
            "analysis_type": analysis_type,
            "predictions": state.get("predictions", {}),
            "recommendations": state.get("recommendations", []),
            "workflow_status": state.get("workflow_status", "failure"),
            "collaboration_metadata": state.get("collaboration_metadata", {})
        }
        
        return jsonify(response), 200 if state["workflow_status"] == "success" else 500
    
    except Exception as e:
        logging.error(f"Error in predictions endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/jira-workflow", methods=["POST"])
def trigger_jira_workflow():
    """Trigger jira article generation workflow"""
    data = request.json
    ticket_id = data.get("ticket_id")  # e.g., "ABDERA-1"
    
    if not ticket_id:
        return jsonify({"error": "ticket_id is required"}), 400
    
    # Extract project from ticket key (ABDERA-1 -> ABDERA)
    project_id = ticket_id.split('-')[0] if '-' in ticket_id else None
    
    if not project_id:
        return jsonify({"error": "Invalid ticket format. Expected format: PROJECT-NUMBER"}), 400
    
    final_state = orchestrator.run_jira_workflow(ticket_id, project_id=project_id)
    
    orchestrator.shared_memory.store(f"workflow_state_{ticket_id}", final_state)
    
    return jsonify({
        "ticket_id": ticket_id,
        "project_id": project_id,
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
    """Approve generated article"""
    data = request.json
    ticket_id = data.get("ticket_id")
    if not ticket_id:
        return jsonify({"error": "ticket_id is required"}), 400

    current_state = orchestrator.shared_memory.get(f"workflow_state_{ticket_id}")
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

    orchestrator.shared_memory.store(f"workflow_state_{ticket_id}", current_state)

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


@app.route("/webhook/jira-ticket-resolved", methods=["POST"])
def jira_ticket_resolved_webhook():
    """Webhook endpoint for Jira ticket resolution events"""
    webhook_logger.info("=" * 50)
    webhook_logger.info("WEBHOOK RECEIVED FROM JIRA")
    webhook_logger.info("=" * 50)
    
    try:
        # Log headers for debugging
        webhook_logger.info(f"Headers: {dict(request.headers)}")
        
        # Get the webhook payload
        payload = request.json
        webhook_logger.info(f"Full Payload: {json.dumps(payload, indent=2)}")
        
        # Extract webhook event type
        webhook_event = payload.get('webhookEvent', '')
        issue_event_type = payload.get('issue_event_type_name', '')
        
        webhook_logger.info(f"Webhook Event: {webhook_event}")
        webhook_logger.info(f"Issue Event Type: {issue_event_type}")
        
        # Extract issue information
        issue = payload.get('issue', {})
        issue_key = issue.get('key')
        issue_fields = issue.get('fields', {})
        
        webhook_logger.info(f"Issue Key: {issue_key}")
        webhook_logger.info(f"Issue Status: {issue_fields.get('status', {}).get('name')}")
        
        # Check current status
        current_status = issue_fields.get('status', {}).get('name', '')
        
        # For real Jira, check if status is Done/Resolved/Closed
        is_resolved = current_status.lower() in ['done', 'resolved', 'closed', 'complete']
        
        # Also check changelog for transition to Done
        if not is_resolved and 'changelog' in payload:
            changelog = payload.get('changelog', {})
            items = changelog.get('items', [])
            
            for item in items:
                field = item.get('field', '')
                to_string = item.get('toString', '')
                from_string = item.get('fromString', '')
                
                webhook_logger.info(f"Change: {field} from '{from_string}' to '{to_string}'")
                
                if field == 'status' and to_string.lower() in ['done', 'resolved', 'closed', 'complete']:
                    is_resolved = True
                    webhook_logger.info("✅ This is a resolution event (from changelog)!")
                    break
        
        if current_status.lower() in ['done', 'resolved', 'closed', 'complete']:
            is_resolved = True
            webhook_logger.info("✅ This is a resolution event (from current status)!")
        
        if not is_resolved:
            webhook_logger.info("❌ Not a resolution event - ignoring")
            return jsonify({
                "status": "ignored",
                "reason": f"Not a resolution event. Current status: {current_status}"
            }), 200
        
        # Extract project from ticket key
        project_id = issue_key.split('-')[0] if issue_key and '-' in issue_key else None
        webhook_logger.info(f"Project ID: {project_id}")
        
        if not project_id:
            webhook_logger.error("Invalid ticket format")
            return jsonify({"error": "Invalid ticket format"}), 400
        
        # Log that we're triggering the workflow
        webhook_logger.info(f"🚀 Triggering article generation for {issue_key}")
        
        # Trigger the article generation workflow
        def run_article_workflow():
            webhook_logger.info(f"Starting workflow thread for {issue_key}")
            try:
                # Run the Jira workflow
                final_state = orchestrator.run_jira_workflow(issue_key, project_id=project_id)
                
                webhook_logger.info(f"Workflow completed for {issue_key}")
                webhook_logger.info(f"Article status: {final_state.get('workflow_status')}")
                webhook_logger.info(f"Article title: {final_state.get('article', {}).get('title', 'No title')}")
                
                # Store the result
                result_data = {
                    "ticket_id": issue_key,
                    "project_id": project_id,
                    "article": final_state.get("article", {}),
                    "status": final_state.get("workflow_status"),
                    "timestamp": datetime.now().isoformat(),
                    "triggered_by": "jira_webhook",
                    "issue_summary": issue_fields.get('summary', 'No summary'),
                    "issue_type": issue_fields.get('issuetype', {}).get('name', 'Unknown'),
                    "reporter": issue_fields.get('reporter', {}).get('displayName', 'Unknown'),
                    "assignee": issue_fields.get('assignee', {}).get('displayName', 'Unassigned') if issue_fields.get('assignee') else 'Unassigned'
                }
                
                # Store in shared memory
                orchestrator.shared_memory.store(f"webhook_result_{issue_key}", result_data)
                webhook_logger.info(f"✅ Result stored for {issue_key}")
                
                # Also store in Redis directly for dashboard
                orchestrator.shared_memory.redis_client.set(
                    f"webhook_result_{issue_key}", 
                    json.dumps(result_data),
                    ex=86400  # Expire after 24 hours
                )
                
            except Exception as e:
                webhook_logger.error(f"❌ Workflow failed for {issue_key}: {str(e)}", exc_info=True)
                
                # Store error result
                error_data = {
                    "ticket_id": issue_key,
                    "project_id": project_id,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                orchestrator.shared_memory.store(f"webhook_result_{issue_key}", error_data)
        
        # Run in background thread
        import threading
        thread = threading.Thread(target=run_article_workflow)
        thread.start()
        
        response_data = {
            "status": "accepted",
            "ticket_id": issue_key,
            "message": "Article generation triggered",
            "project": project_id
        }
        
        webhook_logger.info(f"Response: {json.dumps(response_data)}")
        webhook_logger.info("=" * 50)
        
        return jsonify(response_data), 202
        
    except Exception as e:
        webhook_logger.error(f"❌ Webhook error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/webhook/status/<ticket_id>", methods=["GET"])
def check_webhook_status(ticket_id):
    """Check status of webhook-triggered article generation"""
    result = orchestrator.shared_memory.get(f"webhook_result_{ticket_id}")
    
    if not result:
        return jsonify({
            "error": "No webhook result found for this ticket",
            "ticket_id": ticket_id
        }), 404
    
    return jsonify(result), 200

@app.route("/webhook/dashboard", methods=["GET"])
def webhook_dashboard():
    """Simple dashboard to view webhook results"""
    # Get all webhook results
    webhook_keys = [key for key in orchestrator.shared_memory.redis_client.keys() 
                   if key.startswith("webhook_result_")]
    
    results = []
    for key in webhook_keys:
        # Fix: Remove "webhook_result_" prefix when getting from shared_memory
        ticket_id = key.replace("webhook_result_", "")
        result = orchestrator.shared_memory.get(key)
        if result:
            results.append(result)
    
    # Sort by timestamp
    results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Create simple HTML dashboard - FIXED: Escape the CSS braces
    html = """
    <html>
    <head>
        <title>Webhook Article Generator Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .article {{ border: 1px solid #ddd; padding: 10px; margin: 10px 0; }}
            .success {{ background-color: #d4edda; }}
            .failure {{ background-color: #f8d7da; }}
            pre {{ background: #f5f5f5; padding: 10px; overflow: auto; }}
            h1 {{ color: #333; }}
            .timestamp {{ color: #666; font-size: 0.9em; }}
            .status {{ font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>🚀 Article Generation Dashboard</h1>
        <p>Total webhooks processed: <strong>{count}</strong></p>
        <p>Last updated: {timestamp}</p>
        <hr>
        {articles}
    </body>
    </html>
    """
    
    articles_html = ""
    if not results:
        articles_html = "<p>No webhook results found yet. Trigger a webhook to see results here!</p>"
    else:
        for result in results[:20]:  # Show last 20
            status_class = "success" if result.get("status") == "success" else "failure"
            article = result.get("article", {})
            
            # Safe content preview
            content_preview = article.get('content', 'No content generated')[:1000]
            # Escape any remaining braces in the content
            content_preview = content_preview.replace('{', '{{').replace('}', '}}')
            
            articles_html += f"""
            <div class="article {status_class}">
                <h3>📄 {result.get('ticket_id')} - {article.get('title', 'No title')}</h3>
                <p class="status">Status: <span class="{status_class}">{result.get('status', 'unknown')}</span></p>
                <p class="timestamp">Triggered: {result.get('timestamp', 'Unknown time')}</p>
                <p><strong>Project:</strong> {result.get('project_id', 'Unknown')}</p>
                <details>
                    <summary>View Article Content</summary>
                    <pre>{content_preview}...</pre>
                </details>
            </div>
            """
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return html.format(count=len(results), articles=articles_html, timestamp=current_time)

# Add these endpoints to your Flask app

@app.route("/mcp/status", methods=["GET"])
def mcp_status():
    """Get MCP system status"""
    try:
        status = orchestrator.get_mcp_status()
        return jsonify(status), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/mcp/capabilities", methods=["GET"])
async def mcp_capabilities():
    """Get MCP server capabilities"""
    try:
        capabilities = await orchestrator.get_mcp_capabilities()
        return jsonify(capabilities), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/mcp/tool", methods=["POST"])
async def mcp_tool_call():
    """Call an MCP tool directly"""
    try:
        data = request.json
        server_name = data.get("server")
        tool_name = data.get("tool")
        arguments = data.get("arguments", {})
        
        if not server_name or not tool_name:
            return jsonify({"error": "server and tool are required"}), 400
        
        result = await orchestrator.mcp_manager.mcp_client.call_tool(
            server_name, tool_name, arguments
        )
        
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/mcp/resource", methods=["GET"])
async def mcp_resource():
    """Read an MCP resource"""
    try:
        server_name = request.args.get("server")
        resource_uri = request.args.get("uri")
        
        if not server_name or not resource_uri:
            return jsonify({"error": "server and uri are required"}), 400
        
        result = await orchestrator.mcp_manager.mcp_client.read_resource(
            server_name, resource_uri
        )
        
        return jsonify({"resource": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/mcp/prompt", methods=["POST"])
async def mcp_prompt():
    """Get an MCP prompt"""
    try:
        data = request.json
        server_name = data.get("server")
        prompt_name = data.get("prompt")
        arguments = data.get("arguments", {})
        
        if not server_name or not prompt_name:
            return jsonify({"error": "server and prompt are required"}), 400
        
        result = await orchestrator.mcp_manager.mcp_client.get_prompt(
            server_name, prompt_name, arguments
        )
        
        return jsonify({"prompt": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
mcp_initialized = False

def ensure_mcp_initialized():
    """Ensure MCP is initialized (run once)"""
    global mcp_initialized
    if not mcp_initialized:
        try:
            # Run async MCP connection in sync context
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            connected = loop.run_until_complete(orchestrator.connect_mcp())
            if connected:
                app.logger.info("MCP servers connected")
            else:
                app.logger.warning("Running without MCP - using direct agent calls")
            mcp_initialized = True
        except Exception as e:
            app.logger.error(f"MCP initialization error: {e}")
            mcp_initialized = True  # Don't retry

@app.before_request
def before_request():
    """Run before each request"""
    ensure_mcp_initialized()

@app.route("/recommendations/<ticket_id>", methods=["GET"])
def get_recommendations(ticket_id):
    """Get recommendations for a specific ticket"""
    recommendation_id = request.args.get("recommendation_id")
    
    if recommendation_id:
        recommendations_data = orchestrator.shared_memory.get(recommendation_id)
        if recommendations_data:
            return jsonify(recommendations_data)
    
    current_state = orchestrator.shared_memory.get(f"workflow_state_{ticket_id}")
    if not current_state:
        return jsonify({"error": f"No workflow state found for ticket {ticket_id}"}), 404
    
    return jsonify({
        "ticket_id": ticket_id,
        "recommendation_id": current_state.get("recommendation_id"),
        "recommendations": current_state.get("recommendations", [])
    })

register_aria_routes(app, socketio)

if __name__ == "__main__":
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        # Clean up any MCP connections
        if hasattr(orchestrator, 'mcp_manager') and orchestrator.mcp_manager:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(orchestrator.mcp_manager.mcp_client.close_all())
            except:
                pass
            finally:
                loop.close()
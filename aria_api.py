# api_aria.py - Separate file for ARIA endpoints
from flask import request, jsonify
from orchestrator.core.aria_orchestrator import aria, ARIAContext, ARIAPersonality # type: ignore
import asyncio

def register_aria_routes(app, socketio=None):
    """Register all ARIA routes with the Flask app"""
    
    @app.route("/aria/introduce", methods=["GET"])
    def aria_introduce():
        """ARIA introduces herself"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            intro = loop.run_until_complete(aria.introduce_myself({}))
            return jsonify(intro)
        finally:
            loop.close()

    @app.route("/aria/chat", methods=["POST"])
    def aria_chat():
        """Chat with ARIA"""
        data = request.json
        user_input = data.get("message", "")
        workspace = data.get("workspace", "jira")
        project_id = data.get("project_id", "PROJ123")
        
        context = ARIAContext(
            user_intent=user_input,
            workspace=workspace,
            current_project=project_id,
            current_ticket=data.get("ticket_id"),
            user_history=data.get("history", []),
            active_insights=[],
            mood=ARIAPersonality.HELPFUL
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                aria.process_interaction(user_input, context)
            )
            return jsonify(response)
        finally:
            loop.close()

    @app.route("/aria/dashboard/<project_id>", methods=["GET"])
    def aria_dashboard(project_id):
        """Get live dashboard data from ARIA"""
        context = ARIAContext(
            user_intent="show dashboard",
            workspace="jira",
            current_project=project_id,
            current_ticket=None,
            user_history=[],
            active_insights=[],
            mood=ARIAPersonality.ANALYTICAL
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            dashboard = loop.run_until_complete(
                aria._provide_live_dashboard(context)
            )
            return jsonify(dashboard)
        finally:
            loop.close()

    @app.route("/aria/webhook/ticket-resolved", methods=["POST"])
    def aria_ticket_resolved():
        """Webhook for when ticket is resolved in Jira"""
        data = request.json
        ticket_id = data.get("ticket_id")
        
        context = ARIAContext(
            user_intent="ticket resolved",
            workspace="jira",
            current_project=data.get("project_id", "PROJ123"),
            current_ticket=ticket_id,
            user_history=[],
            active_insights=[],
            mood=ARIAPersonality.PROACTIVE
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                aria._handle_ticket_resolution(ticket_id, context)
            )
            return jsonify(result)
        finally:
            loop.close()
    
    # Register SocketIO events if provided
    if socketio:
        @socketio.on('aria_start_monitoring')
        def handle_monitoring(data):
            """Start ARIA monitoring for real-time updates"""
            project_id = data.get('project_id')
            
            def monitor_and_emit():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Start monitoring
                loop.run_until_complete(aria.start_monitoring(project_id))
                
                # Emit updates every 30 seconds
                while True:
                    # Get dashboard update
                    dashboard = loop.run_until_complete(
                        aria._provide_live_dashboard(
                            ARIAContext(
                                user_intent="",
                                workspace="jira",
                                current_project=project_id,
                                current_ticket=None,
                                user_history=[],
                                active_insights=[],
                                mood=ARIAPersonality.ANALYTICAL
                            )
                        )
                    )
                    socketio.emit('dashboard_update', dashboard, room=request.sid)
                    
                    # Wait 30 seconds
                    import time
                    time.sleep(30)
            
            # Run in a background thread
            import threading
            monitor_thread = threading.Thread(target=monitor_and_emit)
            monitor_thread.daemon = True
            monitor_thread.start()
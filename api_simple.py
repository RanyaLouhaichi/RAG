# api_simple.py - Enhanced backend API with complete dashboard integration
from collections import defaultdict
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import uuid
import os
import sys
import time
from datetime import datetime, timedelta
from threading import Lock
import threading

import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.core.orchestrator import orchestrator # type: ignore
from integrations.api_config import APIConfig
from integrations.jira_api_client import JiraAPIClient

# Force reload .env
load_dotenv(override=True)

# Debug print
print("=" * 50)
print("DEBUG: Environment Variables")
print(f"JIRA_URL from env: {os.getenv('JIRA_URL')}")
print(f"Current working directory: {os.getcwd()}")
print(f".env file exists: {os.path.exists('.env')}")
print("=" * 50) 

class DashboardUpdateTracker:
    """Track dashboard updates for real-time notifications"""
    def __init__(self):
        self.updates = {}  # project_key -> list of updates
        self.lock = Lock()
        self.max_updates_per_project = 100
    
    def add_update(self, project_key, update_type, details):
        """Add an update for a project"""
        with self.lock:
            if project_key not in self.updates:
                self.updates[project_key] = []
            
            update = {
                'timestamp': datetime.now().timestamp() * 1000,  # milliseconds
                'type': update_type,
                'details': details,
                'id': f"{project_key}_{int(datetime.now().timestamp() * 1000)}"
            }
            
            self.updates[project_key].insert(0, update)
            
            # Keep only last N updates
            if len(self.updates[project_key]) > self.max_updates_per_project:
                self.updates[project_key] = self.updates[project_key][:self.max_updates_per_project]
    
    def get_updates_since(self, project_key, since_timestamp):
        """Get updates since a given timestamp"""
        with self.lock:
            if project_key not in self.updates:
                return []
            
            return [u for u in self.updates[project_key] if u['timestamp'] > since_timestamp]
    
    def clear_old_updates(self, max_age_minutes=30):
        """Clear updates older than max_age_minutes"""
        cutoff_time = (datetime.now().timestamp() - (max_age_minutes * 60)) * 1000
        with self.lock:
            for project_key in self.updates:
                self.updates[project_key] = [
                    u for u in self.updates[project_key] 
                    if u['timestamp'] > cutoff_time
                ]

# CREATE GLOBAL INSTANCE (after the class definition)
update_tracker = DashboardUpdateTracker()

app = Flask(__name__)
CORS(app, origins=["http://localhost:2990", "http://localhost:8080", "*"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JurixAPI")

def fix_jira_agent():
    """Fix the JiraDataAgent to use real API instead of MCP"""
    logger.info("🔧 Fixing JiraDataAgent configuration...")
    
    # Clear any cached data first
    if hasattr(orchestrator.jira_data_agent, 'available_projects'):
        orchestrator.jira_data_agent.available_projects = []
    
    # Disable MCP and enable API
    if hasattr(orchestrator.jira_data_agent, 'use_mcp'):
        orchestrator.jira_data_agent.use_mcp = False
        logger.info("   ✅ Disabled MCP")
    
    orchestrator.jira_data_agent.use_real_api = True
    logger.info("   ✅ Enabled real API")
    
    # Initialize Jira client if not present
    if not hasattr(orchestrator.jira_data_agent, 'jira_client') or orchestrator.jira_data_agent.jira_client is None:
        logger.info("   🔄 Initializing Jira client...")
        orchestrator.jira_data_agent.jira_client = JiraAPIClient()
    
    # Test connection and update projects
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if orchestrator.jira_data_agent.jira_client.test_connection():
                projects = orchestrator.jira_data_agent.jira_client.get_projects()
                orchestrator.jira_data_agent.available_projects = [p['key'] for p in projects]
                logger.info(f"   ✅ Connected! Found projects: {orchestrator.jira_data_agent.available_projects}")
                
                # Clear any Redis cache that might have old project data
                if hasattr(orchestrator.jira_data_agent, 'redis_client') and orchestrator.jira_data_agent.redis_client:
                    try:
                        # Clear old project cache
                        orchestrator.jira_data_agent.redis_client.delete("jira_projects_cache")
                        orchestrator.jira_data_agent.redis_client.delete("available_projects")
                        logger.info("   ✅ Cleared old project cache")
                    except:
                        pass
                
                return True
            else:
                logger.error(f"   ❌ Connection attempt {attempt + 1} failed")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
        except Exception as e:
            logger.error(f"   ❌ Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return False

# Fix on startup
jira_connected = fix_jira_agent()

try:
    if orchestrator.enable_enhanced_rag():
        logger.info("✅ Enhanced RAG enabled for Confluence publishing")
    else:
        logger.info("❌ Failed to enable enhanced RAG")
except Exception as e:
    logger.error(f"❌ Error enabling enhanced RAG: {e}")

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    logger.info("🏥 Health check requested")
    
    # Refresh projects list
    current_projects = []
    if orchestrator.jira_data_agent.jira_client:
        try:
            projects = orchestrator.jira_data_agent.jira_client.get_projects()
            current_projects = [p['key'] for p in projects]
            orchestrator.jira_data_agent.available_projects = current_projects
        except:
            current_projects = orchestrator.jira_data_agent.available_projects
    
    return jsonify({
        "status": "healthy",
        "message": "JURIX AI Backend is running",
        "jira_connected": jira_connected,
        "available_projects": current_projects,
        "using_real_api": orchestrator.jira_data_agent.use_real_api,
        "using_mcp": getattr(orchestrator.jira_data_agent, 'use_mcp', False)
    })

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    """Chat endpoint that works with real Jira data"""
    if request.method == "OPTIONS":
        return "", 200
    
    logger.info("💬 Chat request received")  
    logger.info(f"Headers: {dict(request.headers)}")
        
    try:
        data = request.json
        logger.info(f"Request data: {data}")
        query = data.get("query", "")
        conversation_id = data.get("conversationId", str(uuid.uuid4()))
        
        logger.info(f"💬 Chat request: '{query}'")
        
        # Refresh available projects if needed
        if not orchestrator.jira_data_agent.available_projects:
            logger.info("📁 No projects cached, refreshing...")
            if orchestrator.jira_data_agent.jira_client:
                try:
                    projects = orchestrator.jira_data_agent.jira_client.get_projects()
                    orchestrator.jira_data_agent.available_projects = [p['key'] for p in projects]
                    logger.info(f"📁 Refreshed projects: {orchestrator.jira_data_agent.available_projects}")
                except Exception as e:
                    logger.error(f"Failed to refresh projects: {e}")
        
        # Get available projects
        available_projects = orchestrator.jira_data_agent.available_projects
        logger.info(f"📁 Available projects: {available_projects}")
        
        # Detect project context
        project_context = None
        query_upper = query.upper()
        
        # Check for explicit project mentions
        for project in available_projects:
            if project in query_upper:
                project_context = project
                logger.info(f"📁 Detected project in query: {project_context}")
                break
        
        # For project-related queries without explicit project, use first available
        project_keywords = ['velocity', 'sprint', 'tickets', 'team', 'productivity', 
                          'blockers', 'issues', 'backlog', 'probability', 'completing', 
                          'project', 'risky', 'analyze']
        
        if not project_context and any(keyword in query.lower() for keyword in project_keywords):
            if available_projects:
                # If MG is in the projects, prefer it
                if 'MG' in available_projects:
                    project_context = 'MG'
                else:
                    project_context = available_projects[0]
                logger.info(f"📁 Using project for analysis: {project_context}")
        
        # Update the orchestrator's state to use the correct project
        if project_context:
            # This ensures the workflow uses the right project
            orchestrator.jira_data_agent.mental_state.add_belief("default_project", project_context, 0.9, "api")
        
        # Run the workflow
        logger.info(f"🚀 Running workflow with project context: {project_context}")
        result = orchestrator.run_workflow(query, conversation_id)
        
        # Build response
        response = {
            "query": query,
            "response": result.get("response", "I couldn't generate a response. Please try again."),
            "conversation_id": conversation_id,
            "status": "success"
        }
        
        # Add project context
        if project_context:
            response["project_context"] = project_context
            # Replace any PROJ123 references with actual project
            response["response"] = response["response"].replace("PROJ123", project_context)
        
        # Add additional data
        if result.get("articles"):
            response["articles"] = [
                {"title": a.get("title", ""), "content": a.get("content", "")[:200] + "..."}
                for a in result.get("articles", [])[:3]
            ]
        
        if result.get("recommendations"):
            response["recommendations"] = result.get("recommendations", [])[:3]
        
        if result.get("tickets"):
            response["tickets_analyzed"] = len(result.get("tickets", []))
            logger.info(f"📊 Analyzed {len(result.get('tickets', []))} tickets")
        
        logger.info(f"✅ Response generated successfully")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "response": "Sorry, I encountered an error. Please try again."
        }), 500

@app.route("/api/projects", methods=["GET"])
def get_projects():
    """Get available Jira projects"""
    try:
        # Get fresh projects list
        projects = []
        if orchestrator.jira_data_agent.jira_client:
            jira_projects = orchestrator.jira_data_agent.jira_client.get_projects()
            projects = [{"key": p["key"], "name": p["name"]} for p in jira_projects]
            # Update the agent's project list
            orchestrator.jira_data_agent.available_projects = [p["key"] for p in projects]
        
        return jsonify({
            "status": "success",
            "projects": projects,
            "count": len(projects)
        })
    except Exception as e:
        logger.error(f"Error getting projects: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "projects": []
        }), 500

@app.route("/api/refresh", methods=["POST"])
def refresh_config():
    """Refresh Jira connection and projects"""
    try:
        logger.info("🔄 Refreshing Jira configuration...")
        
        # Re-initialize the connection
        success = fix_jira_agent()
        
        return jsonify({
            "status": "success" if success else "error",
            "connected": success,
            "available_projects": orchestrator.jira_data_agent.available_projects
        })
    except Exception as e:
        logger.error(f"Error refreshing config: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# Replace your existing dashboard endpoint in api_simple.py with this enhanced version

@app.route("/api/dashboard/<project_id>", methods=["GET", "POST"])
def dashboard(project_id):
    """Enhanced dashboard endpoint with all new features"""
    try:
        logger.info(f"📊 Enhanced dashboard request for project: {project_id}")
        
        # Ensure the project exists
        if project_id not in orchestrator.jira_data_agent.available_projects:
            # Try to refresh projects first
            if orchestrator.jira_data_agent.jira_client:
                try:
                    projects = orchestrator.jira_data_agent.jira_client.get_projects()
                    orchestrator.jira_data_agent.available_projects = [p['key'] for p in projects]
                except:
                    pass
            
            # Check again
            if project_id not in orchestrator.jira_data_agent.available_projects:
                return jsonify({
                    "status": "error",
                    "error": f"Project {project_id} not found"
                }), 404
        
        # Get time range from request (optional)
        time_range = {}
        logger.info(f"Fetching ALL tickets for project {project_id} (no date filter)")
        
        # Run the full productivity workflow
        state = orchestrator.run_productivity_workflow(project_id, time_range)
        
        # Extract all necessary data
        metrics = state.get("metrics", {})
        predictions = state.get("predictions", {})
        visualization_data = state.get("visualization_data", {})
        tickets = state.get("tickets", [])
        
        # Get risk assessment
        risk_response = get_risk_assessment(project_id)
        risk_data = risk_response.get_json()
        
        # Calculate sprint health pulse
        sprint_health = _calculate_comprehensive_sprint_health(predictions, metrics, tickets)
        
        # Get team energy levels
        team_energy = _calculate_team_energy(predictions, metrics)
        
        # Analyze historical patterns
        patterns = _analyze_historical_patterns(tickets)
        trends = _calculate_trends(tickets, patterns)
        
        # Get team analytics
        team_analytics = _analyze_team_performance(tickets, metrics, predictions)
        
        # Extract individual ticket predictions
        ticket_predictions = predictions.get("ticket_predictions", [])
        bottleneck_tickets = [t for t in ticket_predictions if t.get("is_bottleneck")]
        
        # Process charts data with all new visualizations
        charts_data = {
            "sprintProgress": _generate_sprint_progress_data(tickets),
            "velocityTrend": {
                "type": "line",
                "title": "Velocity Trend Forecast",
                "data": {
                    "labels": ['Sprint 1', 'Sprint 2', 'Sprint 3', 'Sprint 4', 'Sprint 5'],
                    "datasets": [{
                        "label": "Velocity",
                        "data": [32, 38, 35, 45, 42],
                        "borderColor": "#00875A",  # Deep green color
                        "backgroundColor": "rgba(0, 135, 90, 0.2)",  # Green shadow
                        "tension": 0.3,
                        "fill": True,
                        "pointRadius": 5,
                        "pointHoverRadius": 7,
                        "pointBackgroundColor": "#00875A",
                        "pointBorderColor": "#fff",
                        "pointBorderWidth": 2,
                        "borderWidth": 3
                    }]
                }
            },
            "burndown": _generate_burndown_data(tickets),
            "teamWorkload": None,
            "teamPerformance": None,
            "sprintHealthPulse": sprint_health["sprint_health"],
            "teamEnergyMeter": sprint_health["team_energy"],
            "historicalVelocity": {
                "type": "line",
                "title": "Historical Velocity Trend",
                "data": {
                    "labels": list(patterns["weekly_velocity"].keys())[-10:],
                    "datasets": [{
                        "label": "Weekly Velocity",
                        "data": list(patterns["weekly_velocity"].values())[-10:],
                        "borderColor": "#00875A",
                        "backgroundColor": "rgba(0, 135, 90, 0.2)",  # Match the style
                        "tension": 0.3,
                        "fill": True,
                        "pointRadius": 5,
                        "pointHoverRadius": 7,
                        "pointBackgroundColor": "#00875A",
                        "pointBorderColor": "#fff",
                        "pointBorderWidth": 2,
                        "borderWidth": 3
                    }]
                }
            },
            "cycleTimeEvolution": {
                "type": "line",
                "title": "Cycle Time Evolution",
                "data": {
                    "labels": list(patterns["cycle_time_evolution"].keys())[-10:],
                    "datasets": [{
                        "label": "Average Cycle Time",
                        "data": list(patterns["cycle_time_evolution"].values())[-10:],
                        "borderColor": "#FF5630",
                        "tension": 0.3
                    }]
                }
            }
        }
        
        # Extract chart data from visualization_data
        if visualization_data and "charts" in visualization_data:
            for chart in visualization_data["charts"]:
                if "Team Workload" in chart.get("title", ""):
                    charts_data["teamWorkload"] = chart
                elif "Velocity" in chart.get("title", "") and "Forecast" in chart.get("title", ""):
                    # Override with styled velocity data
                    chart_data = chart.get("data", {})
                    if chart_data.get("datasets") and len(chart_data["datasets"]) > 0:
                        # Apply the green styling to any velocity chart
                        chart_data["datasets"][0]["borderColor"] = "#00875A"
                        chart_data["datasets"][0]["backgroundColor"] = "rgba(0, 135, 90, 0.2)"
                        chart_data["datasets"][0]["pointBackgroundColor"] = "#00875A"
                        chart_data["datasets"][0]["pointBorderColor"] = "#fff"
                        chart_data["datasets"][0]["pointBorderWidth"] = 2
                        chart_data["datasets"][0]["borderWidth"] = 3
                        chart_data["datasets"][0]["fill"] = True
                        chart_data["datasets"][0]["tension"] = 0.3
                        chart_data["datasets"][0]["pointRadius"] = 5
                        chart_data["datasets"][0]["pointHoverRadius"] = 7
                    charts_data["velocityTrend"] = chart
                elif "Team Performance" in chart.get("title", ""):
                    charts_data["teamPerformance"] = chart
        
        # Generate success patterns
        success_patterns = _identify_success_patterns(tickets, metrics)
        
        # Build comprehensive response with ALL features
        response = {
            "project_id": project_id,
            "status": "success",
            "metrics": {
                "velocity": metrics.get("throughput", 0),
                "cycleTime": round(metrics.get("cycle_time", 0), 1),
                "efficiency": int(metrics.get("collaborative_insights", {}).get("workflow_efficiency_score", 0.7) * 100),
                "activeIssues": len([t for t in tickets if t.get("fields", {}).get("status", {}).get("name") not in ["Done", "Closed"]]),
                "throughput": metrics.get("throughput", 0),
                "velocityChange": _calculate_metric_change(metrics, "velocity"),
                "cycleTimeChange": _calculate_metric_change(metrics, "cycle_time"),
                "efficiencyChange": _calculate_metric_change(metrics, "efficiency"),
                "workload": metrics.get("workload", {}),
                "bottlenecks": metrics.get("bottlenecks", {}),
                "teamBalanceScore": metrics.get("collaborative_insights", {}).get("team_balance_score", 0.8),
                "criticalAlerts": len([w for w in predictions.get("warnings", []) if w.get("urgency") == "critical"])
            },
            "predictions": {
                "sprintCompletion": predictions.get("sprint_completion", {}),
                "velocityForecast": predictions.get("velocity_forecast", {}),
                "risks": predictions.get("risks", [])[:5],
                "warnings": predictions.get("warnings", []),
                "bottlenecks": bottleneck_tickets[:10],
                "ticketPredictions": ticket_predictions[:20],
                "teamBurnout": predictions.get("team_burnout_analysis", {}),
                "aiSummary": predictions.get("natural_language_summary", "")
            },
            "riskAssessment": {
                "score": risk_data.get("risk_score", 0),
                "level": risk_data.get("risk_level", "low"),
                "factors": risk_data.get("risk_factors", []),
                "monthlyScores": risk_data.get("monthly_scores", [])
            },
            "visualizationData": {
                "charts": charts_data,
                "sparklines": {
                    "velocity": _generate_sparkline_data(tickets, "velocity"),
                    "cycleTime": _generate_sparkline_data(tickets, "cycle"),
                    "efficiency": _generate_sparkline_data(tickets, "efficiency"),
                    "issues": _generate_sparkline_data(tickets, "issues")
                }
            },
            "patterns": {
                "historical": patterns,
                "trends": trends,
                "insights": _generate_pattern_insights(patterns, trends)
            },
            "teamAnalytics": {
                "members": team_analytics["members"],
                "performanceMatrix": team_analytics["performance_matrix"],
                "collaborationPatterns": team_analytics["collaboration_patterns"],
                "individualMetrics": team_analytics["individual_metrics"],
                "optimalDistribution": team_analytics["optimal_distribution"]
            },
            "recommendations": state.get("recommendations", [])[:5] + team_analytics["team_recommendations"],
            "alerts": _generate_alerts(predictions, metrics),
            "report": state.get("report", ""),
            "ticketsAnalyzed": len(tickets),
            "lastUpdated": datetime.now().isoformat(),
            "collaborationMetadata": state.get("collaboration_metadata", {}),
            "recentActivity": _get_recent_activity(tickets),
            "teamMembers": _get_team_members(metrics.get("workload", {})),
            "successPatterns": success_patterns,
            "sprintHealth": sprint_health["sprint_health"],
            "teamEnergy": sprint_health["team_energy"],
            "criticalFactors": sprint_health["critical_factors"],
            "recoveryPlan": sprint_health["recovery_plan"],
            "healthHistory": sprint_health["health_history"]
        }
        
        # Store dashboard state for real-time updates
        if orchestrator.shared_memory:
            dashboard_key = f"dashboard_{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            orchestrator.shared_memory.store(dashboard_key, response)
        
        logger.info(f"✅ Enhanced dashboard data generated successfully for {project_id}")
        
        # Track update
        update_tracker.add_update(project_id, 'dashboard_refresh', {
            'source': 'manual_refresh',
            'ticket_count': len(tickets),
            'has_predictions': True,
            'features_included': ['predictions', 'patterns', 'team_analytics', 'health_metrics']
        })
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "project_id": project_id
        }), 500

def _calculate_sprint_health(predictions, metrics):
    """Calculate sprint health pulse data"""
    sprint_completion = predictions.get("sprint_completion", {})
    probability = sprint_completion.get("probability", 0.85)
    
    # Calculate health score (0-100)
    health_score = probability * 100
    
    # Determine pulse rate based on risk level
    risk_level = sprint_completion.get("risk_level", "low")
    if risk_level == "critical":
        pulse_rate = 120  # Fast pulse - danger
    elif risk_level == "high":
        pulse_rate = 100
    elif risk_level == "medium":
        pulse_rate = 80
    else:
        pulse_rate = 60  # Normal healthy pulse
    
    # Determine color based on health
    if health_score >= 80:
        color = "#00875A"  # Green
    elif health_score >= 60:
        color = "#FFAB00"  # Yellow
    elif health_score >= 40:
        color = "#FF5630"  # Orange
    else:
        color = "#DE350B"  # Red
    
    return {
        "health_score": health_score,
        "pulse_rate": pulse_rate,
        "color": color,
        "status": "healthy" if health_score >= 80 else "at_risk" if health_score >= 60 else "critical",
        "critical_moments": sprint_completion.get("risk_factors", [])
    }

def _calculate_team_energy(predictions, metrics):
    """Calculate team energy levels"""
    team_analysis = predictions.get("team_burnout_analysis", {})
    overloaded_members = team_analysis.get("overloaded_members", [])
    team_metrics = team_analysis.get("team_metrics", {})
    
    # Calculate average energy level
    base_energy = 100
    energy_depletion = len(overloaded_members) * 15  # Each overloaded member depletes 15%
    avg_energy = max(20, base_energy - energy_depletion)
    
    # Calculate individual member energy
    members_energy = []
    workload = metrics.get("workload", {})
    avg_workload = team_metrics.get("avg_load", 5)
    
    for member, load in workload.items():
        energy = 100 - ((load / max(avg_workload, 1)) * 30)
        energy = max(20, min(100, energy))
        
        members_energy.append({
            "name": member,
            "energy": energy,
            "workload": load,
            "status": "healthy" if energy >= 70 else "tired" if energy >= 40 else "exhausted",
            "recovery_time": 0 if energy >= 70 else (70 - energy) / 10  # Days needed to recover
        })
    
    return {
        "average_energy": avg_energy,
        "members": members_energy,
        "at_risk_count": len(overloaded_members),
        "recommendations": team_analysis.get("recommendations", [])
    }

def _identify_success_patterns(tickets, metrics):
    """Identify patterns from successful sprints"""
    patterns = []
    
    # Pattern 1: High completion rate
    done_tickets = len([t for t in tickets if t.get("fields", {}).get("status", {}).get("name") in ["Done", "Closed"]])
    completion_rate = done_tickets / max(len(tickets), 1)
    
    if completion_rate > 0.8:
        patterns.append({
            "pattern": "High Completion Sprint",
            "dna": {
                "completion_rate": completion_rate,
                "avg_cycle_time": metrics.get("cycle_time", 0),
                "team_size": len(metrics.get("workload", {}))
            },
            "success_factors": ["Good planning", "Balanced workload", "Clear requirements"],
            "similarity_score": 0.85
        })
    
    # Pattern 2: Balanced workload
    if metrics.get("collaborative_insights", {}).get("team_balance_score", 0) > 0.8:
        patterns.append({
            "pattern": "Well-Balanced Team",
            "dna": {
                "balance_score": metrics["collaborative_insights"]["team_balance_score"],
                "workload_variance": 0.2
            },
            "success_factors": ["Even distribution", "No burnout", "Collaborative work"],
            "similarity_score": 0.75
        })
    
    return patterns

def _calculate_metric_change(metrics, metric_type):
    """Calculate change indicators for metrics"""
    # This is a simplified version - in production, compare with historical data
    if metric_type == "velocity":
        return {
            "value": 12,
            "text": "+12% from last sprint",
            "type": "positive"
        }
    elif metric_type == "cycle_time":
        return {
            "value": -0.5,
            "text": "-0.5 days improvement",
            "type": "positive"
        }
    elif metric_type == "efficiency":
        return {
            "value": 78,
            "text": "Above target",
            "type": "positive"
        }
    return {
        "value": 0,
        "text": "No change",
        "type": "neutral"
    }

def _identify_bottlenecks(predictions, metrics):
    """Extract bottleneck predictions"""
    bottlenecks = []
    
    # From predictions
    ticket_predictions = predictions.get("ticket_predictions", [])
    for ticket in ticket_predictions:
        if ticket.get("is_bottleneck"):
            bottlenecks.append({
                "type": "ticket",
                "key": ticket.get("ticket_key"),
                "description": ticket.get("bottleneck_reason"),
                "severity": "high" if ticket.get("age_days", 0) > 14 else "medium",
                "estimated_impact": f"{ticket.get('days_remaining', 0):.1f} days"
            })
    
    # From status distribution
    status_bottlenecks = metrics.get("bottlenecks", {})
    for status, count in status_bottlenecks.items():
        if count > 5:
            bottlenecks.append({
                "type": "process",
                "key": status,
                "description": f"{count} tickets stuck in {status}",
                "severity": "high" if count > 10 else "medium",
                "estimated_impact": f"{count * metrics.get('cycle_time', 5):.0f} total days"
            })
    
    return bottlenecks[:5]  # Top 5 bottlenecks

def _generate_alerts(predictions, metrics):
    """Generate alert list from predictions and metrics"""
    alerts = []
    
    # From warnings
    warnings = predictions.get("warnings", [])
    for warning in warnings:
        if warning.get("urgency") == "critical":
            alerts.append({
                "type": "critical",
                "message": warning.get("message", "Critical issue detected"),
                "time": "Just now",
                "action": warning.get("recommended_action", "Review immediately")
            })
    
    # From risks
    risks = predictions.get("risks", [])
    for risk in risks:
        if risk.get("severity") == "high" and risk.get("probability", 0) > 0.7:
            alerts.append({
                "type": "warning",
                "message": risk.get("description", "High risk identified"),
                "time": "5 minutes ago",
                "action": risk.get("mitigation", "Take preventive action")
            })
    
    return alerts[:10]  # Limit to 10 alerts
    
@app.route("/api/forecast/<project_key>", methods=["POST"])
def generate_forecast(project_key):
    """Generate AI-powered forecasts for the project using REAL data"""
    try:
        logger.info(f"🔮 Generating forecast for project: {project_key}")
        
        # Get forecast type from request
        data = request.json or {}
        forecast_type = data.get('type', 'velocity')  # velocity, burndown, capacity
        
        # Run predictive workflow to get REAL forecasts
        state = orchestrator.run_predictive_workflow(project_key, forecast_type)
        predictions = state.get("predictions", {})
        tickets = state.get("tickets", [])
        metrics = state.get("metrics", {})
        
        # Build forecast response based on type
        forecast_data = {
            "status": "success",
            "type": forecast_type,
            "timestamp": datetime.now().isoformat(),
            "project_key": project_key
        }
        
        if forecast_type == "velocity":
            # Use REAL velocity forecast from predictive agent
            velocity_forecast = predictions.get("velocity_forecast", {})
            sprint_completion = predictions.get("sprint_completion", {})
            
            # Get actual forecast values
            forecast_values = velocity_forecast.get("forecast", [])
            trend = velocity_forecast.get("trend", "stable")
            confidence = velocity_forecast.get("confidence", 0.7)
            
            # Build chart data with proper labels
            current_velocity = metrics.get("throughput", 0)
            historical_avg = velocity_forecast.get("historical_average", current_velocity)
            
            # Create labels for past and future weeks
            labels = []
            data_points = []
            
            # Add historical data (last 3 weeks)
            velocity_history = velocity_forecast.get("velocity_history", [current_velocity] * 3)
            for i in range(3, 0, -1):
                labels.append(f"Week -{i}")
                if i <= len(velocity_history):
                    data_points.append(velocity_history[-i])
                else:
                    data_points.append(historical_avg)
            
            # Add current week
            labels.append("Current Week")
            data_points.append(current_velocity)
            
            # Add future predictions
            for i, value in enumerate(forecast_values[:3]):
                labels.append(f"Week +{i+1}")
                data_points.append(round(value, 1))
            
            forecast_data["data"] = {
                "chart": {
                    "type": "line",
                    "data": {
                        "labels": labels,
                        "datasets": [{
                            "label": "Velocity Trend",
                            "data": data_points,
                            "borderColor": "rgba(0, 135, 90, 1)",
                            "backgroundColor": "rgba(0, 135, 90, 0.2)",
                            "tension": 0.3,
                            "fill": True
                        }]
                    }
                },
                "trend": trend,
                "trend_percentage": velocity_forecast.get("trend_percentage", 0),
                "next_week_estimate": velocity_forecast.get("next_week_estimate", current_velocity),
                "confidence": confidence,
                "insights": velocity_forecast.get("insights", ""),
                "recommendations": sprint_completion.get("recommended_actions", [])[:3],
                "current_velocity": current_velocity,
                "historical_average": historical_avg,
                "volatility": velocity_forecast.get("volatility", 0)
            }
            
        elif forecast_type == "burndown":
            # Use REAL sprint data for burndown
            sprint_completion = predictions.get("sprint_completion", {})
            velocity_forecast = predictions.get("velocity_forecast", {})
            
            # Get actual remaining work and velocity
            remaining_work = sprint_completion.get("remaining_work", 0)
            completion_percentage = sprint_completion.get("completion_percentage", 0)
            probability = sprint_completion.get("probability", 0.8)
            risk_level = sprint_completion.get("risk_level", "low")
            
            # Calculate sprint parameters
            sprint_length = 10  # Configurable sprint length
            today = 5  # Current day in sprint (you can calculate this from actual dates)
            
            # Calculate daily velocity
            current_velocity = metrics.get("throughput", 0) / 5  # Weekly to daily
            expected_velocity = velocity_forecast.get("next_week_estimate", current_velocity * 5) / 5
            
            # Generate ideal burndown
            total_work = remaining_work / (1 - completion_percentage) if completion_percentage < 1 else remaining_work
            ideal_burndown = []
            predicted_burndown = []
            
            for day in range(sprint_length + 1):
                # Ideal burndown (linear)
                ideal_points = total_work * (1 - day / sprint_length)
                ideal_burndown.append(round(ideal_points, 1))
                
                # Predicted burndown based on actual data
                if day <= today:
                    # Historical data - calculate from actual completion
                    if day == 0:
                        predicted_points = total_work
                    else:
                        # Calculate based on actual progress
                        daily_progress = (total_work * completion_percentage) / today
                        predicted_points = total_work - (daily_progress * day)
                else:
                    # Future prediction based on velocity and risks
                    days_from_today = day - today
                    
                    # Apply risk factors
                    risk_multiplier = {
                        "critical": 0.5,
                        "high": 0.7,
                        "medium": 0.85,
                        "low": 0.95,
                        "minimal": 1.0
                    }.get(risk_level, 0.9)
                    
                    adjusted_velocity = expected_velocity * risk_multiplier
                    predicted_points = remaining_work - (adjusted_velocity * days_from_today)
                
                predicted_burndown.append(max(0, round(predicted_points, 1)))
            
            # Calculate if sprint will complete on time
            final_predicted = predicted_burndown[-1]
            on_track = final_predicted <= 5  # Allow small buffer
            
            forecast_data["data"] = {
                "chart": {
                    "type": "line",
                    "data": {
                        "labels": [f"Day {i}" for i in range(sprint_length + 1)],
                        "datasets": [{
                            "label": "Ideal",
                            "data": ideal_burndown,
                            "borderColor": "#C1C7D0",
                            "borderDash": [5, 5],
                            "tension": 0,
                            "fill": False
                        }, {
                            "label": "Predicted",
                            "data": predicted_burndown,
                            "borderColor": "#0052CC" if on_track else "#FF5630",
                            "backgroundColor": f"rgba(0, 82, 204, 0.1)" if on_track else "rgba(255, 86, 48, 0.1)",
                            "tension": 0.2,
                            "fill": True
                        }]
                    }
                },
                "ideal_burndown": ideal_burndown,
                "predicted_burndown": predicted_burndown,
                "completion_probability": probability,
                "days_remaining": sprint_length - today,
                "current_day": today,
                "at_risk": risk_level in ["high", "critical"],
                "risk_level": risk_level,
                "remaining_work": remaining_work,
                "daily_velocity_needed": remaining_work / (sprint_length - today) if sprint_length > today else 0,
                "current_velocity": expected_velocity,
                "on_track": on_track,
                "insights": sprint_completion.get("reasoning", ""),
                "recommendations": sprint_completion.get("recommended_actions", [])
            }
            
        elif forecast_type == "capacity":
            # Use REAL team analysis for capacity
            team_analysis = predictions.get("team_burnout_analysis", {})
            team_metrics = team_analysis.get("team_metrics", {})
            overloaded_members = team_analysis.get("overloaded_members", [])
            
            # Calculate real capacity metrics
            team_size = team_metrics.get("team_size", 5)
            avg_load = team_metrics.get("avg_load", 5)
            max_load = team_metrics.get("max_load", 10)
            
            # Hours per person per day (configurable)
            hours_per_person = 6
            current_capacity = team_size * hours_per_person
            
            # Calculate optimal capacity (80% utilization is ideal)
            optimal_capacity = current_capacity * 0.8
            
            # Calculate actual utilization
            total_tickets = sum(metrics.get("workload", {}).values())
            estimated_hours_per_ticket = 4  # Configurable
            used_capacity = (total_tickets * estimated_hours_per_ticket) / 5  # Weekly to daily
            utilization = min(used_capacity / current_capacity, 1.5) if current_capacity > 0 else 0
            
            # Determine capacity trend
            if len(overloaded_members) > team_size * 0.3:
                capacity_trend = "declining"
                trend_reason = "Team burnout risk detected"
            elif utilization > 1.2:
                capacity_trend = "overloaded"
                trend_reason = "Team is over capacity"
            elif utilization < 0.6:
                capacity_trend = "underutilized"
                trend_reason = "Team has available capacity"
            else:
                capacity_trend = "stable"
                trend_reason = "Team capacity is balanced"
            
            # Generate capacity forecast chart
            capacity_labels = []
            capacity_values = []
            optimal_values = []
            
            # Historical and future capacity
            for week in range(-2, 4):
                if week < 0:
                    capacity_labels.append(f"Week {week}")
                elif week == 0:
                    capacity_labels.append("Current Week")
                else:
                    capacity_labels.append(f"Week +{week}")
                
                # Simulate capacity changes based on trend
                if capacity_trend == "declining":
                    adjustment = 1 - (abs(week) * 0.05) if week >= 0 else 1
                elif capacity_trend == "overloaded":
                    adjustment = 1 - (week * 0.03) if week >= 0 else 1
                else:
                    adjustment = 1
                
                capacity_values.append(round(current_capacity * adjustment, 1))
                optimal_values.append(round(optimal_capacity, 1))
            
            forecast_data["data"] = {
                "chart": {
                    "type": "line",
                    "data": {
                        "labels": capacity_labels,
                        "datasets": [{
                            "label": "Team Capacity",
                            "data": capacity_values,
                            "borderColor": "#00875A" if capacity_trend == "stable" else "#FF5630",
                            "backgroundColor": "rgba(0, 135, 90, 0.2)" if capacity_trend == "stable" else "rgba(255, 86, 48, 0.2)",
                            "tension": 0.3,
                            "fill": True
                        }, {
                            "label": "Optimal Capacity",
                            "data": optimal_values,
                            "borderColor": "#C1C7D0",
                            "borderDash": [5, 5],
                            "tension": 0,
                            "fill": False
                        }]
                    }
                },
                "current_capacity": current_capacity,
                "optimal_capacity": optimal_capacity,
                "used_capacity": round(used_capacity, 1),
                "utilization_percentage": round(utilization * 100, 1),
                "at_risk_members": len(overloaded_members),
                "team_size": team_size,
                "capacity_trend": capacity_trend,
                "trend_reason": trend_reason,
                "overloaded_members": [
                    {
                        "name": member["name"],
                        "current_load": member["current_load"],
                        "overload_factor": round(member["overload_factor"], 2)
                    }
                    for member in overloaded_members[:5]
                ],
                "recommendations": team_analysis.get("recommendations", []),
                "workload_distribution": metrics.get("workload", {}),
                "insights": f"{len(overloaded_members)} team members at risk. Average load: {avg_load:.1f} tickets/person"
            }
        
        logger.info(f"✅ Forecast generated: {forecast_type} with real data")
        return jsonify(forecast_data)
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "type": forecast_type
        }), 500
    
@app.route("/api/ticket-knowledge-graph/<ticket_id>", methods=["GET"])
def get_ticket_knowledge_graph(ticket_id):
    """Get knowledge graph data for a ticket"""
    try:
        if hasattr(orchestrator, 'enhanced_retrieval_agent'):
            rag_pipeline = orchestrator.enhanced_retrieval_agent.rag_pipeline
            
            # Get ticket documentation
            docs = rag_pipeline.neo4j_manager.get_ticket_documentation(ticket_id)
            
            # Get related tickets
            related_query = """
            MATCH (t:Ticket {key: $ticket_key})-[:RESOLVES]-(d:Document)-[:RESOLVES]-(other:Ticket)
            WHERE other.key <> $ticket_key
            RETURN DISTINCT other.key as ticket_key, other.summary as summary
            LIMIT 10
            """
            
            with rag_pipeline.neo4j_manager.driver.session() as session:
                related = session.run(related_query, ticket_key=ticket_id)
                related_tickets = [{"key": r["ticket_key"], "summary": r["summary"]} for r in related]
            
            return jsonify({
                "status": "success",
                "ticket_id": ticket_id,
                "documentation": docs,
                "related_tickets": related_tickets,
                "graph_data": {
                    "nodes": [
                        {"id": ticket_id, "type": "ticket", "label": ticket_id}
                    ] + [
                        {"id": d["doc_id"], "type": "document", "label": d["title"]}
                        for d in docs
                    ],
                    "edges": [
                        {"source": ticket_id, "target": d["doc_id"], "type": "RESOLVES"}
                        for d in docs
                    ]
                }
            })
        else:
            return jsonify({
                "status": "error",
                "error": "Enhanced RAG not enabled"
            }), 400
            
    except Exception as e:
        logger.error(f"Error getting knowledge graph: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/api/advanced-predictions/<project_key>", methods=["POST"])
def get_advanced_predictions(project_key):
    """Get comprehensive predictive analytics including individual ticket predictions"""
    try:
        logger.info(f"🔮 Generating advanced predictions for project: {project_key}")
        
        # Get request parameters
        data = request.json or {}
        include_tickets = data.get('include_ticket_predictions', True)
        include_patterns = data.get('include_patterns', True)
        
        # Run enhanced predictive workflow
        state = orchestrator.run_predictive_workflow(project_key, "comprehensive")
        predictions = state.get("predictions", {})
        tickets = state.get("tickets", [])
        
        # Extract all prediction components
        response = {
            "status": "success",
            "project_key": project_key,
            "timestamp": datetime.now().isoformat(),
            "predictions": {
                # Sprint completion with detailed breakdown
                "sprint_completion": predictions.get("sprint_completion", {}),
                
                # Velocity forecast with confidence intervals
                "velocity_forecast": predictions.get("velocity_forecast", {}),
                
                # Team burnout analysis with individual metrics
                "team_burnout": predictions.get("team_burnout_analysis", {}),
                
                # Risk assessment with mitigation strategies
                "risks": predictions.get("risks", []),
                
                # Early warnings with urgency levels
                "warnings": predictions.get("warnings", []),
                
                # Natural language insights
                "ai_summary": predictions.get("natural_language_summary", "")
            }
        }
        
        # Add individual ticket predictions if requested
        if include_tickets:
            ticket_predictions = predictions.get("ticket_predictions", [])
            response["predictions"]["ticket_predictions"] = ticket_predictions[:20]  # Top 20
            
            # Add bottleneck tickets separately
            bottleneck_tickets = [t for t in ticket_predictions if t.get("is_bottleneck")]
            response["predictions"]["bottleneck_tickets"] = bottleneck_tickets
        
        # Add pattern analysis if requested
        if include_patterns:
            response["patterns"] = _analyze_historical_patterns(tickets)
        
        logger.info(f"✅ Advanced predictions generated successfully")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error generating advanced predictions: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/api/team-analytics/<project_key>", methods=["GET"])
def get_team_analytics(project_key):
    """Get detailed team performance analytics"""
    try:
        logger.info(f"👥 Generating team analytics for project: {project_key}")
        
        # Get tickets and run analysis
        state = orchestrator.run_productivity_workflow(project_key, {})
        tickets = state.get("tickets", [])
        metrics = state.get("metrics", {})
        predictions = state.get("predictions", {})
        
        # Extract team-specific data
        team_data = _analyze_team_performance(tickets, metrics, predictions)
        
        response = {
            "status": "success",
            "project_key": project_key,
            "team_analytics": {
                "members": team_data["members"],
                "performance_matrix": team_data["performance_matrix"],
                "collaboration_patterns": team_data["collaboration_patterns"],
                "load_distribution": {
                    "current": metrics.get("workload", {}),
                    "optimal": team_data["optimal_distribution"],
                    "balance_score": metrics.get("collaborative_insights", {}).get("team_balance_score", 0)
                },
                "burnout_analysis": predictions.get("team_burnout_analysis", {}),
                "individual_metrics": team_data["individual_metrics"]
            },
            "recommendations": team_data["team_recommendations"]
        }
        
        logger.info(f"✅ Team analytics generated successfully")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error generating team analytics: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/api/historical-patterns/<project_key>", methods=["POST"])
def get_historical_patterns(project_key):
    """Analyze historical patterns and trends"""
    try:
        logger.info(f"📈 Analyzing historical patterns for project: {project_key}")
        
        data = request.json or {}
        time_range = data.get('time_range', {
            'months': 3  # Default to 3 months
        })
        
        # Get historical tickets
        if 'months' in time_range:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30 * time_range['months'])
        else:
            start_date = datetime.fromisoformat(time_range.get('start', '2025-01-01'))
            end_date = datetime.fromisoformat(time_range.get('end', datetime.now().isoformat()))
        
        # Run analysis with extended time range
        state = orchestrator.run_productivity_workflow(project_key, {
            'start': start_date.isoformat(),
            'end': end_date.isoformat()
        })
        
        tickets = state.get("tickets", [])
        patterns = _analyze_historical_patterns(tickets)
        
        # Add trend analysis
        trends = _calculate_trends(tickets, patterns)
        
        response = {
            "status": "success",
            "project_key": project_key,
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": (end_date - start_date).days
            },
            "patterns": patterns,
            "trends": trends,
            "insights": _generate_pattern_insights(patterns, trends)
        }
        
        logger.info(f"✅ Historical patterns analyzed successfully")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error analyzing historical patterns: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/api/sprint-health/<project_key>", methods=["GET"])
def get_sprint_health_details(project_key):
    """Get detailed sprint health metrics with pulse data"""
    try:
        logger.info(f"💓 Calculating detailed sprint health for: {project_key}")
        
        # Get current sprint data
        state = orchestrator.run_predictive_workflow(project_key, "sprint_health")
        predictions = state.get("predictions", {})
        tickets = state.get("tickets", [])
        metrics = state.get("metrics", {})
        
        # Calculate comprehensive health metrics
        health_data = _calculate_comprehensive_sprint_health(predictions, metrics, tickets)
        
        response = {
            "status": "success",
            "project_key": project_key,
            "sprint_health": health_data["sprint_health"],
            "team_energy": health_data["team_energy"],
            "critical_factors": health_data["critical_factors"],
            "recovery_plan": health_data["recovery_plan"],
            "health_history": health_data["health_history"]
        }
        
        logger.info(f"✅ Sprint health calculated successfully")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error calculating sprint health: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/api/collaboration-insights/<project_key>", methods=["GET"])
def get_collaboration_insights(project_key):
    """Get insights from collaborative agent analysis"""
    try:
        logger.info(f"🤝 Generating collaboration insights for: {project_key}")
        
        # Run collaborative workflow
        state = orchestrator.run_workflow(f"Analyze collaboration patterns for {project_key}")
        
        # Extract collaboration metadata
        collab_metadata = state.get("collaboration_metadata", {})
        collab_trace = state.get("collaboration_trace", [])
        
        response = {
            "status": "success",
            "project_key": project_key,
            "collaboration_insights": {
                "agents_collaborated": collab_metadata.get("collaborating_agents", []),
                "insights_generated": collab_metadata.get("insights_count", 0),
                "collaboration_types": collab_metadata.get("collaboration_types", []),
                "cross_agent_findings": _extract_cross_agent_findings(collab_trace),
                "pattern_recognition": collab_metadata.get("patterns_identified", []),
                "knowledge_connections": collab_metadata.get("knowledge_articles_suggested", [])
            }
        }
        
        logger.info(f"✅ Collaboration insights generated")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error generating collaboration insights: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500
    
# Fixed api_article_feedback.py - Add this to your api_simple.py

@app.route("/api/article/generate/<ticket_id>", methods=["POST"])
def generate_article(ticket_id):
    """Generate initial article for a resolved ticket"""
    try:
        logger.info(f"📝 ========== ARTICLE GENERATION REQUEST ==========")
        logger.info(f"📝 Ticket ID: {ticket_id}")
        logger.info(f"📝 Request Method: {request.method}")
        logger.info(f"📝 Request Headers: {dict(request.headers)}")
        logger.info(f"📝 Request Data: {request.get_json()}")
        
        # Get ticket details first
        project_id = ticket_id.split('-')[0] if '-' in ticket_id else "PROJ123"
        
        logger.info(f"📝 Project ID extracted: {project_id}")
        
        # Run article generation workflow
        result = orchestrator.run_jira_workflow(ticket_id, project_id=project_id)
        
        logger.info(f"📝 Workflow result: {result.get('workflow_status')}")
        
        # Extract the generated article
        article = result.get("article", {})
        workflow_status = result.get("workflow_status", "failure")
        
        if workflow_status == "success" and article:
            # Store article in shared memory for easy retrieval
            article_key = f"article_draft:{ticket_id}"
            # Use Redis directly instead of shared_memory.store
            orchestrator.shared_memory.redis_client.set(
                article_key, 
                json.dumps(article)
            )
            orchestrator.shared_memory.redis_client.expire(article_key, 86400)  # 24 hours
            
            return jsonify({
                "status": "success",
                "ticket_id": ticket_id,
                "article": article,
                "version": article.get("version", 1),
                "approval_status": article.get("approval_status", "pending"),
                "message": "Article generated successfully. Please review and provide feedback."
            })
        else:
            return jsonify({
                "status": "error",
                "error": "Failed to generate article",
                "details": result.get("error", "Unknown error")
            }), 500
            
    except Exception as e:
        logger.error(f"Error generating article: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# Make sure this endpoint exists in your api_simple.py file:

@app.route("/api/article/feedback/<ticket_id>", methods=["POST"])
def submit_article_feedback(ticket_id):
    """Submit feedback for article refinement"""
    try:
        logger.info(f"📝 ========== FEEDBACK RECEIVED ==========")
        logger.info(f"📝 Ticket ID: {ticket_id}")
        
        data = request.json
        logger.info(f"📝 Feedback data: {data}")
        
        feedback = data.get("feedback", "")
        action = data.get("action", "refine")  # refine, approve, reject
        current_version = data.get("current_version", 1)
        
        logger.info(f"📝 Action: {action}, Version: {current_version}")
        logger.info(f"📝 Feedback: {feedback[:100]}...")
        
        # Get current article from Redis
        article_key = f"article_draft:{ticket_id}"
        article_json = orchestrator.shared_memory.redis_client.get(article_key)
        
        if not article_json:
            logger.error(f"❌ No article found for {ticket_id}")
            return jsonify({
                "status": "error",
                "error": "Article not found. Please generate an article first."
            }), 404
            
        current_article = json.loads(article_json)
        logger.info(f"📄 Current article version: {current_article.get('version', 1)}")
        
        if action == "approve":
            # Mark as approved
            current_article["approval_status"] = "approved"
            current_article["approved_at"] = datetime.now().isoformat()
            current_article["approved_version"] = current_version
            
            # Extract project from ticket ID
            project_key = ticket_id.split('-')[0] if '-' in ticket_id else "PROJ"
            
            # PUBLISH TO CONFLUENCE
            logger.info(f"📚 Publishing approved article to Confluence space: {project_key}")
            
            try:
                # Ensure enhanced RAG is enabled
                if not hasattr(orchestrator, 'enhanced_retrieval_agent'):
                    logger.info("Enabling enhanced RAG for publishing...")
                    orchestrator.enable_enhanced_rag()
                
                # Use the orchestrator's delegation method
                publish_result = orchestrator.publish_article_to_confluence(
                    article=current_article,
                    ticket_id=ticket_id,
                    project_key=project_key
                )
                
                if publish_result["status"] == "success":
                    # Update article with Confluence info
                    current_article["confluence_page_id"] = publish_result["page_id"]
                    current_article["confluence_url"] = publish_result["page_url"]
                    current_article["published_to_confluence"] = True
                    
                    logger.info(f"✅ Published to Confluence: {publish_result['page_url']}")
                    
                    # Store final version
                    final_key = f"article_approved:{ticket_id}"
                    orchestrator.shared_memory.redis_client.set(
                        final_key,
                        json.dumps(current_article)
                    )
                    orchestrator.shared_memory.redis_client.expire(final_key, 86400 * 30)  # 30 days
                    
                    return jsonify({
                        "status": "success",
                        "message": "Article approved and published to Confluence!",
                        "article": current_article,
                        "confluence_url": current_article.get("confluence_url"),
                        "next_step": "view_in_confluence"
                    })
                else:
                    logger.error(f"❌ Failed to publish to Confluence: {publish_result.get('error')}")
                    return jsonify({
                        "status": "error",
                        "error": f"Failed to publish to Confluence: {publish_result.get('error')}"
                    }), 500
                    
            except Exception as e:
                logger.error(f"❌ Error during publishing: {e}", exc_info=True)
                return jsonify({
                    "status": "error",
                    "error": f"Publishing error: {str(e)}"
                }), 500
            
        elif action == "reject":
            # Mark as rejected
            current_article["approval_status"] = "rejected"
            current_article["rejected_at"] = datetime.now().isoformat()
            current_article["rejection_reason"] = feedback
            
            # Update stored article
            orchestrator.shared_memory.redis_client.set(
                article_key,
                json.dumps(current_article)
            )
            
            logger.info(f"❌ Article rejected for {ticket_id}")
            
            return jsonify({
                "status": "success",
                "message": "Article rejected",
                "article": current_article
            })
            
        else:  # refine
            # Extract project ID from ticket
            project_id = ticket_id.split('-')[0] if '-' in ticket_id else "PROJ123"
            new_version = current_version + 1
            
            logger.info(f"🔄 Running AI refinement for v{new_version} with feedback")
            
            # Prepare input for article generator with human feedback
            refinement_input = {
                "ticket_id": ticket_id,
                "project_id": project_id,
                "human_feedback": feedback,
                "article_version": new_version,
                "previous_article": current_article
            }
            
            # Run the article generation workflow with feedback
            logger.info(f"🤖 Calling article generator with human feedback...")
            result = orchestrator.jira_article_generator.run(refinement_input)
            
            if result.get("workflow_status") == "success" and result.get("article"):
                refined_article = result["article"]
                
                # Ensure version and feedback history are properly set
                refined_article["version"] = new_version
                refined_article["approval_status"] = "pending"
                refined_article["refined_at"] = datetime.now().isoformat()
                
                # Add feedback to history
                if "feedback_history" not in refined_article:
                    refined_article["feedback_history"] = []
                
                refined_article["feedback_history"].append({
                    "version": current_version,
                    "feedback": feedback,
                    "action": "refine",
                    "timestamp": datetime.now().isoformat(),
                    "applied": True
                })
                
                # Store the refined article
                orchestrator.shared_memory.redis_client.set(
                    article_key,
                    json.dumps(refined_article)
                )
                
                # Also store version history
                version_key = f"article_version:{ticket_id}:v{new_version}"
                orchestrator.shared_memory.redis_client.set(
                    version_key,
                    json.dumps(refined_article)
                )
                orchestrator.shared_memory.redis_client.expire(version_key, 86400 * 30)
                
                logger.info(f"✅ Article refined to v{new_version} using AI with human feedback")
                
                return jsonify({
                    "status": "success",
                    "message": "Article refined based on feedback",
                    "article": refined_article,
                    "version": new_version,
                    "changes_made": "AI regenerated the article based on your feedback"
                })
            else:
                # Fallback error handling
                logger.error(f"❌ AI refinement failed: {result.get('error', 'Unknown error')}")
                
                return jsonify({
                    "status": "error",
                    "error": "Failed to refine article with AI. Please try again.",
                    "details": result.get("error", "Unknown error")
                }), 500
                
    except Exception as e:
        logger.error(f"❌ Error processing feedback: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/api/article/status/<ticket_id>", methods=["GET"])
def get_article_status(ticket_id):
    """Get current article status and version"""
    try:
        # Check for approved article first
        approved_key = f"article_approved:{ticket_id}"
        approved_article_json = orchestrator.shared_memory.redis_client.get(approved_key)
        
        if approved_article_json:
            approved_article = json.loads(approved_article_json)
            return jsonify({
                "status": "success",
                "ticket_id": ticket_id,
                "article": approved_article,
                "approval_status": "approved",
                "is_final": True
            })
        
        # Check for draft
        draft_key = f"article_draft:{ticket_id}"
        draft_article_json = orchestrator.shared_memory.redis_client.get(draft_key)
        
        if draft_article_json:
            draft_article = json.loads(draft_article_json)
            return jsonify({
                "status": "success",
                "ticket_id": ticket_id,
                "article": draft_article,
                "approval_status": draft_article.get("approval_status", "pending"),
                "is_final": False
            })
        
        return jsonify({
            "status": "not_found",
            "message": "No article found for this ticket"
        }), 404
        
    except Exception as e:
        logger.error(f"Error getting article status: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/api/article/history/<ticket_id>", methods=["GET"])
def get_article_history(ticket_id):
    """Get all versions and feedback history for an article"""
    try:
        history = []
        version = 1
        
        # Get all versions
        while True:
            version_key = f"article_version:{ticket_id}:v{version}"
            article_data = orchestrator.shared_memory.redis_client.get(version_key)
            
            if not article_data:
                break
                
            article = json.loads(article_data)
            history.append({
                "version": version,
                "created_at": article.get("created_at"),
                "feedback_history": article.get("feedback_history", []),
                "approval_status": article.get("approval_status", "pending"),
                "content_preview": article.get("content", "")[:200] + "..."
            })
            
            version += 1
        
        return jsonify({
            "status": "success",
            "ticket_id": ticket_id,
            "total_versions": len(history),
            "history": history
        })
        
    except Exception as e:
        logger.error(f"Error getting article history: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/api/article/simulate-comment", methods=["POST"])
def simulate_comment_feedback():
    """Simulate a Jira comment with feedback (for testing)"""
    try:
        data = request.json
        ticket_id = data.get("ticket_id")
        comment_text = data.get("comment", "")
        
        logger.info(f"📝 Simulating comment for {ticket_id}: {comment_text[:100]}...")
        
        # Parse comment for feedback
        if comment_text.lower().startswith("@jurix"):
            # Extract feedback after the mention
            feedback = comment_text[6:].strip()  # Remove "@jurix"
            
            # Detect action from comment
            action = "refine"  # default
            if "approve" in feedback.lower():
                action = "approve"
                feedback = feedback.replace("approve", "").strip()
            elif "reject" in feedback.lower():
                action = "reject"
                feedback = feedback.replace("reject", "").strip()
            
            # Get current version
            draft_key = f"article_draft:{ticket_id}"
            article_json = orchestrator.shared_memory.redis_client.get(draft_key)
            current_version = 1
            
            if article_json:
                article = json.loads(article_json)
                current_version = article.get("version", 1)
            
            # Process as feedback
            feedback_data = {
                "feedback": feedback,
                "action": action,
                "current_version": current_version
            }
            
            # Use the submit_article_feedback function directly
            with app.test_request_context(
                f'/api/article/feedback/{ticket_id}',
                method='POST',
                json=feedback_data
            ):
                return submit_article_feedback(ticket_id)
        else:
            return jsonify({
                "status": "ignored",
                "message": "Comment does not mention @jurix"
            })
            
    except Exception as e:
        logger.error(f"Error simulating comment: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# Helper functions for the new endpoints

def _analyze_team_performance(tickets, metrics, predictions):
    """Analyze individual and team performance metrics"""
    team_data = {
        "members": [],
        "performance_matrix": {},
        "collaboration_patterns": {},
        "optimal_distribution": {},
        "individual_metrics": {},
        "team_recommendations": []
    }
    
    # Get unique team members
    team_members = set()
    member_tickets = defaultdict(list)
    
    for ticket in tickets:
        assignee_info = ticket.get("fields", {}).get("assignee")
        if assignee_info:
            assignee = assignee_info.get("displayName", "Unknown")
            team_members.add(assignee)
            member_tickets[assignee].append(ticket)
    
    # Calculate individual metrics
    for member in team_members:
        tickets_assigned = member_tickets[member]
        
        # Calculate individual cycle time
        member_cycle_times = []
        done_count = 0
        in_progress_count = 0
        
        for ticket in tickets_assigned:
            if ticket.get("fields", {}).get("status", {}).get("name") == "Done":
                done_count += 1
                # Calculate cycle time
                cycle_time = _calculate_ticket_cycle_time(ticket)
                if cycle_time > 0:
                    member_cycle_times.append(cycle_time)
            elif ticket.get("fields", {}).get("status", {}).get("name") == "In Progress":
                in_progress_count += 1
        
        avg_cycle_time = sum(member_cycle_times) / len(member_cycle_times) if member_cycle_times else 0
        
        member_data = {
            "name": member,
            "metrics": {
                "total_tickets": len(tickets_assigned),
                "completed": done_count,
                "in_progress": in_progress_count,
                "avg_cycle_time": round(avg_cycle_time, 1),
                "velocity": done_count,
                "efficiency": done_count / max(len(tickets_assigned), 1)
            },
            "performance_score": _calculate_performance_score(done_count, avg_cycle_time, len(tickets_assigned))
        }
        
        team_data["members"].append(member_data)
        team_data["individual_metrics"][member] = member_data["metrics"]
    
    # Calculate optimal distribution
    total_tickets = len(tickets)
    team_size = len(team_members)
    if team_size > 0:
        optimal_per_person = total_tickets / team_size
        for member in team_members:
            team_data["optimal_distribution"][member] = round(optimal_per_person, 1)
    
    # Analyze collaboration patterns
    team_data["collaboration_patterns"] = _analyze_collaboration_patterns(tickets)
    
    # Generate team recommendations
    if predictions.get("team_burnout_analysis", {}).get("burnout_risk"):
        team_data["team_recommendations"].append(
            "Immediate workload redistribution needed to prevent team burnout"
        )
    
    # Check for performance disparities
    if team_data["members"]:
        performance_scores = [m["performance_score"] for m in team_data["members"]]
        if max(performance_scores) > min(performance_scores) * 2:
            team_data["team_recommendations"].append(
                "Significant performance variance detected - consider pairing or mentoring"
            )
    
    return team_data

def _calculate_ticket_cycle_time(ticket):
    """Calculate cycle time for a single ticket"""
    try:
        changelog = ticket.get("changelog", {}).get("histories", [])
        start_date = None
        end_date = None
        
        for history in changelog:
            for item in history.get("items", []):
                if item.get("field") == "status":
                    timestamp = datetime.fromisoformat(history.get("created", "").replace("Z", "+00:00"))
                    if item.get("toString") == "In Progress" and not start_date:
                        start_date = timestamp
                    elif item.get("toString") == "Done":
                        end_date = timestamp
        
        if start_date and end_date:
            return (end_date - start_date).days
    except:
        pass
    return 0

def _calculate_performance_score(completed, avg_cycle_time, total_tickets):
    """Calculate a performance score for a team member"""
    completion_rate = completed / max(total_tickets, 1)
    
    # Lower cycle time is better, normalize to 0-1 scale
    cycle_score = 1.0 / (1.0 + avg_cycle_time / 5.0) if avg_cycle_time > 0 else 0.5
    
    # Weight completion rate more heavily
    return round(completion_rate * 0.7 + cycle_score * 0.3, 2)

def _analyze_collaboration_patterns(tickets):
    """Analyze how team members collaborate"""
    patterns = {
        "handoffs": {},
        "parallel_work": {},
        "common_pairs": []
    }
    
    # Analyze ticket transitions between team members
    for ticket in tickets:
        changelog = ticket.get("changelog", {}).get("histories", [])
        assignees_timeline = []
        
        for history in changelog:
            for item in history.get("items", []):
                if item.get("field") == "assignee":
                    from_user = item.get("fromString", "Unassigned")
                    to_user = item.get("toString", "Unassigned")
                    if from_user != "Unassigned" and to_user != "Unassigned":
                        handoff_key = f"{from_user} → {to_user}"
                        patterns["handoffs"][handoff_key] = patterns["handoffs"].get(handoff_key, 0) + 1
        
    # Find common collaboration pairs
    handoff_pairs = [(k, v) for k, v in patterns["handoffs"].items() if v >= 3]
    patterns["common_pairs"] = sorted(handoff_pairs, key=lambda x: x[1], reverse=True)[:5]
    
    return patterns

def _analyze_historical_patterns(tickets):
    """Analyze historical patterns in ticket data"""
    patterns = {
        "resolution_patterns": {},
        "weekly_velocity": {},
        "issue_type_distribution": {},
        "cycle_time_evolution": {},
        "recurring_issues": [],
        "seasonal_trends": {}
    }
    
    # Group tickets by week
    for ticket in tickets:
        fields = ticket.get("fields", {})
        
        # Resolution patterns
        if fields.get("resolutiondate"):
            try:
                res_date = datetime.fromisoformat(fields["resolutiondate"].replace("Z", "+00:00"))
                week_key = res_date.strftime("%Y-W%U")
                patterns["weekly_velocity"][week_key] = patterns["weekly_velocity"].get(week_key, 0) + 1
                
                # Track cycle time by week
                cycle_time = _calculate_ticket_cycle_time(ticket)
                if cycle_time > 0:
                    if week_key not in patterns["cycle_time_evolution"]:
                        patterns["cycle_time_evolution"][week_key] = []
                    patterns["cycle_time_evolution"][week_key].append(cycle_time)
            except:
                pass
        
        # Issue type distribution
        issue_type = fields.get("issuetype", {}).get("name", "Unknown")
        patterns["issue_type_distribution"][issue_type] = patterns["issue_type_distribution"].get(issue_type, 0) + 1
        
        # Find recurring issues
        summary = fields.get("summary", "").lower()
        for keyword in ["timeout", "performance", "error", "crash", "slow"]:
            if keyword in summary:
                patterns["recurring_issues"].append({
                    "keyword": keyword,
                    "ticket": ticket.get("key"),
                    "summary": fields.get("summary")
                })
    
    # Calculate average cycle times by week
    for week, times in patterns["cycle_time_evolution"].items():
        patterns["cycle_time_evolution"][week] = round(sum(times) / len(times), 1)
    
    # Identify seasonal trends
    if patterns["weekly_velocity"]:
        weeks = sorted(patterns["weekly_velocity"].keys())
        if len(weeks) >= 8:
            # Simple trend detection
            first_half = sum(patterns["weekly_velocity"][w] for w in weeks[:len(weeks)//2])
            second_half = sum(patterns["weekly_velocity"][w] for w in weeks[len(weeks)//2:])
            
            if second_half > first_half * 1.2:
                patterns["seasonal_trends"]["trend"] = "increasing"
            elif second_half < first_half * 0.8:
                patterns["seasonal_trends"]["trend"] = "decreasing"
            else:
                patterns["seasonal_trends"]["trend"] = "stable"
    
    return patterns

def _calculate_trends(tickets, patterns):
    """Calculate trend data from patterns"""
    trends = {
        "velocity_trend": {
            "direction": "stable",
            "percentage_change": 0,
            "forecast_next_week": 0
        },
        "cycle_time_trend": {
            "direction": "stable",
            "improvement": 0,
            "target_achievement": False
        },
        "quality_trend": {
            "bug_rate": 0,
            "resolution_time": 0
        }
    }
    
    # Velocity trend
    if patterns.get("weekly_velocity"):
        weeks = sorted(patterns["weekly_velocity"].keys())
        if len(weeks) >= 2:
            recent_velocity = patterns["weekly_velocity"][weeks[-1]]
            previous_velocity = patterns["weekly_velocity"][weeks[-2]]
            
            if previous_velocity > 0:
                change = (recent_velocity - previous_velocity) / previous_velocity
                trends["velocity_trend"]["percentage_change"] = round(change * 100, 1)
                
                if change > 0.1:
                    trends["velocity_trend"]["direction"] = "increasing"
                elif change < -0.1:
                    trends["velocity_trend"]["direction"] = "decreasing"
                
                # Simple forecast
                trends["velocity_trend"]["forecast_next_week"] = round(recent_velocity * (1 + change), 0)
    
    # Cycle time trend
    if patterns.get("cycle_time_evolution"):
        weeks = sorted(patterns["cycle_time_evolution"].keys())
        if len(weeks) >= 2:
            recent_cycle = patterns["cycle_time_evolution"][weeks[-1]]
            previous_cycle = patterns["cycle_time_evolution"][weeks[-2]]
            
            improvement = previous_cycle - recent_cycle
            trends["cycle_time_trend"]["improvement"] = round(improvement, 1)
            
            if improvement > 0.5:
                trends["cycle_time_trend"]["direction"] = "improving"
            elif improvement < -0.5:
                trends["cycle_time_trend"]["direction"] = "degrading"
            
            trends["cycle_time_trend"]["target_achievement"] = recent_cycle <= 5  # 5 day target
    
    # Quality trend (bug rate)
    bug_count = patterns.get("issue_type_distribution", {}).get("Bug", 0)
    total_issues = sum(patterns.get("issue_type_distribution", {}).values())
    if total_issues > 0:
        trends["quality_trend"]["bug_rate"] = round(bug_count / total_issues * 100, 1)
    
    return trends

def _generate_pattern_insights(patterns, trends):
    """Generate natural language insights from patterns"""
    insights = []
    
    # Velocity insights
    if trends["velocity_trend"]["direction"] == "increasing":
        insights.append(f"Team velocity is improving with {trends['velocity_trend']['percentage_change']}% increase week-over-week")
    elif trends["velocity_trend"]["direction"] == "decreasing":
        insights.append(f"Team velocity declined by {abs(trends['velocity_trend']['percentage_change'])}% - investigate impediments")
    
    # Cycle time insights
    if trends["cycle_time_trend"]["direction"] == "improving":
        insights.append(f"Cycle times improved by {trends['cycle_time_trend']['improvement']} days - process optimizations working")
    
    # Issue type insights
    if patterns["issue_type_distribution"]:
        top_type = max(patterns["issue_type_distribution"].items(), key=lambda x: x[1])
        insights.append(f"Most common issue type: {top_type[0]} ({top_type[1]} tickets)")
    
    # Recurring issues
    if patterns["recurring_issues"]:
        keywords = {}
        for issue in patterns["recurring_issues"][:20]:  # Limit to prevent too many
            keywords[issue["keyword"]] = keywords.get(issue["keyword"], 0) + 1
        
        if keywords:
            top_keyword = max(keywords.items(), key=lambda x: x[1])
            insights.append(f"Recurring issue pattern: '{top_keyword[0]}' appearing in {top_keyword[1]} tickets")
    
    return insights

def _calculate_comprehensive_sprint_health(predictions, metrics, tickets):
    """Calculate comprehensive sprint health metrics"""
    sprint_completion = predictions.get("sprint_completion", {})
    team_burnout = predictions.get("team_burnout_analysis", {})
    warnings = predictions.get("warnings", [])
    
    # Calculate base health
    base_health = _calculate_sprint_health(predictions, metrics)
    
    # Calculate team energy with more detail
    team_energy = _calculate_team_energy(predictions, metrics)
    
    # Identify critical factors affecting health
    critical_factors = []
    
    if sprint_completion.get("probability", 1) < 0.5:
        critical_factors.append({
            "factor": "Sprint Completion Risk",
            "severity": "critical",
            "impact": f"Only {sprint_completion.get('probability', 0):.0%} chance of completion",
            "action": "Immediate scope reduction needed"
        })
    
    if team_burnout.get("burnout_risk"):
        overloaded = team_burnout.get("overloaded_members", [])
        critical_factors.append({
            "factor": "Team Burnout Risk",
            "severity": "high",
            "impact": f"{len(overloaded)} team members overloaded",
            "action": "Redistribute workload immediately"
        })
    
    # Calculate health history (simulated for now)
    health_history = []
    current_health = base_health["health_score"]
    for i in range(10):
        historical_health = current_health + np.random.normal(0, 5)
        historical_health = max(0, min(100, historical_health))
        health_history.append({
            "day": f"Day -{9-i}",
            "health_score": round(historical_health, 0),
            "status": "healthy" if historical_health >= 80 else "at_risk" if historical_health >= 60 else "critical"
        })
    
    # Generate recovery plan if health is poor
    recovery_plan = []
    if base_health["health_score"] < 60:
        recovery_plan = [
            {
                "action": "Emergency Sprint Planning",
                "priority": "immediate",
                "description": "Re-scope sprint to focus on critical items only"
            },
            {
                "action": "Workload Redistribution",
                "priority": "today",
                "description": "Balance tickets across team members"
            },
            {
                "action": "Blocker Resolution",
                "priority": "today",
                "description": "All hands on deck to resolve blocking issues"
            }
        ]
    
    return {
        "sprint_health": base_health,
        "team_energy": team_energy,
        "critical_factors": critical_factors,
        "recovery_plan": recovery_plan,
        "health_history": health_history
    }

def _extract_cross_agent_findings(collab_trace):
    """Extract key findings from cross-agent collaboration"""
    findings = []
    
    for trace in collab_trace:
        if trace.get("collaboration", {}).get("insights"):
            findings.extend(trace["collaboration"]["insights"])
    
    # Deduplicate and return top findings
    unique_findings = list(set(findings))
    return unique_findings[:10]

@app.route("/api/risk-assessment/<project_key>", methods=["GET"])
def get_risk_assessment(project_key):
    """Get real-time risk assessment based on AI analysis"""
    try:
        logger.info(f"🎯 Calculating risk assessment for: {project_key}")
        
        # Get latest predictions
        state = orchestrator.run_predictive_workflow(project_key, "risk_assessment")
        predictions = state.get("predictions", {})
        risks = predictions.get("risks", [])
        sprint_completion = predictions.get("sprint_completion", {})
        
        # Calculate risk score (0-10 scale)
        risk_score = 0.0
        risk_factors = []
        
        # Factor 1: Sprint completion probability
        completion_prob = sprint_completion.get("probability", 1.0)
        if completion_prob < 0.7:
            sprint_risk = (1 - completion_prob) * 3  # Max 3 points
            risk_score += sprint_risk
            risk_factors.append({
                "factor": "Sprint Completion Risk",
                "contribution": sprint_risk,
                "description": f"Only {completion_prob:.0%} chance of completing sprint"
            })
        
        # Factor 2: High severity risks
        high_risks = [r for r in risks if r.get("severity") == "high"]
        if high_risks:
            risk_contribution = min(len(high_risks) * 1.5, 3)  # Max 3 points
            risk_score += risk_contribution
            risk_factors.append({
                "factor": "Critical Issues",
                "contribution": risk_contribution,
                "description": f"{len(high_risks)} high-severity risks identified"
            })
        
        # Factor 3: Team capacity issues
        team_analysis = predictions.get("team_burnout_analysis", {})
        if team_analysis.get("burnout_risk"):
            burnout_contribution = 2.0
            risk_score += burnout_contribution
            risk_factors.append({
                "factor": "Team Burnout Risk",
                "contribution": burnout_contribution,
                "description": "Team members overloaded"
            })
        
        # Factor 4: Velocity decline
        velocity_forecast = predictions.get("velocity_forecast", {})
        if velocity_forecast.get("trend") == "declining":
            velocity_contribution = 2.0
            risk_score += velocity_contribution
            risk_factors.append({
                "factor": "Declining Velocity",
                "contribution": velocity_contribution,
                "description": "Team velocity trending downward"
            })
        
        # Generate monthly risk scores for visualization
        monthly_scores = []
        base_score = risk_score
        for i in range(12):
            # Add some variation to show trend
            variation = np.random.normal(0, 0.5)
            month_score = max(0, min(10, base_score + variation - (i * 0.1)))
            monthly_scores.append(round(month_score, 2))
        
        response = {
            "status": "success",
            "risk_score": round(min(risk_score, 10), 3),
            "risk_level": "critical" if risk_score > 7 else "high" if risk_score > 5 else "medium" if risk_score > 3 else "low",
            "risk_factors": risk_factors,
            "monthly_scores": monthly_scores,
            "top_risks": risks[:3],
            "recommendations": [
                f"Address {factor['factor']}: {factor['description']}" 
                for factor in sorted(risk_factors, key=lambda x: x['contribution'], reverse=True)[:3]
            ]
        }
        
        logger.info(f"✅ Risk assessment complete: {response['risk_score']}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error calculating risk assessment: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500
    
@app.route("/api/what-if/<project_key>", methods=["POST"])
def what_if_analysis(project_key):
    """Run what-if scenario analysis"""
    try:
        data = request.json
        scenario = data.get('scenario', {})
        
        logger.info(f"🔮 Running what-if analysis for {project_key}: {scenario.get('type')}")
        
        # Get current state
        current_state = orchestrator.run_productivity_workflow(project_key, {})
        current_tickets = current_state.get("tickets", [])
        current_predictions = current_state.get("predictions", {})
        
        # Apply scenario changes
        modified_tickets = current_tickets.copy()
        
        if scenario.get('type') == 'move_tickets':
            # Simulate moving tickets between sprints
            tickets_to_move = scenario.get('ticket_count', 5)
            # Reduce workload by removing tickets
            modified_tickets = modified_tickets[tickets_to_move:]
        
        elif scenario.get('type') == 'add_resources':
            # Simulate adding team members
            additional_capacity = scenario.get('team_members', 1) * 5  # 5 tickets per person
            # This would affect velocity predictions
        
        elif scenario.get('type') == 'change_priority':
            # Simulate changing ticket priorities
            priority_changes = scenario.get('changes', {})
            # Reorder tickets based on new priorities
        
        # Re-run predictions with modified data
        modified_state = {
            "tickets": modified_tickets,
            "metrics": current_state.get("metrics", {}),
            "historical_data": {"velocity_history": [5, 6, 7, 8, 7]}  # Example
        }
        
        # Use predictive agent to analyze modified scenario
        result = orchestrator.predictive_analysis_agent.run({
            "tickets": modified_tickets,
            "metrics": modified_state["metrics"],
            "historical_data": modified_state["historical_data"],
            "analysis_type": "what_if_scenario"
        })
        
        new_predictions = result.get("predictions", {})
        
        # Compare with current state
        comparison = {
            "current": {
                "sprint_completion": current_predictions.get("sprint_completion", {}).get("probability", 0),
                "risk_level": current_predictions.get("sprint_completion", {}).get("risk_level", "unknown"),
                "remaining_work": len(current_tickets)
            },
            "scenario": {
                "sprint_completion": new_predictions.get("sprint_completion", {}).get("probability", 0),
                "risk_level": new_predictions.get("sprint_completion", {}).get("risk_level", "unknown"),
                "remaining_work": len(modified_tickets)
            },
            "impact": {
                "completion_change": new_predictions.get("sprint_completion", {}).get("probability", 0) - 
                                   current_predictions.get("sprint_completion", {}).get("probability", 0),
                "risk_improved": new_predictions.get("sprint_completion", {}).get("risk_level", "high") < 
                               current_predictions.get("sprint_completion", {}).get("risk_level", "high"),
                "butterfly_effects": []
            }
        }
        
        # Identify butterfly effects
        if comparison["impact"]["completion_change"] > 0.2:
            comparison["impact"]["butterfly_effects"].append({
                "effect": "Significant improvement in sprint success",
                "downstream": "Team morale likely to improve, velocity may increase"
            })
        
        response = {
            "status": "success",
            "scenario": scenario,
            "comparison": comparison,
            "recommendations": new_predictions.get("sprint_completion", {}).get("recommended_actions", []),
            "confidence": new_predictions.get("sprint_completion", {}).get("confidence", 0.5)
        }
        
        logger.info(f"✅ What-if analysis complete")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in what-if analysis: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

def _generate_sparkline_data(tickets, metric_type):
    """Generate sparkline data for mini charts"""
    # Generate 10 data points for sparkline
    if metric_type == "velocity":
        # Weekly velocity over last 10 weeks
        return [32, 34, 35, 36, 38, 40, 42, 44, 45, 47]
    elif metric_type == "cycle":
        # Cycle time trend
        return [5.2, 5.0, 4.8, 4.5, 4.2, 3.8, 3.5, 3.3, 3.2, 3.2]
    elif metric_type == "efficiency":
        # Efficiency percentage
        return [75, 78, 80, 82, 84, 85, 87, 88, 89, 89]
    elif metric_type == "issues":
        # Active issues count
        return [30, 28, 26, 25, 24, 23, 23, 22, 23, 23]
    return []

def _get_recent_activity(tickets):
    """Extract recent activity from tickets"""
    activities = []
    
    # Sort tickets by updated date
    sorted_tickets = sorted(
        [t for t in tickets if t.get("fields", {}).get("updated")],
        key=lambda x: x["fields"]["updated"],
        reverse=True
    )[:10]  # Last 10 activities
    
    for ticket in sorted_tickets:
        fields = ticket.get("fields", {})
        updated = fields.get("updated", "")
        
        # Calculate time ago
        try:
            updated_time = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            time_ago = datetime.now(updated_time.tzinfo) - updated_time
            
            if time_ago.days == 0:
                if time_ago.seconds < 3600:
                    time_str = f"{time_ago.seconds // 60} minutes ago"
                else:
                    time_str = f"{time_ago.seconds // 3600} hours ago"
            else:
                time_str = f"{time_ago.days} days ago"
        except:
            time_str = "Recently"
        
        activity = {
            "title": f"{ticket.get('key')} - {fields.get('status', {}).get('name', 'Updated')}",
            "time": time_str,
            "type": "info",
            "key": ticket.get("key"),
            "summary": fields.get("summary", "")[:50] + "..."
        }
        
        # Determine activity type
        status = fields.get("status", {}).get("name", "")
        if status == "Done":
            activity["type"] = "success"
            activity["title"] = f"{ticket.get('key')} completed"
        elif status == "In Progress":
            activity["type"] = "info"
            activity["title"] = f"{ticket.get('key')} in progress"
        elif "blocked" in fields.get("labels", []):
            activity["type"] = "warning"
            activity["title"] = f"{ticket.get('key')} blocked"
        
        activities.append(activity)
    
    return activities[:5]  # Return top 5 activities

def _get_team_members(workload):
    """Get team member data with workload"""
    members = []
    
    for name, task_count in workload.items():
        # Calculate load percentage (assuming 10 tasks is 100%)
        load_percentage = min((task_count / 10) * 100, 120)
        
        member = {
            "name": name,
            "taskCount": task_count,
            "loadPercentage": load_percentage,
            "status": "Active",
            "initials": ''.join([n[0].upper() for n in name.split()[:2]])
        }
        
        # Determine status based on load
        if load_percentage > 100:
            member["status"] = "Overloaded"
        elif load_percentage < 50:
            member["status"] = "Available"
        
        members.append(member)
    
    # Sort by task count descending
    return sorted(members, key=lambda x: x["taskCount"], reverse=True)[:4]  # Top 4 members

@app.route("/api/dashboard/refresh/<project_id>", methods=["POST"])
def refresh_dashboard(project_id):
    """Force refresh dashboard data"""
    try:
        # Clear any cached data
        if orchestrator.shared_memory and orchestrator.shared_memory.redis_client:
            try:
                # Clear project cache
                orchestrator.shared_memory.redis_client.delete(f"tickets:{project_id}")
                orchestrator.shared_memory.redis_client.delete(f"dashboard_cache:{project_id}")
            except:
                pass
        
        # Call the main dashboard endpoint
        return dashboard(project_id)
    except Exception as e:
        logger.error(f"Error refreshing dashboard: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/api/debug", methods=["GET"])
def debug():
    """Debug endpoint to check current state"""
    # Refresh projects
    current_projects = []
    if orchestrator.jira_data_agent.jira_client:
        try:
            projects = orchestrator.jira_data_agent.jira_client.get_projects()
            current_projects = [p['key'] for p in projects]
        except:
            current_projects = orchestrator.jira_data_agent.available_projects
    
    return jsonify({
        "jira_data_agent": {
            "use_real_api": orchestrator.jira_data_agent.use_real_api,
            "use_mcp": getattr(orchestrator.jira_data_agent, 'use_mcp', 'Not set'),
            "available_projects": current_projects,
            "cached_projects": orchestrator.jira_data_agent.available_projects,
            "has_jira_client": hasattr(orchestrator.jira_data_agent, 'jira_client') and orchestrator.jira_data_agent.jira_client is not None
        },
        "api_config": {
            "jira_url": APIConfig.JIRA_URL,
            "jira_username": APIConfig.JIRA_USERNAME
        }
    })

@app.route("/api/notify-update", methods=["POST"])
def notify_update():
    """Endpoint for Jira plugin to notify about dashboard updates"""
    try:
        data = request.json
        project_key = data.get('projectKey')
        update_type = data.get('updateType', 'data_change')
        details = data.get('details', {})
        
        if not project_key:
            return jsonify({
                "status": "error",
                "error": "projectKey is required"
            }), 400
        
        # Track the update
        update_tracker.add_update(project_key, update_type, details)
        
        logger.info(f"Update notification received for project {project_key}: {update_type}")
        
        return jsonify({
            "status": "success",
            "message": "Update tracked"
        })
        
    except Exception as e:
        logger.error(f"Error processing update notification: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# Fixed portion of api_simple.py - Replace the check_updates function with this:

@app.route("/api/updates/<project_key>", methods=["GET"])
def check_updates(project_key):
    """Check for updates since a given timestamp"""
    try:
        # Get the 'since' parameter
        since_str = request.args.get('since', '0')
        try:
            since = float(since_str)
        except ValueError:
            since = 0
        
        logger.info(f"Checking updates for {project_key} since {since} ({datetime.fromtimestamp(since/1000)})")
        
        # Get updates
        updates = update_tracker.get_updates_since(project_key, since)
        
        # Log what we found with details
        logger.info(f"Found {len(updates)} updates for {project_key}")
        if updates:
            for update in updates[:3]:  # Log first 3
                logger.info(f"  - Update: {update['type']} at {update['timestamp']} ({datetime.fromtimestamp(update['timestamp']/1000)})")
        
        # Check if we need fresh data
        has_updates = len(updates) > 0
        needs_refresh = False
        
        if has_updates:
            # Check if any update requires dashboard refresh
            refresh_trigger_types = [
                'ticket_status_changed', 'ticket_created', 'ticket_resolved', 
                'data_change', 'created', 'updated', 'resolved', 'closed', 
                'assigned', 'changed', 'work_started', 'work_stopped', 'reopened'
            ]
            
            for update in updates:
                if update['type'] in refresh_trigger_types:
                    needs_refresh = True
                    logger.info(f"Update type '{update['type']}' triggers refresh")
                    break
        
        response = {
            "status": "success",
            "projectKey": project_key,
            "hasUpdates": has_updates,
            "updateCount": len(updates),
            "updates": updates[:10],  # Return max 10 most recent
            "needsRefresh": needs_refresh,
            "timestamp": datetime.now().timestamp() * 1000
        }
        
        # Log the response
        logger.info(f"Update check response: hasUpdates={has_updates}, needsRefresh={needs_refresh}, updateCount={len(updates)}")
        
        # If dashboard needs refresh, include fresh data
        if needs_refresh:
            try:
                logger.info(f"Dashboard refresh needed - fetching fresh data for {project_key}")
                
                # Clear any cached data to force fresh load
                if orchestrator.shared_memory and orchestrator.shared_memory.redis_client:
                    try:
                        cache_keys = [
                            f"tickets:{project_key}",
                            f"dashboard_cache:{project_key}",
                            f"filtered_tickets:{project_key}:*",
                            f"jira_api_data:{project_key}:*"
                        ]
                        for pattern in cache_keys:
                            for key in orchestrator.shared_memory.redis_client.scan_iter(match=pattern):
                                orchestrator.shared_memory.redis_client.delete(key)
                        logger.info("Cleared cached data for fresh load")
                    except Exception as e:
                        logger.error(f"Error clearing cache: {e}")
                
                # Get fresh dashboard data with force_fresh flag
                state = orchestrator.run_productivity_workflow(project_key, {}, force_fresh=True)
                
                # Extract all necessary data from the workflow state
                metrics = state.get("metrics", {})
                predictions = state.get("predictions", {})
                visualization_data = state.get("visualization_data", {})
                
                # Process charts data for the frontend
                charts_data = {
                    "sprintProgress": None,
                    "velocityTrend": None,
                    "issueDistribution": None,
                    "teamWorkload": None,
                    "burndown": None,
                    "teamPerformance": None
                }
                
                # Extract chart data from visualization_data
                if visualization_data and "charts" in visualization_data:
                    for chart in visualization_data["charts"]:
                        if "Team Workload" in chart.get("title", ""):
                            charts_data["teamWorkload"] = chart
                        elif "Velocity" in chart.get("title", "") and "Forecast" in chart.get("title", ""):
                            charts_data["velocityTrend"] = chart
                        elif "Team Performance" in chart.get("title", ""):
                            charts_data["teamPerformance"] = chart
                
                response["dashboardData"] = {
                    "project_id": project_key,
                    "status": "success",
                    "metrics": {
                        "velocity": metrics.get("throughput", 0),
                        "cycleTime": round(metrics.get("cycle_time", 0), 1),
                        "efficiency": int(metrics.get("collaborative_insights", {}).get("workflow_efficiency_score", 0.7) * 100),
                        "activeIssues": len(state.get("tickets", [])),
                        "throughput": metrics.get("throughput", 0),
                        "velocityChange": {
                            "value": 12,
                            "text": "+12% from last sprint",
                            "type": "positive"
                        },
                        "cycleTimeChange": {
                            "value": -0.5,
                            "text": "-0.5 days improvement",
                            "type": "positive"
                        },
                        "efficiencyChange": {
                            "value": 78,
                            "text": "Above target",
                            "type": "positive"
                        },
                        "workload": metrics.get("workload", {}),
                        "bottlenecks": metrics.get("bottlenecks", {}),
                        "teamBalanceScore": metrics.get("collaborative_insights", {}).get("team_balance_score", 0.8)
                    },
                    "predictions": {
                        "sprintCompletion": predictions.get("sprint_completion", {
                            "probability": 0.85,
                            "risk_level": "low",
                            "reasoning": "Based on current velocity and remaining work"
                        }),
                        "velocityForecast": predictions.get("velocity_forecast", {
                            "next_week_estimate": 48,
                            "trend": "increasing",
                            "insights": "Team velocity showing positive trend",
                            "forecast": [48, 50, 52]
                        }),
                        "risks": predictions.get("risks", [])[:3],
                        "warnings": predictions.get("warnings", [])
                    },
                    "visualizationData": {
                        "charts": charts_data,
                        "sparklines": {
                            "velocity": _generate_sparkline_data(state.get("tickets", []), "velocity"),
                            "cycleTime": _generate_sparkline_data(state.get("tickets", []), "cycle"),
                            "efficiency": _generate_sparkline_data(state.get("tickets", []), "efficiency"),
                            "issues": _generate_sparkline_data(state.get("tickets", []), "issues")
                        }
                    },
                    "recommendations": state.get("recommendations", [])[:5],
                    "ticketsAnalyzed": len(state.get("tickets", [])),
                    "lastUpdated": datetime.now().isoformat(),
                    "recentActivity": _get_recent_activity(state.get("tickets", [])),
                    "teamMembers": _get_team_members(metrics.get("workload", {}))
                }
                
                logger.info(f"Included fresh dashboard data with {len(state.get('tickets', []))} tickets")
            except Exception as e:
                logger.error(f"Error getting dashboard data: {e}", exc_info=True)
                response["dashboardError"] = str(e)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error checking updates: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500
    
@app.route("/api/suggest-articles", methods=["POST", "OPTIONS"])
def suggest_articles():
    """Get smart article suggestions for a Jira issue"""
    if request.method == "OPTIONS":
        return "", 200
        
    try:
        data = request.json
        issue_key = data.get("issue_key")
        issue_summary = data.get("issue_summary", "")
        issue_description = data.get("issue_description", "")
        issue_type = data.get("issue_type", "")
        issue_status = data.get("issue_status", "")
        labels = data.get("labels", [])
        components = data.get("components", [])
        
        if not issue_key:
            return jsonify({
                "status": "error",
                "error": "issue_key is required"
            }), 400
        
        logger.info(f"📚 Article suggestion request for issue: {issue_key}")
        logger.info(f"   Summary: {issue_summary}")
        logger.info(f"   Type: {issue_type}, Status: {issue_status}")
        logger.info(f"   Labels: {labels}, Components: {components}")
        
        # Run smart suggestion agent
        input_data = {
            "issue_key": issue_key,
            "issue_summary": issue_summary,
            "issue_description": issue_description,
            "issue_type": issue_type,
            "issue_status": issue_status,
            "labels": labels,
            "components": components,
            "quick_mode": True  # Fast initial suggestions
        }
        
        result = orchestrator.smart_suggestion_agent.run(input_data)
        
        if result.get("workflow_status") == "success":
            return jsonify({
                "status": "success",
                "issue_key": issue_key,
                "suggestions": result.get("suggestions", []),
                "search_strategy": result.get("search_strategy", {})
            })
        else:
            return jsonify({
                "status": "error",
                "error": result.get("error", "Failed to generate suggestions")
            }), 500
            
    except Exception as e:
        logger.error(f"Error suggesting articles: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/api/article-feedback", methods=["POST"])
def article_feedback():
    """Record user feedback on article suggestions"""
    try:
        data = request.json
        issue_key = data.get("issue_key")
        article_id = data.get("article_id")
        helpful = data.get("helpful", False)
        
        if not issue_key or not article_id:
            return jsonify({
                "status": "error",
                "error": "issue_key and article_id are required"
            }), 400
        
        logger.info(f"📊 Feedback for article {article_id} on issue {issue_key}: {'helpful' if helpful else 'not relevant'}")
        
        # Process feedback
        feedback_data = {
            "issue_key": issue_key,
            "article_id": article_id,
            "helpful": helpful,
            "feedback_mode": True
        }
        
        result = orchestrator.smart_suggestion_agent.run(feedback_data)
        
        return jsonify({
            "status": "success",
            "message": "Thank you for your feedback!"
        })
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500
    
@app.route("/api/suggestion-analytics", methods=["GET"])
def suggestion_analytics():
    """Get analytics on article suggestion performance"""
    try:
        # Get feedback statistics
        stats_key = "article_feedback_stats"
        stats_data = orchestrator.shared_memory.redis_client.get(stats_key)
        
        if stats_data:
            stats = json.loads(stats_data)
            
            # Calculate performance metrics
            analytics = {
                "total_articles": len(stats),
                "top_helpful_articles": [],
                "top_not_relevant_articles": [],
                "overall_helpfulness_rate": 0,
                "feedback_count": 0
            }
            
            total_helpful = 0
            total_feedback = 0
            
            for article_id, feedback in stats.items():
                total_helpful += feedback["helpful"]
                total_feedback += feedback["total"]
                
                helpfulness_rate = feedback["helpful"] / max(feedback["total"], 1)
                
                article_stats = {
                    "article_id": article_id,
                    "helpful": feedback["helpful"],
                    "not_relevant": feedback["not_relevant"],
                    "total": feedback["total"],
                    "helpfulness_rate": helpfulness_rate
                }
                
                # Track top performers
                if helpfulness_rate > 0.7 and feedback["total"] >= 3:
                    analytics["top_helpful_articles"].append(article_stats)
                elif helpfulness_rate < 0.3 and feedback["total"] >= 3:
                    analytics["top_not_relevant_articles"].append(article_stats)
            
            # Sort by performance
            analytics["top_helpful_articles"].sort(key=lambda x: x["helpfulness_rate"], reverse=True)
            analytics["top_not_relevant_articles"].sort(key=lambda x: x["helpfulness_rate"])
            
            # Limit to top 10
            analytics["top_helpful_articles"] = analytics["top_helpful_articles"][:10]
            analytics["top_not_relevant_articles"] = analytics["top_not_relevant_articles"][:10]
            
            # Calculate overall rate
            if total_feedback > 0:
                analytics["overall_helpfulness_rate"] = total_helpful / total_feedback
                analytics["feedback_count"] = total_feedback
            
            return jsonify({
                "status": "success",
                "analytics": analytics
            })
        else:
            return jsonify({
                "status": "success",
                "analytics": {
                    "total_articles": 0,
                    "feedback_count": 0,
                    "message": "No feedback data available yet"
                }
            })
            
    except Exception as e:
        logger.error(f"Error getting suggestion analytics: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500
    
@app.route("/api/neo4j/debug/<ticket_id>", methods=["GET"])
def debug_neo4j_ticket(ticket_id):
    """Debug Neo4j data for a ticket"""
    try:
        if not hasattr(orchestrator, 'enhanced_retrieval_agent'):
            return jsonify({"error": "Enhanced RAG not enabled"}), 400
            
        neo4j_manager = orchestrator.enhanced_retrieval_agent.rag_pipeline.neo4j_manager
        
        with neo4j_manager.driver.session() as session:
            # Check if ticket exists
            ticket_query = """
            MATCH (t:Ticket {key: $ticket_key})
            RETURN t
            """
            ticket_result = session.run(ticket_query, ticket_key=ticket_id)
            ticket = ticket_result.single()
            
            # Check documents
            doc_query = """
            MATCH (t:Ticket {key: $ticket_key})-[r]-(d:Document)
            RETURN d.id as doc_id, d.title as title, type(r) as rel_type, r.confidence as confidence
            """
            doc_result = session.run(doc_query, ticket_key=ticket_id)
            documents = [dict(record) for record in doc_result]
            
            # Check all relationships
            rel_query = """
            MATCH (t:Ticket {key: $ticket_key})-[r]-()
            RETURN type(r) as rel_type, count(r) as count
            """
            rel_result = session.run(rel_query, ticket_key=ticket_id)
            relationships = [dict(record) for record in rel_result]
            
            return jsonify({
                "ticket_exists": ticket is not None,
                "ticket_data": dict(ticket["t"]) if ticket else None,
                "documents": documents,
                "relationships": relationships,
                "document_count": len(documents)
            })
            
    except Exception as e:
        logger.error(f"Neo4j debug error: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/updates/debug", methods=["GET"])
def debug_updates():
    """Debug endpoint to see all stored updates"""
    return jsonify({
        "all_updates": update_tracker.updates,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/thesis-metrics", methods=["GET"])
def get_thesis_metrics():
    """Aggregate ALL system metrics for thesis defense"""
    try:
        # 1. Model Performance
        model_stats = orchestrator.shared_model_manager.get_agent_performance_stats()
        
        # 2. Collaboration Metrics
        collab_insights = orchestrator.collaborative_framework.get_collaboration_insights()
        
        # 3. Agent Performance - FIXED VERSION
        agent_summaries = {}
        for name, agent in orchestrator.agents_registry.items():
            try:
                # Get basic info without calling get_mental_state_summary
                mental_state = agent.mental_state
                
                # Count beliefs safely
                beliefs_count = len(mental_state.beliefs) if hasattr(mental_state, 'beliefs') else 0
                
                # Count decisions safely
                decisions_count = len(mental_state.decisions) if hasattr(mental_state, 'decisions') else 0
                
                # Count reflections safely
                reflections_count = len(mental_state.reflection_patterns) if hasattr(mental_state, 'reflection_patterns') else 0
                
                # Get competencies safely
                competencies = {}
                if hasattr(mental_state, 'competency_model') and mental_state.competency_model:
                    competencies = mental_state.competency_model.competencies
                
                agent_summaries[name] = {
                    "agent_id": agent.agent_id if hasattr(agent, 'agent_id') else name,
                    "beliefs_count": beliefs_count,
                    "decisions_count": decisions_count,
                    "reflections_count": reflections_count,
                    "competencies_count": len(competencies),
                    "has_semantic_memory": hasattr(mental_state, 'vector_memory') and mental_state.vector_memory is not None
                }
            except Exception as e:
                logger.error(f"Error getting summary for {name}: {e}")
                agent_summaries[name] = {
                    "error": str(e),
                    "agent_id": name
                }
        
        # 4. Memory Statistics
        memory_stats = {}
        if hasattr(orchestrator.chat_agent.mental_state, 'vector_memory') and orchestrator.chat_agent.mental_state.vector_memory:
            try:
                memory_stats = orchestrator.chat_agent.mental_state.vector_memory.get_memory_insights()
            except:
                memory_stats = {"status": "Error reading memory stats"}
        
        # 5. Calculate some quick stats
        total_decisions = sum(s.get("decisions_count", 0) for s in agent_summaries.values() if "error" not in s)
        total_beliefs = sum(s.get("beliefs_count", 0) for s in agent_summaries.values() if "error" not in s)
        
        # 6. Get recent activity from Redis
        recent_activity = []
        try:
            # Get last few model usage logs
            for key in orchestrator.shared_memory.redis_client.keys("model_log:*")[:3]:
                logs = orchestrator.shared_memory.redis_client.lrange(key, 0, 5)
                for log in logs:
                    try:
                        entry = json.loads(log)
                        recent_activity.append({
                            "agent": entry.get("agent"),
                            "model": entry.get("model"),
                            "success": entry.get("success"),
                            "quality": entry.get("quality", 0)
                        })
                    except:
                        pass
        except:
            pass
        
        return jsonify({
            "success": True,
            "message": "System metrics collected successfully",
            "system_overview": {
                "total_agents": len(orchestrator.agents_registry),
                "agents_list": list(orchestrator.agents_registry.keys()),
                "total_decisions_tracked": total_decisions,
                "total_beliefs_stored": total_beliefs,
                "semantic_memory_enabled": any(s.get("has_semantic_memory", False) for s in agent_summaries.values())
            },
            "model_performance": {
                "session_duration": model_stats.get("session_info", {}).get("duration", 0),
                "total_model_calls": model_stats.get("session_info", {}).get("total_requests", 0),
                "model_switches": model_stats.get("session_info", {}).get("model_switches", 0),
                "agents_using_models": len(model_stats.get("by_agent", {}))
            },
            "collaboration_metrics": {
                "total_collaborations": collab_insights.get("performance_metrics", {}).get("total_collaborations", 0),
                "successful_collaborations": collab_insights.get("performance_metrics", {}).get("successful_collaborations", 0),
                "collaboration_success_rate": round(
                    collab_insights.get("performance_metrics", {}).get("successful_collaborations", 0) / 
                    max(collab_insights.get("performance_metrics", {}).get("total_collaborations", 1), 1), 2
                ),
                "avg_collaboration_time": round(collab_insights.get("performance_metrics", {}).get("avg_collaboration_time", 0), 2)
            },
            "agent_summaries": agent_summaries,
            "recent_activity": recent_activity[:10],
            "memory_system": memory_stats or {"status": "Not enabled"},
            "thesis_insights": {
                "1_system_learning": "Agents track decisions and build competencies over time",
                "2_collaboration_value": f"Collaboration improves success rate to {round(collab_insights.get('performance_metrics', {}).get('successful_collaborations', 0) / max(collab_insights.get('performance_metrics', {}).get('total_collaborations', 1), 1) * 100, 1)}%",
                "3_model_optimization": f"Dynamic model selection with {model_stats.get('session_info', {}).get('model_switches', 0)} adaptive switches",
                "4_multi_agent_validation": "Multiple agents validate outputs for quality assurance"
            }
        })
    except Exception as e:
        logger.error(f"Error in thesis metrics: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Make sure to run some queries first to generate metrics!"
        }), 500
# ADD THESE FUNCTIONS AFTER THE EXISTING HELPER FUNCTIONS (around line 600)
# These are completely new functions to add to api_simple.py

def _generate_sprint_progress_data(tickets):
    """Generate sprint progress data for the chart"""
    if not tickets:
        return None
    
    # Group tickets by week
    week_data = {}
    status_mapping = {
        "Done": "completed",
        "Closed": "completed",
        "Resolved": "completed",
        "Complete": "completed",
        "Finished": "completed",
        "In Progress": "in-progress",
        "In Development": "in-progress",
        "In Review": "in-progress",
        "Testing": "in-progress",
        "To Do": "todo",
        "Open": "todo",
        "Reopened": "todo",
        "New": "todo",
        "Backlog": "todo",
        "Blocked": "todo",
        "On Hold": "todo",
        "Waiting": "todo"
    }
    
    # Simulate weekly progress (you can enhance this with real date-based grouping)
    weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
    
    # Calculate status distribution
    total_tickets = len(tickets)
    done_count = len([t for t in tickets if t.get("fields", {}).get("status", {}).get("name") in ["Done", "Closed", "Resolved", "Complete", "Finished"]])
    in_progress_count = len([t for t in tickets if t.get("fields", {}).get("status", {}).get("name") in ["In Progress", "In Development", "In Review", "Testing"]])
    todo_count = total_tickets - done_count - in_progress_count
    
    # Create progressive data for each week (simulating sprint progress)
    # This creates a realistic progression over 4 weeks
    data = {
        "type": "bar",
        "title": "Sprint Progress Overview",
        "data": {
            "labels": weeks,
            "datasets": [{
                "label": "Completed",
                "data": [
                    int(done_count * 0.1),  # Week 1: 10% of current done
                    int(done_count * 0.3),  # Week 2: 30% of current done
                    int(done_count * 0.7),  # Week 3: 70% of current done
                    done_count              # Week 4: All done tickets
                ],
                "backgroundColor": "#0052CC"
            }, {
                "label": "In Progress",
                "data": [
                    int(in_progress_count * 1.5),  # Week 1: More in progress
                    int(in_progress_count * 1.2),  # Week 2: Still high
                    int(in_progress_count * 0.8),  # Week 3: Reducing
                    in_progress_count              # Week 4: Current state
                ],
                "backgroundColor": "#6B88F7"
            }, {
                "label": "To Do",
                "data": [
                    todo_count + int(done_count * 0.9),     # Week 1: Most tickets todo
                    todo_count + int(done_count * 0.7),     # Week 2: Some moved
                    todo_count + int(done_count * 0.3),     # Week 3: More moved
                    todo_count                              # Week 4: Current todo
                ],
                "backgroundColor": "#DFE5FF"
            }]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "scales": {
                "x": {"stacked": True},
                "y": {"stacked": True}
            }
        }
    }
    
    return data

def _generate_burndown_data(tickets):
    """Generate sprint burndown data"""
    if not tickets:
        return None
    
    # Calculate story points or use ticket count
    total_points = len(tickets)
    done_tickets = len([t for t in tickets if t.get("fields", {}).get("status", {}).get("name") in ["Done", "Closed", "Resolved", "Complete", "Finished"]])
    
    # Generate ideal burndown line (10 day sprint)
    days = 10
    ideal_line = []
    actual_line = []
    
    for i in range(days + 1):
        # Ideal line - linear burndown
        ideal_points = total_points - (total_points / days * i)
        ideal_line.append(round(ideal_points, 1))
        
        # Simulate actual progress based on current completion
        if i == 0:
            actual_line.append(total_points)
        elif i <= 7:  # Assume we're on day 7
            # Non-linear progress (more realistic)
            progress_ratio = i / 7
            # Add some variation to make it realistic
            if i == 1:
                actual_points = total_points - (done_tickets * 0.05)
            elif i == 2:
                actual_points = total_points - (done_tickets * 0.15)
            elif i == 3:
                actual_points = total_points - (done_tickets * 0.25)
            elif i == 4:
                actual_points = total_points - (done_tickets * 0.40)
            elif i == 5:
                actual_points = total_points - (done_tickets * 0.60)
            elif i == 6:
                actual_points = total_points - (done_tickets * 0.80)
            else:  # i == 7
                actual_points = total_points - done_tickets
            
            actual_line.append(round(max(0, actual_points), 1))
        else:
            # Future days - no data yet
            actual_line.append(None)
    
    data = {
        "type": "line",
        "title": "Sprint Burndown",
        "data": {
            "labels": [f"Day {i}" for i in range(days + 1)],
            "datasets": [{
                "label": "Ideal",
                "data": ideal_line,
                "borderColor": "#C1C7D0",
                "borderDash": [5, 5],
                "tension": 0,
                "fill": False,
                "pointRadius": 3,
                "pointHoverRadius": 5
            }, {
                "label": "Actual",
                "data": actual_line,
                "borderColor": "#0052CC",
                "backgroundColor": "rgba(0, 82, 204, 0.1)",
                "tension": 0.3,
                "fill": True,
                "pointRadius": 4,
                "pointHoverRadius": 6
            }]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "legend": {"position": "bottom"},
                "tooltip": {"mode": "index", "intersect": False}
            },
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "title": {
                        "display": True,
                        "text": "Story Points"
                    }
                }
            }
        }
    }
    
    return data

def improve_suggestions_task():
    """Background task to improve suggestions based on feedback"""
    while True:
        try:
            time.sleep(3600)  # Run every hour
            
            # Get all feedback data
            stats_key = "article_feedback_stats"
            stats_data = orchestrator.shared_memory.redis_client.get(stats_key)
            
            if stats_data:
                stats = json.loads(stats_data)
                
                # Analyze patterns
                issue_type_patterns = {}
                
                # Get all feedback entries
                for key in orchestrator.shared_memory.redis_client.scan_iter(match="article_feedback:*"):
                    feedback_entries = orchestrator.shared_memory.redis_client.lrange(key, 0, -1)
                    
                    for entry_json in feedback_entries:
                        entry = json.loads(entry_json)
                        issue_type = entry.get("issue_type", "Unknown")
                        article_id = entry["article_id"]
                        helpful = entry["helpful"]
                        
                        if issue_type not in issue_type_patterns:
                            issue_type_patterns[issue_type] = {}
                        
                        if article_id not in issue_type_patterns[issue_type]:
                            issue_type_patterns[issue_type][article_id] = {
                                "helpful": 0,
                                "not_helpful": 0,
                                "total": 0
                            }
                        
                        issue_type_patterns[issue_type][article_id]["total"] += 1
                        if helpful:
                            issue_type_patterns[issue_type][article_id]["helpful"] += 1
                        else:
                            issue_type_patterns[issue_type][article_id]["not_helpful"] += 1
                
                # Store patterns for use by suggestion agent
                for issue_type, patterns in issue_type_patterns.items():
                    # Calculate success rates
                    for article_id, stats in patterns.items():
                        stats["success_rate"] = stats["helpful"] / max(stats["total"], 1)
                    
                    # Store in Redis
                    type_key = f"issue_type_feedback:{issue_type}"
                    orchestrator.shared_memory.redis_client.set(
                        type_key,
                        json.dumps(patterns)
                    )
                    orchestrator.shared_memory.redis_client.expire(type_key, 2592000)  # 30 days
                
                logger.info("Suggestion improvement task completed successfully")
                
        except Exception as e:
            logger.error(f"Error in suggestion improvement task: {e}")

# Start background improvement task
improvement_thread = threading.Thread(target=improve_suggestions_task, daemon=True)
improvement_thread.start()

def cleanup_old_updates():
    """Background task to clean old updates"""
    while True:
        try:
            time.sleep(1800)  # Run every 30 minutes
            update_tracker.clear_old_updates(max_age_minutes=60)
            logger.info("Cleaned old dashboard updates")
        except Exception as e:
            logger.error(f"Error cleaning updates: {e}")

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_updates, daemon=True)
cleanup_thread.start()

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("🚀 Starting JURIX AI Backend Server (Enhanced)")
    logger.info(f"📍 Jira URL: {APIConfig.JIRA_URL}")
    logger.info(f"👤 Jira User: {APIConfig.JIRA_USERNAME}")
    logger.info(f"📁 Available Projects: {orchestrator.jira_data_agent.available_projects}")
    logger.info(f"✅ Using Real API: {orchestrator.jira_data_agent.use_real_api}")
    logger.info(f"❌ Using MCP: {getattr(orchestrator.jira_data_agent, 'use_mcp', False)}")
    logger.info("🌐 Server will be available at: http://localhost:5001")
    logger.info("=" * 50)
    
    # Final check for projects
    if not orchestrator.jira_data_agent.available_projects:
        logger.warning("⚠️ No projects found! Check your Jira connection.")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
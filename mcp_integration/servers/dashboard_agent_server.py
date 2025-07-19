import json
from mcp_integration.servers.base_mcp_server import BaseMCPServer
from typing import Dict, Any, List
import asyncio

class DashboardAgentMCPServer(BaseMCPServer):
    """MCP Server for Productivity Dashboard Agent"""
    
    def _register_tools(self):
        """Register dashboard agent tools"""
        
        # Tool: Generate Dashboard
        async def generate_dashboard(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Generate productivity dashboard"""
            result = self.agent.run({
                "tickets": arguments.get("tickets", []),
                "recommendations": arguments.get("recommendations", []),
                "project_id": arguments.get("project_id"),
                "predictions": arguments.get("predictions", {})
            })
            
            return {
                "metrics": result.get("metrics", {}),
                "visualization_data": result.get("visualization_data", {}),
                "report": result.get("report", ""),
                "workflow_status": result.get("workflow_status", "success")
            }
        
        self.register_tool(
            name="generate_dashboard",
            description="Generate comprehensive productivity dashboard",
            input_schema={
                "type": "object",
                "properties": {
                    "tickets": {"type": "array"},
                    "recommendations": {"type": "array"},
                    "project_id": {"type": "string"},
                    "predictions": {"type": "object"}
                },
                "required": ["project_id"]
            },
            handler=generate_dashboard
        )
        
        # Tool: Analyze Metrics
        async def analyze_metrics(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Analyze productivity metrics"""
            tickets = arguments.get("tickets", [])
            
            # Calculate metrics
            metrics = self.agent._analyze_ticket_data(tickets)
            
            # Add analysis insights
            insights = []
            
            if metrics.get("cycle_time", 0) > 7:
                insights.append("High cycle time detected - process optimization recommended")
            
            if metrics.get("throughput", 0) < 5:
                insights.append("Low throughput - consider reviewing blockers")
            
            bottlenecks = metrics.get("bottlenecks", {})
            if any(count > 5 for count in bottlenecks.values()):
                insights.append("Workflow bottlenecks detected - immediate attention needed")
            
            return {
                "metrics": metrics,
                "insights": insights,
                "analysis_confidence": metrics.get("analysis_confidence", 0)
            }
        
        self.register_tool(
            name="analyze_metrics",
            description="Analyze productivity metrics from ticket data",
            input_schema={
                "type": "object",
                "properties": {
                    "tickets": {"type": "array"}
                },
                "required": ["tickets"]
            },
            handler=analyze_metrics
        )
        
        # Tool: Create Visualization
        async def create_visualization(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Create specific visualization"""
            viz_type = arguments.get("type", "bar")
            data = arguments.get("data", {})
            title = arguments.get("title", "")
            
            # Create visualization structure
            visualization = {
                "type": viz_type,
                "title": title,
                "data": data,
                "options": arguments.get("options", {})
            }
            
            # Add type-specific enhancements
            if viz_type == "gauge" and "value" in data:
                visualization["data"]["min"] = 0
                visualization["data"]["max"] = 100
                visualization["data"]["thresholds"] = {
                    "low": 30,
                    "medium": 60,
                    "high": 80
                }
            
            return visualization
        
        self.register_tool(
            name="create_visualization",
            description="Create custom visualization",
            input_schema={
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["bar", "line", "pie", "gauge", "radar"]},
                    "title": {"type": "string"},
                    "data": {"type": "object"},
                    "options": {"type": "object"}
                },
                "required": ["type", "data"]
            },
            handler=create_visualization
        )
        
        # Tool: Export Dashboard
        async def export_dashboard(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Export dashboard data"""
            dashboard_id = arguments.get("dashboard_id")
            format_type = arguments.get("format", "json")
            
            # Get dashboard data from Redis
            if dashboard_id:
                dashboard_data = self.redis_client.get(dashboard_id)
                if dashboard_data:
                    data = json.loads(dashboard_data)
                    
                    if format_type == "json":
                        return data
                    elif format_type == "summary":
                        return {
                            "project_id": data.get("project_id"),
                            "timestamp": data.get("timestamp"),
                            "metrics_summary": {
                                "throughput": data.get("metrics", {}).get("throughput"),
                                "cycle_time": data.get("metrics", {}).get("cycle_time"),
                                "bottlenecks": len(data.get("metrics", {}).get("bottlenecks", {}))
                            },
                            "recommendations_count": len(data.get("recommendations", [])),
                            "has_predictions": bool(data.get("predictions"))
                        }
            
            return {"error": "Dashboard not found"}
        
        self.register_tool(
            name="export_dashboard",
            description="Export dashboard data in various formats",
            input_schema={
                "type": "object",
                "properties": {
                    "dashboard_id": {"type": "string"},
                    "format": {"type": "string", "enum": ["json", "summary"]}
                },
                "required": ["dashboard_id"]
            },
            handler=export_dashboard
        )
    
    def _register_resources(self):
        """Register dashboard resources"""
        
        # Resource: Chart Templates
        self.register_resource(
            uri="dashboard://templates/charts",
            name="Chart Templates",
            description="Pre-configured chart templates",
            content={
                "velocity_trend": {
                    "type": "line",
                    "title": "Velocity Trend",
                    "options": {
                        "scales": {
                            "y": {"beginAtZero": True}
                        }
                    }
                },
                "workload_distribution": {
                    "type": "bar",
                    "title": "Team Workload Distribution",
                    "options": {
                        "indexAxis": "y"
                    }
                },
                "sprint_progress": {
                    "type": "gauge",
                    "title": "Sprint Progress",
                    "options": {
                        "needle": True,
                        "colors": ["#ff4444", "#ffaa00", "#00aa00"]
                    }
                }
            }
        )
        
        # Resource: Report Templates
        self.register_resource(
            uri="dashboard://templates/reports",
            name="Report Templates",
            description="Report generation templates",
            content={
                "executive_summary": "Executive Summary for {project_name}\n\nKey Metrics:\n- Throughput: {throughput}\n- Cycle Time: {cycle_time} days\n- Team Velocity: {velocity}\n\nRecommendations:\n{recommendations}\n\nRisks:\n{risks}",
                "team_performance": "Team Performance Report\n\nProductivity Metrics:\n{metrics}\n\nWorkload Distribution:\n{workload}\n\nBottlenecks Identified:\n{bottlenecks}",
                "sprint_review": "Sprint Review for {sprint_name}\n\nCompleted: {completed_tickets}\nRemaining: {remaining_tickets}\nVelocity: {velocity}\n\nHighlights:\n{highlights}\n\nChallenges:\n{challenges}"
            }
        )
        
        # Resource: Performance Benchmarks
        async def get_benchmarks():
            """Get performance benchmarks"""
            return {
                "industry_benchmarks": {
                    "cycle_time": {
                        "excellent": 3,
                        "good": 5,
                        "average": 7,
                        "poor": 10
                    },
                    "throughput_weekly": {
                        "excellent": 15,
                        "good": 10,
                        "average": 7,
                        "poor": 3
                    },
                    "velocity_stability": {
                        "excellent": 0.9,
                        "good": 0.8,
                        "average": 0.7,
                        "poor": 0.5
                    }
                }
            }
        
        self.register_resource(
            uri="dashboard://benchmarks/performance",
            name="Performance Benchmarks",
            description="Industry performance benchmarks",
            handler=get_benchmarks
        )
    
    def _get_agent_prompts(self) -> List[Dict[str, Any]]:
        """Get dashboard agent prompts"""
        return [
            {
                "name": "executive_dashboard",
                "description": "Generate executive dashboard summary",
                "template": "Create an executive dashboard for {project_name} focusing on {key_metrics}. Include high-level insights and strategic recommendations.",
                "arguments": [
                    {"name": "project_name", "description": "Project name", "required": True},
                    {"name": "key_metrics", "description": "Key metrics to highlight", "required": True}
                ]
            },
            {
                "name": "team_analytics",
                "description": "Analyze team performance",
                "template": "Generate detailed team analytics for {team_name} including workload distribution, productivity trends, and performance insights over {time_period}.",
                "arguments": [
                    {"name": "team_name", "description": "Team name", "required": True},
                    {"name": "time_period", "description": "Analysis time period", "required": True}
                ]
            }
        ]
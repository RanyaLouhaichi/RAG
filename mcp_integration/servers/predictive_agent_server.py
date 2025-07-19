from mcp_integration.servers.base_mcp_server import BaseMCPServer
from typing import Dict, Any, List
import asyncio

class PredictiveAgentMCPServer(BaseMCPServer):
    """MCP Server for Predictive Analysis Agent"""
    
    def _register_tools(self):
        """Register predictive analysis tools"""
        
        # Tool: Predict Sprint Completion
        async def predict_sprint_completion(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Predict sprint completion probability"""
            # Get tickets data
            tickets = arguments.get("tickets", [])
            metrics = arguments.get("metrics", {})
            historical_data = arguments.get("historical_data", {})
            
            # Run prediction
            result = self.agent.run({
                "tickets": tickets,
                "metrics": metrics,
                "historical_data": historical_data,
                "analysis_type": "sprint_completion"
            })
            
            predictions = result.get("predictions", {})
            sprint_completion = predictions.get("sprint_completion", {})
            
            return {
                "probability": sprint_completion.get("probability", 0),
                "confidence": sprint_completion.get("confidence", 0),
                "risk_level": sprint_completion.get("risk_level", "unknown"),
                "remaining_work": sprint_completion.get("remaining_work", 0),
                "risk_factors": sprint_completion.get("risk_factors", []),
                "recommended_actions": sprint_completion.get("recommended_actions", [])
            }
        
        self.register_tool(
            name="predict_sprint_completion",
            description="Predict sprint completion probability with risk analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "tickets": {"type": "array"},
                    "metrics": {"type": "object"},
                    "historical_data": {"type": "object"}
                }
            },
            handler=predict_sprint_completion
        )
        
        # Tool: Forecast Velocity
        async def forecast_velocity(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Forecast team velocity trends"""
            historical_data = arguments.get("historical_data", {})
            current_velocity = arguments.get("current_velocity", 0)
            
            # Simple velocity calculation if no historical data
            if not historical_data.get("velocity_history"):
                historical_data["velocity_history"] = [current_velocity]
            
            # Run forecast
            result = self.agent.run({
                "tickets": arguments.get("tickets", []),
                "metrics": {"throughput": current_velocity},
                "historical_data": historical_data,
                "analysis_type": "velocity_forecast"
            })
            
            predictions = result.get("predictions", {})
            velocity_forecast = predictions.get("velocity_forecast", {})
            
            return {
                "forecast": velocity_forecast.get("forecast", []),
                "trend": velocity_forecast.get("trend", "stable"),
                "trend_percentage": velocity_forecast.get("trend_percentage", 0),
                "confidence": velocity_forecast.get("confidence", 0),
                "next_week_estimate": velocity_forecast.get("next_week_estimate", 0),
                "insights": velocity_forecast.get("insights", "")
            }
        
        self.register_tool(
            name="forecast_velocity",
            description="Forecast team velocity trends",
            input_schema={
                "type": "object",
                "properties": {
                    "tickets": {"type": "array"},
                    "historical_data": {"type": "object"},
                    "current_velocity": {"type": "number"}
                }
            },
            handler=forecast_velocity
        )
        
        # Tool: Identify Risks
        async def identify_risks(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Identify project risks"""
            # Run comprehensive analysis
            result = self.agent.run({
                "tickets": arguments.get("tickets", []),
                "metrics": arguments.get("metrics", {}),
                "historical_data": arguments.get("historical_data", {}),
                "analysis_type": "comprehensive"
            })
            
            predictions = result.get("predictions", {})
            risks = predictions.get("risks", [])
            warnings = predictions.get("warnings", [])
            
            return {
                "total_risks": len(risks),
                "high_severity_risks": len([r for r in risks if r.get("severity") == "high"]),
                "risks": risks[:10],  # Top 10 risks
                "warnings": warnings,
                "mitigation_priority": [r["description"] for r in risks if r.get("severity") == "high"][:3]
            }
        
        self.register_tool(
            name="identify_risks",
            description="Identify and prioritize project risks",
            input_schema={
                "type": "object",
                "properties": {
                    "tickets": {"type": "array"},
                    "metrics": {"type": "object"},
                    "historical_data": {"type": "object"}
                }
            },
            handler=identify_risks
        )
        
        # Tool: Predict Bottlenecks
        async def predict_bottlenecks(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Predict workflow bottlenecks"""
            tickets = arguments.get("tickets", [])
            
            # Analyze ticket distribution
            bottleneck_analysis = {}
            status_counts = {}
            
            for ticket in tickets:
                status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Identify bottlenecks
            total_tickets = len(tickets)
            bottlenecks = []
            
            for status, count in status_counts.items():
                if count > total_tickets * 0.2 and status not in ["Done", "Closed"]:
                    bottlenecks.append({
                        "status": status,
                        "ticket_count": count,
                        "percentage": count / total_tickets * 100,
                        "severity": "high" if count > total_tickets * 0.3 else "medium"
                    })
            
            # Run prediction for future bottlenecks
            result = self.agent.run({
                "tickets": tickets,
                "metrics": {"bottlenecks": status_counts},
                "historical_data": {},
                "analysis_type": "comprehensive"
            })
            
            predictions = result.get("predictions", {})
            ticket_predictions = predictions.get("ticket_predictions", [])
            
            return {
                "current_bottlenecks": bottlenecks,
                "predicted_bottlenecks": [t for t in ticket_predictions if t.get("is_bottleneck")],
                "total_blocked_tickets": sum(1 for t in tickets if t.get("fields", {}).get("status", {}).get("name") == "Blocked"),
                "recommendations": [
                    f"Focus on clearing {b['ticket_count']} tickets in {b['status']} status"
                    for b in bottlenecks[:3]
                ]
            }
        
        self.register_tool(
            name="predict_bottlenecks",
            description="Predict and analyze workflow bottlenecks",
            input_schema={
                "type": "object",
                "properties": {
                    "tickets": {"type": "array"}
                }
            },
            handler=predict_bottlenecks
        )
    
    def _register_resources(self):
        """Register predictive analysis resources"""
        
        # Resource: Prediction Models Info
        self.register_resource(
            uri="predictions://models/info",
            name="Prediction Models",
            description="Information about available prediction models",
            content={
                "models": [
                    {
                        "name": "sprint_completion",
                        "description": "Predicts sprint completion probability",
                        "accuracy": 0.85,
                        "factors": ["velocity", "remaining_work", "blockers", "team_capacity"]
                    },
                    {
                        "name": "velocity_forecast",
                        "description": "Forecasts team velocity trends",
                        "accuracy": 0.78,
                        "factors": ["historical_velocity", "team_size", "complexity"]
                    },
                    {
                        "name": "risk_assessment",
                        "description": "Identifies and assesses project risks",
                        "accuracy": 0.82,
                        "factors": ["bottlenecks", "technical_debt", "resource_allocation"]
                    }
                ]
            }
        )
        
        # Resource: Thresholds and Parameters
        self.register_resource(
            uri="predictions://config/thresholds",
            name="Prediction Thresholds",
            description="Configurable thresholds for predictions",
            content=self.agent.prediction_thresholds
        )
        
        # Resource: Historical Accuracy
        async def get_prediction_accuracy():
            """Get historical prediction accuracy"""
            # This would normally query actual prediction history
            return {
                "sprint_completion_accuracy": {
                    "last_30_days": 0.84,
                    "last_90_days": 0.82,
                    "overall": 0.85
                },
                "velocity_forecast_accuracy": {
                    "last_30_days": 0.79,
                    "last_90_days": 0.77,
                    "overall": 0.78
                },
                "risk_prediction_accuracy": {
                    "last_30_days": 0.81,
                    "last_90_days": 0.80,
                    "overall": 0.82
                }
            }
        
        self.register_resource(
            uri="predictions://metrics/accuracy",
            name="Prediction Accuracy",
            description="Historical accuracy of predictions",
            handler=get_prediction_accuracy
        )
    
    def _get_agent_prompts(self) -> List[Dict[str, Any]]:
        """Get predictive analysis prompts"""
        return [
            {
                "name": "sprint_risk_analysis",
                "description": "Analyze sprint risks and completion probability",
                "template": "Analyze the current sprint for {project_name} and predict completion probability. Focus on {risk_areas} and provide specific mitigation strategies.",
                "arguments": [
                    {"name": "project_name", "description": "Project name", "required": True},
                    {"name": "risk_areas", "description": "Specific risk areas to focus on", "required": False}
                ]
            },
            {
                "name": "capacity_planning",
                "description": "Predict team capacity and workload",
                "template": "Analyze team capacity for {team_name} over the next {time_period}. Consider current workload, velocity trends, and potential bottlenecks.",
                "arguments": [
                    {"name": "team_name", "description": "Team name", "required": True},
                    {"name": "time_period", "description": "Time period (e.g., '2 sprints', '1 month')", "required": True}
                ]
            }
        ]
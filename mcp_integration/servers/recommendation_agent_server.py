import json
from mcp_integration.servers.base_mcp_server import BaseMCPServer
from typing import Dict, Any, List
import asyncio

class RecommendationAgentMCPServer(BaseMCPServer):
    """MCP Server for Recommendation Agent"""
    
    def _register_tools(self):
        """Register recommendation agent tools"""
        
        # Tool: Generate Recommendations
        async def generate_recommendations(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Generate strategic recommendations"""
            result = self.agent.run({
                "session_id": arguments.get("session_id"),
                "user_prompt": arguments.get("prompt"),
                "project": arguments.get("project"),
                "tickets": arguments.get("tickets", []),
                "articles": arguments.get("articles", []),
                "workflow_type": arguments.get("workflow_type", "general"),
                "intent": arguments.get("intent", {"intent": "recommendation"}),
                "predictions": arguments.get("predictions", {})
            })
            
            return {
                "recommendations": result.get("recommendations", []),
                "needs_context": result.get("needs_context", False),
                "collaboration_metadata": result.get("collaboration_metadata", {})
            }
        
        self.register_tool(
            name="generate_recommendations",
            description="Generate intelligent, context-aware recommendations",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "project": {"type": "string"},
                    "tickets": {"type": "array"},
                    "articles": {"type": "array"},
                    "workflow_type": {"type": "string"},
                    "predictions": {"type": "object"}
                },
                "required": ["prompt"]
            },
            handler=generate_recommendations
        )
        
        # Tool: Enhance Recommendations
        async def enhance_recommendations(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Enhance existing recommendations with additional context"""
            existing_recommendations = arguments.get("recommendations", [])
            additional_context = arguments.get("context", {})
            
            # Create enhanced prompt
            enhanced_prompt = f"Enhance these recommendations with the following context: {additional_context}"
            
            result = await generate_recommendations({
                "prompt": enhanced_prompt,
                "project": arguments.get("project"),
                "tickets": arguments.get("tickets", []),
                "articles": arguments.get("articles", [])
            })
            
            # Merge recommendations
            enhanced = existing_recommendations + result.get("recommendations", [])
            
            return {
                "original_count": len(existing_recommendations),
                "enhanced_count": len(enhanced),
                "recommendations": enhanced
            }
        
        self.register_tool(
            name="enhance_recommendations",
            description="Enhance existing recommendations with additional context",
            input_schema={
                "type": "object",
                "properties": {
                    "recommendations": {"type": "array"},
                    "context": {"type": "object"},
                    "project": {"type": "string"},
                    "tickets": {"type": "array"},
                    "articles": {"type": "array"}
                },
                "required": ["recommendations"]
            },
            handler=enhance_recommendations
        )
        
        # Tool: Validate Recommendations
        async def validate_recommendations(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Validate recommendations quality"""
            recommendations = arguments.get("recommendations", [])
            project_context = arguments.get("project_context", {})
            
            validation_results = []
            
            for rec in recommendations:
                # Simple validation logic
                is_specific = len(rec) > 50
                has_action = any(word in rec.lower() for word in ["implement", "create", "review", "schedule", "develop"])
                is_relevant = arguments.get("project") and arguments.get("project") in rec
                
                validation_results.append({
                    "recommendation": rec[:100] + "..." if len(rec) > 100 else rec,
                    "is_specific": is_specific,
                    "has_action": has_action,
                    "is_relevant": is_relevant,
                    "quality_score": (is_specific + has_action + is_relevant) / 3
                })
            
            avg_quality = sum(v["quality_score"] for v in validation_results) / len(validation_results) if validation_results else 0
            
            return {
                "total_recommendations": len(recommendations),
                "average_quality": avg_quality,
                "validation_results": validation_results,
                "needs_improvement": avg_quality < 0.7
            }
        
        self.register_tool(
            name="validate_recommendations",
            description="Validate the quality of recommendations",
            input_schema={
                "type": "object",
                "properties": {
                    "recommendations": {"type": "array"},
                    "project": {"type": "string"},
                    "project_context": {"type": "object"}
                },
                "required": ["recommendations"]
            },
            handler=validate_recommendations
        )
    
    def _register_resources(self):
        """Register recommendation resources"""
        
        # Resource: Recommendation Templates
        self.register_resource(
            uri="recommendations://templates/strategic",
            name="Strategic Recommendation Templates",
            description="Templates for different types of strategic recommendations",
            content={
                "process_improvement": [
                    "Implement {process} to improve {metric} by {percentage}%",
                    "Schedule regular {frequency} reviews of {area} to identify bottlenecks",
                    "Develop automated {tool} for {task} to reduce manual effort"
                ],
                "team_efficiency": [
                    "Redistribute workload from {overloaded_member} to balance team capacity",
                    "Introduce {methodology} practices to enhance collaboration",
                    "Create knowledge sharing sessions focused on {topic}"
                ],
                "technical_debt": [
                    "Allocate {percentage}% of sprint capacity to address technical debt",
                    "Prioritize refactoring of {component} to improve maintainability",
                    "Implement {testing_strategy} to prevent regression"
                ]
            }
        )
        
        # Resource: Quality Metrics
        async def get_quality_metrics():
            """Get recommendation quality metrics"""
            # Get recent recommendations from Redis
            recent_recs = []
            rec_keys = self.redis_client.keys("rec_*")
            
            for key in rec_keys[:20]:  # Last 20
                rec_data = self.redis_client.get(key)
                if rec_data:
                    recent_recs.append(json.loads(rec_data))
            
            return {
                "total_recommendations_generated": len(rec_keys),
                "recent_recommendations": len(recent_recs),
                "average_recommendations_per_request": 4.2,  # Example metric
                "quality_metrics": {
                    "specificity": 0.85,
                    "actionability": 0.92,
                    "relevance": 0.88
                }
            }
        
        self.register_resource(
            uri="recommendations://metrics/quality",
            name="Quality Metrics",
            description="Recommendation quality metrics and statistics",
            handler=get_quality_metrics
        )
    
    def _get_agent_prompts(self) -> List[Dict[str, Any]]:
        """Get recommendation agent prompts"""
        return [
            {
                "name": "process_optimization",
                "description": "Generate process optimization recommendations",
                "template": "Analyze the {process_area} process for {project_name} and provide 5 specific recommendations to improve efficiency, reduce bottlenecks, and enhance team productivity.",
                "arguments": [
                    {"name": "process_area", "description": "Area to optimize (e.g., development, testing)", "required": True},
                    {"name": "project_name", "description": "Project name", "required": True}
                ]
            },
            {
                "name": "risk_mitigation",
                "description": "Generate risk mitigation recommendations",
                "template": "Based on the identified risks in {project_name}, provide strategic recommendations to mitigate {risk_type} risks. Focus on preventive measures and contingency planning.",
                "arguments": [
                    {"name": "project_name", "description": "Project name", "required": True},
                    {"name": "risk_type", "description": "Type of risk (e.g., technical, schedule, resource)", "required": True}
                ]
            }
        ]
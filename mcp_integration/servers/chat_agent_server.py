from mcp_integration.servers.base_mcp_server import BaseMCPServer
from typing import Dict, Any, List
import asyncio

class ChatAgentMCPServer(BaseMCPServer):
    """MCP Server for Chat Agent"""
    
    def _register_tools(self):
        """Register chat agent tools"""
        
        # Tool: Generate Response
        async def generate_response(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Generate a conversational response"""
            result = self.agent.run({
                "session_id": arguments.get("session_id"),
                "user_prompt": arguments.get("prompt"),
                "articles": arguments.get("articles", []),
                "recommendations": arguments.get("recommendations", []),
                "tickets": arguments.get("tickets", []),
                "intent": arguments.get("intent", {"intent": "generic_question"}),
                "predictions": arguments.get("predictions", {})
            })
            
            return {
                "response": result.get("response", ""),
                "articles_used": result.get("articles_used", []),
                "workflow_status": result.get("workflow_status", "completed")
            }
        
        self.register_tool(
            name="generate_response",
            description="Generate an intelligent conversational response",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "articles": {"type": "array"},
                    "recommendations": {"type": "array"},
                    "tickets": {"type": "array"},
                    "intent": {"type": "object"},
                    "predictions": {"type": "object"}
                },
                "required": ["prompt"]
            },
            handler=generate_response
        )
        
        # Tool: Maintain Conversation Context
        async def maintain_context(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Maintain conversation context"""
            session_id = arguments.get("session_id")
            
            if session_id:
                history = self.agent.shared_memory.get_conversation(session_id)
                return {
                    "session_id": session_id,
                    "history": history,
                    "context_maintained": True
                }
            
            return {"error": "No session_id provided"}
        
        self.register_tool(
            name="maintain_context",
            description="Maintain and retrieve conversation context",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"}
                },
                "required": ["session_id"]
            },
            handler=maintain_context
        )
        
        # Tool: Coordinate Agents
        async def coordinate_agents(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Coordinate with other agents for enhanced responses"""
            collaboration_type = arguments.get("collaboration_type", "general")
            context = arguments.get("context", {})
            
            # Trigger collaboration through mental state
            self.agent.mental_state.request_collaboration(
                agent_type=arguments.get("target_agent", "recommendation_agent"),
                reasoning_type=collaboration_type,
                context=context
            )
            
            return {
                "collaboration_requested": True,
                "type": collaboration_type,
                "context": context
            }
        
        self.register_tool(
            name="coordinate_agents",
            description="Coordinate with other agents for collaborative responses",
            input_schema={
                "type": "object",
                "properties": {
                    "collaboration_type": {"type": "string"},
                    "target_agent": {"type": "string"},
                    "context": {"type": "object"}
                }
            },
            handler=coordinate_agents
        )
    
    def _register_resources(self):
        """Register chat agent resources"""
        
        # Resource: Conversation Templates
        self.register_resource(
            uri="chat://templates/conversations",
            name="Conversation Templates",
            description="Pre-defined conversation templates for common scenarios",
            content={
                "greeting": "Hello! I'm here to help with your Agile development questions. What can I assist you with today?",
                "clarification": "I'd like to understand better. Could you provide more details about {topic}?",
                "recommendation_followup": "Based on your question about {topic}, here are some recommendations: {recommendations}",
                "closing": "Is there anything else I can help you with regarding {topic}?"
            }
        )
        
        # Resource: Agent Mental State
        async def get_mental_state():
            """Get current mental state"""
            return self.agent.get_mental_state_summary()
        
        self.register_resource(
            uri="chat://state/mental",
            name="Mental State",
            description="Current mental state and beliefs of the chat agent",
            handler=get_mental_state
        )
        
        # Resource: Semantic Memory Context
        async def get_semantic_context():
            """Get semantic memory insights"""
            query = "recent interactions"
            return self.agent.get_semantic_context(query)
        
        self.register_resource(
            uri="chat://memory/semantic",
            name="Semantic Memory",
            description="Semantic memory context and insights",
            handler=get_semantic_context
        )
    
    def _get_agent_prompts(self) -> List[Dict[str, Any]]:
        """Get chat agent prompts"""
        return [
            {
                "name": "agile_consultation",
                "description": "Consultation about Agile methodology",
                "template": "As an Agile expert, help me understand {topic} in the context of {project_type} projects. Focus on practical applications and best practices.",
                "arguments": [
                    {"name": "topic", "description": "The Agile topic to discuss", "required": True},
                    {"name": "project_type", "description": "Type of project (e.g., software, marketing)", "required": False}
                ]
            },
            {
                "name": "sprint_analysis",
                "description": "Analyze sprint performance and provide insights",
                "template": "Analyze the sprint data for {project_name} and provide insights on {focus_area}. Include recommendations for improvement.",
                "arguments": [
                    {"name": "project_name", "description": "Name of the project", "required": True},
                    {"name": "focus_area", "description": "Area to focus on (velocity, quality, etc.)", "required": True}
                ]
            }
        ]
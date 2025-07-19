from mcp_integration.servers.base_mcp_server import BaseMCPServer
from typing import Dict, Any, List
import asyncio

class RetrievalAgentMCPServer(BaseMCPServer):
    """MCP Server for Retrieval Agent"""
    
    def _register_tools(self):
        """Register retrieval agent tools"""
        
        # Tool: Search Articles
        async def search_articles(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Search for relevant articles"""
            result = self.agent.run({
                "session_id": arguments.get("session_id"),
                "user_prompt": arguments.get("query"),
                "collaboration_purpose": arguments.get("purpose")
            })
            
            return {
                "articles": result.get("articles", []),
                "retrieval_quality": result.get("retrieval_quality", {}),
                "workflow_status": result.get("workflow_status", "success")
            }
        
        self.register_tool(
            name="search_articles",
            description="Search for relevant Confluence articles",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "session_id": {"type": "string"},
                    "purpose": {"type": "string"}
                },
                "required": ["query"]
            },
            handler=search_articles
        )
        
        # Tool: Rank Articles
        async def rank_articles(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Rank articles by relevance"""
            articles = arguments.get("articles", [])
            query = arguments.get("query", "")
            
            # Simple ranking based on title/content matching
            ranked = []
            for article in articles:
                score = 0
                title = article.get("title", "").lower()
                content = article.get("content", "").lower()
                query_lower = query.lower()
                
                # Title match
                if query_lower in title:
                    score += 2
                
                # Content match
                score += content.count(query_lower) * 0.1
                
                article["relevance_score"] = min(score, 1.0)
                ranked.append(article)
            
            # Sort by score
            ranked.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            return {
                "ranked_articles": ranked,
                "top_article": ranked[0] if ranked else None,
                "ranking_method": "keyword_frequency"
            }
        
        self.register_tool(
            name="rank_articles",
            description="Rank articles by relevance to query",
            input_schema={
                "type": "object",
                "properties": {
                    "articles": {"type": "array"},
                    "query": {"type": "string"}
                },
                "required": ["articles", "query"]
            },
            handler=rank_articles
        )
        
        # Tool: Extract Keywords
        async def extract_keywords(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Extract keywords from text"""
            text = arguments.get("text", "")
            
            # Simple keyword extraction
            keywords = self.agent._extract_key_terms(text)
            
            return {
                "keywords": keywords,
                "keyword_count": len(keywords),
                "text_length": len(text)
            }
        
        self.register_tool(
            name="extract_keywords",
            description="Extract keywords from text for better search",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            },
            handler=extract_keywords
        )
    
    def _register_resources(self):
        """Register retrieval resources"""
        
        # Resource: Search Index Status
        async def get_index_status():
            """Get search index status"""
            # Check ChromaDB collection
            try:
                collection = self.agent.collection
                count = collection.count()
                
                return {
                    "index_name": "confluence_articles",
                    "document_count": count,
                    "index_healthy": True,
                    "last_updated": datetime.now().isoformat() # type: ignore
                }
            except Exception as e:
                return {
                    "index_name": "confluence_articles",
                    "index_healthy": False,
                    "error": str(e)
                }
        
        self.register_resource(
            uri="retrieval://index/status",
            name="Search Index Status",
            description="Status of the search index",
            handler=get_index_status
        )
        
        # Resource: Search Templates
        self.register_resource(
            uri="retrieval://templates/searches",
            name="Search Templates",
            description="Pre-configured search templates",
            content={
                "agile_best_practices": "agile scrum kanban best practices methodology",
                "technical_documentation": "implementation guide tutorial technical documentation",
                "troubleshooting": "error troubleshooting solution fix issue problem",
                "architecture": "architecture design patterns microservices system design"
            }
        )
    
    def _get_agent_prompts(self) -> List[Dict[str, Any]]:
        """Get retrieval agent prompts"""
        return [
            {
                "name": "find_documentation",
                "description": "Find relevant documentation",
                "template": "Find all documentation related to {topic} in the context of {project_type}. Focus on practical guides and implementation details.",
                "arguments": [
                    {"name": "topic", "description": "Topic to search for", "required": True},
                    {"name": "project_type", "description": "Type of project", "required": False}
                ]
            },
            {
                "name": "troubleshooting_search",
                "description": "Search for troubleshooting guides",
                "template": "Search for troubleshooting guides and solutions for {error_type} in {component}. Include root cause analysis and prevention strategies.",
                "arguments": [
                    {"name": "error_type", "description": "Type of error", "required": True},
                    {"name": "component", "description": "System component", "required": True}
                ]
            }
        ]
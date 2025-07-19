from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent, AgentCapability
from orchestrator.rag.enhanced_rag_pipeline import EnhancedRAGPipeline # type: ignore
import os
import logging

class EnhancedRetrievalAgent(BaseAgent):
    """Enhanced retrieval agent using the full RAG pipeline"""
    
    def __init__(self, shared_memory):
        super().__init__(name="retrieval_agent", redis_client=shared_memory.redis_client)  # Keep name as retrieval_agent for compatibility
        
        # Initialize RAG pipeline
        self.rag_pipeline = EnhancedRAGPipeline(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "jurix_neo4j_password"),
            confluence_url=os.getenv("CONFLUENCE_URL", "http://localhost:8090"),
            confluence_user=os.getenv("CONFLUENCE_USERNAME", "admin"),
            confluence_password=os.getenv("CONFLUENCE_PASSWORD", "221201Ra!")
        )
        
        self.shared_memory = shared_memory
        
        self.mental_state.capabilities = [
            AgentCapability.RETRIEVE_DATA,
            AgentCapability.RANK_CONTENT,
            AgentCapability.COORDINATE_AGENTS
        ]
        
        self.mental_state.obligations.extend([
            "detect_query_type",
            "retrieve_articles",
            "rank_relevance",
            "handle_collaboration_requests",
            "enrich_context_collaboratively",
            "analyze_retrieval_quality",
            "perform_hybrid_search",
            "track_search_quality",
            "apply_incremental_learning",
            "find_ticket_solutions"
        ])
        
        self.logger = logging.getLogger("EnhancedRetrievalAgent")
        self.log("✨ Enhanced Retrieval Agent initialized with Neo4j and advanced RAG")
    
    def _perceive(self, input_data: Dict[str, Any]) -> None:
        """Enhanced perception with collaboration context awareness"""
        super()._perceive(input_data)
        
        session_id = input_data.get("session_id")
        user_prompt = input_data.get("user_prompt")
        history = self.shared_memory.get_conversation(session_id) if session_id else []
        
        self.log(f"[PERCEPTION] Processing enhanced retrieval request: '{user_prompt}'")
        
        # Store core beliefs
        self.mental_state.add_belief("session_id", session_id, 0.9, "input")
        self.mental_state.add_belief("user_prompt", user_prompt, 0.9, "input")
        self.mental_state.add_belief("conversation_history", history, 0.8, "memory")
        
        # Extract ticket context if present
        project = input_data.get("project")
        ticket_key = self._extract_ticket_key(user_prompt)
        
        if ticket_key:
            self.mental_state.add_belief("ticket_key", ticket_key, 0.9, "extraction")
            self.log(f"[PERCEPTION] Detected ticket reference: {ticket_key}")
        
        if project:
            self.mental_state.add_belief("project", project, 0.9, "input")
        
        # Handle collaborative context
        collaboration_purpose = input_data.get("collaboration_purpose")
        primary_agent_result = input_data.get("primary_agent_result")
        
        if collaboration_purpose:
            self.mental_state.add_belief("collaboration_context", collaboration_purpose, 0.9, "collaboration")
            self.log(f"[COLLABORATION] Operating in collaborative mode: {collaboration_purpose}")
            
            # Extract enhanced search terms from collaboration
            if collaboration_purpose == "context_enrichment":
                self._extract_collaborative_context(input_data)
        
        if primary_agent_result:
            self.mental_state.add_belief("primary_agent_context", primary_agent_result, 0.8, "collaboration")
            self.log(f"[COLLABORATION] Received primary agent context for enhanced retrieval")
    
    def _extract_ticket_key(self, text: str) -> Optional[str]:
        """Extract ticket key from text"""
        import re
        match = re.search(r'[A-Z]{2,}-\d+', text)
        return match.group() if match else None
    
    def _extract_collaborative_context(self, input_data: Dict[str, Any]) -> None:
        """Extract context from collaboration request"""
        primary_result = input_data.get("primary_agent_result", {})
        
        # Extract search enhancement terms
        enhancement_terms = []
        
        # From recommendations
        recommendations = primary_result.get("recommendations", [])
        if recommendations:
            for rec in recommendations[:3]:
                # Extract key technical terms
                import re
                tech_terms = re.findall(r'\b(?:kubernetes|docker|api|database|cache|security|authentication|deployment|monitoring|testing)\b', rec.lower())
                enhancement_terms.extend(tech_terms)
        
        # From tickets
        tickets = primary_result.get("tickets", [])
        if tickets:
            for ticket in tickets[:5]:
                summary = ticket.get("fields", {}).get("summary", "")
                if summary:
                    enhancement_terms.append(summary)
        
        # Store enhancement terms
        unique_terms = list(set(enhancement_terms))
        self.mental_state.add_belief("search_enhancement_terms", unique_terms, 0.8, "collaboration")
        self.log(f"[COLLABORATION] Extracted {len(unique_terms)} enhancement terms for search")
    
    def _retrieve_articles(self) -> List[Dict[str, Any]]:
        """Enhanced article retrieval using hybrid search"""
        user_prompt = self.mental_state.get_belief("user_prompt")
        collaboration_context = self.mental_state.get_belief("collaboration_context")
        enhancement_terms = self.mental_state.get_belief("search_enhancement_terms") or []
        ticket_key = self.mental_state.get_belief("ticket_key")
        project = self.mental_state.get_belief("project")
        
        # Build enhanced query
        search_query = user_prompt
        if enhancement_terms:
            search_query = f"{user_prompt} {' '.join(enhancement_terms[:5])}"
            self.log(f"[SEARCH] Enhanced query with collaboration terms: {search_query}")
        
        # Build context for hybrid search
        search_context = {
            'ticket_key': ticket_key,
            'project': project,
            'collaboration_purpose': collaboration_context
        }
        
        try:
            # Perform hybrid search
            results = self.rag_pipeline.hybrid_search(
                query=search_query,
                ticket_context=search_context if ticket_key else None,
                k=10 if collaboration_context else 5
            )
            
            # Format results
            articles = []
            for result in results:
                article = {
                    "id": result['id'],
                    "title": result['metadata'].get('title', 'Untitled'),
                    "content": result['content'],
                    "relevance_score": result['relevance_score'],
                    "metadata": result['metadata'],
                    "collaborative_retrieval": bool(collaboration_context),
                    "sources": result.get('sources', []),
                    "score_breakdown": result.get('score_breakdown', {})
                }
                
                # Add ticket relationship info
                if 'knowledge_graph' in result.get('sources', []):
                    article["from_knowledge_graph"] = True
                    article["solved_similar_tickets"] = result['metadata'].get('solved_similar', False)
                
                articles.append(article)
            
            self.log(f"[RETRIEVAL] Retrieved {len(articles)} articles using hybrid search")
            self.log(f"[RETRIEVAL] Graph results: {sum(1 for a in articles if a.get('from_knowledge_graph', False))}")
            
            return articles
            
        except Exception as e:
            self.log(f"[ERROR] Enhanced retrieval failed: {e}")
            return []
    
    def _assess_retrieval_quality(self, articles: List[Dict[str, Any]], 
                                collaboration_context: str) -> Dict[str, Any]:
        """Enhanced quality assessment"""
        quality_assessment = {
            "article_count": len(articles),
            "quality_score": 0.0,
            "collaboration_success": False,
            "needs_additional_search": False,
            "used_knowledge_graph": any(a.get('from_knowledge_graph', False) for a in articles),
            "multi_source_results": sum(1 for a in articles if len(a.get('sources', [])) > 1)
        }
        
        if not articles:
            quality_assessment["quality_score"] = 0.0
            quality_assessment["needs_additional_search"] = True
            return quality_assessment
        
        # Calculate quality based on multiple factors
        avg_relevance = sum(a['relevance_score'] for a in articles) / len(articles)
        graph_bonus = 0.2 if quality_assessment["used_knowledge_graph"] else 0
        multi_source_bonus = 0.1 * (quality_assessment["multi_source_results"] / len(articles))
        
        quality_assessment["quality_score"] = min(avg_relevance + graph_bonus + multi_source_bonus, 1.0)
        
        # For collaboration, we want higher quality
        if collaboration_context == "context_enrichment":
            quality_assessment["collaboration_success"] = len(articles) >= 3 and quality_assessment["quality_score"] > 0.6
        else:
            quality_assessment["collaboration_success"] = len(articles) > 0
        
        return quality_assessment
    
    def _act(self) -> Dict[str, Any]:
        """Enhanced action method with collaboration support"""
        try:
            collaboration_context = self.mental_state.get_belief("collaboration_context")
            
            # Store retrieval experience
            if hasattr(self.mental_state, 'add_experience'):
                user_prompt = self.mental_state.get_belief("user_prompt")
                ticket_key = self.mental_state.get_belief("ticket_key")
                
                experience_desc = f"{'Collaborative ' if collaboration_context else ''}enhanced retrieval"
                if ticket_key:
                    experience_desc += f" for ticket {ticket_key}"
                experience_desc += f": {user_prompt[:100]}"
                
                self.mental_state.add_experience(
                    experience_description=experience_desc,
                    outcome="processing_retrieval_request",
                    confidence=0.8,
                    metadata={
                        "collaborative": bool(collaboration_context),
                        "has_ticket_context": bool(ticket_key),
                        "query": user_prompt[:100] if user_prompt else ""
                    }
                )
            
            # Retrieve articles with enhanced logic
            articles = self._retrieve_articles()
            
            # Assess retrieval quality
            quality_assessment = self._assess_retrieval_quality(articles, collaboration_context or "")
            
            # Apply incremental learning if we have feedback
            if self.mental_state.get_belief("user_feedback"):
                self._apply_feedback_learning(articles)
            
            # Build comprehensive response
            result = {
                "articles": articles,
                "workflow_status": "success",
                "retrieval_quality": quality_assessment
            }
            
            # Add collaboration metadata if collaborative request
            if collaboration_context:
                collaboration_metadata = {
                    "is_collaborative": True,
                    "collaboration_context": collaboration_context,
                    "collaboration_success": quality_assessment["collaboration_success"],
                    "articles_for_collaboration": len(articles),
                    "enrichment_provided": quality_assessment["collaboration_success"],
                    "used_knowledge_graph": quality_assessment["used_knowledge_graph"],
                    "multi_source_results": quality_assessment["multi_source_results"]
                }
                result["collaboration_metadata"] = collaboration_metadata
                
                self.log(f"[COLLABORATION] Enhanced retrieval metadata: {collaboration_metadata}")
            
            return result
            
        except Exception as e:
            self.log(f"[ERROR] Failed to retrieve articles: {e}")
            
            return {
                "articles": [],
                "workflow_status": "failure",
                "error": str(e),
                "collaboration_metadata": {
                    "is_collaborative": bool(collaboration_context),
                    "collaboration_success": False,
                    "error_occurred": True
                } if collaboration_context else None
            }
    
    def _apply_feedback_learning(self, articles: List[Dict[str, Any]]):
        """Apply user feedback for incremental learning"""
        feedback = self.mental_state.get_belief("user_feedback")
        if not feedback:
            return
        
        ticket_key = feedback.get("ticket_key")
        helpful_docs = feedback.get("helpful_doc_ids", [])
        
        if ticket_key and helpful_docs:
            # Update Neo4j relationships
            for article in articles:
                if article['id'] in helpful_docs:
                    self.rag_pipeline.neo4j_manager.update_relationship_feedback(
                        doc_id=article['id'],
                        ticket_key=ticket_key,
                        helpful=True
                    )
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for enhanced retrieval with collaboration support"""
        return self.process(input_data)
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent, AgentCapability
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from orchestrator.core.model_manager import ModelManager # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import chromadb # type: ignore
import json
from datetime import datetime

class SmartSuggestionAgent(BaseAgent):
    OBJECTIVE = "Provide immediate, intelligent article suggestions for Jira issues with continuous learning"
    
    def __init__(self, shared_memory: JurixSharedMemory):
        super().__init__(name="smart_suggestion_agent", redis_client=shared_memory.redis_client)
        self.shared_memory = shared_memory
        self.model_manager = ModelManager()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path="c:/Users/tlouh/Desktop/JURIX/chromadb_data")
        self.collection = self.client.get_or_create_collection(name="confluence_articles")
        self.mental_state.capabilities = [
            AgentCapability.RETRIEVE_DATA,
            AgentCapability.RANK_CONTENT,
            AgentCapability.PROVIDE_RECOMMENDATIONS
        ]
        self.mental_state.obligations.extend([
            "analyze_issue_content",
            "suggest_relevant_articles",
            "track_user_feedback",
            "improve_suggestions"
        ])
        self.feedback_key_prefix = "article_feedback:"
        self.suggestion_history_key = "suggestion_history:"
        self.log("SmartSuggestionAgent initialized with learning capabilities")
    
    def _perceive(self, input_data: Dict[str, Any]) -> None:
        super()._perceive(input_data)
        
        issue_key = input_data.get("issue_key")
        issue_summary = input_data.get("issue_summary", "")
        issue_description = input_data.get("issue_description", "")
        issue_type = input_data.get("issue_type", "")
        issue_status = input_data.get("issue_status", "")
        quick_mode = input_data.get("quick_mode", True)
        self.log(f"[PERCEPTION] Analyzing issue {issue_key}: {issue_summary[:50]}...")
        self.mental_state.add_belief("issue_key", issue_key, 0.9, "input")
        self.mental_state.add_belief("issue_summary", issue_summary, 0.9, "input")
        self.mental_state.add_belief("issue_description", issue_description, 0.9, "input")
        self.mental_state.add_belief("issue_type", issue_type, 0.9, "input")
        self.mental_state.add_belief("issue_status", issue_status, 0.9, "input")
        self.mental_state.add_belief("quick_mode", quick_mode, 0.9, "input")
        feedback_stats = self._get_feedback_stats(issue_key)
        self.mental_state.add_belief("feedback_stats", feedback_stats, 0.8, "history")
    
    def _analyze_issue_content(self) -> Dict[str, Any]:
        """Fast analysis of issue content for immediate suggestions"""
        issue_summary = self.mental_state.get_belief("issue_summary") or ""
        issue_description = self.mental_state.get_belief("issue_description") or ""
        issue_type = self.mental_state.get_belief("issue_type") or ""
        combined_content = f"{issue_summary} {issue_description} {issue_type}"
        keywords = self._extract_keywords(combined_content)
        search_strategy = {
            "keywords": keywords,
            "boost_terms": [],
            "search_depth": 3 if self.mental_state.get_belief("quick_mode") else 5
        }
        if "bug" in issue_type.lower():
            search_strategy["boost_terms"].extend(["fix", "resolution", "workaround", "troubleshooting"])
        elif "story" in issue_type.lower():
            search_strategy["boost_terms"].extend(["implementation", "guide", "tutorial", "best practices"])
        elif "task" in issue_type.lower():
            search_strategy["boost_terms"].extend(["process", "procedure", "steps", "howto"])
        
        return search_strategy
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        if not text:
            return []
        tech_terms = [
            "kubernetes", "docker", "deployment", "api", "database", "authentication",
            "authorization", "microservice", "ci/cd", "pipeline", "configuration",
            "error", "exception", "performance", "security", "integration", "testing",
            "monitoring", "logging", "scaling", "cache", "queue", "async", "sync"
        ]
        text_lower = text.lower()
        found_keywords = []
        for term in tech_terms:
            if term in text_lower:
                found_keywords.append(term)
        words = text.split()
        for word in words:
            if word[0].isupper() and len(word) > 3 and word.lower() not in found_keywords:
                found_keywords.append(word.lower())
        return found_keywords[:10]
    
    def _suggest_articles(self, search_strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant articles based on search strategy"""
        issue_summary = self.mental_state.get_belief("issue_summary") or ""
        issue_description = self.mental_state.get_belief("issue_description") or ""
        enhanced_context = self.mental_state.get_belief("enhanced_context") or ""
        keywords = search_strategy.get("keywords", [])
        boost_terms = search_strategy.get("boost_terms", [])
        search_depth = search_strategy.get("search_depth", 3)
        if enhanced_context:
            search_query = enhanced_context
        else:
            search_query = f"{issue_summary} {issue_description} {' '.join(keywords)} {' '.join(boost_terms)}"
        self.log(f"[SEARCH] Query: {search_query[:200]}...")
        try:
            query_embedding = self.model.encode(search_query).tolist()
            initial_search_depth = min(search_depth * 3, 15)  
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_search_depth
            )
            if not results["documents"][0]:
                self.log("[SEARCH] No articles found!")
                return []
            articles = []
            seen_titles = set() 
            for idx, (doc, meta, distance) in enumerate(zip(
                results["documents"][0], 
                results["metadatas"][0],
                results["distances"][0] if "distances" in results else [0] * len(results["documents"][0])
            )):
                title = meta.get("title", "Untitled Article")
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                base_relevance = 1.0 - (distance / 2.0) if distance else 0.8
                doc_lower = doc.lower()
                title_lower = title.lower()
                keyword_boost = 0
                for keyword in keywords:
                    if keyword.lower() in doc_lower or keyword.lower() in title_lower:
                        keyword_boost += 0.1
                for boost_term in boost_terms:
                    if boost_term.lower() in doc_lower or boost_term.lower() in title_lower:
                        keyword_boost += 0.15
                relevance_score = min(base_relevance + keyword_boost, 1.0)
                if relevance_score < 0.3:
                    continue
                article = {
                    "content": doc[:500] + "..." if len(doc) > 500 else doc,
                    "metadata": meta,
                    "title": title,
                    "relevance_score": relevance_score,
                    "suggestion_reason": self._get_suggestion_reason(meta, keywords, boost_terms),
                    "article_id": meta.get("id", f"article_{idx}"),
                    "distance": distance  
                }
                articles.append(article)
            articles.sort(key=lambda x: x["relevance_score"], reverse=True)
            articles = self._apply_feedback_learning(articles)
            final_articles = articles[:search_depth]
            
            self.log(f"[SUGGESTION] Found {len(final_articles)} relevant articles from {len(results['documents'][0])} searched")
            for i, article in enumerate(final_articles):
                self.log(f"   {i+1}. {article['title']} - Score: {article['relevance_score']:.2f}")
            return final_articles
        except Exception as e:
            self.log(f"[ERROR] Article suggestion failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _calculate_relevance_score(self, content: str, metadata: Dict[str, Any], 
                                   keywords: List[str], boost_terms: List[str], 
                                   rank: int) -> float:
        """Calculate relevance score with feedback consideration"""
        base_score = 1.0 - (rank * 0.1)
        content_lower = content.lower()
        keyword_matches = sum(1 for kw in keywords if kw in content_lower)
        boost_matches = sum(1 for bt in boost_terms if bt in content_lower)
        
        keyword_score = keyword_matches * 0.1
        boost_score = boost_matches * 0.15
        feedback_stats = self.mental_state.get_belief("feedback_stats") or {}
        article_id = metadata.get("id", "")
        if article_id in feedback_stats:
            stats = feedback_stats[article_id]
            helpful_rate = stats["helpful"] / max(stats["total"], 1)
            feedback_score = helpful_rate * 0.3
        else:
            feedback_score = 0.0
        total_score = base_score + keyword_score + boost_score + feedback_score
        return min(total_score, 1.0)
    
    def _get_suggestion_reason(self, metadata: Dict[str, Any], 
                              keywords: List[str], boost_terms: List[str]) -> str:
        """Generate explanation for why article was suggested"""
        reasons = []
        title = metadata.get("title", "").lower()
        matched_keywords = [kw for kw in keywords if kw in title]
        if matched_keywords:
            reasons.append(f"Matches: {', '.join(matched_keywords[:3])}")
        matched_boosts = [bt for bt in boost_terms if bt in title]
        if matched_boosts:
            reasons.append(f"Related to: {matched_boosts[0]}")
        feedback_stats = self.mental_state.get_belief("feedback_stats") or {}
        article_id = metadata.get("id", "")
        
        if article_id in feedback_stats:
            stats = feedback_stats[article_id]
            if stats["helpful"] > stats["not_relevant"]:
                reasons.append("Previously helpful for similar issues")
        
        return " | ".join(reasons) if reasons else "General relevance"
    def _apply_feedback_learning(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply feedback learning to re-rank articles"""
        issue_key = self.mental_state.get_belief("issue_key")
        issue_type = self.mental_state.get_belief("issue_type")
        type_feedback = self._get_issue_type_feedback(issue_type)
        for article in articles:
            article_id = article["article_id"]
            if article_id in type_feedback:
                performance = type_feedback[article_id]
                if performance["success_rate"] > 0.7:
                    article["relevance_score"] *= 1.2
                    article["suggestion_reason"] += " | Highly rated for this issue type"
        articles.sort(key=lambda x: x["relevance_score"], reverse=True)
        return articles[:5] 
    
    def _track_feedback(self, issue_key: str, article_id: str, helpful: bool) -> None:
        """Track user feedback on article suggestions"""
        feedback_key = f"{self.feedback_key_prefix}{issue_key}"
        feedback_data = {
            "issue_key": issue_key,
            "article_id": article_id,
            "helpful": helpful,
            "timestamp": datetime.now().isoformat(),
            "issue_type": self.mental_state.get_belief("issue_type")
        }
        self.redis_client.lpush(feedback_key, json.dumps(feedback_data))
        self.redis_client.expire(feedback_key, 2592000)  # 30 days
        self._update_feedback_stats(article_id, helpful)
        if hasattr(self.mental_state, 'add_experience'):
            self.mental_state.add_experience(
                experience_description=f"Article {article_id} was {'helpful' if helpful else 'not relevant'} for issue {issue_key}",
                outcome="feedback_recorded",
                confidence=0.9,
                metadata={
                    "issue_key": issue_key,
                    "article_id": article_id,
                    "helpful": helpful,
                    "issue_type": self.mental_state.get_belief("issue_type")
                }
            )
    
    def _get_feedback_stats(self, issue_key: str) -> Dict[str, Dict[str, int]]:
        """Get feedback statistics for articles"""
        stats_key = "article_feedback_stats"
        stats_data = self.redis_client.get(stats_key)
        if stats_data:
            return json.loads(stats_data)
        return {}
    
    def _update_feedback_stats(self, article_id: str, helpful: bool) -> None:
        """Update aggregated feedback statistics"""
        stats_key = "article_feedback_stats"
        stats_data = self.redis_client.get(stats_key)
        if stats_data:
            stats = json.loads(stats_data)
        else:
            stats = {}
        if article_id not in stats:
            stats[article_id] = {"helpful": 0, "not_relevant": 0, "total": 0}
        stats[article_id]["total"] += 1
        if helpful:
            stats[article_id]["helpful"] += 1
        else:
            stats[article_id]["not_relevant"] += 1
        self.redis_client.set(stats_key, json.dumps(stats))
        self.redis_client.expire(stats_key, 2592000)  
    
    def _get_issue_type_feedback(self, issue_type: str) -> Dict[str, Dict[str, Any]]:
        """Get feedback performance by issue type"""
        type_key = f"issue_type_feedback:{issue_type}"
        type_data = self.redis_client.get(type_key)
        if type_data:
            return json.loads(type_data)
        return {}
    
    def _act(self) -> Dict[str, Any]:
        try:
            search_strategy = self._analyze_issue_content()
            suggestions = self._suggest_articles(search_strategy)
            issue_key = self.mental_state.get_belief("issue_key")
            history_key = f"{self.suggestion_history_key}{issue_key}"
            history_data = {
                "issue_key": issue_key,
                "suggestions": [s["article_id"] for s in suggestions],
                "timestamp": datetime.now().isoformat()
            }
            self.redis_client.set(history_key, json.dumps(history_data))
            self.redis_client.expire(history_key, 86400)
            return {
                "suggestions": suggestions,
                "issue_key": issue_key,
                "workflow_status": "success",
                "search_strategy": search_strategy
            }
        except Exception as e:
            self.log(f"[ERROR] Failed to generate suggestions: {e}")
            return {
                "suggestions": [],
                "workflow_status": "failure",
                "error": str(e)
            }
    
    def process_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user feedback on article suggestions"""
        issue_key = feedback_data.get("issue_key")
        article_id = feedback_data.get("article_id")
        helpful = feedback_data.get("helpful", False)
        self._track_feedback(issue_key, article_id, helpful)
        return {
            "status": "success",
            "message": "Feedback recorded"
        }
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if input_data.get("feedback_mode"):
            return self.process_feedback(input_data)
        
        return self.process(input_data)
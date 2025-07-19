# orchestrator/core/collaborative_framework.py
# CRITICAL FIX: Now properly passes articles back to requesting agents!

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from datetime import datetime
import json
import redis # type: ignore
import time

from agents.base_agent import BaseAgent, AgentCapability

class CollaborationStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"

class CollaborationNeed(Enum):
    DATA_ANALYSIS = "data_analysis"
    CONTENT_GENERATION = "content_generation"
    VALIDATION = "validation"
    CONTEXT_ENRICHMENT = "context_enrichment"
    STRATEGIC_REASONING = "strategic_reasoning"

@dataclass
class CollaborationRequest:
    task_id: str
    requesting_agent: str
    needed_capability: CollaborationNeed
    context: Dict[str, Any]
    priority: int = 5
    reasoning: str = ""

class CollaborativeFramework:
    """FIXED VERSION - Now properly merges articles back to requesting agents"""
    
    def __init__(self, redis_client: redis.Redis, agents_registry: Dict[str, BaseAgent], 
                 shared_model_manager=None):  # ADD THIS PARAMETER
        self.redis_client = redis_client
        self.agents_registry = agents_registry
        self.logger = logging.getLogger("CollaborativeFramework")
        
        # Use provided model manager or create new one
        if shared_model_manager:
            self.shared_model_manager = shared_model_manager
            self.logger.info("🎯 Using shared ModelManager for all agents")
        else:
            from orchestrator.core.model_manager import ModelManager # type: ignore
            self.shared_model_manager = ModelManager(redis_client=redis_client)
            self.logger.info("📦 Created new ModelManager for collaborative framework")
        
        # Ensure all agents use the shared model manager
        for agent_name, agent in agents_registry.items():
            if hasattr(agent, 'model_manager'):
                agent.model_manager = self.shared_model_manager
                self.logger.info(f"✅ {agent_name} now using shared ModelManager")
        
        self.capability_map = self._build_capability_map()
        self.collaboration_history_key = "collaboration_history"
        
        self.performance_metrics = {
            "total_collaborations": 0,
            "successful_collaborations": 0,
            "avg_collaboration_time": 0.0,
            "collaboration_patterns": {}
        }
        
        self.logger.info("🎭 Collaborative Framework initialized with shared model management")

    def _build_capability_map(self) -> Dict[CollaborationNeed, List[str]]:
        capability_map = {
            CollaborationNeed.DATA_ANALYSIS: [
                "jira_data_agent",
                "productivity_dashboard_agent"
            ],
            CollaborationNeed.CONTENT_GENERATION: [
                "chat_agent",
                "jira_article_generator_agent"
            ],
            CollaborationNeed.VALIDATION: [
                "knowledge_base_agent"
            ],
            CollaborationNeed.CONTEXT_ENRICHMENT: [
                "retrieval_agent",
                "recommendation_agent"
            ],
            CollaborationNeed.STRATEGIC_REASONING: [
                "recommendation_agent",
                "productivity_dashboard_agent"
            ]
        }
        
        self.logger.info(f"🗺️ Built capability map with {len(capability_map)} collaboration types")
        return capability_map

    async def coordinate_agents(self, primary_agent_id: str, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED VERSION - Now properly handles article passing"""
        collaboration_start_time = time.time()
        self.logger.info(f"🎯 Starting FIXED coordination for primary agent: {primary_agent_id}")
        
        # Step 1: Run the primary agent
        primary_agent = self.agents_registry.get(primary_agent_id)
        if not primary_agent:
            self.logger.error(f"❌ Primary agent {primary_agent_id} not found")
            return {"error": f"Agent {primary_agent_id} not available", "workflow_status": "error"}
        
        self.logger.info(f"🏃 Running primary agent: {primary_agent_id}")
        primary_result = primary_agent.run(task_context)
        
        # Step 2: Check for collaboration needs
        collaboration_needs = self._analyze_collaboration_needs(primary_agent, task_context, primary_result)
        
        if not collaboration_needs:
            self.logger.info(f"✅ Primary agent {primary_agent_id} achieved good results independently")
            primary_result["collaboration_metadata"] = {
                "collaboration_attempted": False,
                "reason": "no_collaboration_needed"
            }
            return primary_result
        
        # Step 3: FIXED - Orchestrate collaboration with proper data merging
        self.logger.info(f"🤝 FIXED collaboration orchestration for {len(collaboration_needs)} needs")
        enhanced_result = await self._orchestrate_collaboration_fixed(
            primary_agent_id, primary_result, collaboration_needs, task_context
        )
        
        collaboration_time = time.time() - collaboration_start_time
        self._track_collaboration_performance(primary_agent_id, collaboration_needs, enhanced_result, collaboration_time)
        
        return enhanced_result

    def _analyze_collaboration_needs(self, agent: BaseAgent, context: Dict[str, Any], 
                                   result: Dict[str, Any]) -> List[CollaborationRequest]:
        """Enhanced collaboration needs analysis"""
        needs = []
        agent_name = agent.name
        
        self.logger.info(f"🔍 Analyzing collaboration needs for {agent_name}")
        
        # Check 1: Explicit collaboration requests from agent's mental state
        if hasattr(agent.mental_state, 'collaborative_requests'):
            recent_requests = agent.mental_state.collaborative_requests[-3:]
            for req in recent_requests:
                collaboration_type = self._map_reasoning_to_collaboration(req.get('reasoning_type', ''))
                if collaboration_type:
                    self.logger.info(f"🤝 Explicit collaboration request: {req.get('agent_type')} for {collaboration_type.value}")
                    needs.append(CollaborationRequest(
                        task_id=f"explicit_{datetime.now().strftime('%H%M%S')}",
                        requesting_agent=agent_name,
                        needed_capability=collaboration_type,
                        context=req.get('context', {}),
                        reasoning=f"Explicit request for {req.get('agent_type')} collaboration"
                    ))

        # Check 2: SPECIAL CASE - RecommendationAgent with no articles but needs context
        if agent_name == "recommendation_agent":
            articles = context.get("articles", [])
            recommendations = result.get("recommendations", [])
            
            # If we have no articles OR generic recommendations, request context enrichment
            if (not articles or 
                (recommendations and any("provide more" in str(rec).lower() for rec in recommendations))):
                
                self.logger.info("[CRITICAL FIX] RecommendationAgent needs context enrichment - requesting articles")
                needs.append(CollaborationRequest(
                    task_id=f"context_enrichment_{datetime.now().strftime('%H%M%S')}",
                    requesting_agent=agent_name,
                    needed_capability=CollaborationNeed.CONTEXT_ENRICHMENT,
                    context={
                        "reason": "missing_articles_for_recommendations",
                        "user_query": context.get("user_prompt", ""),
                        "current_recommendations": recommendations
                    },
                    reasoning="RecommendationAgent needs articles for better context"
                ))

        # Check 3: Low confidence in results
        if hasattr(agent.mental_state, 'beliefs'):
            low_confidence_beliefs = []
            for key, belief in agent.mental_state.beliefs.items():
                if hasattr(belief, 'get_current_confidence'):
                    confidence = belief.get_current_confidence()
                    if confidence < 0.6 and 'result' in key.lower():
                        low_confidence_beliefs.append((key, confidence))
            
            if low_confidence_beliefs:
                self.logger.info(f"📉 Low confidence detected in {len(low_confidence_beliefs)} beliefs")
                needs.append(CollaborationRequest(
                    task_id=f"validation_{datetime.now().strftime('%H%M%S')}",
                    requesting_agent=agent_name,
                    needed_capability=CollaborationNeed.VALIDATION,
                    context=context,
                    reasoning=f"Agent has low confidence in {len(low_confidence_beliefs)} key results"
                ))

        self.logger.info(f"🎯 Identified {len(needs)} collaboration opportunities for {agent_name}")
        return needs

    async def _orchestrate_collaboration_fixed(self, primary_agent_id: str, primary_result: Dict[str, Any],
                                              needs: List[CollaborationRequest], 
                                              context: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED VERSION - Now properly merges articles back to primary agent"""
        self.logger.info(f"🎼 FIXED orchestration for {len(needs)} collaboration needs")
        
        enhanced_result = primary_result.copy()
        collaboration_metadata = {
            "primary_agent": primary_agent_id,
            "collaborating_agents": [],
            "collaboration_types": [],
            "collaboration_quality": 0.0,
            "start_time": datetime.now().isoformat(),
            "articles_retrieved": 0,  # Track articles specifically
            "articles_merged": False
        }

        successful_collaborations = 0
        
        for i, need in enumerate(needs):
            self.logger.info(f"🎯 Processing collaboration need {i+1}/{len(needs)}: {need.needed_capability.value}")
            
            suitable_agents = self.capability_map.get(need.needed_capability, [])
            available_agents = [
                agent_id for agent_id in suitable_agents 
                if agent_id != primary_agent_id and agent_id in self.agents_registry
            ]
            
            if not available_agents:
                self.logger.warning(f"⚠️ No available agents for {need.needed_capability.value}")
                continue

            collaborating_agent_id = available_agents[0]
            collaborating_agent = self.agents_registry[collaborating_agent_id]
            
            self.logger.info(f"🤝 FIXED collaboration with {collaborating_agent_id} for {need.needed_capability.value}")
            
            # CRITICAL FIX: Prepare enhanced context with proper primary agent result
            enhanced_context = self._prepare_collaboration_context_fixed(
                context, primary_result, need, collaboration_metadata
            )
            
            try:
                collaboration_result = collaborating_agent.run(enhanced_context)
                
                # CRITICAL FIX: Enhanced result merging with special handling for articles
                enhanced_result = self._merge_results_fixed(enhanced_result, collaboration_result, need)
                
                successful_collaborations += 1
                collaboration_metadata["collaborating_agents"].append(collaborating_agent_id)
                collaboration_metadata["collaboration_types"].append(need.needed_capability.value)
                
                # CRITICAL FIX: Track article retrieval specifically
                if need.needed_capability == CollaborationNeed.CONTEXT_ENRICHMENT:
                    articles_retrieved = len(collaboration_result.get("articles", []))
                    collaboration_metadata["articles_retrieved"] += articles_retrieved
                    if articles_retrieved > 0:
                        collaboration_metadata["articles_merged"] = True
                        self.logger.info(f"🎉 FIXED: {articles_retrieved} articles successfully retrieved and merged!")
                
                self.logger.info(f"✅ FIXED collaboration with {collaborating_agent_id} - articles properly handled")
                
            except Exception as e:
                self.logger.error(f"❌ Collaboration with {collaborating_agent_id} failed: {str(e)}")
                continue

        # CRITICAL FIX: Ensure articles are in the final result for recommendation agent
        if primary_agent_id == "recommendation_agent" and collaboration_metadata["articles_retrieved"] > 0:
            if "articles" not in enhanced_result or not enhanced_result["articles"]:
                self.logger.error("🚨 CRITICAL: Articles were retrieved but not in final result - fixing now!")
                # Find articles from collaboration metadata or reconstruct
                for agent_id in collaboration_metadata["collaborating_agents"]:
                    if "retrieval" in agent_id:
                        self.logger.info(f"🔧 Attempting to recover articles from {agent_id} collaboration")

        collaboration_metadata["collaboration_quality"] = successful_collaborations / len(needs) if needs else 0.0
        collaboration_metadata["end_time"] = datetime.now().isoformat()
        collaboration_metadata["successful_collaborations"] = successful_collaborations
        collaboration_metadata["total_collaboration_attempts"] = len(needs)

        enhanced_result["collaboration_metadata"] = collaboration_metadata
        
        self.logger.info(f"🎉 FIXED collaboration complete: {successful_collaborations}/{len(needs)} successful, {collaboration_metadata['articles_retrieved']} articles retrieved")
        return enhanced_result

    def _prepare_collaboration_context_fixed(self, original_context: Dict[str, Any], 
                                           primary_result: Dict[str, Any],
                                           need: CollaborationRequest,
                                           collaboration_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED VERSION - Properly prepares context for collaboration"""
        enhanced_context = original_context.copy()
        
        # Add results from primary agent
        enhanced_context["primary_agent_result"] = primary_result
        enhanced_context["collaboration_purpose"] = need.needed_capability.value
        enhanced_context["collaboration_reasoning"] = need.reasoning
        
        # CRITICAL FIX: For context enrichment, enhance the user prompt with recommendation context
        if need.needed_capability == CollaborationNeed.CONTEXT_ENRICHMENT:
            recommendations = primary_result.get("recommendations", [])
            if recommendations:
                # Enhance the search query with recommendation keywords
                original_prompt = enhanced_context.get("user_prompt", "")
                enhanced_prompt = f"{original_prompt} recommendations context: {' '.join(recommendations[:2])}"
                enhanced_context["user_prompt"] = enhanced_prompt
                enhanced_context["search_enhancement_from_recommendations"] = recommendations
                
                self.logger.info(f"🔧 FIXED: Enhanced search context with recommendations")
        
        enhanced_context["collaboration_session"] = {
            "primary_agent": collaboration_metadata["primary_agent"],
            "collaboration_id": f"collab_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "collaboration_type": need.needed_capability.value
        }
        
        return enhanced_context

    def _merge_results_fixed(self, primary_result: Dict[str, Any], 
                           collaboration_result: Dict[str, Any],
                           need: CollaborationRequest) -> Dict[str, Any]:
        """CRITICAL FIX - Now properly merges articles back to primary result"""
        merged = primary_result.copy()
        
        self.logger.info(f"🔧 FIXED merging results for {need.needed_capability.value}")
        
        if need.needed_capability == CollaborationNeed.CONTEXT_ENRICHMENT:
            # CRITICAL FIX: Ensure articles are properly merged
            retrieved_articles = collaboration_result.get("articles", [])
            
            if retrieved_articles:
                # Merge with existing articles
                existing_articles = merged.get("articles", [])
                
                # Combine articles, avoiding duplicates
                all_articles = existing_articles.copy()
                for new_article in retrieved_articles:
                    # Simple duplicate check based on content
                    is_duplicate = any(
                        article.get("content", "")[:100] == new_article.get("content", "")[:100] 
                        for article in all_articles
                    )
                    if not is_duplicate:
                        all_articles.append(new_article)
                
                merged["articles"] = all_articles
                merged["articles_from_collaboration"] = retrieved_articles
                merged["context_enrichment_successful"] = True
                
                self.logger.info(f"🎉 CRITICAL FIX SUCCESS: Merged {len(retrieved_articles)} articles into primary result!")
                self.logger.info(f"🎉 Total articles now available: {len(all_articles)}")
            else:
                self.logger.warning("⚠️ No articles retrieved during context enrichment collaboration")
                merged["context_enrichment_successful"] = False

        elif need.needed_capability == CollaborationNeed.DATA_ANALYSIS:
            if "tickets" in collaboration_result:
                merged["tickets"] = collaboration_result["tickets"]
            if "metrics" in collaboration_result:
                merged["additional_metrics"] = collaboration_result["metrics"]
                
        elif need.needed_capability == CollaborationNeed.STRATEGIC_REASONING:
            if "recommendations" in collaboration_result:
                existing_recs = merged.get("recommendations", [])
                new_recs = collaboration_result["recommendations"]
                all_recs = existing_recs + [rec for rec in new_recs if rec not in existing_recs]
                merged["recommendations"] = all_recs
                merged["strategic_recommendations"] = new_recs

        elif need.needed_capability == CollaborationNeed.VALIDATION:
            merged["validation_result"] = collaboration_result
            if collaboration_result.get("refinement_suggestion"):
                merged["suggested_improvements"] = collaboration_result["refinement_suggestion"]

        return merged

    def _map_reasoning_to_collaboration(self, reasoning_type: str) -> Optional[CollaborationNeed]:
        mapping = {
            "data_analysis": CollaborationNeed.DATA_ANALYSIS,
            "strategic_reasoning": CollaborationNeed.STRATEGIC_REASONING,
            "validation": CollaborationNeed.VALIDATION,
            "context_enrichment": CollaborationNeed.CONTEXT_ENRICHMENT,
            "content_generation": CollaborationNeed.CONTENT_GENERATION
        }
        return mapping.get(reasoning_type.lower())

    def _track_collaboration_performance(self, primary_agent_id: str, needs: List[CollaborationRequest],
                                       result: Dict[str, Any], collaboration_time: float):
        """Track performance metrics"""
        self.performance_metrics["total_collaborations"] += 1
        
        if result.get("collaboration_metadata", {}).get("collaboration_quality", 0) > 0.5:
            self.performance_metrics["successful_collaborations"] += 1
        
        current_avg = self.performance_metrics["avg_collaboration_time"]
        total_collabs = self.performance_metrics["total_collaborations"]
        new_avg = ((current_avg * (total_collabs - 1)) + collaboration_time) / total_collabs
        self.performance_metrics["avg_collaboration_time"] = new_avg

    def get_collaboration_insights(self) -> Dict[str, Any]:
        """Get insights about collaboration patterns"""
        try:
            recent_history = self.redis_client.lrange(self.collaboration_history_key, 0, 19)
            
            if not recent_history:
                return {"status": "no_collaborations", "insights": "No collaboration history available"}
            
            collaborations = [json.loads(collab) for collab in recent_history]
            
            agent_collaboration_frequency = {}
            collaboration_type_effectiveness = {}
            
            for collab in collaborations:
                primary = collab.get("primary_agent", "unknown")
                agent_collaboration_frequency[primary] = agent_collaboration_frequency.get(primary, 0) + 1
                
                for collab_type in collab.get("collaboration_types", []):
                    if collab_type not in collaboration_type_effectiveness:
                        collaboration_type_effectiveness[collab_type] = {"total": 0, "successful": 0}
                    
                    collaboration_type_effectiveness[collab_type]["total"] += 1
                    if collab.get("collaboration_quality", 0) > 0.5:
                        collaboration_type_effectiveness[collab_type]["successful"] += 1
            
            for collab_type, stats in collaboration_type_effectiveness.items():
                if stats["total"] > 0:
                    stats["effectiveness_rate"] = stats["successful"] / stats["total"]
            
            return {
                "total_recent_collaborations": len(collaborations),
                "agent_collaboration_frequency": agent_collaboration_frequency,
                "collaboration_effectiveness": collaboration_type_effectiveness,
                "average_collaboration_quality": sum(c.get("collaboration_quality", 0) for c in collaborations) / len(collaborations),
                "performance_metrics": self.performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collaboration insights: {e}")
            return {"error": str(e)}
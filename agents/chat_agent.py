from typing import Dict, Any, Optional, List
from agents.base_agent import BaseAgent, AgentCapability
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from orchestrator.graph.state import JurixState # type: ignore
from orchestrator.core.model_manager import ModelManager # type: ignore
from agents.jira_data_agent import JiraDataAgent
from agents.recommendation_agent import RecommendationAgent
import logging
import uuid
from datetime import datetime

class ChatAgent(BaseAgent):
    OBJECTIVE = "Coordinate conversation, maintain context, and deliver intelligent responses by collaborating with specialized agents when needed"
    
    def __init__(self, shared_memory: JurixSharedMemory):
        super().__init__(name="chat_agent", redis_client=shared_memory.redis_client)
        self.shared_memory = shared_memory
        self.model_manager = ModelManager()
        self.jira_data_agent = JiraDataAgent(redis_client=shared_memory.redis_client)
        self.recommendation_agent = RecommendationAgent(shared_memory)
        
        self.mental_state.capabilities = [
            AgentCapability.GENERATE_RESPONSE,
            AgentCapability.MAINTAIN_CONVERSATION,
            AgentCapability.COORDINATE_AGENTS
        ]
        
        self.mental_state.obligations.extend([
            "generate_response",
            "coordinate_agents", 
            "maintain_conversation_context",
            "assess_collaboration_needs",
            "synthesize_multi_agent_results",
            "incorporate_predictive_insights"
        ])
        
        if hasattr(self.mental_state, 'vector_memory') and self.mental_state.vector_memory:
            self.log("✅ ChatAgent initialized with semantic memory and collaboration capabilities")
        else:
            self.log("⚠️ ChatAgent initialized without semantic memory - limited collaboration abilities")

    def _perceive(self, input_data: Dict[str, Any]) -> None:
        super()._perceive(input_data)
        
        session_id = input_data.get("session_id")
        user_prompt = input_data.get("user_prompt")
        history = self.shared_memory.get_conversation(session_id) if session_id else []
        articles = input_data.get("articles", [])
        recommendations = input_data.get("recommendations", [])
        tickets = input_data.get("tickets", [])
        intent = input_data.get("intent", {"intent": "generic_question"})
        predictions = input_data.get("predictions", {})
        predictive_insights = input_data.get("predictive_insights", "")

        self.log(f"[PERCEPTION] Processing query: '{user_prompt}' with {len(articles)} articles, {len(recommendations)} recommendations, {len(tickets)} tickets")
        
        self.mental_state.add_belief("session_id", session_id, 0.9, "input")
        self.mental_state.add_belief("user_prompt", user_prompt, 0.9, "input")
        self.mental_state.add_belief("conversation_history", history, 0.8, "memory")
        self.mental_state.add_belief("articles", articles, 0.9, "input")
        self.mental_state.add_belief("recommendations", recommendations, 0.9, "input")
        self.mental_state.add_belief("tickets", tickets, 0.9, "input")
        self.mental_state.add_belief("intent", intent, 0.9, "input")
        self.mental_state.add_belief("predictions", predictions, 0.9, "input")
        self.mental_state.add_belief("predictive_insights", predictive_insights, 0.9, "input")
        
        if input_data.get("collaboration_purpose"):
            self.mental_state.add_belief(
                "collaboration_context", 
                input_data.get("collaboration_purpose"), 
                0.9, 
                "collaboration"
            )
            self.log(f"[COLLABORATION] Received collaborative request: {input_data.get('collaboration_purpose')}")
        
        if input_data.get("primary_agent_result"):
            self.mental_state.add_belief(
                "primary_agent_context",
                input_data.get("primary_agent_result"),
                0.8,
                "collaboration"
            )
            
        self._assess_collaboration_needs(input_data)

    def _assess_collaboration_needs(self, input_data: Dict[str, Any]) -> None:
        user_prompt = input_data.get("user_prompt", "").lower()
        intent = input_data.get("intent", {}).get("intent", "generic_question")
        articles = input_data.get("articles", [])
        recommendations = input_data.get("recommendations", [])
        tickets = input_data.get("tickets", [])
        predictions = input_data.get("predictions", {})
        
        context_quality = self._assess_context_quality(input_data)
        
        if context_quality < 0.6:
            self.log("[COLLABORATION NEED] Insufficient context detected")
            
            if intent == "recommendation" and not recommendations:
                self.mental_state.request_collaboration(
                    agent_type="recommendation_agent",
                    reasoning_type="strategic_reasoning",
                    context={
                        "reason": "missing_recommendations", 
                        "user_query": user_prompt,
                        "available_tickets": len(tickets)
                    }
                )
                
            if any(keyword in user_prompt for keyword in ["project", "ticket", "jira"]) and not tickets:
                self.mental_state.request_collaboration(
                    agent_type="jira_data_agent",
                    reasoning_type="data_analysis",
                    context={
                        "reason": "missing_project_data",
                        "user_query": user_prompt
                    }
                )
        
        predictive_keywords = ["will we complete", "sprint completion", "predict", "forecast", "risk", "when will", "probability"]
        if any(keyword in user_prompt for keyword in predictive_keywords) and not predictions:
            self.log("[COLLABORATION NEED] Predictive analysis requested but not available")
            self.mental_state.request_collaboration(
                agent_type="predictive_analysis_agent",
                reasoning_type="strategic_reasoning",
                context={
                    "reason": "predictive_analysis_needed",
                    "user_query": user_prompt
                }
            )
        
        query_complexity = self._assess_query_complexity(user_prompt)
        if query_complexity > 0.7 and not self._has_sufficient_expertise(user_prompt):
            self.log("[COLLABORATION NEED] Complex query detected, may need specialist help")
            
            if any(keyword in user_prompt for keyword in ["productivity", "analytics", "dashboard", "metrics", "performance"]):
                self.mental_state.request_collaboration(
                    agent_type="productivity_dashboard_agent",
                    reasoning_type="data_analysis",
                    context={
                        "reason": "analytics_expertise_needed",
                        "complexity_score": query_complexity
                    }
                )

    def _assess_context_quality(self, input_data: Dict[str, Any]) -> float:
        quality_score = 0.0
        
        history = input_data.get("conversation_history", [])
        if history:
            quality_score += 0.2
            
        articles = input_data.get("articles", [])
        if articles:
            quality_score += 0.3
            
        recommendations = input_data.get("recommendations", [])
        if recommendations:
            quality_score += 0.3
            
        tickets = input_data.get("tickets", [])
        intent = input_data.get("intent", {}).get("intent", "")
        if tickets and intent in ["recommendation", "project_analysis"]:
            quality_score += 0.2
            
        predictions = input_data.get("predictions", {})
        if predictions:
            quality_score += 0.2
            
        return min(quality_score, 1.0)

    def _assess_query_complexity(self, user_prompt: str) -> float:
        complexity_indicators = [
            "analyze", "compare", "evaluate", "recommend", "optimize", 
            "forecast", "predict", "trend", "pattern", "correlation",
            "bottleneck", "efficiency", "performance", "metrics",
            "will we complete", "probability", "risk", "when will"
        ]
        
        prompt_lower = user_prompt.lower()
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in prompt_lower)
        
        return min(complexity_score / len(complexity_indicators), 1.0)

    def _has_sufficient_expertise(self, user_prompt: str) -> bool:
        if not hasattr(self.mental_state, 'recall_similar_experiences'):
            return False
            
        try:
            similar_experiences = self.mental_state.recall_similar_experiences(user_prompt, max_results=3)
            successful_experiences = [exp for exp in similar_experiences if exp.confidence > 0.7]
            return len(successful_experiences) >= 2
        except Exception as e:
            self.log(f"[ERROR] Failed to check expertise: {e}")
            return False

    def _generate_response(self) -> str:
        intent = self.mental_state.get_belief("intent")["intent"]
        prompt = self.mental_state.get_belief("user_prompt")
        history = self.mental_state.get_belief("conversation_history")
        articles = self.mental_state.get_belief("articles")
        recommendations = self.mental_state.get_belief("recommendations")
        tickets = self.mental_state.get_belief("tickets")
        predictions = self.mental_state.get_belief("predictions")
        predictive_insights = self.mental_state.get_belief("predictive_insights")
        
        collaboration_context = self.mental_state.get_belief("collaboration_context")
        primary_agent_context = self.mental_state.get_belief("primary_agent_context")
        
        history_context = ""
        if history:
            recent_history = history[-5:]
            history_context = "\n".join([f"{entry['role'].upper()}: {entry['content']}" for entry in recent_history])
        else:
            history_context = "No previous context"
        
        article_context = "\n\n".join([
            f"[Article {i+1}]\nTitle: {a.get('title', 'N/A')}\nContent: {a.get('content', 'N/A')[:200]}..." 
            for i, a in enumerate(articles)
        ]) if articles else ""
        
        rec_context = "\n".join([f"- {rec}" for rec in recommendations]) if recommendations else ""
        ticket_context = "\n".join([
            f"Ticket {t.get('key', 'N/A')}: {t.get('fields', {}).get('summary', 'N/A')}" 
            for t in tickets
        ]) if tickets else ""
        
        predictive_context = ""
        if predictions:
            sprint_completion = predictions.get("sprint_completion", {})
            if sprint_completion:
                predictive_context = f"\n\nPREDICTIVE ANALYSIS:\n"
                predictive_context += f"Sprint Completion Probability: {sprint_completion.get('probability', 0):.0%}\n"
                predictive_context += f"Sprint Progress: {sprint_completion.get('completion_percentage', 0):.0%} complete\n"
                predictive_context += f"Remaining Work: {sprint_completion.get('remaining_work', 0)} tickets\n"
                predictive_context += f"Risk Level: {sprint_completion.get('risk_level', 'unknown')}\n"
                
                if sprint_completion.get("risk_factors"):
                    predictive_context += "\nRisk Factors:\n"
                    for factor in sprint_completion.get("risk_factors", []):
                        predictive_context += f"- {factor}\n"
                
                if sprint_completion.get("recommended_actions"):
                    predictive_context += "\nRecommended Actions:\n"
                    for i, action in enumerate(sprint_completion.get("recommended_actions", [])[:3], 1):
                        predictive_context += f"{i}. {action}\n"
            
            # Add velocity forecast if available
            velocity_forecast = predictions.get("velocity_forecast", {})
            if velocity_forecast and velocity_forecast.get("trend"):
                predictive_context += f"\nVelocity Trend: {velocity_forecast.get('trend')} "
                predictive_context += f"({velocity_forecast.get('trend_percentage', 0):+.1%})\n"
            
            # Add warnings
            warnings = predictions.get("warnings", [])
            if warnings:
                predictive_context += "\n⚠️ WARNINGS:\n"
                for warning in warnings[:3]:
                    predictive_context += f"- {warning['message']}\n"
                    predictive_context += f"  Action: {warning['recommended_action']}\n"
            
            # Add natural language summary if available
            if predictive_insights:
                predictive_context += f"\n💡 Summary: {predictive_insights}\n"
        
        collaborative_context = ""
        if collaboration_context or primary_agent_context:
            collaborative_context = "\n\nCOLLABORATIVE CONTEXT:\n"
            if collaboration_context:
                collaborative_context += f"Collaboration Purpose: {collaboration_context}\n"
            if primary_agent_context:
                collaborative_context += f"Previous Analysis: {str(primary_agent_context)[:200]}...\n"

        prompt_template = f"""You are a conversational AI assistant specializing in Agile methodology and software development topics.

CONVERSATION HISTORY:
{history_context}

CURRENT USER QUERY: {prompt}

IMPORTANT: Pay attention to conversation history. For follow-up questions, relate to previous topics discussed.
"""
        
        if articles:
            prompt_template += f"\nRELEVANT ARTICLES:\n{article_context}\n"
        if recommendations:
            prompt_template += f"\nRECOMMENDATIONS:\n{rec_context}\n"
        if tickets:
            prompt_template += f"\nRELEVANT TICKETS:\n{ticket_context}\n"
        if predictive_context:
            prompt_template += predictive_context
        if collaborative_context:
            prompt_template += collaborative_context
        
        prompt_template += "\nProvide a helpful, accurate response based on the context above. If predictive analysis is provided, incorporate those insights naturally into your response. End with 'Anything else I can help with?'"
        
        self.log(f"[RESPONSE GENERATION] Using cognitive model routing for enhanced response")
        
        try:
            # Use dynamic model selection
            response = self.model_manager.generate_response(
                prompt=prompt_template,
                context={
                    "agent_name": self.name,
                    "agent_capabilities": [cap.value for cap in self.mental_state.capabilities],
                    "task_type": "conversation",
                    "user_query": prompt,
                    "has_predictive_insights": bool(predictive_insights),
                    "collaboration_active": bool(collaboration_context or primary_agent_context)
                }
            )
            self.log(f"✅ {self.name} received response from model")
            
            if not response:
                return "No valid response from model. Anything else I can help with?"
            
            self.log(f"[RESPONSE] Generated {len(response)} character response")
            return response.strip()
            
        except Exception as e:
            self.log(f"[ERROR] Response generation failed: {str(e)}")
            try:
                response = self.model_manager.generate_response(prompt_template, use_cognitive_routing=False)
                return response.strip()
            except Exception as fallback_error:
                self.log(f"[ERROR] Fallback generation error: {str(fallback_error)}")
                return f"Error generating response: {str(e)}. Anything else I can help with?"

    def _act(self) -> Dict[str, Any]:
        try:
            session_id = self.mental_state.get_belief("session_id")
            user_prompt = self.mental_state.get_belief("user_prompt")
            intent = self.mental_state.get_belief("intent")["intent"]
            project = self.mental_state.get_belief("intent").get("project")
            predictions = self.mental_state.get_belief("predictions")
            
            collaboration_context = self.mental_state.get_belief("collaboration_context")
            is_collaborative = bool(collaboration_context)
            
            if not project:
                self.log("[WARNING] No project specified in intent")
            
            if hasattr(self.mental_state, 'add_experience') and self.mental_state.vector_memory:
                experience_content = f"{'Collaborative ' if is_collaborative else ''}response for {intent}: {user_prompt}"
                if project:
                    experience_content += f" for project {project}"
                if predictions:
                    experience_content += " with predictive insights"
                
                self.mental_state.add_experience(
                    experience_description=experience_content,
                    outcome="processing_chat_request",
                    confidence=0.8,
                    metadata={
                        "intent": intent, 
                        "project": project, 
                        "user_query": user_prompt[:100],
                        "is_collaborative": is_collaborative,
                        "collaboration_context": collaboration_context,
                        "has_predictions": bool(predictions)
                    }
                )
                self.log("✅ Stored interaction experience in semantic memory")

            if intent == "recommendation" and not self.mental_state.get_belief("tickets"):
                self.log(f"[COLLABORATION] Requesting ticket context from JiraDataAgent for project {project}")
                jira_input = {
                    "project_id": project
                }
                tickets_result = self.jira_data_agent.run(jira_input)
                tickets = tickets_result.get("tickets", [])
                self.mental_state.add_belief("tickets", tickets, 0.9, "jira_data_agent")
                self.log(f"[COLLABORATION] Received {len(tickets)} tickets from JiraDataAgent")

            if intent == "recommendation" and self.mental_state.get_belief("tickets"):
                self.log(f"[COLLABORATION] Requesting recommendations from RecommendationAgent")
                rec_input = {
                    "session_id": session_id,
                    "user_prompt": user_prompt,
                    "articles": self.mental_state.get_belief("articles"),
                    "project": project,
                    "tickets": self.mental_state.get_belief("tickets"),
                    "workflow_type": "collaborative_chat",
                    "intent": self.mental_state.get_belief("intent"),
                    "predictions": predictions
                }
                rec_result = self.recommendation_agent.run(rec_input)
                recommendations = rec_result.get("recommendations", [])
                self.mental_state.add_belief("recommendations", recommendations, 0.9, "recommendation_agent")
                self.log(f"[COLLABORATION] Received {len(recommendations)} recommendations")

            semantic_context = ""
            if hasattr(self.mental_state, 'get_semantic_context'):
                semantic_context = self.mental_state.get_semantic_context(user_prompt)
                self.log(f"[SEMANTIC] Retrieved {len(semantic_context)} characters of context")

            response = self._generate_response()
            
            self.shared_memory.add_interaction(session_id, "user", user_prompt)
            self.shared_memory.add_interaction(session_id, "assistant", response)
            
            if hasattr(self.mental_state, 'add_experience') and self.mental_state.vector_memory:
                success_content = f"Successfully {'collaborated on' if is_collaborative else 'handled'} {intent} query: {user_prompt[:100]}"
                if project:
                    success_content += f" for project {project}"
                if predictions:
                    success_content += " with predictive insights"
                
                self.mental_state.add_experience(
                    experience_description=success_content,
                    outcome=f"Generated {len(response)} character response",
                    confidence=0.9,
                    metadata={
                        "intent": intent,
                        "project": project,
                        "response_length": len(response),
                        "articles_used": len(self.mental_state.get_belief("articles") or []),
                        "recommendations_used": len(self.mental_state.get_belief("recommendations") or []),
                        "predictions_used": bool(predictions),
                        "success": True,
                        "collaborative": is_collaborative
                    }
                )
                self.log("✅ Stored successful interaction in semantic memory")
            
            decision = {
                "action": "generate_response",
                "intent": intent,
                "response_length": len(response),
                "used_articles": len(self.mental_state.get_belief("articles") or []),
                "used_recommendations": len(self.mental_state.get_belief("recommendations") or []),
                "used_tickets": len(self.mental_state.get_belief("tickets") or []),
                "used_predictions": bool(predictions),
                "collaborative": is_collaborative,
                "collaboration_requests": len(self.mental_state.collaborative_requests),
                "reasoning": f"{'Collaborative ' if is_collaborative else ''}response for {intent} query"
            }
            self.mental_state.add_decision(decision)
            
            return {
                "response": response,
                "articles_used": [
                    {"title": a.get("title", "N/A"), "relevance": a.get("relevance_score", 0)} 
                    for a in self.mental_state.get_belief("articles") or []
                ],
                "tickets": self.mental_state.get_belief("tickets") or [],
                "workflow_status": "completed",
                "collaboration_metadata": {
                    "is_collaborative": is_collaborative,
                    "collaboration_context": collaboration_context,
                    "collaboration_requests_made": len(self.mental_state.collaborative_requests),
                    "predictions_incorporated": bool(predictions)
                },
                "next_agent": None
            }
            
        except Exception as e:
            self.log(f"[ERROR] Action failed: {str(e)}")
            
            if hasattr(self.mental_state, 'add_experience'):
                self.mental_state.add_experience(
                    experience_description=f"Failed to process {user_prompt[:50]}",
                    outcome=f"Error: {str(e)}",
                    confidence=0.3,
                    metadata={"error_type": type(e).__name__}
                )
            
            return {
                "error": str(e),
                "workflow_status": "error",
                "response": f"Error processing request: {str(e)}. Please try again. Anything else I can help with?",
                "tickets": self.mental_state.get_belief("tickets") or [],
                "collaboration_metadata": {
                    "is_collaborative": False,
                    "error_occurred": True
                }
            }

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        super()._rethink(action_result)
        
        response_quality = len(action_result.get("response", "")) > 50
        collaboration_metadata = action_result.get("collaboration_metadata", {})
        was_collaborative = collaboration_metadata.get("is_collaborative", False)
        used_predictions = collaboration_metadata.get("predictions_incorporated", False)
        
        reflection = {
            "interaction_type": "chat_response",
            "response_quality": response_quality,
            "articles_utilized": len(action_result.get("articles_used", [])),
            "context_maintained": bool(self.mental_state.get_belief("conversation_history")),
            "collaborative_interaction": was_collaborative,
            "predictions_incorporated": used_predictions,
            "collaboration_requests_made": collaboration_metadata.get("collaboration_requests_made", 0),
            "performance_notes": f"Generated {'collaborative ' if was_collaborative else ''}response with {len(action_result.get('response', ''))} characters"
        }
        
        if used_predictions:
            reflection["performance_notes"] += " including predictive insights"
        
        self.mental_state.add_reflection(reflection)
        
        if was_collaborative:
            collaboration_success = action_result.get("workflow_status") == "completed"
            self.mental_state.add_experience(
                experience_description=f"Collaborative interaction {'succeeded' if collaboration_success else 'failed'}",
                outcome=f"collaboration_{'success' if collaboration_success else 'failure'}",
                confidence=0.8 if collaboration_success else 0.4,
                metadata={
                    "collaboration_type": collaboration_metadata.get("collaboration_context"),
                    "response_quality": response_quality,
                    "predictions_used": used_predictions
                }
            )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(input_data)
from datetime import datetime
from typing import Dict, Any, List
from agents.base_agent import BaseAgent, AgentCapability
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from orchestrator.core.model_manager import ModelManager # type: ignore
import json
import threading
import time
import logging

logging.basicConfig(
    filename='recommendation_agent.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class RecommendationAgent(BaseAgent):
    OBJECTIVE = "Provide intelligent, context-aware recommendations by collaborating with other agents to gather comprehensive insights"

    def __init__(self, shared_memory: JurixSharedMemory):
        super().__init__(name="recommendation_agent", redis_client=shared_memory.redis_client)
        self.shared_memory = shared_memory
        self.model_manager = ModelManager()
        
        self.mental_state.capabilities = [
            AgentCapability.PROVIDE_RECOMMENDATIONS,
            AgentCapability.COORDINATE_AGENTS,
            AgentCapability.RETRIEVE_DATA
        ]
        
        self.mental_state.obligations.extend([
            "generate_recommendations",
            "assess_data_sufficiency",
            "request_collaboration",
            "synthesize_multi_source_data",
            "validate_recommendation_quality",
            "check_for_updates",
            "integrate_predictive_insights"
        ])
        
        self.log(f"Enhanced RecommendationAgent initialized with collaborative capabilities")

    def _check_for_updates(self):
        try:
            project_id = self.mental_state.get_belief("project")
            if not project_id:
                self.log("No project ID available for updates check")
                return
                
            if self.shared_memory.has_ticket_updates(project_id):
                self.log(f"Detected changes for project {project_id}")
                tickets = self.shared_memory.get_tickets(project_id)
                
                if tickets:
                    self.mental_state.add_belief("tickets", tickets, 0.9, "redis_update")
                    self.log(f"Updated tickets for project {project_id}: {len(tickets)} tickets")
                    
                    recommendations = self._generate_recommendations()
                    self.mental_state.add_belief("recommendations", recommendations, 0.9, "auto_update")
                    self.log(f"Generated new recommendations for project {project_id}: {recommendations}")
                    
                    self.shared_memory.mark_updates_processed(project_id)
            
        except Exception as e:
            self.log(f"[ERROR] Failed to check for updates: {str(e)}")

    def _check_for_updates_loop(self):
        while True:
            try:
                self._check_for_updates()
                self.log("Sleeping for 10 seconds before next update check")
                time.sleep(10)
            except Exception as e:
                self.log(f"[ERROR] Update loop error: {str(e)}")
                time.sleep(30)

    def _perceive(self, input_data: Dict[str, Any]) -> None:
        super()._perceive(input_data)
        
        session_id = input_data.get("session_id")
        user_prompt = input_data.get("user_prompt")
        articles = input_data.get("articles", [])
        project = input_data.get("project")
        tickets = input_data.get("tickets", [])
        history = self.shared_memory.get_conversation(session_id) if session_id else []
        workflow_type = input_data.get("workflow_type", "prompting")
        intent = input_data.get("intent", {}).get("intent", "generic_question")
        predictions = input_data.get("predictions", {})

        self.log(f"[PERCEPTION] Processing recommendation request: session={session_id}, prompt='{user_prompt}', project={project}, tickets={len(tickets)}, workflow={workflow_type}")

        self.mental_state.add_belief("session_id", session_id, 0.9, "input")
        self.mental_state.add_belief("user_prompt", user_prompt, 0.9, "input")
        self.mental_state.add_belief("conversation_history", history, 0.8, "memory")
        self.mental_state.add_belief("articles", articles, 0.9, "input")
        self.mental_state.add_belief("project", project, 0.9, "input")
        self.mental_state.add_belief("tickets", tickets, 0.9, "input")
        self.mental_state.add_belief("workflow_type", workflow_type, 0.9, "input")
        self.mental_state.add_belief("intent", intent, 0.9, "input")
        self.mental_state.add_belief("predictions", predictions, 0.9, "input")
        
        if input_data.get("collaboration_purpose"):
            collaboration_purpose = input_data.get("collaboration_purpose")
            self.mental_state.add_belief("collaboration_context", collaboration_purpose, 0.9, "collaboration")
            self.log(f"[COLLABORATION] Operating in collaborative mode: {collaboration_purpose}")
            
        if input_data.get("primary_agent_result"):
            primary_result = input_data.get("primary_agent_result")
            self.mental_state.add_belief("primary_agent_context", primary_result, 0.8, "collaboration")
            self.log(f"[COLLABORATION] Received primary agent context")
        
        self._assess_collaboration_needs(input_data)

    def _assess_collaboration_needs(self, input_data: Dict[str, Any]) -> None:
        user_prompt = input_data.get("user_prompt", "")
        tickets = input_data.get("tickets", [])
        project = input_data.get("project")
        articles = input_data.get("articles", [])
        predictions = input_data.get("predictions", {})
        
        self.log(f"[COLLABORATION ASSESSMENT] Evaluating needs for query: '{user_prompt}' with {len(tickets)} tickets")
        
        data_quality_score = self._calculate_data_quality_score(input_data)
        self.mental_state.add_belief("data_quality_score", data_quality_score, 0.9, "assessment")
        
        if len(tickets) < 3 and project:
            self.log("[COLLABORATION NEED] Insufficient ticket data detected")
            self.mental_state.request_collaboration(
                agent_type="jira_data_agent",
                reasoning_type="data_analysis", 
                context={
                    "reason": "insufficient_ticket_data", 
                    "project": project,
                    "current_ticket_count": len(tickets),
                    "minimum_needed": 5
                }
            )
        
        if not predictions and len(tickets) >= 5:
            self.log("[COLLABORATION NEED] No predictive insights available")
            self.mental_state.request_collaboration(
                agent_type="predictive_analysis_agent",
                reasoning_type="strategic_reasoning",
                context={
                    "reason": "need_predictive_insights",
                    "ticket_count": len(tickets)
                }
            )
        
        productivity_keywords = ["velocity", "performance", "efficiency", "bottleneck", "productivity", "throughput", "cycle time"]
        if any(keyword in user_prompt.lower() for keyword in productivity_keywords):
            self.log("[COLLABORATION NEED] Productivity analysis expertise needed")
            self.mental_state.request_collaboration(
                agent_type="productivity_dashboard_agent",
                reasoning_type="data_analysis",
                context={
                    "reason": "productivity_analysis_needed", 
                    "query": user_prompt,
                    "keywords_detected": [kw for kw in productivity_keywords if kw in user_prompt.lower()]
                }
            )
        
        if len(articles) == 0 and "article" not in user_prompt.lower():
            self.log("[COLLABORATION NEED] No contextual articles available")
            self.mental_state.request_collaboration(
                agent_type="retrieval_agent",
                reasoning_type="context_enrichment",
                context={
                    "reason": "missing_contextual_articles",
                    "query": user_prompt
                }
            )
        
        if self.mental_state.should_request_help("recommendation_generation", data_quality_score):
            self.log("[COLLABORATION NEED] Low confidence in recommendation quality")
            self.mental_state.request_collaboration(
                agent_type="knowledge_base_agent",
                reasoning_type="validation",
                context={
                    "reason": "quality_validation_needed",
                    "data_quality": data_quality_score
                }
            )

    def _calculate_data_quality_score(self, input_data: Dict[str, Any]) -> float:
        score = 0.0
        
        tickets = input_data.get("tickets", [])
        if tickets:
            ticket_score = min(len(tickets) / 10.0, 0.4)
            
            statuses = set()
            for ticket in tickets[:10]:
                status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
                statuses.add(status)
            
            diversity_bonus = min(len(statuses) / 4.0, 0.1)
            score += ticket_score + diversity_bonus
        
        articles = input_data.get("articles", [])
        if articles:
            article_score = min(len(articles) / 5.0, 0.3)
            score += article_score
        
        history = input_data.get("conversation_history", [])
        if history:
            history_score = min(len(history) / 10.0, 0.2)
            score += history_score
        
        user_prompt = input_data.get("user_prompt", "")
        if len(user_prompt) > 20:
            specificity_score = min(len(user_prompt.split()) / 20.0, 0.1)
            score += specificity_score
        
        predictions = input_data.get("predictions", {})
        if predictions:
            score += 0.2
        
        return min(score, 1.0)

    def _generate_recommendations(self) -> List[str]:
        user_prompt = self.mental_state.get_belief("user_prompt")
        project = self.mental_state.get_belief("project")
        conversation_history = self.mental_state.get_belief("conversation_history")
        articles = self.mental_state.get_belief("articles")
        tickets = self.mental_state.get_belief("tickets") or []
        predictions = self.mental_state.get_belief("predictions") or {}
        
        collaboration_context = self.mental_state.get_belief("collaboration_context")
        primary_agent_context = self.mental_state.get_belief("primary_agent_context")
        data_quality_score = self.mental_state.get_belief("data_quality_score") or 0.5

        self.log(f"[GENERATION] Creating recommendations for project {project} with {len(tickets)} tickets (quality score: {data_quality_score:.2f})")

        if not tickets and not collaboration_context:
            self.log("[GENERATION] Insufficient ticket data, requesting more context")
            self.shared_memory.store("needs_context", {"value": True, "timestamp": datetime.now().isoformat()})
            return ["Please provide more project-specific details for better recommendations."]

        article_context = self._build_article_context(articles)
        history_context = self._build_history_context(conversation_history)
        project_context = self._build_project_context(project, tickets)
        collaborative_context = self._build_collaborative_context(collaboration_context, primary_agent_context)
        predictive_context = self._build_predictive_context(predictions)

        prompt_template = f"""<|system|>You are an AI specialized in Agile methodologies and software development best practices. 
        Generate 3-5 specific, actionable recommendations related to this query: "{user_prompt}".
        
        Use project-specific data to tailor your recommendations, focusing on velocity, bottlenecks, resource allocation, or process improvements.
        
        Provide detailed explanations for each recommendation, formatted as separate strings in a list, without numbering or bullet points.
        Adopt a conversational tone, explaining the reasoning behind each suggestion.
        
        Data Quality Assessment: {data_quality_score:.1%} confidence level
        
        Relevant context from conversation:
        {history_context}
        
        Relevant articles:
        {article_context}
        
        Project-specific data:
        {project_context}
        
        {predictive_context}
        
        {collaborative_context}
        
        Return ONLY the list of recommendations.<|assistant|>"""

        try:
            response = self.model_manager.generate_response(
                prompt=prompt_template,
                context={
                    "agent_name": self.name,
                    "agent_capabilities": [cap.value for cap in self.mental_state.capabilities],
                    "task_type": "strategic_recommendation",
                    "data_quality_score": data_quality_score,
                    "has_tickets": bool(tickets),
                    "has_predictions": bool(predictions),
                    "collaboration_context": collaboration_context
                }
            )
            self.log(f"✅ {self.name} received response from model")
            self.log(f"[GENERATION] Generated response: {response[:200]}...")
            
            recommendations = self._parse_recommendations(response)
            
            if collaboration_context or primary_agent_context:
                recommendations = self._enhance_with_collaborative_insights(recommendations, collaboration_context, primary_agent_context)
            
            if predictions:
                recommendations = self._enhance_with_predictive_insights(recommendations, predictions)
            
            recommendations = self._validate_recommendation_quality(recommendations, data_quality_score)
            
            self.log(f"[GENERATION] Finalized {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            self.log(f"[ERROR] Failed to generate recommendations with LLM: {e}")
            return self._generate_fallback_recommendations(project, tickets, collaboration_context, predictions)

    def _build_collaborative_context(self, collaboration_context: str, primary_agent_context: Any) -> str:
        if not collaboration_context and not primary_agent_context:
            return ""
        
        context_parts = ["Collaborative Intelligence:"]
        
        if collaboration_context:
            context_parts.append(f"Collaboration Purpose: {collaboration_context}")
        
        if primary_agent_context:
            if isinstance(primary_agent_context, dict):
                if primary_agent_context.get("metrics"):
                    context_parts.append(f"Performance Metrics Available: {list(primary_agent_context['metrics'].keys())}")
                if primary_agent_context.get("articles_used"):
                    context_parts.append(f"Related Articles: {len(primary_agent_context['articles_used'])} articles referenced")
                if primary_agent_context.get("response"):
                    response_excerpt = str(primary_agent_context["response"])[:150]
                    context_parts.append(f"Previous Analysis: {response_excerpt}...")
        
        return "\n".join(context_parts) + "\n"

    def _build_predictive_context(self, predictions: Dict[str, Any]) -> str:
        if not predictions:
            return ""
        
        context_parts = ["Predictive Intelligence:"]
        
        sprint_completion = predictions.get("sprint_completion", {})
        if sprint_completion:
            context_parts.append(f"Sprint Completion Probability: {sprint_completion.get('probability', 0):.0%}")
            context_parts.append(f"Risk Level: {sprint_completion.get('risk_level', 'unknown')}")
            
            if sprint_completion.get("recommended_actions"):
                context_parts.append(f"Predictive Recommendations: {sprint_completion['recommended_actions'][:2]}")
        
        warnings = predictions.get("warnings", [])
        if warnings:
            context_parts.append(f"Critical Warnings: {len(warnings)} detected")
            for warning in warnings[:2]:
                context_parts.append(f"- {warning['message']}")
        
        velocity_forecast = predictions.get("velocity_forecast", {})
        if velocity_forecast:
            context_parts.append(f"Velocity Trend: {velocity_forecast.get('trend', 'unknown')}")
        
        return "\n".join(context_parts) + "\n"

    def _build_article_context(self, articles: List[Dict[str, Any]]) -> str:
        if not articles:
            return "No relevant articles available."
        
        article_summaries = []
        for i, article in enumerate(articles[:3]):
            title = article.get('title', 'No Title')
            content = article.get('content', '')[:200]
            article_summaries.append(f"[Article {i+1}] {title}: {content}...")
        
        return "\n".join(article_summaries)

    def _build_history_context(self, conversation_history: List[Dict[str, str]]) -> str:
        if not conversation_history:
            return "No previous conversation context."
        
        recent_messages = conversation_history[-3:]
        context_parts = []
        for msg in recent_messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:100]
            context_parts.append(f"{role}: {content}...")
        
        return "\n".join(context_parts)

    def _build_project_context(self, project: str, tickets: List[Dict[str, Any]]) -> str:
        if not project or not tickets:
            return "Limited project data available."
        
        status_distribution = {"To Do": 0, "In Progress": 0, "Done": 0}
        assignee_distribution = {}
        cycle_times = []
        
        for ticket in tickets:
            fields = ticket.get("fields", {})
            
            status = fields.get("status", {}).get("name", "Unknown")
            if status in status_distribution:
                status_distribution[status] += 1
            
            assignee_info = fields.get("assignee")
            if assignee_info:
                assignee = assignee_info.get("displayName", "Unknown")
                assignee_distribution[assignee] = assignee_distribution.get(assignee, 0) + 1
            
            changelog = ticket.get("changelog", {}).get("histories", [])
            start_date = None
            end_date = None
            
            for history in changelog:
                for item in history.get("items", []):
                    if item.get("field") == "status":
                        if item.get("toString") == "In Progress" and not start_date:
                            start_date = datetime.fromisoformat(history.get("created", "").replace("Z", "+00:00"))
                        elif item.get("toString") == "Done":
                            end_date = datetime.fromisoformat(history.get("created", "").replace("Z", "+00:00"))
            
            if start_date and end_date:
                cycle_time = (end_date - start_date).days
                if cycle_time >= 0:
                    cycle_times.append(cycle_time)
        
        avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0
        
        context_parts = [
            f"Project: {project}",
            f"Total Tickets Analyzed: {len(tickets)}",
            f"Status Distribution: {status_distribution}",
            f"Average Cycle Time: {avg_cycle_time:.1f} days"
        ]
        
        if assignee_distribution:
            top_assignee = max(assignee_distribution.items(), key=lambda x: x[1])
            context_parts.append(f"Most Active Assignee: {top_assignee[0]} ({top_assignee[1]} tickets)")
        
        if status_distribution["In Progress"] > status_distribution["Done"]:
            context_parts.append("⚠️ More tickets in progress than completed - potential bottleneck")
        
        if avg_cycle_time > 7:
            context_parts.append("⚠️ High cycle time detected - process efficiency opportunity")
        
        return "\n".join(context_parts)

    def _parse_recommendations(self, response: str) -> List[str]:
        recommendations = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-") and not line.startswith("*"):
                if line[0].isdigit() and len(line) > 2 and line[1] in [')', '.', ':']:
                    line = line[2:].strip()
                recommendations.append(line)
        
        if not recommendations:
            recommendations = ["Consider reviewing Agile best practices for general improvements."]
        
        return recommendations

    def _enhance_with_collaborative_insights(self, recommendations: List[str], 
                                           collaboration_context: str, 
                                           primary_agent_context: Any) -> List[str]:
        enhanced_recommendations = recommendations.copy()
        
        if collaboration_context == "productivity_analysis_needed":
            enhanced_recommendations.append(
                "Based on collaborative analysis: Consider implementing automated productivity tracking to identify efficiency patterns and bottlenecks in real-time."
            )
        
        if collaboration_context == "data_analysis" and primary_agent_context:
            enhanced_recommendations.append(
                "Cross-agent insight: The data analysis suggests implementing regular performance reviews to maintain the improvements identified in your current metrics."
            )
        
        if collaboration_context == "preventive_action_planning":
            enhanced_recommendations.append(
                "Proactive recommendation: Establish early warning systems based on the identified risk patterns to prevent issues before they impact delivery."
            )
        
        return enhanced_recommendations

    def _enhance_with_predictive_insights(self, recommendations: List[str], 
                                         predictions: Dict[str, Any]) -> List[str]:
        enhanced_recommendations = recommendations.copy()
        
        sprint_completion = predictions.get("sprint_completion", {})
        if sprint_completion and sprint_completion.get("probability", 1.0) < 0.7:
            probability = sprint_completion["probability"]
            actions = sprint_completion.get("recommended_actions", [])
            
            if actions:
                enhanced_recommendations.insert(0, 
                    f"Predictive insight: With only {probability:.0%} sprint completion probability, prioritize: {actions[0]}"
                )
        
        warnings = predictions.get("warnings", [])
        critical_warnings = [w for w in warnings if w.get("urgency") == "critical"]
        
        if critical_warnings:
            warning = critical_warnings[0]
            enhanced_recommendations.insert(0,
                f"⚠️ Critical prediction: {warning['message']}. Immediate action required: {warning['recommended_action']}"
            )
        
        risks = predictions.get("risks", [])
        high_risks = [r for r in risks if r.get("severity") == "high"]
        
        if high_risks and len(high_risks) >= 2:
            enhanced_recommendations.append(
                f"Risk mitigation: {len(high_risks)} high-severity risks detected. Focus on {high_risks[0]['mitigation']} to prevent sprint failure."
            )
        
        return enhanced_recommendations

    def _validate_recommendation_quality(self, recommendations: List[str], data_quality_score: float) -> List[str]:
        if data_quality_score < 0.5:
            qualified_recommendations = []
            for rec in recommendations:
                if "consider" not in rec.lower():
                    rec = f"Consider {rec.lower()}"
                qualified_recommendations.append(rec)
            
            qualified_recommendations.append(
                f"Note: These recommendations are based on limited data (confidence: {data_quality_score:.1%}). "
                "Providing more project context would enable more specific suggestions."
            )
            return qualified_recommendations
        
        return recommendations

    def _generate_fallback_recommendations(self, project: str, tickets: List[Dict[str, Any]], 
                                        collaboration_context: str, predictions: Dict[str, Any]) -> List[str]:
        fallback_recs = [
            "Consider automating testing to reduce delays in the development pipeline, as this can streamline workflows and improve delivery times.",
            "Schedule a team training session on CI/CD optimizations to boost efficiency and reduce manual overhead.",
            "Review workload distribution to balance task allocation and prevent bottlenecks in your development process."
        ]
        
        if tickets:
            status_counts = {}
            for ticket in tickets:
                status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if status_counts.get("In Progress", 0) > status_counts.get("Done", 0):
                fallback_recs.append("Focus on completing in-progress work before starting new tasks to improve throughput.")
        
        if collaboration_context:
            fallback_recs.append(
                f"Based on {collaboration_context}: Consider establishing regular cross-team collaboration sessions to leverage diverse expertise."
            )
        
        if predictions and predictions.get("sprint_completion", {}).get("probability", 1.0) < 0.7:
            fallback_recs.insert(0, "Priority: Take immediate action to improve sprint completion probability by reducing scope or adding resources.")
        
        return fallback_recs

    def _act(self) -> Dict[str, Any]:
        try:
            user_prompt = self.mental_state.get_belief("user_prompt")
            project = self.mental_state.get_belief("project")
            collaboration_context = self.mental_state.get_belief("collaboration_context")
            data_quality_score = self.mental_state.get_belief("data_quality_score") or 0.5
            predictions = self.mental_state.get_belief("predictions")
            
            if hasattr(self.mental_state, 'add_experience'):
                experience_description = f"Processing {'collaborative ' if collaboration_context else ''}recommendation request for project {project}"
                if collaboration_context:
                    experience_description += f" (collaboration: {collaboration_context})"
                if predictions:
                    experience_description += " with predictive insights"
                
                self.mental_state.add_experience(
                    experience_description=experience_description,
                    outcome="generating_recommendations",
                    confidence=0.8,
                    metadata={
                        "project": project,
                        "prompt": user_prompt[:100] if user_prompt else "",
                        "collaboration_context": collaboration_context,
                        "data_quality": data_quality_score,
                        "has_predictions": bool(predictions)
                    }
                )
            
            recommendations = self._generate_recommendations()
            self.log(f"[ACTION] Generated {len(recommendations)} recommendations")
            
            if hasattr(self.mental_state, 'add_experience'):
                self.mental_state.add_experience(
                    experience_description=f"Generated {len(recommendations)} recommendations for {project}",
                    outcome="recommendations_generated",
                    confidence=0.9,
                    metadata={
                        "project": project,
                        "recommendation_count": len(recommendations),
                        "recommendations": recommendations[:3],
                        "collaborative": bool(collaboration_context),
                        "data_quality": data_quality_score,
                        "predictions_integrated": bool(predictions)
                    }
                )
            
            needs_context_data = self.shared_memory.get("needs_context")
            needs_context = needs_context_data.get("value", False) if needs_context_data else False
            
            decision = {
                "action": "generate_recommendations",
                "recommendation_count": len(recommendations),
                "context_sufficient": not needs_context,
                "project": project,
                "collaborative": bool(collaboration_context),
                "predictions_integrated": bool(predictions),
                "collaboration_requests_made": len(self.mental_state.collaborative_requests),
                "data_quality_score": data_quality_score,
                "reasoning": f"Generated {len(recommendations)} {'collaborative ' if collaboration_context else ''}recommendations based on available context"
            }
            self.mental_state.add_decision(decision)
            
            return {
                "recommendations": recommendations,
                "workflow_status": "success",
                "needs_context": needs_context,
                "collaboration_metadata": {
                    "is_collaborative": bool(collaboration_context),
                    "collaboration_context": collaboration_context,
                    "data_quality_score": data_quality_score,
                    "collaboration_requests_made": len(self.mental_state.collaborative_requests),
                    "predictions_integrated": bool(predictions)
                }
            }
            
        except Exception as e:
            self.log(f"[ERROR] Failed to generate recommendations: {e}")
            
            if hasattr(self.mental_state, 'add_experience'):
                self.mental_state.add_experience(
                    experience_description=f"Failed to generate recommendations",
                    outcome=f"Error: {str(e)}",
                    confidence=0.2,
                    metadata={"error_type": type(e).__name__}
                )
            
            return {
                "recommendations": [],
                "workflow_status": "failure",
                "needs_context": True,
                "collaboration_metadata": {
                    "is_collaborative": False,
                    "error_occurred": True,
                    "error_message": str(e)
                }
            }

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        super()._rethink(action_result)
        
        recommendations = action_result.get("recommendations", [])
        collaboration_metadata = action_result.get("collaboration_metadata", {})
        quality_score = min(len(recommendations) / 3.0, 1.0)
        
        reflection = {
            "operation": "recommendation_generation",
            "success": action_result.get("workflow_status") == "success",
            "recommendation_count": len(recommendations),
            "quality_score": quality_score,
            "context_availability": not action_result.get("needs_context", True),
            "collaborative_interaction": collaboration_metadata.get("is_collaborative", False),
            "data_quality_score": collaboration_metadata.get("data_quality_score", 0.5),
            "predictions_integrated": collaboration_metadata.get("predictions_integrated", False),
            "collaboration_requests_made": collaboration_metadata.get("collaboration_requests_made", 0),
            "performance_notes": f"Generated {len(recommendations)} recommendations with quality score {quality_score:.2f}"
        }
        
        if collaboration_metadata.get("is_collaborative"):
            reflection["collaboration_context"] = collaboration_metadata.get("collaboration_context")
            reflection["performance_notes"] += " (collaborative mode)"
        
        if collaboration_metadata.get("predictions_integrated"):
            reflection["performance_notes"] += " with predictive insights"
        
        self.mental_state.add_reflection(reflection)
        
        self.log(f"[REFLECTION] Completed: {reflection}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(input_data)
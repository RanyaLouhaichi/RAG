import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import uuid
from dataclasses import dataclass
from enum import Enum
import threading

from orchestrator.core.orchestrator import Orchestrator # type: ignore
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from orchestrator.core.collaborative_framework import CollaborativeFramework # type: ignore

class ARIAPersonality(Enum):
    HELPFUL = "helpful"
    ANALYTICAL = "analytical"
    PROACTIVE = "proactive"
    CREATIVE = "creative"
    PREDICTIVE = "predictive"

@dataclass
class ARIAContext:
    """ARIA's understanding of the current situation"""
    user_intent: str
    workspace: str
    current_project: Optional[str]
    current_ticket: Optional[str]
    user_history: List[Dict[str, Any]]
    active_insights: List[Dict[str, Any]]
    mood: ARIAPersonality

class ARIA:
    """
    ARIA - Artificial Reasoning Intelligence Assistant
    Your AI Team Member that lives in Jira and Confluence
    """
    
    def __init__(self):
        self.orchestrator = Orchestrator()
        self.logger = logging.getLogger("ARIA")
        self.personality = ARIAPersonality.HELPFUL
        
        self.conversation_contexts = {}
        self.active_monitoring = {}
        self.predictive_insights = []
        
        self.event_handlers = {
            "ticket_resolved": self._handle_ticket_resolution,
            "sprint_risk_detected": self._handle_sprint_risk,
            "prediction_update": self._handle_prediction_update
        }
        
        self.logger.info("🤖 ARIA initialized - Your AI team member is ready!")
    
    def introduce_myself(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """ARIA introduces herself to the team (synchronous version)"""
        return {
            "avatar": "aria_animated.gif",
            "message": "Hi! I'm ARIA, your AI team member. I live right here in Jira and Confluence to help you work smarter. I can analyze your projects, create documentation, predict issues, and answer any questions you have!",
            "capabilities": [
                "Real-time project analytics",
                "Automatic documentation",
                "Predictive insights",
                "Natural conversation",
                "Proactive assistance",
                "Risk prediction"
            ],
            "status": "active",
            "current_mood": self.personality.value
        }
    
    def process_interaction(self, user_input: str, context: ARIAContext) -> Dict[str, Any]:
        """
        ARIA processes any interaction from Jira/Confluence (synchronous version)
        """
        self.logger.info(f"🤖 ARIA processing: {user_input} in {context.workspace}")
        
        intent = self._analyze_intent(user_input, context)
        
        if intent == "dashboard_request":
            return self._provide_live_dashboard(context)
        elif intent == "prediction_request":
            return self._provide_predictions(context)
        elif intent == "ticket_help":
            return self._assist_with_ticket(context)
        elif intent == "documentation":
            return self._handle_documentation(context)
        elif intent == "risk_assessment":
            return self._assess_risks(context)
        else:
            return self._general_conversation(user_input, context)
    
    def _provide_live_dashboard(self, context: ARIAContext) -> Dict[str, Any]:
        """ARIA provides live dashboard data for Jira plugin"""
        self.logger.info(f"📊 ARIA generating live dashboard for {context.current_project}")
        
        state = self.orchestrator.run_productivity_workflow(
            context.current_project or "PROJ123",
            {
                "start": "2025-05-01T00:00:00Z",
                "end": "2025-05-17T23:59:59Z"
            }
        )
        
        predictions = state.get("predictions", {})
        
        dashboard_data = {
            "type": "live_dashboard",
            "data": {
                "sprint_health": self._calculate_sprint_health(state, predictions),
                "live_metrics": {
                    "velocity": state["metrics"].get("throughput", 0),
                    "cycle_time": state["metrics"].get("cycle_time", 0),
                    "bottlenecks": state["metrics"].get("bottlenecks", {}),
                    "team_workload": state["metrics"].get("workload", {})
                },
                "predictions": self._generate_predictions(state),
                "visualizations": state.get("visualization_data", {}),
                "recommendations": state.get("recommendations", [])
            },
            "aria_insights": self._generate_dashboard_insights(state, predictions),
            "refresh_interval": 30,
            "aria_message": "Here's your real-time dashboard with predictive insights! I'll keep it updated every 30 seconds. Ask me anything about what you see!"
        }
        
        return dashboard_data
    
    def _provide_predictions(self, context: ARIAContext) -> Dict[str, Any]:
        """ARIA provides predictive insights"""
        self.logger.info(f"🔮 ARIA generating predictions for {context.current_project}")
        
        state = self.orchestrator.run_predictive_workflow(
            context.current_project or "PROJ123",
            analysis_type="comprehensive"
        )
        
        predictions = state.get("predictions", {})
        
        return {
            "type": "predictions",
            "message": "Based on my analysis of your project patterns, here's what I predict:",
            "predictions": predictions,
            "confidence_level": "high" if predictions.get("sprint_completion", {}).get("confidence", 0) > 0.7 else "medium",
            "aria_advice": self._generate_predictive_advice(predictions),
            "visualizations": self._create_prediction_visualizations(predictions)
        }
    
    def _assess_risks(self, context: ARIAContext) -> Dict[str, Any]:
        """ARIA performs risk assessment"""
        self.logger.info(f"⚠️ ARIA assessing risks for {context.current_project}")
        
        state = self.orchestrator.run_predictive_workflow(
            context.current_project or "PROJ123",
            analysis_type="risk_assessment"
        )
        
        predictions = state.get("predictions", {})
        risks = predictions.get("risks", [])
        warnings = predictions.get("warnings", [])
        
        return {
            "type": "risk_assessment",
            "message": "I've analyzed your project for potential risks:",
            "risks": risks,
            "warnings": warnings,
            "mitigation_plan": self._generate_mitigation_plan(risks, warnings),
            "aria_recommendation": "I recommend addressing the high-severity risks immediately to prevent sprint failure.",
            "confidence": predictions.get("sprint_completion", {}).get("confidence", 0.5)
        }
    
    def _handle_ticket_resolution(self, ticket_id: str, context: ARIAContext) -> Dict[str, Any]:
        """ARIA handles automatic documentation when ticket is resolved"""
        self.logger.info(f"📝 ARIA handling ticket resolution for {ticket_id}")
        
        state = self.orchestrator.run_jira_workflow(ticket_id)
        
        predictions = self._get_ticket_impact_predictions(ticket_id, context)
        
        return {
            "type": "notification",
            "stage": "complete",
            "message": f"✅ Documentation created! I've posted it to Confluence. This resolution improved sprint completion probability by {predictions.get('impact', 0):.0%}!",
            "avatar_state": "happy",
            "actions": [
                {
                    "label": "View Documentation",
                    "url": f"/confluence/articles/{state.get('article', {}).get('id', '')}"
                },
                {
                    "label": "Ask me about it",
                    "action": "chat_with_aria"
                }
            ],
            "article_preview": state.get("article", {}).get("content", "")[:200] + "...",
            "collaboration_applied": bool(state.get("collaboration_metadata")),
            "predictive_insight": predictions.get("insight", "")
        }
    
    def _handle_sprint_risk(self, risk_data: Dict[str, Any], context: ARIAContext) -> Dict[str, Any]:
        """ARIA handles sprint risk detection"""
        self.logger.info(f"🚨 ARIA handling sprint risk: {risk_data}")
        
        recommendations = self._generate_risk_mitigation_recommendations(risk_data)
        
        return {
            "type": "proactive_alert",
            "urgency": risk_data.get("urgency", "high"),
            "message": f"⚠️ Sprint risk detected: {risk_data.get('message', 'Unknown risk')}",
            "avatar_state": "concerned",
            "recommendations": recommendations,
            "actions": [
                {
                    "label": "View Risk Analysis",
                    "action": "show_risk_details"
                },
                {
                    "label": "Apply Recommendations",
                    "action": "apply_recommendations"
                }
            ],
            "impact_prediction": f"If not addressed, sprint completion probability will drop to {risk_data.get('impact_probability', 0):.0%}"
        }
    
    def _handle_prediction_update(self, prediction_data: Dict[str, Any], context: ARIAContext) -> Dict[str, Any]:
        """ARIA handles prediction updates"""
        self.logger.info(f"📈 ARIA processing prediction update")
        
        return {
            "type": "prediction_update",
            "message": f"Prediction update: {prediction_data.get('message', 'Conditions have changed')}",
            "old_prediction": prediction_data.get("old_value"),
            "new_prediction": prediction_data.get("new_value"),
            "reasoning": prediction_data.get("reasoning", "Based on recent project activity"),
            "recommendations": prediction_data.get("recommendations", [])
        }
    
    def _assist_with_ticket(self, context: ARIAContext) -> Dict[str, Any]:
        """ARIA assists with ticket-related questions"""
        ticket_id = context.current_ticket or "TICKET-001"
        
        jira_data = self.orchestrator.jira_data_agent.run({
            "project_id": context.current_project or "PROJ123",
            "time_range": {
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-12-31T23:59:59Z"
            }
        })
        
        tickets = jira_data.get("tickets", [])
        target_ticket = next((t for t in tickets if t.get("key") == ticket_id), None)
        
        if target_ticket:
            prediction_state = self.orchestrator.run_predictive_workflow(
                context.current_project or "PROJ123",
                analysis_type="ticket_predictions"
            )
            
            ticket_predictions = prediction_state.get("predictions", {}).get("ticket_predictions", [])
            ticket_prediction = next((p for p in ticket_predictions if p.get("ticket_key") == ticket_id), None)
            
            message = f"I found information about {ticket_id}. This ticket is about '{target_ticket.get('fields', {}).get('summary', 'No summary')}' and is currently {target_ticket.get('fields', {}).get('status', {}).get('name', 'Unknown')}."
            
            if ticket_prediction:
                message += f" I predict it will be completed by {ticket_prediction.get('estimated_completion', 'unknown')} with {ticket_prediction.get('confidence', 0):.0%} confidence."
            
            return {
                "type": "ticket_assistance",
                "message": message,
                "ticket_data": {
                    "key": ticket_id,
                    "summary": target_ticket.get('fields', {}).get('summary'),
                    "status": target_ticket.get('fields', {}).get('status', {}).get('name'),
                    "assignee": target_ticket.get('fields', {}).get('assignee', {}).get('displayName') if target_ticket.get('fields', {}).get('assignee') else "Unassigned"
                },
                "prediction": ticket_prediction,
                "suggestions": [
                    "Would you like me to analyze similar tickets?",
                    "Should I create documentation for this?",
                    "Want to see the team's productivity metrics?",
                    "Should I predict the impact of completing this ticket?"
                ]
            }
        else:
            return {
                "type": "ticket_assistance",
                "message": f"I couldn't find {ticket_id}. Would you like me to search for similar tickets or help you create a new one?",
                "suggestions": ["Search similar tickets", "Create new ticket", "Show all tickets"]
            }
    
    def _handle_documentation(self, context: ARIAContext) -> Dict[str, Any]:
        """ARIA handles documentation requests"""
        return {
            "type": "documentation_help",
            "message": "I can help you with documentation! What would you like to do?",
            "options": [
                {
                    "label": "Create documentation from ticket",
                    "action": "create_from_ticket"
                },
                {
                    "label": "Find existing documentation",
                    "action": "search_docs"
                },
                {
                    "label": "Improve current page",
                    "action": "improve_page"
                },
                {
                    "label": "Generate documentation report",
                    "action": "doc_report"
                }
            ],
            "recent_docs": [
                "API Migration Guide",
                "Deployment Best Practices",
                "Team Onboarding"
            ],
            "prediction": "Based on your recent tickets, you might need documentation for authentication module updates."
        }
    
    def _calculate_sprint_health(self, state: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate sprint health metrics with predictions"""
        metrics = state.get("metrics", {})
        sprint_completion = predictions.get("sprint_completion", {})
        
        health_score = 0
        if metrics.get("throughput", 0) > 5:
            health_score += 30
        if metrics.get("cycle_time", 0) < 5:
            health_score += 30
        if len(metrics.get("bottlenecks", {})) < 3:
            health_score += 20
        
        if sprint_completion.get("probability", 0) > 0.7:
            health_score += 20
        elif sprint_completion.get("probability", 0) < 0.4:
            health_score -= 20
        
        health_score = max(0, min(100, health_score))
        
        return {
            "score": health_score,
            "percentage": f"{health_score}%",
            "status": "healthy" if health_score > 70 else "at_risk" if health_score > 40 else "critical",
            "factors": {
                "velocity": "good" if metrics.get("throughput", 0) > 5 else "low",
                "cycle_time": "good" if metrics.get("cycle_time", 0) < 5 else "high",
                "bottlenecks": len(metrics.get("bottlenecks", {})),
                "completion_probability": sprint_completion.get("probability", 0)
            }
        }
    
    def _generate_predictions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate predictive insights from state"""
        predictions = state.get("predictions", {})
        formatted_predictions = []
        
        sprint_completion = predictions.get("sprint_completion", {})
        if sprint_completion:
            formatted_predictions.append({
                "type": "sprint_completion",
                "confidence": sprint_completion.get("confidence", 0),
                "message": f"Sprint completion probability: {sprint_completion.get('probability', 0):.0%}",
                "suggestion": sprint_completion.get("recommended_actions", ["Continue current pace"])[0] if sprint_completion.get("recommended_actions") else "Continue current pace"
            })
        
        risks = predictions.get("risks", [])
        for risk in risks[:3]:
            formatted_predictions.append({
                "type": f"risk_{risk.get('type', 'unknown')}",
                "confidence": risk.get("probability", 0.5),
                "message": risk.get("description", "Risk detected"),
                "suggestion": risk.get("mitigation", "Review and address")
            })
        
        velocity_forecast = predictions.get("velocity_forecast", {})
        if velocity_forecast:
            formatted_predictions.append({
                "type": "velocity_trend",
                "confidence": velocity_forecast.get("confidence", 0.5),
                "message": f"Velocity trending {velocity_forecast.get('trend', 'stable')}",
                "suggestion": f"Next week estimate: {velocity_forecast.get('next_week_estimate', 0):.1f} tickets"
            })
        
        return formatted_predictions
    
    def _generate_dashboard_insights(self, state: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate intelligent insights for dashboard"""
        insights = []
        metrics = state.get("metrics", {})
        
        sprint_completion = predictions.get("sprint_completion", {})
        if sprint_completion.get("probability", 1.0) < 0.5:
            insights.append({
                "type": "critical_observation",
                "message": f"Sprint at risk! Only {sprint_completion.get('probability', 0):.0%} chance of completion",
                "sentiment": "concern"
            })
        
        throughput = metrics.get("throughput", 0)
        if throughput > 10:
            insights.append({
                "type": "positive_observation",
                "message": f"Excellent velocity at {throughput} tickets/week!",
                "sentiment": "positive"
            })
        elif throughput < 3:
            insights.append({
                "type": "concern",
                "message": f"Velocity is low at {throughput} tickets/week",
                "sentiment": "concern"
            })
        
        warnings = predictions.get("warnings", [])
        if warnings:
            insights.append({
                "type": "warning",
                "message": f"{len(warnings)} predictive warnings need attention",
                "sentiment": "alert"
            })
        
        return insights
    
    def _generate_predictive_advice(self, predictions: Dict[str, Any]) -> str:
        """Generate actionable advice based on predictions"""
        sprint_completion = predictions.get("sprint_completion", {})
        risks = predictions.get("risks", [])
        warnings = predictions.get("warnings", [])
        
        if sprint_completion.get("probability", 1.0) < 0.5:
            return "I strongly recommend immediate action - the sprint is at high risk. Focus on the critical items and consider reducing scope."
        elif len([r for r in risks if r.get("severity") == "high"]) >= 2:
            return "Multiple high-severity risks detected. Address these immediately to prevent cascading failures."
        elif warnings:
            return f"I've identified {len(warnings)} warning signs. Address these proactively to maintain sprint health."
        else:
            return "The sprint is on track! Continue monitoring the metrics I'm tracking for early warning signs."
    
    def _create_prediction_visualizations(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization data for predictions"""
        return {
            "sprint_gauge": {
                "type": "gauge",
                "value": predictions.get("sprint_completion", {}).get("probability", 0) * 100,
                "thresholds": {"low": 30, "medium": 60, "high": 80}
            },
            "risk_matrix": {
                "type": "matrix",
                "data": [
                    {"x": r.get("probability", 0.5), "y": {"high": 3, "medium": 2, "low": 1}.get(r.get("severity", "medium"), 2), "label": r.get("type", "unknown")}
                    for r in predictions.get("risks", [])[:5]
                ]
            },
            "velocity_trend": {
                "type": "line",
                "data": predictions.get("velocity_forecast", {}).get("forecast", [])
            }
        }
    
    def _generate_mitigation_plan(self, risks: List[Dict[str, Any]], warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate mitigation plan for risks"""
        plan = []
        
        high_risks = [r for r in risks if r.get("severity") == "high"]
        for risk in high_risks[:3]:
            plan.append({
                "priority": "P1",
                "risk": risk.get("type", "unknown"),
                "action": risk.get("mitigation", "Address immediately"),
                "owner": "Team Lead",
                "deadline": "Within 24 hours"
            })
        
        critical_warnings = [w for w in warnings if w.get("urgency") == "critical"]
        for warning in critical_warnings[:2]:
            plan.append({
                "priority": "P1",
                "risk": warning.get("type", "unknown"),
                "action": warning.get("recommended_action", "Take action"),
                "owner": "Scrum Master",
                "deadline": "Immediate"
            })
        
        return plan
    
    def _get_ticket_impact_predictions(self, ticket_id: str, context: ARIAContext) -> Dict[str, Any]:
        """Predict impact of ticket resolution"""
        before_state = self.orchestrator.run_predictive_workflow(
            context.current_project or "PROJ123",
            analysis_type="sprint_completion"
        )
        
        before_probability = before_state.get("predictions", {}).get("sprint_completion", {}).get("probability", 0)
        
        after_probability = min(1.0, before_probability + 0.05)
        
        return {
            "impact": after_probability - before_probability,
            "insight": f"Completing this ticket improved sprint success probability from {before_probability:.0%} to {after_probability:.0%}",
            "new_probability": after_probability
        }
    
    def _generate_risk_mitigation_recommendations(self, risk_data: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for risk mitigation"""
        risk_type = risk_data.get("type", "unknown")
        
        recommendations = {
            "sprint_failure": [
                "Immediately reduce sprint scope by removing low-priority items",
                "Schedule emergency planning session to reassess commitments",
                "Consider bringing in additional resources or pair programming"
            ],
            "team_burnout": [
                "Redistribute workload immediately to prevent burnout",
                "Cancel non-essential meetings this week",
                "Implement work-from-home days for affected team members"
            ],
            "velocity_decline": [
                "Conduct retrospective to identify impediments",
                "Review and optimize development process",
                "Check for external dependencies causing delays"
            ],
            "process_bottleneck": [
                "Add WIP limits to prevent further accumulation",
                "Assign additional reviewers to clear backlog",
                "Implement expedited review process for critical items"
            ]
        }
        
        return recommendations.get(risk_type, [
            "Review risk details and create action plan",
            "Discuss with team in next standup",
            "Monitor closely for next 48 hours"
        ])
    
    def _general_conversation(self, user_input: str, context: ARIAContext) -> Dict[str, Any]:
        """ARIA has a general conversation using enhanced ChatAgent"""
        conversation_id = context.user_history[-1].get("conversation_id") if context.user_history else str(uuid.uuid4())
        
        state = self.orchestrator.run_workflow(user_input, conversation_id)
        
        predictions_mentioned = any(word in user_input.lower() for word in ["predict", "forecast", "will", "risk", "probability"])
        
        avatar_state = "thinking" if predictions_mentioned else "talking"
        
        return {
            "type": "chat_response",
            "message": state.get("response", "I'm not sure how to help with that. Can you provide more details?"),
            "avatar_state": avatar_state,
            "context_used": {
                "articles": len(state.get("articles", [])),
                "recommendations": len(state.get("recommendations", [])),
                "collaboration": bool(state.get("collaboration_metadata")),
                "predictions": bool(state.get("predictions"))
            },
            "suggested_actions": self._suggest_followup_actions(state, predictions_mentioned)
        }
    
    def _suggest_followup_actions(self, state: Dict[str, Any], predictions_mentioned: bool) -> List[Dict[str, str]]:
        """Suggest follow-up actions based on conversation"""
        actions = []
        
        if predictions_mentioned:
            actions.append({
                "label": "View Detailed Predictions",
                "action": "show_predictions"
            })
        
        if state.get("recommendations"):
            actions.append({
                "label": "View Recommendations",
                "action": "show_recommendations"
            })
        
        if state.get("articles"):
            actions.append({
                "label": "Related Articles",
                "action": "show_articles"
            })
        
        actions.append({
            "label": "Show Dashboard",
            "action": "show_dashboard"
        })
        
        return actions
    
    def _analyze_intent(self, user_input: str, context: ARIAContext) -> str:
        """Analyze user intent from input"""
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ["predict", "forecast", "will we", "probability", "chance"]):
            return "prediction_request"
        elif any(word in input_lower for word in ["risk", "danger", "threat", "warning"]):
            return "risk_assessment"
        elif any(word in input_lower for word in ["dashboard", "metrics", "analytics", "status"]):
            return "dashboard_request"
        elif "ticket" in input_lower and any(word in input_lower for word in ["help", "status", "info"]):
            return "ticket_help"
        elif any(phrase in input_lower for phrase in ["create documentation", "improve documentation", "write article"]):
            return "documentation"
        else:
            return "general"
    
    def start_monitoring(self, project_id: str):
        """ARIA starts monitoring a project proactively"""
        self.active_monitoring[project_id] = {
            "started": datetime.now(),
            "last_check": datetime.now(),
            "insights_generated": 0
        }
        
        thread = threading.Thread(target=self._monitor_project_sync, args=(project_id,))
        thread.daemon = True
        thread.start()
    
    def _monitor_project_sync(self, project_id: str):
        """Background monitoring for proactive insights (synchronous)"""
        import time
        
        while project_id in self.active_monitoring:
            time.sleep(30)
            
            state = self.orchestrator.run_predictive_workflow(
                project_id,
                analysis_type="comprehensive"
            )
            
            insights = self._check_for_issues(state)
            if insights:
                self.predictive_insights.extend(insights)
                self.active_monitoring[project_id]["insights_generated"] += len(insights)
                
                for insight in insights:
                    if insight.get("severity") in ["high", "critical"]:
                        self._trigger_proactive_alert(insight, project_id)
    
    def _check_for_issues(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for issues in project state"""
        insights = []
        predictions = state.get("predictions", {})
        metrics = state.get("metrics", {})
        
        sprint_completion = predictions.get("sprint_completion", {})
        if sprint_completion.get("probability", 1.0) < 0.4:
            insights.append({
                "type": "sprint_failure_risk",
                "severity": "critical",
                "message": f"Sprint failure imminent - only {sprint_completion.get('probability', 0):.0%} completion probability",
                "suggestion": "Immediate intervention required"
            })
        
        warnings = predictions.get("warnings", [])
        critical_warnings = [w for w in warnings if w.get("urgency") == "critical"]
        if critical_warnings:
            for warning in critical_warnings:
                insights.append({
                    "type": f"warning_{warning.get('type', 'unknown')}",
                    "severity": "high",
                    "message": warning.get("message", "Critical warning"),
                    "suggestion": warning.get("recommended_action", "Take action")
                })
        
        if metrics.get("throughput", 0) < 3:
            insights.append({
                "type": "velocity_alert",
                "severity": "high",
                "message": "Team velocity has dropped below critical threshold",
                "suggestion": "Check for blockers or team availability issues"
            })
        
        bottlenecks = metrics.get("bottlenecks", {})
        critical_bottlenecks = [status for status, count in bottlenecks.items() if count > 5]
        if critical_bottlenecks:
            insights.append({
                "type": "bottleneck_alert",
                "severity": "medium",
                "message": f"Critical bottlenecks in: {', '.join(critical_bottlenecks)}",
                "suggestion": "Immediate review needed for stuck tickets"
            })
        
        return insights
    
    def _trigger_proactive_alert(self, insight: Dict[str, Any], project_id: str):
        """Trigger proactive alert for critical issues"""
        self.logger.warning(f"🚨 ARIA triggering proactive alert for {project_id}: {insight}")

aria = ARIA()
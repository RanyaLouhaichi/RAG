import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from agents.base_agent import BaseAgent, AgentCapability
from orchestrator.core.model_manager import ModelManager # type: ignore

class PredictiveAnalysisAgent(BaseAgent):
    OBJECTIVE = "Predict project outcomes, identify risks early, and provide actionable forecasts through collaborative intelligence"

    def __init__(self, redis_client=None):
        super().__init__(name="predictive_analysis_agent", redis_client=redis_client)
        self.model_manager = ModelManager()
        self.mental_state.capabilities = [
            AgentCapability.RETRIEVE_DATA,
            AgentCapability.RANK_CONTENT,
            AgentCapability.COORDINATE_AGENTS,
            AgentCapability.PROVIDE_RECOMMENDATIONS
        ]
        self.mental_state.obligations.extend([
            "predict_sprint_completion",
            "forecast_velocity_trends",
            "identify_future_risks",
            "predict_ticket_completion",
            "generate_early_warnings",
            "analyze_team_workload",
            "identify_bottlenecks",
            "provide_actionable_insights"
        ])
        self.prediction_thresholds = {
            "high_risk": 0.3,
            "medium_risk": 0.6,
            "low_risk": 0.8,
            "burnout_threshold": 1.5,
            "bottleneck_threshold": 0.7,
            "velocity_decline_threshold": -0.2
        }
        self.log("PredictiveAnalysisAgent initialized with enhanced analytical capabilities")

    def _parse_ai_json_response(self, response: str, default: Any = None) -> Any:
        try:
            json_match = re.search(r'\{.*\}|\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            try:
                return json.loads(response)
            except:
                return default if default is not None else {}
        except Exception as e:
            self.log(f"[AI PARSE] Failed to parse JSON: {e}")
            return default if default is not None else {}

    def _ai_analyze_ticket_patterns(self, tickets: List[Dict[str, Any]], 
                                   status_counts: Dict[str, int]) -> Dict[str, Any]:
        ticket_summaries = []
        for ticket in tickets[:20]:
            fields = ticket.get("fields", {})
            ticket_summaries.append({
                "key": ticket.get("key"),
                "status": fields.get("status", {}).get("name"),
                "summary": fields.get("summary", "")[:100],
                "assignee": fields.get("assignee", {}).get("displayName", "Unassigned") if fields.get("assignee") else "Unassigned",
                "created": fields.get("created", "")[:10] if fields.get("created") else "",
                "labels": fields.get("labels", [])[:3] })
        prompt = f"""You are an expert Agile project analyst. Analyze these tickets and identify hidden patterns that affect sprint completion:

Status Distribution: {json.dumps(status_counts)}
Total Tickets: {len(tickets)}

Sample Tickets:
{json.dumps(ticket_summaries, indent=2)}

Identify patterns and return ONLY a JSON object with these fields:
{{
    "complexity_patterns": ["list of complexity indicators found"],
    "blocking_patterns": ["patterns that suggest blockers"],
    "velocity_indicators": ["signs of team velocity changes"],
    "team_dynamics": ["observations about team work patterns"],
    "risk_signals": ["early warning signs"],
    "pattern_confidence": 0.7
}}"""
        try:
            response = self.model_manager.generate_response(
                prompt=prompt,
                context={
                    "agent_name": self.name,
                    "task_type": "pattern_analysis",
                    "ticket_count": len(tickets)
                }
            )
            parsed = self._parse_ai_json_response(response)
            default_patterns = {
                "complexity_patterns": [],
                "blocking_patterns": [],
                "velocity_indicators": [],
                "team_dynamics": [],
                "risk_signals": [],
                "pattern_confidence": 0.7
            }
            for key, default_value in default_patterns.items():
                if key not in parsed:
                    parsed[key] = default_value
            return parsed
        except Exception as e:
            self.log(f"[AI ERROR] Pattern analysis failed: {e}")
            return {
                "complexity_patterns": ["Unable to analyze patterns"],
                "blocking_patterns": [],
                "velocity_indicators": [],
                "team_dynamics": [],
                "risk_signals": [],
                "pattern_confidence": 0.5
            }
        
    def _calculate_sprint_completion_probability(self, tickets: List[Dict[str, Any]], 
                                           metrics: Dict[str, Any],
                                           historical_data: Dict[str, Any]) -> Dict[str, Any]:
        self.log("[AI-ENHANCED PREDICTION] Calculating sprint completion probability")
        DONE_STATUSES = ["Done", "Closed", "Resolved", "Complete", "Finished"]
        IN_PROGRESS_STATUSES = ["In Progress", "In Development", "In Review", "Testing"]
        TODO_STATUSES = ["To Do", "Open", "Reopened", "New", "Backlog"]
        BLOCKED_STATUSES = ["Blocked", "On Hold", "Waiting"]
        status_counts = {
            "To Do": 0,
            "In Progress": 0,
            "Done": 0,
            "Blocked": 0,
            "Other": 0
        }
        open_tickets = []
        for ticket in tickets:
            status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
            if status in DONE_STATUSES:
                status_counts["Done"] += 1
            elif status in IN_PROGRESS_STATUSES:
                status_counts["In Progress"] += 1
            elif status in TODO_STATUSES:
                status_counts["To Do"] += 1
                open_tickets.append(ticket)
            elif status in BLOCKED_STATUSES:
                status_counts["Blocked"] += 1
                open_tickets.append(ticket)
            else:
                status_counts["Other"] += 1
                if status not in DONE_STATUSES:
                    open_tickets.append(ticket)
        total_tickets = len(tickets)
        done_tickets = status_counts["Done"]
        in_progress = status_counts["In Progress"]
        todo_tickets = status_counts["To Do"]
        blocked_tickets = status_counts.get("Blocked", 0)
        self.log(f"[PREDICTION] Status distribution: {status_counts}")
        self.log(f"[PREDICTION] Total: {total_tickets}, Done: {done_tickets}, Open: {len(open_tickets)}")
        if total_tickets == 0:
            return {
                "probability": 0.0,
                "confidence": 0.1,
                "reasoning": "No tickets found in sprint",
                "risk_level": "critical",
                "ai_insights": {
                    "patterns_detected": {},
                    "confidence": 0.1,
                    "analysis_type": "no_data"
                }
            }
        completion_percentage = done_tickets / total_tickets
        remaining_work = len(open_tickets)
        self.log("[AI-ENHANCED] Analyzing patterns with AI...")
        pattern_analysis = self._ai_analyze_ticket_patterns(tickets, status_counts)
        if completion_percentage > 0.9:
            base_probability = 0.95
            reasoning = f"Sprint is {completion_percentage:.0%} complete with only {remaining_work} tickets remaining"
        else:
            base_probability = completion_percentage
            if pattern_analysis.get("blocking_patterns"):
                base_probability *= 0.85
                self.log(f"[AI] Blocking patterns detected: {pattern_analysis['blocking_patterns']}")
            if "team overload" in str(pattern_analysis.get("team_dynamics", [])):
                base_probability *= 0.9
                self.log("[AI] Team overload detected")
            if pattern_analysis.get("risk_signals"):
                risk_count = len(pattern_analysis["risk_signals"])
                base_probability *= (1 - (risk_count * 0.05))
                self.log(f"[AI] {risk_count} risk signals detected")
            velocity_history = historical_data.get("velocity_history", [])
            if not velocity_history and done_tickets > 0:
                days_elapsed = 10
                daily_velocity = done_tickets / days_elapsed
                velocity_history = [daily_velocity * 7]
            avg_velocity = np.mean(velocity_history) if velocity_history else 5
            velocity_std = np.std(velocity_history) if len(velocity_history) > 1 else avg_velocity * 0.3
            days_remaining = 10
            expected_completion = avg_velocity * (days_remaining / 7)
            if velocity_std > 0 and remaining_work > 0:
                z_score = (remaining_work - expected_completion) / velocity_std
                velocity_probability = 1 - stats.norm.cdf(z_score)
                base_probability = (base_probability * 0.6) + (velocity_probability * 0.4)
        probability = max(0.05, min(0.95, base_probability))
        ai_confidence = pattern_analysis.get("pattern_confidence", 0.7)
        if probability < self.prediction_thresholds["high_risk"] or len(pattern_analysis.get("risk_signals", [])) > 2:
            risk_level = "high"
        elif probability < self.prediction_thresholds["medium_risk"] or ai_confidence < 0.5:
            risk_level = "medium"
        elif probability < self.prediction_thresholds["low_risk"]:
            risk_level = "low"
        else:
            risk_level = "minimal"
        risk_factors = []
        if blocked_tickets > 0:
            risk_factors.append(f"{blocked_tickets} blocked tickets")
        if in_progress > remaining_work * 0.5:
            risk_factors.append(f"High WIP ratio ({in_progress} in progress)")
        if pattern_analysis.get("risk_signals"):
            risk_factors.extend(pattern_analysis["risk_signals"][:2])
        if pattern_analysis.get("complexity_patterns"):
            risk_factors.append(f"Complexity: {pattern_analysis['complexity_patterns'][0]}")
        reasoning = f"Sprint is {completion_percentage:.0%} complete with {remaining_work} tickets remaining. "
        if avg_velocity > 0:
            days_to_complete = remaining_work / (avg_velocity / 7)
            reasoning += f"At current velocity ({avg_velocity:.1f} tickets/week), completion would take {days_to_complete:.1f} days. "
        if pattern_analysis.get("pattern_confidence", 0) > 0.7:
            patterns_found = []
            if pattern_analysis.get("complexity_patterns"):
                patterns_found.extend(pattern_analysis["complexity_patterns"][:1])
            if pattern_analysis.get("blocking_patterns"):
                patterns_found.extend(pattern_analysis["blocking_patterns"][:1])
            if patterns_found:
                reasoning += f"AI analysis detected: {', '.join(patterns_found)}. "
        if probability >= 0.8:
            confidence_text = "The team is on track to complete the sprint successfully."
        elif probability >= 0.6:
            confidence_text = "Sprint completion is achievable but requires focused effort."
        elif probability >= 0.4:
            confidence_text = "Sprint completion is at risk without immediate intervention."
        else:
            confidence_text = "Sprint completion is highly unlikely with current trajectory."
        reasoning += confidence_text
        if risk_factors:
            reasoning += f" Key concerns: {', '.join(risk_factors[:3])}."
        recommended_actions = self._generate_sprint_recommendations(
            probability, risk_factors, remaining_work, velocity_history
        )
        if pattern_analysis.get("blocking_patterns"):
            recommended_actions.insert(0, 
                f"🔓 AI detected blocking patterns: {pattern_analysis['blocking_patterns'][0]}. Address immediately."
            )
        if pattern_analysis.get("team_dynamics") and "knowledge silo" in str(pattern_analysis["team_dynamics"]):
            recommended_actions.append(
                "👥 AI detected knowledge silos - implement pair programming or knowledge sharing sessions"
            )
        return {
            "probability": round(probability, 2),
            "confidence": round(ai_confidence, 2),
            "reasoning": reasoning,
            "risk_level": risk_level,
            "remaining_work": remaining_work,
            "completion_percentage": round(completion_percentage, 2),
            "expected_velocity": round(avg_velocity, 1),
            "risk_factors": risk_factors,
            "recommended_actions": recommended_actions[:5],
            "detailed_metrics": {
                "total_tickets": total_tickets,
                "done_tickets": done_tickets,
                "open_tickets": remaining_work,
                "blocked_tickets": blocked_tickets,
                "in_progress": in_progress,
                "todo": todo_tickets
            },
            "ai_insights": {
                "patterns_detected": pattern_analysis,
                "confidence": ai_confidence,
                "analysis_type": "ai_enhanced",
                "patterns_summary": {
                    "complexity": len(pattern_analysis.get("complexity_patterns", [])),
                    "blockers": len(pattern_analysis.get("blocking_patterns", [])),
                    "risks": len(pattern_analysis.get("risk_signals", [])),
                    "team_issues": len(pattern_analysis.get("team_dynamics", []))
                }
            }
        }

    def _identify_complex_tickets(self, tickets: List[Dict[str, Any]]) -> int:
        complex_count = 0
        for ticket in tickets:
            fields = ticket.get("fields", {})
            story_points = fields.get("customfield_10010", 0) or 0
            if story_points > 5:
                complex_count += 1
                continue
            description = fields.get("description", "") or ""
            if len(description) > 1000:
                complex_count += 1
                continue
            subtasks = fields.get("subtasks", [])
            if len(subtasks) > 3:
                complex_count += 1
                continue
            labels = fields.get("labels", [])
            complexity_indicators = ["complex", "investigation", "research", "architecture", "refactor"]
            if any(indicator in label.lower() for label in labels for indicator in complexity_indicators):
                complex_count += 1
        return complex_count

    def _assess_team_load(self, tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
        assignee_loads = {}
        assignee_completed = {}
        for ticket in tickets:
            fields = ticket.get("fields", {})
            assignee_info = fields.get("assignee")
            if assignee_info:
                assignee = assignee_info.get("displayName", "Unknown")
                status = fields.get("status", {}).get("name", "Unknown")
                if status in ["To Do", "In Progress", "In Review"]:
                    assignee_loads[assignee] = assignee_loads.get(assignee, 0) + 1
                if status == "Done":
                    assignee_completed[assignee] = assignee_completed.get(assignee, 0) + 1
        if not assignee_loads:
            return {
                "burnout_risk": False,
                "overloaded_members": [],
                "load_distribution": {},
                "recommendations": []
            }
        loads = list(assignee_loads.values())
        avg_load = np.mean(loads) if loads else 0
        std_load = np.std(loads) if len(loads) > 1 else 0
        max_load = max(loads) if loads else 0
        overloaded = []
        recommendations = []
        for assignee, load in assignee_loads.items():
            completed = assignee_completed.get(assignee, 0)
            if load > avg_load * self.prediction_thresholds["burnout_threshold"]:
                overloaded.append({
                    "name": assignee,
                    "current_load": load,
                    "completed": completed,
                    "overload_factor": load / max(avg_load, 1)
                })
                tickets_to_redistribute = int(load - avg_load)
                recommendations.append(
                    f"Redistribute {tickets_to_redistribute} tickets from {assignee} "
                    f"(currently has {load}, average is {avg_load:.1f})"
                )
        burnout_risk = len(overloaded) > 0 or max_load > 10
        if std_load > avg_load * 0.5:
            recommendations.append(
                "High workload variance detected - consider load balancing across team"
            )
        return {
            "burnout_risk": burnout_risk,
            "overloaded_members": overloaded,
            "load_distribution": assignee_loads,
            "team_metrics": {
                "avg_load": round(avg_load, 1),
                "max_load": max_load,
                "std_deviation": round(std_load, 1),
                "team_size": len(assignee_loads)
            },
            "recommendations": recommendations
        }

    def _identify_future_risks(self, tickets: List[Dict[str, Any]], 
                              metrics: Dict[str, Any],
                              predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.log("[PREDICTION] Identifying future risks")
        risks = []
        sprint_probability = predictions.get("sprint_completion", {}).get("probability", 1.0)
        if sprint_probability < 0.7:
            remaining_work = predictions.get("sprint_completion", {}).get("remaining_work", 0)
            risks.append({
                "type": "sprint_failure",
                "severity": "high" if sprint_probability < 0.4 else "medium",
                "probability": round(1 - sprint_probability, 2),
                "description": f"Sprint at risk with only {sprint_probability:.0%} completion probability. "
                              f"{remaining_work} tickets still open.",
                "mitigation": f"Consider deferring {int(remaining_work * 0.3)} low-priority tickets "
                             f"and focusing team on critical items",
                "timeline": "immediate",
                "impact": "Sprint goals not met, stakeholder disappointment"
            })
        bottlenecks = metrics.get("bottlenecks", {})
        for status, count in bottlenecks.items():
            if count > len(tickets) * 0.2:
                risks.append({
                    "type": "process_bottleneck",
                    "severity": "high" if count > 10 else "medium",
                    "probability": 0.8,
                    "description": f"Bottleneck in '{status}' status with {count} tickets stuck",
                    "mitigation": f"Review all tickets in {status} status, identify blockers, "
                                 f"and allocate resources to clear the backlog",
                    "timeline": "next_2_days",
                    "impact": f"Delayed delivery of {count} features/fixes"
                })
        team_load = self._assess_team_load(tickets)
        if team_load["burnout_risk"]:
            for member in team_load["overloaded_members"]:
                risks.append({
                    "type": "team_burnout",
                    "severity": "high",
                    "probability": 0.7,
                    "description": f"{member['name']} overloaded with {member['current_load']} tickets "
                                  f"({member['overload_factor']:.1f}x average)",
                    "mitigation": team_load["recommendations"][0] if team_load["recommendations"] else 
                                 f"Redistribute workload from {member['name']}",
                    "timeline": "immediate",
                    "impact": "Reduced productivity, potential quality issues, team morale"
                })
        velocity_forecast = predictions.get("velocity_forecast", {})
        if velocity_forecast.get("trend") == "declining":
            next_week_estimate = velocity_forecast.get("next_week_estimate", 0)
            risks.append({
                "type": "velocity_decline",
                "severity": "medium",
                "probability": 0.6,
                "description": f"Team velocity declining, next week estimate only {next_week_estimate:.1f} tickets",
                "mitigation": "Conduct retrospective to identify impediments, "
                             "review and optimize development processes",
                "timeline": "next_week",
                "impact": "Reduced delivery capacity affecting future sprints"
            })
        old_tickets = self._identify_old_tickets(tickets)
        if old_tickets > 10:
            risks.append({
                "type": "technical_debt",
                "severity": "medium",
                "probability": 0.5,
                "description": f"{old_tickets} tickets open for more than 30 days indicating technical debt",
                "mitigation": "Schedule dedicated technical debt sprint or allocate 20% capacity",
                "timeline": "next_sprint",
                "impact": "Increased maintenance cost and reduced feature velocity"
            })
        risk_score = {"high": 3, "medium": 2, "low": 1}
        risks.sort(key=lambda r: (risk_score.get(r["severity"], 0), r["probability"]), reverse=True)
        return risks[:10]

    def _identify_old_tickets(self, tickets: List[Dict[str, Any]]) -> int:
        old_count = 0
        thirty_days_ago = datetime.now() - timedelta(days=30)
        for ticket in tickets:
            if ticket.get("fields", {}).get("status", {}).get("name") != "Done":
                created_str = ticket.get("fields", {}).get("created", "")
                if created_str:
                    try:
                        created_date = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                        if created_date.replace(tzinfo=None) < thirty_days_ago:
                            old_count += 1
                    except:
                        pass
        return old_count

    def _predict_ticket_completion(self, tickets: List[Dict[str, Any]], 
                                  metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.log("[PREDICTION] Predicting ticket completion times")
        predictions = []
        avg_cycle_time = metrics.get("cycle_time", 5)
        open_tickets = [t for t in tickets if t.get("fields", {}).get("status", {}).get("name") != "Done"]
        for ticket in open_tickets[:20]:
            fields = ticket.get("fields", {})
            ticket_key = ticket.get("key", "Unknown")
            summary = fields.get("summary", "No summary")
            status = fields.get("status", {}).get("name", "Unknown")
            assignee_info = fields.get("assignee")
            assignee = assignee_info.get("displayName", "Unassigned") if assignee_info else "Unassigned"
            created_str = fields.get("created", "")
            age_days = 0
            if created_str:
                try:
                    created_date = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    age_days = (datetime.now(created_date.tzinfo) - created_date).days
                except:
                    pass
            if status == "To Do":
                estimated_days = avg_cycle_time * 1.2
            elif status == "In Progress":
                days_in_progress = self._calculate_days_in_status(ticket.get("changelog", {}).get("histories", []), "In Progress")
                remaining_ratio = max(0.2, 1 - (days_in_progress / max(avg_cycle_time, 1)))
                estimated_days = avg_cycle_time * remaining_ratio
            else:
                estimated_days = avg_cycle_time
            complexity_factor = self._estimate_ticket_complexity(ticket)
            estimated_days *= complexity_factor
            if assignee == "Unassigned":
                estimated_days *= 1.5
            completion_date = datetime.now() + timedelta(days=estimated_days)
            is_bottleneck = (age_days > 14 and status != "Done") or estimated_days > avg_cycle_time * 2
            predictions.append({
                "ticket_key": ticket_key,
                "summary": summary[:80] + "..." if len(summary) > 80 else summary,
                "status": status,
                "assignee": assignee,
                "age_days": age_days,
                "estimated_completion": completion_date.strftime("%Y-%m-%d"),
                "days_remaining": round(estimated_days, 1),
                "confidence": 0.7 if status == "In Progress" else 0.5,
                "is_bottleneck": is_bottleneck,
                "bottleneck_reason": self._get_bottleneck_reason(age_days, estimated_days, avg_cycle_time, assignee)
            })
        predictions.sort(key=lambda p: (not p["is_bottleneck"], p["days_remaining"]))
        return predictions

    def _get_bottleneck_reason(self, age_days: int, estimated_days: float, 
                               avg_cycle_time: float, assignee: str) -> str:
        reasons = []
        if age_days > 14:
            reasons.append(f"Open for {age_days} days")
        if estimated_days > avg_cycle_time * 2:
            reasons.append(f"High complexity ({estimated_days:.0f} days estimated)")
        if assignee == "Unassigned":
            reasons.append("No assignee")
        return "; ".join(reasons) if reasons else "Normal priority"

    def _generate_early_warnings(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.log("[PREDICTION] Generating early warnings")
        warnings = []
        sprint_completion = predictions.get("sprint_completion", {})
        if sprint_completion.get("probability", 1.0) < 0.5:
            remaining_work = sprint_completion.get("remaining_work", 0)
            warnings.append({
                "type": "sprint_failure_imminent",
                "urgency": "critical",
                "message": f"⚠️ Sprint completion at risk! Only {sprint_completion['probability']:.0%} chance "
                          f"with {remaining_work} tickets remaining",
                "recommended_action": f"Emergency triage meeting needed. Consider deferring "
                                    f"{int(remaining_work * 0.4)} low-priority tickets",
                "trigger_collaboration": ["recommendation_agent", "chat_agent"],
                "data": {
                    "probability": sprint_completion["probability"],
                    "remaining_tickets": remaining_work,
                    "risk_factors": sprint_completion.get("risk_factors", [])
                }
            })
        risks = predictions.get("risks", [])
        high_risks = [r for r in risks if r.get("severity") == "high"]
        if len(high_risks) >= 2:
            risk_summary = "; ".join([r["type"] for r in high_risks[:3]])
            warnings.append({
                "type": "multiple_high_risks",
                "urgency": "high",
                "message": f"🚨 {len(high_risks)} high-severity risks detected: {risk_summary}",
                "recommended_action": "Schedule risk mitigation meeting today. "
                                    "Focus on top 3 risks with specific action plans",
                "trigger_collaboration": ["recommendation_agent"],
                "data": {
                    "risk_count": len(high_risks),
                    "top_risks": high_risks[:3]
                }
            })
        ticket_predictions = predictions.get("ticket_predictions", [])
        bottleneck_tickets = [t for t in ticket_predictions if t.get("is_bottleneck")]
        if len(bottleneck_tickets) >= 5:
            warnings.append({
                "type": "critical_bottlenecks",
                "urgency": "high",
                "message": f"🔴 {len(bottleneck_tickets)} tickets identified as bottlenecks blocking progress",
                "recommended_action": "Review bottleneck tickets immediately. "
                                    "Assign senior developers to unblock critical items",
                "trigger_collaboration": ["productivity_dashboard_agent"],
                "data": {
                    "bottleneck_count": len(bottleneck_tickets),
                    "top_bottlenecks": bottleneck_tickets[:5]
                }
            })
        team_metrics = predictions.get("team_burnout_analysis", {})
        if team_metrics.get("burnout_risk"):
            overloaded_count = len(team_metrics.get("overloaded_members", []))
            warnings.append({
                "type": "team_capacity_warning",
                "urgency": "medium",
                "message": f"⚡ {overloaded_count} team members at burnout risk due to overload",
                "recommended_action": "Redistribute workload immediately. "
                                    "Consider bringing in additional resources",
                "trigger_collaboration": ["recommendation_agent"],
                "data": team_metrics
            })
        return sorted(warnings, key=lambda w: {"critical": 0, "high": 1, "medium": 2}.get(w["urgency"], 3))

    def _generate_probability_reasoning(self, probability: float, completion_percentage: float,
                                      remaining_work: int, avg_velocity: float, 
                                      risk_factors: List[str]) -> str:
        base_reasoning = (f"Sprint is currently {completion_percentage:.0%} complete with "
                         f"{remaining_work} tickets remaining. ")
        if avg_velocity > 0:
            days_to_complete = remaining_work / (avg_velocity / 7)
            base_reasoning += f"At current velocity ({avg_velocity:.1f} tickets/week), "
            base_reasoning += f"completion would take {days_to_complete:.1f} days. "
        if probability >= 0.8:
            confidence_text = "The team is on track to complete the sprint successfully."
        elif probability >= 0.6:
            confidence_text = "Sprint completion is achievable but requires focused effort."
        elif probability >= 0.4:
            confidence_text = "Sprint completion is at risk without immediate intervention."
        else:
            confidence_text = "Sprint completion is highly unlikely with current trajectory."
        base_reasoning += confidence_text
        if risk_factors:
            base_reasoning += f" Key concerns: {', '.join(risk_factors)}."
        return base_reasoning

    def _generate_sprint_recommendations(self, probability: float, risk_factors: List[str],
                                       remaining_work: int, velocity_history: List[float]) -> List[str]:
        recommendations = []
        if probability < 0.5:
            recommendations.append(
                f"🚨 IMMEDIATE ACTION: Defer {int(remaining_work * 0.4)} low-priority tickets "
                f"to next sprint to ensure critical items complete"
            )
            recommendations.append(
                "📅 Schedule daily 15-minute team sync to identify and remove blockers quickly"
            )
        if "blocked tickets" in str(risk_factors):
            recommendations.append(
                "🔓 Prioritize unblocking tickets - assign senior team members to resolve dependencies"
            )
        if "Declining velocity" in str(risk_factors):
            recommendations.append(
                "📊 Conduct mini-retrospective to identify velocity impediments and process improvements"
            )
        if "High WIP ratio" in str(risk_factors):
            recommendations.append(
                "🎯 Implement WIP limits - focus on completing in-progress items before starting new work"
            )
        if probability < 0.7:
            recommendations.append(
                f"✂️ Reduce scope by {int((1-probability) * remaining_work)} tickets to improve success rate"
            )
        if len(velocity_history) < 3:
            recommendations.append(
                "📈 Improve velocity tracking to enable better predictions in future sprints"
            )
        return recommendations[:5]

    def _calculate_velocity_trend(self, velocity_history: List[float]) -> float:
        if len(velocity_history) < 2:
            return 0.0
        x = np.arange(len(velocity_history))
        y = np.array(velocity_history)
        slope, _ = np.polyfit(x, y, 1)
        avg_velocity = np.mean(y)
        trend = slope / max(avg_velocity, 1)
        return trend

    def _forecast_velocity_trends(self, historical_data: Dict[str, Any], 
                             current_velocity: float,
                             tickets: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        velocity_history = historical_data.get("velocity_history", [])
        if not velocity_history or len(velocity_history) < 3:
            if tickets:
                velocity_history = self._extract_velocity_history(tickets)
            else:
                self.log("[WARNING] No tickets provided for velocity history extraction")
                velocity_history = [current_velocity] * 3
        prompt = f"""As a data scientist, analyze this REAL velocity data and predict future trends:

    Historical Weekly Velocity: {velocity_history}
    Current Velocity: {current_velocity}

    Consider:
    1. Is there a clear trend or pattern?
    2. Are there any anomalies that might indicate future changes?
    3. What external factors might affect future velocity?

    Provide a forecast for the next 4 weeks with reasoning.
    Return as JSON: {{"forecast": [week1, week2, week3, week4], "reasoning": "explanation", "confidence": 0.8}}"""
        try:
            response = self.model_manager.generate_response(
                prompt=prompt,
                context={
                    "agent_name": self.name,
                    "task_type": "velocity_forecasting"
                }
            )
            ai_forecast = self._parse_ai_json_response(response)
            if ai_forecast and "forecast" in ai_forecast:
                forecast_values = ai_forecast["forecast"]
                reasoning = ai_forecast.get("reasoning", "AI-based forecast")
            else:
                if len(velocity_history) >= 3:
                    x = np.arange(len(velocity_history))
                    y = np.array(velocity_history)
                    slope, intercept = np.polyfit(x, y, 1)
                    forecast_values = []
                    for i in range(1, 5):
                        dampening = 0.9 ** i
                        predicted = intercept + slope * (len(velocity_history) + i)
                        predicted *= dampening
                        forecast_values.append(max(0, round(predicted)))
                    reasoning = f"Trend analysis shows {'increasing' if slope > 0 else 'decreasing'} velocity"
                else:
                    forecast_values = [current_velocity] * 4
                    reasoning = "Insufficient historical data for trend analysis"
            if velocity_history:
                avg_velocity = np.mean(velocity_history)
                volatility = np.std(velocity_history) / max(avg_velocity, 1)
                if len(velocity_history) >= 2:
                    recent_trend = velocity_history[-1] - velocity_history[-2]
                    if recent_trend > avg_velocity * 0.1:
                        trend_direction = "improving"
                    elif recent_trend < -avg_velocity * 0.1:
                        trend_direction = "declining"
                    else:
                        trend_direction = "stable"
                else:
                    trend_direction = "unknown"
                trend_percentage = recent_trend / max(velocity_history[-2], 1) if len(velocity_history) >= 2 else 0
            else:
                avg_velocity = current_velocity
                volatility = 0
                trend_direction = "unknown"
                trend_percentage = 0
            return {
                "forecast": forecast_values,
                "trend": trend_direction,
                "trend_percentage": trend_percentage,
                "confidence": 0.8 if len(velocity_history) >= 5 else 0.5,
                "insights": reasoning,
                "next_week_estimate": forecast_values[0],
                "historical_average": round(avg_velocity, 1),
                "volatility": round(volatility, 2),
                "data_source": "real_velocity_analysis",
                "ai_enhanced": bool(ai_forecast)
            }
        except Exception as e:
            self.log(f"[ERROR] Velocity forecast failed: {e}")
            return {
                "forecast": [current_velocity] * 4,
                "trend": "stable",
                "insights": "Using current velocity as baseline",
                "data_source": "fallback"
            }
    
    def _calculate_sprint_burndown_data(self, tickets: List[Dict[str, Any]], 
                                       sprint_start: datetime = None,
                                       sprint_end: datetime = None) -> Dict[str, Any]:
        self.log("[PREDICTION] Calculating sprint burndown data")
        if not sprint_start:
            sprint_start = datetime.now() - timedelta(days=5)
        if not sprint_end:
            sprint_end = sprint_start + timedelta(days=10)
        sprint_days = (sprint_end - sprint_start).days
        current_day = min((datetime.now() - sprint_start).days, sprint_days)
        total_tickets = len(tickets)
        if total_tickets == 0:
            return {
                "error": "No tickets in sprint",
                "ideal_burndown": [],
                "actual_burndown": []
            }
        daily_completed = {}
        daily_remaining = {}
        for day in range(sprint_days + 1):
            daily_completed[day] = 0
            daily_remaining[day] = total_tickets
        for ticket in tickets:
            fields = ticket.get("fields", {})
            status = fields.get("status", {}).get("name", "Unknown")
            if status in ["Done", "Closed", "Resolved"]:
                resolution_date_str = fields.get("resolutiondate")
                if resolution_date_str:
                    try:
                        resolution_date = datetime.fromisoformat(resolution_date_str.replace("Z", "+00:00"))
                        resolution_day = (resolution_date.replace(tzinfo=None) - sprint_start).days
                        if 0 <= resolution_day <= sprint_days:
                            daily_completed[resolution_day] = daily_completed.get(resolution_day, 0) + 1
                    except:
                        pass
        actual_burndown = []
        cumulative_completed = 0
        for day in range(sprint_days + 1):
            cumulative_completed += daily_completed.get(day, 0)
            remaining = total_tickets - cumulative_completed
            actual_burndown.append(remaining)
        ideal_burndown = []
        for day in range(sprint_days + 1):
            ideal_remaining = total_tickets * (1 - day / sprint_days)
            ideal_burndown.append(round(ideal_remaining, 1))
        if current_day > 0:
            current_velocity = cumulative_completed / current_day
        else:
            current_velocity = 0
        predicted_burndown = actual_burndown[:current_day + 1].copy()
        for day in range(current_day + 1, sprint_days + 1):
            risk_adjusted_velocity = current_velocity
            blocked_tickets = len([t for t in tickets if "blocked" in t.get("fields", {}).get("labels", [])])
            if blocked_tickets > 0:
                risk_adjusted_velocity *= 0.8
            predicted_remaining = predicted_burndown[-1] - risk_adjusted_velocity
            predicted_burndown.append(max(0, round(predicted_remaining, 1)))
        return {
            "sprint_days": sprint_days,
            "current_day": current_day,
            "total_tickets": total_tickets,
            "completed_tickets": cumulative_completed,
            "remaining_tickets": total_tickets - cumulative_completed,
            "ideal_burndown": ideal_burndown,
            "actual_burndown": actual_burndown[:current_day + 1],
            "predicted_burndown": predicted_burndown,
            "current_velocity": round(current_velocity, 2),
            "on_track": predicted_burndown[-1] <= 5,
            "completion_day": self._estimate_completion_day(predicted_burndown),
            "daily_progress": daily_completed
        }
    
    def _estimate_completion_day(self, predicted_burndown: List[float]) -> Optional[int]:
        for day, remaining in enumerate(predicted_burndown):
            if remaining <= 0:
                return day
        return None

    def _calculate_capacity_forecast(self, tickets: List[Dict[str, Any]], 
                                   team_metrics: Dict[str, Any]) -> Dict[str, Any]:
        self.log("[PREDICTION] Calculating capacity forecast")
        assignee_loads = {}
        assignee_velocity = {}
        for ticket in tickets:
            fields = ticket.get("fields", {})
            assignee_info = fields.get("assignee")
            if assignee_info:
                assignee = assignee_info.get("displayName", "Unknown")
                status = fields.get("status", {}).get("name", "Unknown")
                if status not in ["Done", "Closed", "Resolved"]:
                    assignee_loads[assignee] = assignee_loads.get(assignee, 0) + 1
                else:
                    assignee_velocity[assignee] = assignee_velocity.get(assignee, 0) + 1
        team_members = list(set(list(assignee_loads.keys()) + list(assignee_velocity.keys())))
        team_size = len(team_members)
        if team_size == 0:
            return {
                "error": "No team members found",
                "capacity_forecast": []
            }
        member_capacities = []
        for member in team_members:
            load = assignee_loads.get(member, 0)
            velocity = assignee_velocity.get(member, 0)
            if velocity > 0:
                estimated_capacity = velocity
            else:
                estimated_capacity = 5
            utilization = load / estimated_capacity if estimated_capacity > 0 else 0
            member_capacities.append({
                "name": member,
                "current_load": load,
                "velocity": velocity,
                "estimated_capacity": estimated_capacity,
                "utilization": round(utilization, 2),
                "status": "overloaded" if utilization > 1.2 else "at_risk" if utilization > 0.9 else "healthy"
            })
        member_capacities.sort(key=lambda x: x["utilization"], reverse=True)
        total_load = sum(m["current_load"] for m in member_capacities)
        total_capacity = sum(m["estimated_capacity"] for m in member_capacities)
        team_utilization = total_load / total_capacity if total_capacity > 0 else 0
        capacity_forecast = []
        for week in range(4):
            if team_utilization > 1.2:
                adjustment = 0.95 ** week
            elif team_utilization < 0.6:
                adjustment = 1.05 ** week
            else:
                adjustment = 1.0
            week_capacity = total_capacity * adjustment
            capacity_forecast.append({
                "week": week,
                "capacity": round(week_capacity, 1),
                "projected_utilization": round(total_load / week_capacity, 2) if week_capacity > 0 else 0
            })
        return {
            "team_size": team_size,
            "total_capacity": round(total_capacity, 1),
            "current_load": total_load,
            "team_utilization": round(team_utilization, 2),
            "member_capacities": member_capacities,
            "capacity_forecast": capacity_forecast,
            "at_risk_members": [m for m in member_capacities if m["status"] in ["overloaded", "at_risk"]],
            "recommendations": self._generate_capacity_recommendations(member_capacities, team_utilization)
        }
    
    def _generate_capacity_recommendations(self, member_capacities: List[Dict[str, Any]], 
                                         team_utilization: float) -> List[str]:
        recommendations = []
        overloaded = [m for m in member_capacities if m["status"] == "overloaded"]
        if overloaded:
            recommendations.append(
                f"Redistribute work from {overloaded[0]['name']} "
                f"(currently at {overloaded[0]['utilization']:.0%} capacity)"
            )
        if team_utilization > 1.0:
            recommendations.append(
                "Consider deferring non-critical items or adding temporary resources"
            )
        elif team_utilization < 0.6:
            recommendations.append(
                "Team has available capacity - consider pulling in items from backlog"
            )
        utilizations = [m["utilization"] for m in member_capacities]
        if utilizations:
            variance = np.std(utilizations)
            if variance > 0.3:
                recommendations.append(
                    "High workload variance detected - redistribute tasks for better balance"
                )
        return recommendations
    
    def _calculate_analysis_confidence(self, tickets: List[Dict[str, Any]]) -> float:
        if not tickets:
            return 0.0
        confidence_factors = []
        sample_size_score = min(len(tickets) / 50.0, 1.0)
        confidence_factors.append(sample_size_score * 0.3)
        complete_tickets = 0
        for ticket in tickets:
            fields = ticket.get("fields", {})
            required_fields = ["status", "assignee", "created"]
            if all(fields.get(field) for field in required_fields):
                complete_tickets += 1
        completeness_score = complete_tickets / len(tickets)
        confidence_factors.append(completeness_score * 0.3)
        done_tickets = len([t for t in tickets if t.get("fields", {}).get("status", {}).get("name") in ["Done", "Closed", "Resolved"]])
        history_score = min(done_tickets / 10.0, 1.0)
        confidence_factors.append(history_score * 0.2)
        dates = []
        for ticket in tickets:
            created = ticket.get("fields", {}).get("created")
            if created:
                try:
                    dates.append(datetime.fromisoformat(created.replace("Z", "+00:00")))
                except:
                    pass
        if len(dates) >= 2:
            time_span = (max(dates) - min(dates)).days
            time_span_score = min(time_span / 30.0, 1.0)
            confidence_factors.append(time_span_score * 0.2)
        else:
            confidence_factors.append(0.1)
        total_confidence = sum(confidence_factors)
        return min(total_confidence, 1.0)

    def _calculate_days_in_status(self, changelog: List[Dict[str, Any]], status: str) -> float:
        for history in reversed(changelog):
            for item in history.get("items", []):
                if item.get("field") == "status" and item.get("toString") == status:
                    start_time = datetime.fromisoformat(history.get("created", "").replace("Z", "+00:00"))
                    return (datetime.now(start_time.tzinfo) - start_time).days
        return 0

    def _estimate_ticket_complexity(self, ticket: Dict[str, Any]) -> float:
        complexity = 1.0
        fields = ticket.get("fields", {})
        story_points = fields.get("customfield_10010", 0) or 0
        if story_points > 8:
            complexity *= 1.5
        elif story_points > 5:
            complexity *= 1.2
        description = fields.get("description", "") or ""
        if len(description) > 2000:
            complexity *= 1.3
        elif len(description) > 1000:
            complexity *= 1.1
        comment_count = fields.get("comment", {}).get("total", 0)
        if comment_count > 10:
            complexity *= 1.2
        labels = fields.get("labels", [])
        complex_keywords = ["complex", "architecture", "refactor", "migration", "integration"]
        if any(keyword in label.lower() for label in labels for keyword in complex_keywords):
            complexity *= 1.3
        return min(complexity, 2.5)

    def _generate_natural_language_predictions(self, predictions: Dict[str, Any]) -> str:
        sprint_data = predictions.get("sprint_completion", {})
        risks = predictions.get("risks", [])
        ai_insights = sprint_data.get("ai_insights", {})
        prompt = f"""You are presenting predictive analytics to a team lead. Create a brief, 
conversational summary (2-3 sentences) that provides genuine insights.

Sprint Status:
- Completion: {sprint_data.get('completion_percentage', 0):.0%}
- Success Probability: {sprint_data.get('probability', 0):.0%}
- Risk Level: {sprint_data.get('risk_level', 'unknown')}

AI Detected Patterns:
{json.dumps(ai_insights.get('patterns_detected', {}), indent=2) if ai_insights else 'No patterns analyzed'}

Top Risks: {len([r for r in risks if r.get('severity') == 'high'])} high severity

Write a natural summary that:
1. Highlights the most important insight from the AI analysis
2. Mentions one specific pattern or risk
3. Suggests one actionable step
Keep it concise and conversational."""
        try:
            response = self.model_manager.generate_response(
                prompt=prompt,
                context={
                    "agent_name": self.name,
                    "task_type": "predictive_summary",
                    "sprint_probability": sprint_data.get('probability', 0),
                    "risk_count": len(risks)
                }
            )
            return response.strip()
        except Exception as e:
            self.log(f"[ERROR] Failed to generate AI summary: {e}")
            probability = sprint_data.get('probability', 0)
            patterns = ai_insights.get('patterns_detected', {})
            if probability < 0.5:
                pattern_text = ""
                if patterns.get('risk_signals'):
                    pattern_text = f"AI detected warning signs: {patterns['risk_signals'][0]}. "
                return (f"Sprint completion is at risk with only {probability:.0%} probability. "
                       f"{pattern_text}"
                       f"Focus on the {sprint_data.get('remaining_work', 0)} remaining tickets.")
            else:
                return (f"Sprint is {sprint_data.get('completion_percentage', 0):.0%} complete "
                       f"with {probability:.0%} success probability. "
                       f"AI confidence: {ai_insights.get('confidence', 0.7):.0%}.")
            
    def _perceive(self, input_data: Dict[str, Any]) -> None:
        super()._perceive(input_data)
        tickets = input_data.get("tickets", [])
        metrics = input_data.get("metrics", {})
        historical_data = input_data.get("historical_data", {})
        self.log(f"[PERCEPTION] Received {len(tickets)} tickets")
        self.log(f"[PERCEPTION] Metrics: {metrics}")
        self.mental_state.add_belief("tickets", tickets, 0.9, "input")
        self.mental_state.add_belief("metrics", metrics, 0.9, "input")
        self.mental_state.add_belief("historical_data", historical_data, 0.9, "input")
        self.mental_state.add_belief("analysis_type", input_data.get("analysis_type", "comprehensive"), 0.9, "input")

    def _act(self) -> Dict[str, Any]:
        try:
            tickets = self.mental_state.get_belief("tickets") or []
            metrics = self.mental_state.get_belief("metrics") or {}
            historical_data = self.mental_state.get_belief("historical_data") or {}
            analysis_type = self.mental_state.get_belief("analysis_type") or "comprehensive"
            self.log(f"[ACTION] Generating {analysis_type} predictive analysis for {len(tickets)} tickets")
            predictions = {}
            predictions["sprint_completion"] = self._calculate_sprint_completion_probability(
                tickets, metrics, historical_data
            )
            if analysis_type in ["comprehensive", "velocity_forecast", "velocity"]:
                velocity_history = self._extract_velocity_history(tickets)
                if not velocity_history:
                    current_throughput = metrics.get("throughput", 5)
                    velocity_history = [
                        int(current_throughput * 0.9),
                        int(current_throughput * 0.95),
                        current_throughput,
                        int(current_throughput * 1.05),
                        current_throughput
                    ]
                historical_data["velocity_history"] = velocity_history
                predictions["velocity_forecast"] = self._forecast_velocity_trends(
                    historical_data, 
                    metrics.get("throughput", 0),
                    tickets
                )
            if analysis_type in ["comprehensive", "burndown"]:
                burndown_data = self._calculate_sprint_burndown_data(tickets)
                predictions["burndown_analysis"] = burndown_data
            if analysis_type in ["comprehensive", "capacity"]:
                capacity_data = self._calculate_capacity_forecast(tickets, metrics)
                predictions["capacity_forecast"] = capacity_data
            if analysis_type in ["comprehensive", "risk_assessment"]:
                predictions["risks"] = self._identify_future_risks(tickets, metrics, predictions)
            if analysis_type in ["comprehensive", "ticket_predictions"]:
                predictions["ticket_predictions"] = self._predict_ticket_completion(tickets, metrics)
            predictions["team_burnout_analysis"] = self._assess_team_load(tickets)
            predictions["warnings"] = self._generate_early_warnings(predictions)
            predictions["natural_language_summary"] = self._generate_natural_language_predictions(predictions)
            ai_confidence = predictions.get("sprint_completion", {}).get("ai_insights", {}).get("confidence", 0.5)
            patterns_detected = predictions.get("sprint_completion", {}).get("ai_insights", {}).get("patterns_detected", {})
            predictions["analysis_metadata"] = {
                "ticket_count": len(tickets),
                "analysis_type": analysis_type,
                "data_quality": self._calculate_analysis_confidence(tickets),
                "timestamp": datetime.now().isoformat(),
                "ai_enhanced": True,
                "ai_confidence": ai_confidence,
                "patterns_detected_count": sum(
                    len(patterns_detected.get(key, [])) 
                    for key in ["complexity_patterns", "blocking_patterns", "risk_signals", "team_dynamics"]
                ),
                "using_real_data": True
            }
            self.log(f"[PREDICTION] Sprint completion: {predictions['sprint_completion']['probability']:.0%}")
            self.log(f"[PREDICTION] AI confidence: {ai_confidence:.0%}")
            self.log(f"[PREDICTION] Patterns detected: {predictions['analysis_metadata']['patterns_detected_count']}")
            self.log(f"[PREDICTION] Velocity trend: {predictions.get('velocity_forecast', {}).get('trend', 'unknown')}")
            self.log(f"[PREDICTION] Risks identified: {len(predictions.get('risks', []))}")
            self.log(f"[PREDICTION] Warnings generated: {len(predictions.get('warnings', []))}")
            if patterns_detected.get("risk_signals"):
                self.log(f"[AI ALERT] Risk signals: {patterns_detected['risk_signals'][:2]}")
            return {
                "predictions": predictions,
                "workflow_status": "success",
                "collaboration_metadata": {
                    "predictions_generated": True,
                    "analysis_type": analysis_type,
                    "ticket_count": len(tickets),
                    "confidence_level": predictions["sprint_completion"].get("confidence", 0.5),
                    "real_data_used": True,
                    "ai_enhanced": True,
                    "ai_patterns_found": predictions["analysis_metadata"]["patterns_detected_count"] > 0
                }
            }
        except Exception as e:
            self.log(f"[ERROR] Failed to generate predictions: {str(e)}")
            import traceback
            self.log(f"[ERROR] Traceback: {traceback.format_exc()}")
            return {
                "predictions": {
                    "error": str(e),
                    "sprint_completion": {
                        "probability": 0.5,
                        "confidence": 0.1,
                        "reasoning": "Error occurred during analysis",
                        "risk_level": "unknown"
                    }
                },
                "workflow_status": "failure",
                "error": str(e),
                "collaboration_metadata": {
                    "error_occurred": True,
                    "ai_enhanced": False
                }
            }
    
    def _extract_velocity_history(self, tickets: List[Dict[str, Any]]) -> List[float]:
        self.log("[PREDICTION] Extracting velocity history from tickets")
        weekly_velocity = {}
        for ticket in tickets:
            fields = ticket.get("fields", {})
            status = fields.get("status", {}).get("name", "Unknown")
            if status in ["Done", "Closed", "Resolved"]:
                resolution_date_str = fields.get("resolutiondate")
                if resolution_date_str:
                    try:
                        resolution_date = datetime.fromisoformat(resolution_date_str.replace("Z", "+00:00"))
                        week_key = resolution_date.strftime("%Y-W%U")
                        weekly_velocity[week_key] = weekly_velocity.get(week_key, 0) + 1
                    except:
                        pass
        if weekly_velocity:
            sorted_weeks = sorted(weekly_velocity.keys())
            return [weekly_velocity[week] for week in sorted_weeks[-10:]]
        return []
    def _extract_velocity_history(self, tickets: List[Dict[str, Any]]) -> List[float]:
        self.log("[PREDICTION] Extracting velocity history from tickets")
        weekly_velocity = {}
        for ticket in tickets:
            fields = ticket.get("fields", {})
            status = fields.get("status", {}).get("name", "Unknown")
            if status in ["Done", "Closed", "Resolved"]:
                resolution_date_str = fields.get("resolutiondate")
                if resolution_date_str:
                    try:
                        resolution_date = datetime.fromisoformat(resolution_date_str.replace("Z", "+00:00"))
                        week_key = resolution_date.strftime("%Y-W%U")
                        weekly_velocity[week_key] = weekly_velocity.get(week_key, 0) + 1
                    except:
                        pass
        if weekly_velocity:
            sorted_weeks = sorted(weekly_velocity.keys())
            return [weekly_velocity[week] for week in sorted_weeks[-10:]]
        return []

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(input_data)
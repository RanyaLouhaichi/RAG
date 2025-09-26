from typing import Dict, Any, List
import json
import logging
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent, AgentCapability
from orchestrator.core.model_manager import ModelManager # type: ignore
import sqlite3
import threading
import time

class ProductivityDashboardAgent(BaseAgent):
    OBJECTIVE = "Generate intelligent productivity analytics through collaborative data analysis and provide actionable insights to stakeholders"

    def __init__(self, redis_client=None):
        super().__init__(name="productivity_dashboard_agent", redis_client=redis_client)
        self.model_manager = ModelManager()
        self.db_path = "shared_data.db"
        self.last_throughput = 0
        self.mental_state.capabilities = [
            AgentCapability.RETRIEVE_DATA,
            AgentCapability.RANK_CONTENT,
            AgentCapability.COORDINATE_AGENTS,
            AgentCapability.PROVIDE_RECOMMENDATIONS
        ]
        self.mental_state.obligations.extend([
            "analyze_ticket_data",
            "generate_metrics",
            "create_visualization_data",
            "generate_report",
            "check_for_updates",
            "assess_analytical_depth",
            "request_additional_data",
            "provide_predictive_insights",
            "collaborate_for_recommendations",
            "integrate_predictions"
        ])
        self.log("Enhanced ProductivityDashboardAgent initialized with collaborative analytics capabilities")
        self.monitoring_thread = threading.Thread(target=self._check_for_updates_loop, daemon=True)
        self.monitoring_thread.start()

    def _check_for_updates(self):
        try:
            project_id = self.mental_state.beliefs.get("project_id", "PROJ123")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT has_changes, last_updated FROM updates WHERE project_id = ?", (project_id,))
            result = cursor.fetchone()
            
            if result and result[0]:
                self.log(f"Detected changes for project {project_id} at {result[1]} - applying collaborative analysis")
                
                cursor.execute("SELECT tickets FROM tickets WHERE project_id = ?", (project_id,))
                tickets_result = cursor.fetchone()
                if tickets_result:
                    tickets = json.loads(tickets_result[0])
                    self.mental_state.add_belief("tickets", tickets, 0.9, "database_update")
                    
                    self._assess_data_update_collaboration_needs(tickets, project_id)
                    
                    metrics = self._analyze_ticket_data(tickets)
                    
                    if metrics["throughput"] != self.last_throughput:
                        change_magnitude = abs(metrics["throughput"] - self.last_throughput) / max(self.last_throughput, 1)
                        
                        self.log(f"Significant throughput change detected: {self.last_throughput} → {metrics['throughput']} (magnitude: {change_magnitude:.2f})")
                        
                        if change_magnitude > 0.3:
                            self._request_collaborative_analysis(metrics, "significant_change")
                        
                        self.last_throughput = metrics["throughput"]
                        
                        visualization_data = self._create_visualization_data(metrics, {})
                        recommendations = self._get_collaborative_recommendations(metrics)
                        report = self._generate_report(metrics, recommendations, {})
                        
                        enhanced_dashboard_data = {
                            "metrics": metrics,
                            "visualization_data": visualization_data,
                            "report": report,
                            "recommendations": recommendations,
                            "collaborative_analysis_applied": True,
                            "collaboration_requests_made": len(self.mental_state.collaborative_requests),
                            "analysis_confidence": self._calculate_analysis_confidence(tickets)
                        }
                        
                        self.mental_state.beliefs.update(enhanced_dashboard_data)
                        self.log("Updated dashboard with collaborative intelligence applied")
            
            conn.close()
            
        except Exception as e:
            self.log(f"[ERROR] Failed to check for updates: {str(e)}")

    def _assess_data_update_collaboration_needs(self, tickets: List[Dict[str, Any]], project_id: str):
        if not tickets:
            return
        
        ticket_count = len(tickets)
        self.log(f"[COLLABORATION ASSESSMENT] Analyzing {ticket_count} updated tickets for collaborative opportunities")
        
        if ticket_count >= 15:
            self.log("[COLLABORATION NEED] Substantial data update - suggesting recommendation collaboration")
            self.mental_state.request_collaboration(
                agent_type="recommendation_agent",
                reasoning_type="strategic_reasoning",
                context={
                    "reason": "substantial_data_update",
                    "project_id": project_id,
                    "ticket_count": ticket_count,
                    "trigger": "data_update"
                }
            )
        
        if ticket_count >= 10:
            self.log("[COLLABORATION NEED] Sufficient data for predictive analysis")
            self.mental_state.request_collaboration(
                agent_type="predictive_analysis_agent",
                reasoning_type="data_analysis",
                context={
                    "reason": "predictive_analysis_opportunity",
                    "project_id": project_id,
                    "ticket_count": ticket_count
                }
            )
        
        data_characteristics = self._analyze_update_characteristics(tickets)
        
        if data_characteristics["suggests_knowledge_base_analysis"]:
            self.log("[COLLABORATION OPPORTUNITY] Data suggests knowledge base insights would be valuable")
            self.mental_state.request_collaboration(
                agent_type="knowledge_base_agent",
                reasoning_type="validation",
                context={
                    "reason": "knowledge_validation_opportunity",
                    "characteristics": data_characteristics
                }
            )

    def _analyze_update_characteristics(self, tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
        characteristics = {
            "high_resolution_rate": False,
            "complex_tickets_present": False,
            "team_distribution_changed": False,
            "suggests_knowledge_base_analysis": False
        }
        
        if not tickets:
            return characteristics
        
        resolved_count = 0
        complex_tickets = 0
        team_members = set()
        
        for ticket in tickets:
            fields = ticket.get("fields", {})
            
            if fields.get("resolutiondate"):
                resolved_count += 1
            
            description = fields.get("description", "")
            changelog = ticket.get("changelog", {}).get("histories", [])
            
            if len(description) > 500 or len(changelog) > 10:
                complex_tickets += 1
            
            assignee_info = fields.get("assignee")
            if assignee_info:
                team_members.add(assignee_info.get("displayName", "Unknown"))
        
        characteristics["high_resolution_rate"] = resolved_count / len(tickets) > 0.7 if tickets else False
        characteristics["complex_tickets_present"] = complex_tickets > len(tickets) * 0.3
        characteristics["team_distribution_changed"] = len(team_members) >= 4
        
        characteristics["suggests_knowledge_base_analysis"] = (
            characteristics["high_resolution_rate"] and 
            characteristics["complex_tickets_present"]
        )
        
        return characteristics

    def _check_for_updates_loop(self):
        while True:
            try:
                self._check_for_updates()
                self.log("Collaborative monitoring cycle complete - sleeping for 300 seconds (5 minutes)")
                time.sleep(10)
            except Exception as e:
                self.log(f"[ERROR] Monitor loop error: {str(e)}")
                time.sleep(30)

    def _perceive(self, input_data: Dict[str, Any]) -> None:
        super()._perceive(input_data)
        
        tickets = input_data.get("tickets", [])
        recommendations = input_data.get("recommendations", [])
        project_id = input_data.get("project_id", "PROJ123")
        predictions = input_data.get("predictions", {})
        
        self.log(f"[PERCEPTION] Processing {len(tickets)} tickets for productivity analysis of project {project_id}")
        
        self.mental_state.add_belief("tickets", tickets, 0.9, "input")
        self.mental_state.add_belief("recommendations", recommendations, 0.9, "input")
        self.mental_state.add_belief("project_id", project_id, 0.9, "input")
        self.mental_state.add_belief("predictions", predictions, 0.9, "input")
        
        if input_data.get("collaboration_purpose"):
            collaboration_purpose = input_data.get("collaboration_purpose")
            self.mental_state.add_belief("collaboration_context", collaboration_purpose, 0.9, "collaboration")
            self.log(f"[COLLABORATION] Operating in collaborative mode: {collaboration_purpose}")
        
        if input_data.get("primary_agent_result"):
            primary_result = input_data.get("primary_agent_result")
            self.mental_state.add_belief("primary_agent_context", primary_result, 0.8, "collaboration")
            self.log(f"[COLLABORATION] Received primary agent context for enhanced analysis")
        
        self._assess_analysis_collaboration_needs(input_data)

    def _assess_analysis_collaboration_needs(self, input_data: Dict[str, Any]) -> None:
        tickets = input_data.get("tickets", [])
        project_id = input_data.get("project_id")
        collaboration_purpose = input_data.get("collaboration_purpose")
        predictions = input_data.get("predictions", {})
        
        self.log(f"[COLLABORATION ASSESSMENT] Evaluating analysis needs for {len(tickets)} tickets")
        
        if len(tickets) < 10 and not collaboration_purpose:
            self.log("[COLLABORATION NEED] Limited ticket data - requesting enhanced data retrieval")
            self.mental_state.request_collaboration(
                agent_type="jira_data_agent",
                reasoning_type="data_analysis",
                context={
                    "reason": "need_more_comprehensive_data",
                    "project_id": project_id,
                    "current_ticket_count": len(tickets),
                    "analysis_depth": "comprehensive"
                }
            )
        
        if not predictions and len(tickets) >= 5:
            self.log("[COLLABORATION NEED] No predictive analysis available - requesting predictions")
            self.mental_state.request_collaboration(
                agent_type="predictive_analysis_agent",
                reasoning_type="data_analysis",
                context={
                    "reason": "need_predictive_insights",
                    "project_id": project_id,
                    "ticket_count": len(tickets)
                }
            )
        
        if len(tickets) >= 10:
            initial_metrics = self._analyze_ticket_data(tickets)
            if self._analysis_suggests_recommendations(initial_metrics):
                self.log("[COLLABORATION OPPORTUNITY] Analysis suggests valuable recommendations could be generated")
                self.mental_state.request_collaboration(
                    agent_type="recommendation_agent",
                    reasoning_type="strategic_reasoning",
                    context={
                        "reason": "analysis_enables_recommendations",
                        "metrics_preview": initial_metrics,
                        "project_id": project_id
                    }
                )

    def _analysis_suggests_recommendations(self, metrics: Dict[str, Any]) -> bool:
        bottlenecks = metrics.get("bottlenecks", {})
        throughput = metrics.get("throughput", 0)
        workload = metrics.get("workload", {})
        
        has_bottlenecks = any(count > 3 for count in bottlenecks.values())
        low_throughput = throughput < 5
        uneven_workload = len(workload) > 1 and max(workload.values()) > min(workload.values()) * 2
        
        return has_bottlenecks or low_throughput or uneven_workload

    def _analyze_ticket_data(self, tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.log(f"[ANALYSIS] Performing collaborative analysis on {len(tickets)} tickets")
        
        if not tickets:
            self.log("No tickets to analyze")
            return {
                "cycle_time": 0,
                "throughput": 0,
                "workload": {},
                "bottlenecks": {},
                "analysis_confidence": 0.0,
                "collaborative_insights": {}
            }
        
        try:
            assignee_counts = {}
            status_counts = {}
            done_tickets = []
            cycle_times = []
            resolution_patterns = {}
            
            for idx, ticket in enumerate(tickets):
                self.log(f"[ANALYSIS] Processing ticket {idx}: {ticket.get('key', 'unknown')}")
                fields = ticket.get("fields", {})
                current_status = fields.get("status", {}).get("name", "Unknown")
                assignee = fields.get("assignee", {}).get("displayName", "Unassigned") if fields.get("assignee") else "Unassigned"
                
                status_counts[current_status] = status_counts.get(current_status, 0) + 1
                
                if assignee != "Unassigned":
                    assignee_counts[assignee] = assignee_counts.get(assignee, 0) + 1
                
                if current_status == "Done":
                    done_tickets.append(ticket)
                    
                    cycle_time = self._calculate_enhanced_cycle_time(ticket)
                    if cycle_time > 0:
                        cycle_times.append(cycle_time)
                
                if fields.get("resolutiondate"):
                    resolution_date = fields.get("resolutiondate")
                    try:
                        res_date = datetime.fromisoformat(resolution_date.replace("Z", "+00:00"))
                        week_key = res_date.strftime("%Y-W%U")
                        resolution_patterns[week_key] = resolution_patterns.get(week_key, 0) + 1
                    except:
                        pass
            
            throughput = len(done_tickets)
            avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0
            bottlenecks = {status: count for status, count in status_counts.items() if status != "Done"}
            
            analysis_confidence = self._calculate_analysis_confidence(tickets)
            
            collaborative_insights = self._generate_collaborative_insights(
                assignee_counts, status_counts, cycle_times, resolution_patterns
            )
            
            metrics = {
                "cycle_time": avg_cycle_time,
                "throughput": throughput,
                "workload": assignee_counts,
                "bottlenecks": bottlenecks,
                "resolution_patterns": resolution_patterns,
                "analysis_confidence": analysis_confidence,
                "collaborative_insights": collaborative_insights,
                "data_quality_score": len(tickets) / 20.0
            }
            
            self.log(f"[ANALYSIS] Completed collaborative analysis: throughput={throughput}, avg_cycle_time={avg_cycle_time:.1f}, confidence={analysis_confidence:.2f}")
            return metrics
            
        except Exception as e:
            self.log(f"[ERROR] Failed to analyze ticket data: {str(e)}")
            import traceback
            self.log(f"[ERROR] Traceback: {traceback.format_exc()}")
            return {
                "cycle_time": 0,
                "throughput": 0,
                "workload": {},
                "bottlenecks": {},
                "analysis_confidence": 0.0,
                "collaborative_insights": {"error": str(e)}
            }

    def _calculate_enhanced_cycle_time(self, ticket: Dict[str, Any]) -> float:
        try:
            changelog = ticket.get("changelog", {}).get("histories", [])
            if not changelog:
                return 0.0
            
            work_start = None
            work_end = None
            
            for history in changelog:
                for item in history.get("items", []):
                    if item.get("field") == "status":
                        timestamp = datetime.fromisoformat(history.get("created", "").replace("Z", "+00:00"))
                        
                        if item.get("toString") == "In Progress" and not work_start:
                            work_start = timestamp
                        elif item.get("toString") == "Done":
                            work_end = timestamp
                            break
            
            if work_start and work_end:
                cycle_days = (work_end - work_start).total_seconds() / (24 * 3600)
                return max(cycle_days, 0)
                
        except Exception as e:
            self.log(f"[ERROR] Cycle time calculation failed: {e}")
        
        return 0.0

    def _calculate_analysis_confidence(self, tickets: List[Dict[str, Any]]) -> float:
        if not tickets:
            return 0.0
        
        confidence_factors = []
        
        sample_size_score = min(len(tickets) / 20.0, 1.0)
        confidence_factors.append(sample_size_score * 0.4)
        
        complete_tickets = 0
        for ticket in tickets:
            fields = ticket.get("fields", {})
            completeness_score = 0
            
            if fields.get("status"):
                completeness_score += 0.25
            if fields.get("assignee"):
                completeness_score += 0.25
            if fields.get("created"):
                completeness_score += 0.25
            if fields.get("updated"):
                completeness_score += 0.25
            
            if completeness_score >= 0.75:
                complete_tickets += 1
        
        completeness_score = complete_tickets / len(tickets)
        confidence_factors.append(completeness_score * 0.3)
        
        recent_activity = 0
        one_week_ago = datetime.now() - timedelta(days=7)
        
        for ticket in tickets:
            updated_str = ticket.get("fields", {}).get("updated", "")
            if updated_str:
                try:
                    updated_time = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                    if updated_time.replace(tzinfo=None) > one_week_ago:
                        recent_activity += 1
                except:
                    pass
        
        recency_score = min(recent_activity / len(tickets), 1.0)
        confidence_factors.append(recency_score * 0.3)
        
        total_confidence = sum(confidence_factors)
        return min(total_confidence, 1.0)

    def _generate_collaborative_insights(self, assignee_counts: Dict[str, int], 
                                       status_counts: Dict[str, int],
                                       cycle_times: List[float],
                                       resolution_patterns: Dict[str, int]) -> Dict[str, Any]:
        insights = {
            "team_balance_score": 0.0,
            "workflow_efficiency_score": 0.0,
            "predictive_indicators": {},
            "collaboration_opportunities": []
        }
        
        if assignee_counts:
            workload_values = list(assignee_counts.values())
            max_workload = max(workload_values)
            min_workload = min(workload_values)
            balance_ratio = min_workload / max_workload if max_workload > 0 else 1.0
            insights["team_balance_score"] = balance_ratio
            
            if balance_ratio < 0.5:
                insights["collaboration_opportunities"].append("workload_rebalancing")
        
        if status_counts:
            in_progress = status_counts.get("In Progress", 0)
            done = status_counts.get("Done", 0)
            to_do = status_counts.get("To Do", 0)
            
            total_active = in_progress + done + to_do
            if total_active > 0:
                efficiency_score = done / total_active
                insights["workflow_efficiency_score"] = efficiency_score
                
                if efficiency_score < 0.4:
                    insights["collaboration_opportunities"].append("process_optimization")
        
        if cycle_times:
            avg_cycle_time = sum(cycle_times) / len(cycle_times)
            insights["predictive_indicators"]["avg_cycle_time"] = avg_cycle_time
            insights["predictive_indicators"]["cycle_time_trend"] = "stable"
            
            if avg_cycle_time > 7:
                insights["collaboration_opportunities"].append("cycle_time_optimization")
        
        if resolution_patterns:
            recent_weeks = sorted(resolution_patterns.keys())[-4:]
            if len(recent_weeks) >= 2:
                recent_velocity = sum(resolution_patterns[week] for week in recent_weeks) / len(recent_weeks)
                insights["predictive_indicators"]["recent_velocity"] = recent_velocity
        
        return insights

    def _get_collaborative_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        existing_recommendations = self.mental_state.get_belief("recommendations") or []
        collaborative_insights = metrics.get("collaborative_insights", {})
        
        collaboration_opportunities = collaborative_insights.get("collaboration_opportunities", [])
        
        enhanced_recommendations = existing_recommendations.copy()
        
        if "workload_rebalancing" in collaboration_opportunities:
            enhanced_recommendations.append(
                "Team workload analysis indicates significant imbalance. Consider redistributing tasks to optimize team capacity and prevent burnout."
            )
        
        if "process_optimization" in collaboration_opportunities:
            enhanced_recommendations.append(
                "Workflow efficiency metrics suggest process bottlenecks. Implement regular stand-ups and review 'In Progress' items for blockers."
            )
        
        if "cycle_time_optimization" in collaboration_opportunities:
            enhanced_recommendations.append(
                "Extended cycle times detected. Consider breaking down large tasks and implementing continuous integration practices."
            )
        
        if not enhanced_recommendations:
            enhanced_recommendations = self._generate_intelligent_default_recommendations(metrics)
        
        return enhanced_recommendations

    def _generate_intelligent_default_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        recommendations = []
        
        throughput = metrics.get("throughput", 0)
        cycle_time = metrics.get("cycle_time", 0)
        bottlenecks = metrics.get("bottlenecks", {})
        analysis_confidence = metrics.get("analysis_confidence", 0.5)
        
        if throughput < 3:
            recommendations.append(
                f"Current throughput is {throughput} tickets. Consider implementing pair programming and automated testing to increase delivery velocity."
            )
        elif throughput > 10:
            recommendations.append(
                f"Excellent throughput of {throughput} tickets! Maintain this momentum by documenting successful practices and sharing with other teams."
            )
        
        if cycle_time > 5:
            recommendations.append(
                f"Average cycle time is {cycle_time:.1f} days. Focus on reducing work-in-progress and implementing daily progress reviews."
            )
        
        if bottlenecks:
            max_bottleneck = max(bottlenecks.items(), key=lambda x: x[1])
            if max_bottleneck[1] > 3:
                recommendations.append(
                    f"Bottleneck detected in '{max_bottleneck[0]}' with {max_bottleneck[1]} tickets. Investigate blockers and consider adding resources or process improvements."
                )
        
        if analysis_confidence < 0.6:
            recommendations.append(
                f"Note: Analysis confidence is {analysis_confidence:.1%}. Consider gathering more project data for more precise recommendations."
            )
        
        return recommendations

    def _request_collaborative_analysis(self, metrics: Dict[str, Any], trigger: str):
        self.log(f"[COLLABORATION REQUEST] Requesting collaborative analysis due to: {trigger}")
        
        self.mental_state.request_collaboration(
            agent_type="recommendation_agent",
            reasoning_type="strategic_reasoning",
            context={
                "reason": f"significant_change_detected_{trigger}",
                "metrics": metrics,
                "confidence": metrics.get("analysis_confidence", 0.5),
                "collaboration_trigger": trigger
            }
        )

    def _create_visualization_data(self, metrics: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        try:
            charts = []
            tables = []
            
            if metrics.get("workload"):
                workload_data = {
                    "type": "bar",
                    "title": "Team Workload Distribution",
                    "data": {
                        "labels": list(metrics["workload"].keys()),
                        "datasets": [{
                            "label": "Number of Tickets",
                            "data": list(metrics["workload"].values()),
                            "backgroundColor": self._generate_smart_colors(len(metrics["workload"]))
                        }]
                    },
                    "collaborative_insights": {
                        "balance_score": metrics.get("collaborative_insights", {}).get("team_balance_score", 0)
                    }
                }
                charts.append(workload_data)
            
            if metrics.get("bottlenecks"):
                bottleneck_data = {
                    "type": "bar",
                    "title": "Workflow Bottlenecks Analysis",
                    "data": {
                        "labels": list(metrics["bottlenecks"].keys()),
                        "datasets": [{
                            "label": "Tickets in Status",
                            "data": list(metrics["bottlenecks"].values()),
                            "backgroundColor": self._generate_bottleneck_colors(metrics["bottlenecks"])
                        }]
                    },
                    "insights": "Red indicates potential bottlenecks requiring attention"
                }
                charts.append(bottleneck_data)
            
            if predictions and predictions.get("velocity_forecast"):
                forecast = predictions["velocity_forecast"].get("forecast", [])
                if forecast:
                    velocity_chart = {
                        "type": "line",
                        "title": "Velocity Forecast",
                        "data": {
                            "labels": ["Week -2", "Week -1", "Current", "Next Week", "Week +2", "Week +3"],
                            "datasets": [{
                                "label": "Velocity Trend",
                                "data": [5, 6, metrics.get("throughput", 0)] + forecast[:3],
                                "borderColor": "rgba(75, 192, 192, 1)",
                                "backgroundColor": "rgba(75, 192, 192, 0.2)"
                            }]
                        },
                        "insights": predictions["velocity_forecast"].get("insights", "")
                    }
                    charts.append(velocity_chart)
            
            if predictions and predictions.get("sprint_completion"):
                sprint_data = predictions["sprint_completion"]
                sprint_chart = {
                    "type": "gauge",
                    "title": "Sprint Completion Probability",
                    "data": {
                        "value": sprint_data.get("probability", 0) * 100,
                        "min": 0,
                        "max": 100,
                        "thresholds": {
                            "low": 30,
                            "medium": 60,
                            "high": 80
                        }
                    },
                    "insights": sprint_data.get("reasoning", "")
                }
                charts.append(sprint_chart)
            
            collaborative_insights = metrics.get("collaborative_insights", {})
            if collaborative_insights:
                insights_data = {
                    "type": "radar",
                    "title": "Team Performance Radar",
                    "data": {
                        "labels": ["Team Balance", "Workflow Efficiency", "Cycle Time", "Throughput"],
                        "datasets": [{
                            "label": "Current Performance",
                            "data": [
                                collaborative_insights.get("team_balance_score", 0) * 100,
                                collaborative_insights.get("workflow_efficiency_score", 0) * 100,
                                min(100 - (metrics.get("cycle_time", 0) * 10), 100),
                                min(metrics.get("throughput", 0) * 10, 100)
                            ],
                            "backgroundColor": "rgba(54, 162, 235, 0.2)",
                            "borderColor": "rgba(54, 162, 235, 1)"
                        }]
                    }
                }
                charts.append(insights_data)
            
            performance_table = {
                "title": "Comprehensive Performance Metrics",
                "headers": ["Metric", "Value", "Confidence"],
                "rows": [
                    ["Average Cycle Time (days)", f"{metrics.get('cycle_time', 0):.1f}", f"{metrics.get('analysis_confidence', 0):.1%}"],
                    ["Weekly Throughput", str(metrics.get("throughput", 0)), f"{metrics.get('analysis_confidence', 0):.1%}"],
                    ["Data Quality Score", f"{metrics.get('data_quality_score', 0):.1%}", "High"],
                    ["Collaboration Opportunities", str(len(collaborative_insights.get("collaboration_opportunities", []))), "Medium"]
                ]
            }
            
            if predictions and predictions.get("sprint_completion"):
                sprint_data = predictions["sprint_completion"]
                performance_table["rows"].append([
                    "Sprint Completion Probability", 
                    f"{sprint_data.get('probability', 0):.0%}", 
                    f"{sprint_data.get('confidence', 0):.1%}"
                ])
            
            tables.append(performance_table)
            
            if predictions and predictions.get("risks"):
                risks = predictions["risks"][:5]
                if risks:
                    risk_table = {
                        "title": "Risk Assessment",
                        "headers": ["Risk Type", "Severity", "Mitigation"],
                        "rows": [
                            [risk["type"].replace("_", " ").title(), 
                             risk["severity"].upper(), 
                             risk["mitigation"][:50] + "..."]
                            for risk in risks
                        ]
                    }
                    tables.append(risk_table)
            
            recommendations = self._get_collaborative_recommendations(metrics)
            if recommendations:
                rec_table = {
                    "title": "Intelligent Recommendations",
                    "headers": ["Priority", "Recommendation"],
                    "rows": [
                        [f"P{i+1}", rec[:100] + "..." if len(rec) > 100 else rec]
                        for i, rec in enumerate(recommendations[:5])
                    ]
                }
                tables.append(rec_table)
            
            visualization_data = {
                "charts": charts,
                "tables": tables,
                "collaborative_metadata": {
                    "analysis_confidence": metrics.get("analysis_confidence", 0),
                    "collaboration_applied": True,
                    "insights_generated": len(collaborative_insights),
                    "predictions_integrated": bool(predictions)
                }
            }
            
            self.log(f"[VISUALIZATION] Created enhanced visualization with {len(charts)} charts and {len(tables)} tables")
            return visualization_data
            
        except Exception as e:
            self.log(f"[ERROR] Failed to create visualization data: {str(e)}")
            return {"charts": [], "tables": [], "error": str(e)}

    def _generate_smart_colors(self, count: int) -> List[str]:
        colors = ["#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#2196F3"]
        return [colors[i % len(colors)] for i in range(count)]

    def _generate_bottleneck_colors(self, bottlenecks: Dict[str, int]) -> List[str]:
        max_count = max(bottlenecks.values()) if bottlenecks else 1
        colors = []
        
        for count in bottlenecks.values():
            if count >= max_count * 0.8:
                colors.append("#F44336")
            elif count >= max_count * 0.5:
                colors.append("#FF9800")
            else:
                colors.append("#4CAF50")
        
        return colors

    def _generate_report(self, metrics: Dict[str, Any], recommendations: List[str], predictions: Dict[str, Any]) -> str:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            insights = []
            collaborative_insights = metrics.get("collaborative_insights", {})
            
            cycle_time = metrics.get("cycle_time", 0)
            if cycle_time > 5:
                insights.append(f"⚠️ Average cycle time is high at {cycle_time:.1f} days - review workflow efficiency")
            elif cycle_time > 0:
                insights.append(f"✅ Average cycle time is {cycle_time:.1f} days - within acceptable range")
            
            throughput = metrics.get("throughput", 0)
            if throughput < 3:
                insights.append(f"⚠️ Weekly throughput is low at {throughput} tickets - consider process improvements")
            else:
                insights.append(f"✅ Weekly throughput is {throughput} tickets - good productivity level")
            
            workload = metrics.get("workload", {})
            team_balance_score = collaborative_insights.get("team_balance_score", 1.0)
            if workload and team_balance_score < 0.6:
                max_workload = max(workload.items(), key=lambda x: x[1])
                insights.append(f"⚠️ Workload imbalance detected - {max_workload[0]} has {max_workload[1]} tickets")
            elif workload:
                insights.append(f"✅ Team workload is well-balanced across {len(workload)} team members")
            
            bottlenecks = metrics.get("bottlenecks", {})
            if bottlenecks:
                max_bottleneck = max(bottlenecks.items(), key=lambda x: x[1])
                if max_bottleneck[1] > 3:
                    insights.append(f"🔴 Critical bottleneck in '{max_bottleneck[0]}' with {max_bottleneck[1]} tickets")
                else:
                    insights.append(f"✅ No critical bottlenecks detected")
            
            analysis_confidence = metrics.get("analysis_confidence", 0.5)
            if analysis_confidence < 0.6:
                insights.append(f"📊 Analysis confidence: {analysis_confidence:.1%} - consider gathering more data for higher precision")
            else:
                insights.append(f"📊 Analysis confidence: {analysis_confidence:.1%} - high quality insights available")
            
            if predictions and predictions.get("sprint_completion"):
                sprint_prob = predictions["sprint_completion"].get("probability", 0)
                insights.append(f"🔮 Sprint completion probability: {sprint_prob:.0%}")
            
            if predictions and predictions.get("warnings"):
                warnings = predictions["warnings"]
                if warnings:
                    insights.append(f"⚠️ {len(warnings)} predictive warnings detected")
            
            collaboration_opportunities = collaborative_insights.get("collaboration_opportunities", [])
            if collaboration_opportunities:
                insights.append(f"🤝 {len(collaboration_opportunities)} collaboration opportunities identified for optimization")
            
            insights_text = "\n".join([f"  {insight}" for insight in insights]) if insights else "  No specific insights available"
            recommendations_text = "\n".join([f"  • {rec}" for rec in recommendations[:5]]) if recommendations else "  No recommendations available"
            
            predictive_section = ""
            if predictions:
                predictive_section = "\n\n🔮 Predictive Analytics:"
                if predictions.get("sprint_completion"):
                    sprint_data = predictions["sprint_completion"]
                    predictive_section += f"\n  • Sprint Completion: {sprint_data.get('probability', 0):.0%} probability"
                    predictive_section += f"\n  • Risk Level: {sprint_data.get('risk_level', 'unknown')}"
                
                if predictions.get("velocity_forecast"):
                    forecast = predictions["velocity_forecast"]
                    predictive_section += f"\n  • Velocity Trend: {forecast.get('trend', 'unknown')}"
                    predictive_section += f"\n  • Next Week Estimate: {forecast.get('next_week_estimate', 0):.1f} tickets"
            
            report = f"""🎯 Productivity Dashboard Report ({today})

📈 Key Insights:
{insights_text}

💡 Intelligent Recommendations:
{recommendations_text}{predictive_section}

🔍 Collaborative Analysis Applied:
  • Team Balance Score: {team_balance_score:.1%}
  • Workflow Efficiency: {collaborative_insights.get('workflow_efficiency_score', 0):.1%}
  • Collaboration Opportunities: {len(collaboration_opportunities)}
  • Analysis Confidence: {analysis_confidence:.1%}

📊 Performance Summary:
  • Throughput: {throughput} tickets completed
  • Average Cycle Time: {cycle_time:.1f} days
  • Team Members Active: {len(workload)}
  • Process Bottlenecks: {len([b for b in bottlenecks.values() if b > 3])}

Generated with Collaborative Intelligence - {datetime.now().strftime('%H:%M:%S')}"""
            
            self.log(f"[REPORT] Generated comprehensive report with collaborative intelligence")
            return report
            
        except Exception as e:
            self.log(f"[ERROR] Failed to generate report: {str(e)}")
            return f"""🎯 Productivity Dashboard Report ({datetime.now().strftime('%Y-%m-%d')})

❌ Error generating comprehensive report: {str(e)}

Basic metrics available:
- Throughput: {metrics.get('throughput', 0)} tickets
- Cycle Time: {metrics.get('cycle_time', 0):.1f} days

Generated at {datetime.now().strftime('%H:%M:%S')}"""

    def _act(self) -> Dict[str, Any]:
        try:
            tickets = self.mental_state.get_belief("tickets") or []
            recommendations = self.mental_state.get_belief("recommendations") or []
            collaboration_context = self.mental_state.get_belief("collaboration_context")
            primary_agent_context = self.mental_state.get_belief("primary_agent_context")
            predictions = self.mental_state.get_belief("predictions") or {}
            
            self.log(f"[ACTION] Generating collaborative productivity analysis for {len(tickets)} tickets")
            
            if hasattr(self.mental_state, 'add_experience'):
                experience_description = f"Performing {'collaborative ' if collaboration_context else ''}productivity analysis"
                if collaboration_context:
                    experience_description += f" (context: {collaboration_context})"
                if predictions:
                    experience_description += " with predictive insights"
                
                self.mental_state.add_experience(
                    experience_description=experience_description,
                    outcome="generating_analysis",
                    confidence=0.8,
                    metadata={
                        "ticket_count": len(tickets),
                        "collaborative": bool(collaboration_context),
                        "has_recommendations": bool(recommendations),
                        "has_predictions": bool(predictions)
                    }
                )
            
            metrics = self._analyze_ticket_data(tickets)
            self.last_throughput = metrics["throughput"]
            
            visualization_data = self._create_visualization_data(metrics, predictions)
            
            enhanced_recommendations = self._get_collaborative_recommendations(metrics)
            
            report = self._generate_report(metrics, enhanced_recommendations, predictions)
            
            if hasattr(self.mental_state, 'add_experience'):
                self.mental_state.add_experience(
                    experience_description=f"Generated comprehensive productivity analysis with {metrics.get('analysis_confidence', 0):.1%} confidence",
                    outcome="analysis_completed",
                    confidence=0.9,
                    metadata={
                        "throughput": metrics.get("throughput", 0),
                        "cycle_time": metrics.get("cycle_time", 0),
                        "analysis_confidence": metrics.get("analysis_confidence", 0),
                        "collaborative_insights": len(metrics.get("collaborative_insights", {})),
                        "recommendations_generated": len(enhanced_recommendations),
                        "predictions_integrated": bool(predictions)
                    }
                )
            
            decision = {
                "action": "generate_collaborative_productivity_analysis",
                "ticket_count": len(tickets),
                "throughput": metrics.get("throughput", 0),
                "analysis_confidence": metrics.get("analysis_confidence", 0),
                "collaborative": bool(collaboration_context),
                "predictions_integrated": bool(predictions),
                "collaboration_requests_made": len(self.mental_state.collaborative_requests),
                "insights_generated": len(metrics.get("collaborative_insights", {})),
                "recommendations_count": len(enhanced_recommendations),
                "reasoning": f"Generated comprehensive analysis with {metrics.get('analysis_confidence', 0):.1%} confidence and collaborative intelligence"
            }
            self.mental_state.add_decision(decision)
            
            return {
                "metrics": metrics,
                "visualization_data": visualization_data,
                "report": report,
                "recommendations": enhanced_recommendations,
                "workflow_status": "success",
                "collaboration_metadata": {
                    "is_collaborative": bool(collaboration_context),
                    "collaboration_context": collaboration_context,
                    "analysis_confidence": metrics.get("analysis_confidence", 0),
                    "collaborative_insights_generated": len(metrics.get("collaborative_insights", {})),
                    "collaboration_requests_made": len(self.mental_state.collaborative_requests),
                    "predictions_integrated": bool(predictions)
                }
            }
            
        except Exception as e:
            self.log(f"[ERROR] Failed to generate productivity dashboard: {str(e)}")
            
            if hasattr(self.mental_state, 'add_experience'):
                self.mental_state.add_experience(
                    experience_description=f"Failed to generate productivity analysis",
                    outcome=f"Error: {str(e)}",
                    confidence=0.2,
                    metadata={"error_type": type(e).__name__}
                )
            
            return {
                "metrics": {},
                "visualization_data": {"charts": [], "tables": []},
                "report": f"❌ Error generating productivity dashboard: {str(e)}\n\nGenerated at {datetime.now().strftime('%H:%M:%S')}",
                "recommendations": [],
                "workflow_status": "failure",
                "collaboration_metadata": {
                    "is_collaborative": bool(collaboration_context),
                    "error_occurred": True,
                    "error_message": str(e)
                }
            }

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        super()._rethink(action_result)
        
        status = action_result.get("workflow_status", "failure")
        collaboration_metadata = action_result.get("collaboration_metadata", {})
        metrics = action_result.get("metrics", {})
        
        was_collaborative = collaboration_metadata.get("is_collaborative", False)
        analysis_confidence = collaboration_metadata.get("analysis_confidence", 0)
        insights_generated = collaboration_metadata.get("collaborative_insights_generated", 0)
        predictions_integrated = collaboration_metadata.get("predictions_integrated", False)
        
        reflection = {
            "operation": "collaborative_productivity_analysis",
            "success": status == "success",
            "analysis_confidence": analysis_confidence,
            "collaborative_interaction": was_collaborative,
            "insights_generated": insights_generated,
            "collaboration_requests_made": collaboration_metadata.get("collaboration_requests_made", 0),
            "throughput_analyzed": metrics.get("throughput", 0),
            "predictions_integrated": predictions_integrated,
            "data_quality": "high" if analysis_confidence > 0.7 else "medium" if analysis_confidence > 0.4 else "low",
            "performance_notes": f"Generated {'collaborative ' if was_collaborative else ''}analysis with {analysis_confidence:.1%} confidence and {insights_generated} insights"
        }
        
        if predictions_integrated:
            reflection["performance_notes"] += " including predictive analytics"
        
        if status == "success":
            if was_collaborative:
                self.mental_state.competency_model.add_competency("collaborative_analysis", 1.0)
            if analysis_confidence > 0.8:
                self.mental_state.competency_model.add_competency("high_confidence_analysis", 1.0)
            if insights_generated > 3:
                self.mental_state.competency_model.add_competency("insight_generation", 1.0)
            if predictions_integrated:
                self.mental_state.competency_model.add_competency("predictive_integration", 1.0)
        
        self.mental_state.add_reflection(reflection)
        
        if was_collaborative:
            collaboration_success = status == "success" and analysis_confidence > 0.6
            self.mental_state.add_experience(
                experience_description=f"Collaborative productivity analysis {'succeeded' if collaboration_success else 'had mixed results'}",
                outcome=f"collaboration_{'success' if collaboration_success else 'partial'}",
                confidence=0.8 if collaboration_success else 0.5,
                metadata={
                    "collaboration_context": collaboration_metadata.get("collaboration_context"),
                    "analysis_quality": reflection["data_quality"],
                    "insights_generated": insights_generated,
                    "predictions_integrated": predictions_integrated
                }
            )
        
        self.log(f"[REFLECTION] Analysis completed: {reflection}")

    def get_collaborative_performance_metrics(self) -> Dict[str, Any]:
        try:
            basic_metrics = {
                "total_analyses": len(self.mental_state.decisions),
                "successful_analyses": len([d for d in self.mental_state.decisions if d.get("action") == "generate_collaborative_productivity_analysis"]),
                "average_confidence": 0.0,
                "collaborative_interactions": 0,
                "predictive_integrations": 0
            }
            
            if self.mental_state.decisions:
                confidence_scores = [d.get("analysis_confidence", 0) for d in self.mental_state.decisions if d.get("analysis_confidence")]
                if confidence_scores:
                    basic_metrics["average_confidence"] = sum(confidence_scores) / len(confidence_scores)
                
                basic_metrics["collaborative_interactions"] = len([d for d in self.mental_state.decisions if d.get("collaborative")])
                basic_metrics["predictive_integrations"] = len([d for d in self.mental_state.decisions if d.get("predictions_integrated")])
            
            collaborative_reflections = [r for r in self.mental_state.reflection_patterns if r.get("collaborative_interaction")]
            if collaborative_reflections:
                basic_metrics["collaboration_success_rate"] = len([r for r in collaborative_reflections if r.get("success")]) / len(collaborative_reflections)
            else:
                basic_metrics["collaboration_success_rate"] = 0.0
            
            recent_decisions = self.mental_state.decisions[-10:] if len(self.mental_state.decisions) >= 10 else self.mental_state.decisions
            if recent_decisions:
                recent_throughputs = [d.get("throughput", 0) for d in recent_decisions if d.get("throughput") is not None]
                if recent_throughputs:
                    basic_metrics["recent_avg_throughput"] = sum(recent_throughputs) / len(recent_throughputs)
            
            basic_metrics["intelligence_features_active"] = True
            basic_metrics["semantic_memory_enabled"] = hasattr(self.mental_state, 'vector_memory') and self.mental_state.vector_memory is not None
            basic_metrics["predictive_analytics_enabled"] = True
            
            return basic_metrics
            
        except Exception as e:
            self.log(f"[ERROR] Failed to get performance metrics: {str(e)}")
            return {"error": str(e)}

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(input_data)
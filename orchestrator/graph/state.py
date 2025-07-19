from typing import TypedDict, List, Dict, Any, Optional

class JurixState(TypedDict, total=False):
    query: str
    intent: Dict[str, Any]
    conversation_id: str
    conversation_history: List[Dict[str, str]]
    articles: List[Dict[str, Any]]
    recommendations: List[str]
    tickets: List[Dict[str, Any]]
    status: str
    response: str
    articles_used: List[Dict[str, Any]]
    workflow_status: str
    next_agent: str
    project: Optional[str]
    project_id: Optional[str]
    time_range: Optional[Dict[str, str]]
    metrics: Optional[Dict[str, Any]]
    visualization_data: Optional[Dict[str, Any]]
    report: Optional[str]
    metadata: Optional[Dict[str, Any]]
    ticket_id: Optional[str]
    article: Optional[Dict[str, Any]]
    redundant: Optional[bool]
    refinement_suggestion: Optional[str]
    approved: Optional[bool]
    refinement_count: Optional[int]
    has_refined: Optional[bool]
    iteration_count: Optional[int]
    workflow_stage: Optional[str]
    recommendation_id: Optional[str]
    workflow_history: Optional[List[Dict[str, Any]]]
    error: Optional[str]
    recommendation_status: Optional[str]
    dashboard_id: Optional[str]
    collaboration_metadata: Optional[Dict[str, Any]]
    final_collaboration_summary: Optional[Dict[str, Any]]
    collaboration_insights: Optional[Dict[str, Any]]
    collaboration_trace: Optional[List[Dict[str, Any]]]
    collaborative_agents_used: Optional[List[str]]
    autonomous_refinement_done: Optional[bool]
    predictions: Optional[Dict[str, Any]]
    predictive_insights: Optional[str]
    analysis_type: Optional[str]
    articles_from_collaboration: Optional[List[Dict[str, Any]]]
    context_enrichment_successful: Optional[bool]
    needs_context: Optional[bool]
    force_fresh_data: Optional[bool] 
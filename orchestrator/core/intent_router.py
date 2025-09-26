# orchestrator/core/intent_router.py - COMPLETE FIXED VERSION
from typing import Dict, List
import json
import logging

logger = logging.getLogger("IntentRouter")

def classify_intent(query: str, history: List[Dict[str, str]]) -> Dict:
    """Enhanced intent classification with better keyword detection"""
    query_lower = query.lower()
    
    logger.info(f"[INTENT CLASSIFIER] Analyzing query: {query}")
    result = {"intent": "generic_question"}
    words = query.split()
    project = None
    for word in words:
        if word.isupper() and len(word) > 2 and word not in ["API", "URL", "IDE", "SQL", "AWS", "REST"]:
            project = word
            logger.info(f"[INTENT CLASSIFIER] Found project: {project}")
            break
    recommendation_keywords = [
        "recommend", "recommendation", "recommendations", "suggest", "suggestion", "suggestions",
        "advice", "improve", "improvement", "optimize", "optimization", "enhance",
        "best practice", "should i", "should we", "how can we", "what can we do",
        "tips", "strategy", "approach", "specific recommendations"
    ]
    has_recommendation_keyword = any(keyword in query_lower for keyword in recommendation_keywords)
    
    if has_recommendation_keyword:
        result["intent"] = "recommendation"
        logger.info(f"[INTENT CLASSIFIER] Detected RECOMMENDATION intent")
    elif any(keyword in query_lower for keyword in [
        "predict", "forecast", "probability", "chance", "will we complete",
        "sprint completion", "velocity", "trend", "risk", "burnout",
        "bottleneck", "analyze", "analysis", "when will", "estimate"
    ]):
        result["intent"] = "predictive_analysis"
        logger.info(f"[INTENT CLASSIFIER] Detected PREDICTIVE intent")
    elif any(keyword in query_lower for keyword in [
        "article", "articles", "documentation", "docs", "find", "search",
        "show me", "look for", "guide", "tutorial"
    ]):
        result["intent"] = "article_retrieval"
        logger.info(f"[INTENT CLASSIFIER] Detected ARTICLE intent")
  
    if project:
        result["project"] = project
    
    logger.info(f"[INTENT CLASSIFIER] Final result: {result}")
    return result
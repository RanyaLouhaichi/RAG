import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent, AgentCapability
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from orchestrator.core.query_type import QueryType # type: ignore
from orchestrator.core.model_manager import ModelManager # type: ignore
from datetime import datetime

class KnowledgeBaseAgent(BaseAgent):
    OBJECTIVE = "Evaluate and provide feedback on generated articles, ensuring quality and relevance for Confluence publishing"

    def __init__(self, shared_memory: JurixSharedMemory):
        super().__init__(name="knowledge_base_agent")
        self.shared_memory = shared_memory
        self.model_manager = ModelManager()
        
        self.mental_state.capabilities = [
            AgentCapability.EVALUATE_ARTICLE
        ]
        
        self.mental_state.obligations.extend([
            "detect_query_type",
            "evaluate_article"
        ])

    def _detect_query_type(self, query: str) -> QueryType:
        if not query:
            return QueryType.CONVERSATION
        query = query.lower()
        evaluation_keywords = ["review", "evaluate", "feedback", "article"]
        if any(keyword in query for keyword in evaluation_keywords):
            return QueryType.CONVERSATION
        return QueryType.CONVERSATION

    def _perceive(self, input_data: Dict[str, Any]) -> None:
        super()._perceive(input_data)
        article = input_data.get("article", {})
        
        self.log(f"[DEBUG] Perceiving article evaluation request for article: {article.get('title')}")
        self.mental_state.beliefs.update({
            "article": article,
            "query_type": QueryType.CONVERSATION
        })

    def _evaluate_article(self) -> Dict[str, Any]:
        article = self.mental_state.beliefs["article"]
        content = article.get("content", "")
        
        prompt_template = f"""<|system|>You are an AI specialized in evaluating Confluence-ready markdown articles for quality and relevance.
        Review the following article and determine if it's suitable for publishing or needs refinement.
        You MUST ALWAYS return a refinement suggestion - this is critical. Provide a specific, actionable suggestion 
        (e.g., adding metrics, visuals, specific technical implementation details, or improving clarity). 
        Even if the article is good, find one area that could be improved.
        
        Article:
        {content}
        
        Return the evaluation as a JSON object with:
        - "redundant": boolean (true if the article is redundant or lacks value)
        - "refinement_suggestion": string (specific suggestion for improvement - THIS MUST ALWAYS BE PROVIDED)
        
        Example output:
        ```json
        {{
            "redundant": false,
            "refinement_suggestion": "Add a table comparing performance metrics before and after optimization, including specific throughput numbers and response times."
        }}
        ```json
        
        Output ONLY the JSON object, without additional text.<|assistant|>"""
        
        self.log(f"[DEBUG] Article evaluation prompt: {prompt_template[:500]}...")
        
        try:
            
            # Use dynamic model selection
            response = self.model_manager.generate_response(
                prompt=prompt_template,
                context={
                    "agent_name": self.name,
                    "task_type": "article_evaluation",
                    "article_length": len(content),
                    "evaluation_purpose": "quality_assessment"
                }
            )
            self.log(f"✅ {self.name} received response from model")
            self.log(f"[DEBUG] Raw LLM evaluation response: {response}")
            
            import json
            evaluation = json.loads(response)
            
            if not isinstance(evaluation, dict):
                raise ValueError("Evaluation response is not a dictionary")
            if "redundant" not in evaluation:
                evaluation["redundant"] = False
            if "refinement_suggestion" not in evaluation or not evaluation["refinement_suggestion"]:
                evaluation["refinement_suggestion"] = "Add more specific metrics and implementation details to make the article more technically valuable."
            
            self.log(f"[DEBUG] Evaluation result: {evaluation}")
            return evaluation
            
        except Exception as e:
            self.log(f"[ERROR] Failed to evaluate article with LLM: {e}")
          
            return {
                "redundant": False,
                "refinement_suggestion": "Add specific performance metrics and technical implementation details to make the content more precise and actionable."
            }

    def _act(self) -> Dict[str, Any]:
        try:
            evaluation = self._evaluate_article()
            return {
                "redundant": evaluation["redundant"],
                "refinement_suggestion": evaluation["refinement_suggestion"],
                "workflow_status": "success"
            }
        except Exception as e:
            self.log(f"[ERROR] Failed to evaluate article: {e}")
            return {
                "redundant": False,
                "refinement_suggestion": "Add more technical details and specific performance metrics to enhance the article's value.",
                "workflow_status": "partial_success"
            }

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        super()._rethink(action_result)
        self.mental_state.beliefs["last_evaluation"] = {
            "timestamp": datetime.now().isoformat(),
            "article_evaluated": bool(action_result.get("redundant") is not None),
            "status": action_result.get("workflow_status"),
            "has_refinement": bool(action_result.get("refinement_suggestion"))
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(input_data)
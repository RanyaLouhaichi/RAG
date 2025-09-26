import ollama
import redis 
import logging
import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict
from orchestrator.monitoring.langsmith_config import langsmith_monitor # type: ignore

class ModelManager:
    """Truly Dynamic Model Manager - No Static Mappings!"""
    
    def __init__(self, model_name: str = "mistral", redis_client: Optional[redis.Redis] = None):
        self.default_model = model_name
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.logger = logging.getLogger("ModelManager")

        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:] 
                detected_models = []
                for line in lines:
                    if line:
                        model_name = line.split()[0].split(':')[0]  
                        if model_name not in detected_models:
                            detected_models.append(model_name)
                
                if detected_models:
                    self.available_models = detected_models
                    self.logger.info(f"🔍 Auto-detected models: {self.available_models}")
                else:
                    self.available_models = ["mistral"]  
                    self.logger.warning("⚠️ No models detected, using only mistral")
            else:
                self.available_models = ["mistral"]
                self.logger.warning("⚠️ Could not detect models, using only mistral")
        except Exception as e:
            self.logger.error(f"❌ Model detection failed: {e}")
            self.available_models = ["mistral"]

        self.available_models = ["mistral", "llama3.2", "codellama", "phi"]

        self.learning_rate = 0.15
        self.exploration_rate = 0.30 

        self.agent_performance = defaultdict(lambda: defaultdict(lambda: {
            "successes": 0,
            "failures": 0,
            "total_time": 0.0,
            "quality_sum": 0.0,
            "recent_uses": []
        }))
        self._load_agent_performance()
        
        self.current_session = {
            "requests_by_agent": defaultdict(int),
            "models_used_by_agent": defaultdict(list),
            "switch_count": 0,
            "start_time": datetime.now()
        }
        
        self.logger.info("🚀 Dynamic ModelManager initialized - No static mappings!")


    def start_workflow_tracking(self, workflow_id: str):
        """Start tracking model usage for a specific workflow"""
        self.current_workflow_id = workflow_id
        workflow_key = f"workflow_models:{workflow_id}"
        self.redis_client.set(workflow_key, json.dumps({
            "start_time": datetime.now().isoformat(),
            "agents": {}
        }))
        self.redis_client.expire(workflow_key, 3600)

    def track_workflow_usage(self, agent_name: str, model: str, success: bool, quality: float):
        """Track model usage within current workflow"""
        if hasattr(self, 'current_workflow_id'):
            workflow_key = f"workflow_models:{self.current_workflow_id}"
            workflow_data = self.redis_client.get(workflow_key)
            
            if workflow_data:
                data = json.loads(workflow_data)
                if agent_name not in data["agents"]:
                    data["agents"][agent_name] = []
                
                data["agents"][agent_name].append({
                    "model": model,
                    "timestamp": datetime.now().isoformat(),
                    "success": success,
                    "quality": quality
                })
                
                self.redis_client.set(workflow_key, json.dumps(data))

    def get_workflow_summary(self, workflow_id: str = None) -> Dict[str, Any]:
        """Get summary of model usage in a workflow"""
        if workflow_id is None and hasattr(self, 'current_workflow_id'):
            workflow_id = self.current_workflow_id
        
        if not workflow_id:
            return {}
        
        workflow_key = f"workflow_models:{workflow_id}"
        workflow_data = self.redis_client.get(workflow_key)
        
        if not workflow_data:
            return {}
        
        data = json.loads(workflow_data)
        summary = {
            "workflow_id": workflow_id,
            "start_time": data.get("start_time"),
            "agent_model_usage": {}
        }
        
        for agent_name, uses in data.get("agents", {}).items():
            model_counts = {}
            for use in uses:
                model = use["model"]
                model_counts[model] = model_counts.get(model, 0) + 1
            
            summary["agent_model_usage"][agent_name] = {
                "models_used": list(set(use["model"] for use in uses)),
                "model_counts": model_counts,
                "total_calls": len(uses)
            }
        
        return summary

    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Main entry point - fully dynamic model selection"""
        context = context or {}
        agent_name = context.get("agent_name", "unknown")
        
        self.current_session["requests_by_agent"][agent_name] += 1
        
        selected_model = self._select_model_for_agent(agent_name, prompt, context)
        
        self.logger.info(f"🎯 {agent_name} dynamically selected: {selected_model}")

        start_time = datetime.now()
        try:
            response = ollama.generate(model=selected_model, prompt=prompt)
            success = True
            actual_response = response["response"]
            response_time = (datetime.now() - start_time).total_seconds()
            langsmith_monitor.log_model_usage(
                agent_name=agent_name,
                model_name=selected_model,
                prompt=prompt,
                response=actual_response,
                latency=response_time
            )
        except Exception as e:
            self.logger.warning(f"❌ {selected_model} failed for {agent_name}: {e}")
            fallback_model = self._get_best_fallback(agent_name, selected_model)
            try:
                response = ollama.generate(model=fallback_model, prompt=prompt)
                success = True
                actual_response = response["response"]
                selected_model = fallback_model
                self.current_session["switch_count"] += 1
            except:
                success = False
                actual_response = "I apologize, I'm having technical difficulties."

        response_time = (datetime.now() - start_time).total_seconds()
        quality = self._assess_quality(actual_response, prompt, context)
    
        self._update_agent_performance(agent_name, selected_model, success, response_time, quality)

        self.current_session["models_used_by_agent"][agent_name].append(selected_model)
        self._log_usage(agent_name, selected_model, response_time, success, quality)
        self.track_workflow_usage(agent_name, selected_model, success, quality)
        
        return actual_response

    def _select_model_for_agent(self, agent_name: str, prompt: str, context: Dict[str, Any]) -> str:
        """Dynamically select best model for this specific agent and prompt"""
        if np.random.random() < self.exploration_rate:
            selected = np.random.choice(self.available_models)
            self.logger.info(f"🎲 {agent_name} exploring with {selected}")
            return selected
        model_scores = {}    
        for model in self.available_models: 
            agent_history = self.agent_performance[agent_name][model]         
            if agent_history["successes"] + agent_history["failures"] > 0:
                success_rate = agent_history["successes"] / (agent_history["successes"] + agent_history["failures"])
                avg_quality = agent_history["quality_sum"] / max(agent_history["successes"], 1)
                avg_time = agent_history["total_time"] / max(agent_history["successes"] + agent_history["failures"], 1)
                time_score = 1.0 / (1.0 + avg_time / 10.0)
                base_score = 0.4 * success_rate + 0.4 * avg_quality + 0.2 * time_score
            else:
                base_score = 0.5 
            recent_boost = self._calculate_recent_performance(agent_name, model)
            context_score = self._calculate_context_score(agent_name, model, prompt, context)
            model_scores[model] = base_score * 0.6 + recent_boost * 0.2 + context_score * 0.2
        best_model = max(model_scores, key=model_scores.get) 
        self.logger.info(f"📊 {agent_name} scores: {json.dumps(model_scores, default=lambda x: round(x, 3))}")
        return best_model

    def _calculate_recent_performance(self, agent_name: str, model: str) -> float:
        """Calculate boost based on last 5 uses"""
        recent = self.agent_performance[agent_name][model]["recent_uses"][-5:]
        if not recent:
            return 0.5
        total_score = sum(use["quality"] * (1.0 if use["success"] else 0.3) for use in recent)
        return total_score / len(recent)

    def _calculate_context_score(self, agent_name: str, model: str, prompt: str, context: Dict[str, Any]) -> float:
        """Dynamic context scoring - learns patterns"""
        score = 0.5
        prompt_features = self._extract_prompt_features(prompt)
        pattern_key = f"pattern:{agent_name}:{model}:{prompt_features['type']}"
        
        pattern_success = self.redis_client.get(pattern_key)
        if pattern_success:
            score += float(pattern_success) * 0.3
        if context.get("collaboration_context"):
            collab_key = f"collab_performance:{agent_name}:{model}"
            collab_score = self.redis_client.get(collab_key)
            if collab_score:
                score += float(collab_score) * 0.2
        
        return min(score, 1.0)

    def _extract_prompt_features(self, prompt: str) -> Dict[str, Any]:
        """Extract features for pattern learning"""
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ["analyze", "evaluate", "assess", "measure"]):
            prompt_type = "analytical"
        elif any(word in prompt_lower for word in ["create", "generate", "write", "design"]):
            prompt_type = "creative"
        elif any(word in prompt_lower for word in ["predict", "forecast", "will", "future"]):
            prompt_type = "predictive"
        elif any(word in prompt_lower for word in ["recommend", "suggest", "should", "improve"]):
            prompt_type = "advisory"
        else:
            prompt_type = "general"
        
        return {
            "type": prompt_type,
            "length": len(prompt),
            "complexity": prompt.count(",") + prompt.count(".") + prompt.count("("),
            "has_data": "data" in prompt_lower or "metrics" in prompt_lower,
            "has_code": "code" in prompt_lower or "function" in prompt_lower
        }

    def _assess_quality(self, response: str, prompt: str, context: Dict[str, Any]) -> float:
        """Assess response quality (0-1)"""
        if not response or "error" in response.lower() or "apologize" in response.lower():
            return 0.1
        
        quality = 0.5
        if 50 < len(response) < 2000:
            quality += 0.2
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words & response_words) / max(len(prompt_words), 1)
        quality += overlap * 0.2
        agent_name = context.get("agent_name", "")
        
        if "recommendation" in agent_name and any(word in response.lower() for word in ["recommend", "suggest", "consider"]):
            quality += 0.1
        elif "chat" in agent_name and "?" in response:
            quality += 0.1
        elif "data" in agent_name and any(word in response.lower() for word in ["data", "metrics", "analysis"]):
            quality += 0.1
        
        return min(quality, 1.0)

    def _get_best_fallback(self, agent_name: str, failed_model: str) -> str:
        """Get best fallback model for this agent"""
        candidates = [m for m in self.available_models if m != failed_model]
        scores = {}
        for model in candidates:
            history = self.agent_performance[agent_name][model]
            if history["successes"] > 0:
                scores[model] = history["successes"] / (history["successes"] + history["failures"])
            else:
                scores[model] = 0.4 
        return max(scores, key=scores.get)

    def _update_agent_performance(self, agent_name: str, model: str, success: bool, 
                                response_time: float, quality: float):
        """Update performance tracking for this agent-model combination"""
        perf = self.agent_performance[agent_name][model]
        if success:
            perf["successes"] += 1
            perf["quality_sum"] += quality
        else:
            perf["failures"] += 1
        
        perf["total_time"] += response_time
        perf["recent_uses"].append({
            "success": success,
            "quality": quality,
            "time": response_time,
            "timestamp": datetime.now().isoformat()
        })
        if len(perf["recent_uses"]) > 10:
            perf["recent_uses"] = perf["recent_uses"][-10:]
        prompt_features = self._extract_prompt_features("") 
        pattern_key = f"pattern:{agent_name}:{model}:{prompt_features['type']}"
        
        old_pattern_score = float(self.redis_client.get(pattern_key) or 0.5)
        new_pattern_score = old_pattern_score * 0.9 + (quality if success else 0.1) * 0.1
        self.redis_client.setex(pattern_key, 3600, str(new_pattern_score))
        self._save_agent_performance()

    def _log_usage(self, agent_name: str, model: str, response_time: float, 
                  success: bool, quality: float):
        """Log usage with full details"""
        self.logger.info(
            f"{'✅' if success else '❌'} {agent_name} → {model} | "
            f"Time: {response_time:.3f}s | Quality: {quality:.2f}"
        )
        log_entry = {
            "agent": agent_name,
            "model": model,
            "success": success,
            "quality": quality,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }
        log_key = f"model_log:{datetime.now().strftime('%Y%m%d_%H')}"
        self.redis_client.lpush(log_key, json.dumps(log_entry))
        self.redis_client.expire(log_key, 86400)

    def get_agent_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance stats per agent"""
        stats = {
            "session_info": {
                "duration": (datetime.now() - self.current_session["start_time"]).total_seconds(),
                "total_requests": sum(self.current_session["requests_by_agent"].values()),
                "model_switches": self.current_session["switch_count"]
            },
            "by_agent": {}
        }
        for agent_name in set(list(self.agent_performance.keys()) + list(self.current_session["requests_by_agent"].keys())):
            agent_stats = {
                "total_requests": self.current_session["requests_by_agent"].get(agent_name, 0),
                "models_used": list(set(self.current_session["models_used_by_agent"].get(agent_name, []))),
                "model_performance": {}
            }
            for model in self.available_models:
                perf = self.agent_performance[agent_name][model]
                total = perf["successes"] + perf["failures"]
                
                if total > 0:
                    agent_stats["model_performance"][model] = {
                        "uses": total,
                        "success_rate": perf["successes"] / total,
                        "avg_quality": perf["quality_sum"] / max(perf["successes"], 1),
                        "avg_time": perf["total_time"] / total
                    }
            if agent_stats["model_performance"]:
                scores = {}
                for model, perf in agent_stats["model_performance"].items():
                    scores[model] = perf["success_rate"] * 0.5 + perf["avg_quality"] * 0.5
                agent_stats["preferred_model"] = max(scores, key=scores.get)
            
            stats["by_agent"][agent_name] = agent_stats
        
        return stats

    def _save_agent_performance(self):
        """Save performance data to Redis"""
        for agent_name, models in self.agent_performance.items():
            for model, perf in models.items():
                key = f"agent_model_perf:{agent_name}:{model}"
                self.redis_client.set(key, json.dumps(perf))
                self.redis_client.expire(key, 86400 * 7) 

    def _load_agent_performance(self):
        """Load historical performance data"""
        pattern = "agent_model_perf:*"
        for key in self.redis_client.keys(pattern):
            parts = key.split(":")
            if len(parts) >= 3:
                agent_name = parts[1]
                model = parts[2]
                data = self.redis_client.get(key)
                if data:
                    self.agent_performance[agent_name][model] = json.loads(data)

    def generate_for_agent(self, agent_name: str, prompt: str, **kwargs) -> str:
        context = kwargs.copy()
        context["agent_name"] = agent_name
        return self.generate_response(prompt, context)

    def set_agent_context(self, agent_name: str, **kwargs):
        pass
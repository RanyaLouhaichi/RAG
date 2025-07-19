import ollama
import hashlib
import json
import redis
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum

class ReasoningType(Enum):
    """Types of reasoning that require different cognitive models"""
    CONVERSATIONAL = "conversational"           # Chat, dialogue, explanation
    TEMPORAL_ANALYSIS = "temporal_analysis"     # Time-based patterns, trends
    STRATEGIC_REASONING = "strategic_reasoning" # Planning, recommendations  
    CREATIVE_WRITING = "creative_writing"       # Article generation, content creation
    LOGICAL_REASONING = "logical_reasoning"     # Problem solving, debugging
    DATA_ANALYSIS = "data_analysis"            # Pattern recognition, classification

class CognitiveModelManager:
    """Manages specialized models for different types of reasoning"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.logger = logging.getLogger("CognitiveModelManager")
        
        # Model specialization mapping
        self.specialized_models = {
            ReasoningType.CONVERSATIONAL: "mistral",        # Good for dialogue
            ReasoningType.TEMPORAL_ANALYSIS: "llama3.2",    # Good for patterns
            ReasoningType.STRATEGIC_REASONING: "mistral",    # Good for planning
            ReasoningType.CREATIVE_WRITING: "llama3.2",     # Good for content
            ReasoningType.LOGICAL_REASONING: "codellama",   # Good for logic (if available)
            ReasoningType.DATA_ANALYSIS: "mistral"         # Good for analysis
        }
        
        # Fallback model if specialized model fails
        self.fallback_model = "mistral"
        
        # Cache settings
        self.cache_ttl = 3600  # 1 hour cache
        self.enable_semantic_cache = True
        
        self.logger.info("Initialized CognitiveModelManager with Redis semantic caching")

    def _detect_reasoning_type(self, prompt: str, context: Dict[str, Any] = None) -> ReasoningType:
        """Automatically detect the type of reasoning needed based on prompt and context"""
        prompt_lower = prompt.lower()
        context = context or {}
        
        # Check context clues first
        agent_name = context.get("agent_name", "").lower()
        task_type = context.get("task_type", "").lower()
        
        # Agent-based detection
        if "chat" in agent_name:
            return ReasoningType.CONVERSATIONAL
        elif "jira_data" in agent_name or "data" in agent_name:
            return ReasoningType.DATA_ANALYSIS
        elif "recommendation" in agent_name:
            return ReasoningType.STRATEGIC_REASONING
        elif "article" in agent_name or "generator" in agent_name:
            return ReasoningType.CREATIVE_WRITING
        
        # Content-based detection
        temporal_keywords = ["time", "trend", "pattern", "cycle", "history", "velocity", "over time"]
        if any(keyword in prompt_lower for keyword in temporal_keywords):
            return ReasoningType.TEMPORAL_ANALYSIS
        
        creative_keywords = ["write", "create", "generate", "article", "content", "documentation"]
        if any(keyword in prompt_lower for keyword in creative_keywords):
            return ReasoningType.CREATIVE_WRITING
        
        strategic_keywords = ["recommend", "suggest", "improve", "optimize", "strategy", "plan"]
        if any(keyword in prompt_lower for keyword in strategic_keywords):
            return ReasoningType.STRATEGIC_REASONING
        
        logical_keywords = ["debug", "solve", "problem", "error", "fix", "analyze"]
        if any(keyword in prompt_lower for keyword in logical_keywords):
            return ReasoningType.LOGICAL_REASONING
        
        # Default to conversational
        return ReasoningType.CONVERSATIONAL

    def _get_cache_key(self, prompt: str, reasoning_type: ReasoningType, model: str) -> str:
        """Generate cache key for semantic caching"""
        # Create a hash of the prompt for consistent caching
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return f"model_cache:{reasoning_type.value}:{model}:{prompt_hash}"

    def _get_semantic_cache_key(self, prompt: str, reasoning_type: ReasoningType) -> str:
        """Generate semantic similarity cache key"""
        # For semantic caching, we'll use a simplified version
        # In a full implementation, this would use embeddings
        prompt_words = set(prompt.lower().split())
        key_words = sorted(list(prompt_words))[:10]  # Top 10 words for similarity
        semantic_signature = "_".join(key_words)
        return f"semantic_cache:{reasoning_type.value}:{semantic_signature}"

    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if response is cached"""
        try:
            cached_response = self.redis_client.get(cache_key)
            if cached_response:
                self.logger.info(f"Cache hit for key: {cache_key}")
                return cached_response
        except Exception as e:
            self.logger.error(f"Cache check failed: {e}")
        return None

    def _store_cache(self, cache_key: str, response: str):
        """Store response in cache"""
        try:
            self.redis_client.set(cache_key, response, ex=self.cache_ttl)
            self.logger.info(f"Cached response for key: {cache_key}")
        except Exception as e:
            self.logger.error(f"Cache storage failed: {e}")

    def _track_model_performance(self, reasoning_type: ReasoningType, model: str, 
                                success: bool, response_time: float):
        """Track model performance for future optimization"""
        try:
            metrics_key = f"model_metrics:{reasoning_type.value}:{model}"
            timestamp = datetime.now().isoformat()
            
            metric = {
                "timestamp": timestamp,
                "success": success,
                "response_time": response_time,
                "reasoning_type": reasoning_type.value,
                "model": model
            }
            
            # Store in Redis list (keep last 100 metrics per model)
            self.redis_client.lpush(metrics_key, json.dumps(metric))
            self.redis_client.ltrim(metrics_key, 0, 99)  # Keep only last 100
            self.redis_client.expire(metrics_key, 86400)  # 24 hour expiration
            
        except Exception as e:
            self.logger.error(f"Performance tracking failed: {e}")

    def generate_response(self, prompt: str, reasoning_type: Optional[ReasoningType] = None, 
                         context: Dict[str, Any] = None, force_model: Optional[str] = None) -> str:
        """Generate response using specialized cognitive model"""
        start_time = datetime.now()
        context = context or {}
        
        # Detect reasoning type if not provided
        if reasoning_type is None:
            reasoning_type = self._detect_reasoning_type(prompt, context)
        
        # Select model (allow override for testing)
        selected_model = force_model or self.specialized_models.get(reasoning_type, self.fallback_model)
        
        self.logger.info(f"Using model '{selected_model}' for {reasoning_type.value} reasoning")
        
        # Check cache first
        cache_key = self._get_cache_key(prompt, reasoning_type, selected_model)
        if self.enable_semantic_cache:
            cached_response = self._check_cache(cache_key)
            if cached_response:
                return cached_response
        
        # Generate fresh response
        try:
            # Add reasoning-specific context to prompt
            enhanced_prompt = self._enhance_prompt_for_reasoning(prompt, reasoning_type, context)
            
            response = ollama.generate(model=selected_model, prompt=enhanced_prompt)
            generated_text = response["response"]
            
            # Cache the response
            if self.enable_semantic_cache:
                self._store_cache(cache_key, generated_text)
            
            # Track performance
            response_time = (datetime.now() - start_time).total_seconds()
            self._track_model_performance(reasoning_type, selected_model, True, response_time)
            
            self.logger.info(f"Generated response using {selected_model} in {response_time:.2f}s")
            return generated_text
            
        except Exception as e:
            # Try fallback model
            self.logger.warning(f"Model {selected_model} failed: {e}. Trying fallback.")
            
            try:
                fallback_prompt = self._enhance_prompt_for_reasoning(prompt, reasoning_type, context)
                response = ollama.generate(model=self.fallback_model, prompt=fallback_prompt)
                generated_text = response["response"]
                
                # Track fallback performance
                response_time = (datetime.now() - start_time).total_seconds()
                self._track_model_performance(reasoning_type, f"{selected_model}_fallback", True, response_time)
                
                return generated_text
                
            except Exception as fallback_error:
                self.logger.error(f"Both primary and fallback models failed: {fallback_error}")
                response_time = (datetime.now() - start_time).total_seconds()
                self._track_model_performance(reasoning_type, selected_model, False, response_time)
                raise Exception(f"All models failed: {str(e)}, {str(fallback_error)}")

    def _enhance_prompt_for_reasoning(self, prompt: str, reasoning_type: ReasoningType, 
                                    context: Dict[str, Any]) -> str:
        """Enhance prompt based on reasoning type for better results"""
        
        # Add reasoning-specific instructions
        if reasoning_type == ReasoningType.CONVERSATIONAL:
            enhanced_prompt = f"""<|system|>You are having a natural conversation. Be helpful, engaging, and conversational. Focus on clear communication and understanding the user's needs.

<|user|>{prompt}<|assistant|>"""

        elif reasoning_type == ReasoningType.TEMPORAL_ANALYSIS:
            enhanced_prompt = f"""<|system|>You are analyzing temporal patterns and trends. Focus on time-based relationships, cycles, trends, and historical patterns. Be analytical and data-driven.

<|user|>{prompt}<|assistant|>"""

        elif reasoning_type == ReasoningType.STRATEGIC_REASONING:
            enhanced_prompt = f"""<|system|>You are providing strategic recommendations and planning advice. Think systematically about goals, constraints, trade-offs, and actionable steps. Be practical and results-oriented.

<|user|>{prompt}<|assistant|>"""

        elif reasoning_type == ReasoningType.CREATIVE_WRITING:
            enhanced_prompt = f"""<|system|>You are creating well-structured, informative content. Focus on clarity, organization, and engaging writing. Use appropriate formatting and structure.

<|user|>{prompt}<|assistant|>"""

        elif reasoning_type == ReasoningType.LOGICAL_REASONING:
            enhanced_prompt = f"""<|system|>You are solving problems through logical analysis. Break down complex issues step-by-step, identify root causes, and provide systematic solutions.

<|user|>{prompt}<|assistant|>"""

        else:  # DATA_ANALYSIS
            enhanced_prompt = f"""<|system|>You are analyzing data and identifying patterns. Focus on extracting insights, identifying trends, and providing data-driven conclusions.

<|user|>{prompt}<|assistant|>"""
        
        return enhanced_prompt

    def get_model_performance_stats(self, reasoning_type: Optional[ReasoningType] = None) -> Dict[str, Any]:
        """Get performance statistics for models"""
        try:
            if reasoning_type:
                # Stats for specific reasoning type
                pattern = f"model_metrics:{reasoning_type.value}:*"
            else:
                # Stats for all reasoning types
                pattern = "model_metrics:*"
            
            keys = self.redis_client.keys(pattern)
            stats = {}
            
            for key in keys:
                metrics_raw = self.redis_client.lrange(key, 0, -1)
                metrics = [json.loads(m) for m in metrics_raw]
                
                if metrics:
                    total_requests = len(metrics)
                    successful_requests = sum(1 for m in metrics if m.get("success", False))
                    avg_response_time = sum(m.get("response_time", 0) for m in metrics) / total_requests
                    
                    model_info = key.split(":")
                    reasoning_type_name = model_info[1] if len(model_info) > 1 else "unknown"
                    model_name = model_info[2] if len(model_info) > 2 else "unknown"
                    
                    stats[key] = {
                        "reasoning_type": reasoning_type_name,
                        "model": model_name,
                        "total_requests": total_requests,
                        "success_rate": successful_requests / total_requests,
                        "avg_response_time": avg_response_time,
                        "recent_requests": metrics[:5]  # Last 5 requests
                    }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get performance stats: {e}")
            return {}

    def clear_cache(self, reasoning_type: Optional[ReasoningType] = None):
        """Clear model cache"""
        try:
            if reasoning_type:
                pattern = f"model_cache:{reasoning_type.value}:*"
            else:
                pattern = "model_cache:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                self.logger.info(f"Cleared {len(keys)} cache entries")
            
        except Exception as e:
            self.logger.error(f"Cache clearing failed: {e}")
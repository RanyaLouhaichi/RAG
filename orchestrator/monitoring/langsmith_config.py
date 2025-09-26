# orchestrator/monitoring/langsmith_config.py
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import functools
import time
import json
import uuid
from contextlib import contextmanager

load_dotenv(override=True)
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "JURIX-Thesis-Demo")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

from langsmith import Client, RunTree
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run

logger = logging.getLogger("LangSmithMonitor")

class LangSmithMonitor:
    """Comprehensive LangSmith monitoring for JURIX multi-agent system"""
    
    def __init__(self):
        """Initialize LangSmith client and monitoring"""
        try:
            self.client = Client()
            self.enabled = True
            self.project_name = os.getenv("LANGCHAIN_PROJECT", "JURIX-Thesis-Demo")
            self.metrics = {
                "total_runs": 0,
                "agent_runs": {},
                "workflow_runs": {},
                "errors": [],
                "latencies": [],
                "collaboration_traces": []
            }
            self.active_runs = {}
            self._test_connection()
            
            logger.info(f"✅ LangSmith monitoring initialized for project: {self.project_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LangSmith: {e}")
            self.enabled = False
            self.client = None
    
    def _test_connection(self):
        """Test LangSmith connection"""
        try:
            projects = list(self.client.list_projects(limit=1))
            logger.info(f"✅ LangSmith connection verified. Found {len(projects)} project(s)")
            return True
        except Exception as e:
            logger.error(f"❌ LangSmith connection test failed: {e}")
            return False
    
    @contextmanager
    def trace_workflow(self, workflow_name: str, metadata: Dict[str, Any] = None):
        """Context manager for tracing entire workflows"""
        if not self.enabled:
            yield None
            return
        
        run_id = str(uuid.uuid4())
        run_tree = RunTree(
            name=f"Workflow: {workflow_name}",
            run_type="chain",
            inputs={"workflow": workflow_name, "metadata": metadata or {}},
            project_name=self.project_name,
            tags=["workflow", workflow_name, "thesis_demo"],
            metadata={
                "workflow_type": workflow_name,
                "timestamp": datetime.now().isoformat(),
                "system": "JURIX",
                **(metadata or {})
            }
        )
        
        self.active_runs[run_id] = run_tree
        self.metrics["total_runs"] += 1
        
        if workflow_name not in self.metrics["workflow_runs"]:
            self.metrics["workflow_runs"][workflow_name] = 0
        self.metrics["workflow_runs"][workflow_name] += 1
        
        start_time = time.time()
        
        try:
            yield run_tree
            run_tree.end(outputs={
                "status": "success",
                "duration": time.time() - start_time
            })
            run_tree.post()
            
        except Exception as e:
            run_tree.end(
                outputs={"status": "error", "error": str(e)},
                error=str(e)
            )
            run_tree.post()
            
            self.metrics["errors"].append({
                "workflow": workflow_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise
        
        finally:
            latency = time.time() - start_time
            self.metrics["latencies"].append({
                "workflow": workflow_name,
                "latency": latency,
                "timestamp": datetime.now().isoformat()
            })
            if run_id in self.active_runs:
                del self.active_runs[run_id]
    
    def trace_agent(self, agent_name: str, parent_run: Optional[RunTree] = None):
        """Decorator for tracing individual agent calls"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                run_tree = RunTree(
                    name=f"Agent: {agent_name}",
                    run_type="llm" if "llm" in agent_name.lower() else "tool",
                    inputs={
                        "agent": agent_name,
                        "args": str(args)[:500], 
                        "kwargs": str(kwargs)[:500]
                    },
                    parent_run=parent_run,
                    project_name=self.project_name,
                    tags=["agent", agent_name],
                    metadata={
                        "agent_type": agent_name,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                if agent_name not in self.metrics["agent_runs"]:
                    self.metrics["agent_runs"][agent_name] = 0
                self.metrics["agent_runs"][agent_name] += 1
                
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    run_tree.end(outputs={
                        "result": str(result)[:1000], 
                        "duration": time.time() - start_time,
                        "status": "success"
                    })
                    run_tree.post()
                    
                    return result
                    
                except Exception as e:
                    run_tree.end(
                        outputs={"status": "error", "error": str(e)},
                        error=str(e)
                    )
                    run_tree.post()
                    raise
            
            return wrapper
        return decorator
    
    def trace_collaboration(self, requesting_agent: str, collaborating_agent: str, 
                          collaboration_type: str, parent_run: Optional[RunTree] = None):
        """Trace agent collaboration events"""
        if not self.enabled:
            return None
        
        collab_run = RunTree(
            name=f"Collaboration: {requesting_agent} → {collaborating_agent}",
            run_type="chain",
            inputs={
                "requesting_agent": requesting_agent,
                "collaborating_agent": collaborating_agent,
                "collaboration_type": collaboration_type
            },
            parent_run=parent_run,
            project_name=self.project_name,
            tags=["collaboration", requesting_agent, collaborating_agent],
            metadata={
                "collaboration_type": collaboration_type,
                "timestamp": datetime.now().isoformat()
            }
        )
        self.metrics["collaboration_traces"].append({
            "from": requesting_agent,
            "to": collaborating_agent,
            "type": collaboration_type,
            "timestamp": datetime.now().isoformat()
        })
        
        return collab_run
    
    def log_model_usage(self, agent_name: str, model_name: str, 
                       prompt: str, response: str, 
                       latency: float, parent_run: Optional[RunTree] = None):
        """Log LLM model usage for thesis metrics"""
        if not self.enabled:
            return
        
        model_run = RunTree(
            name=f"Model: {model_name}",
            run_type="llm",
            inputs={
                "agent": agent_name,
                "model": model_name,
                "prompt": prompt[:1000]  
            },
            outputs={
                "response": response[:1000],  
                "latency": latency
            },
            parent_run=parent_run,
            project_name=self.project_name,
            tags=["model", model_name, agent_name],
            metadata={
                "model_name": model_name,
                "agent_name": agent_name,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "latency_ms": latency * 1000
            }
        )
        
        model_run.post()
    
    def create_test_dataset(self, name: str, examples: List[Dict[str, Any]]):
        """Create a test dataset for evaluation"""
        if not self.enabled:
            return None
        
        try:
            dataset = self.client.create_dataset(
                dataset_name=name,
                description=f"Test dataset for {name} - JURIX Thesis Demo"
            )
            for example in examples:
                self.client.create_example(
                    inputs=example.get("inputs", {}),
                    outputs=example.get("outputs", {}),
                    dataset_id=dataset.id
                )
            
            logger.info(f"✅ Created test dataset: {name} with {len(examples)} examples")
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Failed to create dataset: {e}")
            return None
    
    def run_evaluation(self, dataset_name: str, evaluator_func):
        """Run evaluation on a dataset"""
        if not self.enabled:
            return None
        
        try:
            results = evaluate(
                evaluator_func,
                data=dataset_name,
                experiment_prefix="JURIX_Evaluation",
                client=self.client
            )
            
            logger.info(f"✅ Evaluation completed for dataset: {dataset_name}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary for thesis presentation"""
        if not self.enabled:
            return {"status": "LangSmith not enabled"}
        total_agent_runs = sum(self.metrics["agent_runs"].values())
        total_workflow_runs = sum(self.metrics["workflow_runs"].values())
        avg_latency = 0
        if self.metrics["latencies"]:
            avg_latency = sum(l["latency"] for l in self.metrics["latencies"]) / len(self.metrics["latencies"])
        most_active_agents = sorted(
            self.metrics["agent_runs"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        collaboration_network = {}
        for collab in self.metrics["collaboration_traces"]:
            key = f"{collab['from']} → {collab['to']}"
            if key not in collaboration_network:
                collaboration_network[key] = 0
            collaboration_network[key] += 1
        
        return {
            "status": "active",
            "project": self.project_name,
            "total_runs": self.metrics["total_runs"],
            "agent_statistics": {
                "total_agent_runs": total_agent_runs,
                "unique_agents": len(self.metrics["agent_runs"]),
                "agent_breakdown": self.metrics["agent_runs"],
                "most_active_agents": most_active_agents
            },
            "workflow_statistics": {
                "total_workflow_runs": total_workflow_runs,
                "workflow_breakdown": self.metrics["workflow_runs"]
            },
            "performance_metrics": {
                "average_latency_seconds": round(avg_latency, 3),
                "total_errors": len(self.metrics["errors"]),
                "error_rate": len(self.metrics["errors"]) / max(self.metrics["total_runs"], 1)
            },
            "collaboration_insights": {
                "total_collaborations": len(self.metrics["collaboration_traces"]),
                "collaboration_network": collaboration_network,
                "unique_collaboration_pairs": len(collaboration_network)
            },
            "thesis_metrics": {
                "system_complexity": len(self.metrics["agent_runs"]),  
                "system_reliability": 1 - (len(self.metrics["errors"]) / max(self.metrics["total_runs"], 1)),
                "collaboration_density": len(self.metrics["collaboration_traces"]) / max(total_agent_runs, 1),
                "avg_response_time": round(avg_latency, 3)
            }
        }
    
    def export_metrics_for_thesis(self, filepath: str = "thesis_metrics.json"):
        """Export all metrics to a JSON file for thesis presentation"""
        metrics = self.get_metrics_summary()
        metrics["export_timestamp"] = datetime.now().isoformat()
        metrics["raw_data"] = {
            "latencies": self.metrics["latencies"][-100:], 
            "errors": self.metrics["errors"][-50:],  
            "collaborations": self.metrics["collaboration_traces"][-100:]  
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✅ Exported thesis metrics to {filepath}")
        return filepath

langsmith_monitor = LangSmithMonitor()
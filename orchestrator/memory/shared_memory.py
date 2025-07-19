import redis
import json
import uuid
from typing import Any, List, Dict, Optional
from datetime import datetime, timedelta
import logging

class JurixSharedMemory:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.storage: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("JurixSharedMemory")
        
        # Test connection
        try:
            self.redis_client.ping()
            self.logger.info("✅ Connected to Redis successfully!")
            print("✅ Connected to Redis successfully!")
        except redis.ConnectionError:
            self.logger.error("❌ Redis connection failed. Make sure Redis server is running.")
            print("❌ Redis connection failed. Make sure Redis server is running.")
            raise

    def get_conversation(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history from Redis"""
        if not session_id:
            return []
        
        key = f"conversation:{session_id}"
        conversation_data = self.redis_client.get(key)
        if conversation_data:
            try:
                return json.loads(conversation_data)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to decode conversation data for {session_id}")
                return []
        return []

    def add_interaction(self, session_id: str, role: str, content: str) -> None:
        """Add interaction to conversation history in Redis"""
        if not session_id:
            return
            
        key = f"conversation:{session_id}"
        conversation = self.get_conversation(session_id)
        conversation.append({
            "role": role, 
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 50 interactions to prevent memory bloat
        if len(conversation) > 50:
            conversation = conversation[-50:]
            
        self.redis_client.set(key, json.dumps(conversation))
        # Set expiration (24 hours)
        self.redis_client.expire(key, 86400)
    
    def store(self, key: str, value: Dict[str, Any]) -> None:
        """Store a value under the given key in Redis"""
        try:
            serialized_value = json.dumps(value, default=str)  # Handle datetime objects
            self.redis_client.set(f"storage:{key}", serialized_value)
            self.logger.info(f"Stored data with key: storage:{key}")
        except Exception as e:
            self.logger.error(f"Failed to store data with key {key}: {e}")
        
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a value from Redis storage"""
        try:
            data = self.redis_client.get(f"storage:{key}")
            if data:
                return json.loads(data)
        except Exception as e:
            self.logger.error(f"Failed to get data with key {key}: {e}")
        return None

    def store_tickets(self, project_id: str, tickets: List[Dict[str, Any]]) -> None:
        """Store tickets data for a project in Redis"""
        key = f"tickets:{project_id}"
        try:
            serialized_tickets = json.dumps(tickets, default=str)
            self.redis_client.set(key, serialized_tickets)
            # Store metadata
            metadata_key = f"tickets_meta:{project_id}"
            metadata = {
                "last_updated": datetime.now().isoformat(),
                "ticket_count": len(tickets),
                "has_changes": True
            }
            self.redis_client.set(metadata_key, json.dumps(metadata))
            self.logger.info(f"Stored {len(tickets)} tickets for project {project_id}")
        except Exception as e:
            self.logger.error(f"Failed to store tickets for project {project_id}: {e}")

    def get_tickets(self, project_id: str) -> List[Dict[str, Any]]:
        """Get tickets data for a project from Redis"""
        key = f"tickets:{project_id}"
        try:
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            self.logger.error(f"Failed to get tickets for project {project_id}: {e}")
        return []

    def has_ticket_updates(self, project_id: str) -> bool:
        """Check if there are updates for a project"""
        metadata_key = f"tickets_meta:{project_id}"
        try:
            data = self.redis_client.get(metadata_key)
            if data:
                metadata = json.loads(data)
                return metadata.get("has_changes", False)
        except Exception as e:
            self.logger.error(f"Failed to check updates for project {project_id}: {e}")
        return False

    def mark_updates_processed(self, project_id: str) -> None:
        """Mark updates as processed for a project"""
        metadata_key = f"tickets_meta:{project_id}"
        try:
            data = self.redis_client.get(metadata_key)
            if data:
                metadata = json.loads(data)
                metadata["has_changes"] = False
                metadata["last_processed"] = datetime.now().isoformat()
                self.redis_client.set(metadata_key, json.dumps(metadata))
        except Exception as e:
            self.logger.error(f"Failed to mark updates processed for project {project_id}: {e}")

    def store_agent_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        """Store agent mental state in Redis"""
        key = f"agent_state:{agent_id}"
        try:
            serialized_state = json.dumps(state, default=str)
            self.redis_client.set(key, serialized_state)
            # Set expiration (1 hour for agent states)
            self.redis_client.expire(key, 3600)
        except Exception as e:
            self.logger.error(f"Failed to store agent state for {agent_id}: {e}")

    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent mental state from Redis"""
        key = f"agent_state:{agent_id}"
        try:
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            self.logger.error(f"Failed to get agent state for {agent_id}: {e}")
        return None

    def publish_agent_event(self, channel: str, event: Dict[str, Any]) -> None:
        """Publish agent event for real-time coordination"""
        try:
            event_data = json.dumps(event, default=str)
            self.redis_client.publish(channel, event_data)
            self.logger.info(f"Published event to channel {channel}")
        except Exception as e:
            self.logger.error(f"Failed to publish event to {channel}: {e}")

    def subscribe_to_events(self, channels: List[str]):
        """Subscribe to agent events"""
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(*channels)
            return pubsub
        except Exception as e:
            self.logger.error(f"Failed to subscribe to channels {channels}: {e}")
            return None

    def clear_expired_data(self) -> None:
        """Clean up expired data (can be called periodically)"""
        try:
            # Get all conversation keys older than 24 hours
            conversation_keys = self.redis_client.keys("conversation:*")
            for key in conversation_keys:
                ttl = self.redis_client.ttl(key)
                if ttl == -1:  # No expiration set
                    self.redis_client.expire(key, 86400)  # Set 24 hour expiration
            
            self.logger.info("Cleaned up expired data")
        except Exception as e:
            self.logger.error(f"Failed to clean up expired data: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get Redis memory statistics"""
        try:
            info = self.redis_client.info("memory")
            stats = {
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "total_keys": self.redis_client.dbsize(),
                "connected_clients": self.redis_client.info("clients").get("connected_clients", 0)
            }
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {}

    # Backward compatibility property for legacy code
    @property
    def memory(self):
        """Backward compatibility: provide dict-like access to Redis"""
        return RedisCompatDict(self.redis_client)

class RedisCompatDict:
    """Provides dict-like interface for backward compatibility"""
    def __init__(self, redis_client):
        self.redis_client = redis_client
    
    def __getitem__(self, key):
        data = self.redis_client.get(f"compat:{key}")
        if data:
            try:
                return json.loads(data)
            except:
                return data
        raise KeyError(key)
    
    def __setitem__(self, key, value):
        try:
            serialized = json.dumps(value, default=str)
            self.redis_client.set(f"compat:{key}", serialized)
        except:
            self.redis_client.set(f"compat:{key}", str(value))
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
# config/jurix_config.py
JURIX_CONFIG = {
    "aria": {
        "personality": {
            "enabled": True,
            "emoji_usage": True,
            "proactive_suggestions": True,
            "conversation_style": "friendly_professional"
        },
        "capabilities": {
            "real_time_insights": True,
            "plugin_integration": True,
            "auto_documentation": True,
            "predictive_analytics": True
        }
    },
    "plugins": {
        "jira": {
            "ai_insights_tab": True,
            "ticket_ai_panel": True,
            "auto_doc_on_resolve": True,
            "real_time_updates": True,
            "update_interval": 30  # seconds
        },
        "confluence": {
            "ai_dashboard": True,
            "auto_article_generation": True,
            "knowledge_gap_detection": True,
            "content_suggestions": True
        }
    },
    "real_time": {
        "websocket_enabled": True,
        "redis_pubsub": True,
        "dashboard_refresh": 30,
        "notification_delay": 0.5
    }
}
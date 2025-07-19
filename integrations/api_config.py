import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

class APIConfig:
    """Configuration for Jira and Confluence APIs"""
    
    # Jira Configuration
    JIRA_URL = os.getenv("JIRA_URL", "http://localhost:8080")
    JIRA_USERNAME = os.getenv("JIRA_USERNAME", "admin")
    JIRA_PASSWORD = os.getenv("JIRA_PASSWORD", "")
    
    # Confluence Configuration
    CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", "http://localhost:8090")
    CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME", "admin")
    CONFLUENCE_PASSWORD = os.getenv("CONFLUENCE_PASSWORD", "")
    
    # API Settings
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
    API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))
    API_PAGE_SIZE = int(os.getenv("API_PAGE_SIZE", "50"))
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    
    # SSL Verification (disable for local Docker)
    VERIFY_SSL = False if ENVIRONMENT == "development" else True
    
    @classmethod
    def get_jira_auth(cls) -> tuple:
        """Get Jira authentication tuple"""
        return (cls.JIRA_USERNAME, cls.JIRA_PASSWORD)
    
    @classmethod
    def get_confluence_auth(cls) -> tuple:
        """Get Confluence authentication tuple"""
        return (cls.CONFLUENCE_USERNAME, cls.CONFLUENCE_PASSWORD)
    
    @classmethod
    def log_config(cls):
        """Log configuration (hiding sensitive data)"""
        print(f"🔧 API Configuration:")
        print(f"  - Environment: {cls.ENVIRONMENT}")
        print(f"  - Jira URL: {cls.JIRA_URL}")
        print(f"  - Jira User: {cls.JIRA_USERNAME}")
        print(f"  - Confluence URL: {cls.CONFLUENCE_URL}")
        print(f"  - API Timeout: {cls.API_TIMEOUT}s")
        print(f"  - Debug Mode: {cls.DEBUG}")

    STATUS_MAPPING = {
        "DONE_STATUSES": ["Done", "Closed", "Resolved", "Complete", "Completed", "Fixed", "Deployed"],
        "IN_PROGRESS_STATUSES": ["In Progress", "In Development", "In Review", "Testing", "In Test", "Implementing"],
        "TODO_STATUSES": ["To Do", "Open", "New", "Backlog", "Ready", "Created", "Reopened"],
        "BLOCKED_STATUSES": ["Blocked", "On Hold", "Waiting", "Impediment"]
    }
    
    @classmethod
    def normalize_status(cls, status: str) -> str:
        """Normalize Jira status to standard categories"""
        status_upper = status.upper()
        
        for done_status in cls.STATUS_MAPPING["DONE_STATUSES"]:
            if done_status.upper() in status_upper:
                return "Done"
        
        for progress_status in cls.STATUS_MAPPING["IN_PROGRESS_STATUSES"]:
            if progress_status.upper() in status_upper:
                return "In Progress"
        
        for todo_status in cls.STATUS_MAPPING["TODO_STATUSES"]:
            if todo_status.upper() in status_upper:
                return "To Do"
        
        for blocked_status in cls.STATUS_MAPPING["BLOCKED_STATUSES"]:
            if blocked_status.upper() in status_upper:
                return "Blocked"
        
        return status
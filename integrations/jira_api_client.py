import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry # type: ignore
from integrations.api_config import APIConfig

class JiraAPIClient:
    """Client for interacting with Jira REST API"""
    
    def __init__(self):
        self.base_url = APIConfig.JIRA_URL
        self.auth = APIConfig.get_jira_auth()
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.logger = logging.getLogger("JiraAPIClient")
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=APIConfig.API_MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        if APIConfig.DEBUG:
            self.logger.info(f"JiraAPIClient initialized - URL: {self.base_url}, User: {self.auth[0]}")
    
    def test_connection(self) -> bool:
        """Test connection to Jira"""
        try:
            response = self.session.get(
                f"{self.base_url}/rest/api/2/myself",
                auth=self.auth,
                headers=self.headers,
                verify=APIConfig.VERIFY_SSL,
                timeout=APIConfig.API_TIMEOUT
            )
            if response.status_code == 200:
                user_data = response.json()
                self.logger.info(f"✅ Connected to Jira as: {user_data.get('displayName', 'Unknown')}")
                return True
            else:
                self.logger.error(f"❌ Jira connection failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Jira connection error: {e}")
            return False
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make API request with error handling"""
        url = f"{self.base_url}/rest/api/2/{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                auth=self.auth,
                headers=self.headers,
                verify=APIConfig.VERIFY_SSL,
                timeout=APIConfig.API_TIMEOUT,
                **kwargs
            )
            
            if APIConfig.DEBUG:
                self.logger.debug(f"{method} {url} - Status: {response.status_code}")
            
            response.raise_for_status()
            return response.json() if response.text else {}
            
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP Error: {e}")
            self.logger.error(f"Response: {e.response.text if e.response else 'No response'}")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    def get_projects(self) -> List[Dict[str, Any]]:
        """Get all projects accessible to the user"""
        self.logger.info("Fetching all projects from Jira")
        try:
            projects = self._make_request("GET", "project")
            self.logger.info(f"Found {len(projects)} projects")
            return projects
        except Exception as e:
            self.logger.error(f"Failed to get projects: {e}")
            return []
    
    def get_project(self, project_key: str) -> Dict[str, Any]:
        """Get specific project details"""
        self.logger.info(f"Fetching project: {project_key}")
        return self._make_request("GET", f"project/{project_key}")
    
    def search_issues(self, jql: str, start_at: int = 0, max_results: int = None) -> Dict[str, Any]:
        """Search issues using JQL"""
        if max_results is None:
            max_results = APIConfig.API_PAGE_SIZE
        
        params = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": max_results,
            "expand": "changelog,renderedFields,transitions",  # Add transitions
            "fields": "*all,-comment"  # Get all fields except comments (can be large)
        }
        
        self.logger.info(f"Searching issues with JQL: {jql}")
        result = self._make_request("GET", "search", params=params)
        
        # DEBUG: Log what we got
        if result.get("issues"):
            first_issue = result["issues"][0]
            self.logger.info(f"[DEBUG] Sample issue fields: {list(first_issue.get('fields', {}).keys())[:20]}")
        
        return result
    
    def get_issue(self, issue_key: str) -> Dict[str, Any]:
        """Get specific issue details with full expansion"""
        self.logger.info(f"Fetching issue: {issue_key}")
        params = {
            "expand": "changelog,renderedFields,names,schema,transitions,operations,editmeta"
        }
        return self._make_request("GET", f"issue/{issue_key}", params=params)
    
    def get_all_issues_for_project(self, project_key: str, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """Get all issues for a project with optional date filtering"""
        # Build JQL without date filtering if dates not provided
        jql = f"project = {project_key}"
        
        # Only add date filtering if explicitly provided
        if start_date and end_date:
            jql += f" AND updated >= '{start_date}' AND updated <= '{end_date}'"
        elif start_date:
            jql += f" AND updated >= '{start_date}'"
        elif end_date:
            jql += f" AND updated <= '{end_date}'"
        
        jql += " ORDER BY updated DESC"
        
        all_issues = []
        start_at = 0
        total = None
        
        while True:
            try:
                result = self.search_issues(jql, start_at=start_at)
                issues = result.get("issues", [])
                all_issues.extend(issues)
                
                if total is None:
                    total = result.get("total", 0)
                    self.logger.info(f"Total issues matching query: {total}")
                
                if len(all_issues) >= total or len(issues) < self.API_PAGE_SIZE:
                    break
                
                start_at += len(issues)
                self.logger.info(f"Fetched {len(all_issues)}/{total} issues...")
                
            except Exception as e:
                self.logger.error(f"Error fetching issues at offset {start_at}: {e}")
                break
        
        self.logger.info(f"✅ Retrieved {len(all_issues)} issues for project {project_key}")
        return all_issues
    
    def create_issue(self, project_key: str, issue_type: str, summary: str, description: str = None) -> Dict[str, Any]:
        """Create a new issue"""
        issue_data = {
            "fields": {
                "project": {"key": project_key},
                "summary": summary,
                "issuetype": {"name": issue_type}
            }
        }
        
        if description:
            issue_data["fields"]["description"] = description
        
        self.logger.info(f"Creating issue in project {project_key}: {summary}")
        return self._make_request("POST", "issue", json=issue_data)
    
    def add_comment(self, issue_key: str, comment: str) -> Dict[str, Any]:
        """Add a comment to an issue"""
        comment_data = {"body": comment}
        self.logger.info(f"Adding comment to {issue_key}")
        return self._make_request("POST", f"issue/{issue_key}/comment", json=comment_data)
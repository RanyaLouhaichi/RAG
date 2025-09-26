import asyncio
from typing import Dict, Any, List
import json
import os
import logging
import redis # type: ignore
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent, AgentCapability
import time
import threading
import concurrent.futures
from mcp_integration.client.jira_confluence_client import JiraConfluenceMCPClient 

try:
    from integrations.jira_api_client import JiraAPIClient
    JIRA_API_AVAILABLE = True
except ImportError:
    JIRA_API_AVAILABLE = False

class JiraDataAgent(BaseAgent):
    def __init__(self, mock_data_path: str = None, redis_client: redis.Redis = None, use_real_api: bool = True):
        if redis_client is None:
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        super().__init__(name="jira_data_agent", redis_client=redis_client)
        self.mcp_client = None
        self.use_mcp = False
        self.available_projects = []
        try:
            from mcp_integration.client.jira_confluence_client import JiraConfluenceMCPClient
            self.mcp_client = JiraConfluenceMCPClient(redis_client)
            def connect_mcp():
                import asyncio
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(self.mcp_client.connect_to_jira_server())
                    self.available_projects = new_loop.run_until_complete(self.mcp_client.get_available_projects())
                    return True
                except Exception as e:
                    self.log(f"❌ MCP connection failed: {e}")
                    return False
                finally:
                    new_loop.close()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(connect_mcp)
                try:
                    connected = future.result(timeout=5)
                    if connected and self.available_projects:
                        self.use_mcp = True
                        if self.use_mcp:
                            self._load_via_mcp = self._load_via_mcp
                        self.log("✅ Connected to Jira via MCP!")
                        self.log(f"📋 Available projects: {self.available_projects}")
                        self.use_real_api = False
                        self.log("🚀 MCP connection successful - disabling API fallback")
                    else:
                        self.use_mcp = False
                        self.mcp_client = None
                except concurrent.futures.TimeoutError:
                    self.log("❌ MCP connection timeout")
                    self.use_mcp = False
                    self.mcp_client = None
        except ImportError as e:
            self.log(f"❌ MCP client not available: {e}")
            self.use_mcp = False
            self.mcp_client = None
        except Exception as e:
            self.log(f"❌ Error initializing MCP: {e}")
            self.use_mcp = False
            self.mcp_client = None
        self.mock_data_path = mock_data_path or os.path.join("data", "mock_jira_data.json")
        self.last_modified = None 
        if not self.available_projects:
            self.available_projects = []
        self.use_real_api = use_real_api and JIRA_API_AVAILABLE and not self.use_mcp
        if self.use_real_api:
            try:
                self.jira_client = JiraAPIClient()
                if self.jira_client.test_connection():
                    self.log("✅ Connected to Jira API!")
                    projects = self.jira_client.get_projects()
                    self.available_projects = [p['key'] for p in projects]
                    self.log(f"📋 Available projects: {self.available_projects}")
                else:
                    self.log("❌ Jira connection failed, using mock data")
                    self.use_real_api = False
            except Exception as e:
                self.log(f"❌ Failed to connect to Jira: {e}")
                self.use_real_api = False
        if not self.use_real_api and not self.use_mcp:
            self.available_projects = ["PROJ123"]
        try:
            self.redis_client.ping()
            self.log("✅ Connected to Redis successfully for collaborative data operations!")
        except redis.ConnectionError:
            self.log("❌ Redis connection failed - limited collaborative capabilities")
            raise
        self.mental_state.capabilities = [
            AgentCapability.RETRIEVE_DATA,
            AgentCapability.RANK_CONTENT,
            AgentCapability.COORDINATE_AGENTS
        ]
        self.mental_state.obligations.extend([
            "load_jira_data",
            "filter_tickets",
            "monitor_file",
            "cache_efficiently",
            "assess_data_completeness",
            "trigger_analysis_collaboration",
            "provide_data_insights"
        ])
        self.log(f"JiraDataAgent initialized - Using {'MCP' if self.use_mcp else 'API' if self.use_real_api else 'mock data'}")
        if not self.use_real_api and not self.use_mcp:
            self.monitoring_thread = threading.Thread(target=self._monitor_file_loop, daemon=True)
            self.monitoring_thread.start()
    async def _ensure_mcp_connected(self):
        if self.mcp_connection_attempted:
            return self.use_mcp
        self.mcp_connection_attempted = True
        try:
            if not self.mcp_client:
                self.mcp_client = JiraConfluenceMCPClient(self.redis_client)
            await self.mcp_client.connect_to_jira_server()
            self.available_projects = await self.mcp_client.get_available_projects()
            self.use_mcp = True
            self.log("✅ Connected to Jira via MCP!")
            self.log(f"📋 Available projects: {self.available_projects}")
            return True
        except Exception as e:
            self.log(f"❌ MCP connection failed: {e}")
            self.use_mcp = False
            return False
    def _load_jira_data(self) -> List[Dict[str, Any]]:
        try:
            project_id = self.mental_state.get_belief("loading_project") or self.mental_state.get_belief("current_project")
            time_range = self.mental_state.get_belief("loading_time_range") or self.mental_state.get_belief("current_time_range")
            if self.use_mcp and self.mcp_client is not None:
                try:
                    self.log(f"Loading data from Jira MCP for project {project_id}...")
                    import asyncio
                    async def fetch_tickets():
                        if not hasattr(self.mcp_client, 'jira_session') or self.mcp_client.jira_session is None:
                            await self.mcp_client.connect_to_jira_server()
                        return await self.mcp_client.get_project_tickets(
                            project_id,
                            time_range.get('start') if time_range else None,
                            time_range.get('end') if time_range else None
                        )
                    try:
                        loop = asyncio.get_running_loop()
                        def run_in_new_loop():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                return new_loop.run_until_complete(fetch_tickets())
                            finally:
                                new_loop.close()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_in_new_loop)
                            tickets = future.result(timeout=30)
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            tickets = loop.run_until_complete(fetch_tickets())
                        finally:
                            loop.close()
                    self.log(f"✅ Loaded {len(tickets)} tickets via MCP")
                    cache_key = f"jira_mcp_data:{project_id}:{time_range.get('start', 'all') if time_range else 'all'}"
                    self.redis_client.set(cache_key, json.dumps(tickets))
                    self.redis_client.expire(cache_key, 300)
                    self._assess_data_collaboration_needs(tickets, "mcp_data")
                    return tickets
                except Exception as mcp_error:
                    self.log(f"[WARNING] MCP failed, falling back: {str(mcp_error)}")
            if self.use_real_api:
                return self._load_from_api(project_id, time_range)
            return self._load_from_mock_file()
        except Exception as e:
            self.log(f"[ERROR] Failed to load data: {str(e)}")
            return []
    def _load_jira_data(self) -> List[Dict[str, Any]]:
        try:
            project_id = self.mental_state.get_belief("loading_project") or self.mental_state.get_belief("current_project")
            time_range = self.mental_state.get_belief("loading_time_range") or self.mental_state.get_belief("current_time_range")
            if self.use_mcp and hasattr(self, '_load_via_mcp'):
                try:
                    self.log(f"Loading data from Jira MCP for project {project_id}...")
                    tickets = self._load_via_mcp(project_id, time_range)
                    if tickets is not None:
                        self.log(f"✅ Loaded {len(tickets)} tickets via MCP")
                        cache_key = f"jira_mcp_data:{project_id}:{time_range.get('start', 'all') if time_range else 'all'}"
                        self.redis_client.set(cache_key, json.dumps(tickets))
                        self.redis_client.expire(cache_key, 300)
                        self._assess_data_collaboration_needs(tickets, "mcp_data")
                        return tickets
                except Exception as mcp_error:
                    self.log(f"[WARNING] MCP failed, falling back: {str(mcp_error)}")
            if self.use_real_api:
                return self._load_from_api(project_id, time_range)
            return self._load_from_mock_file()
        except Exception as e:
            self.log(f"[ERROR] Failed to load data: {str(e)}")
            return []
    def _load_from_api(self, project_id: str, time_range: Dict[str, str] = None) -> List[Dict[str, Any]]:
        try:
            if not project_id or project_id == "None":
                self.log(f"Invalid project_id: {project_id}")
                if self.available_projects:
                    project_id = self.available_projects[0]
                    self.log(f"Using fallback project: {project_id}")
                else:
                    self.log("No available projects to fall back to")
                    return []
            self.log(f"Loading tickets from API for project: {project_id}")
            if not self.jira_client:
                self.log("[ERROR] No Jira client available")
                return []
            if not self.jira_client.test_connection():
                self.log("[ERROR] Jira connection test failed")
                return []
            all_issues = self.jira_client.get_all_issues_for_project(
                project_id,
                None,
                None
            )
            self.log(f"✅ Loaded {len(all_issues)} tickets via API (all tickets, no date filter)")
            cache_key = f"jira_api_data:{project_id}:all"
            self.redis_client.set(cache_key, json.dumps(all_issues))
            self.redis_client.expire(cache_key, 300)
            self._assess_data_collaboration_needs(all_issues, "api_data")
            return all_issues
        except Exception as e:
            self.log(f"[ERROR] API loading failed: {str(e)}")
            return self._load_from_mock_file()
    def _load_via_mcp(self, project_id: str, time_range: Dict[str, str] = None) -> List[Dict[str, Any]]:
        def fetch_in_thread():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                from mcp_integration.client.jira_confluence_client import JiraConfluenceMCPClient
                thread_mcp_client = JiraConfluenceMCPClient(self.redis_client)
                async def fetch():
                    await thread_mcp_client.connect_to_jira_server()
                    tickets = await thread_mcp_client.get_project_tickets(
                        project_id,
                        time_range.get('start') if time_range else None,
                        time_range.get('end') if time_range else None
                    )
                    return tickets
                tickets = loop.run_until_complete(fetch())
                return tickets
            except Exception as e:
                self.log(f"[ERROR] MCP thread error: {e}")
                raise
            finally:
                loop.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor: 
            future = executor.submit(fetch_in_thread)
            return future.result(timeout=30)
    def _load_from_mock_file(self) -> List[Dict[str, Any]]:
        try:
            cache_key = f"jira_raw_data:{os.path.basename(self.mock_data_path)}"
            current_modified = os.path.getmtime(self.mock_data_path)
            cached_modified = self.redis_client.get(f"{cache_key}:modified")
            if cached_modified and float(cached_modified) == current_modified:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    self.log("Using cached mock data - assessing quality for collaborative needs")
                    data = json.loads(cached_data)
                    self._assess_data_collaboration_needs(data, "cached_data")
                    return data
            self.log(f"Loading fresh mock data from {self.mock_data_path}")
            with open(self.mock_data_path, 'r') as file:
                data = json.load(file)
                issues = data.get("issues", [])
                cache_metadata = {
                    "loaded_at": datetime.now().isoformat(),
                    "issue_count": len(issues),
                    "data_quality_assessed": True
                }
                self.redis_client.set(cache_key, json.dumps(issues))
                self.redis_client.set(f"{cache_key}:modified", str(current_modified))
                self.redis_client.set(f"{cache_key}:metadata", json.dumps(cache_metadata))
                self.redis_client.expire(cache_key, 3600)
                self.redis_client.expire(f"{cache_key}:modified", 3600)
                self.redis_client.expire(f"{cache_key}:metadata", 3600)
                self.last_modified = current_modified
                self.log(f"Successfully loaded and cached {len(issues)} mock tickets")
                self.mental_state.add_belief("data_freshness", "fresh", 0.9, "file_load")
                self._assess_data_collaboration_needs(issues, "fresh_data")
                return issues
        except Exception as e:
            self.log(f"[ERROR] Failed to load mock data: {str(e)}")
            self.mental_state.add_belief("data_availability", "failed", 0.8, "error")
            return []
    def _filter_tickets(self, tickets: List[Dict[str, Any]], project_id: str, time_range: Dict[str, str]) -> List[Dict[str, Any]]:
        if not tickets:
            self.log("No tickets to filter")
            return []
        if tickets is None:
            self.log("[WARNING] Tickets is None, returning empty list")
            return []
        self.log(f"[FILTER] Filtering {len(tickets)} tickets for project {project_id}")
        self.log(f"[FILTER] No date filtering applied - returning all tickets for project")
        status_debug = {}
        project_debug = {}
        for ticket in tickets:
            status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
            status_debug[status] = status_debug.get(status, 0) + 1
            ticket_project = ticket.get("fields", {}).get("project", {}).get("key", "Unknown")
            project_debug[ticket_project] = project_debug.get(ticket_project, 0) + 1
        self.log(f"[DEBUG] Raw ticket status distribution: {status_debug}")
        self.log(f"[DEBUG] Raw ticket project distribution: {project_debug}")
        self.log(f"[DEBUG] Total raw tickets: {len(tickets)}")
        filtered_tickets = []
        if self.use_mcp or self.use_real_api:
            self.log(f"API/MCP data - filtering for project {project_id} only")
            for ticket in tickets:
                ticket_project = None
                ticket_key = ticket.get("key", "")
                if "-" in ticket_key:
                    ticket_project = ticket_key.split("-")[0]
                if not ticket_project:
                    ticket_project = ticket.get("fields", {}).get("project", {}).get("key", "")
                if ticket_project == project_id:
                    filtered_tickets.append(ticket)
            self.log(f"Filtered to {len(filtered_tickets)} tickets for project {project_id}")
        else:
            filtered_tickets = tickets
            self.log(f"Mock data - returning all {len(filtered_tickets)} tickets")
        filtered_status_debug = {}
        for ticket in filtered_tickets:
            status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
            filtered_status_debug[status] = filtered_status_debug.get(status, 0) + 1
        self.log(f"[DEBUG] Filtered ticket status distribution: {filtered_status_debug}")
        cache_key = f"filtered_tickets:{project_id}:all:all"
        filter_metadata = {
            "filtered_at": datetime.now().isoformat(),
            "source_count": len(tickets),
            "filtered_count": len(filtered_tickets),
            "project_id": project_id,
            "status_distribution": filtered_status_debug,
            "date_filter_applied": False
        }
        if filtered_tickets:
            self.redis_client.set(cache_key, json.dumps(filtered_tickets))
            self.redis_client.set(f"{cache_key}:metadata", json.dumps(filter_metadata))
            self.redis_client.expire(cache_key, 1800)
            self.redis_client.expire(f"{cache_key}:metadata", 1800)
        if self.use_mcp or self.use_real_api:
            self._assess_filtered_data_collaboration(filtered_tickets, project_id, 
                                                    "mcp_filtered" if self.use_mcp else "api_filtered")
        return filtered_tickets
    def _act(self) -> Dict[str, Any]:
        try:
            project_id = self.mental_state.get_belief("current_project") or ""
            time_range = self.mental_state.get_belief("current_time_range") or {}
            if not project_id or project_id == "None" or project_id == "":
                if self.available_projects:
                    if "FISCD" in self.available_projects:
                        project_id = "FISCD"
                    elif "MG" in self.available_projects:
                        project_id = "MG"
                    else:
                        project_id = self.available_projects[0]
                    self.log(f"[ACTION] No project specified, using: {project_id}")
                    self.mental_state.add_belief("current_project", project_id, 0.9, "fallback")
                else:
                    self.log("[ERROR] No projects available!")
                    return {
                        "tickets": [],
                        "workflow_status": "failure",
                        "metadata": {"error": "No projects available"},
                        "collaboration_insights": {"error_occurred": True}
                    }
            time_range = {}
            analysis_depth = self.mental_state.get_belief("requested_analysis_depth") or "basic"
            collaboration_context = self.mental_state.get_belief("collaboration_context")
            workflow_context = self.mental_state.get_belief("workflow_context")
            self.log(f"[ACTION] Retrieving data for project {project_id}")
            is_realtime_update = (
                workflow_context == "productivity_analysis" or 
                collaboration_context == "productivity_analysis" or
                self.mental_state.get_belief("force_fresh_data")
            )
            if collaboration_context in ["predictive_analysis", "productivity_analysis"] or is_realtime_update:
                self.log("[ACTION] Forcing fresh data load for analysis/real-time update")
                tickets_key = f"tickets:{project_id}"
                self.redis_client.delete(tickets_key)
                pattern = f"filtered_tickets:{project_id}:*"
                for key in self.redis_client.scan_iter(match=pattern):
                    self.redis_client.delete(key)
            tickets_key = f"tickets:{project_id}"
            cached_tickets = self.redis_client.get(tickets_key) if not is_realtime_update else None
            if cached_tickets and not is_realtime_update:
                filtered_tickets = json.loads(cached_tickets)
                self.log(f"Retrieved {len(filtered_tickets)} tickets from cache")
            else:
                self.log("Loading fresh data...")
                self.mental_state.add_belief("loading_project", project_id, 0.9, "data_loading")
                self.mental_state.add_belief("loading_time_range", {}, 0.9, "data_loading")
                all_tickets = self._load_jira_data()
                filtered_tickets = self._filter_tickets(all_tickets, project_id, {})
                if filtered_tickets:
                    cache_ttl = 300 if is_realtime_update else 7200
                    self.redis_client.set(tickets_key, json.dumps(filtered_tickets))
                    self.redis_client.expire(tickets_key, cache_ttl)
            if filtered_tickets:
                status_counts = {}
                for ticket in filtered_tickets:
                    status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
                    status_counts[status] = status_counts.get(status, 0) + 1
                self.log(f"[ACTION] Ticket status distribution: {status_counts}")
                self.log(f"[ACTION] Total tickets: {len(filtered_tickets)}")
            else:
                self.log(f"[ACTION] No tickets found for project {project_id}")
            enhanced_metadata = self._generate_enhanced_metadata(filtered_tickets, project_id, collaboration_context)
            return {
                "tickets": filtered_tickets,
                "workflow_status": "success",
                "metadata": enhanced_metadata,
                "collaboration_insights": {
                    "collaboration_requests_made": len(self.mental_state.collaborative_requests),
                    "data_quality_assessed": True,
                    "analysis_depth_provided": analysis_depth,
                    "data_source": "api" if self.use_real_api else "mock",
                    "fresh_data_loaded": not bool(cached_tickets) or is_realtime_update
                }
            }
        except Exception as e:
            self.log(f"[ERROR] Failed to process Jira data: {str(e)}")
            return {
                "tickets": [],
                "workflow_status": "failure",
                "metadata": {"error": str(e)},
                "collaboration_insights": {"error_occurred": True}
            }
    def _assess_data_collaboration_needs(self, issues: List[Dict[str, Any]], data_source: str):
        if not issues:
            return
        data_characteristics = self._analyze_data_characteristics(issues)
        self.log(f"[COLLABORATION ASSESSMENT] Analyzing {len(issues)} tickets from {data_source}")
        if data_characteristics["suggests_productivity_analysis"]:
            self.log("[COLLABORATION OPPORTUNITY] Data suggests productivity analysis would be valuable")
            self.mental_state.request_collaboration(
                agent_type="productivity_dashboard_agent",
                reasoning_type="data_analysis",
                context={
                    "reason": "rich_productivity_data_available",
                    "ticket_count": len(issues),
                    "status_diversity": data_characteristics["status_diversity"],
                    "has_cycle_time_data": data_characteristics["has_cycle_time_data"],
                    "data_source": data_source
                }
            )
        if data_characteristics["suggests_recommendation_enhancement"]:
            self.log("[COLLABORATION OPPORTUNITY] Data could enhance recommendation quality")
            self.mental_state.request_collaboration(
                agent_type="recommendation_agent",
                reasoning_type="context_enrichment",
                context={
                    "reason": "data_can_enhance_recommendations",
                    "resolution_patterns": data_characteristics["resolution_patterns"],
                    "team_activity": data_characteristics["team_activity"]
                }
            )
    def _analyze_data_characteristics(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        characteristics = {
            "status_diversity": 0,
            "has_cycle_time_data": False,
            "resolution_patterns": [],
            "team_activity": {},
            "suggests_productivity_analysis": False,
            "suggests_recommendation_enhancement": False
        }
        if not issues:
            return characteristics
        statuses = set()
        assignees = {}
        resolved_tickets = []
        for issue in issues:
            fields = issue.get("fields", {})
            status = fields.get("status", {}).get("name", "Unknown")
            statuses.add(status)
            assignee_info = fields.get("assignee")
            if assignee_info:
                assignee = assignee_info.get("displayName", "Unknown")
                assignees[assignee] = assignees.get(assignee, 0) + 1
            if status == "Done" and fields.get("resolutiondate"):
                resolved_tickets.append(issue)
            changelog = issue.get("changelog", {}).get("histories", [])
            if changelog:
                characteristics["has_cycle_time_data"] = True
        characteristics["status_diversity"] = len(statuses)
        characteristics["team_activity"] = assignees
        characteristics["resolution_patterns"] = [
            {"resolved_count": len(resolved_tickets), "total_tickets": len(issues)}
        ]
        characteristics["suggests_productivity_analysis"] = (
            characteristics["status_diversity"] >= 3 and
            characteristics["has_cycle_time_data"] and
            len(issues) >= 5
        )
        characteristics["suggests_recommendation_enhancement"] = (
            len(assignees) >= 2 and
            len(resolved_tickets) >= 3
        )
        return characteristics
    def _filter_tickets(self, tickets: List[Dict[str, Any]], project_id: str, time_range: Dict[str, str]) -> List[Dict[str, Any]]:
        if not tickets:
            self.log("No tickets to filter")
            return []
        if tickets is None:
            self.log("[WARNING] Tickets is None, returning empty list")
            return []
        if time_range is None:
            time_range = {}
        self.log(f"[FILTER] Filtering {len(tickets)} tickets for project {project_id}")
        self.log(f"[FILTER] Time range: {time_range}")
        status_debug = {}
        project_debug = {}
        for ticket in tickets:
            status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
            status_debug[status] = status_debug.get(status, 0) + 1
            ticket_project = ticket.get("fields", {}).get("project", {}).get("key", "Unknown")
            project_debug[ticket_project] = project_debug.get(ticket_project, 0) + 1
        self.log(f"[DEBUG] Raw ticket status distribution: {status_debug}")
        self.log(f"[DEBUG] Raw ticket project distribution: {project_debug}")
        self.log(f"[DEBUG] Total raw tickets: {len(tickets)}")
        filtered_tickets = []
        if self.use_mcp or self.use_real_api:
            self.log(f"API/MCP data - filtering for project {project_id}")
            for ticket in tickets:
                ticket_project = None
                ticket_key = ticket.get("key", "")
                if "-" in ticket_key:
                    ticket_project = ticket_key.split("-")[0]
                if not ticket_project:
                    ticket_project = ticket.get("fields", {}).get("project", {}).get("key", "")
                if ticket_project:
                    self.log(f"[DEBUG] Ticket {ticket.get('key', 'Unknown')}: Project = {ticket_project}")
                if ticket_project == project_id:
                    filtered_tickets.append(ticket)
                    status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
                    self.log(f"[DEBUG] Matched ticket {ticket.get('key')}: Status = {status}")
            self.log(f"Filtered to {len(filtered_tickets)} tickets for project {project_id}")
        else:
            filtered_tickets = tickets
            self.log(f"Mock data - returning all {len(filtered_tickets)} tickets")
        filtered_status_debug = {}
        for ticket in filtered_tickets:
            status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
            filtered_status_debug[status] = filtered_status_debug.get(status, 0) + 1
        self.log(f"[DEBUG] Filtered ticket status distribution: {filtered_status_debug}")
        start_time = time_range.get('start', 'all') if time_range else 'all'
        end_time = time_range.get('end', 'all') if time_range else 'all'
        cache_key = f"filtered_tickets:{project_id}:{start_time}:{end_time}"
        filter_metadata = {
            "filtered_at": datetime.now().isoformat(),
            "source_count": len(tickets),
            "filtered_count": len(filtered_tickets),
            "project_id": project_id,
            "status_distribution": filtered_status_debug
        }
        if filtered_tickets:
            self.redis_client.set(cache_key, json.dumps(filtered_tickets))
            self.redis_client.set(f"{cache_key}:metadata", json.dumps(filter_metadata))
            self.redis_client.expire(cache_key, 1800)
            self.redis_client.expire(f"{cache_key}:metadata", 1800)
        if self.use_mcp or self.use_real_api:
            self._assess_filtered_data_collaboration(filtered_tickets, project_id, 
                                                    "mcp_filtered" if self.use_mcp else "api_filtered")
        return filtered_tickets
    def _assess_filtered_data_collaboration(self, filtered_tickets: List[Dict[str, Any]], 
                                          project_id: str, data_source: str):
        if not filtered_tickets:
            return
        ticket_count = len(filtered_tickets)
        self.log(f"[COLLABORATION ASSESSMENT] Analyzing {ticket_count} filtered tickets for {project_id}")
        if ticket_count >= 10:
            self.log("[COLLABORATION OPPORTUNITY] Substantial project data - suggesting comprehensive analysis")
            self.mental_state.request_collaboration(
                agent_type="productivity_dashboard_agent",
                reasoning_type="comprehensive_analysis",
                context={
                    "reason": "substantial_project_data",
                    "project_id": project_id,
                    "ticket_count": ticket_count,
                    "analysis_type": "comprehensive",
                    "data_source": data_source
                }
            )
        patterns = self._identify_ticket_patterns(filtered_tickets)
        if patterns["has_interesting_patterns"]:
            self.log("[COLLABORATION OPPORTUNITY] Interesting patterns detected - could enhance recommendations")
            self.mental_state.request_collaboration(
                agent_type="recommendation_agent",
                reasoning_type="pattern_analysis",
                context={
                    "reason": "interesting_patterns_detected",
                    "project_id": project_id,
                    "patterns": patterns,
                    "pattern_confidence": patterns["confidence"]
                }
            )
    def _identify_ticket_patterns(self, tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
        patterns = {
            "has_interesting_patterns": False,
            "confidence": 0.0,
            "pattern_types": []
        }
        if len(tickets) < 3:
            return patterns
        status_counts = {}
        for ticket in tickets:
            status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        in_progress = status_counts.get("In Progress", 0)
        done = status_counts.get("Done", 0)
        if in_progress > done and in_progress > 3:
            patterns["pattern_types"].append("bottleneck_detected")
            patterns["confidence"] += 0.3
        resolved_recently = 0
        for ticket in tickets:
            resolution_date = ticket.get("fields", {}).get("resolutiondate")
            if resolution_date:
                try:
                    resolved_date = datetime.fromisoformat(resolution_date.replace("Z", "+00:00"))
                    days_ago = (datetime.now(resolved_date.tzinfo) - resolved_date).days
                    if days_ago <= 7:
                        resolved_recently += 1
                except:
                    pass
        if resolved_recently >= 3:
            patterns["pattern_types"].append("high_velocity")
            patterns["confidence"] += 0.4
        assignees = set()
        for ticket in tickets:
            assignee_info = ticket.get("fields", {}).get("assignee")
            if assignee_info:
                assignees.add(assignee_info.get("displayName", "Unknown"))
        if len(assignees) >= 3:
            patterns["pattern_types"].append("distributed_team")
            patterns["confidence"] += 0.2
        patterns["has_interesting_patterns"] = patterns["confidence"] >= 0.3
        return patterns
    def _monitor_file(self, project_id: str, time_range: Dict[str, str]):
        try:
            current_modified = os.path.getmtime(self.mock_data_path)
            if self.last_modified is None or current_modified > self.last_modified:
                self.log(f"File {self.mock_data_path} has been modified at {current_modified}")
                self.last_modified = current_modified
                all_tickets = self._load_jira_data()
                filtered_tickets = self._filter_tickets(all_tickets, project_id, time_range)
                tickets_key = f"tickets:{project_id}"
                metadata_key = f"tickets_meta:{project_id}"
                self.redis_client.set(tickets_key, json.dumps(filtered_tickets, default=str))
                self.redis_client.expire(tickets_key, 7200)
                metadata = {
                    "last_updated": datetime.now().isoformat(),
                    "ticket_count": len(filtered_tickets),
                    "has_changes": True,
                    "project_id": project_id,
                    "time_range": time_range,
                    "file_modified": current_modified,
                    "collaborative_opportunities_assessed": True,
                    "collaboration_requests_made": len(self.mental_state.collaborative_requests)
                }
                self.redis_client.set(metadata_key, json.dumps(metadata))
                self.redis_client.expire(metadata_key, 7200)
                update_event = {
                    "event_type": "tickets_updated",
                    "project_id": project_id,
                    "ticket_count": len(filtered_tickets),
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": self.agent_id,
                    "collaborative_assessment_done": True,
                    "collaboration_opportunities": len([req for req in self.mental_state.collaborative_requests if req.get("timestamp", "") > datetime.now().replace(hour=0, minute=0, second=0).isoformat()])
                }
                self.redis_client.publish("jira_updates", json.dumps(update_event))
                self.log(f"Updated Redis with {len(filtered_tickets)} tickets for project {project_id} (collaborative intelligence applied)")
                self.mental_state.add_belief("last_update_success", True, 0.9, "file_monitor")
                self.mental_state.add_belief("tickets_cached", len(filtered_tickets), 0.9, "file_monitor")
                self.mental_state.add_belief("collaborative_assessment_complete", True, 0.9, "intelligence")
        except Exception as e:
            self.log(f"[ERROR] Failed to monitor file: {str(e)}")
            self.mental_state.add_belief("monitor_error", str(e), 0.8, "error")
    def _monitor_file_loop(self):
        project_id = "PROJ123"  
        time_range = {
            "start": "2025-05-01T00:00:00Z",
            "end": "2025-05-15T23:59:59Z"
        }
        while True:
            try:
                self._monitor_file(project_id, time_range)
                self.log("Intelligent monitoring cycle complete - sleeping for 10 seconds")
                time.sleep(10)
            except Exception as e:
                self.log(f"[ERROR] Monitor loop error: {str(e)}")
                time.sleep(30)
    def _perceive(self, input_data: Dict[str, Any]) -> None:
        super()._perceive(input_data)
        project_id = input_data.get("project_id", "")
        time_range = input_data.get("time_range", {})
        analysis_depth = input_data.get("analysis_depth", "basic")
        self.log(f"[PERCEPTION] Processing request for project {project_id} with {analysis_depth} analysis depth")
        self.mental_state.add_belief("current_project", project_id, 0.9, "input")
        self.mental_state.add_belief("current_time_range", time_range, 0.9, "input")
        self.mental_state.add_belief("requested_analysis_depth", analysis_depth, 0.9, "input")
        if input_data.get("collaboration_purpose"):
            self.mental_state.add_belief("collaboration_context", input_data.get("collaboration_purpose"), 0.9, "collaboration")
            self.log(f"[COLLABORATION] Operating in collaborative mode: {input_data.get('collaboration_purpose')}")
        self._assess_request_collaboration_needs(input_data)
    def _assess_request_collaboration_needs(self, input_data: Dict[str, Any]) -> None:
        analysis_depth = input_data.get("analysis_depth", "basic")
        project_id = input_data.get("project_id")
        collaboration_purpose = input_data.get("collaboration_purpose")
        self.log(f"[COLLABORATION ASSESSMENT] Evaluating request for {project_id} with {analysis_depth} depth")
        if analysis_depth == "enhanced":
            self.log("[COLLABORATION OPPORTUNITY] Enhanced analysis requested - suggesting productivity collaboration")
            self.mental_state.request_collaboration(
                agent_type="productivity_dashboard_agent",
                reasoning_type="data_analysis",
                context={
                    "reason": "enhanced_analysis_requested", 
                    "project": project_id,
                    "analysis_type": "comprehensive"
                }
            )
        if collaboration_purpose:
            if "recommendation" in collaboration_purpose.lower():
                self.log("[COLLABORATION OPTIMIZATION] Optimizing data retrieval for recommendation context")
                self.mental_state.add_belief("optimize_for_recommendations", True, 0.9, "collaboration")
            elif "analysis" in collaboration_purpose.lower():
                self.log("[COLLABORATION OPTIMIZATION] Optimizing data retrieval for analysis context")
                self.mental_state.add_belief("optimize_for_analysis", True, 0.9, "collaboration")
    def _provide_enhanced_data_context(self, tickets: List[Dict[str, Any]], project_id: str):
        if not tickets:
            return
        data_summary = {
            "total_tickets": len(tickets),
            "status_distribution": {},
            "team_distribution": {},
            "recent_activity": 0,
            "completion_rate": 0
        }
        completed_tickets = 0
        recent_updates = 0
        one_week_ago = datetime.now() - timedelta(days=7)
        for ticket in tickets:
            fields = ticket.get("fields", {})
            status = fields.get("status", {}).get("name", "Unknown")
            data_summary["status_distribution"][status] = data_summary["status_distribution"].get(status, 0) + 1
            if status == "Done":
                completed_tickets += 1
            assignee_info = fields.get("assignee")
            if assignee_info:
                assignee = assignee_info.get("displayName", "Unassigned")
                data_summary["team_distribution"][assignee] = data_summary["team_distribution"].get(assignee, 0) + 1
            updated_str = fields.get("updated", "")
            if updated_str:
                try:
                    updated_time = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                    if updated_time.replace(tzinfo=None) > one_week_ago:
                        recent_updates += 1
                except:
                    pass
        data_summary["recent_activity"] = recent_updates
        data_summary["completion_rate"] = completed_tickets / len(tickets) if tickets else 0
        self.mental_state.add_belief("enhanced_data_context", data_summary, 0.9, "intelligence")
        self.log(f"[INTELLIGENCE] Generated enhanced data context: {data_summary}")
    def _generate_enhanced_metadata(self, tickets: List[Dict[str, Any]], project_id: str, 
                                  collaboration_context: str) -> Dict[str, Any]:
        metadata = {
            "project_id": project_id,
            "timestamp": datetime.now().isoformat(),
            "cache_hit": self.mental_state.get_belief("cache_hit_main"),
            "ticket_count": len(tickets),
            "agent_id": self.agent_id,
            "data_intelligence_applied": True
        }
        if collaboration_context:
            metadata["collaboration_context"] = collaboration_context
            metadata["optimized_for_collaboration"] = True
        enhanced_context = self.mental_state.get_belief("enhanced_data_context")
        if enhanced_context:
            metadata["data_analysis"] = enhanced_context
        collaboration_requests = self.mental_state.collaborative_requests
        if collaboration_requests:
            metadata["collaboration_opportunities"] = [
                {
                    "agent_type": req.get("agent_type"),
                    "reason": req.get("context", {}).get("reason"),
                    "timestamp": req.get("timestamp")
                }
                for req in collaboration_requests[-5:]
            ]
        return metadata
    def _rethink(self, action_result: Dict[str, Any]) -> None:
        super()._rethink(action_result)
        status = action_result.get("workflow_status", "failure")
        ticket_count = len(action_result.get("tickets", []))
        cache_hit = action_result.get("metadata", {}).get("cache_hit", False)
        collaboration_insights = action_result.get("collaboration_insights", {})
        collaborative_interaction = collaboration_insights.get("collaborative_opportunities_identified", 0) > 0
        if status == "success":
            if cache_hit:
                self.mental_state.competency_model.add_competency("collaborative_cache_retrieval", 1.0)
            else:
                self.mental_state.competency_model.add_competency("collaborative_fresh_data_load", 1.0)
            if collaborative_interaction:
                self.mental_state.competency_model.add_competency("collaborative_intelligence", 1.0)
        else:
            self.mental_state.competency_model.add_competency("error_handling", 0.3)
        reflection = {
            "operation": "intelligent_jira_data_retrieval",
            "success": status == "success",
            "ticket_count": ticket_count,
            "cache_performance": cache_hit,
            "collaborative_interaction": collaborative_interaction,
            "collaboration_opportunities_identified": collaboration_insights.get("collaborative_opportunities_identified", 0),
            "data_intelligence_applied": True,
            "performance_notes": f"Retrieved {ticket_count} tickets with {'cache hit' if cache_hit else 'fresh load'} and {'collaborative intelligence' if collaborative_interaction else 'standard processing'}"
        }
        self.mental_state.add_reflection(reflection)
        if collaborative_interaction:
            self.mental_state.add_experience(
                experience_description=f"Applied collaborative intelligence during data retrieval",
                outcome=f"identified_{collaboration_insights.get('collaborative_opportunities_identified', 0)}_collaboration_opportunities",
                confidence=0.8,
                metadata={
                    "collaboration_success": status == "success",
                    "data_quality": "high" if ticket_count >= 5 else "limited"
                }
            )
        self.log(f"[REFLECTION] Intelligent retrieval completed: {reflection}")
    def get_performance_metrics(self) -> Dict[str, Any]:
        try:
            metrics_key = f"performance:{self.agent_id}:{datetime.now().strftime('%Y%m%d_%H')}"
            raw_metrics = self.redis_client.lrange(metrics_key, 0, -1)
            metrics = []
            collaborative_interactions = 0
            for raw_metric in raw_metrics:
                metric_data = json.loads(raw_metric)
                metrics.append(metric_data)
                if metric_data.get("collaborative"):
                    collaborative_interactions += 1
            if not metrics:
                return {}
            return {
                "total_retrievals": len(metrics),
                "cache_hit_rate": sum(1 for m in metrics if m.get("cache_hit")) / len(metrics),
                "avg_tickets_per_retrieval": sum(m.get("tickets_retrieved", 0) for m in metrics) / len(metrics),
                "collaborative_interaction_rate": collaborative_interactions / len(metrics),
                "intelligence_features_active": True,
                "recent_metrics": metrics[-10:]
            }
        except Exception as e:
            self.log(f"[ERROR] Failed to get performance metrics: {str(e)}")
            return {}
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(input_data)
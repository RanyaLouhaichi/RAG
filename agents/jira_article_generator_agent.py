# agents/jira_article_generator_agent.py - COMPLETE VERSION WITH FEEDBACK
import json
import re
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent, AgentCapability
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from orchestrator.core.query_type import QueryType # type: ignore
from orchestrator.core.model_manager import ModelManager # type: ignore
from datetime import datetime
from agents.jira_data_agent import JiraDataAgent
from agents.knowledge_base_agent import KnowledgeBaseAgent

class JiraArticleGeneratorAgent(BaseAgent):
    OBJECTIVE = "Generate high-quality, Confluence-ready articles based on resolved Jira tickets with human feedback integration"

    def __init__(self, shared_memory: JurixSharedMemory):
        super().__init__(name="jira_article_generator_agent", redis_client=shared_memory.redis_client)
        self.shared_memory = shared_memory
        self.model_manager = ModelManager()
        
        # Initialize agents for collaboration
        self.jira_data_agent = JiraDataAgent(redis_client=shared_memory.redis_client)
        self.knowledge_base_agent = KnowledgeBaseAgent(shared_memory)
        
        self.mental_state.capabilities = [
            AgentCapability.GENERATE_ARTICLE,
            AgentCapability.COORDINATE_AGENTS,
            AgentCapability.PROCESS_FEEDBACK
        ]
        
        self.mental_state.obligations.extend([
            "detect_query_type",
            "generate_article",
            "extract_solution_from_comments",  # NEW: Extract actual solutions
            "analyze_ticket_history",          # NEW: Analyze what was done
            "identify_fix_patterns",           # NEW: Identify the actual fix
            "assess_collaboration_needs",
            "coordinate_with_agents",
            "process_human_feedback",
            "track_article_versions"
        ])

        self.collaboration_threshold = 0.4
        self.always_try_collaboration = True
        self.max_refinement_iterations = 5

    def _extract_solution_from_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the ACTUAL solution from ticket comments and history"""
        self.log(f"[SOLUTION EXTRACTION] Analyzing ticket for actual resolution")
        
        solution_data = {
            "problem_description": "",
            "actual_solution": "",
            "fix_details": [],
            "code_changes": [],
            "configuration_changes": [],
            "resolution_steps": [],
            "root_cause": "",
            "preventive_measures": [],
            "who_fixed": "",
            "when_fixed": "",
            "verification_steps": []
        }
        
        fields = ticket.get("fields", {})
        
        # 1. Extract problem from description - FIX NULL SAFETY
        summary = fields.get("summary") or ""
        description = fields.get("description") or ""
        solution_data["problem_description"] = f"{summary}\n{description}" if summary or description else "No description available"
        
        # 2. CRITICAL: Extract solution from comments
        comments = fields.get("comment", {}).get("comments", [])
        self.log(f"[SOLUTION EXTRACTION] Found {len(comments)} comments to analyze")
        
        for comment in comments:
            comment_body = comment.get("body", "").lower()
            author = comment.get("author", {}).get("displayName", "Unknown")
            created = comment.get("created", "")
            
            # Look for solution indicators in comments
            solution_indicators = [
                "fixed by", "resolved by", "solution:", "fix:", "solved:",
                "the issue was", "the problem was", "root cause", "caused by",
                "applied", "implemented", "changed", "updated", "modified",
                "deployed", "merged", "committed", "pushed"
            ]
            
            if any(indicator in comment_body for indicator in solution_indicators):
                self.log(f"[SOLUTION EXTRACTION] Found solution indicator in comment by {author}")
                solution_data["fix_details"].append({
                    "author": author,
                    "date": created,
                    "content": comment.get("body", "")
                })
                
                # Extract specific fixes
                self._extract_specific_fixes(comment.get("body", ""), solution_data)
        
        # 3. Analyze ticket history for actual changes
        changelog = ticket.get("changelog", {}).get("histories", [])
        self.log(f"[SOLUTION EXTRACTION] Analyzing {len(changelog)} history entries")
        
        for history in changelog:
            author = history.get("author", {}).get("displayName", "Unknown")
            created = history.get("created", "")
            
            for item in history.get("items", []):
                field = item.get("field", "")
                from_val = item.get("fromString", "")
                to_val = item.get("toString", "")
                
                # Track status changes to Done/Resolved
                if field == "status" and to_val in ["Done", "Resolved", "Closed"]:
                    solution_data["who_fixed"] = author
                    solution_data["when_fixed"] = created
                    self.log(f"[SOLUTION EXTRACTION] Ticket resolved by {author} on {created}")
        
        # 4. Build the actual solution narrative
        if solution_data["fix_details"]:
            # Combine all fix details into a coherent solution
            solution_parts = []
            for fix in solution_data["fix_details"]:
                solution_parts.append(fix["content"])
            
            solution_data["actual_solution"] = "\n\n".join(solution_parts)
        else:
            # Fallback: Look for resolution in the last comments
            if comments:
                last_comments = comments[-3:]  # Last 3 comments
                for comment in reversed(last_comments):
                    if "done" in comment.get("body", "").lower() or "fixed" in comment.get("body", "").lower():
                        solution_data["actual_solution"] = comment.get("body", "")
                        break
        
        # 5. Extract resolution field if available
        if fields.get("resolution"):
            resolution_name = fields.get("resolution", {}).get("name", "")
            resolution_desc = fields.get("resolution", {}).get("description", "")
            if resolution_desc:
                solution_data["actual_solution"] += f"\n\nResolution Type: {resolution_name}\n{resolution_desc}"
        
        return solution_data
    
    def _extract_specific_fixes(self, comment_text: str, solution_data: Dict[str, Any]):
        """Extract specific technical fixes from comment text"""
        
        # Look for code changes
        code_patterns = [
            r'```[\s\S]*?```',  # Markdown code blocks
            r'`[^`]+`',         # Inline code
            r'(changed|modified|updated)\s+\w+\.\w+',  # File changes
            r'(set|changed|updated)\s+\w+\s*=\s*\w+',  # Configuration changes
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, comment_text, re.IGNORECASE)
            solution_data["code_changes"].extend(matches)
        
        # Look for configuration changes
        config_indicators = [
            "configuration", "config", "setting", "parameter", "property",
            "environment variable", "env var", "flag"
        ]
        
        lines = comment_text.split('\n')
        for line in lines:
            if any(indicator in line.lower() for indicator in config_indicators):
                solution_data["configuration_changes"].append(line.strip())
        
        # Extract step-by-step instructions
        step_patterns = [
            r'^\d+\.',      # 1. Step
            r'^-\s',        # - Bullet point
            r'^\*\s',       # * Bullet point
            r'^step\s+\d+', # Step 1
        ]
        
        for i, line in enumerate(lines):
            for pattern in step_patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    solution_data["resolution_steps"].append(line.strip())

    def _build_comprehensive_prompt_with_solution(self, ticket_id: str, ticket_data: Dict[str, Any], 
                                            solution_data: Dict[str, Any], 
                                            enhanced_context: Dict[str, Any],
                                            refinement_suggestion: str = None) -> str:
        """Build prompt that focuses on the ACTUAL solution implemented"""
        
        fields = ticket_data.get("fields", {})
        summary = fields.get("summary", "No summary")
        
        # ENSURE solution_data has all required fields
        if not solution_data:
            solution_data = {
                "problem_description": "",
                "actual_solution": "",
                "fix_details": [],
                "code_changes": [],
                "configuration_changes": [],
                "resolution_steps": [],
                "who_fixed": "",
                "when_fixed": "",
            }
        
        # Build safe strings
        problem_desc = solution_data.get('problem_description') or 'No problem description available'
        actual_solution = solution_data.get('actual_solution') or 'No explicit solution found in comments - please derive from context'
        who_fixed = solution_data.get('who_fixed') or 'Unknown'
        when_fixed = solution_data.get('when_fixed') or 'Unknown'
        
        # Safe list handling
        code_changes = solution_data.get('code_changes', [])
        code_changes_str = ', '.join(str(c) for c in code_changes[:5]) if code_changes else 'None documented'
        
        config_changes = solution_data.get('configuration_changes', [])
        config_changes_str = ', '.join(str(c) for c in config_changes[:3]) if config_changes else 'None documented'
        
        resolution_steps = solution_data.get('resolution_steps', [])
        resolution_steps_str = '\n'.join(str(s) for s in resolution_steps) if resolution_steps else 'No explicit steps documented'
        
        # Build fix details section safely
        fix_details_section = ""
        fix_details = solution_data.get('fix_details', [])
        if fix_details:
            for fix in fix_details[:3]:
                if isinstance(fix, dict):
                    author = fix.get('author', 'Unknown')
                    date = fix.get('date', 'Unknown')
                    content = (fix.get('content') or '')[:200]
                    fix_details_section += f"\n- {author} ({date}): {content}..."
        else:
            fix_details_section = "\nNo additional comment details available"
        
        prompt = f"""You must output ONLY the article content. Do not include any preamble, introduction, or meta-commentary.

    DO NOT START WITH:
    - "Sure, I'd be happy to..."
    - "Here's an example..."
    - "As an AI..."
    - "I will create..."
    - Any greeting or acknowledgment

    START DIRECTLY WITH THE ARTICLE TITLE.

    TICKET DATA:
    - ID: {ticket_id}
    - Summary: {summary}
    - Problem: {problem_desc}
    - Solution: {actual_solution}
    - Fixed by: {who_fixed} on {when_fixed}

    OUTPUT THE FOLLOWING ARTICLE DIRECTLY:

    # Know-How: {ticket_id} - {summary}

    ## Problem Overview
    {problem_desc}

    ## Root Cause Analysis
    [Analyze why this issue occurred based on: {actual_solution}]

    ## Solution Implementation
    {actual_solution}

    ### Implementation Steps:
    {resolution_steps_str}

    ## Technical Details
    ### Code Changes
    {code_changes_str}

    ### Configuration Changes
    {config_changes_str}

    ## Verification Steps
    [Document how the fix was verified to work correctly]

    ## Business Impact
    [Describe the impact of this issue and its resolution on users/business]

    ## Related Knowledge
    [List similar issues or patterns: look for tickets with similar symptoms]

    ## Preventive Measures
    [Based on the root cause, list steps to prevent recurrence]

    ## Next Steps
    [Any follow-up actions or monitoring required]

    BEGIN OUTPUT NOW:"""

        if refinement_suggestion:
            prompt += f"\n\nADDITIONAL REQUIREMENT: {refinement_suggestion}"

        return prompt

    def _detect_query_type(self, query: str) -> QueryType:
        if not query:
            return QueryType.CONVERSATION
        query = query.lower()
        article_keywords = ["article", "generate", "create", "write", "ticket"]
        if any(keyword in query for keyword in article_keywords):
            return QueryType.CONVERSATION
        return QueryType.CONVERSATION

    def _perceive(self, input_data: Dict[str, Any]) -> None:
        super()._perceive(input_data)
        ticket_id = input_data.get("ticket_id")
        refinement_suggestion = input_data.get("refinement_suggestion")
        human_feedback = input_data.get("human_feedback")  # New: human feedback
        approval_status = input_data.get("approval_status", "pending")  # New: approval status
        article_version = input_data.get("article_version", 1)  # New: version tracking
        previous_article = input_data.get("previous_article")  # New: previous article data
        
        self.log(f"[PERCEPTION] Ticket: {ticket_id} | Version: {article_version} | Status: {approval_status}")
        if human_feedback:
            self.log(f"[PERCEPTION] Human Feedback: {human_feedback}")
        
        self.mental_state.beliefs.update({
            "ticket_id": ticket_id,
            "refinement_suggestion": refinement_suggestion,
            "human_feedback": human_feedback,
            "approval_status": approval_status,
            "article_version": article_version,
            "previous_article": previous_article,  # Store the previous article
            "query_type": QueryType.CONVERSATION,
            "autonomous_refinement_done": False,
            "collaboration_assessment_done": False,
            "context_richness": 0.0
        })

    def _get_previous_article_version(self, ticket_id: str, version: int) -> Optional[Dict[str, Any]]:
        """Retrieve a previous version of the article"""
        # First try version-specific key
        version_key = f"article_version:{ticket_id}:v{version}"
        article_json = self.redis_client.get(version_key)
        
        if article_json:
            return json.loads(article_json)
        
        # Fallback to draft key for current version
        if version == 0 or version == 1:
            draft_key = f"article_draft:{ticket_id}"
            article_json = self.redis_client.get(draft_key)
            if article_json:
                return json.loads(article_json)
        
        return None

    def _assess_collaboration_needs(self) -> Dict[str, Any]:
        """YOUR EXISTING METHOD - NO CHANGES"""
        ticket_id = self.mental_state.beliefs["ticket_id"]
        
        # Start with assumption that we need collaboration for quality articles
        needs_collaboration = True
        collaboration_reasons = []
        agents_needed = []
        
        self.log(f"[COLLABORATION ASSESSMENT] Analyzing needs for ticket {ticket_id}")
        
        # Always check if we have comprehensive ticket data
        try:
            # Quick test: try to get basic project data
            test_input = {
                "project_id": "PROJ123",
                "time_range": {"start": "2025-05-01T00:00:00Z", "end": "2025-05-17T23:59:59Z"}
            }
            test_result = self.jira_data_agent.run(test_input)
            available_tickets = test_result.get("tickets", [])
            
            target_ticket = next((t for t in available_tickets if t.get("key") == ticket_id), None)
            
            if not target_ticket:
                collaboration_reasons.append(f"Target ticket {ticket_id} not found in available data")
                agents_needed.append("jira_data_agent")
            else:
                # Check if ticket has comprehensive data
                fields = target_ticket.get("fields", {})
                changelog = target_ticket.get("changelog", {}).get("histories", [])
                
                completeness_score = 0
                if fields.get("summary"): completeness_score += 0.25
                if fields.get("description"): completeness_score += 0.25
                if fields.get("resolutiondate"): completeness_score += 0.25
                if changelog: completeness_score += 0.25
                
                if completeness_score < 0.75:
                    collaboration_reasons.append("Ticket data is incomplete - missing key information")
                    agents_needed.append("jira_data_agent")
                
                self.mental_state.beliefs["context_richness"] = completeness_score
                
        except Exception as e:
            self.log(f"[COLLABORATION ASSESSMENT] Error accessing ticket data: {e}")
            collaboration_reasons.append("Unable to assess ticket data quality")
            agents_needed.append("jira_data_agent")
        
        # Always request knowledge base context for comprehensive articles
        collaboration_reasons.append("Need related knowledge articles for comprehensive documentation")
        agents_needed.append("retrieval_agent")
        
        # Remove duplicates
        agents_needed = list(set(agents_needed))
        
        assessment = {
            "needs_collaboration": needs_collaboration,
            "collaboration_reasons": collaboration_reasons,
            "agents_needed": agents_needed,
            "confidence_without_collaboration": self.mental_state.beliefs.get("context_richness", 0.2),
            "assessment_completed": True
        }
        
        self.log(f"[COLLABORATION ASSESSMENT] Result: {assessment}")
        return assessment

    def _coordinate_with_agents(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """YOUR EXISTING METHOD - NO CHANGES"""
        ticket_id = self.mental_state.beliefs["ticket_id"]
        
        # IMPORTANT: Extract the actual project from the ticket ID
        project_id = ticket_id.split('-')[0] if '-' in ticket_id else "PROJ123"
        self.log(f"[COORDINATION] Extracted project {project_id} from ticket {ticket_id}")
        
        enhanced_context = {
            "collaboration_metadata": {
                "collaborating_agents": [],
                "collaboration_types": [],
                "collaboration_start": datetime.now().isoformat()
            }
        }
        
        self.log(f"[COORDINATION] Starting collaboration with {len(assessment['agents_needed'])} agents")
        
        for agent_name in assessment["agents_needed"]:
            try:
                self.log(f"[COORDINATION] Collaborating with {agent_name}")
                
                if agent_name == "jira_data_agent":
                    # Get comprehensive ticket data WITHOUT time range
                    data_input = {
                        "project_id": project_id  # Use the ACTUAL project, not PROJ123!
                    }
                    
                    self.log(f"[COORDINATION] Requesting ALL tickets for project {project_id}")
                    result = self.jira_data_agent.run(data_input)
                    
                    all_tickets = result.get("tickets", [])
                    enhanced_context["tickets"] = all_tickets
                    enhanced_context["ticket_metadata"] = result.get("metadata", {})
                    
                    self.log(f"[COORDINATION] Received {len(all_tickets)} tickets from {project_id}")
                    
                    # Find our specific ticket
                    target_ticket = None
                    for ticket in all_tickets:
                        if ticket.get("key") == ticket_id:
                            target_ticket = ticket
                            break
                    
                    if target_ticket:
                        enhanced_context["target_ticket"] = target_ticket
                        self.log(f"[COORDINATION] ✅ Found target ticket {ticket_id}")
                        
                        # Log the actual ticket details for debugging
                        fields = target_ticket.get("fields", {})
                        self.log(f"[COORDINATION] Ticket Summary: {fields.get('summary', 'No summary')}")
                        self.log(f"[COORDINATION] Ticket Type: {fields.get('issuetype', {}).get('name', 'Unknown')}")
                        self.log(f"[COORDINATION] Ticket Status: {fields.get('status', {}).get('name', 'Unknown')}")
                    else:
                        self.log(f"[COORDINATION ERROR] ❌ Ticket {ticket_id} NOT FOUND in project {project_id}!")
                        # List some available tickets for debugging
                        available_keys = [t.get("key") for t in all_tickets[:10]]
                        self.log(f"[COORDINATION] First 10 available tickets: {available_keys}")
                        
                elif agent_name == "retrieval_agent":
                    # For now, we'll simulate this since RetrievalAgent integration is complex
                    # In your real implementation, you'd call the retrieval agent here
                    enhanced_context["related_articles"] = [
                        {"title": "Similar Resolution Patterns", "content": "Best practices for this type of issue"},
                        {"title": "Prevention Strategies", "content": "How to prevent similar issues"}
                    ]
                    self.log(f"[COORDINATION] Added related articles context")
                
                enhanced_context["collaboration_metadata"]["collaborating_agents"].append(agent_name)
                enhanced_context["collaboration_metadata"]["collaboration_types"].append(f"{agent_name}_context")
                
            except Exception as e:
                self.log(f"[COORDINATION ERROR] Failed to collaborate with {agent_name}: {e}")
                import traceback
                self.log(f"[COORDINATION ERROR] Traceback: {traceback.format_exc()}")
                continue
        
        enhanced_context["collaboration_metadata"]["collaboration_end"] = datetime.now().isoformat()
        enhanced_context["collaboration_metadata"]["total_collaborations"] = len(enhanced_context["collaboration_metadata"]["collaborating_agents"])
        enhanced_context["collaboration_successful"] = len(enhanced_context["collaboration_metadata"]["collaborating_agents"]) > 0
        
        self.log(f"[COORDINATION] Completed collaboration with {enhanced_context['collaboration_metadata']['total_collaborations']} agents")
        return enhanced_context

    # NEW FEEDBACK METHODS
    def _process_human_feedback(self, feedback: str, current_article: Dict[str, Any]) -> Dict[str, Any]:
        """Process human feedback and generate refinement instructions"""
        self.log(f"[FEEDBACK PROCESSING] Processing human feedback for refinement")
        
        # Analyze feedback sentiment and extract key points
        feedback_analysis = {
            "sentiment": self._analyze_feedback_sentiment(feedback),
            "key_points": self._extract_feedback_points(feedback),
            "specific_requests": self._identify_specific_requests(feedback),
            "priority_areas": self._determine_priority_areas(feedback, current_article)
        }
        
        # Generate specific refinement instructions
        refinement_instructions = self._generate_refinement_instructions(
            feedback_analysis, 
            current_article
        )
        
        return {
            "analysis": feedback_analysis,
            "instructions": refinement_instructions,
            "estimated_impact": self._estimate_refinement_impact(feedback_analysis)
        }

    def _analyze_feedback_sentiment(self, feedback: str) -> str:
        """Analyze the sentiment of the feedback"""
        positive_words = ["good", "great", "excellent", "perfect", "love", "helpful"]
        negative_words = ["missing", "unclear", "confusing", "wrong", "incorrect", "bad"]
        improvement_words = ["add", "include", "expand", "clarify", "detail", "more"]
        
        feedback_lower = feedback.lower()
        
        positive_count = sum(1 for word in positive_words if word in feedback_lower)
        negative_count = sum(1 for word in negative_words if word in feedback_lower)
        improvement_count = sum(1 for word in improvement_words if word in feedback_lower)
        
        if negative_count > positive_count:
            return "negative"
        elif improvement_count > positive_count:
            return "constructive"
        elif positive_count > 0:
            return "positive"
        else:
            return "neutral"

    def _extract_feedback_points(self, feedback: str) -> List[str]:
        """Extract key points from feedback"""
        sentences = feedback.split('.')
        key_points = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                if any(word in sentence.lower() for word in ["add", "include", "missing", "need", "should", "must"]):
                    key_points.append(sentence)
        
        return key_points[:5]

    def _identify_specific_requests(self, feedback: str) -> Dict[str, List[str]]:
        """Identify specific types of requests in feedback"""
        requests = {
            "add_content": [],
            "clarify": [],
            "remove": [],
            "restructure": [],
            "technical_details": [],
            "examples": []
        }
        
        feedback_lower = feedback.lower()
        
        if "add" in feedback_lower or "include" in feedback_lower:
            requests["add_content"].append(feedback)
        if "clarify" in feedback_lower or "explain" in feedback_lower:
            requests["clarify"].append(feedback)
        if "remove" in feedback_lower or "delete" in feedback_lower:
            requests["remove"].append(feedback)
        if "restructure" in feedback_lower or "reorganize" in feedback_lower:
            requests["restructure"].append(feedback)
        if "technical" in feedback_lower or "code" in feedback_lower or "implementation" in feedback_lower:
            requests["technical_details"].append(feedback)
        if "example" in feedback_lower or "sample" in feedback_lower:
            requests["examples"].append(feedback)
        
        return requests

    def _determine_priority_areas(self, feedback: str, current_article: Dict[str, Any]) -> List[str]:
        """Determine which sections of the article need priority attention"""
        priority_areas = []
        
        article_content = current_article.get("content", "")
        sections = ["Problem Overview", "Solution Implementation", "Business Impact", 
                   "Related Knowledge", "Strategic Recommendations", "Next Steps"]
        
        for section in sections:
            if section.lower() in feedback.lower():
                priority_areas.append(section)
        
        if not priority_areas:
            if "technical" in feedback.lower():
                priority_areas.append("Solution Implementation")
            if "business" in feedback.lower() or "impact" in feedback.lower():
                priority_areas.append("Business Impact")
            if "next" in feedback.lower() or "steps" in feedback.lower():
                priority_areas.append("Next Steps")
        
        return priority_areas

    def _generate_refinement_instructions(self, analysis: Dict[str, Any], 
                                        current_article: Dict[str, Any]) -> str:
        """Generate specific instructions for article refinement"""
        instructions = []
        
        sentiment = analysis["sentiment"]
        if sentiment == "negative":
            instructions.append("Major revisions needed based on feedback.")
        elif sentiment == "constructive":
            instructions.append("Enhance the article with the suggested improvements.")
        else:
            instructions.append("Refine the article based on the feedback provided.")
        
        for point in analysis["key_points"]:
            instructions.append(f"Address feedback: {point}")
        
        for area in analysis["priority_areas"]:
            instructions.append(f"Focus on improving the '{area}' section.")
        
        requests = analysis["specific_requests"]
        if requests["add_content"]:
            instructions.append("Add the requested content to make the article more comprehensive.")
        if requests["clarify"]:
            instructions.append("Clarify the mentioned points with clearer explanations.")
        if requests["technical_details"]:
            instructions.append("Include more technical details and implementation specifics.")
        if requests["examples"]:
            instructions.append("Add concrete examples to illustrate the concepts.")
        
        return "\n".join(instructions)

    def _estimate_refinement_impact(self, analysis: Dict[str, Any]) -> str:
        """Estimate the impact level of the refinement"""
        sentiment = analysis["sentiment"]
        num_points = len(analysis["key_points"])
        num_requests = sum(len(v) for v in analysis["specific_requests"].values())
        
        if sentiment == "negative" or num_points > 3 or num_requests > 4:
            return "major"
        elif num_points > 1 or num_requests > 2:
            return "moderate"
        else:
            return "minor"

    def _generate_article(self) -> Dict[str, Any]:
        """Generate article based on ACTUAL ticket resolution with feedback support"""
        ticket_id = self.mental_state.beliefs["ticket_id"]
        refinement_suggestion = self.mental_state.beliefs.get("refinement_suggestion")
        human_feedback = self.mental_state.beliefs.get("human_feedback")
        article_version = self.mental_state.beliefs.get("article_version", 1)
        
        self.log(f"[GENERATION] Generating article v{article_version} for resolved ticket: {ticket_id}")
        
        if human_feedback:
            self.log(f"[GENERATION] Processing human feedback: {human_feedback[:100]}...")
        
        # Get ticket data with full details
        project_id = ticket_id.split('-')[0] if '-' in ticket_id else "PROJ123"
        
        # CRITICAL FIX: Clear cache to force fresh data load
        self.log(f"[CACHE] Clearing cache for project {project_id} to get latest comments")
        
        # Clear all cached data for this project
        if self.shared_memory.redis_client:
            cache_patterns = [
                f"tickets:{project_id}",
                f"jira_api_data:{project_id}:*",
                f"filtered_tickets:{project_id}:*",
                f"jira_raw_data:*"
            ]
            
            for pattern in cache_patterns:
                for key in self.shared_memory.redis_client.scan_iter(match=pattern):
                    self.shared_memory.redis_client.delete(key)
                    self.log(f"[CACHE] Deleted cache key: {key}")
        
        # CRITICAL: Force fresh data load
        jira_result = self.jira_data_agent.run({
            "project_id": project_id,
            "force_fresh": True,  # Add flag to force fresh load
            "workflow_context": "article_generation"  # This triggers fresh load in JiraDataAgent
        })
        
        all_tickets = jira_result.get("tickets", [])
        target_ticket = None
        
        self.log(f"[DATA] Loaded {len(all_tickets)} fresh tickets from project {project_id}")
        
        for ticket in all_tickets:
            if ticket.get("key") == ticket_id:
                target_ticket = ticket
                # Log comment count to verify fresh data
                comments = ticket.get("fields", {}).get("comment", {}).get("comments", [])
                self.log(f"[DATA] Found ticket {ticket_id} with {len(comments)} comments (fresh data)")
                break
        
        if not target_ticket:
            self.log(f"[ERROR] Could not find ticket {ticket_id}")
            return self._create_fallback_article(ticket_id)
        
        # CRITICAL: Extract the actual solution from the ticket
        solution_data = self._extract_solution_from_ticket(target_ticket)
        
        if not solution_data["actual_solution"] and not solution_data["fix_details"]:
            self.log(f"[WARNING] No explicit solution found in ticket {ticket_id} - will try to derive from context")
        
        # Get previous version if this is a refinement
        previous_article = None
        if article_version > 1 or human_feedback:
            previous_article = self.mental_state.beliefs.get("previous_article")
            if not previous_article:
                previous_article = self._get_previous_article_version(ticket_id, article_version - 1)
        
        # Check if we need collaboration for additional context
        assessment = self._assess_collaboration_needs()
        enhanced_context = {}
        
        if assessment.get("needs_collaboration", True):
            self.log("[DECISION] Collaboration needed for comprehensive article")
            enhanced_context = self._coordinate_with_agents(assessment)
        
        # Build prompt based on whether we have feedback or not
        if human_feedback and previous_article:
            # Use feedback-aware prompt that ALSO includes solution data
            prompt = self._build_comprehensive_prompt_with_feedback_and_solution(
                ticket_id, 
                target_ticket,
                solution_data,
                enhanced_context,
                refinement_suggestion,
                human_feedback,
                previous_article
            )
        else:
            # Use solution-focused prompt for initial generation
            prompt = self._build_comprehensive_prompt_with_solution(
                ticket_id, 
                target_ticket,
                solution_data,
                enhanced_context, 
                refinement_suggestion
            )
        
        try:
            # Generate article content
            content = self.model_manager.generate_response(
                prompt=prompt,
                context={
                    "agent_name": self.name,
                    "task_type": "article_generation_with_feedback" if human_feedback else "article_generation_from_resolution",
                    "ticket_id": ticket_id,
                    "has_actual_solution": bool(solution_data["actual_solution"]),
                    "solution_extracted": True,
                    "article_version": article_version,
                    "is_human_refinement": bool(human_feedback)
                }
            )
            
            self.log(f"✅ Generated article based on actual resolution")
            
            # Create article with metadata
            article = {
                "content": content,
                "status": "pending_approval" if article_version > 1 else "draft",
                "title": f"Know-How: {ticket_id} - {target_ticket.get('fields', {}).get('summary', 'Resolution Guide')}",
                "created_at": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "version": article_version,
                "approval_status": "pending",
                "feedback_history": [],
                "solution_metadata": {
                    "has_explicit_solution": bool(solution_data["actual_solution"]),
                    "solution_source": "ticket_comments" if solution_data["fix_details"] else "derived",
                    "who_fixed": solution_data["who_fixed"],
                    "when_fixed": solution_data["when_fixed"],
                    "code_changes_found": len(solution_data["code_changes"]),
                    "config_changes_found": len(solution_data["configuration_changes"])
                },
                "collaboration_enhanced": bool(enhanced_context.get("collaboration_successful")),
                "quality_indicators": {
                    "based_on_actual_resolution": True,
                    "solution_extracted": bool(solution_data["actual_solution"]),
                    "technical_details_included": bool(solution_data["code_changes"] or solution_data["configuration_changes"])
                }
            }
            
            # Add feedback to history if provided
            if human_feedback:
                article["feedback_history"].append({
                    "version": article_version - 1,
                    "feedback": human_feedback,
                    "timestamp": datetime.now().isoformat(),
                    "applied": True
                })
            
            # Store version
            self._store_article_version(ticket_id, article_version, article)
            
            return article
            
        except Exception as e:
            self.log(f"[ERROR] Article generation failed: {e}")
            return self._create_fallback_article(ticket_id)
        
    def _build_comprehensive_prompt_with_feedback_and_solution(self, ticket_id: str,
                                                          ticket_data: Dict[str, Any],
                                                          solution_data: Dict[str, Any],
                                                          enhanced_context: Dict[str, Any],
                                                          refinement_suggestion: str = None,
                                                          human_feedback: str = None,
                                                          previous_article: Dict[str, Any] = None) -> str:
        """Build prompt that includes BOTH solution data AND feedback"""
        
        fields = ticket_data.get("fields", {})
        summary = fields.get("summary", "No summary")
        
        # Extract previous content safely
        previous_content = ""
        previous_version = 1
        if previous_article:
            previous_content = previous_article.get("content", "")
            previous_version = previous_article.get("version", 1)
        
        # Ensure solution_data is safe
        if not solution_data:
            solution_data = {
                "actual_solution": "",
                "code_changes": [],
                "configuration_changes": [],
                "resolution_steps": [],
                "fix_details": [],
                "who_fixed": "Unknown",
                "when_fixed": "Unknown"
            }
        
        # Build safe strings
        actual_solution = solution_data.get('actual_solution') or 'No explicit solution found - derive from context below'
        who_fixed = solution_data.get('who_fixed') or 'Unknown'
        when_fixed = solution_data.get('when_fixed') or 'Unknown'
        
        # Safe list handling
        code_changes = solution_data.get('code_changes', [])
        code_changes_str = ', '.join(str(c) for c in code_changes[:5]) if code_changes else 'None documented'
        
        config_changes = solution_data.get('configuration_changes', [])
        config_changes_str = ', '.join(str(c) for c in config_changes[:3]) if config_changes else 'None documented'
        
        resolution_steps = solution_data.get('resolution_steps', [])
        resolution_steps_str = '\n'.join(str(s) for s in resolution_steps) if resolution_steps else 'No explicit steps documented'
        
        # Build fix details section safely
        fix_details_section = ""
        fix_details = solution_data.get('fix_details', [])
        if fix_details:
            for fix in fix_details[:3]:
                if isinstance(fix, dict):
                    author = fix.get('author', 'Unknown')
                    date = fix.get('date', 'Unknown')
                    content = (fix.get('content') or '')[:200]
                    fix_details_section += f"\n- {author} ({date}): {content}..."
        else:
            fix_details_section = "\nNo additional comment details available"
        
        # Now build the prompt with all safe values
        prompt = f"""You are refining an article based on human feedback AND the actual solution implemented. You are a technical documentation writer. Write professional, direct documentation without any meta-commentary or conversational elements. Never refer to yourself or your capabilities. Write as if you are the technical expert who resolved the issue. You must output ONLY the article content. Do not include any preamble, introduction, or meta-commentary.

    DO NOT START WITH:
    - "Sure, I'd be happy to..."
    - "Here's an example..."
    - "As an AI..."
    - "I will create..."
    - Any greeting or acknowledgment

    START DIRECTLY WITH THE ARTICLE TITLE.

    TICKET INFORMATION:
    - Ticket ID: {ticket_id}
    - Summary: {summary}
    - Status: RESOLVED/DONE

    ACTUAL SOLUTION IMPLEMENTED:
    {actual_solution}

    SPECIFIC FIXES APPLIED:
    - Code Changes: {code_changes_str}
    - Configuration Changes: {config_changes_str}
    - Who Fixed It: {who_fixed}
    - When Fixed: {when_fixed}

    RESOLUTION STEPS TAKEN:
    {resolution_steps_str}

    ADDITIONAL CONTEXT FROM COMMENTS:{fix_details_section}

    PREVIOUS ARTICLE (Version {previous_version}):
    {previous_content}

    HUMAN FEEDBACK TO ADDRESS:
    {human_feedback or 'No specific feedback provided'}

    REFINEMENT INSTRUCTIONS:
    1. Read the human feedback carefully
    2. Ensure the article documents the ACTUAL SOLUTION from the comments
    3. Incorporate ALL requested changes from the feedback
    4. Maintain the good parts of the previous version
    5. Make sure technical details from the actual fix are included

    Generate a COMPLETE article with these sections:
    1. **Problem Overview** - What issue was encountered
    2. **Root Cause Analysis** - What caused this issue (from comments if available)
    3. **Solution Implementation** - EXACTLY what was done to fix it (USE THE ACTUAL SOLUTION DATA!)
    4. **Technical Details** - Include code/config changes from comments
    5. **Verification Steps** - How the fix was verified
    6. **Business Impact** - Impact of this fix
    7. **Related Knowledge** - Similar issues that might benefit
    8. **Preventive Measures** - How to prevent this in future
    9. **Next Steps** - Follow-up actions or monitoring

    Write in professional Markdown format. This MUST document the ACTUAL RESOLUTION, not generic advice!"""
        
        if refinement_suggestion:
            prompt += f"\n\nADDITIONAL REFINEMENT REQUEST:\n{str(refinement_suggestion)}"
        
        return prompt

    def _build_comprehensive_prompt(self, ticket_id: str, enhanced_context: Dict[str, Any], 
                                   refinement_suggestion: str = None) -> str:
        """YOUR EXISTING METHOD - KEEPING ALL YOUR LOGIC"""
        target_ticket = enhanced_context.get("target_ticket")
        
        # If we don't have ticket data, log error
        if not target_ticket:
            self.log(f"[ERROR] No ticket data available for {ticket_id}")
            # Try to get it directly
            project_id = ticket_id.split('-')[0] if '-' in ticket_id else "PROJ123"
            self.log(f"[EMERGENCY] Attempting direct ticket retrieval for {ticket_id}")
            
            # Direct attempt to get ticket
            try:
                data_input = {"project_id": project_id}
                result = self.jira_data_agent.run(data_input)
                tickets = result.get("tickets", [])
                
                for ticket in tickets:
                    if ticket.get("key") == ticket_id:
                        target_ticket = ticket
                        enhanced_context["target_ticket"] = target_ticket
                        self.log(f"[EMERGENCY] Found ticket via direct retrieval!")
                        break
            except Exception as e:
                self.log(f"[ERROR] Direct retrieval failed: {e}")
        
        # Build prompt with ACTUAL ticket data
        if target_ticket:
            fields = target_ticket.get("fields", {})
            actual_summary = fields.get("summary", "No summary available")
            actual_description = fields.get("description", "No description available")
            actual_status = fields.get("status", {}).get("name", "Unknown")
            issue_type = fields.get("issuetype", {}).get("name", "Unknown")
            
            prompt = f"""You are an AI specialized in creating comprehensive, Confluence-ready knowledge articles.

    IMPORTANT: Generate an article based on the ACTUAL ticket data below. Do NOT make up generic content!

    ACTUAL TICKET INFORMATION:
    - Ticket ID: {ticket_id}
    - Summary: {actual_summary}
    - Description: {actual_description}
    - Type: {issue_type}
    - Current Status: {actual_status}

    Based on this SPECIFIC ticket about "{actual_summary}", create a comprehensive article with these sections:

    1. Problem Overview - Explain what "{actual_summary}" means and why it needs to be done
    2. Solution Implementation - Provide specific steps for implementing "{actual_summary}"
    3. Business Impact - How does completing "{actual_summary}" benefit the project?
    4. Related Knowledge - Similar tasks or patterns related to this work
    5. Strategic Recommendations - Best practices for this type of work
    6. Next Steps - What should be done after completing this task

    Write the article in professional Markdown format. Be SPECIFIC to the actual ticket - this is about "{actual_summary}", NOT about generic network issues!
    """
        else:
            # Fallback if no data
            self.log(f"[CRITICAL] No ticket data found - using fallback prompt")
            prompt = f"""Create a knowledge article for ticket {ticket_id}. 
    Note: Unable to retrieve specific ticket details. Please create a general template article 
    that can be updated later with specific information."""
        
        return prompt

    def _build_comprehensive_prompt_with_feedback(self, ticket_id: str, 
                                                enhanced_context: Dict[str, Any],
                                                refinement_suggestion: str = None,
                                                human_feedback: str = None,
                                                previous_article: Dict[str, Any] = None) -> str:
        """Build prompt that includes feedback context"""
        
        # Get base prompt from your existing method
        base_prompt = self._build_comprehensive_prompt(ticket_id, enhanced_context, refinement_suggestion)
        
        # Add feedback context if available
        if human_feedback and previous_article:
            # Extract the previous article content
            previous_content = previous_article.get("content", "")
            previous_version = previous_article.get("version", 1)
            
            # Build a comprehensive refinement prompt
            base_prompt = f"""You are an AI specialized in creating comprehensive, Confluence-ready knowledge articles.

    IMPORTANT: You are REFINING an existing article based on human feedback. Generate a COMPLETE NEW VERSION of the article that incorporates the feedback while maintaining all the good aspects of the previous version.

    TICKET INFORMATION:
    {base_prompt.split('TICKET INFORMATION:')[1].split('Based on this')[0]}

    PREVIOUS ARTICLE (Version {previous_version}):
    {previous_content}

    HUMAN FEEDBACK TO ADDRESS:
    {human_feedback}

    REFINEMENT INSTRUCTIONS:
    1. Read and understand the human feedback carefully
    2. Identify what specific improvements are requested
    3. Generate a COMPLETE NEW ARTICLE that:
    - Incorporates ALL the requested changes from the feedback
    - Maintains all the good content from the previous version
    - Enhances sections mentioned in the feedback
    - Keeps the same overall structure unless feedback suggests otherwise

    Generate the FULL article with these sections:
    1. Problem Overview - Enhanced based on feedback if mentioned
    2. Solution Implementation - Add specific details requested in feedback
    3. Business Impact - Update if feedback mentions this
    4. Related Knowledge - Expand if feedback requests more context
    5. Strategic Recommendations - Refine based on feedback
    6. Next Steps - Update if feedback suggests changes

    Write the COMPLETE article in professional Markdown format. Do NOT just append the feedback - regenerate the entire article with improvements."""
            
        elif refinement_suggestion:
            # Handle autonomous refinement suggestion (existing behavior)
            base_prompt += f"\n\nREFINEMENT SUGGESTION:\n{refinement_suggestion}\n\n"
            base_prompt += "Please refine the article based on this suggestion while maintaining all the good parts."
        
        return base_prompt
    
    def _create_article_with_metadata(self, ticket_id: str, content: str, 
                                     enhanced_context: Dict[str, Any],
                                     version: int = 1,
                                     human_feedback: str = None) -> Dict[str, Any]:
        """Enhanced version of your existing method with version tracking"""
        collaboration_metadata = enhanced_context.get("collaboration_metadata", {})
        collaboration_successful = enhanced_context.get("collaboration_successful", False)
        
        article = {
            "content": content,
            "status": "pending_approval" if version > 1 else "draft",
            "title": f"Know-How: {ticket_id} - Comprehensive Resolution Guide",
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "version": version,  # NEW
            "approval_status": "pending",  # NEW
            "feedback_history": [],  # NEW
            "collaboration_enhanced": collaboration_successful,
            "collaboration_metadata": collaboration_metadata,
            "context_sources": {
                "ticket_data": bool(enhanced_context.get("target_ticket")),
                "related_knowledge": bool(enhanced_context.get("related_articles")),
                "productivity_insights": False,
                "strategic_recommendations": False,
                "human_feedback": bool(human_feedback)  # NEW
            },
            "quality_indicators": {
                "comprehensive_context": collaboration_successful,
                "collaboration_applied": collaboration_successful,
                "multi_source_synthesis": len(collaboration_metadata.get("collaborating_agents", [])) >= 2,
                "human_reviewed": version > 1  # NEW
            }
        }
        
        # Add feedback to history if provided
        if human_feedback:
            article["feedback_history"].append({
                "version": version - 1,
                "feedback": human_feedback,
                "timestamp": datetime.now().isoformat(),
                "applied": True
            })
        
        return article

    def _create_fallback_article(self, ticket_id: str) -> Dict[str, Any]:
        """YOUR EXISTING METHOD - NO CHANGES"""
        return {
            "content": f"# Know-How: {ticket_id}\n\n## Problem\nTicket resolution documentation.\n\n## Solution\nSee ticket details for resolution steps.\n\n## Next Steps\nReview implementation and monitor for similar issues.",
            "status": "draft",
            "title": f"Know-How: {ticket_id}",
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "collaboration_enhanced": False,
            "collaboration_metadata": {},
            "context_sources": {
                "ticket_data": False,
                "related_knowledge": False,
                "productivity_insights": False,
                "strategic_recommendations": False
            },
            "quality_indicators": {
                "comprehensive_context": False,
                "collaboration_applied": False,
                "multi_source_synthesis": False
            },
            "fallback_generated": True
        }

    # NEW METHODS FOR VERSION MANAGEMENT
    def _store_article_version(self, ticket_id: str, version: int, article: Dict[str, Any]):
        """Store article version in Redis"""
        version_key = f"article_version:{ticket_id}:v{version}"
        self.redis_client.set(version_key, json.dumps(article))
        self.redis_client.expire(version_key, 86400 * 30)  # Keep for 30 days
        
        # Update latest version pointer
        latest_key = f"article_latest:{ticket_id}"
        self.redis_client.set(latest_key, version)

    def _get_previous_article_version(self, ticket_id: str, version: int) -> Optional[Dict[str, Any]]:
        """Retrieve a previous version of the article"""
        version_key = f"article_version:{ticket_id}:v{version}"
        article_json = self.redis_client.get(version_key)
        
        if article_json:
            return json.loads(article_json)
        return None

    def _act(self) -> Dict[str, Any]:
        """Generate article and return immediately - NO LOOPS"""
        try:
            project_id = self.mental_state.get_belief("current_project") or ""
            workflow_context = self.mental_state.get_belief("workflow_context")
            force_fresh = self.mental_state.get_belief("force_fresh")
            # Generate article
            article = self._generate_article()
            
            # CRITICAL: For article generation, always force fresh data
            is_article_generation = (
                workflow_context == "article_generation" or
                force_fresh == True
            )
            
            if is_article_generation:
                self.log("[ACTION] Article generation context - forcing fresh data load")
                # Clear cache to force fresh load
                tickets_key = f"tickets:{project_id}"
                self.redis_client.delete(tickets_key)
                
                # Also clear any filtered ticket caches
                pattern = f"filtered_tickets:{project_id}:*"
                for key in self.redis_client.scan_iter(match=pattern):
                    self.redis_client.delete(key)
            # Return immediately with success
            return {
                "article": article,
                "workflow_status": "success",
                "workflow_completed": True,  # Signal workflow completion
                "autonomous_refinement_done": False,  # Skip autonomous refinement to prevent loops
                "collaboration_metadata": article.get("collaboration_metadata", {})
            }
            
        except Exception as e:
            self.log(f"[ERROR] Article generation failed: {e}")
            return {
                "article": self._create_fallback_article(self.mental_state.beliefs.get("ticket_id", "UNKNOWN")),
                "workflow_status": "failure",
                "workflow_completed": True,  # Still mark as completed to prevent loops
                "error": str(e),
                "autonomous_refinement_done": False,
                "collaboration_applied": False
            }

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        """YOUR EXISTING METHOD - NO CHANGES"""
        super()._rethink(action_result)
        
        collaboration_applied = action_result.get("collaboration_applied", False)
        collaboration_metadata = action_result.get("collaboration_metadata", {})
        
        self.mental_state.beliefs["last_article"] = {
            "timestamp": datetime.now().isoformat(),
            "article_generated": bool(action_result.get("article")),
            "status": action_result.get("workflow_status"),
            "autonomous_refinement_done": action_result.get("autonomous_refinement_done", False),
            "collaboration_applied": collaboration_applied,
            "agents_collaborated_with": collaboration_metadata.get("collaborating_agents", []),
            "collaboration_success": collaboration_metadata.get("total_collaborations", 0) > 0
        }
        
        # Learn from collaboration outcomes
        if collaboration_applied:
            self.mental_state.add_experience(
                experience_description=f"Generated article with collaboration from {len(collaboration_metadata.get('collaborating_agents', []))} agents",
                outcome="collaborative_article_generation",
                confidence=0.9,
                metadata={
                    "collaboration_agents": collaboration_metadata.get("collaborating_agents", []),
                    "success": action_result.get("workflow_status") == "success"
                }
            )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point with feedback support"""
        # Check if this is a feedback refinement request
        if input_data.get("human_feedback") and input_data.get("article_version", 1) > 1:
            self.log("[WORKFLOW] Processing feedback refinement request")
        
        return self.process(input_data)
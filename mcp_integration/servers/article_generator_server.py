from mcp_integration.servers.base_mcp_server import BaseMCPServer
from typing import Dict, Any, List
import asyncio

class ArticleGeneratorMCPServer(BaseMCPServer):
    """MCP Server for Article Generator Agent"""
    
    def _register_tools(self):
        """Register article generator tools"""
        
        # Tool: Generate Article
        async def generate_article(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Generate article from ticket"""
            result = self.agent.run({
                "ticket_id": arguments.get("ticket_id"),
                "refinement_suggestion": arguments.get("refinement_suggestion")
            })
            
            return {
                "article": result.get("article", {}),
                "workflow_status": result.get("workflow_status", "success"),
                "collaboration_applied": result.get("collaboration_applied", False)
            }
        
        self.register_tool(
            name="generate_article",
            description="Generate knowledge article from resolved ticket",
            input_schema={
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string"},
                    "refinement_suggestion": {"type": "string"}
                },
                "required": ["ticket_id"]
            },
            handler=generate_article
        )
        
        # Tool: Refine Article
        async def refine_article(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Refine existing article"""
            current_article = arguments.get("article", {})
            refinement_type = arguments.get("refinement_type", "general")
            
            # Create refinement suggestion based on type
            suggestions = {
                "technical": "Add more technical implementation details, code examples, and performance metrics",
                "clarity": "Improve clarity by adding step-by-step instructions and visual diagrams",
                "completeness": "Add sections for prerequisites, troubleshooting, and related articles"
            }
            
            refinement = suggestions.get(refinement_type, "Enhance the article with more specific details")
            
            # Generate refined article
            result = await generate_article({
                "ticket_id": current_article.get("title", "").split(" - ")[0],  # Extract ticket ID
                "refinement_suggestion": refinement
            })
            
            return result
        
        self.register_tool(
            name="refine_article",
            description="Refine existing article based on feedback",
            input_schema={
                "type": "object",
                "properties": {
                    "article": {"type": "object"},
                    "refinement_type": {"type": "string", "enum": ["technical", "clarity", "completeness", "general"]}
                },
                "required": ["article"]
            },
            handler=refine_article
        )
    
    def _register_resources(self):
        """Register article generator resources"""
        
        # Resource: Article Templates
        self.register_resource(
            uri="articles://templates/knowledge",
            name="Knowledge Article Templates",
            description="Templates for different types of knowledge articles",
            content={
                "bug_resolution": """# {ticket_id} - {title}

## Problem Description
{problem_description}

## Root Cause Analysis
{root_cause}

## Resolution Steps
1. {step1}
2. {step2}
3. {step3}

## Verification
{verification_steps}

## Prevention
{prevention_measures}

## Related Articles
{related_articles}
""",
                "feature_implementation": """# {ticket_id} - {title}

## Overview
{overview}

## Implementation Details
{implementation}

## Configuration
{configuration}

## Testing
{testing_approach}

## Best Practices
{best_practices}

## References
{references}
""",
                "troubleshooting_guide": """# {ticket_id} - Troubleshooting Guide

## Symptoms
{symptoms}

## Diagnostic Steps
{diagnostics}

## Common Causes
{causes}

## Solutions
{solutions}

## Escalation Path
{escalation}

## Additional Resources
{resources}
"""
            }
        )
        
        # Resource: Quality Checklist
        self.register_resource(
            uri="articles://quality/checklist",
            name="Article Quality Checklist",
            description="Quality criteria for knowledge articles",
            content={
                "structure": [
                    "Clear title with ticket reference",
                    "Problem/overview section",
                    "Detailed solution/implementation",
                    "Verification/testing steps",
                    "Related resources"
                ],
                "content": [
                    "Technical accuracy",
                    "Completeness",
                    "Clarity and readability",
                    "Actionable instructions",
                    "Proper formatting"
                ],
                "metadata": [
                    "Correct categorization",
                    "Relevant tags",
                    "Author information",
                    "Last updated date"
                ]
            }
        )
    
    def _get_agent_prompts(self) -> List[Dict[str, Any]]:
        """Get article generator prompts"""
        return [
            {
                "name": "generate_kb_article",
                "description": "Generate knowledge base article",
                "template": "Create a comprehensive knowledge base article for ticket {ticket_id} about {issue_type}. Include problem description, root cause, solution steps, and prevention measures.",
                "arguments": [
                    {"name": "ticket_id", "description": "Jira ticket ID", "required": True},
                    {"name": "issue_type", "description": "Type of issue", "required": True}]
            },
            {
                "name": "troubleshooting_article",
                "description": "Generate troubleshooting article",
                "template": "Create a troubleshooting guide for {error_description} in {component}. Include symptoms, diagnostic steps, common causes, and solutions.",
                "arguments": [
                    {"name": "error_description", "description": "Description of the error", "required": True},
                    {"name": "component", "description": "Affected component", "required": True}
                ]
            }
        ]
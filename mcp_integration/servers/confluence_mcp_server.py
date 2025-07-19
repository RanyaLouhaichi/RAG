import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from mcp.server import Server
from mcp.types import Tool, TextContent

class ConfluenceMCPServer:
    """MCP Server that exposes Confluence data and operations"""
    
    def __init__(self):
        self.server = Server("confluence_mcp_server")
        self.logger = logging.getLogger("ConfluenceMCPServer")
        
        # For now, we'll simulate Confluence data
        # In production, you'd use the actual Confluence API
        self.mock_articles = self._load_mock_articles()
        
        self._setup_handlers()
    
    def _load_mock_articles(self) -> List[Dict[str, Any]]:
        """Load mock Confluence articles"""
        return [
            {
                "id": "kb001",
                "title": "Agile Best Practices Guide",
                "content": "This guide covers Agile methodology best practices including sprint planning, daily standups, retrospectives, and continuous improvement strategies.",
                "space": "AGILE",
                "tags": ["agile", "scrum", "best-practices"],
                "created": "2024-01-15",
                "updated": "2024-12-20"
            },
            {
                "id": "kb002",
                "title": "Kubernetes Deployment Guide",
                "content": "Step-by-step guide for deploying applications to Kubernetes, including pod configuration, service exposure, and scaling strategies.",
                "space": "TECH",
                "tags": ["kubernetes", "deployment", "devops"],
                "created": "2024-02-20",
                "updated": "2024-12-15"
            },
            {
                "id": "kb003",
                "title": "CI/CD Pipeline Configuration",
                "content": "Complete guide for setting up CI/CD pipelines using Jenkins, GitLab CI, and GitHub Actions. Includes best practices for testing and deployment automation.",
                "space": "DEVOPS",
                "tags": ["ci-cd", "automation", "devops"],
                "created": "2024-03-10",
                "updated": "2024-12-10"
            }
        ]
    
    def _setup_handlers(self):
        """Setup MCP protocol handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List all available Confluence tools"""
            return [
                Tool(
                    name="search_articles",
                    description="Search Confluence articles by keywords",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "space": {
                                "type": "string",
                                "description": "Confluence space to search in (optional)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_article",
                    description="Get a specific Confluence article by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "article_id": {
                                "type": "string",
                                "description": "Article ID"
                            }
                        },
                        "required": ["article_id"]
                    }
                ),
                Tool(
                    name="create_article",
                    description="Create a new Confluence article",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Article title"
                            },
                            "content": {
                                "type": "string",
                                "description": "Article content (Markdown supported)"
                            },
                            "space": {
                                "type": "string",
                                "description": "Confluence space"
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Article tags"
                            }
                        },
                        "required": ["title", "content", "space"]
                    }
                ),
                Tool(
                    name="update_article",
                    description="Update an existing Confluence article",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "article_id": {
                                "type": "string",
                                "description": "Article ID"
                            },
                            "title": {
                                "type": "string",
                                "description": "New title (optional)"
                            },
                            "content": {
                                "type": "string",
                                "description": "New content (optional)"
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "New tags (optional)"
                            }
                        },
                        "required": ["article_id"]
                    }
                ),
                Tool(
                    name="get_related_articles",
                    description="Get articles related to a specific topic or article",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "article_id": {
                                "type": "string",
                                "description": "Base article ID (optional)"
                            },
                            "topic": {
                                "type": "string",
                                "description": "Topic to find related articles for (optional)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 5)"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_article_templates",
                    description="Get available article templates",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "template_type": {
                                "type": "string",
                                "description": "Type of template (optional)",
                                "enum": ["bug_fix", "feature", "how_to", "troubleshooting", "best_practices"]
                            }
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Execute a Confluence tool"""
            try:
                result = None
                
                if name == "search_articles":
                    query = arguments["query"].lower()
                    space = arguments.get("space")
                    limit = arguments.get("limit", 10)
                    
                    # Search mock articles
                    results = []
                    for article in self.mock_articles:
                        if space and article["space"] != space:
                            continue
                        
                        # Simple keyword search
                        if (query in article["title"].lower() or 
                            query in article["content"].lower() or
                            any(query in tag for tag in article["tags"])):
                            results.append(article)
                    
                    result = {
                        "query": arguments["query"],
                        "results": results[:limit],
                        "total": len(results)
                    }
                
                elif name == "get_article":
                    article_id = arguments["article_id"]
                    article = next((a for a in self.mock_articles if a["id"] == article_id), None)
                    
                    if article:
                        result = article
                    else:
                        result = {"error": f"Article {article_id} not found"}
                
                elif name == "create_article":
                    # In production, this would create an actual Confluence page
                    new_article = {
                        "id": f"kb{len(self.mock_articles) + 1:03d}",
                        "title": arguments["title"],
                        "content": arguments["content"],
                        "space": arguments["space"],
                        "tags": arguments.get("tags", []),
                        "created": datetime.now().strftime("%Y-%m-%d"),
                        "updated": datetime.now().strftime("%Y-%m-%d")
                    }
                    
                    self.mock_articles.append(new_article)
                    result = {
                        "status": "created",
                        "article": new_article
                    }
                
                elif name == "update_article":
                    article_id = arguments["article_id"]
                    article = next((a for a in self.mock_articles if a["id"] == article_id), None)
                    
                    if article:
                        if "title" in arguments:
                            article["title"] = arguments["title"]
                        if "content" in arguments:
                            article["content"] = arguments["content"]
                        if "tags" in arguments:
                            article["tags"] = arguments["tags"]
                        
                        article["updated"] = datetime.now().strftime("%Y-%m-%d")
                        
                        result = {
                            "status": "updated",
                            "article": article
                        }
                    else:
                        result = {"error": f"Article {article_id} not found"}
                
                elif name == "get_related_articles":
                    base_article_id = arguments.get("article_id")
                    topic = arguments.get("topic", "").lower()
                    limit = arguments.get("limit", 5)
                    
                    related = []
                    
                    if base_article_id:
                        base_article = next((a for a in self.mock_articles if a["id"] == base_article_id), None)
                        if base_article:
                            # Find articles with similar tags
                            for article in self.mock_articles:
                                if article["id"] != base_article_id:
                                    common_tags = set(article["tags"]) & set(base_article["tags"])
                                    if common_tags:
                                        related.append({
                                            "article": article,
                                            "relevance": len(common_tags) / len(base_article["tags"])
                                        })
                    
                    elif topic:
                        # Find articles related to topic
                        for article in self.mock_articles:
                            if (topic in article["title"].lower() or
                                topic in article["content"].lower() or
                                any(topic in tag for tag in article["tags"])):
                                related.append({
                                    "article": article,
                                    "relevance": 0.8
                                })
                    
                    # Sort by relevance
                    related.sort(key=lambda x: x["relevance"], reverse=True)
                    
                    result = {
                        "related_articles": [r["article"] for r in related[:limit]],
                        "count": len(related)
                    }
                
                elif name == "get_article_templates":
                    template_type = arguments.get("template_type")
                    
                    templates = {
                        "bug_fix": {
                            "title": "Bug Fix - {TICKET_ID}",
                            "content": """## Problem Description
{PROBLEM_DESCRIPTION}

## Root Cause
{ROOT_CAUSE}

## Solution
{SOLUTION_STEPS}

## Testing
{TESTING_STEPS}

## Impact
{IMPACT_ANALYSIS}
"""
                        },
                        "how_to": {
                            "title": "How to {TASK}",
                            "content": """## Overview
{OVERVIEW}

## Prerequisites
{PREREQUISITES}

## Steps
1. {STEP_1}
2. {STEP_2}
3. {STEP_3}

## Verification
{VERIFICATION_STEPS}

## Troubleshooting
{COMMON_ISSUES}
"""
                        },
                        "best_practices": {
                            "title": "{TOPIC} Best Practices",
                            "content": """## Introduction
{INTRODUCTION}

## Best Practices
### 1. {PRACTICE_1}
{PRACTICE_1_DETAILS}

### 2. {PRACTICE_2}
{PRACTICE_2_DETAILS}

## Common Pitfalls
{PITFALLS}

## Examples
{EXAMPLES}

## References
{REFERENCES}
"""
                        }
                    }
                    
                    if template_type:
                        result = templates.get(template_type, {"error": "Template not found"})
                    else:
                        result = {
                            "available_templates": list(templates.keys()),
                            "templates": templates
                        }
                
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
                
            except Exception as e:
                self.logger.error(f"Tool execution failed: {name} - {str(e)}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.list_resources()
        async def handle_list_resources() -> list[dict]:
            """List available Confluence resources"""
            return [
                {
                    "uri": "confluence://spaces/list",
                    "name": "Confluence Spaces",
                    "description": "List of available Confluence spaces",
                    "mimeType": "application/json"
                },
                {
                    "uri": "confluence://templates/article",
                    "name": "Article Templates",
                    "description": "Pre-defined article templates",
                    "mimeType": "application/json"
                },
                {
                    "uri": "confluence://guidelines/writing",
                    "name": "Writing Guidelines",
                    "description": "Confluence writing style guidelines",
                    "mimeType": "application/json"
                }
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read a Confluence resource"""
            if uri == "confluence://spaces/list":
                return json.dumps({
                    "spaces": [
                        {"key": "AGILE", "name": "Agile Practices", "description": "Agile methodology and practices"},
                        {"key": "TECH", "name": "Technical Docs", "description": "Technical documentation"},
                        {"key": "DEVOPS", "name": "DevOps", "description": "DevOps practices and tools"},
                        {"key": "KB", "name": "Knowledge Base", "description": "General knowledge base"}
                    ]
                }, indent=2)
            
            elif uri == "confluence://templates/article":
                return json.dumps({
                    "templates": {
                        "standard": "# {TITLE}\n\n## Overview\n{OVERVIEW}\n\n## Details\n{DETAILS}\n\n## References\n{REFERENCES}",
                        "tutorial": "# {TITLE}\n\n## What You'll Learn\n{OBJECTIVES}\n\n## Prerequisites\n{PREREQUISITES}\n\n## Steps\n{STEPS}\n\n## Summary\n{SUMMARY}",
                        "troubleshooting": "# {TITLE}\n\n## Problem\n{PROBLEM}\n\n## Symptoms\n{SYMPTOMS}\n\n## Solution\n{SOLUTION}\n\n## Prevention\n{PREVENTION}"
                    }
                }, indent=2)
            
            elif uri == "confluence://guidelines/writing":
                return json.dumps({
                    "guidelines": {
                        "tone": "Professional yet approachable",
                        "structure": "Use clear headings and sections",
                        "formatting": {
                            "headings": "Use markdown # for main sections, ## for subsections",
                            "lists": "Use bullet points for unordered lists, numbers for steps",
                            "code": "Use code blocks for technical content",
                            "emphasis": "Use **bold** for important points"
                        },
                        "best_practices": [
                            "Start with an overview",
                            "Include practical examples",
                            "Add troubleshooting sections",
                            "Reference related articles",
                            "Keep content up to date"
                        ]
                    }
                }, indent=2)
            
            else:
                raise ValueError(f"Unknown resource: {uri}")
    
    async def start(self):
        """Start the MCP server"""
        self.logger.info("Starting Confluence MCP Server")
        async with self.server.run_stdio():
            await asyncio.Event().wait()
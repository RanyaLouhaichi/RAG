# Jira MCP Server

## Key Responsibilities:

- **Jira Connection:** Initializes and maintains a connection to the Jira API.
- **Tool Registration:** Exposes tools for interacting with Jira, such as fetching projects, tickets, and performing searches.
- **Resource Registration:** Provides access to Jira-related resources like status mappings, JQL templates, and field definitions.
- **Prompt Registration:** Offers prompts for common analysis tasks like sprint analysis and team workload assessment.

## Registered Tools:

- **`get_projects`**: Retrieves all available Jira projects.
- **`get_project_tickets`**: Fetches all tickets for a specific project, with optional date filtering.
- **`get_ticket`**: Gets the details of a single Jira ticket.
- **`search_tickets`**: Searches for tickets using a JQL query.
- **`analyze_project_metrics`**: Analyzes project metrics like velocity and cycle time over a specified period.

## Registered Resources:

- **`jira://config/status_mappings`**: Provides mappings for Jira status categories (e.g., "To Do", "In Progress", "Done").
- **`jira://templates/jql_queries`**: Offers common JQL query templates for reuse.
- **`jira://metadata/field_definitions`**: Exposes standard and custom Jira field definitions.

## Registered Prompts:

- **`sprint_analysis`**: A prompt to analyze the progress of the current sprint for a project.
- **`team_workload`**: A prompt to analyze the workload distribution among team members.

## Usage:

This server is run as a separate process and communicates over standard I/O. Other agents can connect to it using an MCP client to access Jira data and functionality without needing to directly implement the Jira API client themselves. This modular approach encapsulates Jira interactions and provides a standardized interface for the multi-agent system.

# Jira Data Agent MCP Server

The `JiraDataAgentMCPServer` is an MCP (Multi-Agent Communication Protocol) server that exposes the capabilities of the `JiraDataAgent`. It provides tools and resources for retrieving, analyzing, and monitoring Jira data.

## Key Responsibilities:

- **Tool Registration:** Registers tools for retrieving Jira tickets, analyzing project data, and monitoring changes.
- **Resource Registration:** Provides access to available Jira projects, a summary of cached data, and agent performance metrics.
- **Prompt Registration:** Exposes prompts for generating project summaries and analyzing sprint data.

## Registered Tools:

- **`retrieve_tickets`**:
    - **Description:** Retrieves and analyzes Jira tickets for a given project.
    - **Input:** `project_id` (string), optional `time_range` (object), and `analysis_depth` (enum: "basic", "enhanced").
    - **Output:** A dictionary containing `tickets`, `metadata`, `ticket_count`, and `workflow_status`.

- **`analyze_project`**:
    - **Description:** Analyzes project data patterns to identify potential issues.
    - **Input:** `project_id` (string).
    - **Output:** A dictionary with `project_id`, `total_tickets`, `status_distribution`, `assignee_workload`, and a `has_bottlenecks` flag.

- **`monitor_changes`**:
    - **Description:** Monitors a Jira project for any data changes.
    - **Input:** `project_id` (string).
    - **Output:** A dictionary indicating if there have been changes, along with the `project_id`, `ticket_count`, and `last_check` timestamp.

## Registered Resources:

- **`jira://projects/available`**:
    - **Name:** Available Projects
    - **Description:** Provides a list of available Jira projects.

- **`jira://cache/summary`**:
    - **Name:** Cache Summary
    - **Description:** Offers a summary of cached Jira data stored in Redis.

- **`jira://metrics/performance`**:
    - **Name:** Performance Metrics
    - **Description:** Exposes the performance metrics of the `JiraDataAgent`.

## Registered Prompts:

- **`project_summary`**: A prompt to generate a comprehensive summary of a Jira project.
- **`sprint_data_analysis`**: A prompt to analyze sprint data for a project within a specified date range.

## Usage:

This server acts as the interface for the `JiraDataAgent`, allowing other agents to access and analyze Jira data. It is a critical component for any agent that needs to reason about or make decisions based on project management data.

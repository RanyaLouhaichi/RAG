# Dashboard Agent MCP Server

The `DashboardAgentMCPServer` is an MCP (Multi-Agent Communication Protocol) server that exposes the capabilities of the `ProductivityDashboardAgent`. It allows other agents to generate productivity dashboards, analyze metrics, and create visualizations.

## Key Responsibilities:

- **Tool Registration:** Registers tools for generating dashboards, analyzing metrics, creating visualizations, and exporting dashboard data.
- **Resource Registration:** Provides access to chart templates, report templates, and performance benchmarks.
- **Prompt Registration:** Exposes prompts for generating executive-level dashboards and team performance analytics.

## Registered Tools:

- **`generate_dashboard`**:
    - **Description:** Generates a comprehensive productivity dashboard.
    - **Input:** `project_id` (string), and optional context like `tickets`, `recommendations`, and `predictions`.
    - **Output:** A dictionary containing `metrics`, `visualization_data`, a `report`, and `workflow_status`.

- **`analyze_metrics`**:
    - **Description:** Analyzes productivity metrics from ticket data.
    - **Input:** `tickets` (array).
    - **Output:** A dictionary with `metrics`, `insights`, and `analysis_confidence`.

- **`create_visualization`**:
    - **Description:** Creates a custom visualization.
    - **Input:** `type` (enum), `data` (object), and optional `title` and `options`.
    - **Output:** A dictionary representing the visualization structure.

- **`export_dashboard`**:
    - **Description:** Exports dashboard data in various formats.
    - **Input:** `dashboard_id` (string) and `format` (enum: "json", "summary").
    - **Output:** The exported dashboard data.

## Registered Resources:

- **`dashboard://templates/charts`**:
    - **Name:** Chart Templates
    - **Description:** Provides pre-configured templates for common charts like velocity trends and workload distribution.

- **`dashboard://templates/reports`**:
    - **Name:** Report Templates
    - **Description:** Contains templates for generating reports such as executive summaries and team performance reviews.

- **`dashboard://benchmarks/performance`**:
    - **Name:** Performance Benchmarks
    - **Description:** Provides industry benchmarks for key performance metrics like cycle time and throughput.

## Registered Prompts:

- **`executive_dashboard`**: A prompt to generate a high-level executive dashboard summary.
- **`team_analytics`**: A prompt to generate detailed analytics on team performance.

## Usage:

This server enables other agents to leverage the `ProductivityDashboardAgent`'s capabilities for data visualization and reporting. It plays a key role in providing actionable insights into team and project performance.

# API Simple Documentation

## Overview

The `api_simple.py` file contains the main API for the JURIX AI system. It provides endpoints for interacting with the system, generating articles, retrieving dashboard data, and more. The API is built using the Flask web framework and integrates with a Jira instance.

## Important Endpoints

Here are some of the most important endpoints provided by the API:

### Article Generation

- **POST /api/article/generate/<ticket_id>**: Generates an initial article for a resolved Jira ticket.
- **POST /api/article/feedback/<ticket_id>**: Submits feedback for a generated article, which can be used for refinement, approval, or rejection.
- **GET /api/article/status/<ticket_id>**: Retrieves the current status and version of an article.
- **GET /api/article/history/<ticket_id>**: Gets the version history and feedback for an article.

### Dashboard and Analytics

- **GET /api/dashboard/<project_id>**: Retrieves comprehensive data for a project dashboard, including metrics, predictions, visualizations, and AI-powered insights.
- **GET /api/sprint-health/<project_key>**: Provides detailed sprint health metrics, including a "pulse" to indicate the sprint's status.
- **GET /api/risk-assessment/<project_key>**: Returns a real-time risk assessment for a project, based on AI analysis.
- **POST /api/what-if/<project_key>**: Allows for "what-if" scenario analysis to see the potential impact of changes.
- **GET /api/team-analytics/<project_key>**: Provides detailed analytics on team performance.
- **POST /api/historical-patterns/<project_key>**: Analyzes historical data to identify patterns and trends.

### Chat and Suggestions

- **POST /api/chat**: A chat endpoint for interacting with the JURIX AI system.
- **POST /api/suggest-articles**: Provides smart article suggestions for a given Jira issue.

### Project and Health

- **GET /api/projects**: Retrieves a list of available Jira projects.
- **GET /health**: A health check endpoint to verify that the API is running and connected to Jira.
- **POST /api/refresh**: Refreshes the Jira connection and project list.

## Other Features

The API also includes several other features, such as:

- **Real-time updates**: The API uses a tracking mechanism to provide real-time updates for dashboards.
- **Debugging endpoints**: Several endpoints are available for debugging and inspecting the system's state.
- **Thesis metrics**: An endpoint is included to aggregate system metrics for a thesis presentation.
- **LangSmith integration**: The API can be integrated with LangSmith for monitoring and testing.

This documentation provides a high-level overview of the `api_simple.py` file. For more detailed information, please refer to the source code.

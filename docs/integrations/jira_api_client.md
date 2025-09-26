# Jira API Client Documentation

## Overview

The `JiraAPIClient` class is a client for interacting with the Jira API. It provides methods for performing common Jira operations, such as retrieving projects, searching for issues, and creating issues. The client handles authentication, error handling, and pagination, which makes it easy to use in other parts of the application.

## `JiraAPIClient` Class

### Initialization

The client is initialized with the Jira URL and authentication credentials from the `APIConfig` class. It also sets up a `requests.Session` with a retry strategy for handling API errors.

### Methods

- **`test_connection()`**: Tests the connection to the Jira API.

- **`get_projects()`**: Retrieves a list of all projects accessible to the user.

- **`get_project(project_key: str)`**: Retrieves the details of a specific project.

- **`search_issues(jql: str, start_at: int = 0, max_results: int = None)`**: Searches for issues using a JQL query.

- **`get_issue(issue_key: str)`**: Retrieves the details of a specific issue.

- **`get_all_issues_for_project(project_key: str, start_date: str = None, end_date: str = None)`**: Retrieves all issues for a given project, with optional date filtering.

- **`create_issue(project_key: str, issue_type: str, summary: str, description: str = None)`**: Creates a new issue in a project.

- **`add_comment(issue_key: str, comment: str)`**: Adds a comment to an issue.

## Usage

The `JiraAPIClient` can be used to interact with the Jira API from anywhere in the application. For example:

```python
from integrations.jira_api_client import JiraAPIClient

client = JiraAPIClient()
projects = client.get_projects()
```

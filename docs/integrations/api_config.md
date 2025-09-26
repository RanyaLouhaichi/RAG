# API Config Documentation

## Overview

The `APIConfig` class is used to manage the configuration for the Jira and Confluence APIs. It loads API credentials and other configuration settings from environment variables, and provides a centralized place to access these settings.

## `APIConfig` Class

### Attributes

- **`JIRA_URL`**: The URL of the Jira instance.
- **`JIRA_USERNAME`**: The username for authenticating with the Jira API.
- **`JIRA_PASSWORD`**: The password for authenticating with the Jira API.
- **`CONFLUENCE_URL`**: The URL of the Confluence instance.
- **`CONFLUENCE_USERNAME`**: The username for authenticating with the Confluence API.
- **`CONFLUENCE_PASSWORD`**: The password for authenticating with the Confluence API.
- **`API_TIMEOUT`**: The timeout for API requests in seconds.
- **`API_MAX_RETRIES`**: The maximum number of retries for failed API requests.
- **`API_PAGE_SIZE`**: The number of items to request per page for paginated API responses.
- **`ENVIRONMENT`**: The environment in which the application is running (e.g., "development", "production").
- **`DEBUG`**: A boolean flag to enable or disable debug mode.
- **`VERIFY_SSL`**: A boolean flag to enable or disable SSL verification.

### Methods

- **`get_jira_auth()`**: Returns a tuple containing the Jira username and password.

- **`get_confluence_auth()`**: Returns a tuple containing the Confluence username and password.

- **`log_config()`**: Prints the current API configuration to the console, while hiding sensitive data.

- **`normalize_status(status: str)`**: Normalizes a given Jira status string into one of the following standard categories: "Done", "In Progress", "To Do", or "Blocked".

## Usage

The `APIConfig` class can be used to access API configuration settings from anywhere in the application. For example:

```python
from integrations.api_config import APIConfig

jira_url = APIConfig.JIRA_URL
```

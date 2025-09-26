# Ingest Confluence Documentation

## Overview

The `ingest_confluence.py` script is a command-line tool that allows you to populate the knowledge graph with data from Confluence and Jira. It uses the `EnhancedRAGPipeline` to process the data and store it in the graph.

## Functions

### `ingest_confluence_space(space_key: str, limit: int = 100)`

This function ingests a Confluence space into the RAG system. It initializes an `EnhancedRAGPipeline` and then calls the `ingest_confluence_space` method to perform the ingestion.

- **space_key**: The key of the Confluence space to ingest.
- **limit**: The maximum number of documents to ingest from the space.

### `sync_jira_tickets(project_key: str)`

This function syncs Jira tickets for a given project to the knowledge graph. It uses the `JiraDataAgent` to retrieve the tickets and then adds them to the graph using the `EnhancedRAGPipeline`.

- **project_key**: The key of the Jira project to sync.

### `main()`

The main function of the script, which parses command-line arguments and calls the appropriate functions to perform the ingestion. It supports the following arguments:

- **--space**: The Confluence space key to ingest.
- **--project**: The Jira project key to sync.
- **--limit**: The maximum number of documents to ingest.
- **--all-spaces**: A flag to ingest all available Confluence spaces.

## Usage

To use the script, you can run it from the command line with the desired arguments. For example:

```bash
python ingest_confluence.py --space MYSPACE --limit 200
```

This will ingest the first 200 documents from the "MYSPACE" Confluence space.

```bash
python ingest_confluence.py --project MYPROJECT
```

This will sync all tickets from the "MYPROJECT" Jira project.

```bash
python ingest_confluence.py --all-spaces
```

This will ingest all available Confluence spaces.

# Neo4jManager: Knowledge Graph Orchestration

## Overview

`Neo4jManager` is the core component responsible for managing the JURIX knowledge graph in Neo4j. It provides a high-level API for creating, updating, and querying documents, tickets, projects, and their relationships, enabling advanced graph-based reasoning and hybrid retrieval.

## Graph Schema

- **Nodes:**
  - `Document`: Represents knowledge articles, Confluence pages, or generated content.
  - `Ticket`: Represents Jira issues/tickets.
  - `Project`: Represents a Jira project or workspace.
  - `Chunk`: Represents semantic chunks of documents for fine-grained retrieval.

- **Relationships:**
  - `HAS_CHUNK`: Links a Document to its semantic Chunks.
  - `BELONGS_TO`: Links a Ticket to its Project.
  - `REFERENCES`, `RESOLVES`, `MENTIONS`: Link Documents to Tickets, capturing semantic relationships and resolution evidence.

## Key Features

- **Schema Initialization:** Automatically creates indexes and constraints for fast lookup and data integrity.
- **Document & Chunk Ingestion:** Adds documents and their semantic chunks, storing embeddings and metadata for hybrid search.
- **Ticket Synchronization:** Ingests Jira tickets, linking them to projects and related documents.
- **Relationship Management:** Supports dynamic creation and updating of semantic relationships, including confidence scores and feedback-driven adjustments.
- **Graph-Based Retrieval:** Enables advanced queries for related documents, solution patterns, and impact scoring.
- **Feedback Integration:** Updates relationship strengths based on user feedback, supporting incremental learning.

## Example Usage

```python
neo4j = Neo4jManager(uri, username, password)
neo4j.add_document(doc_id, title, content, metadata)
neo4j.add_ticket(ticket_key, summary, project_key, status, metadata)
neo4j.link_document_to_ticket(doc_id, ticket_key, relationship_type="REFERENCES")
related_docs = neo4j.find_related_documents(ticket_key)
```

## Advanced Capabilities

- **Solution Pattern Mining:** Finds common resolution patterns for similar tickets within a project.
- **Impact Scoring:** Quantifies the influence of documents based on their resolution and reference relationships.
- **Metadata Enrichment:** Stores rich metadata (JSON) for all nodes and relationships, enabling flexible analytics.

## Integration

Neo4jManager is used by the Enhanced RAG pipeline, productivity workflows, and all agents requiring graph-based context or retrieval.

---

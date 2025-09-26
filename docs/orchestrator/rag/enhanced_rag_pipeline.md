# EnhancedRAGPipeline: Multi-Level Retrieval-Augmented Generation

## Overview

`EnhancedRAGPipeline` is the advanced RAG engine in JURIX, combining multi-level embeddings, semantic chunking, knowledge graph integration, and incremental learning for state-of-the-art retrieval and recommendation.

## Architecture

- **Multi-Level Embeddings:** Utilizes both document-level and chunk-level embeddings for hybrid semantic search.
- **Semantic Chunking:** Leverages LLM-based chunking and quality evaluation for fine-grained retrieval and ranking.
- **Knowledge Graph Integration:** Connects to Neo4j for graph-based reasoning, relationship mining, and impact scoring.
- **Incremental Learning:** Incorporates user feedback and workflow outcomes to adapt and improve retrieval quality over time.

## Key Features

- **Confluence Ingestion:** Extracts, chunks, and indexes Confluence documents with rich metadata and quality scores.
- **Hybrid Search:** Combines vector similarity (document/chunk) with graph-based relevance for robust recommendations.
- **Graph-Based Re-Ranking:** Boosts results based on knowledge graph relationships, solution patterns, and impact scores.
- **Feedback-Driven Learning:** Updates relationship strengths and retrieval models based on user feedback and ticket resolutions.
- **Article Publishing:** Supports publishing generated articles to Confluence and linking them in the knowledge graph.

## Example Usage

```python
pipeline = EnhancedRAGPipeline(...)
pipeline.ingest_confluence_space("SPACEKEY")
results = pipeline.hybrid_search(query="How to resolve timeout errors?", ticket_context={"ticket_key": "MG-123"})
pipeline.publish_article_to_confluence(article, ticket_id="MG-123", project_key="MG")
```

## Advanced Capabilities

- **Quality-Aware Ranking:** Adjusts chunk relevance based on LLM-evaluated quality scores.
- **Solution Pattern Mining:** Identifies and boosts documents that have resolved similar tickets.
- **Impact Scoring:** Quantifies document influence based on graph relationships and ticket outcomes.
- **Semantic Feedback Loop:** Integrates user feedback into both the knowledge graph and retrieval ranking.

## Integration

Used by the enhanced retrieval agent, orchestrator workflows, and dashboard endpoints for all advanced retrieval, recommendation, and knowledge management tasks.

---

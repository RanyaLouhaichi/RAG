# SemanticChunkerWithLLMJudge: LLM-Powered Semantic Chunking

## Overview

`SemanticChunkerWithLLMJudge` is the advanced semantic chunking engine in JURIX, leveraging LLMs to segment documents into high-quality, semantically coherent chunks and evaluate their quality for downstream retrieval and ranking.

## Key Features

- **LLM-Based Chunking:** Uses large language models to identify logical sections, topics, and boundaries within documents.
- **Quality Evaluation:** Each chunk is scored and annotated with quality metrics (clarity, completeness, relevance) using LLM-based evaluation.
- **Metadata Enrichment:** Chunks are enriched with section labels, quality issues, and evaluation metadata for advanced filtering and ranking.
- **Integration with RAG:** Chunks are indexed with embeddings and quality scores, enabling quality-aware retrieval in the EnhancedRAGPipeline.

## Example Usage

```python
chunker = SemanticChunkerWithLLMJudge()
chunks = chunker.chunk_confluence_document(document)
for chunk in chunks:
    print(chunk.content, chunk.quality_score, chunk.metadata)
```

## Advanced Capabilities

- **Section Detection:** Identifies document structure (headings, sections) for context-aware chunking.
- **Issue Detection:** Flags chunks with potential issues (ambiguity, incompleteness) for downstream filtering.
- **Adaptive Chunk Sizing:** Dynamically adjusts chunk size based on semantic boundaries and content density.

## Integration

Used by the EnhancedRAGPipeline for all document ingestion, chunk-level retrieval, and quality-aware ranking.

---

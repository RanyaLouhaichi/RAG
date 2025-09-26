# Enhanced Retrieval Agent Documentation


## Overview

The `EnhancedRetrievalAgent` is a specialized agent responsible for retrieving and ranking information from various sources, including a Neo4j knowledge graph. It uses an `EnhancedRAGPipeline` to perform hybrid searches, combining vector search, keyword search, and graph-based search to find the most relevant information for a given query.

## `EnhancedRetrievalAgent` Class

### Initialization

The agent is initialized with a `JurixSharedMemory` instance and an `EnhancedRAGPipeline`. The `EnhancedRAGPipeline` is the core component that provides the retrieval and ranking capabilities.

### Capabilities

- `RETRIEVE_DATA`: Can retrieve data from various sources.
- `RANK_CONTENT`: Can rank the retrieved content based on its relevance to the query.
- `COORDINATE_AGENTS`: Can coordinate with other agents to fulfill a user's request.

### Cognitive Cycle

The `EnhancedRetrievalAgent` follows the same cognitive cycle as the `BaseAgent`:

- **`_perceive(input_data)`**: In this phase, the agent processes the user's input, including the prompt and any collaboration context. It also extracts relevant information, such as ticket keys, from the prompt.

- **`_act()`**: This is the main phase of the agent's operation. It orchestrates the retrieval process by calling the `_retrieve_articles` method to get the articles and then `_assess_retrieval_quality` to evaluate the results. It can also handle collaboration and feedback to improve the retrieval process.

- **`_rethink(action_result)`**: In this phase, the agent reflects on the outcome of the retrieval process and updates its mental state and competency model.

### Key Methods

- **`_retrieve_articles()`**: This is the core method of the agent. It uses the `EnhancedRAGPipeline` to perform a hybrid search, which combines vector search, keyword search, and graph-based search. It can also enhance the search query with terms from the collaboration context to improve the search results.

- **`_assess_retrieval_quality()`**: This method assesses the quality of the retrieved articles based on their relevance scores and sources. This helps the agent to determine if the retrieved information is good enough to be used in a response.

## Hybrid Search

The `EnhancedRetrievalAgent` uses a hybrid search approach, which combines the strengths of different search techniques:

- **Vector search**: Finds semantically similar documents.
- **Keyword search**: Finds documents that contain specific keywords.
- **Graph-based search**: Uses the relationships in the knowledge graph to find related information.

This hybrid approach allows the agent to retrieve more accurate and context-aware information than a simple keyword search.

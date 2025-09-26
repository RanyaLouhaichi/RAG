# Basic Retrieval Agent Documentation

## Overview

The `RetrievalAgent` is a specialized agent responsible for retrieving relevant Confluence articles. It uses a `SentenceTransformer` model to encode the search query and then queries a ChromaDB collection to find relevant articles. The agent is also designed to collaborate with other agents, which allows it to provide more relevant and context-aware information.

## `RetrievalAgent` Class

### Objective

To retrieve relevant Confluence articles with intelligent collaboration support for context enrichment.

### Capabilities

- `RETRIEVE_DATA`: Can retrieve data from a knowledge base.
- `RANK_CONTENT`: Can rank the retrieved content based on its relevance to the query.
- `COORDINATE_AGENTS`: Can coordinate with other agents to fulfill a user's request.

### Cognitive Cycle

The `RetrievalAgent` follows the same cognitive cycle as the `BaseAgent`:

- **`_perceive(input_data)`**: In this phase, the agent processes the input data, which includes the user's prompt and any collaboration context. It then updates its beliefs about the task at hand.

- **`_act()`**: This is the main phase of the agent's operation. It orchestrates the retrieval process by calling the `_retrieve_articles` method to get the articles and then `_assess_retrieval_quality` to evaluate the results.

- **`_rethink(action_result)`**: In this phase, the agent reflects on the outcome of the retrieval process and updates its mental state and competency model.

### Key Methods

- **`_retrieve_articles()`**: This is the core method of the agent. It uses a `SentenceTransformer` model to encode the search query and then queries a ChromaDB collection to find relevant articles. It can also enhance the search query with terms from the collaboration context to improve the search results.

- **`_assess_retrieval_quality()`**: This method assesses the quality of the retrieved articles based on their relevance scores and sources.

## Usage

The `RetrievalAgent` is used to provide contextual information to other agents. It can be triggered to retrieve relevant articles from a knowledge base, which can then be used to answer user queries or generate more comprehensive responses.

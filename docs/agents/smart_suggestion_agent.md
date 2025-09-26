# Smart Suggestion Agent Documentation

## Overview

The `SmartSuggestionAgent` is a specialized agent responsible for providing immediate and intelligent article suggestions for Jira issues. It can analyze the content of a Jira issue, retrieve relevant articles from a knowledge base, and learn from user feedback to improve its suggestions over time.

## `SmartSuggestionAgent` Class

### Objective

To provide immediate, intelligent article suggestions for Jira issues with continuous learning.

### Capabilities

- `RETRIEVE_DATA`: Can retrieve data from a knowledge base.
- `RANK_CONTENT`: Can rank the retrieved content based on its relevance to the query.
- `PROVIDE_RECOMMENDATIONS`: Can provide recommendations based on its analysis.

### Cognitive Cycle

The `SmartSuggestionAgent` follows the same cognitive cycle as the `BaseAgent`:

- **`_perceive(input_data)`**: In this phase, the agent processes the input data, which includes the Jira issue key, summary, description, type, and status. It then updates its beliefs about the task at hand.

- **`_act()`**: This is the main phase of the agent's operation. It orchestrates the suggestion generation process, which includes:
    1. **Analyzing the issue content**: The agent analyzes the content of the Jira issue to extract keywords and determine a search strategy.
    2. **Suggesting articles**: The agent uses a `SentenceTransformer` model to encode the search query and then queries a ChromaDB collection to find relevant articles. It also applies feedback learning to re-rank the articles.

- **`_rethink(action_result)`**: In this phase, the agent reflects on the outcome of the suggestion generation process and updates its mental state and competency model.

### Key Methods

- **`_analyze_issue_content()`**: This method analyzes the content of a Jira issue to extract keywords and determine a search strategy.

- **`_suggest_articles()`**: This is the core method of the agent. It retrieves relevant articles from a ChromaDB collection and then re-ranks them based on feedback learning.

- **`_track_feedback()`**: This method tracks user feedback on article suggestions.

- **`_apply_feedback_learning()`**: This method applies feedback learning to re-rank articles based on their past performance.

## Usage

The `SmartSuggestionAgent` is used to improve the efficiency of resolving Jira issues. It can be triggered to provide relevant article suggestions for a given Jira issue, which can help users to find solutions to their problems more quickly.

# Intent Router

The `intent_router.py` module provides a simple yet effective intent classification function. Its primary purpose is to analyze a user's query and determine the underlying intent, which helps the orchestrator route the request to the appropriate agent or workflow.

## Function: `classify_intent`

This function takes a user's query and conversation history to classify the intent.

### Key Responsibilities:

- **Intent Classification:** Analyzes the user's query to determine the primary intent. The possible intents are:
    - `recommendation`: When the user is asking for advice, suggestions, or improvements.
    - `predictive_analysis`: When the user is asking for forecasts, predictions, or analysis of trends.
    - `article_retrieval`: When the user is looking for documentation, articles, or guides.
    - `generic_question`: The default intent if no other specific intent is detected.
- **Project Detection:** Identifies project keys (typically in all caps) within the query.

### Parameters:

- `query` (str): The user's input query.
- `history` (List[Dict[str, str]]): The conversation history (currently not used in the logic but available for future enhancements).

### Returns:

- A dictionary containing the detected `intent` and, if found, the `project` key.

## Usage:

The `classify_intent` function is called by the orchestrator at the beginning of the request handling process. The returned intent is then used to decide which agent or sequence of agents should handle the user's request.

### Example:

- **Query:** "Can you recommend some improvements for the PHOENIX project?"
- **Result:** `{"intent": "recommendation", "project": "PHOENIX"}`

- **Query:** "Find the documentation for setting up the new API."
- **Result:** `{"intent": "article_retrieval"}`

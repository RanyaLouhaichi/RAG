# Jira Article Generator Agent Documentation

## Overview

The `JiraArticleGeneratorAgent` is a specialized agent responsible for generating high-quality, Confluence-ready articles from resolved Jira tickets. It can extract the solution from a ticket, collaborate with other agents to gather more context, and even incorporate human feedback to refine the articles.

## `JiraArticleGeneratorAgent` Class

### Objective

To generate high-quality, Confluence-ready articles based on resolved Jira tickets, with human feedback integration.

### Capabilities

- `GENERATE_ARTICLE`: Can generate a complete article from a Jira ticket.
- `COORDINATE_AGENTS`: Can coordinate with other agents to gather information.
- `PROCESS_FEEDBACK`: Can process human feedback to refine an article.

### Cognitive Cycle

The `JiraArticleGeneratorAgent` follows the same cognitive cycle as the `BaseAgent`:

- **`_perceive(input_data)`**: In this phase, the agent processes the input data, which includes the ticket ID, any human feedback, and the article version. It then updates its beliefs about the task at hand.

- **`_act()`**: This is the main phase of the agent's operation. It orchestrates the entire article generation process, which includes:
    1. **Extracting the solution**: The agent analyzes the Jira ticket's comments and changelogs to extract the solution.
    2. **Assessing collaboration needs**: The agent determines if it needs to collaborate with other agents to get more context.
    3. **Coordinating with other agents**: If collaboration is needed, the agent interacts with other agents (like `JiraDataAgent` and `KnowledgeBaseAgent`) to gather more information.
    4. **Building a prompt**: The agent constructs a detailed prompt for the language model, including the problem description, solution, and any additional context.
    5. **Generating the article**: The agent calls the language model to generate the article.
    6. **Processing human feedback**: If human feedback is provided, the agent analyzes it and generates a refined version of the article.

- **`_rethink(action_result)`**: In this phase, the agent reflects on the generated article and the overall process. It updates its mental state and competency model based on the outcome.

### Key Methods

- **`_extract_solution_from_ticket()`**: This method extracts the solution from a Jira ticket by analyzing its comments and changelogs.

- **`_build_comprehensive_prompt_with_solution()`**: This method builds a detailed prompt for the language model to generate an article from scratch.

- **`_build_comprehensive_prompt_with_feedback_and_solution()`**: This method builds a prompt for refining an existing article based on human feedback.

- **`_process_human_feedback()`**: This method analyzes human feedback to understand the sentiment, key points, and specific requests for refinement.

- **`_generate_article()`**: This is the core method of the agent. It orchestrates the entire article generation process.

## Usage

The `JiraArticleGeneratorAgent` is used to automate the creation of technical documentation from Jira tickets. It can be triggered to generate a new article for a resolved ticket, or to refine an existing article based on feedback.

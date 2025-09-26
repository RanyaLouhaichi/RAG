# Orchestrator

The `Orchestrator` is the central component of the system, responsible for managing and coordinating the various agents and workflows. It acts as the brain of the operation, deciding which agents to involve, in what order, and how they should collaborate to fulfill a user's request.

## Key Responsibilities:

- **Agent Management:** Initializes and maintains a registry of all available agents.
- **Workflow Management:** Defines and executes different workflows (e.g., general chat, productivity analysis, JIRA article generation) using a state graph.
- **Intent Routing:** Uses an intent classifier to determine the user's intent and route the request to the appropriate workflow.
- **Collaboration Orchestration:** Leverages the `CollaborativeFramework` to enable and manage collaboration between agents.
- **State Management:** Manages the state of the workflow, passing relevant information between agents.
- **LangSmith Integration:** Provides comprehensive tracing and monitoring of workflows and agent interactions using LangSmith.
- **MCP Integration:** Manages the connection to and interaction with MCP (Multi-Agent Communication Protocol) servers.

## Workflows

The orchestrator defines several key workflows:

- **General Workflow:** A flexible workflow that handles general user queries. It starts with intent classification and then routes the request to the appropriate sequence of agents (e.g., data retrieval, recommendation, chat).
- **Productivity Workflow:** A specialized workflow for generating a productivity analysis for a specific project. It involves the JIRA Data Agent, Predictive Analysis Agent, Recommendation Agent, and Productivity Dashboard Agent.
- **JIRA Workflow:** A workflow for generating a knowledge base article from a JIRA ticket. It uses the JIRA Article Generator and Knowledge Base Agent.
- **Predictive Workflow:** A workflow for generating predictive analysis for a project, involving the JIRA Data Agent, Predictive Analysis Agent, and Recommendation Agent.

## Methods:

- `__init__(...)`: Initializes the orchestrator, including all agents, the shared memory, the model manager, and the collaborative framework.
- `run_workflow(...)`: The main entry point for executing the general chat workflow.
- `run_productivity_workflow(...)`: Executes the productivity analysis workflow.
- `run_jira_workflow(...)`: Executes the JIRA article generation workflow.
- `run_predictive_workflow(...)`: Executes the predictive analysis workflow.
- `_build_*_workflow()`: A series of methods that define the structure of each workflow using a `StateGraph`.
- `_*_node()`: A series of methods that define the logic for each node in the state graphs, typically involving running an agent and updating the state.

## Usage:

The `Orchestrator` is the primary interface for interacting with the multi-agent system. By calling one of its `run_*_workflow` methods, a user can initiate a complex, multi-agent task. The orchestrator handles all the underlying complexity, providing a seamless and intelligent user experience.

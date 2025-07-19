# JURIX Workflow Architecture

## Overview

JURIX is a modular orchestration system for Agile and software development support, integrating conversational AI, document retrieval, recommendation, and productivity analytics. The system is built around several specialized agents and workflows that interact to answer user queries, retrieve relevant documentation, generate recommendations, and analyze productivity data.

---

## Workflows

### 1. General Orchestration Workflow

Handles user queries, classifies intent, and routes to chat, retrieval, or recommendation agents.

```mermaid
flowchart TD
    A1([API Trigger:<br>/ask-orchestrator]) --> B1[Init JurixState<br>+ SharedMemory]
    B1 --> C1[ClassifyIntentNode<br>Classify user query]
    C1 --generic_question, followup, summarize, conversation--> D1[ChatAgent<br>Conversational AI]
    C1 --search, article_retrieval--> E1[RetrievalAgent<br>Semantic search over docs]
    C1 --recommendation--> F1[RecommendationAgent<br>Generate recommendations]
    C1 --ticket_query, jira_data--> G1[JiraDataAgent<br>Fetch Jira tickets]

    %% ChatAgent can call RetrievalAgent or JiraDataAgent as subroutines
    D1 --needs articles--> E1
    D1 --needs tickets--> G1
    D1 --needs recommendations--> F1

    %% Data flows back to ChatAgent
    E1 --retrieved_articles--> D1
    G1 --jira_tickets--> D1
    F1 --recommendations--> D1

    D1 --response, context, status--> Z1([API Response:<br>answer, workflow_history, recommendations, etc.])

    %% State and Shared Memory
    B1 -.->|JurixState<br>+ SharedMemory| S1[JurixSharedMemory]
    D1 -.->|Reads/Writes state| S1
    E1 -.->|Reads/Writes state| S1
    F1 -.->|Reads/Writes state| S1
    G1 -.->|Reads/Writes state| S1
    C1 -.->|Reads/Writes state| S1

    %% Outputs
    style A1 fill:#e0f7fa,stroke:#00796b,stroke-width:2px
    style C1 fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
    style D1 fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
    style E1 fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style F1 fill:#b3e5fc,stroke:#0288d1,stroke-width:2px
    style G1 fill:#ffe082,stroke:#ff6f00,stroke-width:2px
    style S1 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

### 2. Productivity Workflow

Analyzes Jira ticket data for a project and time range, generates recommendations, and produces a productivity dashboard.

```mermaid
flowchart TD
    A1([API Trigger:<br>/trigger-productivity-workflow]) --> B1[Init JurixState<br>+ SharedMemory]
    B1 --> C1[JiraDataAgent<br>Load & filter Jira tickets]
    C1 --success: jira_data--> D1[RecommendationAgent<br>Generate productivity recommendations]
    C1 --failure--> Z1([API Response:<br>error, workflow_history])

    D1 --recommendations--> E1[ProductivityDashboardAgent<br>Generate metrics & dashboard]
    E1 --dashboard, metrics, status--> F1([API Response:<br>dashboard, recommendations, workflow_history])

    %% State and Shared Memory
    B1 -.->|JurixState<br>+ SharedMemory| S1[JurixSharedMemory]
    C1 -.->|Reads/Writes state| S1
    D1 -.->|Reads/Writes state| S1
    E1 -.->|Reads/Writes state| S1

    %% Outputs
    style A1 fill:#e0f7fa,stroke:#00796b,stroke-width:2px
    style C1 fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
    style D1 fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
    style E1 fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style S1 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

### 3. Jira Article Generation Workflow

Automates the creation and refinement of Confluence-ready articles from Jira tickets, with quality evaluation and iterative refinement.

```mermaid
flowchart TD
    A1([API Trigger:<br>/trigger-jira-workflow]) --> B1[Init JurixState<br>+ SharedMemory]
    B1 --> C1[Run RecommendationAgent<br>with ticket_id, project, articles]
    C1 --recommendation_id, recommendations--> B2
    B2[Update JurixState<br>with recommendations]
    B2 --> D1[JiraArticleGeneratorAgent<br>Generate article]
    D1 --article, status, stage--> E1[KnowledgeBaseAgent<br>Evaluate/refine article]
    E1 --needs refinement & not yet refined--> D1
    E1 --done or already refined--> F1{Approval Needed?}
    F1 --Yes--> G1([User/API Approval])
    G1 --Approved--> H1[workflow_stage: complete]
    G1 --Rejected--> D1
    F1 --No--> H1
    H1 --> Z1([API Response:<br>ticket_id, workflow_history,<br>article, recommendations, etc.])

    %% State and Shared Memory
    B1 -.->|JurixState<br>+ SharedMemory| S1[JurixSharedMemory]
    D1 -.->|Reads/Writes state| S1
    E1 -.->|Reads/Writes state| S1
    C1 -.->|Reads/Writes recommendations| S1
    G1 -.->|Updates approval in state| S1

    %% Outputs
    style A1 fill:#e0f7fa,stroke:#00796b,stroke-width:2px
    style C1 fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
    style D1 fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
    style E1 fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style H1 fill:#b3e5fc,stroke:#0288d1,stroke-width:2px
    style S1 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```
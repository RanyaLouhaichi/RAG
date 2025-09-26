# JURIX Test Suite

## Overview

The `tests` folder contains the automated and manual test scripts for the JURIX platform. These tests are designed to validate the correctness, robustness, and integration of all major components, including agent workflows, semantic memory, predictive analytics, API endpoints, and hybrid multi-agent collaboration.

## Testing Philosophy

- **Integration-First:** Most tests are integration or system-level, validating real interactions between agents, memory, APIs, and external services (Jira, Redis, Neo4j).
- **Realistic Data:** Tests use realistic or mock data to simulate actual Jira tickets, Confluence articles, and workflow scenarios.
- **Hybrid Architecture Validation:** Special focus is given to collaborative agent workflows and hybrid RAG+graph+predictive pipelines.
- **API Contract:** API endpoints are tested for correctness, error handling, and data structure compliance.
- **Manual & Automated:** Some tests are designed for manual inspection (with print/log output), while others use `pytest` for automation and assertions.

## Test Modules

### 1. `test_semantic_memory.py`

**Purpose:**  
Validates the semantic memory subsystem, including vector memory storage, retrieval, and agent integration.

**Key Features:**
- Checks for required dependencies (`sentence_transformers`, `numpy`).
- Verifies Redis connectivity.
- Tests the `VectorMemoryManager` for storing and searching memories.
- Validates agent integration with semantic memory (experience storage and recall).
- Prints detailed step-by-step results for manual inspection.

### 2. `test_predictive_calculations.py`

**Purpose:**  
Unit and integration tests for the predictive analytics agent, focusing on velocity, burndown, capacity, and sprint completion calculations.

**Key Features:**
- Uses `pytest` fixtures for sample ticket data and agent instantiation.
- Tests velocity history extraction, burndown chart generation, capacity forecasting, and sprint completion probability.
- Asserts correctness of returned data structures and value ranges.

### 3. `test_hybrid_architecture.py`

**Purpose:**  
Validates the hybrid multi-agent architecture and collaborative workflows.

**Key Features:**
- Runs end-to-end workflows using the orchestrator.
- Checks for correct intent classification, agent collaboration, and recommendation generation.
- Prints collaboration details and workflow results for manual review.

### 4. `test_forecast_api.py`

**Purpose:**  
Tests the REST API endpoints for predictive analytics and dashboard forecasting.

**Key Features:**
- Uses Flask test client and `pytest` for endpoint testing.
- Mocks predictive agent responses for deterministic results.
- Validates velocity, burndown, and capacity forecast endpoints.
- Checks error handling for invalid requests.

### 5. `test_api.py`

**Purpose:**  
Tests Jira API connectivity and data retrieval.

**Key Features:**
- Verifies connection to Jira instance.
- Fetches projects and issues, printing summaries for manual verification.
- Useful for debugging API credentials and connectivity.


## Running the Tests

- **Manual Tests:**  
  Run scripts directly with Python for step-by-step output:
  ```
  python tests/test_semantic_memory.py
  python tests/test_hybrid_architecture.py
  ```

- **Automated Tests:**  
  Use `pytest` for automated assertion-based tests:
  ```
  pytest tests/
  ```

- **API Tests:**  
  Ensure the backend server is running, then run API tests.

## Best Practices

- Ensure all required services (Redis, Neo4j, Jira, Confluence) are running for integration tests.
- Review printed/logged output for manual tests to verify correct behavior.
- Extend tests with new scenarios as new features and agents are added.

---

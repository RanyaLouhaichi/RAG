# Predictive Agent MCP Server

The `PredictiveAgentMCPServer` is an MCP (Multi-Agent Communication Protocol) server that exposes the capabilities of the `PredictiveAnalysisAgent`. It provides tools for making predictions about sprint completion, team velocity, project risks, and workflow bottlenecks.

## Key Responsibilities:

- **Tool Registration:** Registers tools for various predictive analysis tasks.
- **Resource Registration:** Provides access to information about the prediction models, configuration thresholds, and historical accuracy.
- **Prompt Registration:** Exposes prompts for common predictive analysis scenarios.

## Registered Tools:

- **`predict_sprint_completion`**:
    - **Description:** Predicts the probability of sprint completion and provides a risk analysis.
    - **Input:** `tickets` (array), `metrics` (object), and `historical_data` (object).
    - **Output:** A dictionary with the `probability`, `confidence`, `risk_level`, and other related details.

- **`forecast_velocity`**:
    - **Description:** Forecasts team velocity trends.
    - **Input:** `historical_data` (object), `current_velocity` (number), and optional `tickets` (array).
    - **Output:** A dictionary containing the velocity `forecast`, `trend`, and `confidence`.

- **`identify_risks`**:
    - **Description:** Identifies and prioritizes project risks.
    - **Input:** `tickets` (array), `metrics` (object), and `historical_data` (object).
    - **Output:** A dictionary with a summary of risks, warnings, and mitigation priorities.

- **`predict_bottlenecks`**:
    - **Description:** Predicts and analyzes workflow bottlenecks.
    - **Input:** `tickets` (array).
    - **Output:** A dictionary with current and predicted bottlenecks, along with recommendations.

## Registered Resources:

- **`predictions://models/info`**:
    - **Name:** Prediction Models
    - **Description:** Provides information about the available prediction models, including their accuracy and the factors they consider.

- **`predictions://config/thresholds`**:
    - **Name:** Prediction Thresholds
    - **Description:** Exposes the configurable thresholds used for making predictions.

- **`predictions://metrics/accuracy`**:
    - **Name:** Prediction Accuracy
    - **Description:** Provides historical data on the accuracy of the prediction models.

## Registered Prompts:

- **`sprint_risk_analysis`**: A prompt to analyze sprint risks and predict completion probability.
- **`capacity_planning`**: A prompt to analyze team capacity and predict future workload.

## Usage:

This server allows other agents to leverage the predictive capabilities of the `PredictiveAnalysisAgent`. It is a key component for proactive project management, enabling the system to anticipate issues and recommend preventive actions.

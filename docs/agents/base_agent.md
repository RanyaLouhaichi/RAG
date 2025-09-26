# Base Agent Documentation

## Overview

The `base_agent.py` file is the foundation for all agents in the system. It defines the core components of an agent's mental state and behavior, including beliefs, capabilities, and a cognitive cycle.

## Key Classes

### `AgentCapability`

An `Enum` that defines the possible capabilities of an agent. These capabilities are used to determine which agent is best suited for a particular task.

### `ConfidentBelief`

A class that represents a belief held by an agent. Each belief has a `value`, a `confidence` score (from 0.0 to 1.0), a `source`, and a `timestamp`. The confidence of a belief can decay over time, making the agent's knowledge more dynamic.

### `CompetencyModel`

A class that models an agent's competency at various tasks. It tracks the `success_rate` for each task type, allowing the system to learn which agents are best at certain operations.

### `EnhancedMentalState`

This class represents the complete mental state of an agent. It includes:

- **Capabilities**: A list of the agent's capabilities.
- **Obligations**: A list of tasks the agent is obligated to perform.
- **Beliefs**: A dictionary of `ConfidentBelief` objects.
- **Decisions**: A history of the agent's past decisions.
- **Competency Model**: An instance of the `CompetencyModel` class.
- **Vector Memory**: An optional `VectorMemoryManager` for semantic storage and retrieval of memories.

### `BaseAgent`

The main base class for all agents in the system. It provides the following:

- **Initialization**: Each agent is initialized with a `name`, a unique `agent_id`, and an `EnhancedMentalState`.
- **Cognitive Cycle**: The `BaseAgent` class defines a basic cognitive cycle with three phases:
    - **`_perceive(input_data)`**: The agent perceives new information from its environment and updates its beliefs.
    - **`_act()`**: The agent decides on an action to take based on its current mental state.
    - **`_rethink(action_result)`**: The agent reflects on the outcome of its action and updates its competencies and beliefs.
- **`run(input_data)`**: The main method to execute the agent's cognitive cycle.

## Usage

To create a new agent, you should inherit from the `BaseAgent` class and override the `_perceive`, `_act`, and `_rethink` methods to implement the agent's specific logic. You should also define the agent's `OBJECTIVE` and `capabilities`.

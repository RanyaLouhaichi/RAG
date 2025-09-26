
# Cognitive Core Documentation

## Overview

The `cognitive_core.py` file provides a more advanced and human-like reasoning mechanism for the agents. It focuses on belief management, reasoning under uncertainty, and understanding the quality and reliability of knowledge. This allows the agents to make better decisions in complex and dynamic environments.

## Key Classes

### `ConfidenceLevel`

An `Enum` that defines different levels of confidence for beliefs, ranging from `VERY_LOW` to `CERTAIN`.

### `InformationSource`

An `Enum` that represents the source of information for a belief. Each source has a different credibility pattern, which is used to estimate the initial confidence of a belief.

### `ConfidentBelief`

A dataclass that represents a belief with a rich set of attributes:

- **`value`**: The value of the belief.
- **`confidence`**: The confidence score of the belief (from 0.0 to 1.0).
- **`source`**: The source of the information.
- **`created_at`** and **`last_updated`**: Timestamps to track the age of the belief.
- **`evidence_count`** and **`contradiction_count`**: Counters to track supporting and conflicting evidence.
- **`temporal_decay_rate`**: A rate at which the confidence of the belief decays over time.

This class allows the agent to reason about its beliefs in a more nuanced way, taking into account factors like age, source, and evidence.

### `CognitiveBeliefSystem`

A class that manages the agent's beliefs. It provides the following functionalities:

- **Belief management**: It can add new beliefs, update existing ones, and resolve conflicts between beliefs based on their confidence, source, and recency.
- **Knowledge quality assessment**: It can assess the overall quality of the agent's knowledge base, providing insights into the reliability and stability of its beliefs.
- **Proactive learning**: It can suggest information needs, enabling the agent to proactively seek out information to fill gaps in its understanding.

### `EnhancedMentalState`

An enhanced version of the agent's mental state that uses the `CognitiveBeliefSystem` to manage beliefs. It also includes confidence thresholds for different types of actions, such as taking an action or requesting collaboration. This allows the agent to make more informed decisions based on its level of confidence.

### `CognitiveBeliefProxy`

A proxy class that makes the `CognitiveBeliefSystem` compatible with the existing agent code. It provides a dictionary-like interface for accessing and manipulating beliefs, while still leveraging the advanced features of the cognitive core.

## Usage

The components in this file are used to build more intelligent and autonomous agents. By using the `EnhancedMentalState` and `CognitiveBeliefSystem`, agents can reason about their knowledge in a more human-like way, leading to better decision-making and performance.

# RAG Setup Documentation

## Overview

The `setup_rag.py` script is a command-line tool that automates the process of setting up the enhanced RAG system. It ensures that all prerequisites are met, starts the necessary services, and initializes the RAG pipeline.

## Functions

### `check_docker()`

This function checks if Docker is installed and running on the system. It does this by running the `docker --version` and `docker ps` commands.

### `start_neo4j()`

This function starts a Neo4j container using `docker-compose`. It uses the `docker-compose-neo4j.yml` file to define the Neo4j service. After starting the container, it waits for 30 seconds and then tests the connection to the Neo4j database.

### `install_dependencies()`

This function installs the required Python dependencies for the RAG system. It reads the dependencies from the `requirements_rag.txt` file and installs them using `pip`.

### `test_imports()`

This function tests that all the required Python packages can be imported successfully. This helps to ensure that the dependencies were installed correctly.

### `initialize_rag_system()`

This function initializes the `EnhancedRAGPipeline`. It creates an instance of the `EnhancedRAGPipeline` class, which is the main component of the RAG system.

### `main()`

The main function of the script, which runs all the setup steps in the correct order:

1. Checks for Docker.
2. Installs dependencies.
3. Tests imports.
4. Starts Neo4j.
5. Initializes the RAG system.

If all steps are successful, it prints a message with the next steps to take.

## Usage

To use the script, you can run it from the command line:

```bash
python setup_rag.py
```

This will execute all the setup steps and prepare the RAG system for use.

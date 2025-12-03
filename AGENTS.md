# Graphiti MCP Server - Codebase Guide for AI Agents

This document explains the modular structure of the Graphiti MCP Server codebase to help AI agents navigate and understand the code organization.

## Overview

The Graphiti MCP Server exposes Graphiti (a knowledge graph memory service for AI agents) through the Model Context Protocol (MCP). The codebase has been refactored from a single monolithic file into a well-organized Python package with focused modules.

## Entry Points

There are two ways to run the server:

```bash
# As a Python package (recommended)
python -m graphiti_mcp_server [options]

# Legacy wrapper script
python run_server.py [options]
```

Both entry points call `graphiti_mcp_server.main:main()`.

**Important**: The wrapper script is named `run_server.py` (not `graphiti_mcp_server.py`) to avoid name conflicts with the package directory.

## Package Structure

```
graphiti_mcp_server/
├── __init__.py              # Package exports
├── __main__.py              # Entry point for `python -m graphiti_mcp_server`
├── main.py                  # CLI parsing, server startup, signal handling
│
├── config/                  # Configuration classes
│   ├── __init__.py          # Exports all config classes
│   ├── settings.py          # Global constants (DEFAULT_LLM_MODEL, SEMAPHORE_LIMIT, etc.)
│   ├── llm.py               # GraphitiLLMConfig - LLM client configuration
│   ├── embedder.py          # GraphitiEmbedderConfig - Embedding configuration
│   ├── neo4j.py             # Neo4jConfig - Database connection settings
│   ├── redis.py             # RedisConfig - Queue connection settings
│   └── graphiti.py          # GraphitiConfig, MCPConfig - Main configuration classes
│
├── models/                  # Data models and type definitions
│   ├── __init__.py          # Exports all models
│   ├── entities.py          # Pydantic models: Requirement, Preference, Procedure, ENTITY_TYPES
│   ├── responses.py         # TypedDict definitions for API responses
│   └── queue.py             # QueuedEpisode model for Redis queue
│
├── auth/                    # Authentication and authorization
│   ├── __init__.py          # Exports auth components
│   ├── nonce.py             # Nonce token validation (ALLOWED_NONCE_TOKENS, is_nonce_valid)
│   ├── principal.py         # Principal authentication (get_authenticated_principal)
│   └── middleware.py        # AuthenticationMiddleware - ASGI middleware for auth
│
├── group_id/                # Group ID management
│   ├── __init__.py          # Exports context functions
│   └── context.py           # Context variable functions for group_id allowlisting
│                            # (get_allowed_group_ids, get_effective_group_id, etc.)
│
├── queue/                   # Redis-based episode queue
│   ├── __init__.py          # Exports queue components
│   ├── state.py             # Global state variables (redis_client, queue_manager, etc.)
│   ├── manager.py           # RedisQueueManager class - queue operations
│   ├── worker.py            # process_episode_queue - background worker
│   └── init.py              # initialize_redis, shutdown_redis functions
│
├── client/                  # Graphiti client management
│   ├── __init__.py          # Exports client functions
│   └── graphiti.py          # Graphiti client initialization and singleton access
│                            # (get_graphiti_client, initialize_graphiti, etc.)
│
├── tools/                   # MCP Tool implementations (11 tools)
│   ├── __init__.py          # Exports all tools
│   ├── memory.py            # add_memory - Add episodes to the knowledge graph
│   ├── search.py            # search_memory_nodes, search_memory_facts
│   ├── episodes.py          # get_episodes, delete_episode
│   ├── edges.py             # get_entity_edge, delete_entity_edge
│   ├── groups.py            # list_group_ids, delete_everything_by_group_id
│   ├── queue_status.py      # get_queue_status - Monitor processing queues
│   └── admin.py             # clear_graph - Administrative operations
│
├── resources/               # MCP Resources
│   ├── __init__.py          # Exports resources
│   └── status.py            # get_status - Server and Neo4j connection status
│
└── utils/                   # Utility functions
    ├── __init__.py          # Exports utilities
    └── formatters.py        # format_fact_result - Format EntityEdge for output
```

## Key Components

### Configuration (`config/`)

All configuration is managed through Pydantic models that can be initialized from:
1. Environment variables (`.env` file)
2. CLI arguments (which override environment variables)

Key classes:
- `GraphitiConfig`: Main configuration aggregating all sub-configs
- `GraphitiLLMConfig`: OpenAI API settings (model, temperature, API key)
- `Neo4jConfig`: Database connection (uri, user, password)
- `RedisConfig`: Queue connection and key prefixes

### Authentication (`auth/`)

The server supports nonce-based authentication via query parameters:
- Set `MCP_SERVER_NONCE_TOKENS` environment variable with comma-separated tokens
- Requests must include `?nonce=<token>` for authentication
- `AuthenticationMiddleware` is a pure ASGI middleware that handles auth

### Group ID Management (`group_id/`)

Group IDs namespace data in the knowledge graph. The `X-Group-Id` HTTP header can restrict which group_ids are allowed for a request:
- `get_effective_group_id()`: Resolves the group_id to use, respecting allowlists
- `get_effective_group_ids()`: For search operations with multiple group_ids

### Queue System (`queue/`)

Episodes are processed asynchronously via Redis queues:
- `RedisQueueManager`: Handles enqueue/dequeue with crash recovery
- `process_episode_queue()`: Worker function that processes episodes sequentially per group_id
- Uses BRPOPLPUSH pattern for reliable processing

### Tools (`tools/`)

Each tool file contains one or more MCP tool functions:

| File | Tools | Purpose |
|------|-------|---------|
| `memory.py` | `add_memory` | Add episodes (text, JSON, messages) to the graph |
| `search.py` | `search_memory_nodes`, `search_memory_facts` | Query the knowledge graph |
| `episodes.py` | `get_episodes`, `delete_episode` | Retrieve/delete episodes |
| `edges.py` | `get_entity_edge`, `delete_entity_edge` | Manage entity relationships |
| `groups.py` | `list_group_ids`, `delete_everything_by_group_id` | Group management |
| `queue_status.py` | `get_queue_status` | Monitor background processing |
| `admin.py` | `clear_graph` | Destructive admin operations (password protected) |

### Client (`client/`)

Manages the Graphiti client singleton:
- `get_graphiti_client()`: Access the initialized client
- `initialize_graphiti()`: Setup the client with Neo4j and LLM connections
- `get_config()` / `set_config()`: Access global configuration

## Important Patterns

### Global State Access

Global state is accessed through getter functions rather than direct imports:
```python
from graphiti_mcp_server.client import get_graphiti_client, get_config
from graphiti_mcp_server.queue import get_queue_manager, get_queue_workers
```

### Tool Registration

Tools are registered with the MCP server in `main.py:register_tools_and_resources()`:
```python
mcp.tool()(add_memory)
mcp.tool()(search_memory_nodes)
# ... etc
```

### Error Handling

Tools return either a success response or an `ErrorResponse`:
```python
async def some_tool(...) -> SuccessResponse | ErrorResponse:
    if error_condition:
        return ErrorResponse(error='Description of error')
    return SuccessResponse(message='Success message')
```

## Environment Variables

Key environment variables (see `.env.example`):
- `OPENAI_API_KEY`: Required for LLM and embeddings
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Database connection
- `REDIS_HOST`, `REDIS_PORT`: Queue connection
- `MCP_SERVER_NONCE_TOKENS`: Authentication tokens (comma-separated)
- `CLEAR_GRAPH_PASSWORD`: Password for `clear_graph` tool
- `MODEL_NAME`, `SMALL_MODEL_NAME`: LLM model selection
- `SEMAPHORE_LIMIT`: Concurrent operation limit

## Adding New Features

### Adding a New Tool

1. Create a new file in `tools/` or add to an existing one
2. Import required dependencies from other modules
3. Define the async function with proper type hints
4. Add the function to `tools/__init__.py` exports
5. Register the tool in `main.py:register_tools_and_resources()`

### Adding Configuration

1. Add new fields to the appropriate config class in `config/`
2. Update `from_env()` and/or `from_cli_and_env()` methods
3. Add CLI argument in `main.py:initialize_server()` if needed

## Testing

Run syntax checks on all files:
```bash
uv run python -m py_compile graphiti_mcp_server/**/*.py
```

Test imports:
```bash
uv run python -c "from graphiti_mcp_server.main import main; print('OK')"
```

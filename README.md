# Graphiti MCP Server - Enhanced Fork

Graphiti is a framework for building and querying temporally-aware knowledge graphs, specifically tailored for AI agents
operating in dynamic environments. Unlike traditional retrieval-augmented generation (RAG) methods, Graphiti
continuously integrates user interactions, structured and unstructured enterprise data, and external information into a
coherent, queryable graph. The framework supports incremental data updates, efficient retrieval, and precise historical
queries without requiring complete graph recomputation, making it suitable for developing interactive, context-aware AI
applications.

This is an enhanced Model Context Protocol (MCP) server implementation for Graphiti. The MCP server exposes
Graphiti's key functionality through the MCP protocol, allowing AI assistants to interact with Graphiti's knowledge
graph capabilities.

## Key Enhancements in This Fork

This enhanced version includes several important improvements over the original implementation:

1. **🚀 Latest Graphiti Core Compatibility** - Uses the current version of graphiti-core with all latest features and improvements
2. **🤖 GPT-5, O1, O3 Model Support** - Proper handling of OpenAI's reasoning models with automatic parameter adjustment (disables temperature, reasoning, and verbosity parameters)
3. **🔒 Token-Based Authentication** - Production-ready nonce token authentication system enabling secure public deployment via SSE transport
4. **📊 New `list_group_ids` Tool** - Discover and manage all group IDs across nodes and relationships in your knowledge graph
5. **🛡️ Enhanced Security** - Pure ASGI middleware-based authentication with constant-time token comparison to prevent timing attacks
6. **🔇 Telemetry Control** - Automatic disabling of telemetry for privacy-focused deployments (set before graphiti_core imports)
7. **⚡ Simplified Dependencies** - Removed Azure OpenAI dependencies for easier setup and deployment

### About Azure Support

**Note on Azure OpenAI:** Azure OpenAI support was removed during refactoring due to implementation conflicts with the new authentication middleware. If you need Azure OpenAI support in this enhanced MCP server, pull requests are welcome! The original implementation can be found in the [upstream Graphiti repository](https://github.com/getzep/graphiti).

### About This Fork

This fork maintains compatibility with the latest Graphiti core while adding production-ready features for secure public deployment. It focuses on OpenAI API compatibility and enhanced security features.

## Features

The Graphiti MCP server exposes the following key high-level functions of Graphiti:

- **Episode Management**: Add, retrieve, and delete episodes (text, messages, or JSON data)
- **Entity Management**: Search and manage entity nodes and relationships in the knowledge graph
- **Search Capabilities**: Search for facts (edges) and node summaries using semantic and hybrid search
- **Group Management**: Organize and manage groups of related data with group_id filtering
- **Graph Maintenance**: Clear the graph and rebuild indices

## Quick Start

### Clone this enhanced fork

```bash
git clone https://github.com/michabbb/graphiti-mcp-but-working.git
cd graphiti-mcp-but-working
```

or

```bash
gh repo clone michabbb/graphiti-mcp-but-working
cd graphiti-mcp-but-working
```

### For Claude Desktop and other `stdio` only clients

1. Note the full path to this directory.

```bash
pwd
```

2. Install the [Graphiti prerequisites](#prerequisites).

3. Configure Claude, Cursor, or other MCP client to use [Graphiti with a `stdio` transport](#integrating-with-mcp-clients). See the client documentation on where to find their MCP configuration files.

### For Cursor and other `sse`-enabled clients (Recommended)

1. Configure your environment variables (copy `.env.example` to `.env` and set your `OPENAI_API_KEY`)

2. Start the service using Docker Compose

```bash
docker compose up
```

3. Point your MCP client to `http://localhost:8000/sse`

**For secure public deployment**, see the [Authentication Guide](auth.md) for setting up nonce token authentication.

## Installation

### Prerequisites

1. Ensure you have Python 3.10 or higher installed.
2. A running Neo4j database (version 5.26 or later required)
3. OpenAI API key for LLM operations

### Setup

1. Clone the repository and navigate to the mcp_server directory
2. Use `uv` to create a virtual environment and install dependencies:

```bash
# Install uv if you don't have it already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies in one step
uv sync
```

## Configuration

The server uses the following environment variables:

- `NEO4J_URI`: URI for the Neo4j database (default: `bolt://localhost:7687`)
- `NEO4J_USER`: Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD`: Neo4j password (default: `demodemo`)
- `OPENAI_API_KEY`: OpenAI API key (required for LLM operations)
- `OPENAI_BASE_URL`: Optional base URL for OpenAI API
- `MODEL_NAME`: OpenAI model name to use for LLM operations.
- `SMALL_MODEL_NAME`: OpenAI model name to use for smaller LLM operations.
- `LLM_TEMPERATURE`: Temperature for LLM responses (0.0-2.0).
- `AZURE_OPENAI_ENDPOINT`: Optional Azure OpenAI LLM endpoint URL
- `AZURE_OPENAI_DEPLOYMENT_NAME`: Optional Azure OpenAI LLM deployment name
- `AZURE_OPENAI_API_VERSION`: Optional Azure OpenAI LLM API version
- `AZURE_OPENAI_EMBEDDING_API_KEY`: Optional Azure OpenAI Embedding deployment key (if other than `OPENAI_API_KEY`)
- `AZURE_OPENAI_EMBEDDING_ENDPOINT`: Optional Azure OpenAI Embedding endpoint URL
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`: Optional Azure OpenAI embedding deployment name
- `AZURE_OPENAI_EMBEDDING_API_VERSION`: Optional Azure OpenAI API version
- `AZURE_OPENAI_USE_MANAGED_IDENTITY`: Optional use Azure Managed Identities for authentication
- `SEMAPHORE_LIMIT`: Episode processing concurrency. See [Concurrency and LLM Provider 429 Rate Limit Errors](#concurrency-and-llm-provider-429-rate-limit-errors)

You can set these variables in a `.env` file in the project directory.

## Running the Server

To run the Graphiti MCP server directly using `uv`:

```bash
uv run graphiti_mcp_server.py
```

With options:

```bash
uv run graphiti_mcp_server.py --model gpt-4.1-mini --transport sse
```

Available arguments:

- `--model`: Overrides the `MODEL_NAME` environment variable.
- `--small-model`: Overrides the `SMALL_MODEL_NAME` environment variable.
- `--temperature`: Overrides the `LLM_TEMPERATURE` environment variable.
- `--transport`: Choose the transport method (sse or stdio, default: sse)
- `--group-id`: Set a namespace for the graph (optional). If not provided, defaults to "default".
- `--destroy-graph`: If set, destroys all Graphiti graphs on startup.
- `--use-custom-entities`: Enable entity extraction using the predefined ENTITY_TYPES

### Concurrency and LLM Provider 429 Rate Limit Errors

Graphiti's ingestion pipelines are designed for high concurrency, controlled by the `SEMAPHORE_LIMIT` environment variable.
By default, `SEMAPHORE_LIMIT` is set to `10` concurrent operations to help prevent `429` rate limit errors from your LLM provider. If you encounter such errors, try lowering this value.

If your LLM provider allows higher throughput, you can increase `SEMAPHORE_LIMIT` to boost episode ingestion performance.

### Docker Deployment

The Graphiti MCP server can be deployed using Docker. The Dockerfile uses `uv` for package management, ensuring
consistent dependency installation.

#### Environment Configuration

Before running the Docker Compose setup, you need to configure the environment variables. You have two options:

1. **Using a .env file** (recommended):

   - Copy the provided `.env.example` file to create a `.env` file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file to set your OpenAI API key and other configuration options:
     ```
     # Required for LLM operations
     OPENAI_API_KEY=your_openai_api_key_here
     MODEL_NAME=gpt-4.1-mini
     # Optional: OPENAI_BASE_URL only needed for non-standard OpenAI endpoints
     # OPENAI_BASE_URL=https://api.openai.com/v1
     ```
   - The Docker Compose setup is configured to use this file if it exists (it's optional)

2. **Using environment variables directly**:
   - You can also set the environment variables when running the Docker Compose command:
     ```bash
     OPENAI_API_KEY=your_key MODEL_NAME=gpt-4.1-mini docker compose up
     ```

#### Neo4j Configuration

The Docker Compose setup includes a Neo4j container with the following default configuration:

- Username: `neo4j`
- Password: `demodemo`
- URI: `bolt://neo4j:7687` (from within the Docker network)
- Memory settings optimized for development use

#### Running with Docker Compose

A Graphiti MCP container is available at: `zepai/knowledge-graph-mcp`. The latest build of this container is used by the Compose setup below.

Start the services using Docker Compose:

```bash
docker compose up
```

Or if you're using an older version of Docker Compose:

```bash
docker-compose up
```

This will start both the Neo4j database and the Graphiti MCP server. The Docker setup:

- Uses `uv` for package management and running the server
- Installs dependencies from the `pyproject.toml` file
- Connects to the Neo4j container using the environment variables
- Exposes the server on port 8000 for HTTP-based SSE transport
- Includes a healthcheck for Neo4j to ensure it's fully operational before starting the MCP server

## Integrating with MCP Clients

### Configuration

To use the Graphiti MCP server with an MCP-compatible client, configure it to connect to the server:

> [!IMPORTANT]
> You will need the Python package manager, `uv` installed. Please refer to the [`uv` install instructions](https://docs.astral.sh/uv/getting-started/installation/).
>
> Ensure that you set the full path to the `uv` binary and your Graphiti project folder.

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "transport": "stdio",
      "command": "/Users/<user>/.local/bin/uv",
      "args": [
        "run",
        "--isolated",
        "--directory",
        "/Users/<user>>/dev/zep/graphiti/mcp_server",
        "--project",
        ".",
        "graphiti_mcp_server.py",
        "--transport",
        "stdio"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password",
        "OPENAI_API_KEY": "sk-XXXXXXXX",
        "MODEL_NAME": "gpt-4.1-mini"
      }
    }
  }
}
```

For SSE transport (HTTP-based), you can use this configuration:

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "transport": "sse",
      "url": "http://localhost:8000/sse"
    }
  }
}
```

## Available Tools

The Graphiti MCP server exposes the following tools:

- `add_episode`: Add an episode to the knowledge graph (supports text, JSON, and message formats)
- `search_nodes`: Search the knowledge graph for relevant node summaries
- `search_facts`: Search the knowledge graph for relevant facts (edges between entities)
- `delete_entity_edge`: Delete an entity edge from the knowledge graph
- `delete_episode`: Delete an episode from the knowledge graph
- `get_entity_edge`: Get an entity edge by its UUID
- `get_episodes`: Get the most recent episodes for a specific group
- `clear_graph`: Clear all data from the knowledge graph and rebuild indices
- `get_status`: Get the status of the Graphiti MCP server and Neo4j connection

## Using the X-Group-Id Header

When using SSE transport, you can pass one or more `group_id` values via the `X-Group-Id` HTTP header. This header supports comma-separated values and acts as an **allowlist** for group_ids.

### Behavior

- **Single group_id in header**: Used as the fixed group_id for all tool calls (tool parameters are ignored)
- **Multiple group_ids in header (comma-separated)**: Acts as an allowlist - only these group_ids are permitted
  - Tool parameters that match an allowed group_id are accepted
  - Tool parameters not in the allowlist are rejected with an error message that shows which group_ids are allowed
  - If no tool parameter is provided, the first allowed group_id is used

This is useful for:
- **Multi-tenant deployments**: Each client can send their tenant ID(s) in the header, ensuring data isolation without relying on tool parameters
- **API gateways**: Upstream proxies can inject the allowed group_ids based on authentication/authorization
- **Security**: Clients cannot access group_ids not specified in the header allowlist

### Error Messages

When a tool call uses a group_id not in the allowlist, the error message includes the allowed group_ids:

```
group_id 'wrong-tenant' is not permitted. Allowed group_ids: ['tenant-a', 'tenant-b']
```

For tools that accept multiple group_ids:

```
Provided group_ids ['wrong1', 'wrong2'] are not permitted. Allowed group_ids: ['tenant-a', 'tenant-b']
```

### Priority Order

The `group_id` is determined based on the header configuration:

**With single group_id in header:**
1. Header group_id is always used (tool parameter ignored)

**With multiple group_ids in header (allowlist):**
1. Tool parameter (if in allowlist)
2. CLI default (if in allowlist)
3. First entry in allowlist (fallback)

**Without header:**
1. Tool parameter
2. CLI default (from `--group-id` argument)
3. Empty string (fallback)

### Example Usage

```bash
# Single group_id - always used
curl "http://localhost:8000/sse" \
  -H "X-Group-Id: tenant-123"

# Multiple group_ids (comma-separated) - acts as allowlist
curl "http://localhost:8000/sse" \
  -H "X-Group-Id: tenant-123, tenant-456, tenant-789"
```

When the header contains `tenant-123, tenant-456, tenant-789`, tool calls can only use one of these three group_ids. Any attempt to use a different group_id will be rejected.

### MCP Client Configuration with Custom Headers

If your MCP client supports custom headers, configure it like this:

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "url": "http://localhost:8000/sse",
      "headers": {
        "X-Group-Id": "tenant-a, tenant-b"
      }
    }
  }
}
```

## Working with JSON Data

The Graphiti MCP server can process structured JSON data through the `add_episode` tool with `source="json"`. This
allows you to automatically extract entities and relationships from structured data:

```

add_episode(
name="Customer Profile",
episode_body="{\"company\": {\"name\": \"Acme Technologies\"}, \"products\": [{\"id\": \"P001\", \"name\": \"CloudSync\"}, {\"id\": \"P002\", \"name\": \"DataMiner\"}]}",
source="json",
source_description="CRM data"
)

```

## Integrating with the Cursor IDE

To integrate the Graphiti MCP Server with the Cursor IDE, follow these steps:

1. Run the Graphiti MCP server using the SSE transport:

```bash
python graphiti_mcp_server.py --transport sse --use-custom-entities --group-id <your_group_id>
```

Hint: specify a `group_id` to namespace graph data. If you do not specify a `group_id`, the server will use "default" as the group_id.

or

```bash
docker compose up
```

2. Configure Cursor to connect to the Graphiti MCP server.

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

3. Add the Graphiti rules to Cursor's User Rules. See [cursor_rules.md](cursor_rules.md) for details.

4. Kick off an agent session in Cursor.

The integration enables AI assistants in Cursor to maintain persistent memory through Graphiti's knowledge graph
capabilities.

## Integrating with Claude Desktop (Docker MCP Server)

The Graphiti MCP Server container uses the SSE MCP transport. Claude Desktop does not natively support SSE, so you'll need to use a gateway like `mcp-remote`.

1.  **Run the Graphiti MCP server using SSE transport**:

    ```bash
    docker compose up
    ```

2.  **(Optional) Install `mcp-remote` globally**:
    If you prefer to have `mcp-remote` installed globally, or if you encounter issues with `npx` fetching the package, you can install it globally. Otherwise, `npx` (used in the next step) will handle it for you.

    ```bash
    npm install -g mcp-remote
    ```

3.  **Configure Claude Desktop**:
    Open your Claude Desktop configuration file (usually `claude_desktop_config.json`) and add or modify the `mcpServers` section as follows:

    ```json
    {
      "mcpServers": {
        "graphiti-memory": {
          // You can choose a different name if you prefer
          "command": "npx", // Or the full path to mcp-remote if npx is not in your PATH
          "args": [
            "mcp-remote",
            "http://localhost:8000/sse" // Ensure this matches your Graphiti server's SSE endpoint
          ]
        }
      }
    }
    ```

    If you already have an `mcpServers` entry, add `graphiti-memory` (or your chosen name) as a new key within it.

4.  **Restart Claude Desktop** for the changes to take effect.

## Requirements

- Python 3.10 or higher
- Neo4j database (version 5.26 or later required)
- OpenAI API key (for LLM operations and embeddings)
- MCP-compatible client

## Telemetry

The Graphiti MCP server uses the Graphiti core library, which includes anonymous telemetry collection. When you initialize the Graphiti MCP server, anonymous usage statistics are collected to help improve the framework.

### What's Collected

- Anonymous identifier and system information (OS, Python version)
- Graphiti version and configuration choices (LLM provider, database backend, embedder type)
- **No personal data, API keys, or actual graph content is ever collected**

### How to Disable

To disable telemetry in the MCP server, set the environment variable:

```bash
export GRAPHITI_TELEMETRY_ENABLED=false
```

Or add it to your `.env` file:

```
GRAPHITI_TELEMETRY_ENABLED=false
```

For complete details about what's collected and why, see the [Telemetry section in the main Graphiti README](https://github.com/getzep/graphiti#telemetry).

## License

This project is licensed under the same license as the parent Graphiti project.

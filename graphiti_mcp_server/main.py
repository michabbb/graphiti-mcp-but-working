"""
Main entry point for Graphiti MCP Server.

This module handles CLI argument parsing, server initialization,
and the main run loop.
"""

import argparse
import asyncio
import logging
import os
import signal
import sys

import uvicorn
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# Load .env file first
load_dotenv()

# CRITICAL: Disable Graphiti telemetry BEFORE any imports from graphiti_core
# Graphiti uses GRAPHITI_TELEMETRY_ENABLED (not POSTHOG_DISABLED)
# Must be AFTER load_dotenv() but BEFORE graphiti_core imports
os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'false'

from graphiti_mcp_server.auth import ALLOWED_NONCE_TOKENS, AuthenticationMiddleware
from graphiti_mcp_server.client import get_graphiti_client, initialize_graphiti, set_config
from graphiti_mcp_server.config import GraphitiConfig, MCPConfig
from graphiti_mcp_server.config.settings import DEFAULT_LLM_MODEL, SMALL_LLM_MODEL
from graphiti_mcp_server.queue import (
    get_shutdown_event,
    get_worker_tasks,
    initialize_redis,
    shutdown_redis,
)
from graphiti_mcp_server.resources import get_status
from graphiti_mcp_server.tools import (
    add_memory,
    clear_graph,
    delete_entity_edge,
    delete_episode,
    delete_everything_by_group_id,
    get_entity_edge,
    get_episodes,
    get_queue_status,
    list_group_ids,
    search_memory_facts,
    search_memory_nodes,
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see middleware calls
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Track shutdown tasks to prevent garbage collection
_shutdown_tasks: list[asyncio.Task] = []

# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a memory service for AI agents built on a knowledge graph. Graphiti performs well
with dynamic data such as user interactions, changing enterprise data, and external information.

Graphiti transforms information into a richly connected knowledge network, allowing you to
capture relationships between concepts, entities, and information. The system organizes data as episodes
(content snippets), nodes (entities), and facts (relationships between entities), creating a dynamic,
queryable memory store that evolves with new information. Graphiti supports multiple data formats, including
structured JSON data, enabling seamless integration with existing data pipelines and systems.

Facts contain temporal metadata, allowing you to track the time of creation and whether a fact is invalid
(superseded by new information).

Key capabilities:
1. Add episodes (text, messages, or JSON) to the knowledge graph with the add_memory tool
2. Search for nodes (entities) in the graph using natural language queries with search_nodes
3. Find relevant facts (relationships between entities) with search_facts
4. Retrieve specific entity edges or episodes by UUID
5. Manage the knowledge graph with tools like delete_episode, delete_entity_edge, delete_everything_by_group_id, and clear_graph

The server connects to a database for persistent storage and uses language models for certain operations.
Each piece of information is organized by group_id, allowing you to maintain separate knowledge domains.

When adding information, provide descriptive names and detailed content to improve search quality.
When searching, use specific queries and consider filtering by group_id for more relevant results.

For optimal performance, ensure the database is properly configured and accessible, and valid
API keys are provided for any language model operations.
"""

# MCP server instance
mcp = FastMCP(
    'Graphiti Agent Memory',
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
)


def register_tools_and_resources() -> None:
    """Register all tools and resources with the MCP server."""
    # Register tools
    mcp.tool()(add_memory)
    mcp.tool()(search_memory_nodes)
    mcp.tool()(search_memory_facts)
    mcp.tool()(get_episodes)
    mcp.tool()(delete_episode)
    mcp.tool()(get_entity_edge)
    mcp.tool()(delete_entity_edge)
    mcp.tool()(list_group_ids)
    mcp.tool()(delete_everything_by_group_id)
    mcp.tool()(get_queue_status)
    mcp.tool()(clear_graph)

    # Register resources
    mcp.resource('http://graphiti/status')(get_status)


async def initialize_server() -> MCPConfig:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    parser = argparse.ArgumentParser(
        description='Run the Graphiti MCP server with optional LLM client'
    )
    parser.add_argument(
        '--group-id',
        help='Namespace for the graph. This is an arbitrary string used to organize related data. '
        'If not provided, a random UUID will be generated.',
    )
    parser.add_argument(
        '--transport',
        choices=['streamable-http', 'sse', 'stdio'],
        default='streamable-http',
        help='Transport to use for communication with the client. (default: streamable-http, the MCP 2025-06-18 standard)',
    )
    parser.add_argument(
        '--model', help=f'Model name to use with the LLM client. (default: {DEFAULT_LLM_MODEL})'
    )
    parser.add_argument(
        '--small-model',
        help=f'Small model name to use with the LLM client. (default: {SMALL_LLM_MODEL})',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        help='Temperature setting for the LLM (0.0-2.0). Lower values make output more deterministic. (default: 0.7)',
    )
    parser.add_argument('--destroy-graph', action='store_true', help='Destroy all Graphiti graphs')
    parser.add_argument(
        '--use-custom-entities',
        action='store_true',
        help='Enable entity extraction using the predefined ENTITY_TYPES',
    )
    parser.add_argument(
        '--host',
        default=os.environ.get('MCP_SERVER_HOST'),
        help='Host to bind the MCP server to (default: MCP_SERVER_HOST environment variable)',
    )

    args = parser.parse_args()

    # Build configuration from CLI arguments and environment variables
    config = GraphitiConfig.from_cli_and_env(args)
    set_config(config)

    # Log the group ID configuration
    if args.group_id:
        logger.info(f'Using provided group_id: {config.group_id}')
    else:
        logger.info(f'Generated random group_id: {config.group_id}')

    # Log entity extraction configuration
    if config.use_custom_entities:
        logger.info('Entity extraction enabled using predefined ENTITY_TYPES')
    else:
        logger.info('Entity extraction disabled (no custom entities will be used)')

    # Register tools and resources
    register_tools_and_resources()

    # Initialize Graphiti
    await initialize_graphiti()

    # Initialize Redis queue manager
    await initialize_redis()

    if args.host:
        logger.info(f'Setting MCP server host to: {args.host}')
        # Set MCP server host from CLI or env
        mcp.settings.host = args.host

    # Configure transport security based on host binding
    # When binding to 0.0.0.0 (all interfaces), we need to configure allowed hosts
    # to prevent DNS rebinding protection from blocking legitimate requests
    allowed_hosts_env = os.environ.get('ALLOWED_HOSTS', '')
    if mcp.settings.host == '0.0.0.0':
        if allowed_hosts_env:
            # Parse comma-separated allowed hosts
            allowed_hosts = [h.strip() for h in allowed_hosts_env.split(',') if h.strip()]
            # Add wildcard port patterns for each host
            allowed_hosts_with_ports = [f'{h}:*' for h in allowed_hosts]
            allowed_origins = [f'http://{h}:*' for h in allowed_hosts] + [
                f'https://{h}:*' for h in allowed_hosts
            ]
            logger.info(f'Configuring transport security with allowed hosts: {allowed_hosts}')
            mcp.settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=allowed_hosts_with_ports,
                allowed_origins=allowed_origins,
            )
        else:
            # No ALLOWED_HOSTS configured - check security requirements
            # When binding to 0.0.0.0 without authentication, require explicit opt-in
            allow_insecure = os.environ.get('ALLOW_UNAUTHENTICATED_PUBLIC_ACCESS', '').lower() == 'true'

            if not ALLOWED_NONCE_TOKENS and not allow_insecure:
                # Dangerous configuration: public binding without authentication
                raise RuntimeError(
                    'SECURITY ERROR: Server is configured to bind to 0.0.0.0 (all interfaces) '
                    'without authentication (MCP_SERVER_NONCE_TOKENS not set) and without '
                    'ALLOWED_HOSTS restriction.\n\n'
                    'This configuration would expose your server to the public internet '
                    'without any access control.\n\n'
                    'To fix this, either:\n'
                    '  1. Set MCP_SERVER_NONCE_TOKENS to enable authentication, or\n'
                    '  2. Set ALLOWED_HOSTS to restrict access to specific hosts, or\n'
                    '  3. Set ALLOW_UNAUTHENTICATED_PUBLIC_ACCESS=true if you explicitly '
                    'want to run without security (NOT RECOMMENDED)\n\n'
                    'For local development, use --host 127.0.0.1 instead of 0.0.0.0'
                )

            # User explicitly opted in or has authentication enabled
            logger.warning(
                'Host is 0.0.0.0 but ALLOWED_HOSTS not set - disabling DNS rebinding protection'
            )
            if not ALLOWED_NONCE_TOKENS:
                logger.warning(
                    'SECURITY WARNING: Running with ALLOW_UNAUTHENTICATED_PUBLIC_ACCESS=true. '
                    'Server accepts requests from ANY source without authentication!'
                )
            mcp.settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=False,
            )

    # Return MCP configuration
    return MCPConfig.from_cli(args)


async def graceful_shutdown(sig: signal.Signals | None = None) -> None:
    """Perform graceful shutdown of all services.

    This function:
    1. Sets the shutdown event to signal workers to stop
    2. Waits for active workers to complete their current task
    3. Closes Redis connection
    4. Closes Graphiti client connection
    """
    shutdown_event = get_shutdown_event()
    worker_tasks = get_worker_tasks()
    graphiti_client = get_graphiti_client()

    if sig:
        logger.info(f'Received signal {sig.name}, initiating graceful shutdown...')
    else:
        logger.info('Initiating graceful shutdown...')

    # Signal all workers to stop
    shutdown_event.set()

    # Wait for active workers to complete (with timeout)
    if worker_tasks:
        logger.info(f'Waiting for {len(worker_tasks)} worker(s) to complete current task...')

        # Give workers time to finish their current task
        # Workers check shutdown_event every 1 second, so this should be enough
        try:
            # Wait up to 30 seconds for workers to finish
            done, pending = await asyncio.wait(
                list(worker_tasks.values()),
                timeout=30.0,
                return_when=asyncio.ALL_COMPLETED
            )

            if pending:
                logger.warning(f'{len(pending)} worker(s) did not finish in time, cancelling...')
                for task in pending:
                    task.cancel()
                # Wait for cancelled tasks to clean up
                await asyncio.gather(*pending, return_exceptions=True)

            logger.info('All workers stopped')
        except Exception as e:
            logger.error(f'Error waiting for workers: {e}')

    # Close Redis connection
    await shutdown_redis()

    # Close Graphiti client connection
    if graphiti_client is not None:
        try:
            await graphiti_client.close()
            logger.info('Graphiti client connection closed')
        except Exception as e:
            logger.error(f'Error closing Graphiti client: {e}')

    logger.info('Graceful shutdown complete')


def setup_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Set up signal handlers for graceful shutdown."""

    def signal_handler(sig: signal.Signals) -> None:
        """Handle shutdown signals."""
        logger.info(f'Received signal {sig.name}')
        # Schedule the shutdown coroutine and track the task
        task = asyncio.create_task(graceful_shutdown(sig))
        _shutdown_tasks.append(task)
        # Remove from list when done to allow cleanup
        task.add_done_callback(lambda t: _shutdown_tasks.remove(t) if t in _shutdown_tasks else None)

    # Register signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(
                sig,
                lambda s=sig: signal_handler(s)
            )
            logger.debug(f'Registered signal handler for {sig.name}')
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            logger.warning(f'Signal handler for {sig.name} not supported on this platform')


async def run_mcp_server() -> None:
    """Run the MCP server in the current event loop."""
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    setup_signal_handlers(loop)

    # Initialize the server
    mcp_config = await initialize_server()

    # Run the server with the configured transport
    logger.info(f'Starting MCP server with transport: {mcp_config.transport}')
    if mcp_config.transport == 'stdio':
        await mcp.run_stdio_async()
    elif mcp_config.transport == 'streamable-http':
        # Get the Streamable HTTP app instance (MCP 2025-06-18 standard)
        http_app = mcp.streamable_http_app()
        logger.debug(f'Streamable HTTP app type: {type(http_app)}')
        logger.debug(f'Streamable HTTP app id: {id(http_app)}')

        # Wrap with authentication middleware if tokens are configured
        if ALLOWED_NONCE_TOKENS:
            logger.info('Wrapping Streamable HTTP app with authentication middleware')
            wrapped_app = AuthenticationMiddleware(http_app)
            logger.info('Authentication middleware applied')
        else:
            logger.warning('No authentication middleware - all requests allowed')
            wrapped_app = http_app

        # Start uvicorn directly with the wrapped app
        logger.info(
            f'Running MCP server with Streamable HTTP transport on {mcp.settings.host}:{mcp.settings.port}/mcp'
        )

        # Create uvicorn config with the wrapped app instance
        config = uvicorn.Config(
            wrapped_app,
            host=mcp.settings.host,
            port=mcp.settings.port,
            log_level='debug',
        )
        server = uvicorn.Server(config)
        await server.serve()
    elif mcp_config.transport == 'sse':
        # Legacy SSE transport (deprecated in MCP 2025-06-18)
        logger.warning('SSE transport is deprecated. Consider using streamable-http instead.')
        sse_app = mcp.sse_app()
        logger.debug(f'SSE app type: {type(sse_app)}')
        logger.debug(f'SSE app id: {id(sse_app)}')

        # Wrap with authentication middleware if tokens are configured
        if ALLOWED_NONCE_TOKENS:
            logger.info('Wrapping SSE app with authentication middleware')
            wrapped_app = AuthenticationMiddleware(sse_app)
            logger.info('Authentication middleware applied')
        else:
            logger.warning('No authentication middleware - all requests allowed')
            wrapped_app = sse_app

        # Start uvicorn directly with the wrapped app
        logger.info(
            f'Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}/sse'
        )

        # Create uvicorn config with the wrapped app instance
        config = uvicorn.Config(
            wrapped_app,
            host=mcp.settings.host,
            port=mcp.settings.port,
            log_level='debug',
        )
        server = uvicorn.Server(config)
        await server.serve()


def main() -> None:
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
    except KeyboardInterrupt:
        logger.info('Server stopped by user (KeyboardInterrupt)')
    except Exception as e:
        logger.error(f'Error initializing Graphiti MCP server: {str(e)}')
        raise
    finally:
        logger.info('Graphiti MCP server shut down')


if __name__ == '__main__':
    main()

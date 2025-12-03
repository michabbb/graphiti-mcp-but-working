"""
Graphiti client module for Graphiti MCP Server.

This module handles the initialization and management of the Graphiti client.
"""

from graphiti_mcp_server.client.graphiti import (
    get_config,
    get_graphiti_client,
    initialize_graphiti,
    set_config,
    set_graphiti_client,
)

__all__ = [
    'get_graphiti_client',
    'set_graphiti_client',
    'get_config',
    'set_config',
    'initialize_graphiti',
]

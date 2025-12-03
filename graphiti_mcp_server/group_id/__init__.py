"""
Group ID context management for Graphiti MCP Server.

This module handles the context variable for allowed group_ids
from HTTP headers and provides helper functions for group_id validation.
"""

from graphiti_mcp_server.group_id.context import (
    get_allowed_group_ids,
    get_effective_group_id,
    get_effective_group_ids,
    is_group_id_allowed,
    set_allowed_group_ids,
)

__all__ = [
    'get_allowed_group_ids',
    'set_allowed_group_ids',
    'is_group_id_allowed',
    'get_effective_group_id',
    'get_effective_group_ids',
]

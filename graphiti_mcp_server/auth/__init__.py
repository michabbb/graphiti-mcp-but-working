"""
Authentication module for Graphiti MCP Server.

This module handles nonce token validation, bearer token extraction,
and authentication middleware.
"""

from graphiti_mcp_server.auth.middleware import AuthenticationMiddleware
from graphiti_mcp_server.auth.nonce import ALLOWED_NONCE_TOKENS, is_nonce_valid
from graphiti_mcp_server.auth.principal import (
    extract_bearer_token,
    get_authenticated_principal,
)

__all__ = [
    'ALLOWED_NONCE_TOKENS',
    'is_nonce_valid',
    'extract_bearer_token',
    'get_authenticated_principal',
    'AuthenticationMiddleware',
]

"""
Authentication middleware for Graphiti MCP Server.
"""

import logging

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from graphiti_mcp_server.auth.nonce import ALLOWED_NONCE_TOKENS
from graphiti_mcp_server.auth.principal import get_authenticated_principal
from graphiti_mcp_server.config.settings import GROUP_ID_HEADER_NAME
from graphiti_mcp_server.group_id.context import set_allowed_group_ids

logger = logging.getLogger(__name__)


class AuthenticationMiddleware:
    """Pure ASGI middleware to enforce nonce token authentication and extract group_id header.

    This is a pure ASGI middleware (not BaseHTTPMiddleware) to avoid conflicts
    with HTTP streaming responses.

    The nonce token must be provided as a query parameter on the FIRST request:
    - /mcp for Streamable HTTP transport (MCP 2025-06-18 standard)
    - /sse for legacy SSE transport

    Subsequent requests in the same session (like /messages/ and /register) are part of
    the authenticated session.

    Additionally, this middleware extracts the X-Group-Id header if present and stores it
    in a context variable. The header can contain multiple comma-separated group_ids which
    act as an allowlist - only these group_ids will be permitted for tool calls.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    def _extract_allowed_group_ids(self, scope: Scope) -> list[str] | None:
        """Extract the X-Group-Id header value(s) from the request scope.

        The header can contain multiple comma-separated group_ids which act as an allowlist.

        Args:
            scope: ASGI connection scope containing headers

        Returns:
            List of allowed group_ids from the X-Group-Id header, or None if not present.
        """
        headers = scope.get('headers', [])
        # Headers are stored as list of tuples: [(name_bytes, value_bytes), ...]
        header_name_lower = GROUP_ID_HEADER_NAME.lower().encode('latin-1')
        for name, value in headers:
            if name.lower() == header_name_lower:
                header_value = value.decode('latin-1')
                # Parse comma-separated values and strip whitespace
                group_ids = [gid.strip() for gid in header_value.split(',') if gid.strip()]
                return group_ids if group_ids else None
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI middleware entry point.

        Args:
            scope: ASGI connection scope
            receive: ASGI receive callable
            send: ASGI send callable
        """
        # Only process HTTP requests
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        # Get path from scope
        path = scope['path']
        method = scope['method']

        # Debug logging
        logger.debug(f'MIDDLEWARE CALLED: {method} {path}')

        # Extract and store the X-Group-Id header if present (supports comma-separated values)
        allowed_group_ids = self._extract_allowed_group_ids(scope)
        if allowed_group_ids:
            logger.info(f'X-Group-Id header found with allowed group_ids: {allowed_group_ids}')

        # Set the context variable for the current request context
        token = set_allowed_group_ids(allowed_group_ids)

        try:
            # Internal MCP endpoints that are part of an authenticated session
            # These endpoints are called AFTER initial connection authentication succeeds
            internal_endpoints = ['/register', '/messages/', '/.well-known/']

            # Check if this is an internal endpoint (part of session, not initial auth)
            is_internal = any(path.startswith(ep) for ep in internal_endpoints)

            # Authenticate the initial connection endpoint
            # - /sse for legacy SSE transport
            # - /mcp for new Streamable HTTP transport (MCP 2025-06-18)
            is_initial_connection = path == '/sse' or path == '/mcp'

            if is_initial_connection or not is_internal:
                # Build Request object for authentication
                request = Request(scope, receive)

                try:
                    # Authenticate the request
                    await get_authenticated_principal(request)
                except HTTPException as exc:
                    # Return error response for authentication failures
                    logger.warning(f'MIDDLEWARE BLOCKED: {method} {path} - {exc.detail}')

                    # Send 401 response directly via ASGI interface
                    response = PlainTextResponse(
                        content=f'Error: {exc.detail}',
                        status_code=exc.status_code,
                        headers=exc.headers or {},
                    )
                    await response(scope, receive, send)
                    return

            # If authentication succeeds (or is internal endpoint), proceed
            await self.app(scope, receive, send)
        finally:
            # Reset the context variable after the request is done
            from graphiti_mcp_server.group_id.context import _allowed_group_ids_var
            _allowed_group_ids_var.reset(token)

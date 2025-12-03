"""
Administrative tools for graph management.
"""

import logging
import os
import secrets
from typing import cast

from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from graphiti_mcp_server.client import get_graphiti_client
from graphiti_mcp_server.models.responses import ErrorResponse, SuccessResponse

logger = logging.getLogger(__name__)


async def clear_graph(password: str) -> SuccessResponse | ErrorResponse:
    """Clear all data from the graph memory and rebuild indices.

    This is a destructive operation that requires password authentication.
    The password must match the CLEAR_GRAPH_PASSWORD environment variable.

    Args:
        password: The password required to authorize clearing the graph.
                  Must match the CLEAR_GRAPH_PASSWORD environment variable.
    """
    graphiti_client = get_graphiti_client()

    # Check if CLEAR_GRAPH_PASSWORD is configured
    expected_password = os.environ.get('CLEAR_GRAPH_PASSWORD')
    if not expected_password:
        return ErrorResponse(
            error='The clear_graph tool is not available because CLEAR_GRAPH_PASSWORD is not set on the MCP server'
        )

    # Validate the password using constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(password, expected_password):
        logger.warning('clear_graph called with invalid password')
        return ErrorResponse(error='Invalid password provided for clear_graph')

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # clear_data is already imported at the top
        await clear_data(client.driver)
        await client.build_indices_and_constraints()
        logger.info('Graph cleared successfully by authenticated user')
        return SuccessResponse(message='Graph cleared successfully and indices rebuilt')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error clearing graph: {error_msg}')
        return ErrorResponse(error=f'Error clearing graph: {error_msg}')

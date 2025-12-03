"""
Entity edge tools for retrieving and deleting edges.
"""

import logging
from typing import Any, cast

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge

from graphiti_mcp_server.client import get_graphiti_client
from graphiti_mcp_server.models.responses import ErrorResponse, SuccessResponse
from graphiti_mcp_server.utils import format_fact_result

logger = logging.getLogger(__name__)


def _is_not_found_error(error: Exception) -> bool:
    """Check if an exception indicates a 'not found' condition.

    Args:
        error: The exception to check

    Returns:
        True if the error indicates the entity was not found
    """
    error_msg = str(error).lower()
    # Common patterns for "not found" errors in Neo4j/graphiti
    not_found_patterns = ['not found', 'does not exist', 'no such', 'cannot find']
    return any(pattern in error_msg for pattern in not_found_patterns)


async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
    """Get an entity edge from the graph memory by its UUID.

    Args:
        uuid: UUID of the entity edge to retrieve
    """
    graphiti_client = get_graphiti_client()

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge directly using the EntityEdge class method
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)

        # Use the format_fact_result function to serialize the edge
        # Return the Python dict directly - MCP will handle serialization
        return format_fact_result(entity_edge)
    except Exception as e:
        error_msg = str(e)
        if _is_not_found_error(e):
            logger.warning(f'Entity edge not found: {uuid}')
            return ErrorResponse(error=f'Entity edge with UUID {uuid} not found')
        logger.error(f'Error getting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error getting entity edge: {error_msg}')


async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an entity edge from the graph memory.

    Args:
        uuid: UUID of the entity edge to delete
    """
    graphiti_client = get_graphiti_client()

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge by UUID
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
        # Delete the edge using its delete method
        await entity_edge.delete(client.driver)
        return SuccessResponse(message=f'Entity edge with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        if _is_not_found_error(e):
            logger.warning(f'Entity edge not found for deletion: {uuid}')
            return ErrorResponse(error=f'Entity edge with UUID {uuid} not found')
        logger.error(f'Error deleting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error deleting entity edge: {error_msg}')

"""
Episode tools for retrieving and deleting episodes.
"""

import logging
from datetime import datetime, timezone
from typing import Any, cast

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodicNode

from graphiti_mcp_server.client import get_config, get_graphiti_client
from graphiti_mcp_server.group_id import get_allowed_group_ids, get_effective_group_id
from graphiti_mcp_server.models.responses import (
    EpisodeSearchResponse,
    ErrorResponse,
    SuccessResponse,
)

logger = logging.getLogger(__name__)


async def get_episodes(
    group_id: str | None = None, last_n: int = 10
) -> list[dict[str, Any]] | EpisodeSearchResponse | ErrorResponse:
    """Get the most recent memory episodes for a specific group.

    Args:
        group_id: ID of the group to retrieve episodes from. If not provided, uses the default group_id.
        last_n: Number of most recent episodes to retrieve (default: 10)
    """
    graphiti_client = get_graphiti_client()
    config = get_config()

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Use get_effective_group_id to determine the group_id, respecting the header allowlist
        effective_group_id = get_effective_group_id(group_id, config.group_id)

        # Check if the group_id was rejected (not in allowlist)
        if effective_group_id is None:
            allowed = get_allowed_group_ids()
            return ErrorResponse(
                error=f"group_id '{group_id}' is not permitted. Allowed group_ids: {allowed}"
            )

        # Log if header-based allowlist is being used
        allowed = get_allowed_group_ids()
        if allowed is not None and group_id is not None and group_id != effective_group_id:
            logger.info(
                f"Using group_id '{effective_group_id}' from allowlist (tool parameter '{group_id}' was overridden)"
            )

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        episodes = await client.retrieve_episodes(
            group_ids=[effective_group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
        )

        if not episodes:
            return EpisodeSearchResponse(
                message=f'No episodes found for group {effective_group_id}', episodes=[]
            )

        # Use Pydantic's model_dump method for EpisodicNode serialization
        formatted_episodes = [
            # Use mode='json' to handle datetime serialization
            episode.model_dump(mode='json')
            for episode in episodes
        ]

        # Return the Python list directly - MCP will handle serialization
        return formatted_episodes
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting episodes: {error_msg}')
        return ErrorResponse(error=f'Error getting episodes: {error_msg}')


async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an episode from the graph memory.

    Args:
        uuid: UUID of the episode to delete
    """
    graphiti_client = get_graphiti_client()

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the episodic node by UUID - EpisodicNode is already imported at the top
        episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
        # Delete the node using its delete method
        await episodic_node.delete(client.driver)
        return SuccessResponse(message=f'Episode with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting episode: {error_msg}')
        return ErrorResponse(error=f'Error deleting episode: {error_msg}')

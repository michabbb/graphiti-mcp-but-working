"""
Memory tool for adding episodes to the knowledge graph.
"""

import asyncio
import logging
from datetime import datetime, timezone

from graphiti_mcp_server.client import get_config, get_graphiti_client
from graphiti_mcp_server.group_id import get_allowed_group_ids, get_effective_group_id
from graphiti_mcp_server.models.queue import QueuedEpisode
from graphiti_mcp_server.models.responses import ErrorResponse, SuccessResponse
from graphiti_mcp_server.queue import get_queue_manager, get_queue_workers, get_worker_tasks
from graphiti_mcp_server.queue.worker import process_episode_queue

logger = logging.getLogger(__name__)


async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    source: str = 'text',
    source_description: str = '',
    uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add an episode to memory. This is the primary way to add information to the graph.

    This function returns immediately and processes the episode addition in the background.
    Episodes for the same group_id are processed sequentially to avoid race conditions.

    Args:
        name (str): Name of the episode
        episode_body (str): The content of the episode to persist to memory. When source='json', this must be a
                           properly escaped JSON string, not a raw Python dictionary. The JSON data will be
                           automatically processed to extract entities and relationships.
        group_id (str, optional): A unique ID for this graph. If not provided, uses the default group_id from CLI
                                 or a generated one.
        source (str, optional): Source type, must be one of:
                               - 'text': For plain text content (default)
                               - 'json': For structured data
                               - 'message': For conversation-style content
        source_description (str, optional): Description of the source
        uuid (str, optional): Optional UUID for the episode

    Examples:
        # Adding plain text content
        add_memory(
            name="Company News",
            episode_body="Acme Corp announced a new product line today.",
            source="text",
            source_description="news article",
            group_id="some_arbitrary_string"
        )

        # Adding structured JSON data
        # NOTE: episode_body must be a properly escaped JSON string. Note the triple backslashes
        add_memory(
            name="Customer Profile",
            episode_body="{\\\"company\\\": {\\\"name\\\": \\\"Acme Technologies\\\"}, \\\"products\\\": [{\\\"id\\\": \\\"P001\\\", \\\"name\\\": \\\"CloudSync\\\"}, {\\\"id\\\": \\\"P002\\\", \\\"name\\\": \\\"DataMiner\\\"}]}",
            source="json",
            source_description="CRM data"
        )

        # Adding message-style content
        add_memory(
            name="Customer Conversation",
            episode_body="user: What's your return policy?\\nassistant: You can return items within 30 days.",
            source="message",
            source_description="chat transcript",
            group_id="some_arbitrary_string"
        )

    Notes:
        When using source='json':
        - The JSON must be a properly escaped string, not a raw Python dictionary
        - The JSON will be automatically processed to extract entities and relationships
        - Complex nested structures are supported (arrays, nested objects, mixed data types), but keep nesting to a minimum
        - Entities will be created from appropriate JSON properties
        - Relationships between entities will be established based on the JSON structure
    """
    graphiti_client = get_graphiti_client()
    queue_manager = get_queue_manager()
    queue_workers = get_queue_workers()
    worker_tasks = get_worker_tasks()
    config = get_config()

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    if queue_manager is None:
        return ErrorResponse(error='Redis queue manager not initialized')

    try:
        # Use get_effective_group_id to determine the group_id, respecting the header allowlist
        group_id_str = get_effective_group_id(group_id, config.group_id)

        # Check if the group_id was rejected (not in allowlist)
        if group_id_str is None:
            allowed = get_allowed_group_ids()
            return ErrorResponse(
                error=f"group_id '{group_id}' is not permitted. Allowed group_ids: {allowed}"
            )

        # Log if header-based allowlist is being used
        allowed = get_allowed_group_ids()
        if allowed is not None and group_id is not None and group_id != group_id_str:
            logger.info(
                f"Using group_id '{group_id_str}' from allowlist (tool parameter '{group_id}' was overridden)"
            )

        # Create a serializable episode object for Redis queue
        queued_episode = QueuedEpisode(
            name=name,
            episode_body=episode_body,
            source=source.lower(),
            source_description=source_description,
            group_id=group_id_str,
            uuid=uuid,
            reference_time=datetime.now(timezone.utc).isoformat(),
            use_custom_entities=config.use_custom_entities,
        )

        # Add to Redis queue
        queue_length = await queue_manager.enqueue(queued_episode)

        # Start a worker for this queue if one isn't already running
        if not queue_workers.get(group_id_str, False):
            task = asyncio.create_task(process_episode_queue(group_id_str))
            worker_tasks[group_id_str] = task

        # Return immediately with a success message
        return SuccessResponse(
            message=f"Episode '{name}' queued for processing (position: {queue_length}, job_id: {queued_episode.job_id})"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error queuing episode task: {error_msg}')
        return ErrorResponse(error=f'Error queuing episode task: {error_msg}')

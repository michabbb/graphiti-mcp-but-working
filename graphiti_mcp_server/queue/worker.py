"""
Queue worker for processing episodes.

This module contains the worker function that processes episodes from the queue.
"""

import asyncio
import logging
from datetime import datetime
from typing import cast

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from graphiti_mcp_server.models.entities import ENTITY_TYPES
from graphiti_mcp_server.queue.state import (
    get_queue_manager,
    get_queue_workers,
    get_shutdown_event,
    get_worker_tasks,
)

logger = logging.getLogger(__name__)


async def process_episode_queue(group_id: str) -> None:
    """Process episodes for a specific group_id sequentially using Redis queue.

    This function runs as a long-lived task that processes episodes
    from the Redis queue one at a time. It supports graceful shutdown
    and crash recovery.
    """
    # Import here to avoid circular imports
    from graphiti_mcp_server.client import get_graphiti_client

    queue_workers = get_queue_workers()
    queue_manager = get_queue_manager()
    shutdown_event = get_shutdown_event()
    worker_tasks = get_worker_tasks()

    logger.info(f'Starting episode queue worker for group_id: {group_id}')
    queue_workers[group_id] = True

    # Track idle time for 30-second inactivity timeout
    idle_seconds = 0
    idle_timeout = 30

    try:
        while not shutdown_event.is_set():
            if queue_manager is None:
                logger.error(f'Queue manager not initialized for group_id {group_id}')
                await asyncio.sleep(1)
                continue

            # Try to get an episode from the queue
            # Short timeout allows us to check shutdown_event regularly
            episode = await queue_manager.dequeue(group_id, timeout=1.0)

            if episode is None:
                # No items in queue, increment idle counter
                idle_seconds += 1

                # Check if we've reached the inactivity timeout
                if idle_seconds >= idle_timeout and not shutdown_event.is_set():
                    queue_length = await queue_manager.get_queue_length(group_id)
                    if queue_length == 0:
                        logger.info(
                            f'Queue for group_id {group_id} is empty after {idle_timeout}s '
                            f'of inactivity, stopping worker'
                        )
                        break
                    # Queue has items now, reset idle counter
                    idle_seconds = 0
                continue

            # Work found - reset idle counter
            idle_seconds = 0

            # Process the episode
            try:
                logger.info(
                    f"Processing episode '{episode.name}' for group_id: {group_id} "
                    f"(job_id: {episode.job_id})"
                )

                graphiti_client = get_graphiti_client()
                if graphiti_client is None:
                    raise RuntimeError('Graphiti client not initialized')

                # Use cast to help the type checker
                client = cast(Graphiti, graphiti_client)

                # Map source string to EpisodeType enum
                source_type = EpisodeType.text
                if episode.source.lower() == 'message':
                    source_type = EpisodeType.message
                elif episode.source.lower() == 'json':
                    source_type = EpisodeType.json

                # Use custom entity types if enabled
                entity_types = ENTITY_TYPES if episode.use_custom_entities else {}

                # Parse reference time
                reference_time = datetime.fromisoformat(episode.reference_time)

                await client.add_episode(
                    name=episode.name,
                    episode_body=episode.episode_body,
                    source=source_type,
                    source_description=episode.source_description,
                    group_id=episode.group_id,
                    uuid=episode.uuid,
                    reference_time=reference_time,
                    entity_types=entity_types,
                )

                logger.info(
                    f"Episode '{episode.name}' processed successfully (job_id: {episode.job_id})"
                )

                # Mark as completed (remove from processing list)
                await queue_manager.complete(group_id, episode)

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error processing episode '{episode.name}' for group_id {group_id}: {error_msg}"
                )
                # Don't requeue on error - the item stays in processing list
                # and will be recovered on next startup
                # This prevents infinite retry loops for bad data
                if queue_manager is not None:
                    await queue_manager.fail(group_id, episode, requeue=False)

    except asyncio.CancelledError:
        logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
        # Don't re-raise - we want to complete the finally block
    except Exception as e:
        logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
    finally:
        queue_workers[group_id] = False
        if group_id in worker_tasks:
            del worker_tasks[group_id]
        logger.info(f'Stopped episode queue worker for group_id: {group_id}')

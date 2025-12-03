"""
Redis initialization for Graphiti MCP Server.

This module handles Redis connection setup and shutdown.
"""

import asyncio
import logging

import redis.asyncio as aioredis

from graphiti_mcp_server.queue.manager import RedisQueueManager
from graphiti_mcp_server.queue.state import (
    get_queue_manager,
    get_queue_workers,
    get_redis_client,
    get_worker_tasks,
    set_queue_manager,
    set_redis_client,
)
from graphiti_mcp_server.queue.worker import process_episode_queue

logger = logging.getLogger(__name__)


async def initialize_redis() -> None:
    """Initialize Redis connection and queue manager."""
    # Import here to avoid circular imports
    from graphiti_mcp_server.client import get_config

    config = get_config()

    try:
        # Create Redis connection
        redis_client = aioredis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            password=config.redis.password,
            decode_responses=False,  # We handle decoding ourselves
        )

        # Test connection
        await redis_client.ping()
        logger.info(f"Connected to Redis at {config.redis.host}:{config.redis.port}")

        # Store client globally
        set_redis_client(redis_client)

        # Create queue manager
        queue_manager = RedisQueueManager(redis_client, config.redis)
        set_queue_manager(queue_manager)

        # Recover any items that were being processed when we crashed
        recovered = await queue_manager.recover_all()
        if recovered > 0:
            logger.info(f"Recovered {recovered} episode(s) from previous crash")

        # Start workers for any existing queues
        await start_workers_for_existing_queues()

    except Exception as e:
        logger.error(f"Failed to initialize Redis: {str(e)}")
        raise


async def start_workers_for_existing_queues() -> None:
    """Start workers for any queues that have pending items."""
    queue_manager = get_queue_manager()
    queue_workers = get_queue_workers()
    worker_tasks = get_worker_tasks()

    if queue_manager is None:
        return

    group_ids = await queue_manager.get_all_group_ids()

    for group_id in group_ids:
        queue_length = await queue_manager.get_queue_length(group_id)
        if queue_length > 0 and not queue_workers.get(group_id, False):
            logger.info(f"Starting worker for existing queue '{group_id}' with {queue_length} items")
            task = asyncio.create_task(process_episode_queue(group_id))
            worker_tasks[group_id] = task


async def shutdown_redis() -> None:
    """Gracefully shutdown Redis connection."""
    redis_client = get_redis_client()

    if redis_client is not None:
        await redis_client.close()
        logger.info("Redis connection closed")

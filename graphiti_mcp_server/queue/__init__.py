"""
Queue management module for Graphiti MCP Server.

This module handles Redis-based episode queuing and background processing.
"""

from graphiti_mcp_server.queue.init import (
    initialize_redis,
    shutdown_redis,
    start_workers_for_existing_queues,
)
from graphiti_mcp_server.queue.manager import RedisQueueManager
from graphiti_mcp_server.queue.state import (
    get_queue_manager,
    get_queue_workers,
    get_redis_client,
    get_shutdown_event,
    get_worker_tasks,
    set_queue_manager,
    set_redis_client,
)
from graphiti_mcp_server.queue.worker import process_episode_queue

__all__ = [
    'RedisQueueManager',
    'process_episode_queue',
    'initialize_redis',
    'shutdown_redis',
    'start_workers_for_existing_queues',
    'get_redis_client',
    'set_redis_client',
    'get_queue_manager',
    'set_queue_manager',
    'get_shutdown_event',
    'get_queue_workers',
    'get_worker_tasks',
]

"""
Global state for queue management.

This module contains the global state variables for Redis client,
queue manager, and worker tracking.
"""

import asyncio
from typing import TYPE_CHECKING

import redis.asyncio as aioredis

if TYPE_CHECKING:
    from graphiti_mcp_server.queue.manager import RedisQueueManager

# Redis client instance
_redis_client: aioredis.Redis | None = None

# Global queue manager instance
_queue_manager: 'RedisQueueManager | None' = None

# Shutdown event for graceful shutdown
_shutdown_event: asyncio.Event = asyncio.Event()

# Track active worker tasks for cleanup
_worker_tasks: dict[str, asyncio.Task] = {}

# Dictionary to track if a worker is running for each group_id
_queue_workers: dict[str, bool] = {}


def get_redis_client() -> aioredis.Redis | None:
    """Get the Redis client instance."""
    return _redis_client


def set_redis_client(client: aioredis.Redis | None) -> None:
    """Set the Redis client instance."""
    global _redis_client
    _redis_client = client


def get_queue_manager() -> 'RedisQueueManager | None':
    """Get the queue manager instance."""
    return _queue_manager


def set_queue_manager(manager: 'RedisQueueManager | None') -> None:
    """Set the queue manager instance."""
    global _queue_manager
    _queue_manager = manager


def get_shutdown_event() -> asyncio.Event:
    """Get the shutdown event."""
    return _shutdown_event


def get_worker_tasks() -> dict[str, asyncio.Task]:
    """Get the worker tasks dictionary."""
    return _worker_tasks


def get_queue_workers() -> dict[str, bool]:
    """Get the queue workers dictionary."""
    return _queue_workers

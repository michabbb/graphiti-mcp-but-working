"""
Redis queue manager for Graphiti MCP Server.

This module provides the RedisQueueManager class for managing episode queues
using Redis for persistence.
"""

import logging

import redis.asyncio as aioredis

from graphiti_mcp_server.config.redis import RedisConfig
from graphiti_mcp_server.models.queue import QueuedEpisode

logger = logging.getLogger(__name__)


class RedisQueueManager:
    """Manages episode queues using Redis for persistence.

    Uses Redis Lists with BRPOPLPUSH pattern for reliable queue processing:
    - Episodes are added to a queue list (LPUSH)
    - Workers pop from queue and push to processing list (BRPOPLPUSH)
    - On success, item is removed from processing list
    - On startup, any items in processing list are moved back to queue (recovery)
    """

    def __init__(self, redis: aioredis.Redis, config: RedisConfig):
        self.redis = redis
        self.config = config
        self._logger = logging.getLogger(__name__)

    def _queue_key(self, group_id: str) -> str:
        """Get the Redis key for a group's episode queue."""
        return f"{self.config.queue_prefix}{group_id}"

    def _processing_key(self, group_id: str) -> str:
        """Get the Redis key for a group's processing list."""
        return f"{self.config.processing_prefix}{group_id}"

    async def enqueue(self, episode: QueuedEpisode) -> int:
        """Add an episode to the queue for its group_id.

        Returns:
            The new queue length after adding the episode.
        """
        queue_key = self._queue_key(episode.group_id)
        # LPUSH adds to the left (head), workers will BRPOP from right (tail) = FIFO
        length = await self.redis.lpush(queue_key, episode.to_json())
        self._logger.info(
            f"Enqueued episode '{episode.name}' for group_id '{episode.group_id}' "
            f"(job_id: {episode.job_id}, queue length: {length})"
        )
        return length

    async def dequeue(self, group_id: str, timeout: float = 1.0) -> QueuedEpisode | None:
        """Pop an episode from the queue and move it to processing list.

        Uses BRPOPLPUSH for atomic move from queue to processing list.
        This ensures that if the worker crashes, the item can be recovered.

        Args:
            group_id: The group_id to dequeue from
            timeout: How long to wait for an item (seconds)

        Returns:
            The dequeued episode, or None if timeout
        """
        queue_key = self._queue_key(group_id)
        processing_key = self._processing_key(group_id)

        # BRPOPLPUSH: Pop from queue tail, push to processing list head
        # This is atomic and ensures no data loss
        result = await self.redis.brpoplpush(queue_key, processing_key, timeout=int(timeout))

        if result is None:
            return None

        try:
            # Result is bytes, decode and parse
            if isinstance(result, bytes):
                result = result.decode('utf-8')
            return QueuedEpisode.from_json(result)
        except Exception as e:
            self._logger.error(f"Failed to parse episode from queue: {e}")
            # Remove invalid item from processing list
            await self.redis.lrem(processing_key, 1, result)
            return None

    async def complete(self, group_id: str, episode: QueuedEpisode) -> None:
        """Mark an episode as completed by removing it from the processing list."""
        processing_key = self._processing_key(group_id)
        # Remove the item from the processing list
        await self.redis.lrem(processing_key, 1, episode.to_json())
        self._logger.debug(f"Completed episode '{episode.name}' (job_id: {episode.job_id})")

    async def fail(self, group_id: str, episode: QueuedEpisode, requeue: bool = True) -> None:
        """Handle a failed episode.

        Args:
            group_id: The group_id
            episode: The failed episode
            requeue: If True, move back to queue for retry. If False, just remove from processing.
        """
        processing_key = self._processing_key(group_id)

        # Remove from processing list
        await self.redis.lrem(processing_key, 1, episode.to_json())

        if requeue:
            # Re-add to queue (at the end, so it's processed last)
            queue_key = self._queue_key(group_id)
            await self.redis.rpush(queue_key, episode.to_json())
            self._logger.warning(
                f"Re-queued failed episode '{episode.name}' (job_id: {episode.job_id})"
            )
        else:
            self._logger.error(
                f"Dropped failed episode '{episode.name}' (job_id: {episode.job_id})"
            )

    async def recover_processing(self, group_id: str) -> int:
        """Recover any items left in the processing list (from a crash).

        Moves all items from processing list back to the queue.

        Returns:
            Number of items recovered.
        """
        processing_key = self._processing_key(group_id)
        queue_key = self._queue_key(group_id)

        recovered = 0
        while True:
            # Pop from processing, push to queue (at the front for priority)
            item = await self.redis.rpoplpush(processing_key, queue_key)
            if item is None:
                break
            recovered += 1

        if recovered > 0:
            self._logger.info(
                f"Recovered {recovered} episode(s) from processing list for group_id '{group_id}'"
            )
        return recovered

    async def get_queue_length(self, group_id: str) -> int:
        """Get the current queue length for a group_id."""
        queue_key = self._queue_key(group_id)
        return await self.redis.llen(queue_key)

    async def get_processing_items(self, group_id: str) -> list[QueuedEpisode]:
        """Get all items currently being processed for a group_id."""
        processing_key = self._processing_key(group_id)
        items = await self.redis.lrange(processing_key, 0, -1)

        result = []
        for item in items:
            try:
                if isinstance(item, bytes):
                    item = item.decode('utf-8')
                result.append(QueuedEpisode.from_json(item))
            except Exception as e:
                self._logger.error(f"Failed to parse processing item: {e}")
        return result

    async def get_processing_count(self, group_id: str) -> int:
        """Get the count of items currently being processed for a group_id."""
        processing_key = self._processing_key(group_id)
        return await self.redis.llen(processing_key)

    async def get_all_group_ids(self) -> list[str]:
        """Get all group_ids that have queues (either pending or processing)."""
        # Scan for all queue keys
        group_ids = set()

        # Check queue keys
        async for key in self.redis.scan_iter(match=f"{self.config.queue_prefix}*"):
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            group_id = key.replace(self.config.queue_prefix, '')
            group_ids.add(group_id)

        # Check processing keys
        async for key in self.redis.scan_iter(match=f"{self.config.processing_prefix}*"):
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            group_id = key.replace(self.config.processing_prefix, '')
            group_ids.add(group_id)

        return list(group_ids)

    async def recover_all(self) -> int:
        """Recover all processing items across all group_ids.

        Called on startup to handle any items that were being processed
        when the server crashed.

        Returns:
            Total number of items recovered.
        """
        total_recovered = 0
        group_ids = await self.get_all_group_ids()

        for group_id in group_ids:
            recovered = await self.recover_processing(group_id)
            total_recovered += recovered

        if total_recovered > 0:
            self._logger.info(f"Total recovered episodes from crash: {total_recovered}")

        return total_recovered

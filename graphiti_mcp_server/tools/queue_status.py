"""
Queue status tool for monitoring episode processing.
"""

import logging

from graphiti_mcp_server.models.responses import (
    ErrorResponse,
    ProcessingJobInfo,
    QueueInfo,
    QueueStatusResponse,
)
from graphiti_mcp_server.queue import get_queue_manager, get_queue_workers

logger = logging.getLogger(__name__)


async def get_queue_status() -> QueueStatusResponse | ErrorResponse:
    """Get the current status of all episode processing queues.

    This tool provides visibility into the background processing queues that handle
    episodes after they are submitted via add_memory. It shows:
    - Total number of pending tasks across all queues (waiting to be processed)
    - Total number of processing tasks (currently being processed by workers)
    - Number of active worker processes
    - Per-group_id queue details including pending tasks, processing tasks, and worker status

    Jobs go through these states:
    1. pending: Waiting in queue to be picked up by a worker
    2. processing: Currently being processed (extracting entities, creating facts, etc.)
    3. completed: Finished and removed from queue (not shown in status)

    Use this tool to monitor the processing status after adding memories, especially
    when adding multiple episodes in succession.
    """
    queue_manager = get_queue_manager()
    queue_workers = get_queue_workers()

    if queue_manager is None:
        return ErrorResponse(error='Redis queue manager not initialized')

    try:
        queues_info: list[QueueInfo] = []
        total_pending = 0
        total_processing = 0
        active_workers = 0

        # Get all known group_ids from Redis queues and active workers
        redis_group_ids = await queue_manager.get_all_group_ids()
        all_group_ids = set(redis_group_ids) | set(queue_workers.keys())

        for group_id in sorted(all_group_ids):
            # Get queue size from Redis (pending items)
            pending = await queue_manager.get_queue_length(group_id)
            # Get processing items from Redis
            processing_items = await queue_manager.get_processing_items(group_id)
            processing_count = len(processing_items)
            # Check if worker is active
            worker_active = queue_workers.get(group_id, False)

            total_pending += pending
            total_processing += processing_count
            if worker_active:
                active_workers += 1

            # Convert processing items to ProcessingJobInfo
            processing_jobs: list[ProcessingJobInfo] = [
                ProcessingJobInfo(
                    job_id=item.job_id,
                    name=item.name,
                    group_id=item.group_id,
                    queued_at=item.queued_at,
                )
                for item in processing_items
            ]

            queues_info.append(
                QueueInfo(
                    group_id=group_id,
                    pending_tasks=pending,
                    processing_tasks=processing_count,
                    processing_jobs=processing_jobs,
                    worker_active=worker_active,
                )
            )

        logger.info(
            f'Queue status: {total_pending} pending, {total_processing} processing, {active_workers} active workers'
        )

        return QueueStatusResponse(
            total_pending=total_pending,
            total_processing=total_processing,
            active_workers=active_workers,
            queues=queues_info,
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting queue status: {error_msg}')
        return ErrorResponse(error=f'Error getting queue status: {error_msg}')

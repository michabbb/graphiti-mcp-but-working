"""
Response type definitions for Graphiti MCP Server.

These TypedDict classes define the structure of API responses.
"""

from typing import Any

from typing_extensions import TypedDict


class ErrorResponse(TypedDict):
    error: str


class SuccessResponse(TypedDict):
    message: str


class NodeResult(TypedDict):
    uuid: str
    name: str
    summary: str
    labels: list[str]
    group_id: str
    created_at: str
    attributes: dict[str, Any]


class NodeSearchResponse(TypedDict):
    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    message: str
    facts: list[dict[str, Any]]


class EpisodeSearchResponse(TypedDict):
    message: str
    episodes: list[dict[str, Any]]


class StatusResponse(TypedDict):
    status: str
    message: str


class ProcessingJobInfo(TypedDict):
    job_id: str
    name: str
    group_id: str
    queued_at: str


class QueueInfo(TypedDict):
    group_id: str
    pending_tasks: int
    processing_tasks: int
    processing_jobs: list[ProcessingJobInfo]
    worker_active: bool


class QueueStatusResponse(TypedDict):
    total_pending: int
    total_processing: int
    active_workers: int
    queues: list[QueueInfo]


class GroupIdResult(TypedDict):
    entity: str
    group_id: str


class GroupIdListResponse(TypedDict):
    message: str
    group_ids: list[GroupIdResult]


class DeleteGroupResponse(TypedDict):
    """Response for delete_everything_by_group_id tool."""
    message: str
    deleted_episodes: int
    deleted_nodes: int
    deleted_entity_edges: int

"""
Queue-related models for Graphiti MCP Server.
"""

import uuid as uuid_module
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class QueuedEpisode(BaseModel):
    """Serializable episode data for Redis queue."""

    job_id: str = Field(default_factory=lambda: str(uuid_module.uuid4()))
    name: str
    episode_body: str
    source: str  # 'text', 'json', 'message'
    source_description: str
    group_id: str
    uuid: str | None = None
    reference_time: str  # ISO format datetime
    use_custom_entities: bool = False
    queued_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        """Serialize to JSON string for Redis storage."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> 'QueuedEpisode':
        """Deserialize from JSON string."""
        return cls.model_validate_json(data)

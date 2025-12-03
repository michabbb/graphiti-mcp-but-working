"""
Redis configuration for Graphiti MCP Server.
"""

import os

from pydantic import BaseModel


class RedisConfig(BaseModel):
    """Configuration for Redis connection."""

    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    password: str | None = None
    queue_prefix: str = 'graphiti:episodes:'
    processing_prefix: str = 'graphiti:processing:'

    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Create Redis configuration from environment variables."""
        return cls(
            host=os.environ.get('REDIS_HOST', 'localhost'),
            port=int(os.environ.get('REDIS_PORT', '6379')),
            db=int(os.environ.get('REDIS_DB', '0')),
            password=os.environ.get('REDIS_PASSWORD'),
            queue_prefix=os.environ.get('REDIS_QUEUE_PREFIX', 'graphiti:episodes:'),
            processing_prefix=os.environ.get('REDIS_PROCESSING_PREFIX', 'graphiti:processing:'),
        )

"""
Global settings and constants for the Graphiti MCP Server.
"""

import os

# Default model names
DEFAULT_LLM_MODEL = 'gpt-4.1-mini'
SMALL_LLM_MODEL = 'gpt-4.1-nano'
DEFAULT_EMBEDDER_MODEL = 'text-embedding-3-small'

# Semaphore limit for concurrent Graphiti operations.
# Decrease this if you're experiencing 429 rate limit errors from your LLM provider.
# Increase if you have high rate limits.
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 10))

# HTTP Header name for passing allowed group_ids
# When this header is present in the request, its value(s) define the allowed group_ids.
# Multiple group_ids can be provided as comma-separated values.
# Only these group_ids will be permitted for tool calls - any other group_id will be rejected.
GROUP_ID_HEADER_NAME = 'X-Group-Id'

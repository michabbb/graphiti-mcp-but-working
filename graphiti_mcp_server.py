#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
"""

import argparse
import asyncio
import contextvars
import json
import logging
import os
import secrets
import signal
import sys
import uuid as uuid_module
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, cast
from typing_extensions import TypedDict

import redis.asyncio as aioredis

from dotenv import load_dotenv

# Load .env file first
load_dotenv()

# CRITICAL: Disable Graphiti telemetry BEFORE any imports from graphiti_core
# Graphiti uses GRAPHITI_TELEMETRY_ENABLED (not POSTHOG_DISABLED)
# Must be AFTER load_dotenv() but BEFORE graphiti_core imports
os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'false'

from fastapi import Depends, HTTPException, Request, status
from fastapi.security.utils import get_authorization_scheme_param
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send
import uvicorn

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data


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

# Context variable to store the allowed group_ids from the header for the current request
# This allows tool functions to access the header values without direct access to the HTTP request
# When set, this acts as an allowlist - only group_ids in this list are permitted
_allowed_group_ids_var: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    'allowed_group_ids', default=None
)


def get_allowed_group_ids() -> list[str] | None:
    """Get the allowed group_ids from the HTTP header if set for the current request.
    
    Returns:
        List of allowed group_ids from the X-Group-Id header, or None if not set.
    """
    return _allowed_group_ids_var.get()


def set_allowed_group_ids(group_ids: list[str] | None) -> contextvars.Token[list[str] | None]:
    """Set the allowed group_ids from the HTTP header for the current request context.
    
    Args:
        group_ids: List of allowed group_id values from the header, or None to clear.
        
    Returns:
        A token that can be used to reset the context variable.
    """
    return _allowed_group_ids_var.set(group_ids)


def is_group_id_allowed(group_id: str) -> bool:
    """Check if a group_id is allowed based on the header allowlist.
    
    Args:
        group_id: The group_id to check.
        
    Returns:
        True if the group_id is allowed (or no allowlist is set), False otherwise.
    """
    allowed = get_allowed_group_ids()
    if allowed is None:
        # No allowlist set, all group_ids are allowed
        return True
    return group_id in allowed


def get_effective_group_id(
    tool_group_id: str | None, 
    default_group_id: str | None = None
) -> str | None:
    """Get the effective group_id to use for an operation, respecting the header allowlist.
    
    Behavior:
    - If X-Group-Id header is set with one or more group_ids, these act as an allowlist
    - If only one group_id is in the allowlist, it is used as the fixed group_id
    - If multiple group_ids are in the allowlist, the tool parameter must be in the list
    - If the tool parameter is not in the allowlist, returns None (rejected)
    
    Priority order (when allowlist has multiple entries):
    1. Tool-provided group_id (must be in allowlist)
    2. Default group_id (must be in allowlist)
    3. First entry in allowlist as fallback
    
    Args:
        tool_group_id: The group_id passed in the tool call parameters.
        default_group_id: The default group_id from config (usually from CLI --group-id).
        
    Returns:
        The effective group_id to use for the operation, or None if rejected.
    """
    allowed = get_allowed_group_ids()
    
    if allowed is None:
        # No allowlist set - use original priority: tool param > default > empty string
        if tool_group_id is not None:
            return tool_group_id
        if default_group_id is not None:
            return default_group_id
        return ''
    
    # Allowlist is set
    if len(allowed) == 1:
        # Single entry in allowlist - use it as fixed group_id, ignore tool parameter
        return allowed[0]
    
    # Multiple entries in allowlist - tool parameter must be validated
    if tool_group_id is not None:
        if tool_group_id in allowed:
            return tool_group_id
        else:
            # Tool parameter not in allowlist - rejected
            return None
    
    # No tool parameter provided
    if default_group_id is not None and default_group_id in allowed:
        return default_group_id
    
    # Fall back to first entry in allowlist
    return allowed[0]


def get_effective_group_ids(
    tool_group_ids: list[str] | None,
    default_group_id: str | None = None
) -> list[str] | None:
    """Get the effective group_ids to use for search operations, respecting the header allowlist.
    
    Behavior:
    - If X-Group-Id header is set, only group_ids in the allowlist are permitted
    - If tool provides group_ids, they are filtered to only include allowed ones
    - If the result would be empty (all tool group_ids rejected), returns None
    
    Args:
        tool_group_ids: List of group_ids passed in the tool call parameters.
        default_group_id: The default group_id from config (usually from CLI --group-id).
        
    Returns:
        List of effective group_ids to use, or None if all were rejected.
    """
    allowed = get_allowed_group_ids()
    
    if allowed is None:
        # No allowlist set - use original behavior
        if tool_group_ids is not None:
            return tool_group_ids
        if default_group_id is not None:
            return [default_group_id]
        return []
    
    # Allowlist is set
    if tool_group_ids is not None:
        # Filter tool_group_ids to only include allowed ones
        filtered = [gid for gid in tool_group_ids if gid in allowed]
        if not filtered:
            # All provided group_ids were rejected
            return None
        return filtered
    
    # No tool group_ids provided - use the full allowlist
    return allowed


class Requirement(BaseModel):
    """A Requirement represents a specific need, feature, or functionality that a product or service must fulfill.

    Always ensure an edge is created between the requirement and the project it belongs to, and clearly indicate on the
    edge that the requirement is a requirement.

    Instructions for identifying and extracting requirements:
    1. Look for explicit statements of needs or necessities ("We need X", "X is required", "X must have Y")
    2. Identify functional specifications that describe what the system should do
    3. Pay attention to non-functional requirements like performance, security, or usability criteria
    4. Extract constraints or limitations that must be adhered to
    5. Focus on clear, specific, and measurable requirements rather than vague wishes
    6. Capture the priority or importance if mentioned ("critical", "high priority", etc.)
    7. Include any dependencies between requirements when explicitly stated
    8. Preserve the original intent and scope of the requirement
    9. Categorize requirements appropriately based on their domain or function
    """

    project_name: str = Field(
        ...,
        description='The name of the project to which the requirement belongs.',
    )
    description: str = Field(
        ...,
        description='Description of the requirement. Only use information mentioned in the context to write this description.',
    )


class Preference(BaseModel):
    """A Preference represents a user's expressed like, dislike, or preference for something.

    Instructions for identifying and extracting preferences:
    1. Look for explicit statements of preference such as "I like/love/enjoy/prefer X" or "I don't like/hate/dislike X"
    2. Pay attention to comparative statements ("I prefer X over Y")
    3. Consider the emotional tone when users mention certain topics
    4. Extract only preferences that are clearly expressed, not assumptions
    5. Categorize the preference appropriately based on its domain (food, music, brands, etc.)
    6. Include relevant qualifiers (e.g., "likes spicy food" rather than just "likes food")
    7. Only extract preferences directly stated by the user, not preferences of others they mention
    8. Provide a concise but specific description that captures the nature of the preference
    """

    category: str = Field(
        ...,
        description="The category of the preference. (e.g., 'Brands', 'Food', 'Music')",
    )
    description: str = Field(
        ...,
        description='Brief description of the preference. Only use information mentioned in the context to write this description.',
    )


class Procedure(BaseModel):
    """A Procedure informing the agent what actions to take or how to perform in certain scenarios. Procedures are typically composed of several steps.

    Instructions for identifying and extracting procedures:
    1. Look for sequential instructions or steps ("First do X, then do Y")
    2. Identify explicit directives or commands ("Always do X when Y happens")
    3. Pay attention to conditional statements ("If X occurs, then do Y")
    4. Extract procedures that have clear beginning and end points
    5. Focus on actionable instructions rather than general information
    6. Preserve the original sequence and dependencies between steps
    7. Include any specified conditions or triggers for the procedure
    8. Capture any stated purpose or goal of the procedure
    9. Summarize complex procedures while maintaining critical details
    """

    description: str = Field(
        ...,
        description='Brief description of the procedure. Only use information mentioned in the context to write this description.',
    )


ENTITY_TYPES: dict[str, BaseModel] = {
    'Requirement': Requirement,  # type: ignore
    'Preference': Preference,  # type: ignore
    'Procedure': Procedure,  # type: ignore
}


# Type definitions for API responses
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


# Server configuration classes
# The configuration system has a hierarchy:
# - GraphitiConfig is the top-level configuration
#   - LLMConfig handles all OpenAI/LLM related settings
#   - EmbedderConfig manages embedding settings
#   - Neo4jConfig manages database connection details
#   - Various other settings like group_id and feature flags
# Configuration values are loaded from:
# 1. Default values in the class definitions
# 2. Environment variables (loaded via load_dotenv())
# 3. Command line arguments (which override environment variables)
class GraphitiLLMConfig(BaseModel):
    """Configuration for the LLM client.

    Centralizes all LLM-specific configuration parameters including API keys and model selection.
    """

    api_key: str | None = None
    model: str = DEFAULT_LLM_MODEL
    small_model: str = SMALL_LLM_MODEL
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> 'GraphitiLLMConfig':
        """Create LLM configuration from environment variables."""
        # Get model from environment, or use default if not set or empty
        model_env = os.environ.get('MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_LLM_MODEL

        # Get small_model from environment, or use default if not set or empty
        small_model_env = os.environ.get('SMALL_MODEL_NAME', '')
        small_model = small_model_env if small_model_env.strip() else SMALL_LLM_MODEL

        # Log if empty model was provided
        if model_env == '':
            logger.debug(
                f'MODEL_NAME environment variable not set, using default: {DEFAULT_LLM_MODEL}'
            )
        elif not model_env.strip():
            logger.warning(
                f'Empty MODEL_NAME environment variable, using default: {DEFAULT_LLM_MODEL}'
            )

        return cls(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model=model,
            small_model=small_model,
            temperature=float(os.environ.get('LLM_TEMPERATURE', '0.0')),
        )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiLLMConfig':
        """Create LLM configuration from CLI arguments, falling back to environment variables."""
        # Start with environment-based config
        config = cls.from_env()

        # CLI arguments override environment variables when provided
        if hasattr(args, 'model') and args.model:
            # Only use CLI model if it's not empty
            if args.model.strip():
                config.model = args.model
            else:
                # Log that empty model was provided and default is used
                logger.warning(f'Empty model name provided, using default: {DEFAULT_LLM_MODEL}')

        if hasattr(args, 'small_model') and args.small_model:
            if args.small_model.strip():
                config.small_model = args.small_model
            else:
                logger.warning(f'Empty small_model name provided, using default: {SMALL_LLM_MODEL}')

        if hasattr(args, 'temperature') and args.temperature is not None:
            config.temperature = args.temperature

        return config

    def create_client(self) -> LLMClient:
        """Create an LLM client based on this configuration.

        Returns:
            LLMClient instance
        """
        if not self.api_key:
            raise ValueError('OPENAI_API_KEY must be set when using OpenAI API')

        llm_client_config = LLMConfig(
            api_key=self.api_key, model=self.model, small_model=self.small_model
        )

        # Only set temperature if not using gpt-5, o1, or o3 models (they don't support temperature)
        if not any(x in self.model.lower() for x in ['gpt-5', 'o1', 'o3']):
            llm_client_config.temperature = self.temperature

        # Disable reasoning and verbosity parameters for gpt-5, o1, o3 models
        return OpenAIClient(config=llm_client_config, reasoning=None, verbosity=None)


class GraphitiEmbedderConfig(BaseModel):
    """Configuration for the embedder client.

    Centralizes all embedding-related configuration parameters.
    """

    model: str = DEFAULT_EMBEDDER_MODEL
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> 'GraphitiEmbedderConfig':
        """Create embedder configuration from environment variables."""

        # Get model from environment, or use default if not set or empty
        model_env = os.environ.get('EMBEDDER_MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_EMBEDDER_MODEL

        return cls(
            model=model,
            api_key=os.environ.get('OPENAI_API_KEY'),
        )

    def create_client(self) -> EmbedderClient | None:
        if not self.api_key:
            return None

        embedder_config = OpenAIEmbedderConfig(api_key=self.api_key, embedding_model=self.model)

        return OpenAIEmbedder(config=embedder_config)


class Neo4jConfig(BaseModel):
    """Configuration for Neo4j database connection."""

    uri: str = 'bolt://localhost:7687'
    user: str = 'neo4j'
    password: str = 'password'

    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Create Neo4j configuration from environment variables."""
        return cls(
            uri=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
            user=os.environ.get('NEO4J_USER', 'neo4j'),
            password=os.environ.get('NEO4J_PASSWORD', 'password'),
        )


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


class GraphitiConfig(BaseModel):
    """Configuration for Graphiti client.

    Centralizes all configuration parameters for the Graphiti client.
    """

    llm: GraphitiLLMConfig = Field(default_factory=GraphitiLLMConfig)
    embedder: GraphitiEmbedderConfig = Field(default_factory=GraphitiEmbedderConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    group_id: str | None = None
    use_custom_entities: bool = False
    destroy_graph: bool = False

    @classmethod
    def from_env(cls) -> 'GraphitiConfig':
        """Create a configuration instance from environment variables."""
        return cls(
            llm=GraphitiLLMConfig.from_env(),
            embedder=GraphitiEmbedderConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
            redis=RedisConfig.from_env(),
        )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiConfig':
        """Create configuration from CLI arguments, falling back to environment variables."""
        # Start with environment configuration
        config = cls.from_env()

        # Apply CLI overrides
        if args.group_id:
            config.group_id = args.group_id
        else:
            config.group_id = 'default'

        config.use_custom_entities = args.use_custom_entities
        config.destroy_graph = args.destroy_graph

        # Update LLM config using CLI args
        config.llm = GraphitiLLMConfig.from_cli_and_env(args)

        return config


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    transport: str = 'streamable-http'  # Default to Streamable HTTP transport (MCP 2025-06-18)

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> 'MCPConfig':
        """Create MCP configuration from CLI arguments."""
        return cls(transport=args.transport)


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see middleware calls
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# --- Security Configuration ---
# Nonce tokens for query parameter authentication
# Set MCP_SERVER_NONCE_TOKENS environment variable with comma-separated tokens
# Example: export MCP_SERVER_NONCE_TOKENS="token1,token2,token3"
ALLOWED_NONCE_TOKENS = [
    token.strip()
    for token in os.environ.get('MCP_SERVER_NONCE_TOKENS', '').split(',')
    if token.strip()
]
if ALLOWED_NONCE_TOKENS:
    logger.info(
        '🔒 AUTHENTICATION ENABLED: Loaded %d nonce token(s) for authentication',
        len(ALLOWED_NONCE_TOKENS),
    )
    logger.info('🔒 Requests must include valid nonce token (?nonce=<token>)')
else:
    logger.warning('⚠️  AUTHENTICATION DISABLED: MCP_SERVER_NONCE_TOKENS not configured')
    logger.warning('⚠️  Server will accept ALL requests without authentication!')


def _is_nonce_valid(candidate: str) -> bool:
    """Validate a nonce token using constant-time comparison.

    Args:
        candidate: The nonce token to validate

    Returns:
        True if the nonce is valid, False otherwise
    """
    for token in ALLOWED_NONCE_TOKENS:
        if secrets.compare_digest(candidate, token):
            return True
    return False


def _extract_bearer_token(request: Request) -> str | None:
    """Extract bearer token from Authorization header.

    Args:
        request: The FastAPI request object

    Returns:
        The bearer token if found, None otherwise

    Raises:
        HTTPException: If authorization scheme is not 'bearer'
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return None
    scheme, param = get_authorization_scheme_param(auth_header)
    if not scheme:
        return None
    if scheme.lower() != 'bearer':
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Unsupported authorization scheme',
            headers={'WWW-Authenticate': 'Bearer'},
        )
    return param


async def get_authenticated_principal(request: Request) -> dict[str, str]:
    """Authenticate the incoming request using nonce token.

    This function checks for a nonce token in the query parameters.
    If nonce tokens are not configured (ALLOWED_NONCE_TOKENS is empty),
    authentication is bypassed and a default principal is returned.

    Args:
        request: The FastAPI request object

    Returns:
        A dictionary containing authentication information:
        - client_id: Identifier for the authenticated client
        - auth_method: The authentication method used
        - scope: OAuth scope (empty for nonce auth)

    Raises:
        HTTPException: If authentication fails (invalid nonce or missing credentials)
    """
    # If no nonce tokens are configured, bypass authentication
    if not ALLOWED_NONCE_TOKENS:
        return {
            'client_id': 'unauthenticated',
            'auth_method': 'none',
            'scope': '',
        }

    # Check for nonce token in query parameters
    nonce = request.query_params.get('nonce')
    if nonce is not None:
        if _is_nonce_valid(nonce):
            logger.info('✓ Authentication successful with nonce token')
            return {
                'client_id': f'nonce:{nonce}',
                'auth_method': 'query_token',
                'scope': '',
            }
        logger.warning('✗ Authentication failed: Invalid nonce token provided')
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid nonce token',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    # If nonce tokens are configured but no valid nonce was provided, reject
    logger.warning('✗ Authentication failed: No nonce token provided')
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Not authenticated',
        headers={'WWW-Authenticate': 'Bearer'},
    )


class AuthenticationMiddleware:
    """Pure ASGI middleware to enforce nonce token authentication and extract group_id header.

    This is a pure ASGI middleware (not BaseHTTPMiddleware) to avoid conflicts
    with HTTP streaming responses.

    The nonce token must be provided as a query parameter on the FIRST request:
    - /mcp for Streamable HTTP transport (MCP 2025-06-18 standard)
    - /sse for legacy SSE transport
    
    Subsequent requests in the same session (like /messages/ and /register) are part of
    the authenticated session.
    
    Additionally, this middleware extracts the X-Group-Id header if present and stores it
    in a context variable. The header can contain multiple comma-separated group_ids which
    act as an allowlist - only these group_ids will be permitted for tool calls.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    def _extract_allowed_group_ids(self, scope: Scope) -> list[str] | None:
        """Extract the X-Group-Id header value(s) from the request scope.
        
        The header can contain multiple comma-separated group_ids which act as an allowlist.
        
        Args:
            scope: ASGI connection scope containing headers
            
        Returns:
            List of allowed group_ids from the X-Group-Id header, or None if not present.
        """
        headers = scope.get('headers', [])
        # Headers are stored as list of tuples: [(name_bytes, value_bytes), ...]
        header_name_lower = GROUP_ID_HEADER_NAME.lower().encode('latin-1')
        for name, value in headers:
            if name.lower() == header_name_lower:
                header_value = value.decode('latin-1')
                # Parse comma-separated values and strip whitespace
                group_ids = [gid.strip() for gid in header_value.split(',') if gid.strip()]
                return group_ids if group_ids else None
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI middleware entry point.

        Args:
            scope: ASGI connection scope
            receive: ASGI receive callable
            send: ASGI send callable
        """
        # Only process HTTP requests
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        # Get path from scope
        path = scope['path']
        method = scope['method']

        # Debug logging
        logger.debug(f'🔍 MIDDLEWARE CALLED: {method} {path}')

        # Extract and store the X-Group-Id header if present (supports comma-separated values)
        allowed_group_ids = self._extract_allowed_group_ids(scope)
        if allowed_group_ids:
            logger.info(f'🔑 X-Group-Id header found with allowed group_ids: {allowed_group_ids}')
        
        # Set the context variable for the current request context
        token = set_allowed_group_ids(allowed_group_ids)
        
        try:
            # Internal MCP endpoints that are part of an authenticated session
            # These endpoints are called AFTER initial connection authentication succeeds
            internal_endpoints = ['/register', '/messages/', '/.well-known/']

            # Check if this is an internal endpoint (part of session, not initial auth)
            is_internal = any(path.startswith(ep) for ep in internal_endpoints)

            # Authenticate the initial connection endpoint
            # - /sse for legacy SSE transport
            # - /mcp for new Streamable HTTP transport (MCP 2025-06-18)
            is_initial_connection = path == '/sse' or path == '/mcp'
            
            if is_initial_connection or not is_internal:
                # Build Request object for authentication
                from starlette.requests import Request
                request = Request(scope, receive)

                try:
                    # Authenticate the request
                    await get_authenticated_principal(request)
                except HTTPException as exc:
                    # Return error response for authentication failures
                    logger.warning(f'🔍 MIDDLEWARE BLOCKED: {method} {path} - {exc.detail}')

                    # Send 401 response directly via ASGI interface
                    response = PlainTextResponse(
                        content=f'Error: {exc.detail}',
                        status_code=exc.status_code,
                        headers=exc.headers or {},
                    )
                    await response(scope, receive, send)
                    return

            # If authentication succeeds (or is internal endpoint), proceed
            await self.app(scope, receive, send)
        finally:
            # Reset the context variable after the request is done
            _allowed_group_ids_var.reset(token)


# Create global config instance - will be properly initialized later
config = GraphitiConfig()

# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a memory service for AI agents built on a knowledge graph. Graphiti performs well
with dynamic data such as user interactions, changing enterprise data, and external information.

Graphiti transforms information into a richly connected knowledge network, allowing you to 
capture relationships between concepts, entities, and information. The system organizes data as episodes 
(content snippets), nodes (entities), and facts (relationships between entities), creating a dynamic, 
queryable memory store that evolves with new information. Graphiti supports multiple data formats, including 
structured JSON data, enabling seamless integration with existing data pipelines and systems.

Facts contain temporal metadata, allowing you to track the time of creation and whether a fact is invalid 
(superseded by new information).

Key capabilities:
1. Add episodes (text, messages, or JSON) to the knowledge graph with the add_memory tool
2. Search for nodes (entities) in the graph using natural language queries with search_nodes
3. Find relevant facts (relationships between entities) with search_facts
4. Retrieve specific entity edges or episodes by UUID
5. Manage the knowledge graph with tools like delete_episode, delete_entity_edge, delete_everything_by_group_id, and clear_graph

The server connects to a database for persistent storage and uses language models for certain operations. 
Each piece of information is organized by group_id, allowing you to maintain separate knowledge domains.

When adding information, provide descriptive names and detailed content to improve search quality. 
When searching, use specific queries and consider filtering by group_id for more relevant results.

For optimal performance, ensure the database is properly configured and accessible, and valid 
API keys are provided for any language model operations.
"""

# MCP server instance
mcp = FastMCP(
    'Graphiti Agent Memory',
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
)

# Store SSE app instance globally to ensure middleware is applied to the same instance
_sse_app_instance = None

# Initialize Graphiti client
graphiti_client: Graphiti | None = None

# Redis client and queue manager (initialized later)
redis_client: aioredis.Redis | None = None

# Shutdown event for graceful shutdown
shutdown_event: asyncio.Event = asyncio.Event()

# Track active worker tasks for cleanup
worker_tasks: dict[str, asyncio.Task] = {}


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


# Global queue manager instance
queue_manager: RedisQueueManager | None = None

# Dictionary to track if a worker is running for each group_id
queue_workers: dict[str, bool] = {}


async def initialize_redis() -> None:
    """Initialize Redis connection and queue manager."""
    global redis_client, queue_manager, config

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

        # Create queue manager
        queue_manager = RedisQueueManager(redis_client, config.redis)

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
    global queue_manager, queue_workers

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
    global redis_client

    if redis_client is not None:
        await redis_client.close()
        logger.info("Redis connection closed")


async def initialize_graphiti():
    """Initialize the Graphiti client with the configured settings."""
    global graphiti_client, config

    try:
        # Create LLM client if possible
        llm_client = config.llm.create_client()
        if not llm_client and config.use_custom_entities:
            # If custom entities are enabled, we must have an LLM client
            raise ValueError('OPENAI_API_KEY must be set when custom entities are enabled')

        # Validate Neo4j configuration
        if not config.neo4j.uri or not config.neo4j.user or not config.neo4j.password:
            raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

        embedder_client = config.embedder.create_client()

        # Initialize Graphiti client
        graphiti_client = Graphiti(
            uri=config.neo4j.uri,
            user=config.neo4j.user,
            password=config.neo4j.password,
            llm_client=llm_client,
            embedder=embedder_client,
            max_coroutines=SEMAPHORE_LIMIT,
        )

        # Destroy graph if requested
        if config.destroy_graph:
            logger.info('Destroying graph...')
            await clear_data(graphiti_client.driver)

        # Initialize the graph database with Graphiti's indices
        await graphiti_client.build_indices_and_constraints()
        logger.info('Graphiti client initialized successfully')

        # Log configuration details for transparency
        if llm_client:
            logger.info(f'Using OpenAI model: {config.llm.model}')
            logger.info(f'Using temperature: {config.llm.temperature}')
        else:
            logger.info('No LLM client configured - entity extraction will be limited')

        logger.info(f'Using group_id: {config.group_id}')
        logger.info(
            f'Custom entity extraction: {"enabled" if config.use_custom_entities else "disabled"}'
        )
        logger.info(f'Using concurrency limit: {SEMAPHORE_LIMIT}')

    except Exception as e:
        logger.error(f'Failed to initialize Graphiti: {str(e)}')
        raise


def format_fact_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an entity edge into a readable result.

    Since EntityEdge is a Pydantic BaseModel, we can use its built-in serialization capabilities.

    Args:
        edge: The EntityEdge to format

    Returns:
        A dictionary representation of the edge with serialized dates and excluded embeddings
    """
    result = edge.model_dump(
        mode='json',
        exclude={
            'fact_embedding',
        },
    )
    result.get('attributes', {}).pop('fact_embedding', None)
    return result


async def process_episode_queue(group_id: str):
    """Process episodes for a specific group_id sequentially using Redis queue.

    This function runs as a long-lived task that processes episodes
    from the Redis queue one at a time. It supports graceful shutdown
    and crash recovery.
    """
    global queue_workers, queue_manager, graphiti_client, shutdown_event

    logger.info(f'Starting episode queue worker for group_id: {group_id}')
    queue_workers[group_id] = True

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
                # No items in queue, check if we should continue
                queue_length = await queue_manager.get_queue_length(group_id)
                if queue_length == 0 and not shutdown_event.is_set():
                    # Queue is empty and no shutdown - wait a bit and check again
                    # After 30 seconds of no activity, stop the worker
                    # It will be restarted when new items are added
                    await asyncio.sleep(1)
                    queue_length = await queue_manager.get_queue_length(group_id)
                    if queue_length == 0:
                        logger.info(f'Queue for group_id {group_id} is empty, stopping worker')
                        break
                continue

            # Process the episode
            try:
                logger.info(f"Processing episode '{episode.name}' for group_id: {group_id} (job_id: {episode.job_id})")

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

                logger.info(f"Episode '{episode.name}' processed successfully (job_id: {episode.job_id})")

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


@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    source: str = 'text',
    source_description: str = '',
    uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add an episode to memory. This is the primary way to add information to the graph.

    This function returns immediately and processes the episode addition in the background.
    Episodes for the same group_id are processed sequentially to avoid race conditions.

    Args:
        name (str): Name of the episode
        episode_body (str): The content of the episode to persist to memory. When source='json', this must be a
                           properly escaped JSON string, not a raw Python dictionary. The JSON data will be
                           automatically processed to extract entities and relationships.
        group_id (str, optional): A unique ID for this graph. If not provided, uses the default group_id from CLI
                                 or a generated one.
        source (str, optional): Source type, must be one of:
                               - 'text': For plain text content (default)
                               - 'json': For structured data
                               - 'message': For conversation-style content
        source_description (str, optional): Description of the source
        uuid (str, optional): Optional UUID for the episode

    Examples:
        # Adding plain text content
        add_memory(
            name="Company News",
            episode_body="Acme Corp announced a new product line today.",
            source="text",
            source_description="news article",
            group_id="some_arbitrary_string"
        )

        # Adding structured JSON data
        # NOTE: episode_body must be a properly escaped JSON string. Note the triple backslashes
        add_memory(
            name="Customer Profile",
            episode_body="{\\\"company\\\": {\\\"name\\\": \\\"Acme Technologies\\\"}, \\\"products\\\": [{\\\"id\\\": \\\"P001\\\", \\\"name\\\": \\\"CloudSync\\\"}, {\\\"id\\\": \\\"P002\\\", \\\"name\\\": \\\"DataMiner\\\"}]}",
            source="json",
            source_description="CRM data"
        )

        # Adding message-style content
        add_memory(
            name="Customer Conversation",
            episode_body="user: What's your return policy?\nassistant: You can return items within 30 days.",
            source="message",
            source_description="chat transcript",
            group_id="some_arbitrary_string"
        )

    Notes:
        When using source='json':
        - The JSON must be a properly escaped string, not a raw Python dictionary
        - The JSON will be automatically processed to extract entities and relationships
        - Complex nested structures are supported (arrays, nested objects, mixed data types), but keep nesting to a minimum
        - Entities will be created from appropriate JSON properties
        - Relationships between entities will be established based on the JSON structure
    """
    global graphiti_client, queue_manager, queue_workers, worker_tasks

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    if queue_manager is None:
        return ErrorResponse(error='Redis queue manager not initialized')

    try:
        # Use get_effective_group_id to determine the group_id, respecting the header allowlist
        group_id_str = get_effective_group_id(group_id, config.group_id)

        # Check if the group_id was rejected (not in allowlist)
        if group_id_str is None:
            allowed = get_allowed_group_ids()
            return ErrorResponse(
                error=f"group_id '{group_id}' is not permitted. Allowed group_ids: {allowed}"
            )

        # Log if header-based allowlist is being used
        allowed = get_allowed_group_ids()
        if allowed is not None and group_id is not None and group_id != group_id_str:
            logger.info(
                f"Using group_id '{group_id_str}' from allowlist (tool parameter '{group_id}' was overridden)"
            )

        # Create a serializable episode object for Redis queue
        queued_episode = QueuedEpisode(
            name=name,
            episode_body=episode_body,
            source=source.lower(),
            source_description=source_description,
            group_id=group_id_str,
            uuid=uuid,
            reference_time=datetime.now(timezone.utc).isoformat(),
            use_custom_entities=config.use_custom_entities,
        )

        # Add to Redis queue
        queue_length = await queue_manager.enqueue(queued_episode)

        # Start a worker for this queue if one isn't already running
        if not queue_workers.get(group_id_str, False):
            task = asyncio.create_task(process_episode_queue(group_id_str))
            worker_tasks[group_id_str] = task

        # Return immediately with a success message
        return SuccessResponse(
            message=f"Episode '{name}' queued for processing (position: {queue_length}, job_id: {queued_episode.job_id})"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error queuing episode task: {error_msg}')
        return ErrorResponse(error=f'Error queuing episode task: {error_msg}')


@mcp.tool()
async def search_memory_nodes(
    query: str,
    group_ids: list[str] | None = None,
    max_nodes: int = 10,
    center_node_uuid: str | None = None,
    entity: str = '',  # cursor seems to break with None
) -> NodeSearchResponse | ErrorResponse:
    """Search the graph memory for relevant node summaries.
    These contain a summary of all of a node's relationships with other nodes.

    Note: entity is a single entity type to filter results (permitted: "Preference", "Procedure").

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_nodes: Maximum number of nodes to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
        entity: Optional single entity type to filter results (permitted: "Preference", "Procedure")
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Use get_effective_group_ids to determine allowed group_ids, respecting the header allowlist
        effective_group_ids = get_effective_group_ids(group_ids, config.group_id)
        
        # Check if all provided group_ids were rejected (not in allowlist)
        if effective_group_ids is None:
            allowed = get_allowed_group_ids()
            return ErrorResponse(
                error=f"Provided group_ids {group_ids} are not permitted. Allowed group_ids: {allowed}"
            )
        
        # Log if header-based allowlist is filtering the group_ids
        allowed = get_allowed_group_ids()
        if allowed is not None and group_ids is not None:
            filtered_out = [gid for gid in group_ids if gid not in allowed]
            if filtered_out:
                logger.info(
                    f"Filtered out group_ids not in allowlist: {filtered_out}"
                )

        # Configure the search
        if center_node_uuid is not None:
            search_config = NODE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
        else:
            search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        search_config.limit = max_nodes

        filters = SearchFilters()
        if entity != '':
            filters.node_labels = [entity]

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Perform the search using the _search method
        search_results = await client._search(
            query=query,
            config=search_config,
            group_ids=effective_group_ids,
            center_node_uuid=center_node_uuid,
            search_filter=filters,
        )

        if not search_results.nodes:
            return NodeSearchResponse(message='No relevant nodes found', nodes=[])

        # Format the node results
        formatted_nodes: list[NodeResult] = [
            {
                'uuid': node.uuid,
                'name': node.name,
                'summary': node.summary if hasattr(node, 'summary') else '',
                'labels': node.labels if hasattr(node, 'labels') else [],
                'group_id': node.group_id,
                'created_at': node.created_at.isoformat(),
                'attributes': node.attributes if hasattr(node, 'attributes') else {},
            }
            for node in search_results.nodes
        ]

        return NodeSearchResponse(message='Nodes retrieved successfully', nodes=formatted_nodes)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching nodes: {error_msg}')
        return ErrorResponse(error=f'Error searching nodes: {error_msg}')


@mcp.tool()
async def search_memory_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
    center_node_uuid: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Search the graph memory for relevant facts.

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_facts: Maximum number of facts to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Validate max_facts parameter
        if max_facts <= 0:
            return ErrorResponse(error='max_facts must be a positive integer')

        # Use get_effective_group_ids to determine allowed group_ids, respecting the header allowlist
        effective_group_ids = get_effective_group_ids(group_ids, config.group_id)
        
        # Check if all provided group_ids were rejected (not in allowlist)
        if effective_group_ids is None:
            allowed = get_allowed_group_ids()
            return ErrorResponse(
                error=f"Provided group_ids {group_ids} are not permitted. Allowed group_ids: {allowed}"
            )
        
        # Log if header-based allowlist is filtering the group_ids
        allowed = get_allowed_group_ids()
        if allowed is not None and group_ids is not None:
            filtered_out = [gid for gid in group_ids if gid not in allowed]
            if filtered_out:
                logger.info(
                    f"Filtered out group_ids not in allowlist: {filtered_out}"
                )

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        relevant_edges = await client.search(
            group_ids=effective_group_ids,
            query=query,
            num_results=max_facts,
            center_node_uuid=center_node_uuid,
        )

        if not relevant_edges:
            return FactSearchResponse(message='No relevant facts found', facts=[])

        facts = [format_fact_result(edge) for edge in relevant_edges]
        return FactSearchResponse(message='Facts retrieved successfully', facts=facts)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching facts: {error_msg}')
        return ErrorResponse(error=f'Error searching facts: {error_msg}')


@mcp.tool()
async def list_group_ids(limit: int = 500) -> GroupIdListResponse | ErrorResponse:
    """Return distinct group IDs present on nodes and relationships in the graph."""
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    safe_limit = max(1, min(limit, 500))
    query = (
        "MATCH (n) WHERE n.group_id IS NOT NULL "
        "RETURN DISTINCT 'node' AS entity, n.group_id AS group_id "
        "LIMIT $node_limit "
        "UNION ALL "
        "MATCH ()-[r]-() WHERE r.group_id IS NOT NULL "
        "RETURN DISTINCT 'relationship' AS entity, r.group_id AS group_id "
        "LIMIT $relationship_limit"
    )

    try:
        assert graphiti_client is not None
        client = cast(Graphiti, graphiti_client)
        records, _, _ = await client.driver.execute_query(
            query,
            params={'node_limit': safe_limit, 'relationship_limit': safe_limit},
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error listing group IDs: {error_msg}')
        return ErrorResponse(error=f'Error listing group IDs: {error_msg}')

    group_ids: list[GroupIdResult] = [
        {'entity': record['entity'], 'group_id': record['group_id']}
        for record in records
        if record['group_id'] is not None
    ]

    if not group_ids:
        return GroupIdListResponse(message='No group IDs found', group_ids=[])

    return GroupIdListResponse(message='Group IDs retrieved successfully', group_ids=group_ids)


@mcp.tool()
async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an entity edge from the graph memory.

    Args:
        uuid: UUID of the entity edge to delete
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge by UUID
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
        # Delete the edge using its delete method
        await entity_edge.delete(client.driver)
        return SuccessResponse(message=f'Entity edge with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error deleting entity edge: {error_msg}')


@mcp.tool()
async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an episode from the graph memory.

    Args:
        uuid: UUID of the episode to delete
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the episodic node by UUID - EpisodicNode is already imported at the top
        episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
        # Delete the node using its delete method
        await episodic_node.delete(client.driver)
        return SuccessResponse(message=f'Episode with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting episode: {error_msg}')
        return ErrorResponse(error=f'Error deleting episode: {error_msg}')


@mcp.tool()
async def delete_everything_by_group_id(group_id: str) -> DeleteGroupResponse | ErrorResponse:
    """Delete all data associated with a group_id from the graph memory.

    This tool completely removes all episodes, nodes, and entity edges (facts/relationships)
    that belong to the specified group_id. After deletion, the group_id will no longer
    appear in list_group_ids results.

    This is a destructive operation that cannot be undone.

    Args:
        group_id: The group_id whose data should be completely deleted.
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Validate group_id against allowlist (respects X-Group-Id header)
        effective_group_id = get_effective_group_id(group_id, config.group_id)

        if effective_group_id is None:
            allowed = get_allowed_group_ids()
            return ErrorResponse(
                error=f"group_id '{group_id}' is not permitted. Allowed group_ids: {allowed}"
            )

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # First, count all entities that will be deleted for reporting
        count_query = """
        OPTIONAL MATCH (episode:Episodic {group_id: $group_id})
        WITH count(episode) AS episode_count
        OPTIONAL MATCH (node {group_id: $group_id})
        WHERE NOT node:Episodic
        WITH episode_count, count(node) AS node_count
        OPTIONAL MATCH ()-[rel {group_id: $group_id}]->()
        RETURN episode_count, node_count, count(rel) AS edge_count
        """

        count_records, _, _ = await client.driver.execute_query(
            count_query,
            params={'group_id': effective_group_id}
        )

        episode_count = 0
        node_count = 0
        edge_count = 0

        if count_records:
            record = count_records[0]
            episode_count = record['episode_count'] or 0
            node_count = record['node_count'] or 0
            edge_count = record['edge_count'] or 0

        # Delete all relationships with this group_id first
        delete_edges_query = """
        MATCH ()-[r {group_id: $group_id}]->()
        DELETE r
        """
        await client.driver.execute_query(
            delete_edges_query,
            params={'group_id': effective_group_id}
        )

        # Delete all nodes (including episodes) with this group_id
        # Using DETACH DELETE to handle any remaining relationships
        delete_nodes_query = """
        MATCH (n {group_id: $group_id})
        DETACH DELETE n
        """
        await client.driver.execute_query(
            delete_nodes_query,
            params={'group_id': effective_group_id}
        )

        total_deleted = episode_count + node_count + edge_count
        logger.info(
            f"Deleted all data for group_id '{effective_group_id}': "
            f"{episode_count} episodes, {node_count} nodes, {edge_count} edges"
        )

        return DeleteGroupResponse(
            message=f"Group '{effective_group_id}' completely removed ({total_deleted} total entities deleted)",
            deleted_episodes=episode_count,
            deleted_nodes=node_count,
            deleted_entity_edges=edge_count
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting data by group_id: {error_msg}')
        return ErrorResponse(error=f'Error deleting data by group_id: {error_msg}')


@mcp.tool()
async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
    """Get an entity edge from the graph memory by its UUID.

    Args:
        uuid: UUID of the entity edge to retrieve
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge directly using the EntityEdge class method
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)

        # Use the format_fact_result function to serialize the edge
        # Return the Python dict directly - MCP will handle serialization
        return format_fact_result(entity_edge)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error getting entity edge: {error_msg}')


@mcp.tool()
async def get_episodes(
    group_id: str | None = None, last_n: int = 10
) -> list[dict[str, Any]] | EpisodeSearchResponse | ErrorResponse:
    """Get the most recent memory episodes for a specific group.

    Args:
        group_id: ID of the group to retrieve episodes from. If not provided, uses the default group_id.
        last_n: Number of most recent episodes to retrieve (default: 10)
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Use get_effective_group_id to determine the group_id, respecting the header allowlist
        effective_group_id = get_effective_group_id(group_id, config.group_id)
        
        # Check if the group_id was rejected (not in allowlist)
        if effective_group_id is None:
            allowed = get_allowed_group_ids()
            return ErrorResponse(
                error=f"group_id '{group_id}' is not permitted. Allowed group_ids: {allowed}"
            )
        
        # Log if header-based allowlist is being used
        allowed = get_allowed_group_ids()
        if allowed is not None and group_id is not None and group_id != effective_group_id:
            logger.info(
                f"Using group_id '{effective_group_id}' from allowlist (tool parameter '{group_id}' was overridden)"
            )

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        episodes = await client.retrieve_episodes(
            group_ids=[effective_group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
        )

        if not episodes:
            return EpisodeSearchResponse(
                message=f'No episodes found for group {effective_group_id}', episodes=[]
            )

        # Use Pydantic's model_dump method for EpisodicNode serialization
        formatted_episodes = [
            # Use mode='json' to handle datetime serialization
            episode.model_dump(mode='json')
            for episode in episodes
        ]

        # Return the Python list directly - MCP will handle serialization
        return formatted_episodes
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting episodes: {error_msg}')
        return ErrorResponse(error=f'Error getting episodes: {error_msg}')


@mcp.tool()
async def clear_graph(password: str) -> SuccessResponse | ErrorResponse:
    """Clear all data from the graph memory and rebuild indices.

    This is a destructive operation that requires password authentication.
    The password must match the CLEAR_GRAPH_PASSWORD environment variable.

    Args:
        password: The password required to authorize clearing the graph.
                  Must match the CLEAR_GRAPH_PASSWORD environment variable.
    """
    global graphiti_client

    # Check if CLEAR_GRAPH_PASSWORD is configured
    expected_password = os.environ.get('CLEAR_GRAPH_PASSWORD')
    if not expected_password:
        return ErrorResponse(
            error='The clear_graph tool is not available because CLEAR_GRAPH_PASSWORD is not set on the MCP server'
        )

    # Validate the password using constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(password, expected_password):
        logger.warning('clear_graph called with invalid password')
        return ErrorResponse(error='Invalid password provided for clear_graph')

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # clear_data is already imported at the top
        await clear_data(client.driver)
        await client.build_indices_and_constraints()
        logger.info('Graph cleared successfully by authenticated user')
        return SuccessResponse(message='Graph cleared successfully and indices rebuilt')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error clearing graph: {error_msg}')
        return ErrorResponse(error=f'Error clearing graph: {error_msg}')


@mcp.tool()
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
    global queue_manager, queue_workers

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


@mcp.resource('http://graphiti/status')
async def get_status() -> StatusResponse:
    """Get the status of the Graphiti MCP server and Neo4j connection."""
    global graphiti_client

    if graphiti_client is None:
        return StatusResponse(status='error', message='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Test database connection
        await client.driver.client.verify_connectivity()  # type: ignore

        return StatusResponse(
            status='ok', message='Graphiti MCP server is running and connected to Neo4j'
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error checking Neo4j connection: {error_msg}')
        return StatusResponse(
            status='error',
            message=f'Graphiti MCP server is running but Neo4j connection failed: {error_msg}',
        )


async def initialize_server() -> MCPConfig:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    global config

    parser = argparse.ArgumentParser(
        description='Run the Graphiti MCP server with optional LLM client'
    )
    parser.add_argument(
        '--group-id',
        help='Namespace for the graph. This is an arbitrary string used to organize related data. '
        'If not provided, a random UUID will be generated.',
    )
    parser.add_argument(
        '--transport',
        choices=['streamable-http', 'sse', 'stdio'],
        default='streamable-http',
        help='Transport to use for communication with the client. (default: streamable-http, the MCP 2025-06-18 standard)',
    )
    parser.add_argument(
        '--model', help=f'Model name to use with the LLM client. (default: {DEFAULT_LLM_MODEL})'
    )
    parser.add_argument(
        '--small-model',
        help=f'Small model name to use with the LLM client. (default: {SMALL_LLM_MODEL})',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        help='Temperature setting for the LLM (0.0-2.0). Lower values make output more deterministic. (default: 0.7)',
    )
    parser.add_argument('--destroy-graph', action='store_true', help='Destroy all Graphiti graphs')
    parser.add_argument(
        '--use-custom-entities',
        action='store_true',
        help='Enable entity extraction using the predefined ENTITY_TYPES',
    )
    parser.add_argument(
        '--host',
        default=os.environ.get('MCP_SERVER_HOST'),
        help='Host to bind the MCP server to (default: MCP_SERVER_HOST environment variable)',
    )

    args = parser.parse_args()

    # Build configuration from CLI arguments and environment variables
    config = GraphitiConfig.from_cli_and_env(args)

    # Log the group ID configuration
    if args.group_id:
        logger.info(f'Using provided group_id: {config.group_id}')
    else:
        logger.info(f'Generated random group_id: {config.group_id}')

    # Log entity extraction configuration
    if config.use_custom_entities:
        logger.info('Entity extraction enabled using predefined ENTITY_TYPES')
    else:
        logger.info('Entity extraction disabled (no custom entities will be used)')

    # Initialize Graphiti
    await initialize_graphiti()

    # Initialize Redis queue manager
    await initialize_redis()

    if args.host:
        logger.info(f'Setting MCP server host to: {args.host}')
        # Set MCP server host from CLI or env
        mcp.settings.host = args.host

    # Configure transport security based on host binding
    # When binding to 0.0.0.0 (all interfaces), we need to configure allowed hosts
    # to prevent DNS rebinding protection from blocking legitimate requests
    allowed_hosts_env = os.environ.get('ALLOWED_HOSTS', '')
    if mcp.settings.host == '0.0.0.0':
        if allowed_hosts_env:
            # Parse comma-separated allowed hosts
            allowed_hosts = [h.strip() for h in allowed_hosts_env.split(',') if h.strip()]
            # Add wildcard port patterns for each host
            allowed_hosts_with_ports = [f'{h}:*' for h in allowed_hosts]
            allowed_origins = [f'http://{h}:*' for h in allowed_hosts] + [
                f'https://{h}:*' for h in allowed_hosts
            ]
            logger.info(f'Configuring transport security with allowed hosts: {allowed_hosts}')
            mcp.settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=allowed_hosts_with_ports,
                allowed_origins=allowed_origins,
            )
        else:
            # No ALLOWED_HOSTS configured - disable DNS rebinding protection
            logger.warning(
                '⚠️  Host is 0.0.0.0 but ALLOWED_HOSTS not set - disabling DNS rebinding protection'
            )
            mcp.settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=False,
            )

    # Return MCP configuration
    return MCPConfig.from_cli(args)


async def graceful_shutdown(sig: signal.Signals | None = None) -> None:
    """Perform graceful shutdown of all services.

    This function:
    1. Sets the shutdown event to signal workers to stop
    2. Waits for active workers to complete their current task
    3. Closes Redis connection
    4. Closes Graphiti client connection
    """
    global shutdown_event, worker_tasks, graphiti_client

    if sig:
        logger.info(f'Received signal {sig.name}, initiating graceful shutdown...')
    else:
        logger.info('Initiating graceful shutdown...')

    # Signal all workers to stop
    shutdown_event.set()

    # Wait for active workers to complete (with timeout)
    if worker_tasks:
        logger.info(f'Waiting for {len(worker_tasks)} worker(s) to complete current task...')

        # Give workers time to finish their current task
        # Workers check shutdown_event every 1 second, so this should be enough
        try:
            # Wait up to 30 seconds for workers to finish
            done, pending = await asyncio.wait(
                list(worker_tasks.values()),
                timeout=30.0,
                return_when=asyncio.ALL_COMPLETED
            )

            if pending:
                logger.warning(f'{len(pending)} worker(s) did not finish in time, cancelling...')
                for task in pending:
                    task.cancel()
                # Wait for cancelled tasks to clean up
                await asyncio.gather(*pending, return_exceptions=True)

            logger.info('All workers stopped')
        except Exception as e:
            logger.error(f'Error waiting for workers: {e}')

    # Close Redis connection
    await shutdown_redis()

    # Close Graphiti client connection
    if graphiti_client is not None:
        try:
            await graphiti_client.close()
            logger.info('Graphiti client connection closed')
        except Exception as e:
            logger.error(f'Error closing Graphiti client: {e}')

    logger.info('Graceful shutdown complete')


def setup_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Set up signal handlers for graceful shutdown."""

    def signal_handler(sig: signal.Signals) -> None:
        """Handle shutdown signals."""
        logger.info(f'Received signal {sig.name}')
        # Schedule the shutdown coroutine
        asyncio.create_task(graceful_shutdown(sig))

    # Register signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(
                sig,
                lambda s=sig: signal_handler(s)
            )
            logger.debug(f'Registered signal handler for {sig.name}')
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            logger.warning(f'Signal handler for {sig.name} not supported on this platform')


async def run_mcp_server():
    """Run the MCP server in the current event loop."""
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    setup_signal_handlers(loop)

    # Initialize the server
    mcp_config = await initialize_server()

    # Run the server with the configured transport
    logger.info(f'Starting MCP server with transport: {mcp_config.transport}')
    if mcp_config.transport == 'stdio':
        await mcp.run_stdio_async()
    elif mcp_config.transport == 'streamable-http':
        # Get the Streamable HTTP app instance (MCP 2025-06-18 standard)
        http_app = mcp.streamable_http_app()
        logger.debug(f'🔍 Streamable HTTP app type: {type(http_app)}')
        logger.debug(f'🔍 Streamable HTTP app id: {id(http_app)}')

        # Wrap with authentication middleware if tokens are configured
        if ALLOWED_NONCE_TOKENS:
            logger.info('🔒 Wrapping Streamable HTTP app with authentication middleware')
            wrapped_app = AuthenticationMiddleware(http_app)
            logger.info('🔒 Authentication middleware applied')
        else:
            logger.warning('⚠️  No authentication middleware - all requests allowed')
            wrapped_app = http_app

        # Start uvicorn directly with the wrapped app
        logger.info(
            f'Running MCP server with Streamable HTTP transport on {mcp.settings.host}:{mcp.settings.port}/mcp'
        )

        # Create uvicorn config with the wrapped app instance
        config = uvicorn.Config(
            wrapped_app,
            host=mcp.settings.host,
            port=mcp.settings.port,
            log_level='debug',
        )
        server = uvicorn.Server(config)
        await server.serve()
    elif mcp_config.transport == 'sse':
        # Legacy SSE transport (deprecated in MCP 2025-06-18)
        logger.warning('⚠️  SSE transport is deprecated. Consider using streamable-http instead.')
        sse_app = mcp.sse_app()
        logger.debug(f'🔍 SSE app type: {type(sse_app)}')
        logger.debug(f'🔍 SSE app id: {id(sse_app)}')

        # Wrap with authentication middleware if tokens are configured
        if ALLOWED_NONCE_TOKENS:
            logger.info('🔒 Wrapping SSE app with authentication middleware')
            wrapped_app = AuthenticationMiddleware(sse_app)
            logger.info('🔒 Authentication middleware applied')
        else:
            logger.warning('⚠️  No authentication middleware - all requests allowed')
            wrapped_app = sse_app

        # Start uvicorn directly with the wrapped app
        logger.info(
            f'Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}/sse'
        )

        # Create uvicorn config with the wrapped app instance
        config = uvicorn.Config(
            wrapped_app,
            host=mcp.settings.host,
            port=mcp.settings.port,
            log_level='debug',
        )
        server = uvicorn.Server(config)
        await server.serve()


def main():
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
    except KeyboardInterrupt:
        logger.info('Server stopped by user (KeyboardInterrupt)')
    except Exception as e:
        logger.error(f'Error initializing Graphiti MCP server: {str(e)}')
        raise
    finally:
        logger.info('Graphiti MCP server shut down')


if __name__ == '__main__':
    main()

"""
Graphiti client initialization and management.

This module provides functions for initializing and accessing the Graphiti client.
"""

import logging

from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from graphiti_mcp_server.config import GraphitiConfig
from graphiti_mcp_server.config.settings import SEMAPHORE_LIMIT

logger = logging.getLogger(__name__)

# Global Graphiti client instance
_graphiti_client: Graphiti | None = None

# Global configuration instance
_config: GraphitiConfig = GraphitiConfig()


def get_graphiti_client() -> Graphiti | None:
    """Get the Graphiti client instance."""
    return _graphiti_client


def set_graphiti_client(client: Graphiti | None) -> None:
    """Set the Graphiti client instance."""
    global _graphiti_client
    _graphiti_client = client


def get_config() -> GraphitiConfig:
    """Get the configuration instance."""
    return _config


def set_config(config: GraphitiConfig) -> None:
    """Set the configuration instance."""
    global _config
    _config = config


async def initialize_graphiti() -> None:
    """Initialize the Graphiti client with the configured settings."""
    global _graphiti_client, _config

    try:
        # Create LLM client if possible
        llm_client = _config.llm.create_client()
        if not llm_client and _config.use_custom_entities:
            # If custom entities are enabled, we must have an LLM client
            raise ValueError('OPENAI_API_KEY must be set when custom entities are enabled')

        # Validate Neo4j configuration
        if not _config.neo4j.uri or not _config.neo4j.user or not _config.neo4j.password:
            raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

        embedder_client = _config.embedder.create_client()

        # Initialize Graphiti client
        _graphiti_client = Graphiti(
            uri=_config.neo4j.uri,
            user=_config.neo4j.user,
            password=_config.neo4j.password,
            llm_client=llm_client,
            embedder=embedder_client,
            max_coroutines=SEMAPHORE_LIMIT,
        )

        # Destroy graph if requested
        if _config.destroy_graph:
            logger.info('Destroying graph...')
            await clear_data(_graphiti_client.driver)

        # Initialize the graph database with Graphiti's indices
        await _graphiti_client.build_indices_and_constraints()
        logger.info('Graphiti client initialized successfully')

        # Log configuration details for transparency
        if llm_client:
            logger.info(f'Using OpenAI model: {_config.llm.model}')
            logger.info(f'Using temperature: {_config.llm.temperature}')
        else:
            logger.info('No LLM client configured - entity extraction will be limited')

        logger.info(f'Using group_id: {_config.group_id}')
        logger.info(
            f'Custom entity extraction: {"enabled" if _config.use_custom_entities else "disabled"}'
        )
        logger.info(f'Using concurrency limit: {SEMAPHORE_LIMIT}')

    except Exception as e:
        logger.error(f'Failed to initialize Graphiti: {str(e)}')
        raise

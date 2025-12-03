"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)

This package provides a modular MCP server for Graphiti, a memory service for AI agents
built on a knowledge graph.
"""

from graphiti_mcp_server.config.settings import (
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_LLM_MODEL,
    SEMAPHORE_LIMIT,
    SMALL_LLM_MODEL,
)
from graphiti_mcp_server.main import main, run_mcp_server

__all__ = [
    'main',
    'run_mcp_server',
    'DEFAULT_LLM_MODEL',
    'SMALL_LLM_MODEL',
    'DEFAULT_EMBEDDER_MODEL',
    'SEMAPHORE_LIMIT',
]

#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)

This file is a compatibility wrapper. The actual implementation has been refactored
into the graphiti_mcp_server package for better maintainability.

Usage:
    python run_server.py [options]

Or use the package directly (recommended):
    python -m graphiti_mcp_server [options]
"""

from graphiti_mcp_server.main import main

if __name__ == '__main__':
    main()

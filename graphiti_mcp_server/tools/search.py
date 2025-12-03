"""
Search tools for querying the knowledge graph.
"""

import logging
from typing import cast

from graphiti_core import Graphiti
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters

from graphiti_mcp_server.client import get_config, get_graphiti_client
from graphiti_mcp_server.group_id import get_allowed_group_ids, get_effective_group_ids
from graphiti_mcp_server.models.responses import (
    ErrorResponse,
    FactSearchResponse,
    NodeResult,
    NodeSearchResponse,
)
from graphiti_mcp_server.utils import format_fact_result

logger = logging.getLogger(__name__)


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
    graphiti_client = get_graphiti_client()
    config = get_config()

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
    graphiti_client = get_graphiti_client()
    config = get_config()

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

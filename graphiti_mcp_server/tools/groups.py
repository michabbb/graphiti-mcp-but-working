"""
Group management tools for listing and deleting groups.
"""

import logging
from typing import cast

from graphiti_core import Graphiti

from graphiti_mcp_server.client import get_config, get_graphiti_client
from graphiti_mcp_server.group_id import get_allowed_group_ids, get_effective_group_id
from graphiti_mcp_server.models.responses import (
    DeleteGroupResponse,
    ErrorResponse,
    GroupIdListResponse,
    GroupIdResult,
)

logger = logging.getLogger(__name__)


async def list_group_ids(limit: int = 500) -> GroupIdListResponse | ErrorResponse:
    """Return distinct group IDs present on nodes and relationships in the graph."""
    graphiti_client = get_graphiti_client()

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


async def delete_everything_by_group_id(group_id: str) -> DeleteGroupResponse | ErrorResponse:
    """Delete all data associated with a group_id from the graph memory.

    This tool completely removes all episodes, nodes, and entity edges (facts/relationships)
    that belong to the specified group_id. After deletion, the group_id will no longer
    appear in list_group_ids results.

    This is a destructive operation that cannot be undone.

    Args:
        group_id: The group_id whose data should be completely deleted.
    """
    graphiti_client = get_graphiti_client()
    config = get_config()

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

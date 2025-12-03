"""
Context variable management for group_id allowlist.

This module provides context variables and helper functions for managing
the allowed group_ids from HTTP headers on a per-request basis.
"""

import contextvars

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

"""
Principal authentication for Graphiti MCP Server.
"""

import hashlib
import logging

from fastapi import HTTPException, Request, status
from fastapi.security.utils import get_authorization_scheme_param

from graphiti_mcp_server.auth.nonce import ALLOWED_NONCE_TOKENS, is_nonce_valid

logger = logging.getLogger(__name__)


def _hash_nonce(nonce: str) -> str:
    """Create a truncated SHA-256 hash of a nonce for safe logging/identification.

    Args:
        nonce: The raw nonce token

    Returns:
        A truncated hex digest (first 12 characters) of the SHA-256 hash
    """
    return hashlib.sha256(nonce.encode()).hexdigest()[:12]


def extract_bearer_token(request: Request) -> str | None:
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
        if is_nonce_valid(nonce):
            # Use hashed nonce in client_id to avoid exposing raw secret in logs/identifiers
            nonce_hash = _hash_nonce(nonce)
            logger.info(f'Authentication successful (client: nonce:{nonce_hash})')
            return {
                'client_id': f'nonce:{nonce_hash}',
                'auth_method': 'query_token',
                'scope': '',
            }
        logger.warning('Authentication failed: Invalid nonce token provided')
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid nonce token',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    # If nonce tokens are configured but no valid nonce was provided, reject
    logger.warning('Authentication failed: No nonce token provided')
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Not authenticated',
        headers={'WWW-Authenticate': 'Bearer'},
    )

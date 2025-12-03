"""
Nonce token validation for Graphiti MCP Server.
"""

import logging
import os
import secrets

logger = logging.getLogger(__name__)

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
        'AUTHENTICATION ENABLED: Loaded %d nonce token(s) for authentication',
        len(ALLOWED_NONCE_TOKENS),
    )
    logger.info('Requests must include valid nonce token (?nonce=<token>)')
else:
    logger.warning('AUTHENTICATION DISABLED: MCP_SERVER_NONCE_TOKENS not configured')
    logger.warning('Server will accept ALL requests without authentication!')


def is_nonce_valid(candidate: str) -> bool:
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

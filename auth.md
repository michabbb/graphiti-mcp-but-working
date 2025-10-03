# Authentication Guide for Graphiti MCP Server

This guide explains how to configure and use nonce-based authentication for the Graphiti MCP Server.

## Overview

The Graphiti MCP Server supports nonce token authentication via query parameters. This provides a simple yet secure way to authenticate requests without the complexity of OAuth flows.

## Prerequisites

### Dependencies

The authentication feature requires `fastapi` and `uvicorn`, which are included in the `pyproject.toml`:

```toml
dependencies = [
    # ... other dependencies
    "fastapi>=0.115.0",
    "uvicorn>=0.32.1"
]
```

### Installation

**With venv (local development):**
```bash
cd mcp_server
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
# Or if using pip with pyproject.toml:
pip install -e .
```

**With Docker:**
The Dockerfile automatically installs all dependencies from `pyproject.toml`.

## Configuration

### Environment Variable Setup

Set the `MCP_SERVER_NONCE_TOKENS` environment variable with comma-separated token values:

```bash
export MCP_SERVER_NONCE_TOKENS="token1,token2,token3"
```

**Example with real tokens:**
```bash
export MCP_SERVER_NONCE_TOKENS="ShjDhieHfxxxxxx,AnotherSecretToken123,MyToken456"
```

### Generating Secure Tokens

You can generate secure random tokens using Python:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Or use this command to generate multiple tokens:

```bash
for i in {1..3}; do python3 -c "import secrets; print(secrets.token_hex(32))"; done
```

### Docker Setup

If using Docker Compose, add the environment variable to your `docker-compose.yml`:

```yaml
services:
  mcp-server:
    environment:
      - MCP_SERVER_NONCE_TOKENS=token1,token2,token3
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

Or use an `.env` file:

```bash
# .env file
MCP_SERVER_NONCE_TOKENS=ShjDhieHfxxxxxx,AnotherSecretToken123,MyToken456
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
OPENAI_API_KEY=your-openai-api-key
```

## Usage

### Starting the Server

Start the MCP server with SSE transport (required for HTTP authentication):

```bash
cd mcp_server
python3 graphiti_mcp_server.py --transport sse
```

**With authentication enabled**, the server will log:

```
INFO - 🔒 AUTHENTICATION ENABLED: Loaded 3 nonce token(s) for authentication
INFO - 🔒 Requests must include valid nonce token (?nonce=<token>)
INFO - 🔒 Adding authentication middleware to SSE endpoints
INFO - Running MCP server with SSE transport on 0.0.0.0:8000
```

**Without authentication** (MCP_SERVER_NONCE_TOKENS not set), you'll see:

```
WARNING - ⚠️  AUTHENTICATION DISABLED: MCP_SERVER_NONCE_TOKENS not configured
WARNING - ⚠️  Server will accept ALL requests without authentication!
INFO - Running MCP server with SSE transport on 0.0.0.0:8000
```

### Making Authenticated Requests

Add the `nonce` query parameter to your requests:

```bash
# Example: Check server status
curl "http://localhost:8000/sse?nonce=your-token-here"
```

```bash
# Example with a real token
curl "http://localhost:8000/sse?nonce=ShjDhieHfxxxxxx"
```

### Authentication Logging

Each authentication attempt is logged to help you monitor access.

**Server startup logs** (authentication enabled):
```
INFO - 🔒 AUTHENTICATION ENABLED: Loaded 2 nonce token(s) for authentication
INFO - 🔒 Requests must include valid nonce token (?nonce=<token>)
INFO - 🔒 Registering authentication middleware on SSE app
DEBUG - 🔍 SSE app type: <class 'starlette.applications.Starlette'>
DEBUG - 🔍 SSE app id: 140234567890123
INFO - 🔒 Middleware registered. App has 1 middleware(s)
INFO - Running MCP server with SSE transport on 0.0.0.0:8000
```

**Per-request logging** (when DEBUG level enabled):
```
DEBUG - 🔍 MIDDLEWARE CALLED: GET /sse
INFO - ✓ Authentication successful with nonce token
```

**Failed authentication (invalid token):**
```
DEBUG - 🔍 MIDDLEWARE CALLED: POST /messages/
WARNING - ✗ Authentication failed: Invalid nonce token provided
WARNING - 🔍 MIDDLEWARE BLOCKED: POST /messages/ - Invalid nonce token
```

**Failed authentication (missing token):**
```
DEBUG - 🔍 MIDDLEWARE CALLED: GET /sse
WARNING - ✗ Authentication failed: No nonce token provided
WARNING - 🔍 MIDDLEWARE BLOCKED: GET /sse - Not authenticated
```

### MCP Client Configuration

Configure your MCP client to include the nonce token in requests. Example for Claude Desktop or similar MCP clients:

```json
{
  "mcpServers": {
    "graphiti": {
      "url": "http://localhost:8000/sse?nonce=ShjDhieHfxxxxxx",
      "transport": "sse"
    }
  }
}
```

## Security Features

### Middleware-Based Authentication

The authentication is implemented as **Starlette middleware** that intercepts HTTP requests before they reach endpoints. This ensures:
- The initial SSE connection (`/sse`) requires authentication via nonce token
- Internal MCP endpoints (`/messages/`, `/register`, etc.) are protected by session management
- Early rejection of unauthorized requests

**How it works:**
1. Client connects to `/sse?nonce=<token>` - **Authentication required**
2. If valid, a session is established
3. Subsequent requests to `/messages/`, `/register` use the authenticated session
4. No need to include nonce on every request after initial connection

### Constant-Time Comparison

The implementation uses `secrets.compare_digest()` to prevent timing attacks when validating tokens. This ensures that token comparison takes the same amount of time regardless of whether the token matches or not.

### Multiple Token Support

You can configure multiple tokens for different clients or purposes:

```bash
export MCP_SERVER_NONCE_TOKENS="client1-token,client2-token,admin-token"
```

Each token is validated independently, and any valid token will authenticate the request.

## Authentication Behavior

### When Tokens Are Configured

If `MCP_SERVER_NONCE_TOKENS` is set:
- Requests **must** include a valid `nonce` query parameter
- Requests without a nonce or with an invalid nonce receive `401 Unauthorized`
- Valid nonce grants access with `client_id: "nonce:<token>"`

### When Tokens Are Not Configured

If `MCP_SERVER_NONCE_TOKENS` is not set or empty:
- **Authentication is bypassed** (backward compatible)
- All requests are allowed
- Client ID is set to `"unauthenticated"`

## Error Responses

### Invalid Nonce Token

**Request:**
```bash
curl "http://localhost:8000/sse?nonce=invalid-token"
```

**Response:**
```json
{
  "error": "Invalid nonce token"
}
```
Status Code: `401 Unauthorized`

### Missing Authentication

**Request:**
```bash
curl "http://localhost:8000/sse"
```

**Response (when tokens are configured):**
```json
{
  "error": "Not authenticated"
}
```
Status Code: `401 Unauthorized`

## Testing Authentication

### Test Token Validation

```bash
# Set test tokens
export MCP_SERVER_NONCE_TOKENS="test-token-123,test-token-456"

# Start server
python3 graphiti_mcp_server.py --transport sse

# Test valid token
curl "http://localhost:8000/sse?nonce=test-token-123"
# Expected: Connection established

# Test invalid token
curl "http://localhost:8000/sse?nonce=wrong-token"
# Expected: 401 Unauthorized

# Test missing token
curl "http://localhost:8000/sse"
# Expected: 401 Unauthorized
```

## Best Practices

1. **Use Strong Tokens**: Generate cryptographically secure random tokens (minimum 32 characters)
2. **Rotate Tokens Regularly**: Update tokens periodically for enhanced security
3. **Limit Token Distribution**: Only share tokens with authorized clients
4. **Use HTTPS in Production**: Always use HTTPS/TLS to encrypt tokens in transit
5. **Environment Variables**: Never commit tokens to version control; use environment variables
6. **Separate Tokens per Client**: Use different tokens for different clients to enable selective revocation

## Troubleshooting

### Middleware Not Being Called

**Issue:** Requests bypass authentication (no middleware logs)

**Symptoms:**
```
# You see this:
INFO - Running MCP server with SSE transport on 0.0.0.0:8000
# But NOT this:
DEBUG - 🔍 MIDDLEWARE CALLED: GET /sse
```

**Solution:**
1. Check that `MCP_SERVER_NONCE_TOKENS` is set:
   ```bash
   echo $MCP_SERVER_NONCE_TOKENS
   ```
2. Verify transport is SSE (not stdio):
   ```bash
   python graphiti_mcp_server.py --transport sse
   ```
3. Enable DEBUG logging to see middleware calls:
   - The code now uses `logging.DEBUG` by default
   - Check logs for "🔍 MIDDLEWARE CALLED" messages

4. Verify middleware registration:
   ```
   # Should see in logs:
   INFO - 🔒 Registering authentication middleware on SSE app
   INFO - 🔒 Middleware registered. App has 1 middleware(s)
   ```

### Server Doesn't Start

**Issue:** Server fails to start with authentication errors

**Solution:** Check that environment variables are properly set:
```bash
echo $MCP_SERVER_NONCE_TOKENS
```

### Authentication Always Fails

**Issue:** Valid tokens are rejected

**Common causes:**

1. **Docker Compose Environment Variable Quotes**
   ```yaml
   # ❌ WRONG - includes quotes in the token value
   - MCP_SERVER_NONCE_TOKENS="token123"

   # ✅ CORRECT - no quotes
   - MCP_SERVER_NONCE_TOKENS=token123
   ```

2. **Token Whitespace/Format Issues**
   - Verify token has no extra whitespace
   - Check URL encoding of special characters
   - Ensure token matches exactly (case-sensitive)

3. **Debugging Steps**
   ```bash
   # Inside the container, check what token was actually set
   docker exec <container> env | grep MCP_SERVER_NONCE_TOKENS

   # Should output EXACTLY the token without quotes:
   # MCP_SERVER_NONCE_TOKENS=token123
   # NOT: MCP_SERVER_NONCE_TOKENS="token123"
   ```

4. **Confirm Middleware Is Active**
   - Look for "🔍 MIDDLEWARE CALLED" in DEBUG logs
   - If you see this, middleware is working
   - If token still fails, it's a token mismatch issue

### No Authentication Required

**Issue:** Server accepts requests without tokens

**Solution:** This is expected behavior when `MCP_SERVER_NONCE_TOKENS` is not set. Set the environment variable to enable authentication.

## Example: Complete Setup

### Local Setup

```bash
# 1. Generate secure tokens
TOKEN1=$(python3 -c "import secrets; print(secrets.token_hex(32))")
TOKEN2=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# 2. Configure environment
export MCP_SERVER_NONCE_TOKENS="$TOKEN1,$TOKEN2"
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
export OPENAI_API_KEY=your-openai-api-key

# 3. Start server
cd mcp_server
python3 graphiti_mcp_server.py --transport sse --group-id my-app

# 4. Test connection
curl "http://localhost:8000/sse?nonce=$TOKEN1"

# 5. Save token for client configuration
echo "Your MCP client URL: http://localhost:8000/sse?nonce=$TOKEN1"
```

### Docker Setup with Authentication

```bash
# 1. Generate secure token
TOKEN=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# 2. Build Docker image
cd mcp_server
docker build -f Dockerfile.new -t graphiti-mcp-server .

# 3. Run with authentication
docker run -d \
  --name graphiti-mcp \
  -p 8000:8000 \
  -e MCP_SERVER_NONCE_TOKENS="$TOKEN" \
  -e NEO4J_URI=bolt://neo4j:7687 \
  -e NEO4J_USER=neo4j \
  -e NEO4J_PASSWORD=password \
  -e OPENAI_API_KEY=your-openai-api-key \
  graphiti-mcp-server

# 4. Check logs to verify authentication is enabled
docker logs graphiti-mcp

# Expected output:
# INFO - 🔒 AUTHENTICATION ENABLED: Loaded 1 nonce token(s) for authentication
# INFO - 🔒 Requests must include valid nonce token (?nonce=<token>)
# INFO - Running MCP server with SSE transport on 0.0.0.0:8000

# 5. Test authentication
curl "http://localhost:8000/sse?nonce=$TOKEN"

# 6. Test without token (should fail)
curl "http://localhost:8000/sse"
# Expected: 401 Unauthorized
```

## Additional Resources

- [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)
- [Python Secrets Module](https://docs.python.org/3/library/secrets.html)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)

from __future__ import annotations

import hmac
import logging

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import Settings

log = logging.getLogger(__name__)


class BearerAuth:
    def __init__(self, app: ASGIApp, token: str, path: str):
        self.app = app
        self.expected = f"Bearer {token}".encode()
        self.path = path.rstrip("/") or "/"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"].rstrip("/").startswith(self.path):
            auth = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(auth, self.expected):
                response = JSONResponse({"error": "unauthorized"}, status_code=401,
                                        headers={"WWW-Authenticate": "Bearer"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_app(server: MCPServer, s: Settings) -> ASGIApp:
    extra_hosts = Settings.split_csv(s.mcp_allowed_hosts)
    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*", "host.docker.internal:*"]
    origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]
    for h in extra_hosts:
        hosts += [h, f"{h}:*"]
        origins += [f"https://{h}", f"http://{h}"]
    origins += Settings.split_csv(s.mcp_allowed_origins)
    security = TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts,
                                         allowed_origins=origins)
    app: ASGIApp = server.streamable_http_app(streamable_http_path=s.mcp_path, transport_security=security,
                                              host=s.mcp_host)
    if s.mcp_auth_token:
        app = BearerAuth(app, s.mcp_auth_token, s.mcp_path)
    return app


def serve(server: MCPServer, s: Settings, host: str, port: int) -> None:
    app = build_app(server, s)
    auth = "bearer token required" if s.mcp_auth_token else "NO AUTH (set MCP_AUTH_TOKEN before exposing it)"
    log.info("MCP over HTTP on http://%s:%s%s (%s)", host, port, s.mcp_path, auth)
    uvicorn.run(app, host=host, port=port, log_level="info", proxy_headers=True, forwarded_allow_ips="*")

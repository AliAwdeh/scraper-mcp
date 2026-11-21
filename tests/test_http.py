"""The HTTP transport: health check, bearer auth, DNS-rebinding protection."""

from starlette.testclient import TestClient

from scraper_mcp.config import Settings
from scraper_mcp.http_app import build_app
from scraper_mcp.server import mcp

INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def test_auth_and_host_checks():
    app = build_app(mcp, Settings(mcp_auth_token="s3cret", mcp_host="0.0.0.0"))
    with TestClient(app, base_url="http://localhost:8000") as client:
        assert client.get("/health").status_code == 200
        assert client.post("/mcp", json=INIT, headers=HEADERS).status_code == 401
        assert client.post("/mcp", json=INIT, headers={**HEADERS, "Authorization": "Bearer wrong"}).status_code == 401
        ok = client.post("/mcp", json=INIT, headers={**HEADERS, "Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        evil = client.post("/mcp", json=INIT, headers={**HEADERS, "Authorization": "Bearer s3cret", "Host": "evil.example"})
        assert evil.status_code in (400, 403, 421)

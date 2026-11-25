from __future__ import annotations

import argparse
import logging
import sys

from .config import get_settings


def main() -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(prog="scraper-mcp", description="LLM web-scraping MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default=s.mcp_transport)
    parser.add_argument("--host", default=s.mcp_host)
    parser.add_argument("--port", type=int, default=s.mcp_port)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "primp", "ddgs", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from .server import mcp

    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        from .http_app import serve

        serve(mcp, s, args.host, args.port)


if __name__ == "__main__":
    main()

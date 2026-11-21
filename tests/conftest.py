"""Shared test setup: an isolated data dir, a local fixture website, and a scripted fake LLM."""

from __future__ import annotations

import os
import re
import tempfile
import threading
from functools import lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Configure before any scraper_mcp import: isolated storage, no politeness delay, local site allowed,
# and no real API key (the LLM is replaced by a fake in the pipeline tests).
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="scraper-mcp-test-")
os.environ["ALLOW_PRIVATE_NETWORKS"] = "true"
os.environ["REQUEST_DELAY"] = "0"
os.environ["OPENROUTER_API_KEY"] = ""

import pytest  # noqa: E402

SITE = Path(__file__).parent / "fixtures" / "site"


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep test output clean
        pass

    def do_GET(self):
        if self.path == "/forbidden":  # a site that refuses bots, like Wikipedia without a contact UA
            body = b"Please respect our robot policy when crawling us."
            self.send_response(403)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


@pytest.fixture(scope="session")
def site() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=str(SITE)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture(autouse=True)
def fresh_runtime():
    """Each test gets its own event loop, so it also gets its own HTTP client / browser."""
    from scraper_mcp import server

    server._runtime = None
    yield
    server._runtime = None


@lru_cache(maxsize=1)
def chromium_available() -> bool:
    """True only if the Chromium build that *this* Playwright version needs can actually launch
    (always the case in the Docker image; usually not on a dev laptop)."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            p.chromium.launch(headless=True).close()
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------- fake LLM

_CARD = re.compile(
    r"\[(?P<name>[^\]]+)\]\((?P<url>[^)]+)\)\s*\|\s*(?P<brand>\w+)\s*\|\s*AED (?P<price>[\d,]+)"
    r"(?:\s*\|\s*was AED (?P<was>[\d,]+))?\s*\|\s*(?P<rating>[\d.]+)/5 \((?P<reviews>[\d,]+) reviews\)"
    r"\s*\|\s*(?P<stock>In stock|Sold out)")


def _num(text: str | None) -> float | None:
    return float(text.replace(",", "")) if text else None


class FakeLLM:
    """Deterministic stand-in for the model: parses the fixture shop like a specialist would,
    hallucinates exactly one product on page 1 (to test the Verifier), and plans a fixed analysis."""

    def __init__(self):
        self.calls: list[str] = []

    async def structured(self, schema, system, user, *, agent, fast=False):
        from scraper_mcp.agents import roles

        self.calls.append(agent)
        if agent == "planner":
            return roles.ResearchPlan(mode="ecommerce", search_queries=[], extraction_focus="laptops only",
                                      analysis_question="How do prices differ by brand?", dataset_name="fake_shop")
        if agent.startswith("specialist"):
            records = []
            for m in _CARD.finditer(user):
                records.append({
                    "name": m["name"], "brand": m["brand"], "price": _num(m["price"]),
                    "original_price": _num(m["was"]), "currency": "AED", "rating": float(m["rating"]),
                    "review_count": int(_num(m["reviews"])), "in_stock": m["stock"] == "In stock", "url": m["url"],
                })
            if "page 1" in user:
                records.append({"name": "Quantum Hyperbook X", "brand": "Quantum", "price": 99999.0,
                                "currency": "AED", "rating": 5.0, "review_count": 31337, "url": None})
            return schema(page_kind="listing", records=records, next_page_url=None)
        if agent == "source_selector":  # rank every result in order; the pipeline decides the backups
            return roles.SourceSelection(chosen=[roles.SourceChoice(index=i, reason="test")
                                                 for i in range(user.count("\n[") + 1)])
        if agent == "analyst":
            from scraper_mcp.analysis.charts import ChartSpec

            return roles.AnalysisPlan(
                rationale="Compare price by brand and check rating vs price.",
                filters=[{"column": "_grounding", "op": ">=", "value": 0.5}],
                operations=[{"op": "groupby", "by": ["brand"], "metric": "price", "aggs": ["count", "median"]},
                            {"op": "correlation", "columns": ["price", "rating"]}],
                chart=ChartSpec(kind="box", x="brand", y="price"),
            )
        raise AssertionError(f"unexpected agent {agent}")

    async def text(self, system, user, *, agent, fast=False):
        self.calls.append(agent)
        median = re.search(r'"median": ([\d.]+)', user)
        return (f"**Key findings**\n- The median price for one brand is {median.group(1)} AED.\n"
                f"- Prices peak at 12345 AED.\n**Caveats**\n- Small sample.")


@pytest.fixture
def fake_llm(monkeypatch):
    from scraper_mcp import llm
    from scraper_mcp.config import get_settings

    fake = FakeLLM()
    monkeypatch.setattr(llm, "structured", fake.structured)
    monkeypatch.setattr(llm, "text", fake.text)
    monkeypatch.setattr(get_settings(), "openrouter_api_key", "fake-key-for-tests")
    return fake

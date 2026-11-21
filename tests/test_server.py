"""End-to-end through the MCP protocol (in-process client), with the fake LLM."""

import json

import pytest
from mcp import Client

from scraper_mcp.server import mcp


def _json(result):
    assert not result.is_error, result.content[0].text
    return json.loads(result.content[0].text)


async def test_tool_list():
    async with Client(mcp) as c:
        names = {t.name for t in (await c.list_tools()).tools}
    assert {"research", "scrape", "analyze", "run_stat", "query_sql", "chart", "scrape_tables",
            "fetch_page", "search_web", "scrape_stats", "export_dataset", "describe_dataset"} <= names


async def test_llm_tools_explain_missing_key():
    async with Client(mcp) as c:
        r = await c.call_tool("scrape", {"urls": ["https://example.com"]})
    assert r.is_error and "OPENROUTER_API_KEY" in r.content[0].text


async def test_scrape_tables_without_llm(site):
    async with Client(mcp) as c:
        out = _json(await c.call_tool("scrape_tables", {"url": f"{site}/table.html", "dataset": "countries"}))
        assert out["table_used"] == 1 and out["saved"]["rows"] == 8
        assert out["columns"]["population"] == "float"  # "114,535,772[a]" parsed, footnote dropped
        assert out["columns"]["census_date"] == "date"  # "1 Jul 2025" is a date, not 172025
        assert out["columns"]["source"] == "str"  # "Official estimate[3]" stays text
        stat = _json(await c.call_tool("run_stat", {"dataset": "countries", "operation":
                                                    {"op": "groupby", "by": ["continent"], "metric": "population", "aggs": ["sum"]}}))
        top = stat["result"]["rows"][0]
        assert top["continent"] == "Asia" and top["sum"] == 1_574_750_000


async def test_fetch_page(site):
    async with Client(mcp) as c:
        out = _json(await c.call_tool("fetch_page", {"url": f"{site}/shop.html"}))
    assert out["fetched_with"] == "static" and out["next_page_url"].endswith("shop_page2.html")
    assert "Nova Studio 16" in out["markdown"]


async def test_full_pipeline_with_pagination_and_verification(site, fake_llm):
    async with Client(mcp) as c:
        r = await c.call_tool("scrape", {"urls": [f"{site}/shop.html"], "mode": "ecommerce", "dataset": "laptops",
                                         "max_pages_per_source": 2, "analyze": True})
        out = _json(r)
        # 6 products on page 1 + 3 on page 2 (pagination followed) + 1 hallucination on page 1
        assert out["rows_added"] == 10 and len(out["sources"]) == 2
        assert out["quality"]["low_grounding_rows"] == 1
        trace = " ".join(out["agent_trace"])
        assert "planner: skipped" in trace and "following 1 next-page link" in trace
        # derived metric computed by pandas, not the LLM
        rows = _json(await c.call_tool("query_sql", {"sql": "select name, discount_pct, _grounding from laptops order by name"}))["rows"]
        acme = next(r for r in rows if r["name"] == "Acme Book 14")
        assert acme["discount_pct"] == 12.5 and acme["_grounding"] == 1.0
        fake = next(r for r in rows if r["name"] == "Quantum Hyperbook X")
        assert fake["_grounding"] == 0.0
        # analysis: filters out the hallucination, numbers come from pandas, prose is audited
        a = out["analysis"]
        assert a["rows_analysed"] == 9 and a["rows_total"] == 10
        groups = {g["brand"]: g for g in a["operations"][0]["result"]["rows"]}
        assert groups["Acme"]["count"] == 3 and groups["Acme"]["median"] == 3499.0
        assert a["unverified_numbers"] == ["12345"]
        assert r.content[1].type == "image"

        prof = _json(await c.call_tool("describe_dataset", {"dataset": "laptops", "sample_rows": 2}))
        assert prof["profile"]["rows"] == 10 and len(prof["sample"]) == 2
        exp = _json(await c.call_tool("export_dataset", {"dataset": "laptops", "format": "csv", "inline": True}))
        assert exp["content"].startswith("name,brand,category,price")
        chart = await c.call_tool("chart", {"dataset": "laptops", "kind": "scatter", "x": "rating", "y": "price"})
        assert chart.content[1].type == "image"
        stats = _json(await c.call_tool("scrape_stats", {}))
        assert stats["runs"]["records_kept"] >= 10 and stats["fetches"]["ok_rate"] > 0
    assert fake_llm.calls.count("specialist:ecommerce") == 2 and "analyst" in fake_llm.calls


async def test_delete_needs_confirmation(site):
    async with Client(mcp) as c:
        _json(await c.call_tool("scrape_tables", {"url": f"{site}/table.html", "dataset": "tmp_table"}))
        assert _json(await c.call_tool("delete_dataset", {"dataset": "tmp_table"}))["deleted"] is False
        assert _json(await c.call_tool("delete_dataset", {"dataset": "tmp_table", "confirm": True}))["deleted"] is True


def test_table_column_names_with_symbols():
    import pandas as pd

    from scraper_mcp.server import _flatten_columns

    df = _flatten_columns(pd.DataFrame(columns=["Team Name", "Win %", "Goals For (GF)", "+ / -", "%", "Win %"]))
    assert list(df.columns) == ["team_name", "win_pct", "goals_for_gf", "plus_minus", "pct", "win_pct_2"]


async def test_unexpected_errors_are_readable(monkeypatch):
    from scraper_mcp import server

    async def boom(*a, **k):
        raise ValueError("something specific broke")

    monkeypatch.setattr(server, "web_search", boom)
    async with Client(mcp) as c:
        r = await c.call_tool("search_web", {"query": "x"})
    assert r.is_error and "ValueError: something specific broke" in r.content[0].text


async def test_refusing_sources_are_replaced_by_backups(site, fake_llm, monkeypatch):
    from scraper_mcp.agents import pipeline
    from scraper_mcp.scraping.search import SearchResult

    async def fake_search(query, max_results=10, region="us-en"):
        urls = ["forbidden", "missing.html", "shop.html", "shop_page2.html", "table.html"]
        return [SearchResult(title=u, url=f"{site}/{u}", snippet="laptops") for u in urls]

    monkeypatch.setattr(pipeline, "web_search", fake_search)
    async with Client(mcp) as c:
        out = _json(await c.call_tool("research", {"query": "laptop prices", "max_sources": 2, "analyze": False,
                                                    "dataset": "backup_test"}))
    by_url = {s["url"].rsplit("/", 1)[1]: s for s in out["sources"]}
    assert not by_url["forbidden"]["ok"] and not by_url["missing.html"]["ok"]
    assert by_url["shop.html"]["ok"] and by_url["shop_page2.html"]["ok"]  # the two backups filled in
    assert "table.html" not in by_url                                       # third backup not needed
    assert out["rows_added"] == 10 and "+ 2 backup(s)" in " ".join(out["agent_trace"])

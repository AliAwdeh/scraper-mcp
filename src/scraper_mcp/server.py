"""The MCP server: the tools a harness (Claude, ChatGPT, any MCP client) can call."""

from __future__ import annotations

import asyncio
import functools
import io
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

import pandas as pd
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, TextContent, ToolAnnotations
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import __version__, llm
from .agents.pipeline import Runtime, progress_cb, run_analysis, run_pipeline
from .analysis.charts import ChartSpec, render_chart
from .analysis.ops import Filter, OpError, Operation, apply_filters, run_operation
from .analysis.profile import profile
from .analysis.profile import sample_rows as _sample_rows
from .config import get_settings
from .modes import MODE_NAMES, describe_modes
from .scraping.cleaner import clean_html, clean_non_html
from .scraping.fetcher import Fetcher
from .scraping.search import SearchError, web_search
from .storage import Storage, StorageError, sanitize_name

log = logging.getLogger(__name__)

INSTRUCTIONS = """\
Web-scraping and data-analysis toolbox with its own team of agents inside.

Typical flow
1. research(query): the server's agents plan, search the web, pick sources, scrape them with a specialist
   (ecommerce, jobs, news, real_estate, reviews, research_papers or generic), verify every record against the
   page text, store a dataset and analyse it. Use scrape(urls, mode) instead when you already know the pages.
2. analyze(dataset, question): the Analyst agent picks statistics and pandas/scipy compute them.
   No number in any result is produced by an LLM; the Reporter's prose is audited against the results
   (see unverified_numbers).
3. Follow up precisely with describe_dataset, run_stat, query_sql and chart; get the data with export_dataset.

Tools that need no LLM: search_web, fetch_page, scrape_tables, list_datasets, describe_dataset, run_stat,
query_sql, chart, export_dataset, scrape_stats, delete_dataset.

Columns starting with "_" are provenance: _source_url, _fetch_method (static/browser), _run_id, _scraped_at, and
_grounding = share of the record's copied values found verbatim on the source page (below 0.5 = check it).
Report numbers exactly as returned."""

mcp = MCPServer(name="scraper-mcp", title="LLM Web Scraper & Analyst", instructions=INSTRUCTIONS,
                version=__version__)

_runtime: Runtime | None = None


def runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        s = get_settings()
        _runtime = Runtime(settings=s, fetcher=Fetcher(s), storage=Storage(s.db_path, s.exports_dir))
    return _runtime


def _json(data: Any) -> str:
    return json.dumps(data, indent=1, ensure_ascii=False, default=str)


def friendly_errors(fn):
    """Turn unexpected exceptions into readable tool errors, so the harness's LLM can react to them
    (the SDK otherwise reports only 'Error executing tool <name>')."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except ToolError:
            raise
        except StorageError as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            log.exception("tool %s failed", fn.__name__)
            raise ToolError(f"{fn.__name__} failed: {type(exc).__name__}: {exc}") from exc

    return wrapper


def _storage_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except StorageError as exc:
        raise ToolError(str(exc)) from exc


READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
READ_WEB = ToolAnnotations(read_only_hint=True, open_world_hint=True)
WRITES_DATA = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)

ModeArg = Annotated[str, Field(description=f"Specialist: {', '.join(MODE_NAMES)}. 'auto' lets the Planner choose.")]
RenderArg = Annotated[Literal["auto", "static", "browser"], Field(
    description="auto = plain HTTP, switching to headless Chromium if the page needs JavaScript")]


# ============================================================================ discovery

@mcp.tool(annotations=READ_ONLY)
def list_modes() -> dict:
    """List the specialist extraction modes, their fields and their default analysis."""
    return {"modes": describe_modes()}


@mcp.tool(annotations=READ_WEB)
@friendly_errors
async def search_web(
    query: Annotated[str, Field(description="Search query")],
    max_results: Annotated[int, Field(ge=1, le=20)] = 10,
) -> dict:
    """Search the web (DuckDuckGo). Returns titles, URLs and snippets; no scraping, no LLM."""
    try:
        results = await web_search(query, max_results=max_results)
    except SearchError as exc:
        raise ToolError(str(exc)) from exc
    return {"query": query, "results": [r.as_dict() for r in results]}


@mcp.tool(annotations=READ_WEB)
@friendly_errors
async def fetch_page(
    url: Annotated[str, Field(description="http(s) URL")],
    render: RenderArg = "auto",
    max_chars: Annotated[int, Field(ge=500, le=60000)] = 12000,
) -> dict:
    """Fetch one page politely (robots.txt, rate limit) and return it as clean Markdown, plus its
    schema.org JSON-LD and the next-page link. No LLM: useful to inspect a page before scraping it."""
    rt = runtime()
    res = await rt.fetcher.fetch(url, render)
    await asyncio.to_thread(rt.storage.log_fetches, f"fetch_page:{uuid.uuid4().hex[:8]}", [res.log_row()])
    if not res.ok:
        raise ToolError(f"could not fetch {url}: {res.error}")
    cp = (clean_html(res.html, res.final_url or url, max_chars=max_chars) if res.is_html
          else clean_non_html(res.html, res.final_url or url, res.content_type, max_chars=max_chars))
    return {"url": res.final_url or url, "status": res.status, "fetched_with": res.method,
            "js_fallback": res.js_fallback, "elapsed_ms": res.elapsed_ms, "title": cp.title,
            "description": cp.description, "next_page_url": cp.next_page_url, "json_ld": cp.json_ld[:10],
            "text_chars_total": cp.text_chars, "truncated": cp.truncated, "markdown": cp.text}


# ============================================================================ deterministic table scraping

def _column_name(raw: Any, position: int) -> str:
    text = str(raw)
    text = re.sub(r"\+\s*/\s*[-\u2212]", " plus_minus ", text)
    text = text.replace("%", " pct ").replace("#", " num ").replace("+", " plus ").replace("№", " num ")
    try:
        return sanitize_name(text)
    except StorageError:
        return f"column_{position + 1}"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(p) for p in col if "Unnamed" not in str(p)).strip() for col in df.columns]
    seen: dict[str, int] = {}
    cols = []
    for i, c in enumerate(df.columns):
        name = _column_name(c, i)
        seen[name] = seen.get(name, 0) + 1
        cols.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    df.columns = cols
    return df


_FOOTNOTE = re.compile(r"\[(?:[a-z]|\d{1,3}|note \d+|citation needed)\]", re.I)
_PLAIN_NUMBER = re.compile(r"^[-+]?\d+(?:\.\d+)?$")
_CURRENCY_AFFIX = re.compile(r"^(?:[$€£¥₹]|AED|USD|EUR|GBP|SAR)\s*|\s*(?:AED|USD|EUR|GBP|SAR)$")


def _parse_number(value: Any) -> float | None:
    """Strict: '1,234.5', '$99', '17.3%', '-2' parse; '13 Nov 2025' or 'Official estimate' do not."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _FOOTNOTE.sub("", str(value)).strip().replace("\u2212", "-").replace("\xa0", " ")
    text = _CURRENCY_AFFIX.sub("", text).rstrip("%").strip()
    text = re.sub(r"(?<=\d)[, ](?=\d{3}\b)", "", text)
    return float(text) if _PLAIN_NUMBER.match(text) else None


def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        raw = df[c].map(lambda v: _FOOTNOTE.sub("", v).strip() or None if isinstance(v, str) else v)
        present = int(raw.notna().sum())
        if not present:
            df[c] = raw
            continue
        numbers = raw.map(_parse_number)
        if numbers.notna().sum() >= 0.8 * present:
            df[c] = pd.to_numeric(numbers, errors="coerce")
            continue
        has_year = raw.astype("string").str.contains(r"\b(?:1[89]|20)\d{2}\b", na=False)
        if has_year.sum() >= 0.8 * present:
            dates = pd.to_datetime(raw, errors="coerce", format="mixed")
            if dates.notna().sum() >= 0.8 * present:
                df[c] = dates
                continue
        df[c] = raw
    return df


@mcp.tool(annotations=WRITES_DATA)
@friendly_errors
async def scrape_tables(
    url: Annotated[str, Field(description="Page containing one or more HTML <table> elements")],
    dataset: Annotated[str | None, Field(description="Dataset name to save into (default: derived from the URL)")] = None,
    table_index: Annotated[int | None, Field(description="Which table to save (default: the largest)")] = None,
    render: RenderArg = "auto",
) -> dict:
    """Classic, LLM-free scraping: parse the HTML tables on a page with pandas.read_html and save one as a
    dataset (numbers are cleaned automatically). A good baseline to compare against LLM extraction."""
    rt = runtime()
    res = await rt.fetcher.fetch(url, render)
    await asyncio.to_thread(rt.storage.log_fetches, f"scrape_tables:{uuid.uuid4().hex[:8]}", [res.log_row()])
    if not res.ok:
        raise ToolError(f"could not fetch {url}: {res.error}")
    try:
        tables = pd.read_html(io.StringIO(res.html))
    except ValueError:
        raise ToolError("no <table> elements found on that page; try scrape(urls=[...]) with an LLM specialist")
    summaries = [{"index": i, "rows": len(t), "columns": [str(c) for c in t.columns][:12]} for i, t in enumerate(tables)]
    idx = table_index if table_index is not None else max(range(len(tables)), key=lambda i: len(tables[i]) * len(tables[i].columns))
    if not 0 <= idx < len(tables):
        raise ToolError(f"table_index must be between 0 and {len(tables) - 1}")
    df = _coerce_columns(_flatten_columns(tables[idx].copy())).dropna(how="all")
    df["_source_url"] = res.final_url or url
    df["_scraped_at"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    from urllib.parse import urlsplit
    name = sanitize_name(dataset or f"table_{urlsplit(url).hostname or 'web'}")
    types = {c: ("float" if pd.api.types.is_numeric_dtype(df[c]) else
                 "date" if pd.api.types.is_datetime64_any_dtype(df[c]) else "str")
             for c in df.columns if not c.startswith("_")}
    saved = await asyncio.to_thread(_storage_call, rt.storage.save_dataset, name, df, mode="table",
                                    description=f"HTML table {idx} from {url}", source=url, columns=types,
                                    list_columns=[], append=bool(dataset))
    return {"saved": saved, "table_used": idx, "tables_found": summaries,
            "columns": types, "sample": _sample_rows(df, 5)}


# ============================================================================ the agent pipeline

async def _run(ctx: Context, tool: str, inputs: dict) -> tuple[dict, llm.RunUsage, str]:
    rt = runtime()
    if not rt.settings.llm_configured:
        raise ToolError(
            "OPENROUTER_API_KEY is not set, so the extraction agents cannot run. Add it to .env and restart. "
            "LLM-free alternatives: scrape_tables (HTML tables) and fetch_page (clean page text).")
    usage = llm.RunUsage()
    run_id = uuid.uuid4().hex[:12]
    step = 0

    async def report(message: str) -> None:
        nonlocal step
        step += 1
        await ctx.report_progress(step, None, message)

    usage_token = llm.current_usage.set(usage)
    progress_token = progress_cb.set(report)
    started = datetime.now(timezone.utc).replace(tzinfo=None)
    state: dict = {}
    status, error = "ok", None
    try:
        state = await run_pipeline(rt, run_id=run_id, **inputs)
    except llm.LLMNotConfigured as exc:
        status, error = "error", str(exc)
        raise ToolError(str(exc)) from exc
    except Exception as exc:
        log.exception("%s failed", tool)
        status, error = "error", f"{type(exc).__name__}: {exc}"
        raise ToolError(f"{tool} failed: {error}") from exc
    finally:
        llm.current_usage.reset(usage_token)
        progress_cb.reset(progress_token)
        q = state.get("quality") or {}
        fetch_log = state.get("fetch_log") or []
        run_row = {
            "run_id": run_id, "tool": tool, "request": inputs.get("request") or ", ".join(inputs.get("seed_urls") or []),
            "mode": (state.get("plan") or {}).get("mode") or inputs.get("mode"),
            "dataset": (state.get("store") or {}).get("dataset"), "started_at": started,
            "finished_at": datetime.now(timezone.utc).replace(tzinfo=None), "status": status, "error": error,
            "pages_attempted": len(fetch_log), "pages_ok": sum(1 for f in fetch_log if f["ok"]),
            "records_extracted": q.get("extracted", 0), "records_kept": (state.get("store") or {}).get("rows_added", 0),
            "low_grounding": q.get("low_grounding", 0), "llm_calls": usage.calls, "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens, "cost_usd": round(usage.cost_usd, 6),
            "agent_trace": json.dumps(state.get("trace") or []),
        }
        try:
            await asyncio.to_thread(rt.storage.log_run, run_row)
            await asyncio.to_thread(rt.storage.log_fetches, run_id, fetch_log)
        except Exception:
            log.exception("could not log run %s", run_id)
    return state, usage, run_id


def _visible(records: list[dict], n: int = 5) -> list[dict]:
    return [{k: v for k, v in r.items() if not k.startswith("_") or k == "_grounding"} for r in records[:n]]


def _pipeline_response(state: dict, usage: llm.RunUsage, run_id: str) -> list[TextContent | ImageContent]:
    store = state.get("store") or {}
    q = state.get("quality") or {}
    plan = state.get("plan") or {}
    out: dict[str, Any] = {
        "run_id": run_id,
        "dataset": store.get("dataset"),
        "mode": plan.get("mode"),
        "rows_added": store.get("rows_added", 0),
        "dataset_total_rows": store.get("rows"),
    }
    if plan.get("mode") == "generic":
        out["fields"] = state.get("fields")
    if plan.get("search_queries"):
        out["search"] = {"queries": plan["search_queries"], "results_considered": len(state.get("search_results") or [])}
    out["sources"] = [{k: v for k, v in p.items() if v is not None} for p in state.get("page_log") or []]
    out["quality"] = {
        "records_extracted": q.get("extracted", 0), "records_kept": store.get("rows_added", 0),
        "empty_dropped": q.get("empty", 0), "duplicates_dropped": q.get("duplicates", 0),
        "low_grounding_rows": q.get("low_grounding", 0),
        "note": "_grounding = share of copied values found verbatim on the source page",
    }
    out["sample"] = _visible(state.get("records") or [])
    content: list[TextContent | ImageContent] = []
    analysis = state.get("analysis")
    if analysis:
        png = analysis.pop("chart_png", None)
        out["analysis"] = analysis
        if png:
            content.append(ImageContent(type="image", data=__import__("base64").b64encode(png).decode(), mime_type="image/png"))
    out["agent_trace"] = state.get("trace") or []
    if state.get("errors"):
        out["errors"] = state["errors"][:10]
    out["llm_usage"] = usage.as_dict()
    if not out["rows_added"]:
        out["hint"] = ("No records were kept. Check 'sources' for fetch errors (robots.txt, HTTP errors, empty pages); "
                       "try render='browser', other URLs, or scrape_tables for plain HTML tables.")
    elif not analysis:
        out["next"] = f"analyze(dataset='{out['dataset']}', question=...) or describe_dataset(dataset='{out['dataset']}')"
    return [TextContent(type="text", text=_json(out)), *content]


@mcp.tool(annotations=WRITES_DATA, structured_output=False)
@friendly_errors
async def research(
    query: Annotated[str, Field(description="What to find out, in plain words, e.g. 'rent for 1-bedroom flats in Dubai Marina'")],
    ctx: Context,
    mode: ModeArg = "auto",
    max_sources: Annotated[int, Field(ge=1, le=10, description="Pages the Source Selector may pick")] = 5,
    max_pages_per_source: Annotated[int, Field(ge=1, le=5, description="Follow 'next page' links up to this depth")] = 1,
    dataset: Annotated[str | None, Field(description="Save into (and append to) this dataset; default: a new one")] = None,
    analyze: Annotated[bool, Field(description="Run the Analyst + Reporter on the result")] = True,
    question: Annotated[str | None, Field(description="Specific analysis question (default: the Planner's)")] = None,
    render: RenderArg = "auto",
) -> list[TextContent | ImageContent]:
    """End-to-end research with the server's agent team: Planner -> web search -> Source Selector ->
    Fetcher -> Specialist extraction -> Verifier (grounding check) -> dataset -> Analyst + Reporter.
    Takes 30-120 s depending on the number of pages."""
    if mode not in MODE_NAMES:
        raise ToolError(f"mode must be one of {MODE_NAMES}")
    state, usage, run_id = await _run(ctx, "research", {
        "request": query, "mode": mode, "max_sources": max_sources, "max_pages_per_source": max_pages_per_source,
        "dataset": dataset, "analyze": analyze, "question": question, "render": render})
    return _pipeline_response(state, usage, run_id)


@mcp.tool(annotations=WRITES_DATA, structured_output=False)
@friendly_errors
async def scrape(
    urls: Annotated[list[str], Field(min_length=1, max_length=10, description="Pages to scrape")],
    ctx: Context,
    mode: ModeArg = "auto",
    fields: Annotated[list[str] | None, Field(description=(
        "Generic mode: fields as 'name', 'name:type' or 'name:type:description' "
        "(types: str, int, float, bool, date, list). Omit to let the Schema Designer propose them."))] = None,
    instructions: Annotated[str | None, Field(description="What to include / exclude, e.g. 'only laptops, skip accessories'")] = None,
    dataset: Annotated[str | None, Field(description="Save into (and append to) this dataset")] = None,
    max_pages_per_source: Annotated[int, Field(ge=1, le=5)] = 1,
    render: RenderArg = "auto",
    analyze: bool = False,
    question: str | None = None,
) -> list[TextContent | ImageContent]:
    """Scrape known URLs with a specialist agent, verify the records, and save them as a dataset."""
    if mode not in MODE_NAMES:
        raise ToolError(f"mode must be one of {MODE_NAMES}")
    if fields and mode == "auto":
        mode = "generic"
    state, usage, run_id = await _run(ctx, "scrape", {
        "request": instructions or "", "mode": mode, "fields": fields, "seed_urls": urls, "dataset": dataset,
        "max_pages_per_source": max_pages_per_source, "render": render, "analyze": analyze, "question": question})
    return _pipeline_response(state, usage, run_id)


# ============================================================================ datasets

@mcp.tool(annotations=READ_ONLY)
@friendly_errors
async def list_datasets() -> dict:
    """List stored datasets with their mode and row counts."""
    return {"datasets": await asyncio.to_thread(_storage_call, runtime().storage.list_datasets)}


@mcp.tool(annotations=READ_ONLY)
@friendly_errors
async def describe_dataset(
    dataset: str,
    sample_rows: Annotated[int, Field(ge=0, le=50)] = 5,
) -> dict:
    """Profile a dataset: columns, types, missing values, ranges, top values, data quality, sample rows."""
    st = runtime().storage
    df = await asyncio.to_thread(_storage_call, st.load, dataset)
    meta = await asyncio.to_thread(_storage_call, st.meta, dataset)
    return {"dataset": meta["name"], "mode": meta["mode"], "description": meta["description"],
            "sources": meta["source"], "updated_at": str(meta["updated_at"]),
            "profile": profile(df, meta["list_columns"]), "sample": _sample_rows(df, sample_rows)}


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True))
@friendly_errors
async def delete_dataset(dataset: str, confirm: bool = False) -> dict:
    """Delete a dataset permanently. Requires confirm=true."""
    if not confirm:
        return {"deleted": False, "message": f"Call again with confirm=true to permanently delete '{dataset}'."}
    ok = await asyncio.to_thread(_storage_call, runtime().storage.delete, dataset)
    return {"deleted": ok, "dataset": dataset}


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
@friendly_errors
async def export_dataset(
    dataset: str,
    format: Literal["csv", "json", "parquet"] = "csv",
    inline: Annotated[bool, Field(description="Also return the file content (CSV/JSON up to ~200 KB)")] = False,
) -> dict:
    """Export a dataset to /data/exports (the ./data/exports folder on the host when ./data is mounted)."""
    st = runtime().storage
    path = await asyncio.to_thread(_storage_call, st.export, dataset, format)
    out: dict[str, Any] = {"dataset": dataset, "format": format, "path_in_container": str(path),
                           "path_on_host_if_mounted": f"./data/exports/{path.name}", "bytes": path.stat().st_size}
    if inline and format != "parquet":
        text = path.read_text(encoding="utf-8")
        out["content"] = text if len(text) <= 200_000 else text[:200_000]
        out["content_truncated"] = len(text) > 200_000
    return out


# ============================================================================ analysis

@mcp.tool(annotations=READ_ONLY, structured_output=False)
@friendly_errors
async def analyze(
    dataset: str,
    question: Annotated[str, Field(description="What you want to know, e.g. 'do remote jobs pay more?'")],
    ctx: Context,
    chart: bool = True,
) -> list[TextContent | ImageContent]:
    """Answer a question about a dataset. The Analyst agent chooses statistics from a fixed menu,
    pandas/scipy compute them, the Reporter writes findings, and every number in the findings is
    audited against the computed results."""
    rt = runtime()
    if not rt.settings.llm_configured:
        raise ToolError("OPENROUTER_API_KEY is not set; use run_stat / query_sql / chart, which need no LLM.")
    usage = llm.RunUsage()
    utok = llm.current_usage.set(usage)

    step = 0

    async def report(message: str) -> None:
        nonlocal step
        step += 1
        await ctx.report_progress(step, None, message)

    ptok = progress_cb.set(report)
    try:
        result = await run_analysis(rt, sanitize_name(dataset), question, want_chart=chart)
    except StorageError as exc:
        raise ToolError(str(exc)) from exc
    finally:
        llm.current_usage.reset(utok)
        progress_cb.reset(ptok)
    png = result.pop("chart_png", None)
    result["llm_usage"] = usage.as_dict()
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=_json(result))]
    if png:
        import base64
        content.append(ImageContent(type="image", data=base64.b64encode(png).decode(), mime_type="image/png"))
    return content


@mcp.tool(annotations=READ_ONLY)
@friendly_errors
async def run_stat(
    dataset: str,
    operation: Annotated[Operation, Field(description="One statistical operation; 'op' selects which")],
    filters: Annotated[list[Filter] | None, Field(description="Row filters applied first")] = None,
) -> dict:
    """Run one deterministic statistical operation (no LLM): describe, value_counts, groupby, correlation,
    compare_groups (t-test / Mann-Whitney / ANOVA / Kruskal), outliers, time_series, crosstab (chi-square),
    top_n, histogram, regression (OLS)."""
    st = runtime().storage
    df = await asyncio.to_thread(_storage_call, st.load, dataset)
    meta = await asyncio.to_thread(_storage_call, st.meta, dataset)
    try:
        filtered = apply_filters(df, filters or [])
    except OpError as exc:
        raise ToolError(str(exc)) from exc
    result = run_operation(filtered, operation, meta["list_columns"])
    if "error" in result:
        raise ToolError(result["error"])
    return {"dataset": meta["name"], "rows_total": len(df), "rows_used": len(filtered), **result}


@mcp.tool(annotations=READ_ONLY)
@friendly_errors
async def query_sql(
    sql: Annotated[str, Field(description=(
        "One read-only DuckDB query. Datasets are tables by name; telemetry is in _runs and _fetches. "
        "Example: SELECT brand, median(price) FROM laptops GROUP BY brand"))],
    max_rows: Annotated[int, Field(ge=1, le=1000)] = 100,
) -> dict:
    """Run read-only SQL over the datasets in an isolated sandbox (no file, network or settings access)."""
    df = await asyncio.to_thread(_storage_call, runtime().storage.run_sql, sql, max_rows)
    return {"columns": list(df.columns), "row_count": len(df), "truncated": len(df) >= max_rows,
            "rows": json.loads(df.to_json(orient="records", date_format="iso"))}


@mcp.tool(annotations=READ_ONLY, structured_output=False)
@friendly_errors
async def chart(
    dataset: str,
    kind: Literal["bar", "hist", "line", "scatter", "box"],
    x: Annotated[str, Field(description="bar: category | hist: numeric | line: date | scatter: numeric | box: group")],
    y: Annotated[str | None, Field(description="Numeric column (scatter, box; bar/line with agg)")] = None,
    agg: Literal["count", "mean", "median", "sum"] = "count",
    group: Annotated[str | None, Field(description="scatter only: colour by this column")] = None,
    title: str | None = None,
    top_n: Annotated[int, Field(ge=2, le=40)] = 15,
    filters: list[Filter] | None = None,
) -> list[TextContent | ImageContent]:
    """Draw a chart of a dataset (PNG). No LLM."""
    st = runtime().storage
    df = await asyncio.to_thread(_storage_call, st.load, dataset)
    meta = await asyncio.to_thread(_storage_call, st.meta, dataset)
    try:
        df = apply_filters(df, filters or [])
        spec = ChartSpec(kind=kind, x=x, y=y, agg=agg, group=group, title=title, top_n=top_n)
        png = await asyncio.to_thread(render_chart, df, spec, meta["list_columns"])
    except OpError as exc:
        raise ToolError(str(exc)) from exc
    return [TextContent(type="text", text=_json({"dataset": meta["name"], "rows_used": len(df), "chart": spec.model_dump()})),
            Image(data=png, format="png").to_image_content()]


# ============================================================================ telemetry

@mcp.tool(annotations=READ_ONLY)
@friendly_errors
async def scrape_stats() -> dict:
    """Statistics about the scraper itself: runs, pages fetched, success rates, static vs browser,
    robots.txt blocks, latency, top domains, records kept, LLM calls, tokens and cost."""
    frames = await asyncio.to_thread(runtime().storage.telemetry_frames)
    runs, fetches = frames["_runs"], frames["_fetches"]
    out: dict[str, Any] = {}
    if len(runs):
        out["runs"] = {
            "total": len(runs), "by_tool": runs["tool"].value_counts().to_dict(),
            "by_status": runs["status"].value_counts().to_dict(),
            "records_kept": int(runs["records_kept"].fillna(0).sum()),
            "records_extracted": int(runs["records_extracted"].fillna(0).sum()),
            "low_grounding_records": int(runs["low_grounding"].fillna(0).sum()),
            "llm_calls": int(runs["llm_calls"].fillna(0).sum()),
            "input_tokens": int(runs["input_tokens"].fillna(0).sum()),
            "output_tokens": int(runs["output_tokens"].fillna(0).sum()),
            "cost_usd": round(float(runs["cost_usd"].fillna(0).sum()), 4),
            "median_seconds_per_run": round(float((runs["finished_at"] - runs["started_at"]).dt.total_seconds().median()), 1),
        }
    if len(fetches):
        ok = fetches["ok"].astype(bool)
        by_method = fetches.groupby("method").agg(pages=("url", "size"), ok_rate=("ok", "mean"), median_ms=("elapsed_ms", "median"))
        top = (fetches.groupby("domain").agg(pages=("url", "size"), ok_rate=("ok", "mean"), median_ms=("elapsed_ms", "median"))
               .sort_values("pages", ascending=False).head(10))
        out["fetches"] = {
            "pages": len(fetches), "ok_rate": round(float(ok.mean()), 3),
            "by_method": json.loads(by_method.round(3).to_json(orient="index")),
            "js_fallbacks": int(fetches["js_fallback"].fillna(False).astype(bool).sum()),
            "blocked_by_robots": int((~fetches["robots_allowed"].fillna(True).astype(bool)).sum()),
            "status_codes": {str(k): int(v) for k, v in fetches["status"].value_counts(dropna=False).items()},
            "latency_ms": {"p50": float(fetches["elapsed_ms"].median()), "p90": float(fetches["elapsed_ms"].quantile(0.9))},
            "megabytes": round(float(fetches["bytes"].fillna(0).sum()) / 1e6, 2),
            "top_domains": json.loads(top.round(3).to_json(orient="index")),
        }
    return out or {"message": "nothing scraped yet"}


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    s = get_settings()
    return JSONResponse({"status": "ok", "version": __version__, "llm_configured": s.llm_configured, "model": s.llm_model})

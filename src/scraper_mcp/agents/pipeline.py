"""The multi-agent pipeline, orchestrated with LangGraph.

    START -> planner -> searcher -> source_selector -> fetcher -> specialists -> verifier
                 \\______(URLs given)_______________/^                              |
                                                     \\_____ next page (pagination)__/
                                                                                     v
                                                                  store -> analyst -> END
"""

from __future__ import annotations

import asyncio
import operator
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Awaitable, Callable, TypedDict
from urllib.parse import urlsplit

import pandas as pd
from langgraph.graph import END, START, StateGraph

from ..analysis.charts import render_chart
from ..analysis.ops import OpError, apply_filters, run_operation
from ..analysis.profile import profile, sample_rows
from ..config import Settings
from ..modes import get_mode
from ..scraping.cleaner import chunk_text, clean_html, clean_non_html
from ..scraping.fetcher import Fetcher
from ..scraping.search import web_search
from ..storage import Storage, sanitize_name
from . import roles
from .audit import audit_numbers
from .verifier import META_COLUMNS, dedupe, verify_records

ProgressFn = Callable[[str], Awaitable[None]]
progress_cb: ContextVar[ProgressFn | None] = ContextVar("progress_cb", default=None)


async def emit(message: str) -> None:
    cb = progress_cb.get()
    if cb is not None:
        try:
            await cb(message)
        except Exception:
            pass


@dataclass
class Runtime:
    settings: Settings
    fetcher: Fetcher
    storage: Storage


class PipelineState(TypedDict, total=False):
    # inputs
    run_id: str
    request: str
    mode: str
    fields: list[str] | None
    seed_urls: list[str]
    dataset: str | None
    max_sources: int
    max_pages_per_source: int
    render: str
    analyze: bool
    question: str | None
    # working state
    plan: dict
    search_results: list[dict]
    frontier: list[dict]
    backups: list[dict]
    pages: list[dict]
    extracted: list[dict]
    pages_fetched: int
    quality: dict
    store: dict
    analysis: dict
    # accumulated
    visited: Annotated[list[str], operator.add]
    records: Annotated[list[dict], operator.add]
    fetch_log: Annotated[list[dict], operator.add]
    page_log: Annotated[list[dict], operator.add]
    trace: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


def build_graph(rt: Runtime):
    s = rt.settings

    def mode_of(state: PipelineState):
        return get_mode(state["plan"]["mode"], state.get("fields"))

    # ------------------------------------------------------------------ planner
    async def planner(state: PipelineState) -> dict:
        seeds = state.get("seed_urls") or []
        forced = None if state.get("mode") in (None, "", "auto") else state["mode"]
        if forced and seeds:
            mode = get_mode(forced, state.get("fields"))
            plan = {"mode": forced, "search_queries": [], "extraction_focus": state.get("request") or "",
                    "analysis_question": state.get("question") or mode.default_question,
                    "dataset_name": state.get("dataset") or f"{forced}_{urlsplit(seeds[0]).hostname or 'web'}"}
            return {"plan": plan, "trace": [f"planner: skipped (mode '{forced}' and {len(seeds)} URL(s) given)"]}
        await emit("Planner: deciding mode, search queries and analysis")
        rp = await roles.plan_research(state.get("request") or ", ".join(seeds), urls_given=bool(seeds), forced_mode=forced)
        plan = rp.model_dump()
        if state.get("question"):
            plan["analysis_question"] = state["question"]
        fields = state.get("fields") or (rp.fields if rp.mode == "generic" else None)
        line = f"planner: mode={rp.mode}" + (f", queries={rp.search_queries}" if rp.search_queries else "")
        return {"plan": plan, "fields": fields, "trace": [line]}

    # ------------------------------------------------------------------ searcher
    async def searcher(state: PipelineState) -> dict:
        queries = state["plan"].get("search_queries") or [state.get("request", "")]
        await emit(f"Searcher: running {len(queries)} web search(es)")
        results: list[dict] = []
        seen: set[str] = set()
        errors: list[str] = []
        for q in queries:
            try:
                for r in await web_search(q, max_results=8):
                    if r.url not in seen:
                        seen.add(r.url)
                        results.append(r.as_dict())
            except Exception as exc:
                errors.append(f"search {q!r}: {exc}")
        return {"search_results": results[:20], "errors": errors,
                "trace": [f"searcher: {len(results)} unique results from {len(queries)} queries"]}

    # ------------------------------------------------------------------ source selector
    async def source_selector(state: PipelineState) -> dict:
        results = state.get("search_results") or []
        n = state["max_sources"]
        if not results:
            return {"frontier": [], "trace": ["source_selector: nothing to choose from"]}
        await emit("Source Selector: choosing the pages worth scraping")
        try:
            picks = await roles.select_sources(state.get("request", ""), state["plan"]["mode"], results, n)
            how = "LLM"
        except Exception as exc:
            picks, how = list(range(min(n, len(results)))), f"fallback to top results ({exc})"
        picks = picks or list(range(min(n, len(results))))
        ranked = [{"url": results[i]["url"], "depth": 0} for i in picks]
        frontier, backups = ranked[:n], ranked[n:]
        line = f"source_selector: picked {len(frontier)} of {len(results)}"
        if backups:
            line += f" + {len(backups)} backup(s)"
        return {"frontier": frontier, "backups": backups, "trace": [f"{line} ({how})"]}

    # ------------------------------------------------------------------ fetcher
    async def fetcher(state: PipelineState) -> dict:
        frontier = state.get("frontier")
        if frontier is None or (not frontier and not state.get("pages_fetched")):
            frontier = frontier or [{"url": u, "depth": 0} for u in state.get("seed_urls") or []]
        visited = set(state.get("visited") or [])
        budget = s.max_total_pages - (state.get("pages_fetched") or 0)
        todo = []
        for item in frontier:
            if item["url"] not in visited and len(todo) < budget:
                visited.add(item["url"])
                todo.append(item)
        await emit(f"Fetcher: downloading {len(todo)} page(s)")
        article = get_mode(state["plan"]["mode"], state.get("fields") or ["x"]).use_article_extractor

        async def one(item: dict):
            res = await rt.fetcher.fetch(item["url"], state.get("render"))
            page = {"url": item["url"], "final_url": res.final_url or item["url"], "depth": item["depth"],
                    "method": res.method, "ok": res.ok, "error": res.error, "js_fallback": res.js_fallback}
            if res.ok:
                if res.is_html:
                    cp = clean_html(res.html, page["final_url"], max_chars=s.max_page_chars, article=article)
                else:
                    cp = clean_non_html(res.html, page["final_url"], res.content_type, max_chars=s.max_page_chars)
                page.update(title=cp.title, text=cp.text, description=cp.description, json_ld=cp.json_ld_compact(),
                            next_page_url=cp.next_page_url, truncated=cp.truncated)
                if not cp.text.strip():
                    page.update(ok=False, error="no readable text after cleaning")
            return res, page

        done = list(await asyncio.gather(*(one(i) for i in todo)))
        # A first-round source that refused or failed is replaced by the next-ranked backup.
        backups = list(state.get("backups") or [])
        needed = sum(1 for _, p in done if not p["ok"])  # first-round failures to replace
        replaced = 0
        while backups and not state.get("pages_fetched"):
            missing = needed - replaced
            if missing <= 0:
                break
            batch = []
            while backups and len(batch) < missing and len(todo) + len(batch) < budget:
                item = backups.pop(0)
                if item["url"] not in visited:
                    visited.add(item["url"])
                    batch.append(item)
            if not batch:
                break
            await emit(f"Fetcher: {missing} source(s) failed, trying {len(batch)} backup(s)")
            more = await asyncio.gather(*(one(i) for i in batch))
            done.extend(more)
            todo.extend(batch)
            replaced += sum(1 for _, p in more if p["ok"])
        pages = [p for _, p in done]
        failed = [p for p in pages if not p["ok"]]
        methods = ", ".join(f"{p['method']}{'(js-fallback)' if p.get('js_fallback') else ''}" for p in pages if p["ok"])
        return {
            "pages": pages, "frontier": [], "backups": [], "visited": [i["url"] for i in todo],
            "pages_fetched": (state.get("pages_fetched") or 0) + len(todo),
            "fetch_log": [r.log_row() for r, _ in done],
            "page_log": [{"url": p["url"], "ok": False, "error": p["error"], "records": 0} for p in failed],
            "trace": [f"fetcher: {len(pages) - len(failed)}/{len(pages)} pages ok [{methods}]"],
        }

    # ------------------------------------------------------------------ specialists
    async def specialists(state: PipelineState) -> dict:
        pages = [p for p in state.get("pages") or [] if p["ok"]]
        update: dict[str, Any] = {"trace": []}
        fields = state.get("fields")
        if state["plan"]["mode"] == "generic" and not fields and pages:
            await emit("Schema Designer: proposing fields for the generic extractor")
            fields = await roles.design_schema(state["plan"].get("extraction_focus", ""), pages[0]["title"], pages[0]["text"])
            update["fields"] = fields
            update["trace"].append(f"schema_designer: fields={[f.split(':')[0] for f in fields]}")
        mode = get_mode(state["plan"]["mode"], fields)
        sem = asyncio.Semaphore(s.max_concurrency)
        focus = state["plan"].get("extraction_focus", "")
        finished = 0

        async def run_page(page: dict) -> dict:
            nonlocal finished
            chunks = chunk_text(page["text"], s.chunk_chars, s.max_chunks_per_page)

            async def run_chunk(i: int, chunk: str):
                async with sem:
                    return await roles.extract_chunk(
                        mode, focus=focus, url=page["final_url"], title=page.get("title", ""),
                        description=page.get("description", ""), json_ld=page.get("json_ld", "") if i == 0 else "",
                        chunk=chunk, chunk_no=i + 1, chunks=len(chunks))

            outs = await asyncio.gather(*(run_chunk(i, c) for i, c in enumerate(chunks)), return_exceptions=True)
            item = {"page": page, "raw": [], "page_kind": None, "next_llm": None, "notes": [], "errors": []}
            for out in outs:
                if isinstance(out, BaseException):
                    item["errors"].append(f"{type(out).__name__}: {out}")
                    continue
                item["raw"].extend(r.model_dump(mode="json") for r in out.records)
                item["page_kind"] = item["page_kind"] or out.page_kind
                item["next_llm"] = item["next_llm"] or out.next_page_url
                if out.notes:
                    item["notes"].append(out.notes)
            finished += 1
            await emit(f"Specialist ({mode.name}): {finished}/{len(pages)} pages extracted")
            return item

        await emit(f"Specialist ({mode.name}): extracting from {len(pages)} page(s)")
        extracted = await asyncio.gather(*(run_page(p) for p in pages))
        n = sum(len(e["raw"]) for e in extracted)
        update["extracted"] = list(extracted)
        update["trace"].append(f"specialist[{mode.name}]: {n} raw records from {len(pages)} page(s)")
        update["errors"] = [f"{e['page']['url']}: {err}" for e in extracted for err in e["errors"]]
        return update

    # ------------------------------------------------------------------ verifier
    async def verifier(state: PipelineState) -> dict:
        mode = mode_of(state)
        quality = dict(state.get("quality") or {"extracted": 0, "kept": 0, "empty": 0, "low_grounding": 0})
        records, page_log, frontier = [], [], []
        for item in state.get("extracted") or []:
            page = item["page"]
            source = "\n".join([page.get("title", ""), page.get("description", ""), page.get("json_ld", ""), page["text"]])
            kept, st = verify_records(item["raw"], mode, source_url=page["final_url"], page_title=page.get("title", ""),
                                      source_text=source, fetch_method=page["method"], run_id=state["run_id"])
            records.extend(kept)
            quality["extracted"] += st["extracted"]
            quality["kept"] += len(kept)
            quality["empty"] += st["empty"]
            quality["low_grounding"] += st["low_grounding"]
            page_log.append({"url": page["url"], "ok": True, "method": page["method"], "page_kind": item["page_kind"],
                             "records": len(kept), "low_grounding": st["low_grounding"],
                             "notes": "; ".join(item["notes"]) or None,
                             "error": "; ".join(item["errors"]) or None})
            nxt = page.get("next_page_url") or item.get("next_llm")
            if nxt and kept and page["depth"] + 1 < state["max_pages_per_source"]:
                frontier.append({"url": nxt, "depth": page["depth"] + 1})
        line = f"verifier: kept {quality['kept']} of {quality['extracted']} records so far, {quality['low_grounding']} low-grounding"
        if frontier:
            line += f"; following {len(frontier)} next-page link(s)"
        return {"records": records, "page_log": page_log, "quality": quality, "frontier": frontier, "trace": [line]}

    def after_verifier(state: PipelineState) -> str:
        if state.get("frontier") and (state.get("pages_fetched") or 0) < s.max_total_pages:
            return "fetcher"
        return "store"

    # ------------------------------------------------------------------ store
    async def store(state: PipelineState) -> dict:
        records, dups = dedupe(state.get("records") or [], mode_of(state).dedupe_on)
        quality = dict(state.get("quality") or {})
        quality["duplicates"] = dups
        if not records:
            return {"store": {"dataset": None, "rows_added": 0}, "quality": quality,
                    "trace": ["store: nothing to save"]}
        mode = mode_of(state)
        types = {f["name"]: f["type"] for f in mode.fields()}
        field_names = list(types)
        df = pd.DataFrame(records)
        for name, typ in types.items():
            if name not in df.columns:
                df[name] = None
            if typ == "date":
                df[name] = pd.to_datetime(df[name], errors="coerce")
            elif typ == "float":
                df[name] = pd.to_numeric(df[name], errors="coerce")
            elif typ == "int":
                df[name] = pd.to_numeric(df[name], errors="coerce").round().astype("Int64")
            elif typ == "bool":
                df[name] = df[name].astype("boolean")
        if mode.derive:
            df = mode.derive(df)
        derived = [c for c in df.columns if c not in types and not c.startswith("_")]
        for c in derived:
            types[c] = "float"
        meta = [c for c in META_COLUMNS if c in df.columns]
        df = df[[*field_names, *derived, *meta]]

        name, append = _resolve_name(rt.storage, state.get("dataset"), state["plan"].get("dataset_name"), mode.name)
        saved = await asyncio.to_thread(
            rt.storage.save_dataset, name, df, mode=mode.name, description=state.get("request") or "",
            source=", ".join(sorted({urlsplit(r["_source_url"]).netloc for r in records})),
            columns=types, list_columns=mode.list_fields(), append=append)
        saved["rows_added"] = len(df)
        return {"store": saved, "quality": quality,
                "trace": [f"store: {len(df)} rows -> dataset '{saved['dataset']}' ({saved['rows']} total)"]}

    def after_store(state: PipelineState) -> str:
        return "analyst" if state.get("analyze") and (state.get("store") or {}).get("rows_added") else END

    # ------------------------------------------------------------------ analyst (+ reporter)
    async def analyst(state: PipelineState) -> dict:
        question = state.get("question") or state["plan"].get("analysis_question") or mode_of(state).default_question
        try:
            result = await run_analysis(rt, state["store"]["dataset"], question)
            line = f"analyst: {len(result['operations'])} operations" + (", 1 chart" if result.get("chart_png") else "")
            return {"analysis": result, "trace": [line, "reporter: findings written and number-audited"]}
        except Exception as exc:
            return {"analysis": {"error": f"{type(exc).__name__}: {exc}"}, "errors": [f"analysis: {exc}"],
                    "trace": ["analyst: failed"]}

    g = StateGraph(PipelineState)
    for name, fn in [("planner", planner), ("searcher", searcher), ("source_selector", source_selector),
                     ("fetcher", fetcher), ("specialists", specialists), ("verifier", verifier),
                     ("store", store), ("analyst", analyst)]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", lambda st: "fetcher" if st.get("seed_urls") else "searcher",
                            ["fetcher", "searcher"])
    g.add_edge("searcher", "source_selector")
    g.add_edge("source_selector", "fetcher")
    g.add_edge("fetcher", "specialists")
    g.add_edge("specialists", "verifier")
    g.add_conditional_edges("verifier", after_verifier, ["fetcher", "store"])
    g.add_conditional_edges("store", after_store, ["analyst", END])
    g.add_edge("analyst", END)
    return g.compile()


def _resolve_name(storage: Storage, requested: str | None, suggested: str | None, mode: str) -> tuple[str, bool]:
    """A name the user chose is appended to; an auto-generated name never overwrites existing data."""
    if requested:
        return sanitize_name(requested), True
    base = sanitize_name(suggested or f"{mode}_{datetime.now():%Y%m%d_%H%M}")[:40]
    name, i = base, 2
    while storage.exists(name):
        name, i = f"{base}_{i}", i + 1
    return name, False


async def run_pipeline(rt: Runtime, **inputs: Any) -> PipelineState:
    s = rt.settings
    state: PipelineState = {
        "run_id": inputs.get("run_id") or uuid.uuid4().hex[:12],
        "request": inputs.get("request") or "",
        "mode": inputs.get("mode") or "auto",
        "fields": inputs.get("fields"),
        "seed_urls": inputs.get("seed_urls") or [],
        "dataset": inputs.get("dataset"),
        "max_sources": max(1, min(int(inputs.get("max_sources") or 5), s.max_sources_cap)),
        "max_pages_per_source": max(1, min(int(inputs.get("max_pages_per_source") or 1), s.max_pages_per_source_cap)),
        "render": inputs.get("render") or s.render_mode,
        "analyze": bool(inputs.get("analyze")),
        "question": inputs.get("question"),
        "visited": [], "records": [], "fetch_log": [], "page_log": [], "trace": [], "errors": [],
    }
    graph = build_graph(rt)
    return await graph.ainvoke(state, {"recursion_limit": 60})


# ----------------------------------------------------------------------------- analysis

def _execute(df: pd.DataFrame, plan: roles.AnalysisPlan, lists: list[str]):
    try:
        filtered, filter_error = apply_filters(df, plan.filters), None
    except OpError as exc:
        filtered, filter_error = df, str(exc)
    return [run_operation(filtered, op, lists) for op in plan.operations], filtered, filter_error


async def run_analysis(rt: Runtime, dataset: str, question: str, want_chart: bool = True) -> dict:
    df = await asyncio.to_thread(rt.storage.load, dataset)
    meta = await asyncio.to_thread(rt.storage.meta, dataset)
    lists = meta["list_columns"]
    prof = profile(df, lists)
    samples = sample_rows(df, 5)

    await emit("Analyst: choosing statistics")
    plan = await roles.plan_analysis(question, prof, samples)
    results, filtered, filter_error = _execute(df, plan, lists)
    failed = [r for r in results if "error" in r]
    repaired = False
    if failed and (len(failed) == len(results) or filter_error):
        feedback = "\n".join(f"- {r['op']}: {r['error']}" for r in failed)
        if filter_error:
            feedback += f"\n- filters: {filter_error}"
        await emit("Analyst: repairing the plan")
        plan = await roles.plan_analysis(question, prof, samples, feedback=feedback)
        results, filtered, filter_error = _execute(df, plan, lists)
        repaired = True

    chart_png, chart_error = None, None
    if want_chart and plan.chart is not None:
        try:
            chart_png = await asyncio.to_thread(render_chart, filtered, plan.chart, lists)
        except Exception as exc:
            chart_error = f"{type(exc).__name__}: {exc}"

    context = {"dataset": dataset, "rows_total": len(df), "rows_analysed": len(filtered),
               "filters": [f.model_dump() for f in plan.filters], "filter_error": filter_error,
               "quality": prof.get("quality")}
    await emit("Reporter: writing findings")
    try:
        findings = await roles.write_report(question, context, results)
    except Exception as exc:
        findings = f"(reporter unavailable: {exc}; the computed results above are complete)"
    return {
        "question": question,
        "dataset": dataset,
        "rows_total": len(df),
        "rows_analysed": len(filtered),
        "filters": context["filters"],
        "filter_error": filter_error,
        "rationale": plan.rationale,
        "plan_repaired": repaired,
        "operations": results,
        "findings": findings,
        "unverified_numbers": audit_numbers(findings, results, context),
        "chart": plan.chart.model_dump() if plan.chart else None,
        "chart_error": chart_error,
        "chart_png": chart_png,
    }

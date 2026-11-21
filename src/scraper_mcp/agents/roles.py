"""The LLM agents. Each one has a narrow job, a typed output and its own prompt.

    Planner          request -> mode, search queries, analysis question
    Source Selector  search results -> the pages worth scraping
    Schema Designer  (generic mode, no fields given) page sample -> record fields
    Specialists      page text -> typed records (one specialist per mode)
    Analyst          dataset profile + question -> which statistics to run (never the numbers)
    Reporter         computed results -> short findings, audited against the results
"""

from __future__ import annotations

import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from .. import llm
from ..analysis.charts import ChartSpec
from ..analysis.ops import OPERATION_HELP, Filter, Operation
from ..modes import COMMON_RULES, MODES, Mode, describe_modes, extraction_model

ModeName = Literal[tuple(["generic", *MODES.keys()])]  # type: ignore[valid-type]


# ----------------------------------------------------------------------------- Planner

class ResearchPlan(BaseModel):
    mode: ModeName = Field(description="The specialist best suited to the request")
    search_queries: list[str] = Field(default_factory=list, description="1-3 web search queries (empty if URLs given)")
    fields: list[str] | None = Field(None, description="Generic mode only: fields as 'name:type' (str/int/float/bool/date/list)")
    extraction_focus: str = Field(description="One or two sentences telling the specialist exactly what to extract and what to skip")
    analysis_question: str = Field(description="The statistical question to answer once the data is collected")
    dataset_name: str = Field(description="Short snake_case dataset name, max 30 characters")


async def plan_research(request: str, *, urls_given: bool, forced_mode: str | None) -> ResearchPlan:
    modes = "\n".join(f"- {m['mode']}: {m['description']}" for m in describe_modes() if m["mode"] != "auto")
    system = f"""You are the Planner of a web-scraping research team. Today is {date.today().isoformat()}.
A request arrives; you decide how the team should collect and analyse data for it.

Specialist modes:
{modes}

Rules:
- Pick the most specific mode that fits. Use "generic" only when no specialist fits, and then list 4-10 fields
  with types, putting numeric fields as int/float so they can be analysed and including at least one
  categorical field useful for grouping.
- Search queries must surface pages that LIST many comparable items (category pages, search results pages,
  job boards, data tables, archives). A statistic needs many records, so single-item pages are a poor choice.
  Mention a site or a region if the request does.
- {"The user already supplied URLs: return an empty search_queries list." if urls_given else "Write 1-3 distinct queries."}
- extraction_focus states inclusion and exclusion criteria taken from the request (price range, location,
  date window, category).
- analysis_question is concrete: which distributions, comparisons or trends answer the request."""
    if forced_mode:
        system += f"\n- The user chose the mode: use exactly \"{forced_mode}\"."
    plan = await llm.structured(ResearchPlan, system, f"Request: {request}", agent="planner")
    if forced_mode:
        plan.mode = forced_mode
    if urls_given:
        plan.search_queries = []
    plan.search_queries = plan.search_queries[:3]
    return plan


# ----------------------------------------------------------------------------- Source Selector

class SourceChoice(BaseModel):
    index: int = Field(description="Index of the search result")
    reason: str = Field(description="Why this page is likely to contain many relevant records")


class SourceSelection(BaseModel):
    chosen: list[SourceChoice]


async def select_sources(request: str, mode: str, results: list[dict], max_sources: int) -> list[int]:
    """Returns indices ranked best first: the first `max_sources` are scraped, the rest are backups
    used in place of any site that refuses or fails."""
    listing = "\n".join(f"[{i}] {r['title']}\n    {r['url']}\n    {r['snippet']}" for i, r in enumerate(results))
    wanted = max_sources * 2
    system = f"""You are the Source Selector. Rank up to {wanted} search results to scrape for a "{mode}"
dataset, best first. The first {max_sources} are scraped; the rest are backups used if a site refuses.
Prefer, in order:
1. pages that list MANY comparable items (category / search / listing pages, data tables, archives);
2. sites likely to serve plain HTML to an automated, honestly identified client. Large job boards,
   marketplaces and review platforms (Indeed, Glassdoor, LinkedIn, Amazon, Zillow, Yelp, TripAdvisor)
   usually refuse such clients; rank smaller aggregators, publishers and the organisations' own pages higher;
3. different domains, so the data is not from a single source.
Avoid PDFs, video, social media, login walls, and generic articles *about* the topic when the mode is not news.
Indices must come from the list."""
    out = await llm.structured(SourceSelection, system, f"Request: {request}\n\nResults:\n{listing}",
                               agent="source_selector", fast=True)
    seen: list[int] = []
    for choice in out.chosen:
        if 0 <= choice.index < len(results) and choice.index not in seen:
            seen.append(choice.index)
    return seen[:wanted]


# ----------------------------------------------------------------------------- Schema Designer

class FieldSpec(BaseModel):
    name: str = Field(description="snake_case field name")
    type: Literal["str", "int", "float", "bool", "date", "list"]
    description: str


class SchemaDesign(BaseModel):
    fields: list[FieldSpec] = Field(description="4-12 fields")


async def design_schema(request: str, page_title: str, page_sample: str) -> list[str]:
    system = """You are the Schema Designer. Look at the page and the user's instructions and design the
record fields to extract, one record per repeated item on the page (product, row, card, entry).
- 4 to 12 fields, snake_case names. Do not include a url field (it is added automatically).
- Numbers that someone would want to average or compare are int/float (prices, counts, scores, sizes).
- Include at least one categorical field that is useful for grouping (brand, type, location, category).
- Only fields that the page actually shows."""
    user = f"Instructions: {request or '(none: extract the main repeated items)'}\n\nPage title: {page_title}\n\nPage sample:\n{page_sample[:8000]}"
    out = await llm.structured(SchemaDesign, system, user, agent="schema_designer")
    return [f"{f.name}:{f.type}:{f.description}" for f in out.fields[:12]]


# ----------------------------------------------------------------------------- Specialists

async def extract_chunk(mode: Mode, *, focus: str, url: str, title: str, description: str, json_ld: str,
                        chunk: str, chunk_no: int, chunks: int) -> BaseModel:
    schema = extraction_model(mode.record_model)
    system = f"{mode.guidance}\n\n{COMMON_RULES}"
    if focus:
        system += f"\nUser's focus (what to include / exclude): {focus}\n"
    parts = [f"URL: {url}", f"Title: {title}"]
    if description:
        parts.append(f"Meta description: {description}")
    if json_ld:
        parts.append(f"Structured data (schema.org JSON-LD):\n{json_ld}")
    parts.append(f"Page text (part {chunk_no} of {chunks}):\n{chunk}")
    return await llm.structured(schema, system, "\n\n".join(parts), agent=f"specialist:{mode.name}")


# ----------------------------------------------------------------------------- Analyst

class AnalysisPlan(BaseModel):
    rationale: str = Field(description="One or two sentences: why these operations answer the question")
    filters: list[Filter] = Field(default_factory=list, description="Row filters applied before every operation")
    operations: list[Operation] = Field(description="1-6 operations from the menu")
    chart: ChartSpec | None = Field(None, description="The single most informative chart, if any")


async def plan_analysis(question: str, profile: dict, samples: list[dict], feedback: str | None = None) -> AnalysisPlan:
    menu = "\n".join(f"- {k}: {v}" for k, v in OPERATION_HELP.items())
    system = f"""You are the Analyst. You never calculate numbers yourself: you choose operations from a fixed
menu and a statistics engine (pandas / scipy) executes them exactly.

Menu:
{menu}

Rules:
- Use exact column names from the profile. Columns starting with "_" are metadata; use them only for
  data-quality questions.
- Never aggregate money across different currencies: if a currency column has several values, add a filter
  for the dominant currency or group by the currency column.
- Prefer spearman correlation and median for skewed data (prices, salaries).
- For groupby use min_group_size >= 2 so single rows do not look like a trend.
- compare_groups only when there is a real question about a difference between groups.
- 1 to 6 operations; the fewer that answer the question well, the better.
- Choose one chart that would help a reader most, or none."""
    user = (f"Question: {question}\n\nDataset profile:\n{json.dumps(profile, default=str)[:12000]}\n\n"
            f"Sample rows:\n{json.dumps(samples, default=str)[:4000]}")
    if feedback:
        user += f"\n\nYour previous plan failed with these errors; fix them:\n{feedback}"
    plan = await llm.structured(AnalysisPlan, system, user, agent="analyst")
    plan.operations = plan.operations[:6]
    return plan


# ----------------------------------------------------------------------------- Reporter

async def write_report(question: str, context: dict, results: list[dict]) -> str:
    system = """You are the Reporter. Write the findings for the user using ONLY the numbers in RESULTS.
- Every number you write must appear in RESULTS (you may round it). Do not compute new numbers,
  ratios or differences yourself.
- When a test is not significant (p >= 0.05), say there is no evidence of a difference/relationship.
- Mention data-quality caveats: small samples, few sources, rows with low grounding, mixed currencies.
Format (markdown, max ~250 words):
**Key findings**: 3-7 bullets
**Caveats**: 1-3 bullets"""
    user = (f"Question: {question}\n\nCONTEXT:\n{json.dumps(context, default=str)[:3000]}\n\n"
            f"RESULTS:\n{json.dumps(results, default=str)[:14000]}")
    return await llm.text(system, user, agent="reporter", fast=True)

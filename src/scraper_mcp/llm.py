from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from .config import get_settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMNotConfigured(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


@dataclass
class RunUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    by_agent: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "calls_by_agent": dict(self.by_agent),
        }


current_usage: ContextVar[RunUsage | None] = ContextVar("current_usage", default=None)


@lru_cache(maxsize=4)
def _model(model_name: str) -> ChatOpenAI:
    s = get_settings()
    headers = {"X-Title": s.openrouter_app_name}
    if s.openrouter_app_url:
        headers["HTTP-Referer"] = s.openrouter_app_url
    return ChatOpenAI(
        model=model_name,
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base_url,
        max_tokens=s.llm_max_tokens,
        timeout=s.llm_timeout,
        max_retries=s.llm_max_retries,
        default_headers=headers,
        extra_body={"usage": {"include": True}},
    )


def _require_key() -> None:
    if not get_settings().llm_configured:
        raise LLMNotConfigured(
            "OPENROUTER_API_KEY is not set. Put it in .env (see .env.example) and restart the container. "
            "Tools that do not need the LLM (fetch_page, scrape_tables, search_web, run_stat, query_sql, "
            "chart, describe_dataset, export_dataset) still work."
        )


def _record(raw, agent: str) -> None:
    usage = current_usage.get()
    if usage is None or raw is None:
        return
    usage.calls += 1
    usage.by_agent[agent] = usage.by_agent.get(agent, 0) + 1
    um = getattr(raw, "usage_metadata", None) or {}
    usage.input_tokens += int(um.get("input_tokens", 0) or 0)
    usage.output_tokens += int(um.get("output_tokens", 0) or 0)
    token_usage = (getattr(raw, "response_metadata", None) or {}).get("token_usage") or {}
    cost = token_usage.get("cost")
    if isinstance(cost, (int, float)):
        usage.cost_usd += float(cost)


def _portable(node):
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in ("discriminator", "$defs"):
                continue
            out["anyOf" if key == "oneOf" else key] = _portable(value)
        return out
    if isinstance(node, list):
        return [_portable(v) for v in node]
    return node


@lru_cache(maxsize=64)
def tool_schema(schema: type[BaseModel]) -> dict:
    fn = convert_to_openai_tool(schema)["function"]
    params = _portable(fn["parameters"])
    return {"title": fn["name"], "description": fn.get("description") or fn["name"], **params}


def _methods() -> list[str]:
    m = get_settings().llm_structured_method
    return ["function_calling", "json_schema"] if m == "auto" else [m]


async def structured(schema: type[T], system: str, user: str, *, agent: str, fast: bool = False) -> T:
    _require_key()
    s = get_settings()
    model = _model(s.fast_model if fast else s.llm_model)
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    last_error: Exception | None = None
    json_schema = tool_schema(schema)
    for method in _methods():
        try:
            chain = model.with_structured_output(json_schema, method=method, include_raw=True)
            out = await chain.ainvoke(messages)
        except Exception as exc:
            log.warning("%s: structured call via %s failed: %s", agent, method, exc)
            last_error = exc
            continue
        _record(out.get("raw"), agent)
        parsed = out.get("parsed")
        if parsed is None:
            last_error = out.get("parsing_error") or LLMError("model returned no parseable output")
            log.warning("%s: could not parse output via %s: %s", agent, method, last_error)
            continue
        try:
            return schema.model_validate(parsed)
        except ValidationError as exc:
            last_error = exc
            log.warning("%s: output failed validation via %s: %s", agent, method, exc)
    raise LLMError(f"{agent}: the model did not return valid structured output ({last_error})")


async def text(system: str, user: str, *, agent: str, fast: bool = False) -> str:
    _require_key()
    s = get_settings()
    model = _model(s.fast_model if fast else s.llm_model)
    msg = await model.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    _record(msg, agent)
    content = msg.content
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content).strip()

import json

import pytest

from scraper_mcp import llm
from scraper_mcp.agents.roles import AnalysisPlan, ResearchPlan
from scraper_mcp.config import get_settings


class FakeChain:
    def __init__(self, outputs, seen):
        self.outputs, self.seen = outputs, seen

    async def ainvoke(self, messages):
        return self.outputs.pop(0)


class FakeModel:
    def __init__(self, outputs):
        self.outputs, self.seen = outputs, []

    def with_structured_output(self, schema, method, include_raw):
        self.seen.append((method, schema))
        return FakeChain(self.outputs, self.seen)


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setattr(get_settings(), "openrouter_api_key", "k")


def test_schema_sent_to_provider_is_portable():
    blob = json.dumps(llm.tool_schema(AnalysisPlan))
    assert "discriminator" not in blob and "$defs" not in blob and "oneOf" not in blob and "$ref" not in blob
    assert llm.tool_schema(AnalysisPlan)["title"] == "AnalysisPlan"


async def test_dict_output_is_validated_into_the_model(monkeypatch, with_key):
    plan = {"rationale": "r", "operations": [{"op": "describe"}, {"op": "value_counts", "column": "brand"}]}
    fake = FakeModel([{"raw": None, "parsed": plan, "parsing_error": None}])
    monkeypatch.setattr(llm, "_model", lambda name: fake)
    out = await llm.structured(AnalysisPlan, "s", "u", agent="analyst")
    assert isinstance(out, AnalysisPlan) and out.operations[1].op == "value_counts"
    assert fake.seen[0][0] == "function_calling"


async def test_invalid_output_falls_back_to_json_schema(monkeypatch, with_key):
    bad = {"mode": "not-a-mode", "extraction_focus": "x", "analysis_question": "q", "dataset_name": "d"}
    good = {**bad, "mode": "jobs"}
    fake = FakeModel([{"raw": None, "parsed": bad, "parsing_error": None},
                      {"raw": None, "parsed": good, "parsing_error": None}])
    monkeypatch.setattr(llm, "_model", lambda name: fake)
    out = await llm.structured(ResearchPlan, "s", "u", agent="planner")
    assert out.mode == "jobs" and [m for m, _ in fake.seen] == ["function_calling", "json_schema"]


async def test_gives_up_with_a_clear_error(monkeypatch, with_key):
    fake = FakeModel([{"raw": None, "parsed": None, "parsing_error": ValueError("junk")}] * 2)
    monkeypatch.setattr(llm, "_model", lambda name: fake)
    with pytest.raises(llm.LLMError, match="planner"):
        await llm.structured(ResearchPlan, "s", "u", agent="planner")


async def test_missing_key_is_explained(monkeypatch):
    monkeypatch.setattr(get_settings(), "openrouter_api_key", "")
    with pytest.raises(llm.LLMNotConfigured, match="OPENROUTER_API_KEY"):
        await llm.structured(ResearchPlan, "s", "u", agent="planner")

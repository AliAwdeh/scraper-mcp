from __future__ import annotations

import re
import types
import typing
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Literal

import pandas as pd
from pydantic import BaseModel, Field, create_model


@dataclass
class Mode:
    name: str
    description: str
    record_model: type[BaseModel]
    guidance: str
    default_question: str
    use_article_extractor: bool = False
    derive: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    keywords: list[str] = field(default_factory=list)
    inferred: frozenset[str] = frozenset()
    dedupe_on: tuple[str, ...] | None = None

    def fields(self) -> list[dict]:
        out = []
        for name, f in self.record_model.model_fields.items():
            out.append({"name": name, "type": _type_name(f.annotation), "description": f.description or ""})
        return out

    def list_fields(self) -> list[str]:
        return [n for n, f in self.record_model.model_fields.items() if _type_name(f.annotation) == "list"]


def _type_name(annotation: Any) -> str:
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        return _type_name(non_none[0]) if non_none else "str"
    if origin is list:
        return "list"
    if origin is typing.Literal:
        return "category"
    if annotation is bool:
        return "bool"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is date:
        return "date"
    return "str"


COMMON_RULES = """\
Extraction rules (apply to every field):
- Extract ONLY what is present in the page text or the structured data. Never guess, never invent.
  If a value is not on the page, use null. An empty list is better than a fabricated record.
- Numbers are plain numbers: no currency symbols, no thousands separators, no units (1,299.00 AED -> 1299.0).
- Currencies are ISO 4217 codes (AED, USD, EUR, GBP, SAR, ...). Infer from symbols only when unambiguous
  ($ on a US site -> USD); otherwise null.
- Dates are ISO 8601 (YYYY-MM-DD). Convert relative dates ("3 days ago") ONLY if the page shows today's date;
  otherwise keep null.
- url is the absolute link to the item's own page when the page provides one.
- One record per distinct item. Do not merge different items, do not duplicate one item.
- Skip navigation, ads, "related items" widgets and cookie banners unless the user explicitly wants them.
"""


_TYPE_MAP: dict[str, Any] = {
    "str": str, "string": str, "text": str,
    "int": int, "integer": int,
    "float": float, "number": float, "decimal": float,
    "bool": bool, "boolean": bool,
    "date": date,
    "list": list[str], "list[str]": list[str], "tags": list[str],
}

_FIELD_NAME = re.compile(r"[^a-z0-9_]+")


def parse_field_spec(spec: str) -> tuple[str, Any, str]:
    parts = [p.strip() for p in spec.split(":", 2)]
    name = _FIELD_NAME.sub("_", parts[0].lower()).strip("_") or "field"
    if name[0].isdigit():
        name = f"f_{name}"
    type_ = _TYPE_MAP.get(parts[1].lower(), str) if len(parts) > 1 and parts[1] else str
    desc = parts[2] if len(parts) > 2 else ""
    return name, type_, desc


def build_record_model(field_specs: list[str], name: str = "Record") -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for spec in field_specs:
        fname, ftype, desc = parse_field_spec(spec)
        if fname in fields or fname in ("url",):
            continue
        default: Any = Field(default_factory=list, description=desc) if ftype == list[str] else Field(None, description=desc)
        fields[fname] = (ftype if ftype == list[str] else ftype | None, default)
    fields["url"] = (str | None, Field(None, description="Absolute URL of the item's own page, if any"))
    return create_model(name, **fields)


PageKind = Literal["listing", "detail", "article", "table", "other", "irrelevant"]


def extraction_model(record_model: type[BaseModel]) -> type[BaseModel]:
    return create_model(
        f"{record_model.__name__}Extraction",
        page_kind=(PageKind, Field(description="What kind of page this is")),
        records=(list[record_model], Field(default_factory=list, description="Every matching item on the page")),
        next_page_url=(str | None, Field(None, description="Absolute URL of the next results page, if there is one")),
        notes=(str | None, Field(None, description="Anything that limited extraction (login wall, missing data)")),
    )

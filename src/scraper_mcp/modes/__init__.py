from __future__ import annotations

from .base import COMMON_RULES, Mode, build_record_model, extraction_model, parse_field_spec
from .catalog import BUILTIN_MODES

GENERIC_DESCRIPTION = ("Any page, any schema: you name the fields (e.g. ['name', 'price:float', 'tags:list']) "
                       "or the Schema Designer agent proposes them from the page and your instructions.")

MODES: dict[str, Mode] = {m.name: m for m in BUILTIN_MODES}
MODE_NAMES = ["auto", "generic", *MODES.keys()]


def generic_mode(field_specs: list[str]) -> Mode:
    return Mode(
        name="generic",
        description=GENERIC_DESCRIPTION,
        record_model=build_record_model(field_specs, "Record"),
        guidance="You are the Generic Extraction Specialist. Extract every item on the page that matches the "
                 "requested fields and the user's instructions.",
        default_question="Profile the dataset: distributions of numeric fields, the most common values of "
                         "categorical fields, and any notable relationships between them.",
    )


def get_mode(name: str, field_specs: list[str] | None = None) -> Mode:
    if name == "generic":
        return generic_mode(field_specs or ["name", "description"])
    if name not in MODES:
        raise ValueError(f"unknown mode {name!r}; choose one of {', '.join(MODE_NAMES)}")
    return MODES[name]


def describe_modes() -> list[dict]:
    out = [{"mode": "auto", "description": "The Planner agent picks the best specialist for your request."},
           {"mode": "generic", "description": GENERIC_DESCRIPTION}]
    for m in MODES.values():
        out.append({"mode": m.name, "description": m.description,
                    "fields": [f["name"] for f in m.fields()], "default_analysis": m.default_question})
    return out


__all__ = ["COMMON_RULES", "MODES", "MODE_NAMES", "Mode", "describe_modes", "extraction_model", "generic_mode",
           "get_mode", "parse_field_spec"]

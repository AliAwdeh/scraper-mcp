"""Dataset profiling: what the Analyst agent (and the user) sees before choosing statistics."""

from __future__ import annotations

import pandas as pd

from ..agents.verifier import LOW_GROUNDING
from ..storage import parse_list_cell
from .ops import _clean


def profile(df: pd.DataFrame, list_columns: list[str] | None = None, top: int = 5) -> dict:
    list_columns = list_columns or []
    cols = []
    for c in df.columns:
        if c.startswith("_"):
            continue
        s = df[c]
        info: dict = {"name": c, "non_null": int(s.notna().sum()), "missing_pct": 100 * s.isna().mean()}
        if c in list_columns:
            items = s.map(parse_list_cell).explode().dropna()
            info.update(kind="list", distinct_items=int(items.nunique()),
                        top_items=items.value_counts().head(top).to_dict())
        elif pd.api.types.is_bool_dtype(s):
            info.update(kind="bool", true_pct=100 * s.mean() if s.notna().any() else None)
        elif pd.api.types.is_numeric_dtype(s):
            info.update(kind="numeric", min=s.min(), median=s.median(), mean=s.mean(), max=s.max(), std=s.std())
        elif pd.api.types.is_datetime64_any_dtype(s):
            info.update(kind="date", min=s.min(), max=s.max())
        else:
            nunique = int(s.nunique())
            avg_len = s.dropna().astype(str).str.len().mean() if s.notna().any() else 0
            kind = "text" if (avg_len and avg_len > 60) or nunique > 0.8 * max(1, s.notna().sum()) else "category"
            info.update(kind=kind, distinct=nunique)
            if kind == "category" or c == "url":
                info["top_values"] = s.value_counts().head(top).to_dict() if c != "url" else None
        cols.append(info)
    out = {"rows": len(df), "columns": cols}
    if "_grounding" in df.columns:
        g = pd.to_numeric(df["_grounding"], errors="coerce")
        out["quality"] = {"mean_grounding": g.mean(), "low_grounding_rows": int((g < LOW_GROUNDING).sum()),
                          "sources": int(df["_source_url"].nunique()) if "_source_url" in df else None}
    return _clean(out)


def sample_rows(df: pd.DataFrame, n: int = 5) -> list[dict]:
    visible = [c for c in df.columns if not c.startswith("_")]
    return _clean(df[visible].head(n).to_dict("records"))

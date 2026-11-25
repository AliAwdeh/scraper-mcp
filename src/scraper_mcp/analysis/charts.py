from __future__ import annotations

import io
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..storage import parse_list_cell
from .ops import OpError, _cat, _col, _num


class ChartSpec(BaseModel):
    kind: Literal["bar", "hist", "line", "scatter", "box"] = Field(
        description="bar: categories (count or aggregate of y) | hist: distribution of x | "
                    "line: x is a date, count or aggregate of y over time | scatter: x vs y | box: y by group x")
    x: str
    y: str | None = None
    agg: Literal["count", "mean", "median", "sum"] = "count"
    group: str | None = Field(None, description="scatter only: colour points by this column")
    title: str | None = None
    top_n: int = 15


def _numeric_labels(labels) -> bool:
    try:
        [float(str(x)) for x in labels]
        return len(labels) > 0
    except ValueError:
        return False


def render_chart(df: pd.DataFrame, spec: ChartSpec, list_columns: list[str] | None = None) -> bytes:
    list_columns = list_columns or []
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=110)
    try:
        title = spec.title or _default_title(spec)
        if spec.kind == "bar":
            if spec.y and spec.agg != "count":
                data = pd.DataFrame({"x": _cat(_col(df, spec.x)), "y": _num(df, spec.y)}).dropna()
                series = data.groupby("x")["y"].agg(spec.agg).sort_values(ascending=False).head(spec.top_n)
                ax.set_xlabel(f"{spec.agg} of {spec.y}")
            else:
                s = _col(df, spec.x)
                if spec.x in list_columns:
                    s = s.map(parse_list_cell).explode().dropna()
                series = _cat(s).value_counts().head(spec.top_n)
                ax.set_xlabel("count")
            if _numeric_labels(series.index):
                series = series.sort_index(key=lambda idx: idx.astype(str).astype(float))
            series = series.iloc[::-1]
            ax.barh([str(i)[:40] for i in series.index], series.values, color="#4C72B0")
            for i, v in enumerate(series.values):
                ax.text(v, i, f" {v:,.4g}", va="center", fontsize=8)
        elif spec.kind == "hist":
            s = _num(df, spec.x).dropna()
            ax.hist(s, bins=min(30, max(5, int(np.sqrt(len(s))))), color="#4C72B0", edgecolor="white")
            ax.axvline(s.median(), color="#C44E52", linestyle="--", label=f"median {s.median():,.4g}")
            ax.legend()
            ax.set_xlabel(spec.x)
            ax.set_ylabel("count")
        elif spec.kind == "line":
            dates = pd.to_datetime(_col(df, spec.x), errors="coerce")
            data = pd.DataFrame({"d": dates, "n": 1})
            if spec.y:
                data["y"] = _num(df, spec.y)
            data = data.dropna(subset=["d"]).set_index("d").sort_index()
            if len(data) == 0:
                raise OpError(f"{spec.x!r} has no parseable dates")
            span = (data.index.max() - data.index.min()).days
            freq = "QS" if span > 730 else "MS" if span > 90 else "W" if span > 21 else "D"
            r = data.resample(freq)
            series = r.size() if (not spec.y or spec.agg == "count") else getattr(r["y"], spec.agg)()
            ax.plot(series.index, series.values, marker="o", color="#4C72B0")
            ax.set_ylabel("count" if not spec.y or spec.agg == "count" else f"{spec.agg} of {spec.y}")
            fig.autofmt_xdate()
        elif spec.kind == "scatter":
            if not spec.y:
                raise OpError("scatter needs both x and y")
            data = pd.DataFrame({"x": _num(df, spec.x), "y": _num(df, spec.y)})
            if spec.group:
                data["g"] = _cat(_col(df, spec.group))
            data = data.dropna(subset=["x", "y"])
            if spec.group:
                for name, grp in data[data["g"].isin(data["g"].value_counts().head(8).index)].groupby("g"):
                    ax.scatter(grp["x"], grp["y"], label=str(name)[:25], alpha=0.75, s=28)
                ax.legend(fontsize=8)
            else:
                ax.scatter(data["x"], data["y"], alpha=0.75, s=28, color="#4C72B0")
            if len(data) >= 3 and data["x"].nunique() > 1:
                slope, intercept = np.polyfit(data["x"], data["y"], 1)
                xs = np.linspace(data["x"].min(), data["x"].max(), 50)
                r = data["x"].corr(data["y"])
                ax.plot(xs, slope * xs + intercept, color="#C44E52", linestyle="--", label=f"trend (r = {r:.2f})")
                ax.legend(fontsize=8)
            ax.set_xlabel(spec.x)
            ax.set_ylabel(spec.y)
        elif spec.kind == "box":
            if not spec.y:
                raise OpError("box needs y (numeric) and x (group)")
            data = pd.DataFrame({"g": _cat(_col(df, spec.x)), "y": _num(df, spec.y)}).dropna()
            groups = list(data["g"].value_counts().head(spec.top_n).index)
            if _numeric_labels(groups):
                groups.sort(key=lambda g: float(str(g)))
            values = [data.loc[data["g"] == g, "y"].to_numpy() for g in groups]
            ax.boxplot(values, tick_labels=[str(g)[:20] for g in groups], orientation="vertical")
            ax.set_ylabel(spec.y)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        return buf.getvalue()
    finally:
        plt.close(fig)


def _default_title(spec: ChartSpec) -> str:
    if spec.kind == "bar":
        return f"{spec.agg} of {spec.y} by {spec.x}" if spec.y and spec.agg != "count" else f"Most common {spec.x}"
    if spec.kind == "hist":
        return f"Distribution of {spec.x}"
    if spec.kind == "line":
        return f"{spec.agg} of {spec.y} over time" if spec.y and spec.agg != "count" else f"Records over time ({spec.x})"
    if spec.kind == "scatter":
        return f"{spec.y} vs {spec.x}"
    return f"{spec.y} by {spec.x}"

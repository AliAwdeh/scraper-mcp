from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats as sps

from ..storage import parse_list_cell

AggName = Literal["count", "mean", "median", "min", "max", "std", "sum", "nunique"]


class Filter(BaseModel):
    column: str
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "not_contains", "is_null", "not_null"]
    value: str | float | int | bool | list[str | float | int] | None = None


class Describe(BaseModel):
    op: Literal["describe"] = "describe"
    columns: list[str] | None = Field(None, description="Numeric columns to summarise; null = every numeric column")


class ValueCounts(BaseModel):
    op: Literal["value_counts"] = "value_counts"
    column: str
    top_n: int = 15
    explode: bool = Field(False, description="Count the individual items of a list column (skills, tags, topics)")


class GroupBy(BaseModel):
    op: Literal["groupby"] = "groupby"
    by: list[str] = Field(description="One or two grouping columns")
    metric: str | None = Field(None, description="Numeric column to aggregate; null = count rows per group")
    aggs: list[AggName] = Field(default_factory=lambda: ["count", "mean", "median"])
    min_group_size: int = Field(1, description="Hide groups with fewer rows than this")
    sort_by: AggName | None = Field(None, description="Aggregate to sort by (default: the first one)")
    ascending: bool = False
    top_n: int = 20


class Correlation(BaseModel):
    op: Literal["correlation"] = "correlation"
    columns: list[str] | None = Field(None, description="Numeric columns (2-8); null = all numeric columns")
    method: Literal["pearson", "spearman"] = "spearman"


class CompareGroups(BaseModel):
    op: Literal["compare_groups"] = "compare_groups"
    group_column: str
    value_column: str
    groups: list[str] | None = Field(None, description="Groups to compare; null = the 5 largest")


class Outliers(BaseModel):
    op: Literal["outliers"] = "outliers"
    column: str
    method: Literal["iqr", "zscore"] = "iqr"
    show_columns: list[str] | None = None
    limit: int = 10


class TimeSeries(BaseModel):
    op: Literal["time_series"] = "time_series"
    date_column: str
    value_column: str | None = Field(None, description="Numeric column; null = count rows per period")
    freq: Literal["D", "W", "M", "Q", "Y"] = "M"
    agg: Literal["count", "mean", "median", "sum"] = "count"


class Crosstab(BaseModel):
    op: Literal["crosstab"] = "crosstab"
    row: str
    column: str
    normalize: bool = Field(False, description="Show row percentages instead of counts")
    top_n: int = 10


class TopN(BaseModel):
    op: Literal["top_n"] = "top_n"
    column: str
    n: int = 10
    ascending: bool = False
    show_columns: list[str] | None = None


class Histogram(BaseModel):
    op: Literal["histogram"] = "histogram"
    column: str
    bins: int = 10


class Regression(BaseModel):
    op: Literal["regression"] = "regression"
    target: str = Field(description="Numeric column to explain")
    features: list[str] = Field(description="Numeric (or boolean) predictor columns")


Operation = Annotated[
    Union[Describe, ValueCounts, GroupBy, Correlation, CompareGroups, Outliers, TimeSeries, Crosstab, TopN,
          Histogram, Regression],
    Field(discriminator="op"),
]

OPERATION_HELP = {
    "describe": "count, mean, std, min, quartiles, max, missing share of numeric columns",
    "value_counts": "frequency table of a column (explode=true for list columns like skills)",
    "groupby": "aggregate a numeric metric by one or two categorical columns",
    "correlation": "pairwise correlations with p-values (pearson or spearman)",
    "compare_groups": "are groups different? Welch t-test + Mann-Whitney (2 groups) or ANOVA + Kruskal-Wallis",
    "outliers": "rows outside 1.5xIQR or |z|>3",
    "time_series": "count or aggregate per day/week/month/quarter/year",
    "crosstab": "contingency table of two categorical columns (+ chi-square test)",
    "top_n": "rows with the largest / smallest values of a column",
    "histogram": "binned distribution of a numeric column",
    "regression": "ordinary least squares: coefficients, p-values, R²",
}


class OpError(ValueError):
    pass


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        if f != 0 and abs(f) < 1e-3:
            return float(f"{f:.3g}")
        return round(f, 2) if abs(f) >= 100 else round(f, 4)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        visible = [c for c in df.columns if not c.startswith("_")]
        raise OpError(f"column {name!r} not found; available: {', '.join(visible)}")
    return df[name]


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    s = _col(df, name)
    if s.dtype == bool:
        return s.astype(float)
    out = pd.to_numeric(s, errors="coerce")
    if out.notna().sum() == 0:
        raise OpError(f"column {name!r} has no numeric values")
    return out


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("_") and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def _cat(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("(missing)")


def apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    for f in filters or []:
        s = _col(df, f.column)
        v = f.value
        if f.op in (">", ">=", "<", "<="):
            if isinstance(v, str):
                try:
                    v = float(v)
                except ValueError:
                    s = pd.to_datetime(s, errors="coerce")
                    v = pd.Timestamp(v)
            if not isinstance(v, pd.Timestamp):
                s = pd.to_numeric(s, errors="coerce")
        if f.op == "==":
            mask = s.astype("string").str.lower() == str(v).lower() if isinstance(v, str) else s == v
        elif f.op == "!=":
            mask = s.astype("string").str.lower() != str(v).lower() if isinstance(v, str) else s != v
        elif f.op == ">":
            mask = s > v
        elif f.op == ">=":
            mask = s >= v
        elif f.op == "<":
            mask = s < v
        elif f.op == "<=":
            mask = s <= v
        elif f.op in ("in", "not_in"):
            values = [str(x).lower() for x in (v if isinstance(v, list) else [v])]
            mask = s.astype("string").str.lower().isin(values)
            mask = ~mask if f.op == "not_in" else mask
        elif f.op in ("contains", "not_contains"):
            mask = s.astype("string").str.contains(str(v), case=False, regex=False, na=False)
            mask = ~mask if f.op == "not_contains" else mask
        elif f.op == "is_null":
            mask = s.isna()
        else:
            mask = s.notna()
        df = df[mask.fillna(False).astype(bool)]
    return df


def _describe(df: pd.DataFrame, op: Describe, _lists) -> dict:
    cols = op.columns or numeric_columns(df)
    out = {}
    for c in cols:
        s = _num(df, c)
        d = s.describe()
        out[c] = {"count": d["count"], "mean": d["mean"], "std": d.get("std"), "min": d["min"],
                  "p25": d["25%"], "median": d["50%"], "p75": d["75%"], "max": d["max"],
                  "missing_pct": 100 * s.isna().mean(), "skew": s.skew()}
    return {"columns": out}


def _value_counts(df: pd.DataFrame, op: ValueCounts, lists: list[str]) -> dict:
    s = _col(df, op.column)
    if op.explode or op.column in lists:
        s = s.map(parse_list_cell).explode().dropna().astype("string").str.strip()
        s = s[s != ""]
    counts = _cat(s).value_counts()
    total = int(counts.sum())
    top = counts.head(op.top_n)
    return {"total": total, "distinct": int(len(counts)),
            "top": [{"value": k, "count": int(v), "pct": 100 * v / total if total else None} for k, v in top.items()]}


def _groupby(df: pd.DataFrame, op: GroupBy, _lists) -> dict:
    work = df.copy()
    for b in op.by:
        work[b] = _cat(_col(work, b))
    if op.metric:
        work["_metric"] = _num(work, op.metric)
        table = work.groupby(op.by)["_metric"].agg(list(op.aggs))
        sizes = work.groupby(op.by).size()
    else:
        sizes = work.groupby(op.by).size()
        table = sizes.to_frame("count")
    table = table[sizes.reindex(table.index) >= op.min_group_size]
    sort_col = op.sort_by if op.sort_by in table.columns else table.columns[0]
    table = table.sort_values(sort_col, ascending=op.ascending).head(op.top_n)
    rows = []
    for idx, row in table.iterrows():
        key = dict(zip(op.by, idx if isinstance(idx, tuple) else (idx,)))
        rows.append({**key, **row.to_dict()})
    return {"metric": op.metric, "groups_shown": len(rows), "groups_total": int(len(sizes)),
            "groups_hidden": int(len(sizes)) - len(rows), "rows": rows}


def _correlation(df: pd.DataFrame, op: Correlation, _lists) -> dict:
    cols = op.columns or numeric_columns(df)
    cols = [c for c in cols if c in df.columns][:8]
    if len(cols) < 2:
        raise OpError("correlation needs at least two numeric columns")
    data = pd.DataFrame({c: _num(df, c) for c in cols})
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            sub = data[[a, b]].dropna()
            if len(sub) < 3 or sub[a].nunique() < 2 or sub[b].nunique() < 2:
                continue
            fn = sps.pearsonr if op.method == "pearson" else sps.spearmanr
            r, p = fn(sub[a], sub[b])
            pairs.append({"a": a, "b": b, "r": float(r), "p_value": float(p), "n": len(sub),
                          "significant_at_0.05": bool(p < 0.05)})
    pairs.sort(key=lambda x: -abs(x["r"]))
    return {"method": op.method, "pairs": pairs}


def _compare_groups(df: pd.DataFrame, op: CompareGroups, _lists) -> dict:
    g = _cat(_col(df, op.group_column))
    v = _num(df, op.value_column)
    data = pd.DataFrame({"g": g, "v": v}).dropna()
    available = data["g"].value_counts()
    if op.groups:
        wanted = {str(x).lower() for x in op.groups}
        missing = sorted(wanted - set(available.index.str.lower()))
        if missing:
            raise OpError(f"group(s) {missing} not found in {op.group_column!r}; available: "
                          f"{available.head(15).index.tolist()}")
        data = data[data["g"].str.lower().isin(wanted)]
    else:
        data = data[data["g"].isin(data["g"].value_counts().head(5).index)]
    samples = {k: grp["v"].to_numpy() for k, grp in data.groupby("g") if len(grp) >= 2}
    if len(samples) < 2:
        raise OpError("need at least two groups with 2+ numeric values each")
    summary = [{"group": k, "n": len(x), "mean": x.mean(), "median": np.median(x), "std": x.std(ddof=1)}
               for k, x in samples.items()]
    out: dict[str, Any] = {"value_column": op.value_column, "groups": summary}
    arrays = list(samples.values())
    if len(arrays) == 2:
        a, b = arrays
        t = sps.ttest_ind(a, b, equal_var=False)
        u = sps.mannwhitneyu(a, b, alternative="two-sided")
        pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        out["tests"] = {
            "welch_t_test": {"t": t.statistic, "p_value": t.pvalue},
            "mann_whitney_u": {"u": u.statistic, "p_value": u.pvalue},
            "cohens_d": (a.mean() - b.mean()) / pooled if pooled else None,
        }
        p_main = t.pvalue
    else:
        f = sps.f_oneway(*arrays)
        k = sps.kruskal(*arrays)
        out["tests"] = {"anova": {"f": f.statistic, "p_value": f.pvalue},
                        "kruskal_wallis": {"h": k.statistic, "p_value": k.pvalue}}
        p_main = f.pvalue
    out["significant_at_0.05"] = bool(p_main < 0.05)
    if min(len(x) for x in arrays) < 5:
        out["warning"] = "some groups have fewer than 5 values; treat the tests as indicative only"
    return out


def _outliers(df: pd.DataFrame, op: Outliers, _lists) -> dict:
    s = _num(df, op.column)
    clean = s.dropna()
    if op.method == "iqr":
        q1, q3 = clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
        bounds = {"lower": lo, "upper": hi}
    else:
        z = (s - clean.mean()) / clean.std(ddof=1)
        mask = z.abs() > 3
        bounds = {"z_threshold": 3}
    show = [c for c in (op.show_columns or []) if c in df.columns]
    rows = df.loc[mask.fillna(False), list(dict.fromkeys([*show, op.column]))].head(op.limit)
    return {"method": op.method, "bounds": bounds, "outlier_count": int(mask.sum()), "n": int(clean.size),
            "examples": rows.to_dict("records")}


_FREQ = {"D": "D", "W": "W", "M": "MS", "Q": "QS", "Y": "YS"}


def _time_series(df: pd.DataFrame, op: TimeSeries, _lists) -> dict:
    dates = pd.to_datetime(_col(df, op.date_column), errors="coerce")
    data = pd.DataFrame({"d": dates, "n": 1})
    if op.value_column:
        data["v"] = _num(df, op.value_column)
    data = data.dropna(subset=["d"]).set_index("d").sort_index()
    if len(data) == 0:
        raise OpError(f"column {op.date_column!r} has no parseable dates")
    r = data.resample(_FREQ[op.freq])
    series = r.size() if (op.agg == "count" or not op.value_column) else getattr(r["v"], op.agg)()
    series = series.tail(60)
    fmt = {"D": "%Y-%m-%d", "W": "%Y-%m-%d", "M": "%Y-%m", "Q": "%Y-%m", "Y": "%Y"}[op.freq]
    points = [{"period": idx.strftime(fmt), "value": v} for idx, v in series.items()]
    return {"freq": op.freq, "agg": op.agg if op.value_column else "count", "points": points,
            "first": points[0]["period"], "last": points[-1]["period"]}


def _crosstab(df: pd.DataFrame, op: Crosstab, _lists) -> dict:
    a, b = _cat(_col(df, op.row)), _cat(_col(df, op.column))
    keep_a = a.value_counts().head(op.top_n).index
    keep_b = b.value_counts().head(op.top_n).index
    mask = a.isin(keep_a) & b.isin(keep_b)
    table = pd.crosstab(a[mask], b[mask])
    out: dict[str, Any] = {"row": op.row, "column": op.column}
    if table.shape[0] >= 2 and table.shape[1] >= 2:
        chi2, p, dof, _ = sps.chi2_contingency(table)
        out["chi_square"] = {"chi2": chi2, "p_value": p, "dof": dof, "significant_at_0.05": bool(p < 0.05)}
    shown = table.div(table.sum(axis=1), axis=0) * 100 if op.normalize else table
    out["table"] = {str(r): row.to_dict() for r, row in shown.iterrows()}
    out["values"] = "row %" if op.normalize else "counts"
    return out


def _top_n(df: pd.DataFrame, op: TopN, _lists) -> dict:
    s = _num(df, op.column)
    order = s.sort_values(ascending=op.ascending, na_position="last").dropna().head(op.n).index
    show = [c for c in (op.show_columns or []) if c in df.columns]
    cols = list(dict.fromkeys([*show, op.column]))
    return {"column": op.column, "ascending": op.ascending, "rows": df.loc[order, cols].to_dict("records")}


def _histogram(df: pd.DataFrame, op: Histogram, _lists) -> dict:
    s = _num(df, op.column).dropna()
    counts, edges = np.histogram(s, bins=max(2, min(op.bins, 50)))
    return {"column": op.column, "n": int(s.size),
            "bins": [{"from": edges[i], "to": edges[i + 1], "count": int(c)} for i, c in enumerate(counts)]}


def _regression(df: pd.DataFrame, op: Regression, _lists) -> dict:
    cols = [op.target, *op.features]
    data = pd.DataFrame({c: _num(df, c) for c in cols}).dropna()
    n, k = len(data), len(op.features)
    if n < k + 3:
        raise OpError(f"regression needs at least {k + 3} complete rows, found {n}")
    X = np.column_stack([np.ones(n), data[op.features].to_numpy(dtype=float)])
    y = data[op.target].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k - 1
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvals = 2 * sps.t.sf(np.abs(tvals), dof)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot else None
    names = ["intercept", *op.features]
    return {"target": op.target, "n": n, "r_squared": r2,
            "adj_r_squared": (1 - (1 - r2) * (n - 1) / dof) if r2 is not None else None,
            "coefficients": [{"term": names[i], "estimate": beta[i], "std_error": se[i], "t": tvals[i],
                              "p_value": pvals[i]} for i in range(len(names))]}


_DISPATCH = {
    "describe": _describe, "value_counts": _value_counts, "groupby": _groupby, "correlation": _correlation,
    "compare_groups": _compare_groups, "outliers": _outliers, "time_series": _time_series,
    "crosstab": _crosstab, "top_n": _top_n, "histogram": _histogram, "regression": _regression,
}


def run_operation(df: pd.DataFrame, op: BaseModel, list_columns: list[str] | None = None) -> dict:
    params = op.model_dump(exclude={"op"}, exclude_none=True)
    try:
        result = _DISPATCH[op.op](df, op, list_columns or [])
        return {"op": op.op, "params": params, "result": _clean(result)}
    except OpError as exc:
        return {"op": op.op, "params": params, "error": str(exc)}
    except Exception as exc:
        return {"op": op.op, "params": params, "error": f"{type(exc).__name__}: {exc}"}

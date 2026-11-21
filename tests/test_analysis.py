import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from scraper_mcp.agents.audit import audit_numbers
from scraper_mcp.analysis.charts import ChartSpec, render_chart
from scraper_mcp.analysis.ops import Filter, Operation, apply_filters, run_operation
from scraper_mcp.analysis.profile import profile

OPS = TypeAdapter(Operation)


@pytest.fixture
def df():
    rng = np.random.default_rng(7)
    n = 60
    brand = rng.choice(["Acme", "Zeta"], n)
    price = np.where(brand == "Acme", rng.normal(150, 10, n), rng.normal(100, 10, n))
    return pd.DataFrame({"brand": brand, "price": price, "rating": rng.uniform(2, 5, n),
                         "skills": [["python", "sql"] if i % 3 else ["sql"] for i in range(n)],
                         "posted": pd.date_range("2026-01-01", periods=n, freq="3D")})


def run(df, spec, lists=("skills",)):
    out = run_operation(df, OPS.validate_python(spec), list(lists))
    assert "error" not in out, out.get("error")
    return out["result"]


def test_compare_groups_detects_real_difference(df):
    r = run(df, {"op": "compare_groups", "group_column": "brand", "value_column": "price"})
    p = r["tests"]["welch_t_test"]["p_value"]
    assert r["significant_at_0.05"] and 0 < p < 1e-6  # tiny p-values keep significant figures, never 0.0
    assert abs(r["tests"]["cohens_d"]) > 2


def test_value_counts_explodes_lists(df):
    r = run(df, {"op": "value_counts", "column": "skills", "explode": True})
    assert {row["value"] for row in r["top"]} == {"python", "sql"} and r["total"] == 100


def test_time_series_counts_every_row(df):
    r = run(df, {"op": "time_series", "date_column": "posted", "freq": "M"})
    assert sum(p["value"] for p in r["points"]) == 60


def test_regression_recovers_slope():
    x = np.arange(50, dtype=float)
    d = pd.DataFrame({"x": x, "y": 3 * x + 2})
    r = run(d, {"op": "regression", "target": "y", "features": ["x"]}, lists=())
    assert r["r_squared"] == 1.0 and r["coefficients"][1]["estimate"] == 3.0


def test_groupby_reports_hidden_groups(df):
    r = run(df, {"op": "groupby", "by": ["brand"], "metric": "price", "top_n": 1})
    assert r["groups_shown"] == 1 and r["groups_total"] == 2 and r["groups_hidden"] == 1


def test_unknown_column_is_a_clear_error(df):
    out = run_operation(df, OPS.validate_python({"op": "groupby", "by": ["nope"]}), [])
    assert "not found" in out["error"]


def test_filters(df):
    out = apply_filters(df, [Filter(column="brand", op="==", value="acme"), Filter(column="price", op=">", value="140")])
    assert len(out) and (out["brand"] == "Acme").all() and (out["price"] > 140).all()


def test_profile_and_charts(df):
    p = profile(df, ["skills"])
    kinds = {c["name"]: c["kind"] for c in p["columns"]}
    assert kinds == {"brand": "category", "price": "numeric", "rating": "numeric", "skills": "list", "posted": "date"}
    for spec in [ChartSpec(kind="bar", x="skills"), ChartSpec(kind="hist", x="price"),
                 ChartSpec(kind="line", x="posted"), ChartSpec(kind="scatter", x="rating", y="price", group="brand"),
                 ChartSpec(kind="box", x="brand", y="price")]:
        assert render_chart(df, spec, ["skills"]).startswith(b"\x89PNG")


def test_number_audit():
    results = [{"result": {"median": 149.87, "p_value": 0.0012, "pct": 66.6667}}]
    text = "Median 149.9 AED, p = 0.001, 66.7% of rows; that is 1.5x more, and 4,500 at most."
    assert audit_numbers(text, results) == ["1.5", "4,500"]


def test_compare_groups_names_missing_groups(df):
    out = run_operation(df, OPS.validate_python(
        {"op": "compare_groups", "group_column": "brand", "value_column": "price", "groups": ["Acme", "Nova"]}), [])
    assert "['nova'] not found" in out["error"] and "Acme" in out["error"]


def test_number_audit_accepts_numbers_inside_keys():
    results = [{"result": {"columns": {"price": {"p25": 18.59, "p75": 47.95}}}}]
    assert audit_numbers("The 25th and 75th percentiles are 18.59 and 47.95.", results) == []

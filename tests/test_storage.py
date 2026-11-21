import pandas as pd
import pytest

from scraper_mcp.storage import Storage, StorageError


@pytest.fixture
def st(tmp_path):
    return Storage(tmp_path / "t.duckdb", tmp_path / "exports")


def test_save_append_dedupe_load(st):
    df = pd.DataFrame({"name": ["a", "b"], "price": [1.0, 2.0], "tags": [["x"], []], "_run_id": ["1", "1"]})
    assert st.save_dataset("My Shop!", df, mode="m", description="", source="", columns={}, list_columns=["tags"])["dataset"] == "my_shop"
    again = df.assign(_run_id="2")
    assert st.save_dataset("my_shop", again, mode="m", description="", source="", columns={}, list_columns=["tags"])["rows"] == 2
    assert st.load("my_shop")["tags"].tolist() == ['["x"]', "[]"]
    assert st.meta("my_shop")["list_columns"] == ["tags"]


@pytest.mark.parametrize("sql", [
    "select getenv('OPENROUTER_API_KEY')",
    "select * from read_text('/etc/passwd')",
    "select * from read_csv('/etc/hosts')",
    "copy t to '/tmp/x.csv'",
    "select 1; drop table t",
    "attach '/tmp/x.db'",
    "install httpfs",
])
def test_sql_sandbox_blocks(st, sql):
    st.save_dataset("t", pd.DataFrame({"a": [1]}), mode="m", description="", source="", columns={}, list_columns=[])
    with pytest.raises(StorageError):
        st.run_sql(sql)


def test_sql_reads_datasets_and_telemetry(st):
    st.save_dataset("t", pd.DataFrame({"a": [1, 2, 3]}), mode="m", description="", source="", columns={}, list_columns=[])
    assert st.run_sql("select sum(a) s from t").iloc[0]["s"] == 6
    st.log_fetches("r", [{"url": "https://a.com/1", "final_url": "", "status": 200, "method": "static",
                          "content_type": "text/html", "elapsed_ms": 5, "bytes": 1, "robots_allowed": True,
                          "js_fallback": False, "ok": True, "error": None}])
    assert st.run_sql("select domain from _fetches").iloc[0]["domain"] == "a.com"


def test_export_formats(st):
    st.save_dataset("t", pd.DataFrame({"a": [1]}), mode="m", description="", source="", columns={}, list_columns=[])
    for fmt in ("csv", "json", "parquet"):
        assert st.export("t", fmt).stat().st_size > 0

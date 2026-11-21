"""DuckDB storage: one table per dataset, plus run and fetch telemetry.

A connection is opened per operation and closed right away, so a stdio container and
an HTTP container can share the same /data volume without holding the file lock.
"""

from __future__ import annotations

import json
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

import duckdb
import pandas as pd

_NAME = re.compile(r"[^a-z0-9_]+")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SQL_START = re.compile(r"^\s*(select|with|from|summarize|describe|pivot|unpivot|values|table)\b", re.I)
TELEMETRY_VIEWS = {"_runs": "runs", "_fetches": "fetches"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    name VARCHAR PRIMARY KEY, mode VARCHAR, description VARCHAR, source VARCHAR,
    columns_json VARCHAR, list_columns_json VARCHAR, row_count BIGINT,
    created_at TIMESTAMP, updated_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY, tool VARCHAR, request VARCHAR, mode VARCHAR, dataset VARCHAR,
    started_at TIMESTAMP, finished_at TIMESTAMP, status VARCHAR, error VARCHAR,
    pages_attempted INTEGER, pages_ok INTEGER, records_extracted INTEGER, records_kept INTEGER,
    low_grounding INTEGER, llm_calls INTEGER, input_tokens BIGINT, output_tokens BIGINT,
    cost_usd DOUBLE, agent_trace VARCHAR
);
CREATE TABLE IF NOT EXISTS fetches (
    run_id VARCHAR, fetched_at TIMESTAMP, url VARCHAR, final_url VARCHAR, domain VARCHAR,
    status INTEGER, method VARCHAR, content_type VARCHAR, elapsed_ms INTEGER, bytes BIGINT,
    robots_allowed BOOLEAN, js_fallback BOOLEAN, ok BOOLEAN, error VARCHAR
);
"""


class StorageError(RuntimeError):
    pass


def sanitize_name(name: str) -> str:
    clean = _NAME.sub("_", (name or "").strip().lower()).strip("_")[:60]
    if not clean:
        raise StorageError("dataset name is empty after cleaning; use letters, digits and underscores")
    if clean[0].isdigit():
        clean = f"d_{clean}"
    return clean


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Storage:
    def __init__(self, path: Path, exports_dir: Path):
        self.path = Path(path)
        self.exports_dir = Path(exports_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self.conn() as c:
            c.execute(_SCHEMA)

    @contextmanager
    def conn(self) -> Iterator[duckdb.DuckDBPyConnection]:
        with self._lock:
            deadline = time.monotonic() + 15
            while True:
                try:
                    con = duckdb.connect(str(self.path))
                    break
                except duckdb.IOException as exc:  # another process holds the write lock
                    if time.monotonic() > deadline:
                        raise StorageError(f"database is busy (locked by another process): {exc}") from exc
                    time.sleep(0.2)
            try:
                yield con
            finally:
                con.close()

    # ------------------------------------------------------------------ datasets
    def exists(self, name: str) -> bool:
        with self.conn() as c:
            return bool(c.execute("SELECT count(*) FROM datasets WHERE name = ?", [name]).fetchone()[0])

    def save_dataset(self, name: str, df: pd.DataFrame, *, mode: str, description: str, source: str,
                     columns: dict[str, str], list_columns: list[str], append: bool = True) -> dict:
        name = sanitize_name(name)
        df = _to_storable(df, list_columns)
        with self.conn() as c:
            existing = c.execute("SELECT columns_json, list_columns_json, created_at FROM datasets WHERE name = ?",
                                 [name]).fetchone()
            if append and existing:
                old = c.execute(f'SELECT * FROM "ds_{name}"').df()
                df = pd.concat([old, df], ignore_index=True)
                content_cols = [col for col in df.columns if not col.startswith("_")]
                df = df.drop_duplicates(subset=content_cols, keep="first").reset_index(drop=True)
                columns = {**json.loads(existing[0] or "{}"), **columns}
                list_columns = sorted(set(json.loads(existing[1] or "[]")) | set(list_columns))
            created = existing[2] if existing else _now()
            c.register("incoming_df", df)
            c.execute(f'CREATE OR REPLACE TABLE "ds_{name}" AS SELECT * FROM incoming_df')
            c.unregister("incoming_df")
            c.execute(
                "INSERT OR REPLACE INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [name, mode, description[:500], source[:1000], json.dumps(columns), json.dumps(list_columns),
                 len(df), created, _now()],
            )
        return {"dataset": name, "rows": len(df), "appended": bool(append and existing)}

    def load(self, name: str) -> pd.DataFrame:
        name = sanitize_name(name)
        if not self.exists(name):
            raise StorageError(f"no dataset named {name!r}; call list_datasets to see what exists")
        with self.conn() as c:
            return c.execute(f'SELECT * FROM "ds_{name}"').df()

    def meta(self, name: str) -> dict:
        name = sanitize_name(name)
        with self.conn() as c:
            row = c.execute("SELECT * FROM datasets WHERE name = ?", [name]).fetchone()
            if not row:
                raise StorageError(f"no dataset named {name!r}; call list_datasets to see what exists")
            cols = [d[0] for d in c.description]
        meta = dict(zip(cols, row))
        meta["columns"] = json.loads(meta.pop("columns_json") or "{}")
        meta["list_columns"] = json.loads(meta.pop("list_columns_json") or "[]")
        return meta

    def list_datasets(self) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT name, mode, row_count, description, created_at, updated_at "
                             "FROM datasets ORDER BY updated_at DESC").fetchall()
        return [{"dataset": r[0], "mode": r[1], "rows": r[2], "description": r[3],
                 "created_at": str(r[4]), "updated_at": str(r[5])} for r in rows]

    def delete(self, name: str) -> bool:
        name = sanitize_name(name)
        with self.conn() as c:
            c.execute(f'DROP TABLE IF EXISTS "ds_{name}"')
            n = c.execute("DELETE FROM datasets WHERE name = ? RETURNING name", [name]).fetchall()
        return bool(n)

    def export(self, name: str, fmt: str) -> Path:
        name = sanitize_name(name)
        df = self.load(name)
        stamp = _now().strftime("%Y%m%d_%H%M%S")
        path = self.exports_dir / f"{name}_{stamp}.{fmt}"
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "json":
            df.to_json(path, orient="records", date_format="iso", force_ascii=False, indent=1)
        elif fmt == "parquet":
            with self.conn() as c:
                c.register("export_df", df)
                target = str(path).replace("'", "''")
                c.execute(f"COPY export_df TO '{target}' (FORMAT parquet)")
                c.unregister("export_df")
        else:
            raise StorageError("format must be csv, json or parquet")
        return path

    # ------------------------------------------------------------------ telemetry
    def log_run(self, run: dict) -> None:
        cols = ["run_id", "tool", "request", "mode", "dataset", "started_at", "finished_at", "status", "error",
                "pages_attempted", "pages_ok", "records_extracted", "records_kept", "low_grounding", "llm_calls",
                "input_tokens", "output_tokens", "cost_usd", "agent_trace"]
        values = [run.get(c) for c in cols]
        with self.conn() as c:
            c.execute(f"INSERT OR REPLACE INTO runs VALUES ({', '.join('?' * len(cols))})", values)

    def log_fetches(self, run_id: str, rows: list[dict]) -> None:
        if not rows:
            return
        now = _now()
        values = [[run_id, now, r["url"], r["final_url"], urlsplit(r["final_url"] or r["url"]).netloc,
                   r["status"], r["method"], r["content_type"], r["elapsed_ms"], r["bytes"], r["robots_allowed"],
                   r["js_fallback"], r["ok"], r["error"]] for r in rows]
        with self.conn() as c:
            c.executemany(f"INSERT INTO fetches VALUES ({', '.join('?' * 14)})", values)

    def telemetry_frames(self) -> dict[str, pd.DataFrame]:
        with self.conn() as c:
            return {view: c.execute(f"SELECT * FROM {table}").df() for view, table in TELEMETRY_VIEWS.items()}

    # ------------------------------------------------------------------ read-only SQL
    def run_sql(self, sql: str, max_rows: int = 200, timeout_s: float = 15.0) -> pd.DataFrame:
        """Run a read-only query in an isolated in-memory DuckDB.

        Only the datasets named in the query are copied in. File, network and extension
        access are disabled and the configuration is locked, so the query cannot read
        files, environment variables or the main database.
        """
        statement = _strip_sql(sql)
        if not _SQL_START.match(statement):
            raise StorageError("only read-only queries are allowed (SELECT / WITH / FROM / SUMMARIZE / DESCRIBE)")
        if ";" in statement.rstrip(";").strip():
            raise StorageError("send one statement at a time")
        statement = statement.rstrip(";").strip()

        names = {d["dataset"] for d in self.list_datasets()}
        idents = set(_IDENT.findall(statement))
        frames = {n: self.load(n) for n in names & idents}
        if idents & set(TELEMETRY_VIEWS):
            frames.update({k: v for k, v in self.telemetry_frames().items() if k in idents})

        con = duckdb.connect(":memory:")
        try:
            for name, frame in frames.items():
                con.register(name, frame)
            con.execute("SET enable_external_access = false")
            con.execute("SET lock_configuration = true")
            timer = threading.Timer(timeout_s, con.interrupt)
            timer.start()
            try:
                return con.execute(f"SELECT * FROM ({statement}) AS q LIMIT {int(max_rows)}").df()
            except duckdb.Error as exc:
                raise StorageError(f"SQL error: {exc}") from exc
            finally:
                timer.cancel()
        finally:
            con.close()


def _strip_sql(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return sql.strip()


def _to_storable(df: pd.DataFrame, list_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in list_columns or df[col].map(lambda v: isinstance(v, (list, dict))).any():
            df[col] = df[col].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
    return df


def parse_list_cell(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except ValueError:
            return []
    return []

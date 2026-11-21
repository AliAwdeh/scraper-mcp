"""The Verifier: deterministic quality control between extraction and storage.

No LLM here. It cleans types, drops empty and duplicate records, and computes a
*grounding score* per record: the share of copied values (names, numbers, ...) that
can actually be found in the source page. Low scores flag likely hallucinations.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

from ..modes import Mode

META_COLUMNS = ["_source_url", "_page_title", "_fetch_method", "_grounding", "_run_id", "_scraped_at"]
LOW_GROUNDING = 0.5
_THOUSANDS = re.compile(r"(?<=\d)[,  ' ](?=\d{3}(?!\d))")
_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"})


def normalise_source(*parts: str) -> str:
    text = " ".join(p for p in parts if p)
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES).lower()
    text = _THOUSANDS.sub("", text)
    return _WS.sub(" ", text)


def _norm_str(value: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", value).translate(_QUOTES).lower()).strip()


def _number_found(value: float, source: str) -> bool:
    """56.5 matches '56.5' and '56.50'; 1299 matches '1299', '1,299' (separators already removed) and '1299.00'."""
    if float(value).is_integer():
        # ...but not inside '56.50' (a decimal continuation that is not all zeros)
        pattern = rf"(?<![\d.]){int(value)}(?:[.,]0+)?(?!\d|[.,]\d)"
        return re.search(pattern, source) is not None
    forms = {f"{value:g}", f"{value:.2f}".rstrip("0").rstrip(".")}
    return any(re.search(rf"(?<![\d.]){re.escape(form)}0*(?!\d)", source) for form in forms)


_ABBREVIATIONS = ((1e9, r"(?:bn|b|billion)"), (1e6, r"(?:m|mn|million)"), (1e3, r"(?:k|thousand)"))


def _abbreviated_found(value: float, source: str) -> bool:
    """150000 matches '$150k', 119800 matches '119.8k', 132000 matches '132.0k', 1200000 matches '1.2m'."""
    for divisor, suffix in _ABBREVIATIONS:
        if abs(value) >= divisor:
            scaled = f"{value / divisor:.3f}".rstrip("0").rstrip(".")
            if re.search(rf"(?<![\d.]){re.escape(scaled)}(?:[.,]0+)?\s?{suffix}\b", source):
                return True
    return False


def _string_found(value: str, source: str) -> bool:
    v = _norm_str(value)
    if not v:
        return False
    if len(v) > 60:  # long strings: the opening words are enough
        v = v[:40]
    return v in source


def grounding_score(record: dict, mode: Mode, source: str) -> float | None:
    checked = found = 0
    for name, value in record.items():
        if name == "url" or name in mode.inferred or name.startswith("_") or value is None:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            checked += 1
            found += _number_found(float(value), source) or _abbreviated_found(float(value), source)
        elif isinstance(value, str):
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):  # dates are often reformatted; skip
                continue
            if len(value) <= 3 and value.isupper():  # currency codes are usually inferred from symbols
                continue
            checked += 1
            found += _string_found(value, source)
        elif isinstance(value, list) and value:
            items = [x for x in value if isinstance(x, str) and x.strip()]
            if items:
                checked += 1
                found += sum(_string_found(x, source) for x in items) / len(items)
    return round(found / checked, 3) if checked else None


def _clean_value(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, list):
        return [v.strip() if isinstance(v, str) else v for v in value if v not in (None, "")]
    return value


def verify_records(
    raw_records: list[dict], mode: Mode, *, source_url: str, page_title: str, source_text: str,
    fetch_method: str, run_id: str,
) -> tuple[list[dict], dict]:
    """Return (kept_records, stats) for one page."""
    source = normalise_source(source_text)
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    kept: list[dict] = []
    stats = {"extracted": len(raw_records), "empty": 0, "low_grounding": 0}
    for raw in raw_records:
        rec = {k: _clean_value(v) for k, v in raw.items()}
        for key in ("currency", "salary_currency"):
            if isinstance(rec.get(key), str):
                rec[key] = rec[key].upper()[:3]
        content = {k: v for k, v in rec.items() if k != "url" and v not in (None, [], "")}
        if not content:
            stats["empty"] += 1
            continue
        score = grounding_score(rec, mode, source)
        if score is not None and score < LOW_GROUNDING:
            stats["low_grounding"] += 1
        rec.update({
            "_source_url": source_url, "_page_title": page_title[:200], "_fetch_method": fetch_method,
            "_grounding": score, "_run_id": run_id, "_scraped_at": now,
        })
        kept.append(rec)
    return kept, stats


def dedupe(records: list[dict], keys: tuple[str, ...] | None = None) -> tuple[list[dict], int]:
    """Drop repeated records. With `keys` (a mode's identity fields) the same job posted twice under two
    IDs counts once; without them, records must match on every content field, URL included."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for rec in records:
        fields = keys if keys and any(rec.get(k) not in (None, "", []) for k in keys) else None
        key = tuple(
            (k, tuple(v) if isinstance(v, list) else (v.lower().strip() if isinstance(v, str) and fields else v))
            for k, v in sorted(rec.items())
            if not k.startswith("_") and (fields is None or k in fields)
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out, len(records) - len(out)

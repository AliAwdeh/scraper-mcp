from __future__ import annotations

import re
from typing import Any

_NUM_IN_TEXT = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?")
_NUM_ANYWHERE = re.compile(r"\d+(?:\.\d+)?")


def _collect(value: Any, out: list[float]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, str):
        for m in _NUM_IN_TEXT.findall(value) + _NUM_ANYWHERE.findall(value):
            try:
                out.append(float(m.replace(",", "")))
            except ValueError:
                pass
    elif isinstance(value, dict):
        for k, v in value.items():
            _collect(k, out)
            _collect(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _collect(v, out)


def audit_numbers(text: str, *sources: Any) -> list[str]:
    known: list[float] = []
    for s in sources:
        _collect(s, known)
    unverified: list[str] = []
    for raw in _NUM_IN_TEXT.findall(text or ""):
        num = float(raw.replace(",", ""))
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if abs(num) <= 10 and decimals == 0:
            continue
        ok = False
        for k in known:
            if round(k, decimals) == num or abs(k - num) <= 0.01 * max(abs(k), 1e-9):
                ok = True
                break
            if abs(k) <= 1 and (round(k * 100, decimals) == num or abs(k * 100 - num) <= 0.5):
                ok = True
                break
        if not ok and raw not in unverified:
            unverified.append(raw)
    return unverified

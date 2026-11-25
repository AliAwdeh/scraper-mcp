from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

log = logging.getLogger(__name__)

SKIP_DOMAINS = (
    "youtube.com", "youtu.be", "facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com",
    "linkedin.com", "pinterest.com", "reddit.com", "quora.com",
)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def as_dict(self) -> dict:
        return asdict(self)


class SearchError(RuntimeError):
    pass


def _skip(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in SKIP_DOMAINS)


async def web_search(query: str, max_results: int = 10, region: str = "us-en") -> list[SearchResult]:
    def _run() -> list[dict]:
        return DDGS().text(query, region=region, safesearch="moderate", max_results=max_results * 2)

    last: Exception | None = None
    rows: list[dict] = []
    for attempt in range(3):
        try:
            rows = await asyncio.to_thread(_run)
            break
        except RatelimitException as exc:
            last = exc
            await asyncio.sleep(2.0 * (attempt + 1))
        except (DDGSException, TimeoutException) as exc:
            last = exc
            await asyncio.sleep(1.0)
    else:
        raise SearchError(f"search failed after 3 attempts: {last}")

    seen: set[str] = set()
    out: list[SearchResult] = []
    for row in rows or []:
        url = row.get("href") or row.get("url") or ""
        if not url.startswith("http") or url in seen or _skip(url):
            continue
        seen.add(url)
        out.append(SearchResult(title=row.get("title", "")[:200], url=url, snippet=row.get("body", "")[:400]))
        if len(out) >= max_results:
            break
    return out

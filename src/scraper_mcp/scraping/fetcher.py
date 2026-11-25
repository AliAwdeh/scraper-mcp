from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

from ..config import Settings
from .browser import BrowserPool
from .cleaner import looks_js_rendered
from .politeness import BlockedURL, DomainRateLimiter, RobotsCache, check_url_allowed

log = logging.getLogger(__name__)

_TEXT_TYPES = ("text/html", "application/xhtml", "text/plain", "application/json", "application/ld+json",
               "text/xml", "application/xml", "application/rss", "application/atom")


@dataclass
class FetchResult:
    url: str
    final_url: str = ""
    status: int | None = None
    method: str = "static"
    html: str = ""
    content_type: str = ""
    elapsed_ms: int = 0
    bytes: int = 0
    robots_allowed: bool = True
    js_fallback: bool = False
    error: str | None = None
    note: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.html)

    @property
    def is_html(self) -> bool:
        return not self.content_type or "html" in self.content_type

    def log_row(self) -> dict:
        return {
            "url": self.url, "final_url": self.final_url or self.url, "status": self.status,
            "method": self.method, "content_type": self.content_type, "elapsed_ms": self.elapsed_ms,
            "bytes": self.bytes, "robots_allowed": self.robots_allowed, "js_fallback": self.js_fallback,
            "ok": self.ok, "error": self.error,
        }


async def _read_some(resp: httpx.Response, limit: int) -> str:
    data = bytearray()
    try:
        async for chunk in resp.aiter_bytes():
            data.extend(chunk)
            if len(data) >= limit:
                break
    except httpx.HTTPError:
        pass
    return " ".join(bytes(data[:limit]).decode("utf-8", errors="replace").split())


class Fetcher:
    def __init__(self, settings: Settings):
        self.s = settings
        self.client = httpx.AsyncClient(
            headers={
                "User-Agent": settings.effective_user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.5",
                "Accept-Language": "en-US,en;q=0.8",
            },
            timeout=settings.request_timeout,
            follow_redirects=False,
        )
        self.robots = RobotsCache(self.client, settings.effective_user_agent)
        self.limiter = DomainRateLimiter(settings.request_delay)
        self.browser = BrowserPool(settings.effective_user_agent, settings.browser_timeout, settings.allow_private_networks)
        self._sem = asyncio.Semaphore(settings.max_concurrency)

    async def fetch(self, url: str, render: str | None = None) -> FetchResult:
        render = render or self.s.render_mode
        res = FetchResult(url=url)
        started = time.perf_counter()
        try:
            await check_url_allowed(url, self.s.allow_private_networks)
            delay = None
            if self.s.respect_robots:
                if not await self.robots.allowed(url):
                    res.robots_allowed = False
                    res.error = "disallowed by robots.txt"
                    return res
                delay = await self.robots.crawl_delay(url)
            async with self._sem:
                if render != "browser":
                    await self.limiter.wait(url, delay)
                    await self._static(url, res)
                if render == "browser" or (render == "auto" and self._needs_browser(res)):
                    res.js_fallback = render == "auto"
                    await self.limiter.wait(url, delay)
                    await self._render(url, res)
        except BlockedURL as exc:
            res.error = str(exc)
        except httpx.HTTPError as exc:
            res.error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        except Exception as exc:
            log.exception("fetch failed for %s", url)
            res.error = f"{type(exc).__name__}: {exc}"
        finally:
            res.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return res

    async def _static(self, url: str, res: FetchResult) -> None:
        current = url
        for _ in range(6):
            async with self.client.stream("GET", current) as resp:
                if resp.is_redirect and resp.headers.get("location"):
                    current = str(resp.url.join(resp.headers["location"]))
                    await check_url_allowed(current, self.s.allow_private_networks)
                    continue
                res.status = resp.status_code
                res.final_url = str(resp.url)
                res.content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if resp.status_code >= 400:
                    res.error = f"HTTP {resp.status_code}"
                    snippet = await _read_some(resp, 600)
                    if snippet and "html" not in res.content_type:
                        res.error += f": {snippet[:240]}"
                    if resp.status_code in (401, 403, 429, 503):
                        res.error += (" (the site refused the request; check its robots/bot policy, set "
                                      "SCRAPER_CONTACT, slow down with REQUEST_DELAY, or use another source)")
                    return
                if res.content_type and not res.content_type.startswith(_TEXT_TYPES):
                    res.error = f"unsupported content type {res.content_type}"
                    return
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self.s.max_bytes:
                        res.note = f"truncated at {self.s.max_bytes} bytes"
                        break
                res.bytes = len(body)
                res.html = bytes(body).decode(resp.encoding or "utf-8", errors="replace")
                return
        res.error = "too many redirects"

    def _needs_browser(self, res: FetchResult) -> bool:
        return res.error is None and res.is_html and looks_js_rendered(res.html)

    async def _render(self, url: str, res: FetchResult) -> None:
        try:
            status, final_url, html = await self.browser.render(url)
        except Exception as exc:
            msg = f"browser render failed: {type(exc).__name__}: {str(exc)[:200]}"
            if res.html and res.error is None:
                res.note = msg + " (kept the static HTML)"
            else:
                res.error = msg
            return
        res.method = "browser"
        res.status = status
        res.final_url = final_url
        res.html = html
        res.bytes = len(html.encode("utf-8", errors="ignore"))
        res.content_type = "text/html"
        res.error = None if (status is None or status < 400) else f"HTTP {status}"

    async def close(self) -> None:
        await self.browser.close()
        await self.client.aclose()

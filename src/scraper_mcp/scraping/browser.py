"""Headless Chromium (Playwright) for pages that only render with JavaScript."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit

from .politeness import host_is_private_literal

log = logging.getLogger(__name__)

_BLOCKED_RESOURCES = {"image", "media", "font"}


class BrowserPool:
    def __init__(self, user_agent: str, timeout_s: float, allow_private: bool, max_pages: int = 3):
        self._ua = user_agent
        self._timeout_ms = int(timeout_s * 1000)
        self._allow_private = allow_private
        self._sem = asyncio.Semaphore(max_pages)
        self._lock = asyncio.Lock()
        self._pw = None
        self._browser = None

    async def _ensure(self):
        if self._browser is not None:
            return self._browser
        async with self._lock:
            if self._browser is None:
                from playwright.async_api import async_playwright

                self._pw = await async_playwright().start()
                # /dev/shm is tiny in containers; let Chromium use /tmp instead.
                self._browser = await self._pw.chromium.launch(headless=True, args=["--disable-dev-shm-usage"])
        return self._browser

    async def render(self, url: str) -> tuple[int | None, str, str]:
        """Load `url` in a fresh browser context. Returns (status, final_url, html)."""
        browser = await self._ensure()
        async with self._sem:
            context = await browser.new_context(user_agent=self._ua, locale="en-US")
            try:
                page = await context.new_page()

                async def _filter(route):
                    req = route.request
                    host = urlsplit(req.url).hostname or ""
                    if req.resource_type in _BLOCKED_RESOURCES:
                        return await route.abort()
                    if not self._allow_private and host_is_private_literal(host):
                        return await route.abort()
                    return await route.continue_()

                await page.route("**/*", _filter)
                response = await page.goto(url, wait_until="domcontentloaded", timeout=self._timeout_ms)
                # Give client-side rendering a moment; do not fail if the network never idles.
                try:
                    await page.wait_for_load_state("networkidle", timeout=min(8000, self._timeout_ms))
                except Exception:
                    pass
                # Nudge lazy-loaded lists.
                try:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(600)
                except Exception:
                    pass
                html = await page.content()
                return (response.status if response else None), page.url, html
            finally:
                await context.close()

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._pw is not None:
            await self._pw.stop()
            self._pw = None

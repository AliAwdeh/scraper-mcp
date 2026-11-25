from __future__ import annotations

import asyncio
import ipaddress
import socket
import time
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx


class BlockedURL(ValueError):
    pass


def _is_private_ip(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


async def check_url_allowed(url: str, allow_private: bool) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise BlockedURL(f"only http(s) URLs are allowed, got {parts.scheme or 'no scheme'!r}")
    host = parts.hostname
    if not host:
        raise BlockedURL("URL has no host")
    if allow_private:
        return
    if host.lower() in ("localhost", "localhost.localdomain") or host.lower().endswith(".localhost"):
        raise BlockedURL(f"{host} is a local address (set ALLOW_PRIVATE_NETWORKS=true to allow)")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise BlockedURL(f"cannot resolve host {host}: {exc}") from exc
    for info in infos:
        ip = info[4][0]
        if _is_private_ip(ip):
            raise BlockedURL(f"{host} resolves to private address {ip} (set ALLOW_PRIVATE_NETWORKS=true to allow)")


def host_is_private_literal(host: str) -> bool:
    if host.lower() in ("localhost",) or host.lower().endswith(".localhost"):
        return True
    try:
        return _is_private_ip(host.strip("[]"))
    except ValueError:
        return False


class RobotsCache:
    def __init__(self, client: httpx.AsyncClient, user_agent: str, ttl: float = 3600):
        self._client = client
        self._ua = user_agent
        self._ttl = ttl
        self._cache: dict[str, tuple[float, RobotFileParser | None, bool]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _get(self, origin: str) -> tuple[RobotFileParser | None, bool]:
        now = time.monotonic()
        cached = self._cache.get(origin)
        if cached and now - cached[0] < self._ttl:
            return cached[1], cached[2]
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            cached = self._cache.get(origin)
            if cached and now - cached[0] < self._ttl:
                return cached[1], cached[2]
            parser: RobotFileParser | None = None
            allow_all = False
            try:
                resp = await self._client.get(f"{origin}/robots.txt", follow_redirects=True, timeout=10)
                if resp.status_code >= 500:
                    allow_all = False
                elif resp.status_code >= 400:
                    allow_all = True
                else:
                    parser = RobotFileParser()
                    parser.parse(resp.text.splitlines())
            except httpx.HTTPError:
                allow_all = False
            self._cache[origin] = (now, parser, allow_all)
            return parser, allow_all

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        parser, allow_all = await self._get(f"{parts.scheme}://{parts.netloc}")
        if parser is None:
            return allow_all
        return parser.can_fetch(self._ua, url)

    async def crawl_delay(self, url: str) -> float | None:
        parts = urlsplit(url)
        parser, _ = await self._get(f"{parts.scheme}://{parts.netloc}")
        if parser is None:
            return None
        delay = parser.crawl_delay(self._ua)
        return float(delay) if delay is not None else None


class DomainRateLimiter:
    def __init__(self, default_delay: float):
        self._default = default_delay
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def wait(self, url: str, delay: float | None = None) -> None:
        domain = urlsplit(url).netloc.lower()
        gap = max(self._default, delay or 0.0)
        lock = self._locks.setdefault(domain, asyncio.Lock())
        async with lock:
            last = self._last.get(domain)
            if last is not None:
                remaining = gap - (time.monotonic() - last)
                if remaining > 0:
                    await asyncio.sleep(remaining)
            self._last[domain] = time.monotonic()

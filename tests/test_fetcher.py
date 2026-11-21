import pytest

from scraper_mcp.config import Settings
from scraper_mcp.scraping.fetcher import Fetcher
from scraper_mcp.scraping.politeness import BlockedURL, check_url_allowed

from .conftest import chromium_available


def _settings(**kw) -> Settings:
    base = dict(allow_private_networks=True, request_delay=0, respect_robots=True)
    base.update(kw)
    return Settings(**base)


async def test_static_fetch(site):
    f = Fetcher(_settings())
    try:
        res = await f.fetch(f"{site}/shop.html")
        assert res.ok and res.method == "static" and not res.js_fallback
        assert res.status == 200 and "Acme Book 14" in res.html
    finally:
        await f.close()


async def test_robots_txt_is_respected(site):
    f = Fetcher(_settings())
    try:
        res = await f.fetch(f"{site}/private/secret.html")
        assert not res.ok and not res.robots_allowed and "robots" in res.error
    finally:
        await f.close()


async def test_http_errors_are_reported(site):
    f = Fetcher(_settings())
    try:
        res = await f.fetch(f"{site}/missing.html")
        assert not res.ok and res.status == 404 and res.error == "HTTP 404"
    finally:
        await f.close()


async def test_ssrf_guard_blocks_private_addresses(site):
    f = Fetcher(_settings(allow_private_networks=False))
    try:
        res = await f.fetch(f"{site}/shop.html")
        assert not res.ok and "private" in res.error
    finally:
        await f.close()
    for bad in ["file:///etc/passwd", "http://169.254.169.254/latest/meta-data", "http://localhost:8000/"]:
        with pytest.raises(BlockedURL):
            await check_url_allowed(bad, allow_private=False)


@pytest.mark.skipif(not chromium_available(), reason="Playwright Chromium not installed (runs in Docker)")
async def test_js_page_falls_back_to_browser(site):
    f = Fetcher(_settings())
    try:
        res = await f.fetch(f"{site}/spa.html")
        assert res.ok and res.method == "browser" and res.js_fallback
        assert "Rendered laptops" in res.html and "Nova Air 13" in res.html
    finally:
        await f.close()


async def test_refusal_is_reported_not_retried_in_a_browser(site):
    f = Fetcher(_settings(render_mode="auto"))
    try:
        res = await f.fetch(f"{site}/forbidden")
        assert not res.ok and res.status == 403 and res.method == "static" and not res.js_fallback
        assert "respect our robot policy" in res.error and "SCRAPER_CONTACT" in res.error
    finally:
        await f.close()


def test_contact_is_added_to_user_agent():
    assert Settings(scraper_contact="https://github.com/me/repo").effective_user_agent == (
        "scraper-mcp/0.1 (educational web-scraping project; +https://github.com/me/repo)")
    assert Settings(scraper_contact="").effective_user_agent == "scraper-mcp/0.1 (educational web-scraping project)"

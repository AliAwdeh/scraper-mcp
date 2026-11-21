"""HTML -> compact Markdown the extraction agents can read, plus the structured
signals that make extraction more accurate (JSON-LD, meta tags, pagination links)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup, Comment, NavigableString
from markdownify import markdownify

_NOISE_TAGS = ["script", "style", "noscript", "svg", "iframe", "template", "canvas", "form",
               "button", "input", "select", "textarea", "link", "meta", "picture", "video", "audio"]
_LAYOUT_TAGS = ["nav", "footer", "aside"]
_NEXT_TEXT = re.compile(r"^\s*(?:(?:next|next page|older posts|older|more results)\s*[›»>→]{0,2}|[›»→]|>>?)\s*$", re.I)
_NEXT_CLASS = re.compile(r"(?:^|[-_])next(?:$|[-_])", re.I)  # "next", "pagination-next", "next-page"
_JS_MARKERS = re.compile(r"(enable javascript|requires javascript|javascript is (disabled|required)|"
                         r"please turn on javascript|you need to enable javascript)", re.I)
_EMPTY_APP_ROOT = re.compile(r'<div[^>]+id=["\'](root|app|__next|__nuxt|svelte)["\'][^>]*>\s*</div>', re.I)


@dataclass
class CleanPage:
    url: str
    title: str = ""
    description: str = ""
    text: str = ""
    json_ld: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    next_page_url: str | None = None
    text_chars: int = 0
    truncated: bool = False

    def json_ld_compact(self, limit: int = 8000) -> str:
        if not self.json_ld:
            return ""
        blob = json.dumps(self.json_ld, ensure_ascii=False, separators=(",", ":"))
        return blob[:limit] + ("…" if len(blob) > limit else "")


def _visible_text(soup: BeautifulSoup) -> str:
    body = soup.body or soup
    return re.sub(r"\s+", " ", body.get_text(" ", strip=True))


def looks_js_rendered(html: str) -> bool:
    """Heuristic: does this page need a real browser to show its content?"""
    if not html:
        return False
    soup = BeautifulSoup(html, "lxml")
    noscript_text = " ".join(n.get_text(" ", strip=True) for n in soup.find_all("noscript"))
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    text_len = len(_visible_text(soup))
    if text_len < 400:
        return True
    if text_len < 1500 and (_JS_MARKERS.search(noscript_text) or _EMPTY_APP_ROOT.search(html)):
        return True
    return False


def _parse_json_ld(soup: BeautifulSoup) -> list[dict]:
    items: list[dict] = []
    for tag in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw = tag.string or tag.get_text() or ""
        try:
            data = json.loads(raw.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        stack = data if isinstance(data, list) else [data]
        for entry in stack:
            if not isinstance(entry, dict):
                continue
            if "@graph" in entry and isinstance(entry["@graph"], list):
                items.extend(e for e in entry["@graph"] if isinstance(e, dict))
            else:
                items.append(entry)
    # Drop pure-navigation noise.
    return [i for i in items if str(i.get("@type", "")).lower() not in ("breadcrumblist", "website", "searchaction")]


def _meta(soup: BeautifulSoup) -> dict:
    out: dict[str, str] = {}
    for m in soup.find_all("meta"):
        key = m.get("property") or m.get("name") or m.get("itemprop")
        val = m.get("content")
        if not key or not val:
            continue
        key = key.lower()
        if key in ("description", "author", "keywords") or key.startswith(("og:", "article:", "product:")):
            out[key] = val.strip()[:500]
    return out


def _find_next_page(soup: BeautifulSoup, base_url: str) -> str | None:
    base_host = urlsplit(base_url).hostname
    candidates: list[str] = []
    link = soup.find("link", rel=lambda v: v and "next" in (v if isinstance(v, list) else [v]))
    if link and link.get("href"):
        candidates.append(link["href"])
    def classed_next(tag) -> bool:
        return tag is not None and any(_NEXT_CLASS.search(c) for c in (tag.get("class") or []))

    for a in soup.find_all("a", href=True):
        rel = a.get("rel") or []
        label = (a.get("aria-label") or "").lower()
        if ("next" in rel or _NEXT_TEXT.match(a.get_text(" ", strip=True) or "") or label.startswith("next")
                or classed_next(a) or classed_next(a.parent)):
            candidates.append(a["href"])
    for href in candidates:
        absolute = urljoin(base_url, href)
        if absolute.startswith("http") and urlsplit(absolute).hostname == base_host and absolute != base_url:
            return absolute
    return None


_SIGNAL = re.compile(r"rating|rated|\bstars?\b|score|reviews?|price|availab|stock|out of \d", re.I)
_RATING_CLASS = re.compile(r"rating|rated|stars?", re.I)
_DATA_ATTR = re.compile(r"^data-[\w-]*(rating|rated|stars?|score|price|value|count|reviews?|date|time)", re.I)
_VOID = {"img", "meta", "input", "br", "hr", "source"}
_SKIP = {"script", "style", "noscript", "template", "svg", "head", "html", "body"}


def _annotate_attributes(soup: BeautifulSoup) -> None:
    """Many pages keep data in attributes rather than text: a CSS class 'star-rating Three', an
    aria-label 'Rated 4.5 out of 5', a full title in title="" behind truncated link text, data-*
    attributes, microdata <meta itemprop content>, <time datetime>, star images with alt text.
    Markdown conversion drops attributes, so surface them as bracketed text next to the element."""
    rating_containers: set[int] = set()
    for tag in soup.find_all(True):  # document order: a container is visited before its children
        if tag.name in _SKIP:
            continue
        # Children of an annotated rating (e.g. the five <i class="icon-star"> inside
        # <p class="star-rating Three">) would mislead: all five are drawn, CSS colours three.
        if rating_containers and any(id(parent) in rating_containers for parent in tag.parents):
            continue
        attrs = tag.attrs
        if not attrs:
            continue
        notes: list[str] = []
        text: str | None = None

        def visible() -> str:
            nonlocal text
            if text is None:
                text = tag.get_text(" ", strip=True).lower()
            return text

        classes = " ".join(attrs.get("class") or [])
        if classes and _RATING_CLASS.search(classes) and len(visible()) < 40:
            notes.append(f"class: {classes}")
            if tag.find(True) is not None:
                rating_containers.add(id(tag))
        aria = attrs.get("aria-label")
        if isinstance(aria, str) and len(aria) <= 100 and _SIGNAL.search(aria) and aria.lower() not in visible():
            notes.append(aria)
        title = attrs.get("title")
        # (links keep their title in the Markdown link syntax already)
        if tag.name != "a" and isinstance(title, str) and 0 < len(title.strip()) <= 200 and title.strip().lower() not in visible():
            notes.append(f"title: {title.strip()}")
        for key, val in attrs.items():
            if isinstance(val, str) and len(val) <= 60 and _DATA_ATTR.match(key):
                notes.append(f"{key}: {val}")
        if tag.name == "time" and attrs.get("datetime") and attrs["datetime"].lower() not in visible():
            notes.append(f"datetime: {attrs['datetime']}")
        if tag.name == "meta" and attrs.get("itemprop") and attrs.get("content"):
            notes.append(f"{attrs['itemprop']}: {attrs['content'][:100]}")
        if tag.name == "img" and isinstance(attrs.get("alt"), str) and _SIGNAL.search(attrs["alt"]) and len(attrs["alt"]) <= 80:
            notes.append(f"image: {attrs['alt']}")
        if notes:
            marker = NavigableString(f" [{'; '.join(notes)}] ")
            if tag.name in _VOID:
                tag.insert_after(marker)
            else:
                tag.append(marker)


def _separate_cards(soup: BeautifulSoup) -> None:
    """Put a rule between repeated <article> cards so a rating or price is not attributed to the
    neighbouring item once the HTML is flattened to text."""
    articles = soup.find_all("article")
    if len(articles) >= 2:
        for article in articles:
            article.insert_before(soup.new_tag("hr"))


def _collapse(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _article_text(html: str, url: str) -> str | None:
    try:
        import trafilatura

        return trafilatura.extract(html, url=url, output_format="markdown", include_tables=True,
                                   include_links=False, include_comments=False, favor_recall=True)
    except Exception:
        return None


_ARTICLE_TYPES = ("article", "newsarticle", "blogposting", "report", "scholarlyarticle", "reportagenewsarticle")


def _looks_like_article(page: CleanPage) -> bool:
    """Use the article extractor only on single-article pages; section pages keep their lists."""
    if page.meta.get("og:type", "").lower() == "article":
        return True
    for item in page.json_ld:
        types = item.get("@type", "")
        types = types if isinstance(types, list) else [types]
        if any(str(t).lower() in _ARTICLE_TYPES for t in types):
            return True
    return False


def clean_html(html: str, base_url: str, *, max_chars: int = 60_000, article: bool = False) -> CleanPage:
    soup = BeautifulSoup(html, "lxml")
    page = CleanPage(url=base_url)
    page.title = (soup.title.get_text(strip=True) if soup.title else "")[:300]
    page.json_ld = _parse_json_ld(soup)
    page.meta = _meta(soup)
    page.description = page.meta.get("description") or page.meta.get("og:description", "")
    page.next_page_url = _find_next_page(soup, base_url)

    text = _article_text(html, base_url) if (article and _looks_like_article(page)) else None
    if not text:
        for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
            comment.extract()
        _annotate_attributes(soup)
        _separate_cards(soup)
        for tag in soup(_NOISE_TAGS):
            tag.decompose()
        for tag in soup(_LAYOUT_TAGS):
            tag.decompose()
        for header in soup.find_all("header"):
            if not header.find_parent(["article", "main"]):
                header.decompose()
        for tag in soup.find_all(attrs={"aria-hidden": "true"}):
            tag.decompose()
        for tag in soup.find_all(hidden=True):
            tag.decompose()
        for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            tag.decompose()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith(("javascript:", "mailto:", "tel:", "#")):
                a.unwrap()
            else:
                a["href"] = urljoin(base_url, href)
        root = soup.find("main") or soup.body or soup
        text = markdownify(str(root), heading_style="ATX", strip=["img"], bullets="-")

    text = _collapse(text)
    page.text_chars = len(text)
    if len(text) > max_chars:
        text = text[:max_chars]
        page.truncated = True
    page.text = text
    return page


def clean_non_html(body: str, url: str, content_type: str, *, max_chars: int) -> CleanPage:
    page = CleanPage(url=url, title=url)
    if "json" in content_type:
        try:
            body = json.dumps(json.loads(body), ensure_ascii=False, indent=1)
        except (json.JSONDecodeError, ValueError):
            pass
    page.text_chars = len(body)
    page.truncated = len(body) > max_chars
    page.text = body[:max_chars]
    return page


def chunk_text(text: str, chunk_chars: int, max_chunks: int) -> list[str]:
    """Split on paragraph boundaries into at most `max_chunks` chunks of ~`chunk_chars`."""
    if len(text) <= chunk_chars:
        return [text]
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        while len(para) > chunk_chars:  # a single huge paragraph
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:chunk_chars])
            para = para[chunk_chars:]
        if len(current) + len(para) + 2 > chunk_chars and current:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
        if len(chunks) >= max_chunks:
            break
    if current and len(chunks) < max_chunks:
        chunks.append(current)
    return chunks[:max_chunks]

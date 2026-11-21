from pathlib import Path

from scraper_mcp.scraping.cleaner import chunk_text, clean_html, looks_js_rendered

SITE = Path(__file__).parent / "fixtures" / "site"


def test_clean_listing_page():
    html = (SITE / "shop.html").read_text()
    page = clean_html(html, "http://shop.test/shop.html")
    assert page.title.startswith("Gadget Shop")
    assert page.next_page_url == "http://shop.test/shop_page2.html"
    assert page.json_ld and page.json_ld[0]["@type"] == "ItemList"
    assert "[Acme Book 14](http://shop.test/p/acme-14)" in page.text  # links made absolute
    assert "My account" not in page.text  # navigation removed
    assert "All rights reserved" not in page.text  # footer removed
    assert ".card" not in page.text  # CSS removed


def test_js_detection():
    assert looks_js_rendered((SITE / "spa.html").read_text())
    assert not looks_js_rendered((SITE / "shop.html").read_text())


def test_chunking_respects_limits():
    text = "\n\n".join(f"paragraph {i} " + "x" * 80 for i in range(100))
    chunks = chunk_text(text, chunk_chars=1000, max_chunks=3)
    assert len(chunks) == 3
    assert all(len(c) <= 1000 for c in chunks)
    assert chunk_text("short", 1000, 3) == ["short"]


def test_data_hidden_in_attributes_is_surfaced():
    html = """<html><body><main><ol>
      <li><article><p class="star-rating Three"><i class="icon-star"></i><i class="icon-star"></i></p>
        <h3><a href="/b/1" title="A Light in the Attic">A Light in the ...</a></h3>
        <p class="price">£51.77</p></article></li>
      <li><article><div class="stars" aria-label="Rated 4.5 out of 5"></div>
        <h3><a href="/b/2" title="Soumission">Soumission</a></h3>
        <span data-review-count="1204"></span><time datetime="2026-03-01">last spring</time>
        <meta itemprop="ratingValue" content="4.5"><img src="s.png" alt="4.5 out of 5 stars"></article></li>
    </ol></main></body></html>"""
    text = clean_html(html, "http://x.test/").text
    assert "[class: star-rating Three]" in text           # rating stored as a CSS class
    assert "icon-star" not in text                        # ...without the misleading child icons
    assert '"A Light in the Attic")' in text              # full title behind truncated link text (link title)
    assert "title:" not in text                           # ...and not duplicated as a note
    assert "Rated 4.5 out of 5" in text                   # aria-label
    assert "data-review-count: 1204" in text and "datetime: 2026-03-01" in text
    assert "ratingValue: 4.5" in text and "image: 4.5 out of 5 stars" in text
    assert text.count("---") >= 2                         # cards are separated
    assert text.index("star-rating Three") < text.index("A Light in the Attic") < text.index("---", text.index("51.77"))


def test_icon_only_ratings_are_kept():
    html = """<html><body><main><p>Some product with a long enough description to keep.</p>
      <div><i class="fa fa-star"></i><i class="fa fa-star"></i><i class="fa fa-star-o"></i> Widget</div>
      </main></body></html>"""
    text = clean_html(html, "http://x.test/").text
    assert text.count("fa-star]") == 2 and text.count("fa-star-o]") == 1


def test_next_page_detection_variants():
    base = "http://x.test/list/1"
    def nxt(body):
        return clean_html(f"<html><body><main><p>{'text ' * 100}</p>{body}</main></body></html>", base).next_page_url
    # Bootstrap pager (quotes.toscrape.com): text "Next →", class on the parent <li>
    assert nxt('<ul class="pager"><li class="previous"><a href="/list/0">&larr; Previous</a></li>'
               '<li class="next"><a href="/list/2">Next <span aria-hidden="true">&rarr;</span></a></li></ul>') == "http://x.test/list/2"
    assert nxt('<a class="pagination-next" href="/list/2"><svg></svg></a>') == "http://x.test/list/2"
    assert nxt('<a href="/list/2">Next page »</a>') == "http://x.test/list/2"
    assert nxt('<a href="/list/2">›</a>') == "http://x.test/list/2"
    assert nxt('<a href="/list/0">&larr; Previous</a><a href="/news">Next.js news</a>') is None
    assert nxt('<a href="https://other.test/2">Next</a>') is None  # pagination stays on the same site

from scraper_mcp.agents.verifier import dedupe, grounding_score, normalise_source, verify_records
from scraper_mcp.modes import get_mode

SOURCE = "Acme Book 14 | Acme | AED 3,499 | was AED 3,999 | 4.5/5 (1,204 reviews) | In stock"


def test_grounded_record_scores_high():
    rec = {"name": "Acme Book 14", "brand": "Acme", "price": 3499.0, "original_price": 3999.0,
           "review_count": 1204, "currency": "AED", "url": "http://x/p"}
    assert grounding_score(rec, get_mode("ecommerce"), normalise_source(SOURCE)) == 1.0


def test_hallucinated_record_scores_low():
    rec = {"name": "Quantum Hyperbook X", "brand": "Quantum", "price": 99999.0, "review_count": 31337}
    assert grounding_score(rec, get_mode("ecommerce"), normalise_source(SOURCE)) == 0.0


def test_verify_drops_empty_and_tags_provenance():
    kept, stats = verify_records(
        [{"name": "Acme Book 14", "price": 3499.0}, {"name": None, "price": None, "url": "http://x"}],
        get_mode("ecommerce"), source_url="http://x/shop", page_title="Shop", source_text=SOURCE,
        fetch_method="static", run_id="r1")
    assert len(kept) == 1 and stats["empty"] == 1
    assert kept[0]["_source_url"] == "http://x/shop" and kept[0]["_grounding"] == 1.0


def test_dedupe_ignores_metadata():
    a = {"name": "A", "price": 1.0, "_run_id": "1"}
    b = {"name": "A", "price": 1.0, "_run_id": "2"}
    out, dropped = dedupe([a, b, {"name": "B", "price": 2.0}])
    assert dropped == 1 and len(out) == 2


def test_prices_with_trailing_zeros_are_found():
    source = normalise_source("The Past Never Ends £56.50 | Roger Ackroyd £44.10 | Gift card AED 1,299.00")
    for value in (56.5, 44.1, 1299.0):
        assert grounding_score({"price": value}, get_mode("ecommerce"), source) == 1.0
    assert grounding_score({"price": 56.0}, get_mode("ecommerce"), source) == 0.0
    assert grounding_score({"price": 5.5}, get_mode("ecommerce"), source) == 0.0


def test_abbreviated_salaries_are_found():
    source = normalise_source("Highnote · Remote $150k - $200k/yr · $119.8k - $132.0k · budget €1.2M")
    jobs = get_mode("jobs")
    for value in (150000.0, 200000.0, 119800.0, 132000.0, 1200000.0):
        assert grounding_score({"salary_min": value}, jobs, source) == 1.0, value
    for value in (15000.0, 1500000.0, 120000.0):
        assert grounding_score({"salary_min": value}, jobs, source) == 0.0, value


def test_identity_dedupe_for_reposted_jobs():
    a = {"title": "BI Data Analyst", "company": "Openx", "location": "US", "salary_min": 93500.0, "url": "https://b/1"}
    b = {**a, "url": "https://b/2", "title": "BI data analyst "}
    c = {**a, "location": "Canada", "url": "https://b/3"}
    out, dropped = dedupe([a, b, c], get_mode("jobs").dedupe_on)
    assert dropped == 1 and [r["url"] for r in out] == ["https://b/1", "https://b/3"]
    assert dedupe([a, b], None)[1] == 0

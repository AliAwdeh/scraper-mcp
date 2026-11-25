from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .base import Mode


class Product(BaseModel):
    name: str | None = Field(None, description="Product name as shown")
    brand: str | None = Field(None, description="Brand or manufacturer")
    category: str | None = Field(None, description="Product category, if shown")
    price: float | None = Field(None, description="Current selling price, number only")
    original_price: float | None = Field(None, description="Price before discount (strike-through), number only")
    currency: str | None = Field(None, description="ISO 4217 code")
    rating: float | None = Field(None, description="Average rating normalised to a 0-5 scale")
    review_count: int | None = Field(None, description="Number of ratings or reviews")
    in_stock: bool | None = Field(None, description="True if available, False if sold out")
    seller: str | None = Field(None, description="Seller or store, if a marketplace")
    url: str | None = Field(None, description="Absolute URL of the product page")


def _derive_ecommerce(df: pd.DataFrame) -> pd.DataFrame:
    if {"price", "original_price"} <= set(df.columns):
        p = pd.to_numeric(df["price"], errors="coerce")
        o = pd.to_numeric(df["original_price"], errors="coerce")
        df["discount_pct"] = np.where((o > 0) & (p > 0) & (o > p), ((o - p) / o * 100).round(2), np.nan)
    return df


ECOMMERCE = Mode(
    name="ecommerce",
    description="Products and prices from shops and marketplaces: price, discount, rating, stock.",
    record_model=Product,
    guidance="""You are the E-commerce Specialist.
- A listing page holds many products: extract every product card. A product page holds one product.
- price is what the customer pays now. original_price is the crossed-out price, only if shown.
- Ratings on other scales are converted to 0-5 (e.g. 8/10 -> 4.0, 90% -> 4.5).
- review_count is the integer next to the stars ("(1,234)" -> 1234).
- Prefer schema.org Product / Offer data in the structured data when present: it is more reliable.""",
    default_question="Summarise the price distribution, compare prices and ratings across brands, "
                     "check whether rating correlates with price, and flag price outliers.",
    derive=_derive_ecommerce,
    inferred=frozenset({"category", "in_stock", "rating"}),
    keywords=["price", "buy", "shop", "product", "deal", "cheapest", "store", "amazon", "noon"],
)


class JobPosting(BaseModel):
    title: str | None = Field(None, description="Job title")
    company: str | None = Field(None, description="Hiring company")
    location: str | None = Field(None, description="City and/or country")
    remote: Literal["remote", "hybrid", "onsite"] | None = Field(None, description="Work arrangement, if stated")
    employment_type: Literal["full_time", "part_time", "contract", "internship", "temporary"] | None = None
    seniority: Literal["intern", "junior", "mid", "senior", "lead", "executive"] | None = Field(
        None, description="Only if stated or obvious from the title")
    salary_min: float | None = Field(None, description="Lower bound of the advertised pay")
    salary_max: float | None = Field(None, description="Upper bound (same as min for a single figure)")
    salary_currency: str | None = Field(None, description="ISO 4217 code")
    salary_period: Literal["hour", "day", "week", "month", "year"] | None = None
    skills: list[str] = Field(default_factory=list, description="Skills / technologies explicitly required")
    posted_date: date | None = Field(None, description="Date posted, ISO format")
    url: str | None = Field(None, description="Absolute URL of the posting")


_ANNUAL = {"hour": 2080, "day": 260, "week": 52, "month": 12, "year": 1}


def _derive_jobs(df: pd.DataFrame) -> pd.DataFrame:
    if {"salary_min", "salary_max", "salary_period"} <= set(df.columns):
        lo = pd.to_numeric(df["salary_min"], errors="coerce")
        hi = pd.to_numeric(df["salary_max"], errors="coerce").fillna(lo)
        factor = df["salary_period"].map(_ANNUAL)
        df["salary_mid_annual"] = (((lo.fillna(hi) + hi) / 2) * factor).round(0)
    return df


JOBS = Mode(
    name="jobs",
    description="Job postings: title, company, location, salary (annualised), skills, seniority.",
    record_model=JobPosting,
    guidance="""You are the Jobs Specialist.
- Extract every job card on a listing page; on a single posting page, one record.
- Salary: copy the numbers as written, set salary_period from the text ("per hour", "/yr", "monthly").
  If only one number is given, salary_min = salary_max = that number. Never invent a salary.
- skills: concrete skills or tools that are required (Python, SQL, AWS), normalised to their common name.
  Not soft phrases like "team player".
- Prefer schema.org JobPosting data in the structured data when present.""",
    default_question="Summarise annualised salaries (by currency), compare pay by seniority and remote type, "
                     "list the most demanded skills and the top hiring companies and locations.",
    derive=_derive_jobs,
    inferred=frozenset({"seniority", "remote", "employment_type", "salary_period", "skills"}),
    dedupe_on=("title", "company", "location", "salary_min", "salary_max", "salary_period"),
    keywords=["job", "jobs", "hiring", "career", "vacancy", "salary", "engineer", "developer", "position"],
)


class Article(BaseModel):
    headline: str | None = Field(None, description="Headline")
    source: str | None = Field(None, description="Publication name")
    author: str | None = Field(None, description="Author(s)")
    published_date: date | None = Field(None, description="Publication date, ISO format")
    section: str | None = Field(None, description="Section or category (Business, Tech, ...)")
    summary: str | None = Field(None, description="One or two sentence factual summary")
    sentiment: Literal["positive", "neutral", "negative"] | None = Field(
        None, description="Overall tone of the article toward its main subject")
    topics: list[str] = Field(default_factory=list, description="2-5 short topic tags")
    entities: list[str] = Field(default_factory=list, description="Main people / organisations / places named")
    url: str | None = Field(None, description="Absolute URL of the article")


NEWS = Mode(
    name="news",
    description="News and blog articles: headline, date, author, sentiment, topics, entities.",
    record_model=Article,
    guidance="""You are the News Specialist.
- On an article page extract that one article. On a section or search page extract each headline listed.
- summary must be factual and based on the text shown. If only a headline is visible, summary is null.
- sentiment is the tone toward the main subject, judged from the text you can read. If you only see a
  headline, judge it from the headline.
- topics are short lowercase tags (e.g. "interest rates", "ai regulation").""",
    default_question="Show the volume of articles over time, the sentiment mix overall and by source, "
                     "and the most frequent topics and entities.",
    use_article_extractor=True,
    inferred=frozenset({"summary", "sentiment", "topics", "entities", "section"}),
    dedupe_on=("headline", "source"),
    keywords=["news", "article", "headline", "press", "blog", "report", "latest"],
)


class PropertyListing(BaseModel):
    title: str | None = Field(None, description="Listing title")
    listing_type: Literal["rent", "sale"] | None = None
    property_type: str | None = Field(None, description="apartment, villa, studio, townhouse, office...")
    price: float | None = Field(None, description="Asking price or rent, number only")
    currency: str | None = Field(None, description="ISO 4217 code")
    price_period: Literal["year", "month", "week", "day", "total"] | None = Field(
        None, description="'total' for a sale price")
    bedrooms: float | None = Field(None, description="Studio = 0")
    bathrooms: float | None = None
    area_sqm: float | None = Field(None, description="Area in square metres (convert sqft * 0.092903)")
    location: str | None = Field(None, description="Area / neighbourhood and city")
    furnished: bool | None = None
    agency: str | None = None
    url: str | None = Field(None, description="Absolute URL of the listing")


def _derive_real_estate(df: pd.DataFrame) -> pd.DataFrame:
    if {"price", "area_sqm"} <= set(df.columns):
        p = pd.to_numeric(df["price"], errors="coerce")
        a = pd.to_numeric(df["area_sqm"], errors="coerce")
        df["price_per_sqm"] = np.where(a > 0, (p / a).round(2), np.nan)
    return df


REAL_ESTATE = Mode(
    name="real_estate",
    description="Property listings for rent or sale: price, size, bedrooms, location, price per m².",
    record_model=PropertyListing,
    guidance="""You are the Real Estate Specialist.
- Extract every listing card. Convert square feet to square metres (x 0.092903).
- price_period: "AED 85,000 yearly" -> year; "per month" -> month; sale prices -> total.
- bedrooms: "Studio" -> 0. Keep half-baths as decimals if written that way.""",
    default_question="Summarise prices by listing type and period, compare price per square metre by location "
                     "and bedroom count, and flag outliers.",
    derive=_derive_real_estate,
    inferred=frozenset({"listing_type", "price_period", "property_type", "area_sqm", "furnished"}),
    keywords=["rent", "apartment", "villa", "property", "real estate", "house", "flat", "bedroom"],
)


class Review(BaseModel):
    subject: str | None = Field(None, description="What is being reviewed (product, place, service)")
    rating: float | None = Field(None, description="Rating normalised to 0-5")
    title: str | None = Field(None, description="Review title, if any")
    text: str | None = Field(None, description="Review text, up to ~300 characters")
    sentiment: Literal["positive", "neutral", "negative"] | None = None
    aspects: list[str] = Field(default_factory=list, description="Aspects discussed: price, quality, delivery...")
    review_date: date | None = None
    verified: bool | None = Field(None, description="Verified purchase / stay, if indicated")
    url: str | None = None


REVIEWS = Mode(
    name="reviews",
    description="Customer reviews: rating, sentiment, aspects mentioned, date.",
    record_model=Review,
    guidance="""You are the Reviews Specialist.
- One record per review. Truncate long review text to ~300 characters, do not paraphrase it.
- sentiment comes from the review text; rating comes only from the stars/score shown.
- aspects are short lowercase nouns for what the reviewer talks about (price, quality, delivery, staff).""",
    default_question="Show the rating distribution, the sentiment mix, which aspects are mentioned most "
                     "and how ratings differ by aspect, and the trend over time if dates exist.",
    inferred=frozenset({"sentiment", "aspects", "subject", "rating"}),
    dedupe_on=("subject", "title", "text", "rating"),
    keywords=["review", "reviews", "rating", "opinion", "feedback", "complaints"],
)


class Paper(BaseModel):
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = Field(None, description="Journal, conference or archive (arXiv)")
    abstract: str | None = Field(None, description="First ~300 characters of the abstract, if shown")
    citations: int | None = Field(None, description="Citation count, if shown")
    fields_of_study: list[str] = Field(default_factory=list, description="Subject areas / categories")
    url: str | None = None


PAPERS = Mode(
    name="research_papers",
    description="Academic papers (arXiv, journals, proceedings): authors, year, venue, citations, fields.",
    record_model=Paper,
    guidance="""You are the Research Papers Specialist.
- One record per paper. authors as individual names. year as a 4-digit integer.
- fields_of_study from the categories shown (e.g. arXiv cs.LG -> "machine learning").""",
    default_question="Show publications per year, the most frequent fields and venues, the most prolific "
                     "authors, and the citation distribution if citation counts exist.",
    inferred=frozenset({"fields_of_study", "abstract"}),
    dedupe_on=("title",),
    keywords=["paper", "papers", "arxiv", "research", "journal", "publication", "study", "scholar"],
)

BUILTIN_MODES = [ECOMMERCE, JOBS, NEWS, REAL_ESTATE, REVIEWS, PAPERS]

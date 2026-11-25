# scraper-mcp

An MCP server in Docker that scrapes the web with a team of LLM agents and returns statistics about
the data. Any MCP client (Claude, ChatGPT, DeepSeek through an MCP client) can use it.

## How it works

```
question -> Planner -> Searcher -> Source Selector -> Fetcher -> Specialist -> Verifier -> Store -> Analyst -> Reporter
```

- **Fetcher**: plain HTTP first, headless Chromium (Playwright) only when a page needs JavaScript.
  Respects robots.txt, waits between requests and follows pagination.
- **Specialist**: an LLM that extracts records in one of the modes: `ecommerce`, `jobs`, `news`,
  `real_estate`, `reviews`, `research_papers` or `generic`.
- **Verifier**: checks every record against the page text (`_grounding`, 0 to 1) and removes duplicates.
- **Analyst**: the LLM picks the statistics; pandas and scipy compute them. No number comes from the LLM.

The LLM runs through OpenRouter (LangChain + LangGraph). Data is stored in DuckDB in `./data`.

## Setup

1. Create the config and add your OpenRouter key:

   ```bash
   cp .env.example .env
   ```

2. Build the image:

   ```bash
   docker build -t scraper-mcp:latest .
   ```

3. Connect it to Claude Code (run from this folder):

   ```bash
   claude mcp add scraper -- docker run -i --rm --init --env-file "$(pwd)/.env" -v "$(pwd)/data:/data" scraper-mcp:latest
   ```

   For Claude Desktop, use `configs/claude_desktop_config.json` and replace the paths.
   For an HTTP server instead, run `docker compose up -d` and connect to `http://localhost:8000/mcp`.

## Tools

| Tool | What it does |
|---|---|
| `research` | Full pipeline from a question: search, scrape, verify, store, analyse |
| `scrape` | Same pipeline on URLs you give it |
| `analyze` | Answers a question about a dataset with statistics |
| `search_web`, `fetch_page` | Search results, or one page as clean text |
| `scrape_tables` | HTML tables into a dataset (no LLM) |
| `run_stat`, `query_sql`, `chart` | Statistics, read-only SQL, charts |
| `list_datasets`, `describe_dataset`, `export_dataset`, `delete_dataset` | Manage datasets |
| `scrape_stats` | Success rate, browser use, blocked pages and LLM cost |
| `list_modes` | The extraction modes and their fields |

## Example prompts

- "Use scrape_tables on https://www.scrapethissite.com/pages/forms/?per_page=100 and regress wins on goals for and against."
- "Fetch https://quotes.toscrape.com/js/ and tell me how it was fetched."
- "Scrape https://books.toscrape.com/catalogue/category/books/mystery_3/index.html in ecommerce mode, follow 2 pages, and compare prices by rating."
- "Research remote data analyst salaries."

## Tests

```bash
uv run pytest
```

The tests use a local fixture website and a fake LLM, so they need no network or API key.

## Project layout

```
src/scraper_mcp/
  server.py     MCP tools
  scraping/     fetching, robots.txt, browser, HTML cleaning, search
  modes/        extraction schemas
  agents/       LangGraph pipeline and agents
  analysis/     statistics and charts
  storage.py    DuckDB
tests/          pytest suite
```

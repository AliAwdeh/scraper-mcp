from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "anthropic/claude-sonnet-5.5"
    llm_fast_model: str = ""
    llm_max_tokens: int = 8000
    llm_timeout: float = 120.0
    llm_max_retries: int = 2
    llm_structured_method: Literal["auto", "function_calling", "json_schema"] = "auto"
    openrouter_app_name: str = "scraper-mcp"
    openrouter_app_url: str = ""

    mcp_transport: Literal["stdio", "http"] = "stdio"
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8000
    mcp_path: str = "/mcp"
    mcp_auth_token: str = ""
    mcp_allowed_hosts: str = ""
    mcp_allowed_origins: str = ""

    data_dir: Path = Path("data")

    user_agent: str = "scraper-mcp/0.1 (educational web-scraping project)"
    scraper_contact: str = ""
    respect_robots: bool = True
    request_delay: float = 1.0
    request_timeout: float = 20.0
    max_concurrency: int = 4
    max_bytes: int = 5_000_000
    render_mode: Literal["auto", "static", "browser"] = "auto"
    browser_timeout: float = 30.0
    allow_private_networks: bool = False

    max_page_chars: int = 60_000
    chunk_chars: int = 20_000
    max_chunks_per_page: int = 3

    max_sources_cap: int = 10
    max_pages_per_source_cap: int = 5
    max_total_pages: int = 25

    @property
    def db_path(self) -> Path:
        return self.data_dir / "scraper.duckdb"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def effective_user_agent(self) -> str:
        contact = self.scraper_contact.strip()
        if not contact:
            return self.user_agent
        if self.user_agent.endswith(")"):
            return f"{self.user_agent[:-1]}; +{contact})"
        return f"{self.user_agent} (+{contact})"

    @property
    def fast_model(self) -> str:
        return self.llm_fast_model or self.llm_model

    @property
    def llm_configured(self) -> bool:
        return bool(self.openrouter_api_key.strip())

    @staticmethod
    def split_csv(value: str) -> list[str]:
        return [v.strip() for v in value.split(",") if v.strip()]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.exports_dir.mkdir(parents=True, exist_ok=True)
    return s

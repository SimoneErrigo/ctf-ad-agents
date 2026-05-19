from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    janus_url: str = Field("http://janus:8080", description="Base URL of the Janus REST API.")
    janus_password: str = Field(..., description="TEAM_PASSWORD from Janus' .env.")
    janus_display_name: str = Field("mcp-agent", description="Login display name.")
    janus_timeout: float = Field(20.0, description="HTTP request timeout (seconds).")

    mcp_host: str = Field("0.0.0.0", description="Bind address for MCP transport.")
    mcp_port: int = Field(8765, description="Port for MCP transport.")
    mcp_path: str = Field("/mcp", description="HTTP path the MCP server is served on.")

    janus_max_limit: int = Field(1000, description="Hard cap on list_packets `limit`.")
    janus_summary_max_limit: int = Field(
        50,
        description="Hard cap on summary-mode list_packets rows returned to agents.",
    )
    janus_body_max_bytes: int = Field(8192, description="Max body bytes per packet (0 = unlimited).")
    janus_max_flow_packets: int = Field(20, description="Max packets in a flow.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

# Centralized configuration loaded from environment variables

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider"""

    name: str
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    priority: int = 0


class Settings(BaseModel):
    """Application-wide settings parsed from environment"""

    # LLM providers
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:3b"))

    # Team shared 
    team_shared_provider: str = Field(default_factory=lambda: os.getenv("TEAM_SHARED_PROVIDER", "anthropic"))
    team_shared_api_key: str = Field(default_factory=lambda: os.getenv("TEAM_SHARED_API_KEY", ""))
    team_shared_rate_limit: int = Field(default_factory=lambda: int(os.getenv("TEAM_SHARED_RATE_LIMIT", "30")))

    # Agent endpoints
    orchestrator_host: str = Field(default_factory=lambda: os.getenv("ORCHESTRATOR_HOST", "0.0.0.0"))
    orchestrator_port: int = Field(default_factory=lambda: int(os.getenv("ORCHESTRATOR_PORT", "8000")))
    vuln_agent_host: str = Field(default_factory=lambda: os.getenv("VULN_AGENT_HOST", "0.0.0.0"))
    vuln_agent_port: int = Field(default_factory=lambda: int(os.getenv("VULN_AGENT_PORT", "8001")))

    # CTF environment
    vm_ip: str = Field(default_factory=lambda: os.getenv("VM_IP", "10.60.1.1"))
    flag_regex: str = Field(default_factory=lambda: os.getenv("FLAG_REGEX", r"[A-Z0-9]{31}="))
    tick_duration: int = Field(default_factory=lambda: int(os.getenv("TICK_DURATION", "120")))

    # Vulnbox SSH
    vulnbox_ssh_host: str = Field(default_factory=lambda: os.getenv("VULNBOX_SSH_HOST", "10.60.1.1"))
    vulnbox_ssh_port: int = Field(default_factory=lambda: int(os.getenv("VULNBOX_SSH_PORT", "22")))
    vulnbox_ssh_user: str = Field(default_factory=lambda: os.getenv("VULNBOX_SSH_USER", "root"))
    vulnbox_ssh_key_path: str = Field(default_factory=lambda: os.getenv("VULNBOX_SSH_KEY_PATH", "~/.ssh/id_rsa"))

    # Redis
    redis_url: str = Field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    def get_available_providers(self) -> list[LLMProviderConfig]:
        """Return LLM providers in priority order, only those with keys set"""

        providers: list[LLMProviderConfig] = []

        if self.anthropic_api_key:
            providers.append(
                LLMProviderConfig(
                    name="anthropic",
                    api_key=self.anthropic_api_key,
                    model=self.anthropic_model,
                    priority=1,
                )
            )
        if self.openai_api_key:
            providers.append(
                LLMProviderConfig(
                    name="openai",
                    api_key=self.openai_api_key,
                    model=self.openai_model,
                    priority=2,
                )
            )

        # Assuming that Ollama is always available as fallback (no key needed)
        providers.append(
            LLMProviderConfig(
                name="ollama",
                base_url=self.ollama_base_url,
                model=self.ollama_model,
                priority=99,
            )
        )
        return sorted(providers, key=lambda p: p.priority)


@lru_cache
def get_settings() -> Settings:
    return Settings()
from __future__ import annotations

import shlex
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Workspace
    patcher_workspace_root: Path = Field(
        Path("/var/lib/patcher/workspace"),
        description=(
            "Root directory containing one git working tree per service. "
            "Each service lives at <root>/<service>/ with its own .git/."
        ),
    )
    patcher_service_aliases: dict[str, str] = Field(
        default_factory=lambda: {
            "CC-Forms-backend": "cc-forms",
            "CCForms": "cc-forms",
            "web1.1": "cc-forms",
        },
        description=(
            "Map externally visible service names/ids to patcher workspace names. "
            "Example env JSON: {'web1.1':'cc-forms'}."
        ),
    )

    # Competition VM (git push target)
    vm_ip: str = Field(
        ...,
        description="Hostname or IP of the competition VM that receives `git push`.",
    )
    vm_ssh_user: str = Field("root", description="SSH user on the competition VM.")
    vm_ssh_port: int = Field(22, ge=1, le=65535, description="SSH port on the competition VM.")
    vm_ssh_key_path: Path = Field(
        Path("/root/.ssh/id_ed25519"),
        description="Path to the private SSH key used to push to the VM.",
    )
    vm_git_remote_root: str = Field(
        "/srv/git",
        description=(
            "Absolute path on the VM holding the bare repos. The remote URL "
            "for service <s> is built as ssh://<user>@<ip>:<port><remote_root>/<service>.git"
        ),
    )

    # Git author / branch
    patcher_default_branch: str = Field("main", description="Default branch to push.")
    patcher_git_user_name: str = Field("patch-agent", description="git user.name for commits.")
    patcher_git_user_email: str = Field(
        "patch-agent@ctf-ad-agents.local",
        description="git user.email for commits.",
    )

    # Safety caps
    patcher_max_file_bytes: int = Field(
        262_144,
        ge=1024,
        description="Cap on read_file output sent back to the agent (256 KiB default).",
    )
    patcher_max_diff_bytes: int = Field(
        524_288,
        ge=1024,
        description="Cap on get_diff output sent back to the agent (512 KiB default).",
    )
    patcher_max_list_entries: int = Field(
        500,
        ge=10,
        description="Cap on list_files entries returned per call.",
    )

    # MCP transport
    patcher_mcp_host: str = Field("0.0.0.0", description="Bind address for MCP transport.")
    patcher_mcp_port: int = Field(8766, description="Port for MCP transport.")
    patcher_mcp_path: str = Field("/mcp", description="HTTP path the MCP server is served on.")

    # Misc
    patcher_git_ssh_strict_host_check: bool = Field(
        False,
        description=(
            "If false (default), GIT_SSH_COMMAND passes "
            "-o StrictHostKeyChecking=no — convenient for first-time deploys. "
            "Set to true in production after the host key is in known_hosts."
        ),
    )
    patcher_deploy_worktrees: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional map of canonical service name to deployed VM worktree path. "
            "When set, deploy verifies that this worktree moved to the pushed commit."
        ),
    )

    def remote_url(self, service: str) -> str:
        return (
            f"ssh://{self.vm_ssh_user}@{self.vm_ip}:{self.vm_ssh_port}"
            f"{self.vm_git_remote_root.rstrip('/')}/{service}.git"
        )

    def deploy_worktree(self, service: str) -> str | None:
        return self.patcher_deploy_worktrees.get(service)

    def ssh_base_command(self) -> list[str]:
        cmd = ["ssh", "-i", str(self.vm_ssh_key_path)]
        if not self.patcher_git_ssh_strict_host_check:
            cmd.extend([
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
            ])
        cmd.extend(["-p", str(self.vm_ssh_port), f"{self.vm_ssh_user}@{self.vm_ip}"])
        return cmd

    def git_ssh_command(self) -> str:
        return " ".join(shlex.quote(arg) for arg in self.ssh_base_command()[:-1])


# Cache the settings instance so it's only created once per process. 
# we don't want to do it on every tool call.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

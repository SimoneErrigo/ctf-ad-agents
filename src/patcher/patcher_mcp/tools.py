"""Patcher MCP tools.
HITL is not enforced here
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from patcher_mcp import git_functions
from patcher_mcp.git_functions import PatcherError

log = logging.getLogger(__name__)


def _as_error(e: Exception) -> dict[str, Any]:
    return {"ok": False, "error": str(e)}


def register_tools(mcp: FastMCP) -> None:
    """Register every patcher tool on `mcp`. All tools are tagged 'patch'."""

    # Discovery

    @mcp.tool(
        tags={"patch", "read"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )
    async def list_workspace_services() -> list[str]:
        """List services currently materialized in the patcher workspace.

        If a service you need is missing, call `ensure_service_repo(service)`
        to clone/init it from the configured remote.
        """
        return git_functions.list_workspace_services()

    @mcp.tool(
        tags={"patch", "read"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )
    async def list_vm_services() -> dict[str, Any]:
        """List the services on the competition VM, read live over SSH.

        Returns `running_containers` (the Docker containers actually running on
        the VM: name, image, status, ports) and `service_dirs` (the service
        folders present on the VM). Unlike Janus' list_services, this shows services
        that run on the VM (or merely exist on disk) but are NOT behind Janus. Pair the two for a
        full inventory. Returns empty lists if the VM/Docker is unreachable.
        """
        return await git_functions.list_vm_services()

    @mcp.tool(
        tags={"patch", "read"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )
    async def resolve_service(
        service: Annotated[
            str,
            Field(min_length=1, description="Service name/id to canonicalize, e.g. web1.1 or CC-Forms-backend."),
        ],
    ) -> dict[str, Any]:
        """Resolve a Janus/service display name to the patcher workspace service name.

        Also reports `available_services` (materialized workspace + repos seeded
        on the VM), so a wrong name shows the valid options instead of dead-ending.
        """
        try:
            return await git_functions.resolve_service(service)
        except PatcherError as e:
            return _as_error(e)

    @mcp.tool(
        tags={"patch", "read"},
        annotations={"destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def ensure_service_repo(
        service: Annotated[
            str,
            Field(min_length=1, description="Service name (also the workspace directory name)."),
        ],
    ) -> dict[str, Any]:
        """Make sure the service repo exists locally, cloning from the VM if needed."""
        try:
            return await git_functions.ensure_service_repo(service)
        except PatcherError as e:
            return _as_error(e)

    # Read

    @mcp.tool(
        tags={"patch", "read"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )
    async def list_files(
        service: Annotated[str, Field(min_length=1, description="Service name.")],
        path: Annotated[
            str,
            Field(description="Relative directory path inside the service repo. Default '.'."),
        ] = ".",
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Directory listing scoped to the service repo (.git/ is hidden)."""
        try:
            return await git_functions.list_files(service, path)
        except PatcherError as e:
            return _as_error(e)

    @mcp.tool(
        tags={"patch", "read"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )
    async def read_file(
        service: Annotated[str, Field(min_length=1, description="Service name.")],
        path: Annotated[str, Field(min_length=1, description="Relative file path inside the service repo.")],
    ) -> dict[str, Any]:
        """Return file contents (truncated to PATCHER_MAX_FILE_BYTES)."""
        try:
            content = await git_functions.read_file(service, path)
            return {"service": service, "path": path, "content": content}
        except PatcherError as e:
            return _as_error(e)

    @mcp.tool(
        tags={"patch", "read"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )
    async def git_status(
        service: Annotated[str, Field(min_length=1, description="Service name.")],
    ) -> dict[str, Any]:
        """`git status --short --branch` inside the service repo."""
        try:
            return {"status": await git_functions.git_status(service)}
        except PatcherError as e:
            return _as_error(e)

    @mcp.tool(
        tags={"patch", "read"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )
    async def git_log(
        service: Annotated[str, Field(min_length=1, description="Service name.")],
        n: Annotated[int, Field(ge=1, le=100, description="How many recent commits to show.")] = 10,
    ) -> dict[str, Any]:
        """`git log -nN` with iso-strict dates."""
        try:
            return {"log": await git_functions.git_log(service, n)}
        except PatcherError as e:
            return _as_error(e)

    @mcp.tool(
        tags={"patch", "read"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )
    async def get_diff(
        service: Annotated[str, Field(min_length=1, description="Service name.")],
        ref: Annotated[
            str | None,
            Field(description="Optional base ref. Default: uncommitted diff (None). Use 'origin/main' to see what will be pushed."),
        ] = None,
    ) -> dict[str, Any]:
        """Return a unified diff for the service repo."""
        try:
            return {"service": service, "ref": ref, "diff": await git_functions.get_diff(service, ref)}
        except PatcherError as e:
            return _as_error(e)

    # Write

    @mcp.tool(
        tags={"patch"},
        annotations={"destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
    )
    async def write_files(
        service: Annotated[str, Field(min_length=1, description="Service name.")],
        files: Annotated[
            list[dict[str, Any]],
            Field(
                min_length=1,
                description=(
                    "List of files to write, each `{path: <rel>, content: <text>}`. "
                    "Existing files are overwritten; missing directories are created. "
                    "Pass MINIMAL files — only the ones you intend to change."
                ),
            ),
        ],
        message: Annotated[
            str,
            Field(min_length=1, description="Commit message (single short sentence preferred)."),
        ],
        branch: Annotated[
            str | None,
            Field(description="Branch to commit on. Defaults to PATCHER_DEFAULT_BRANCH."),
        ] = None,
    ) -> dict[str, Any]:
        """Stage and commit a set of files in the service repo. Does NOT push.

        After this call, inspect the returned diff and only then call `deploy`
        (which is HITL-gated on the agent side).
        """
        try:
            return await git_functions.write_files(service, files, message, branch)
        except PatcherError as e:
            return _as_error(e)

    @mcp.tool(
        tags={"patch"},
        annotations={"destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
    )
    async def replace_text(
        service: Annotated[str, Field(min_length=1, description="Service name or alias.")],
        path: Annotated[str, Field(min_length=1, description="Relative file path inside the service repo.")],
        old: Annotated[str, Field(min_length=1, description="Exact old text to replace.")],
        new: Annotated[str, Field(description="Replacement text.")],
        message: Annotated[str, Field(min_length=1, description="Commit message.")],
        branch: Annotated[
            str | None,
            Field(description="Branch to commit on. Defaults to PATCHER_DEFAULT_BRANCH."),
        ] = None,
        expected_replacements: Annotated[
            int,
            Field(ge=1, description="Fail unless exactly this many matches are replaced."),
        ] = 1,
    ) -> dict[str, Any]:
        """Replace an exact snippet, stage, and commit. Prefer this for small patches."""
        try:
            return await git_functions.replace_text(
                service,
                path,
                old,
                new,
                message,
                branch,
                expected_replacements,
            )
        except PatcherError as e:
            return _as_error(e)

    @mcp.tool(
        tags={"patch"},
        annotations={"destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
    )
    async def apply_patch(
        service: Annotated[str, Field(min_length=1, description="Service name or alias.")],
        patch: Annotated[str, Field(min_length=1, description="Unified diff accepted by git apply.")],
        message: Annotated[str, Field(min_length=1, description="Commit message.")],
        branch: Annotated[
            str | None,
            Field(description="Branch to commit on. Defaults to PATCHER_DEFAULT_BRANCH."),
        ] = None,
    ) -> dict[str, Any]:
        """Apply a compact unified diff, stage, and commit."""
        try:
            return await git_functions.apply_patch(service, patch, message, branch)
        except PatcherError as e:
            return _as_error(e)

    @mcp.tool(
        tags={"patch"},
        annotations={"destructiveHint": True, "idempotentHint": True, "openWorldHint": False},
    )
    async def discard_changes(
        service: Annotated[str, Field(min_length=1, description="Service name.")],
        branch: Annotated[
            str | None,
            Field(description="Branch to reset to (origin/<branch>). Defaults to PATCHER_DEFAULT_BRANCH."),
        ] = None,
    ) -> dict[str, Any]:
        """`git reset --hard origin/<branch>` + `git clean -fd`. Use when a draft patch is wrong."""
        try:
            return await git_functions.discard_changes(service, branch)
        except PatcherError as e:
            return _as_error(e)

    # Deploy / rollback (HITL-gated on the agent side)

    @mcp.tool(
        tags={"patch"},
        annotations={"destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
    )
    async def deploy(
        service: Annotated[str, Field(min_length=1, description="Service name.")],
        branch: Annotated[
            str | None,
            Field(description="Branch to push. Defaults to PATCHER_DEFAULT_BRANCH."),
        ] = None,
        message: Annotated[
            str | None,
            Field(description="Optional changelog note for the operator approval prompt."),
        ] = None,
    ) -> dict[str, Any]:
        """Push the current branch to origin. The VM post-receive hook rebuilds the service.

        This tool MUST be HITL-gated upstream — never call without operator approval.
        """
        try:
            return await git_functions.deploy(service, branch)
        except PatcherError as e:
            return _as_error(e)

    @mcp.tool(
        tags={"patch"},
        annotations={"destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
    )
    async def rollback(
        service: Annotated[str, Field(min_length=1, description="Service name.")],
        commit_sha: Annotated[
            str,
            Field(min_length=4, max_length=64, description="Commit to revert (short or full SHA)."),
        ],
        branch: Annotated[
            str | None,
            Field(description="Branch to push the revert on. Defaults to PATCHER_DEFAULT_BRANCH."),
        ] = None,
    ) -> dict[str, Any]:
        """Create a revert commit for `commit_sha` and push it. HITL-gated upstream."""
        try:
            return await git_functions.rollback(service, commit_sha, branch)
        except PatcherError as e:
            return _as_error(e)

    @mcp.tool(tags={"patch"}, annotations={"readOnlyHint": True, "openWorldHint": True})
    async def list_commits(
        service: Annotated[str, Field(min_length=1, description="Service name.")],
        branch: Annotated[
            str | None,
            Field(description="Branch to list. Defaults to PATCHER_DEFAULT_BRANCH."),
        ] = None,
        n: Annotated[int, Field(ge=1, le=200, description="Max commits, newest first.")] = 50,
    ) -> dict[str, Any]:
        """Structured commit history for a service (newest first).

        Each entry has sha/short_sha/date/author/subject and an `is_seed` flag on
        the first unpatched commit. Use this to disambiguate a vague rollback
        request before choosing a target.
        """
        try:
            return await git_functions.list_commits(service, branch, n)
        except PatcherError as e:
            return _as_error(e)

    @mcp.tool(
        tags={"patch"},
        annotations={"destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
    )
    async def rollback_to(
        service: Annotated[str, Field(min_length=1, description="Service name.")],
        target_commit: Annotated[
            str | None,
            Field(
                description=(
                    "Commit to restore the service to (short or full SHA). "
                    "Omit to restore the first (seed) commit, dropping all patches."
                )
            ),
        ] = None,
        branch: Annotated[
            str | None,
            Field(description="Branch to push on. Defaults to PATCHER_DEFAULT_BRANCH."),
        ] = None,
    ) -> dict[str, Any]:
        """Restore a service to `target_commit` (or the seed commit) and push it.

        Forward-only (creates a new commit matching the target tree, scoped to the
        service subpath). HITL-gated upstream.
        """
        try:
            return await git_functions.rollback_to(service, target_commit, branch)
        except PatcherError as e:
            return _as_error(e)

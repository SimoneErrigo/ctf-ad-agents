from __future__ import annotations

import contextlib
import logging
from typing import AsyncIterator

import uvicorn
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from patcher_mcp.config import get_settings
from patcher_mcp.tools import register_tools

INSTRUCTIONS = (
    "Read service source code, propose patches as git commits in a local "
    "workspace, and deploy them to the competition VM via SSH-backed git push. "
    "Workflow: resolve_service if needed -> ensure_service_repo -> list_files / "
    "read_file to understand the vulnerability -> replace_text or apply_patch "
    "for a compact minimal patch -> get_diff to review -> deploy (HUMAN "
    "APPROVAL REQUIRED). Use write_files only for whole-file rewrites. Use "
    "rollback(commit_sha) to undo a bad deploy; discard_changes drops "
    "uncommitted work."
)

INSTRUCTIONS_READ = (
    "Read-only view of service source code: resolve_service, ensure_service_repo "
    "(clones locally), list_files, read_file, git_log, get_diff. Use it to "
    "understand a vulnerable service. You cannot write, commit, or deploy."
)

# View name -> (instructions, tag set, mount prefix). The full patch view lives
# at /mcp; the read-only view (consumed by the exploit agent) at /read/mcp.
_VIEWS = {
    "patch": (INSTRUCTIONS, {"patch"}, ""),
    "read": (INSTRUCTIONS_READ, {"read"}, "/read"),
}


def _build_view(instructions: str, tags: set[str]) -> FastMCP:
    view = FastMCP(name="patcher", instructions=instructions)
    register_tools(view)
    view.enable(tags=tags, only=True)
    return view


def build_app() -> Starlette:
    settings = get_settings()

    routes: list = []
    sub_apps: list[Starlette] = []
    for instructions, tags, prefix in _VIEWS.values():
        view = _build_view(instructions, tags)
        sub_app = view.http_app(path=settings.patcher_mcp_path)
        sub_apps.append(sub_app)
        if prefix:
            routes.append(Mount(prefix, app=sub_app))

    async def index(_request):
        return JSONResponse({
            "service": "patcher-mcp",
            "endpoints": {"patch": settings.patcher_mcp_path, "read": "/read" + settings.patcher_mcp_path},
            "workspace_root": str(settings.patcher_workspace_root),
            "vm": {
                "host": settings.vm_ip,
                "user": settings.vm_ssh_user,
                "port": settings.vm_ssh_port,
                "remote_root": settings.vm_git_remote_root,
            },
        })

    routes.append(Route("/_status", endpoint=index))
    # The full "patch" view is mounted last at "/" so it does not shadow /read.
    routes.append(Mount("/", app=sub_apps[0]))

    @contextlib.asynccontextmanager
    async def combined_lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with contextlib.AsyncExitStack() as stack:
            for sub in sub_apps:
                await stack.enter_async_context(sub.router.lifespan_context(sub))
            yield

    app = Starlette(routes=routes, lifespan=combined_lifespan)
    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    app = build_app()
    logging.info(
        "Patcher MCP listening on http://%s:%s%s (workspace=%s, vm=%s@%s:%s)",
        settings.patcher_mcp_host,
        settings.patcher_mcp_port,
        settings.patcher_mcp_path,
        settings.patcher_workspace_root,
        settings.vm_ssh_user,
        settings.vm_ip,
        settings.vm_ssh_port,
    )
    uvicorn.run(app, host=settings.patcher_mcp_host, port=settings.patcher_mcp_port)


if __name__ == "__main__":
    main()

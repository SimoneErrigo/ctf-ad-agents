from __future__ import annotations

import logging

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


def build_app() -> Starlette:
    mcp = FastMCP(name="patcher", instructions=INSTRUCTIONS)
    register_tools(mcp)

    settings = get_settings()
    sub_app = mcp.http_app(path=settings.patcher_mcp_path)

    async def index(_request):
        return JSONResponse({
            "service": "patcher-mcp",
            "endpoint": settings.patcher_mcp_path,
            "workspace_root": str(settings.patcher_workspace_root),
            "vm": {
                "host": settings.vm_ip,
                "user": settings.vm_ssh_user,
                "port": settings.vm_ssh_port,
                "remote_root": settings.vm_git_remote_root,
            },
        })

    app = Starlette(
        routes=[
            Route("/_status", endpoint=index),
            Mount("/", app=sub_app),  # /mcp comes from sub_app's own routes
        ],
        lifespan=sub_app.router.lifespan_context,
    )
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

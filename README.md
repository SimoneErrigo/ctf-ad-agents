# ctf-ad-agents

A multi-agent AI assistant for Attack & Defense CTF competitions, built on
LangGraph. An operator chats with a supervisor agent; security goals are routed
to specialist agents that analyze captured traffic, patch vulnerable services,
and write/run exploits with every dangerous action gated behind human
approval.

## Architecture

One LangGraph graph (`src/graph.py:make_graph`, served by the LangGraph Agent
Server):

![alt text](diagrams/LangGraphAgentServer-white-2026-06-10-083300.png "LangGraph Agent Server")

- **supervisor** (`src/agents/conversational.py`, Haiku) -> operator-facing front
  door. Replies directly to small talk; hands any security goal to the pipeline
  via its single `dispatch_security_task` tool, which jumps to `classify` with
  `Command(goto=..., graph=PARENT)`.
- **classify** (`src/agents/orchestrator.py`, Haiku, structured output) -> splits
  the request into one scoped sub-question per specialist; an "all services"
  task is expanded into one task per live service (read from Janus).
- **specialists** (Sonnet, ReAct via `create_agent`, prompt caching) -> run in
  parallel via the `Send` API, each with a least-privilege MCP toolset:
  - **traffic** (`traffic_agent.py`) -> analyzes HTTP traffic captured by Janus
    (reverse proxy), writes alert/drop rules. Tools: janus-mcp `defender` view.
  - **patch** (`patch_agent.py`) -> minimal source fixes, deployed to the
    competition VM via git push (post-receive hook rebuilds). Tools: patcher-mcp.
  - **exploit** (`exploit_agent.py`) -> writes/tests/runs exploits through
    exploitfarm (xfarm). Tools: exploiter-mcp + janus-mcp `exploit` (read-only
    packets) + patcher-mcp `read` (read-only source).
- **final** (`orchestrator.py`) -> fan-in; synthesizes the specialists' reports
  into one operator-facing answer.

A current Mermaid diagram lives in `diagrams/graph_overview.mmd`.

### Human-in-the-loop

`HumanInTheLoopMiddleware` (`src/tools/hitl.py`) interrupts on the critical
tools only: Janus rule create/update/delete, patch deploy/rollback, exploit
push/start. Approval cards (with the diff / exploit source) render in the Agent
Chat UI; everything else runs autonomously.

### MCP servers

Each capability is a separate FastMCP server; agents consume tag-filtered
HTTP endpoints so each agent only ever sees the tools its role allows:

| Server           | Path                                 | Endpoints                                                   |
| ---------------- | ------------------------------------ | ----------------------------------------------------------- |
| `src/janus/`     | Janus traffic + rules                | `/traffic/mcp`, `/defender/mcp`, `/exploit/mcp` (port 8765) |
| `src/patcher/`   | git workspace + SSH deploy to the VM | `/mcp`, read-only `/read/mcp` (8766)                        |
| `src/exploiter/` | xfarm authoring + attack lifecycle   | `/mcp` (8767)                                               |

### Persistence

The graph compiles with `checkpointer=None`: the Agent Server injects
persistence (in-memory for `langgraph dev`, Postgres/Redis in the Helm
deployment), which is also what pauses/resumes HITL interrupts.

## Running locally

```bash
cp .env_example .env        # AWS Bedrock creds + model ids + MCP base URLs
# start the three MCP servers (each has its own .env, README and compose file):
#   src/janus, src/patcher, src/exploiter
uv sync --group dev
uv run langgraph dev        # Agent Server on :2024
# UI: ../agent-chat-ui (pnpm dev), point it at http://localhost:2024 / graph "conversational"
```

Tests (pure functions only, no network):

```bash
uv run --group dev pytest -q
```

## Deploying to Kubernetes

`deploy-k8s/` contains kustomize bases (namespace + config, Postgres/Redis,
the three MCP servers) and a values file for the official `langgraph-cloud`
Helm chart (self-hosted Lite). The step-by-step commands used on Minikube are
in `deploy-k8s/notes.txt`. The agent image is built with
`uv run langgraph build`.

## Safety

This project is for authorized CTF competitions only. Outward-facing actions
(starting exploits, deploying patches, changing live proxy rules) always
require explicit operator approval through the HITL gates.

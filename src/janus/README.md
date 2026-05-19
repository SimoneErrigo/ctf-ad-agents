# janus-mcp

MCP server that wraps the [Janus](https://github.com/SimoneErrigo/Janus.git) REST
API and exposes it to the agents in `ctf-ad-agents`. Streamable-HTTP transport,
dockerized, joins the Janus Docker network so the URL is just `http://janus:8080`.

## Tools

| Tool                 | What it does                                                                        |
| -------------------- | ----------------------------------------------------------------------------------- |
| `list_services`      | List proxied services (id, name, ports, protocol).                                  |
| `list_packets`       | Query captured packets using the Janus filter DSL (`q`).                            |
| `get_packet`         | Full headers + body for a single packet (body truncated to `JANUS_BODY_MAX_BYTES`). |
| `get_flow`           | Reconstruct the full request/response flow for a packet.                            |
| `validate_filter`    | Validate a filter DSL expression before using it.                                   |
| `get_capture_status` | Capture mode (live/static), whether capture is running, current window.             |
| `get_filter_dsl`     | Short DSL cheatsheet (full reference: resource `docs://janus/filters`).             |

The filter DSL is documented in `Janus/FILTERS.md`
Example: `service == "web1" AND method == "POST" AND status >= 400`.

## Quick start

1. **Make sure Janus is up.** From the Janus folder: `docker compose up -d`.
   That creates the Docker network `janus_default` and runs Janus on
   `http://janus:8080` inside the network.

2. **Configure.** From this folder:

   ```bash
   cp .env.example .env
   # Edit JANUS_PASSWORD to match TEAM_PASSWORD in Janus' .env
   ```

3. **Run.**

   ```bash
   docker compose up -d --build
   ```

   The MCP is now reachable at:
   - `http://janus-mcp:8765/mcp` from any container on `janus_default`
   - `http://127.0.0.1:8765/mcp` from the host (local development)

## Configuration

All settings come from environment variables (see `.env.example`):

| Variable                | Default             | Notes                                                        |
| ----------------------- | ------------------- | ------------------------------------------------------------ |
| `JANUS_URL`             | `http://janus:8080` | Container-network URL. Override for host-network setups.     |
| `JANUS_PASSWORD`        | —                   | **Required.** Equal to Janus' `TEAM_PASSWORD`.               |
| `JANUS_DISPLAY_NAME`    | `mcp-agent`         | Login name shown in the Janus sidebar.                       |
| `MCP_HOST` / `MCP_PORT` | `0.0.0.0` / `8765`  | Bind for the streamable-HTTP transport.                      |
| `MCP_PATH`              | `/mcp`              | HTTP path the MCP is served on.                              |
| `JANUS_MAX_LIMIT`       | `1000`              | Hard cap on `list_packets` `limit` (protects context).       |
| `JANUS_SUMMARY_MAX_LIMIT` | `50`              | Hard cap for summary-mode `list_packets` rows sent to agents. |
| `JANUS_BODY_MAX_BYTES`  | `8192`              | Truncate per-packet body; `0` = unlimited.                   |
| `JANUS_TIMEOUT`         | `20`                | HTTP request timeout (seconds).                              |
| `JANUS_NETWORK`         | `janus_default`     | External Docker network to attach to (compose-level).        |

## Auth model

- **MCP ↔ agents:** no auth. The MCP is bound to the private Docker network
  shared with Janus and (later) the agents. The localhost publish is for dev.
- **MCP → Janus:** the client logs in with `JANUS_PASSWORD`, caches the
  bearer token, and refreshes it once on any 401 (Janus issues 24h tokens).

# patcher-mcp

MCP server consumed by the **patch agent**. Exposes a surface to read
service source code, apply patches as git commits in a local workspace, and
deploy them to the competition VM via SSH `git push` to a bare repo
that has a `post-receive` hook rebuilding automatically the container.

## Competition VM setup (one-time)

Use `scripts/init-vm.sh`, run it **on the VM** (or on your host for local
testing). For each service it creates the bare repo, installs the
`post-receive` hook, seeds the repo from the service source, and pushes the
first commit (which performs the initial deploy). POSIX sh; macOS + Linux.

```bash
PATCHER_SERVICE_ALIASES={"web1.1":"cc-forms","CC-Forms-backend":"cc-forms","CCForms":"cc-forms"}
```

```bash
# Real VM: copy sources into /srv/services, bare repos in /srv/git.
scripts/init-vm.sh --remote-root /srv/git --worktree-root /srv/services \
  --exclude db/data \
  --authorized-key ~/patch-agent.pub \
  cc-forms=/home/ctf/src/cc-forms rce=/home/ctf/src/service2

# Local testing: deploy each service in place (source == deploy target).
scripts/init-vm.sh --remote-root ~/AD-demo/patcher-test/git \
  --exclude db/data --exclude api/forms \
  cc-forms=~/AD-demo/patcher-test/cc-forms
```

The service name must match the patcher's canonical repo name (see
`PATCHER_SERVICE_ALIASES`) and the deploy path must match
`PATCHER_DEPLOY_WORKTREES` so `deploy()` can verify it. `--exclude` keeps
runtime/persistent dirs (DB data, uploads) out of git and untouched across
deploys. `--authorized-key` installs the patch agent's public key (counterpart
of the private key mounted in the container) into
`~${VM_SSH_USER}/.ssh/authorized_keys`. Re-run with `--force` to refresh the
hook and re-seed.

### Hook (why the generated hook prepends PATH)

A `post-receive` hook runs in a **non-interactive** shell whose `PATH` is
minimal and usually excludes Docker (`/usr/local/bin` on macOS, `/snap/bin` on
Linux). If `docker` isn't found the hook dies, but **the push still succeeds**
(a post-receive hook cannot reject a push, refs are already updated), so the
patcher reports a deploy that never rebuilt the container. The generated hook
therefore prepends the usual Docker locations to `PATH` and prints a
`DEPLOY-FAILED` on any failure, which
`patcher_mcp/git_functions._remote_output_has_hook_error` detects so `deploy()`
raises instead of silently succeeding.

### macOS / Docker Desktop testing note

On the competition Linux VM the hook just works. On macOS (Docker Desktop) the
deploy build runs over a **non-interactive SSH session**, where the login
keychain can't be unlocked. Docker Desktop's builder resolves base-image
credentials through that keychain _during `load metadata`_, so a build that has
to pull a base image fails with:

```
error getting credentials … keychain cannot be accessed because the current
session does not allow user interaction
```

Neither `DOCKER_CONFIG` nor removing `credsStore` from `~/.docker/config.json`
reliably bypasses this for the build path. What does work: **have the base
images cached locally**, a build that finds `node:22`, `postgres:15`, etc.
already present makes no registry call and never touches the keychain.

`init-vm.sh` handles this automatically: it runs the first
`docker compose up -d --build` via a _local_ push (interactive session, keychain
unlocked), which pulls and caches the base images. Subsequent push/deploys from
the patcher (over SSH) reuse that cache. If you prune images or add a service
with new base images, pre-pull them once interactively:

```bash
docker compose -f <service>/docker-compose.yml pull   # or: docker pull node:22 postgres:15
```

None of this applies to the Linux VM, which has no macOS keychain.

## Tools (MCP surface)

Read-only:

| Tool                      | What it does                                                                                          |
| ------------------------- | ----------------------------------------------------------------------------------------------------- |
| `list_workspace_services` | List services currently materialized in the patcher workspace.                                        |
| `resolve_service`         | Canonicalize a Janus/display name or alias (e.g. `web1.1`, `CC-Forms-backend`) to the workspace name. |
| `list_files`              | Directory listing scoped to a service repo (`.git/` hidden).                                          |
| `read_file`               | File contents (truncated to `PATCHER_MAX_FILE_BYTES`).                                                |
| `git_status`              | `git status --short --branch` for the service repo.                                                   |
| `git_log`                 | Recent commits (`git log -nN`, iso-strict dates).                                                     |
| `get_diff`                | Unified diff: uncommitted by default, or vs a ref (e.g. `origin/main`) to preview what a push sends.  |

Stage & commit in the local workspace, no push:

| Tool                  | What it does                                                                |
| --------------------- | --------------------------------------------------------------------------- |
| `ensure_service_repo` | Clone the service repo from the VM; re-clones a stale empty workspace, fails with ok=false if the bare repo isn't seeded. |
| `write_files`         | Write and commit a minimal set of `{path, content}` files. Does not push.   |
| `replace_text`        | Replace an exact snippet and commit. Preferred for small patches.           |
| `apply_patch`         | Apply a compact unified diff and commit.                                    |
| `discard_changes`     | `git reset --hard origin/<branch>` + `git clean -fd` to drop a wrong draft. |

Deploy/push to the VM, HITL-gated on the agent side (operator approval required):

| Tool       | What it does                                                                          |
| ---------- | ------------------------------------------------------------------------------------- |
| `deploy`   | Push the branch to origin; the VM `post-receive` hook rebuilds the service container. |
| `rollback` | Create and push a revert commit for a given commit SHA.                               |

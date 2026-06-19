"""Sandboxed git operations for patcher tools, implemented with subprocess calls to the git CLI.

Why subprocess and not GitPython/pygit2: we already need the openssh client
for `git push`, and shelling out to `git` keeps the surface tiny and matches
the operator's mental model (every action you can see in `git reflog`).

https://github.com/modelcontextprotocol/servers-archived/tree/main/src/git 

"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

from patcher_mcp.config import Settings, get_settings

log = logging.getLogger(__name__)


# Service names are filesystem path components: keep them strict.
_SERVICE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")


class PatcherError(RuntimeError):
    """Raised when a patcher operation fails. The message is safe to return to the agent."""


@dataclass(slots=True)
class GitResult:
    returnCode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returnCode == 0


# Path / service helpers

def _validate_service_name(service: str) -> None:
    if not _SERVICE_NAME_RE.match(service):
        raise PatcherError(
            f"invalid service name {service!r}: must be 1-63 chars, "
            "alphanumeric / dot / dash / underscore, starting with alphanumeric"
        )


def _normalize(name: str) -> str:
    """Lowercase and drop every non-alphanumeric char: a case/separator-insensitive key."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _match_candidate(requested: str, candidates: list[str]) -> str | None:
    """Map `requested` to one of `candidates` (real service repo names).

    Tries, in order: exact, case-insensitive, normalized equality (so `CCForms`,
    `cc-forms` and `CC_Forms` collapse together), then a UNIQUE normalized prefix
    match (so a role suffix like `cc-forms-backend` still finds `cc-forms`).
    Returns None when nothing matches or the match is ambiguous, leaving the
    caller to fall back or raise a candidate-listing error rather than guess.
    """
    if requested in candidates:
        return requested
    by_lower = {c.lower(): c for c in candidates}
    if requested.lower() in by_lower:
        return by_lower[requested.lower()]
    req = _normalize(requested)
    if not req:
        return None
    by_norm: dict[str, str] = {}
    for c in candidates:
        by_norm.setdefault(_normalize(c), c)
    if req in by_norm:
        return by_norm[req]
    if len(req) >= 3:
        hits = [
            orig for norm, orig in by_norm.items()
            if len(norm) >= 3 and (req.startswith(norm) or norm.startswith(req))
        ]
        if len(hits) == 1:
            return hits[0]
    return None


def _configured_subpath(service: str, settings: Settings, *, exact: bool = False) -> str:
    """Subpath from PATCHER_SERVICE_SUBPATHS for `service`.

    exact=True matches only the original-case key, so a real repo name can never
    pick up a subpath by case-folding (the bug where requesting repo `cyberuni`
    silently scoped to subpath `auth_service`, because a config key `CyberUni`
    lowercases to the repo name).
    """
    subs = settings.patcher_service_subpaths
    if service in subs:
        return subs[service].strip("/")
    if exact:
        return ""
    low = {k.lower(): v for k, v in subs.items()}
    val = low.get(service.lower(), "")
    return val.strip("/") if val else ""


def _repo_subdirs(repo: str, settings: Settings) -> list[str]:
    """Top-level subdirectories of a MATERIALIZED repo workspace (excluding .git)."""
    root = settings.patcher_workspace_root / repo
    try:
        return [e.name for e in root.iterdir() if e.is_dir() and e.name != ".git"]
    except OSError:
        return []


def _subdir_index(repos: list[str], settings: Settings) -> dict[str, tuple[str, str]]:
    """normalized(subdir) -> (repo, subdir) across materialized repos; ambiguous names dropped.

    This is what lets a SUB-service inside a shared repo be addressed by its own
    name (e.g. `examnotes` -> repo `cyberuni`, subpath `examnotes`) with no
    per-service config, once that repo is cloned.
    """
    idx: dict[str, list[tuple[str, str]]] = {}
    for repo in repos:
        for sub in _repo_subdirs(repo, settings):
            idx.setdefault(_normalize(sub), []).append((repo, sub))
    return {k: v[0] for k, v in idx.items() if len(v) == 1}


def _resolve_service(
    service: str,
    settings: Settings,
    candidates: list[str] | None = None,
    subdir_index: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, str]:
    """Resolve a user-facing service id/name to (repo, subpath).

    Order: PATCHER_SERVICE_ALIASES override (case-insensitive); then an exact REPO
    name -> repo root (so a multi-service repo addressed by its own name shows its
    subdirs, and a repo name never folds into a subpath key by case); then an
    explicit PATCHER_SERVICE_SUBPATHS key (e.g. a Janus id like CyberUni2) ->
    (repo, subpath); then a SUB-service whose name matches a repo subdir ->
    (repo, subdir); else the fuzzy repo match, or the name unchanged so the clone
    path can report it against the real repo list.
    """
    requested = service.strip()
    _validate_service_name(requested)

    aliases = settings.patcher_service_aliases
    by_lower = {k.lower(): v for k, v in aliases.items()}
    if requested in aliases or requested.lower() in by_lower:
        repo = aliases.get(requested) or by_lower[requested.lower()]
        return repo, _configured_subpath(requested, settings)

    known = candidates if candidates is not None else list_workspace_services(settings)

    # A genuine repo name wins and maps to the repo ROOT (collision fix).
    direct = _match_candidate(requested, known)
    if direct and _normalize(direct) == _normalize(requested):
        return direct, _configured_subpath(requested, settings, exact=True)

    # Explicit subpath config key (e.g. a Janus id: CyberUni2 -> examnotes).
    sub = _configured_subpath(requested, settings)
    if sub:
        repo = direct or _match_candidate(re.sub(r"\d+$", "", requested), known) or requested
        return repo, sub

    # A sub-service addressed by its own name -> (repo, subdir), auto-detected.
    idx = subdir_index if subdir_index is not None else _subdir_index(known, settings)
    hit = idx.get(_normalize(requested))
    if hit:
        return hit

    return (direct or requested), ""


def canonical_service_name(
    service: str,
    settings: Settings | None = None,
    candidates: list[str] | None = None,
) -> str:
    """Return the patcher workspace repo name for a user-facing service id/name.

    See `_resolve_service` for the full order. An unresolved name is returned
    unchanged, leaving the clone path to report it against the real repo list.
    """
    s = settings or get_settings()
    repo, _ = _resolve_service(service, s, candidates)
    _validate_service_name(repo)
    return repo


async def resolve_service(service: str, settings: Settings | None = None) -> dict:
    """Canonicalize a service name and report what the patcher can actually reach.

    Lists the real service repos (materialized workspace + best-effort VM
    discovery), resolves `service` against them, and returns the canonical name
    plus the available set, so a wrong/unknown name surfaces the valid options
    immediately instead of a silent dead end.
    """
    s = settings or get_settings()
    local = list_workspace_services(s)
    remote = await list_remote_services(s)
    candidates = sorted(set(local) | set(remote))
    canonical, sub = _resolve_service(service, s, candidates=candidates)
    # If the name resolved to neither a known repo nor a materialized sub-service,
    # it may be a sub-service of a repo not yet cloned -> discover it over SSH.
    if canonical not in candidates and not sub:
        hit = (await _discover_remote_subservices(s)).get(_normalize(service))
        if hit:
            canonical, sub = hit
    repo_root = (s.patcher_workspace_root / canonical).resolve()
    materialized = canonical in local
    result = {
        "requested_service": service,
        "service": canonical,
        "aliased": service != canonical,
        "matched": canonical in candidates,
        "materialized": materialized,
        "available_services": candidates,
        "subpath": sub,
        "root": str(repo_root / sub if sub else repo_root),
        "repo_root": str(repo_root),
        "remote": s.remote_url(canonical),
        "next": (
            None if materialized
            else "call ensure_service_repo to clone it before reading or patching"
        ),
    }
    # When the repo holds several services, list the sibling subdirs so the agent
    # can see it must scope into one (e.g. read examnotes/app/..., not the root).
    if materialized:
        subdirs = _repo_subdirs(canonical, s)
        if len(subdirs) > 1:
            result["subservices"] = sorted(subdirs)
    return result


def service_root(service: str, settings: Settings | None = None) -> Path:
    """Return the repo working tree path for `service` (clone/push/deploy root)."""
    s = settings or get_settings()
    canonical = canonical_service_name(service, s)
    return (s.patcher_workspace_root / canonical).resolve()


def service_subpath(service: str, settings: Settings | None = None) -> str:
    """Subdirectory inside the repo this service lives in ("" if it owns the repo).

    Resolved by `_resolve_service`: an explicit PATCHER_SERVICE_SUBPATHS key, or a
    subdir auto-matched to the requested name in a multi-service repo. Keyed by the
    *requested* id, so several ids sharing one repo each scope to their own subdir.
    """
    s = settings or get_settings()
    _, sub = _resolve_service(service, s)
    return sub


def work_root(service: str, settings: Settings | None = None) -> Path:
    """Root for file/edit ops: the repo root, or its per-service subpath if set."""
    root = service_root(service, settings)
    sub = service_subpath(service, settings)
    if not sub:
        return root
    candidate = (root / sub).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise PatcherError(f"service subpath escapes the repo root: {sub!r}")
    return candidate


def _scoped_path(service: str, rel_path: str, settings: Settings | None = None) -> Path:
    """Resolve `rel_path` inside the service work root, refusing traversal."""
    root = work_root(service, settings)
    # Reject absolute paths up front, the agent should only ever pass paths
    # relative to the service root.
    if os.path.isabs(rel_path):
        raise PatcherError(f"path must be relative to the service root: {rel_path!r}")
    candidate = (root / rel_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise PatcherError(f"path escapes the service root: {rel_path!r}")
    return candidate


def list_workspace_services(settings: Settings | None = None) -> list[str]:
    """Return the list of services currently materialized in the workspace."""
    s = settings or get_settings()
    root = s.patcher_workspace_root
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / ".git").exists() and _SERVICE_NAME_RE.match(p.name)
    )


# Subprocess helpers

async def _run(
    cmd: list[str],
    *,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
) -> GitResult:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdin=asyncio.subprocess.PIPE if input_bytes is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await proc.communicate(input=input_bytes)
    return GitResult(
        returnCode=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout_bytes.decode(errors="replace"),
        stderr=stderr_bytes.decode(errors="replace"),
    )


def _git_env(settings: Settings) -> dict[str, str]:
    """Environment for git invocations.

    - GIT_SSH_COMMAND injects the right private key and ssh options so
      `git push` works without an ssh-agent.
    - GIT_TERMINAL_PROMPT=0 prevents git from blocking on credential prompts.
    """
    env = os.environ.copy()
    env["GIT_SSH_COMMAND"] = settings.git_ssh_command()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


async def _git(
    service: str,
    args: list[str],
    *,
    settings: Settings | None = None,
    check: bool = True,
) -> GitResult:
    """Run `git <args>` inside the service working tree."""
    s = settings or get_settings()
    root = service_root(service, s)
    if not root.exists():
        raise PatcherError(f"service workspace missing: {root}")
    res = await _run(["git", *args], cwd=root, env=_git_env(s))
    if check and not res.ok:
        # stderr from git is small and operator-friendly, surface it.
        msg = res.stderr.strip() or res.stdout.strip() or f"git {args[0]} failed"
        raise PatcherError(f"git {args[0]}: {msg}")
    return res


async def list_remote_services(settings: Settings | None = None) -> list[str]:
    """List of the seeded bare repos on the VM (their service names).

    Runs `ls` in VM_GIT_REMOTE_ROOT over SSH and strips the `.git` suffix: this
    is the authoritative set of canonical service names the patcher can clone.
    Returns [] on any failure (SSH down, dir missing) so name resolution can
    degrade to the requested name instead of breaking.
    """
    s = settings or get_settings()
    remote_root = s.vm_git_remote_root.rstrip("/")
    script = f"ls -1 {shlex.quote(remote_root)} 2>/dev/null || true"
    # This runs on the hot resolve path, so a down or
    # unreachable VM must not hang the agent waiting on the default SSH timeout.
    ssh = s.ssh_base_command()
    ssh = [ssh[0], "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", *ssh[1:]]
    try:
        res = await _run(ssh + [f"sh -c {shlex.quote(script)}"])
    except Exception as exc:
        log.warning("list_remote_services: SSH listing failed: %s", exc)
        return []
    names: list[str] = []
    for line in res.stdout.splitlines():
        name = line.strip()
        if name.endswith(".git"):
            name = name[: -len(".git")]
        if name and _SERVICE_NAME_RE.match(name):
            names.append(name)
    return sorted(set(names))


async def _discover_remote_subservices(settings: Settings | None = None) -> dict[str, tuple[str, str]]:
    """normalized(subdir) -> (repo, subdir) for the top-level dirs of every bare repo.

    The materialized `_subdir_index` only sees repos already cloned; this lets a
    SUB-service be resolved by name BEFORE its (multi-service) repo is cloned, so
    e.g. `ensure_service_repo("examnotes")` knows to clone `cyberuni` and scope to
    its `examnotes/` subdir. One SSH round-trip, ambiguous names dropped, degrades
    to {} on any failure (caller falls back to the requested name).
    """
    s = settings or get_settings()
    root = s.vm_git_remote_root.rstrip("/")
    script = (
        f"cd {shlex.quote(root)} 2>/dev/null || exit 0; "
        'for d in *.git; do [ -d "$d" ] || continue; repo=${d%.git}; '
        'git --git-dir="$d" ls-tree -d --name-only HEAD 2>/dev/null | '
        'while IFS= read -r sub; do printf "%s\\t%s\\n" "$repo" "$sub"; done; done'
    )
    ssh = s.ssh_base_command()
    ssh = [ssh[0], "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", *ssh[1:]]
    try:
        res = await _run(ssh + [f"sh -c {shlex.quote(script)}"])
    except Exception as exc:
        log.warning("_discover_remote_subservices: SSH failed: %s", exc)
        return {}
    idx: dict[str, list[tuple[str, str]]] = {}
    for line in res.stdout.splitlines():
        if "\t" not in line:
            continue
        repo, sub = (p.strip() for p in line.split("\t", 1))
        if repo and sub and _SERVICE_NAME_RE.match(repo):
            idx.setdefault(_normalize(sub), []).append((repo, sub))
    return {k: v[0] for k, v in idx.items() if len(v) == 1}


async def list_vm_services(settings: Settings | None = None) -> dict:
    """Inventory of services on the competition VM, gathered over SSH.

    Returns `running_containers` (the Docker containers actually running on the
    VM: name/image/status/ports) and `service_dirs` (the service folders present
    on the VM, i.e. the bare repos under VM_GIT_REMOTE_ROOT). This is deliberately
    broader than Janus' list_services, which only knows the services it proxies:
    it surfaces services that run on the VM (or merely exist on disk) but are NOT
    behind Janus. Degrades to empty lists on any SSH/Docker failure so the
    inventory never crashes the agent.
    """
    s = settings or get_settings()
    remote_root = s.vm_git_remote_root.rstrip("/")
    # One SSH round-trip carries both queries; sentinel lines split the sections.
    # `docker ps --format '{{json .}}'` emits one JSON object per container, which
    # parses cleanly regardless of spaces in the image/status/ports fields.
    # Prepend the usual macOS/Linux docker install dirs: a non-interactive SSH
    # shell often has a minimal PATH, so a bare `docker` may be "command not
    # found" and look like "no containers". Capture docker's own exit code +
    # stderr (no 2>/dev/null) so a FAILED query is reported as such, not as an
    # empty list that the agent then mis-reads as "nothing is running".
    script = (
        'export PATH="$PATH:/usr/local/bin:/opt/homebrew/bin:'
        '/Applications/Docker.app/Contents/Resources/bin"; '
        "echo __CONTAINERS__; "
        "{ docker ps --format '{{json .}}'; echo \"__DOCKER_RC__:$?\"; } 2>&1; "
        "echo __SERVICE_DIRS__; "
        f"ls -1 {shlex.quote(remote_root)} 2>/dev/null || true"
    )
    # Bound the SSH wait so a down/unreachable VM can't hang the agent.
    ssh = s.ssh_base_command()
    ssh = [ssh[0], "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", *ssh[1:]]
    try:
        res = await _run(ssh + [f"sh -c {shlex.quote(script)}"])
    except Exception as exc:
        log.warning("list_vm_services: SSH failed: %s", exc)
        return {
            "running_containers": [], "service_dirs": [],
            "containers_query_failed": True, "containers_error": f"SSH failed: {exc}",
        }

    containers: list[dict[str, str]] = []
    service_dirs: list[str] = []
    docker_rc: int | None = None
    docker_err: list[str] = []
    section: str | None = None
    for line in res.stdout.splitlines():
        stripped = line.strip()
        if stripped == "__CONTAINERS__":
            section = "containers"
            continue
        if stripped == "__SERVICE_DIRS__":
            section = "dirs"
            continue
        if not stripped:
            continue
        if section == "containers":
            if stripped.startswith("__DOCKER_RC__:"):
                try:
                    docker_rc = int(stripped.split(":", 1)[1])
                except ValueError:
                    docker_rc = None
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                docker_err.append(stripped)  # docker error text, not a container
                continue
            containers.append({
                "name": str(row.get("Names", "")),
                "image": str(row.get("Image", "")),
                "status": str(row.get("Status", "")),
                "ports": str(row.get("Ports", "")),
            })
        elif section == "dirs":
            name = stripped[: -len(".git")] if stripped.endswith(".git") else stripped
            if _SERVICE_NAME_RE.match(name):
                service_dirs.append(name)
    result: dict = {
        "running_containers": containers,
        "service_dirs": sorted(set(service_dirs)),
    }
    # Distinguish "docker ran, found nothing" from "docker query failed": only the
    # former means no containers are running.
    if docker_rc not in (0, None) or (not containers and docker_err):
        result["containers_query_failed"] = True
        result["containers_error"] = (
            "\n".join(docker_err)[:500] or f"docker ps exited {docker_rc}"
        )
    return result


# Read operations


async def read_file(service: str, rel_path: str) -> str:
    """Return file contents (text), truncated to `patcher_max_file_bytes`."""
    s = get_settings()
    path = _scoped_path(service, rel_path, s)
    if not path.exists():
        raise PatcherError(f"file not found: {rel_path}")
    if not path.is_file():
        raise PatcherError(f"not a regular file: {rel_path}")
    size = path.stat().st_size
    cap = s.patcher_max_file_bytes
    with path.open("rb") as fileHandle:
        data = fileHandle.read(cap + 1)
    truncated = len(data) > cap
    text = data[:cap].decode(errors="replace")
    if truncated:
        text += f"\n\n[... truncated, file is {size} bytes, cap is {cap} bytes ...]"
    return text


async def list_files(service: str, rel_path: str = ".") -> list[dict]:
    """List directory entries. Skips `.git/`. Caps at `patcher_max_list_entries`."""
    s = get_settings()
    root = work_root(service, s)
    base = _scoped_path(service, rel_path, s) if rel_path not in ("", ".") else root
    if not base.exists():
        raise PatcherError(f"path not found: {rel_path}")
    if not base.is_dir():
        raise PatcherError(f"not a directory: {rel_path}")
    out: list[dict] = []
    for entry in sorted(base.iterdir()):
        if entry.name == ".git":
            continue
        rel = entry.relative_to(root).as_posix()
        out.append({
            "path": rel,
            "type": "dir" if entry.is_dir() else "file",
            "size": entry.stat().st_size if entry.is_file() else None,
        })
        if len(out) >= s.patcher_max_list_entries:
            break
    return out


async def git_status(service: str) -> str:
    res = await _git(service, ["status", "--short", "--branch"])
    return res.stdout.strip()


async def git_log(service: str, n: int = 10) -> str:
    res = await _git(
        service,
        ["log", f"-n{n}", "--pretty=format:%h %ad %an %s", "--date=iso-strict"],
    )
    return res.stdout.strip()


async def get_diff(service: str, ref: str | None = None) -> str:
    """Return a diff. Default: working tree + index vs HEAD (i.e. uncommitted changes).

    If `ref` is set, returns `git diff ref..HEAD` (commits since `ref` on the
    current branch). Useful as `ref="origin/main"` to see what will be pushed.
    """
    s = get_settings()
    args = ["diff", "--patch", "--stat"] if ref is None else ["diff", "--patch", "--stat", f"{ref}..HEAD"]
    sub = service_subpath(service, s)
    if sub:
        # Scope the diff to this service's subdir and show subpath-relative paths.
        args += [f"--relative={sub}", "--", sub]
    res = await _git(service, args)
    cap = s.patcher_max_diff_bytes
    out = res.stdout
    if len(out) > cap:
        out = out[:cap] + f"\n\n[... diff truncated at {cap} bytes ...]"
    return out


# Workspace initialization


async def _repo_has_commit(root: Path, env: dict[str, str]) -> bool:
    """True if the repo at `root` has at least one commit (a valid HEAD)."""
    res = await _run(["git", "rev-parse", "--verify", "HEAD"], cwd=root, env=env)
    return res.ok


async def ensure_service_repo(service: str) -> dict:
    """Materialize the service repo locally WITH source, or fail loudly.

    The bare repo on the VM is seeded by scripts/init-vm.sh; here we only ever
    CLONE it. We deliberately do NOT `git init` an empty local repo: a
    content-less repo is indistinguishable from "source present" to the agent
    and sends it into an infinite list_files/git_log loop.

    - local repo exists AND has a commit -> "already-present".
    - nothing local, OR a leftover empty `.git` (an earlier failed clone) ->
      wipe and re-clone (self-heals the stale-empty-workspace bug).
    - clone fails (remote unreachable / repo missing), OR the clone is empty
      (bare repo never seeded) -> raise PatcherError, so the tool returns
      ok=false and the agent stops instead of looping on empty reads.

    Sets local user.name/user.email so commits don't need a global config.
    """
    s = get_settings()
    requested_service = service
    canonical = canonical_service_name(service, s)
    root = service_root(canonical, s)
    env = _git_env(s)

    if (root / ".git").exists() and await _repo_has_commit(root, env):
        await _git(canonical, ["config", "user.name", s.patcher_git_user_name])
        await _git(canonical, ["config", "user.email", s.patcher_git_user_email])
        return {
            "requested_service": requested_service,
            "service": canonical,
            "root": str(root),
            "remote": s.remote_url(canonical),
            "action": "already-present",
            "created": False,
            "has_source": True,
        }

    # Nothing usable locally. Discover the real repo names on the VM and
    # re-resolve, so a Janus/display-name mismatch (case, separators, a role
    # suffix) finds the right bare repo instead of cloning a ghost.
    # if discovery fails we keep the resolved name and let the clone error talk.
    remote_services = await list_remote_services(s)
    if remote_services:
        match = _match_candidate(canonical, remote_services) or _match_candidate(
            requested_service, remote_services
        )
        if not match and canonical not in remote_services:
            # Not a repo name: maybe a SUB-service of a multi-service repo not yet
            # cloned (e.g. "examnotes" lives in repo "cyberuni"). Discover it.
            hit = (await _discover_remote_subservices(s)).get(_normalize(requested_service))
            if hit:
                match = hit[0]
        if match:
            canonical = match
        elif canonical not in remote_services:
            raise PatcherError(
                f"no service repo matching {requested_service!r} on the VM. "
                f"Available services: {', '.join(remote_services)}. "
                "Pass one of these names exactly, or set an override "
                f'PATCHER_SERVICE_ALIASES={{"{requested_service}": "<repo>"}}.'
            )
        root = service_root(canonical, s)

    remote = s.remote_url(canonical)

    # remove whatever is there and clone fresh from the remote.
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    clone = await _run(["git", "clone", remote, str(root)], env=env)
    if not clone.ok:
        raise PatcherError(
            f"no source for {canonical}: clone from {remote} failed "
            f"({clone.stderr.strip() or 'remote unreachable'}). "
            "Seed the bare repo on the VM with scripts/init-vm.sh, then retry."
        )

    # Per-repo identity
    await _git(canonical, ["config", "user.name", s.patcher_git_user_name])
    await _git(canonical, ["config", "user.email", s.patcher_git_user_email])

    # A clone can succeed yet be empty when the bare repo has no commits. That
    # is not usable source: fail instead of returning a hollow success.
    if not await _repo_has_commit(root, env):
        raise PatcherError(
            f"no source for {canonical}: repo at {remote} is empty (no commits). "
            "Seed it with scripts/init-vm.sh, then retry."
        )

    return {
        "requested_service": requested_service,
        "service": canonical,
        "root": str(root),
        "remote": remote,
        "action": "cloned",
        "created": True,
        "has_source": True,
    }


# Write operations

async def _switch_branch(service: str, branch: str, settings: Settings) -> None:
    """Switch to / create a branch before writing files."""
    has_head = (await _git(service, ["rev-parse", "--verify", "HEAD"], check=False)).ok
    if has_head:
        sw_br = await _git(service, ["switch", branch], check=False)
        if not sw_br.ok:
            await _git(service, ["switch", "-c", branch])
    else:
        await _git(service, ["symbolic-ref", "HEAD", f"refs/heads/{branch}"], check=False)


async def _commit_paths(service: str, paths: list[str], message: str) -> dict:
    if not paths:
        raise PatcherError("no files changed")

    await _git(service, ["add", "--", *paths])
    diff_cached = await _git(service, ["diff", "--cached", "--patch", "--stat"])
    diff = diff_cached.stdout
    cap = get_settings().patcher_max_diff_bytes
    if len(diff) > cap:
        diff = diff[:cap] + f"\n\n[... diff truncated at {cap} bytes ...]"

    no_changes = await _git(service, ["diff", "--cached", "--quiet"], check=False)
    if no_changes.ok:
        return {"committed": False, "reason": "no changes to commit", "files": paths}

    commit = await _git(service, ["commit", "-m", message])
    sha = (await _git(service, ["rev-parse", "HEAD"])).stdout.strip()
    return {
        "committed": True,
        "commit_sha": sha,
        "files": paths,
        "message": message,
        "diff": diff,
        "stderr": commit.stderr.strip(),
    }


async def write_files(
    service: str,
    files: list[dict],
    message: str,
    branch: str | None = None,
) -> dict:
    """Write a set of files, stage, commit. Does NOT push.

    `files` is a list of `{"path": "<rel>", "content": "<text>"}` items.
    Existing files are overwritten; missing parent directories are created.
    """
    if not files:
        raise PatcherError("no files provided")
    if not message.strip():
        raise PatcherError("commit message is required")
    s = get_settings()
    requested_service = service
    service = canonical_service_name(service, s)
    branch = branch or s.patcher_default_branch

    # Switch before writing so we don't accidentally commit on a wrong branch.
    await _switch_branch(service, branch, s)

    written: list[str] = []
    for item in files:
        rel = item.get("path")
        content = item.get("content")
        if not isinstance(rel, str) or not isinstance(content, str):
            raise PatcherError("each file must have string 'path' and 'content'")
        dest = _scoped_path(service, rel, s)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        written.append(rel)

    result = await _commit_paths(service, written, message)
    return {
        "requested_service": requested_service,
        "service": service,
        "branch": branch,
        **result,
    }


async def replace_text(
    service: str,
    rel_path: str,
    old: str,
    new: str,
    message: str,
    branch: str | None = None,
    expected_replacements: int = 1,
) -> dict:
    """Replace an exact text snippet and commit the result.

    This is the preferred path for small patches because it keeps full file
    contents out of the LLM/tool history.
    """
    if not old:
        raise PatcherError("'old' text must be non-empty")
    if expected_replacements < 1:
        raise PatcherError("expected_replacements must be >= 1")
    if not message.strip():
        raise PatcherError("commit message is required")

    s = get_settings()
    requested_service = service
    service = canonical_service_name(service, s)
    branch = branch or s.patcher_default_branch
    await _switch_branch(service, branch, s)

    path = _scoped_path(service, rel_path, s)
    if not path.exists():
        raise PatcherError(f"file not found: {rel_path}")
    if not path.is_file():
        raise PatcherError(f"not a regular file: {rel_path}")

    text = path.read_text()
    found = text.count(old)
    if found != expected_replacements:
        raise PatcherError(
            f"replace_text expected {expected_replacements} match(es) in {rel_path}, found {found}"
        )
    path.write_text(text.replace(old, new, expected_replacements))

    result = await _commit_paths(service, [rel_path], message)
    return {
        "requested_service": requested_service,
        "service": service,
        "branch": branch,
        "replacements": expected_replacements,
        **result,
    }


async def apply_patch(
    service: str,
    patch: str,
    message: str,
    branch: str | None = None,
) -> dict:
    """Apply a unified diff with `git apply`, stage, and commit."""
    if not patch.strip():
        raise PatcherError("patch is empty")
    if not message.strip():
        raise PatcherError("commit message is required")

    s = get_settings()
    requested_service = service
    service = canonical_service_name(service, s)
    branch = branch or s.patcher_default_branch
    await _switch_branch(service, branch, s)

    root = service_root(service, s)
    # Patch paths are relative to the service work root; --directory prefixes the
    # subpath so it applies correctly at the repo root (no-op when no subpath).
    sub = service_subpath(requested_service, s)
    apply_cmd = ["git", "apply", f"--directory={sub}"] if sub else ["git", "apply"]
    patch_bytes = patch.encode()
    check = await _run([*apply_cmd, "--check", "-"], cwd=root, env=_git_env(s), input_bytes=patch_bytes)
    if not check.ok:
        raise PatcherError(f"git apply --check failed: {(check.stderr or check.stdout).strip()}")

    applied = await _run([*apply_cmd, "-"], cwd=root, env=_git_env(s), input_bytes=patch_bytes)
    if not applied.ok:
        raise PatcherError(f"git apply failed: {(applied.stderr or applied.stdout).strip()}")

    names = (await _git(service, ["diff", "--name-only"])).stdout.splitlines()
    result = await _commit_paths(service, names, message)
    return {
        "requested_service": requested_service,
        "service": service,
        "branch": branch,
        **result,
    }


async def discard_changes(service: str, branch: str | None = None) -> dict:
    """Reset the working tree to `origin/<branch>`. Destructive."""
    s = get_settings()
    branch = branch or s.patcher_default_branch
    await _git(service, ["fetch", "origin", branch], check=False)
    res = await _git(service, ["reset", "--hard", f"origin/{branch}"], check=False)
    if not res.ok:
        # No remote branch yet, reset to HEAD instead.
        res = await _git(service, ["reset", "--hard"])
    await _git(service, ["clean", "-fd"])
    return {"reset_to": f"origin/{branch}", "output": res.stdout.strip() or res.stderr.strip()}


# Deploy / rollback


def _remote_output_has_hook_error(output: str) -> bool:
    """Detection of a failed post-receive hook in push output.

    A post-receive hook cannot reject the push, refs are already updated by the
    time it runs, so a failed rebuild surfaces ONLY as text on the remote's
    stderr (relayed as `remote: ...` lines). We scan for that text.
    
    """
    lowered = output.lower()
    markers = (
        "remote: fatal:",
        "remote: error:",
        "remote: failed",
        "remote: cannot",
        "remote: no such file",
        "remote: not found",
        "hook declined",
        # Shell / tooling failures from inside the hook. These never appear in a
        # successful `docker compose up --build`, so they are safe to treat as
        # hard failures rather than risk a silent bad deploy.
        "deploy-failed",
        "command not found",
        "cannot connect to the docker daemon",
    )
    return any(marker in lowered for marker in markers)


async def _verify_remote_branch(service: str, branch: str, sha: str) -> dict:
    """Confirm that `origin/<branch>` really points at the SHA we just pushed.

    `git push` reporting success is not enough: we re-query the remote with
    `ls-remote` and compare the ref it advertises against the local HEAD we
    intended to deploy. A mismatch (or a missing ref) means the push did not
    land what we expected, so we fail loudly instead of reporting a good deploy.
    """
    remote = await _git(service, ["ls-remote", "origin", f"refs/heads/{branch}"])
    # `ls-remote` prints "<sha>\t<ref>"; take the first token as the remote SHA.
    remote_sha = remote.stdout.split(maxsplit=1)[0] if remote.stdout.strip() else ""
    if remote_sha != sha:
        raise PatcherError(
            f"deploy verification failed: origin/{branch} is {remote_sha or '<missing>'}, expected {sha}"
        )
    return {"origin_branch": f"origin/{branch}", "commit_sha": remote_sha}


async def _verify_remote_worktree(
    service: str,
    branch: str,
    sha: str,
    worktree: str,
    settings: Settings,
) -> dict:
    """Verify, over SSH, that the checked-out copy on the VM is at our SHA.

    The bare repo on the VM advancing (checked by `_verify_remote_branch`) does
    not guarantee the post-receive hook actually checked the new code out into
    the running worktree. Here we SSH in and inspect that directory directly.

    The remote shell script is fully quoted with `shlex.quote` to avoid any
    injection from service/branch/sha values, and uses custom exit codes so the
    failure reason survives the SSH round-trip:
      - 42: the worktree directory does not exist on the VM.
      - 43: the worktree exists but its HEAD is not the SHA we deployed.
    On success it prints the resolved SHA on stdout.
    """
    remote_git = f"{settings.vm_git_remote_root.rstrip('/')}/{service}.git"
    script = "\n".join([
        "set -eu",
        f"WORKTREE={shlex.quote(worktree)}",
        f"REMOTE_GIT={shlex.quote(remote_git)}",
        f"BRANCH={shlex.quote(branch)}",
        f"EXPECTED={shlex.quote(sha)}",
        'if [ ! -d "$WORKTREE" ]; then echo "worktree missing: $WORKTREE"; exit 42; fi',
        # A real checkout has its own .git; read HEAD from there. Otherwise fall
        # back to the branch ref in the bare repo as the best available signal.
        'if [ -d "$WORKTREE/.git" ]; then',
        '  ACTUAL=$(git -C "$WORKTREE" rev-parse HEAD)',
        "else",
        '  ACTUAL=$(git --git-dir="$REMOTE_GIT" rev-parse "$BRANCH")',
        "fi",
        'if [ "$ACTUAL" != "$EXPECTED" ]; then',
        '  echo "worktree HEAD mismatch: got $ACTUAL expected $EXPECTED"',
        "  exit 43",
        "fi",
        'printf "%s" "$ACTUAL"',
    ])
    res = await _run(settings.ssh_base_command() + [f"sh -c {shlex.quote(script)}"])
    if not res.ok:
        raise PatcherError(
            f"deploy verification failed for worktree {worktree}: "
            f"{(res.stderr or res.stdout).strip()}"
        )
    return {"worktree": worktree, "commit_sha": res.stdout.strip()}


_HOOK_KEYCHAIN_MARKER = "patcher-docker-config"
_HOOK_KEYCHAIN_BLOCK = (
    '# [patcher self-heal] macOS Docker Desktop credsStore="desktop" routes every\n'
    "# image pull through docker-credential-desktop, which reads the login keychain.\n"
    "# This hook runs in the non-interactive ssh session of the push, where the\n"
    '# keychain is locked ("user interaction is not allowed") and the build dies.\n'
    "# Point docker at a throwaway config with no credsStore so public base images\n"
    "# pull anonymously; cliPluginsExtraDirs keeps `docker compose`/buildx resolvable.\n"
    'DOCKER_CONFIG="${TMPDIR:-/tmp}/patcher-docker-config"\n'
    'mkdir -p "$DOCKER_CONFIG"\n'
    'cat > "$DOCKER_CONFIG/config.json" <<JSON\n'
    '{ "cliPluginsExtraDirs": ["$HOME/.docker/cli-plugins"] }\n'
    "JSON\n"
    "export DOCKER_CONFIG\n"
)


def _heal_hook_text(text: str) -> str | None:
    """Return patched hook text if it needs the keychain fix, else None.

    Idempotent: inserts the DOCKER_CONFIG sanitation block immediately before the
    `docker compose up` line (so it is exported before any registry pull) and adds
    `--force-recreate` so a rebuilt image actually replaces the running container.
    Returns None when the hook already has both (nothing to do) or is unrecognised
    (no `docker compose up` line — leave it untouched rather than corrupt it).
    """
    new = text
    if _HOOK_KEYCHAIN_MARKER not in new:
        i = new.find("docker compose up")
        if i == -1:
            return None
        line_start = new.rfind("\n", 0, i) + 1
        new = new[:line_start] + _HOOK_KEYCHAIN_BLOCK + new[line_start:]
    if "--build --force-recreate" not in new:
        new = new.replace("compose up -d --build", "compose up -d --build --force-recreate")
    return new if new != text else None


async def _ensure_remote_hook_keychain(service: str, settings: Settings) -> dict:
    """Make sure the live VM post-receive hook has the keychain fix before we push.

    Reads the hook over SSH, heals it in memory if needed, and writes it back.
    A missing hook (repo not seeded) is left to init-vm.sh; a write failure is
    raised so the operator sees it rather than deploying through a stale hook.
    """
    remote_git = f"{settings.vm_git_remote_root.rstrip('/')}/{service}.git"
    hook = f"{remote_git}/hooks/post-receive"
    read = await _run(
        settings.ssh_base_command()
        + [f"sh -c {shlex.quote(f'cat {shlex.quote(hook)} 2>/dev/null')}"]
    )
    text = read.stdout
    if not text.strip():
        return {"hook": "absent"}
    patched = _heal_hook_text(text)
    if patched is None:
        return {"hook": "current"}
    write_cmd = f"cat > {shlex.quote(hook)} && chmod +x {shlex.quote(hook)}"
    res = await _run(
        settings.ssh_base_command() + [f"sh -c {shlex.quote(write_cmd)}"],
        input_bytes=patched.encode(),
    )
    if not res.ok:
        raise PatcherError(
            f"failed to refresh post-receive hook on VM ({hook}): "
            f"{(res.stderr or res.stdout).strip()}"
        )
    return {"hook": "healed"}


async def deploy(service: str, branch: str | None = None) -> dict:
    """`git push origin <branch>`. The VM's post-receive hook does the rebuild."""
    s = get_settings()
    requested_service = service
    # Normalise aliases to the real service name, but keep the caller's original
    # spelling so we can echo it back in the result.
    service = canonical_service_name(service, s)
    branch = branch or s.patcher_default_branch
    # Self-heal the live hook before pushing so the macOS keychain does not break
    # the rebuild on a hook seeded before the credsStore fix landed.
    hook_state = await _ensure_remote_hook_keychain(service, s)
    # Snapshot the local HEAD now: this is the SHA we expect to find live after
    # the push, and what every verification step below compares against.
    sha = (await _git(service, ["rev-parse", "HEAD"])).stdout.strip()
    push = await _git(service, ["push", "origin", branch], check=False)
    if not push.ok:
        raise PatcherError(
            f"git push failed: {(push.stderr or push.stdout).strip()}"
        )
    # Push progress and any `remote:` lines come through on stderr; fall back to
    # stdout just in case.
    output = (push.stderr or push.stdout).strip()
    # The push can succeed (refs updated) while the rebuild hook fails. Scan the
    # relayed remote output so a broken rebuild does not pass as a good deploy.
    if _remote_output_has_hook_error(output):
        raise PatcherError(
            "git push updated the remote, but the remote hook reported an error: "
            f"{output}"
        )

    # Two-level verification: the remote ref advanced, and (if configured) the
    # live worktree on the VM was actually checked out to that SHA.
    verified = {"remote": await _verify_remote_branch(service, branch, sha)}
    worktree = s.deploy_worktree(service)
    if worktree:
        verified["worktree"] = await _verify_remote_worktree(service, branch, sha, worktree, s)

    return {
        "deployed": True,
        "requested_service": requested_service,
        "service": service,
        "branch": branch,
        "commit_sha": sha,
        "remote": s.remote_url(service),
        "output": output,
        "verified": verified,
        "hook": hook_state,
    }


async def rollback(service: str, commit_sha: str, branch: str | None = None) -> dict:
    """Create a revert commit for `commit_sha` and push it.

    Rollback is "forward, not backward": instead of rewriting history we add a
    new commit that undoes `commit_sha`, then deploy it the normal way. This
    keeps the branch fast-forwardable and avoids force-pushes on the VM remote.
    """
    s = get_settings()
    requested_service = service
    service = canonical_service_name(service, s)
    branch = branch or s.patcher_default_branch
    # Validate the SHA before it ever reaches a git command (hex, 4-64 chars).
    if not re.match(r"^[0-9a-fA-F]{4,64}$", commit_sha):
        raise PatcherError(f"invalid commit sha: {commit_sha!r}")
    # Make sure we are on the target branch; create it if it does not exist yet.
    sw = await _git(service, ["switch", branch], check=False)
    if not sw.ok:
        await _git(service, ["switch", "-c", branch])
    revert = await _git(service, ["revert", "--no-edit", commit_sha], check=False)
    if not revert.ok:
        # A failed revert (e.g. conflicts) leaves the tree mid-revert; abort so
        # the repo is left clean rather than in a half-applied state.
        await _git(service, ["revert", "--abort"], check=False)
        raise PatcherError(
            f"git revert failed: {(revert.stderr or revert.stdout).strip()}"
        )
    # Reuse deploy so the revert goes through the same push + verification path.
    deploy_res = await deploy(service, branch)
    return {
        "rolled_back": commit_sha,
        "requested_service": requested_service,
        "service": service,
        "new_commit_sha": deploy_res["commit_sha"],
        "branch": branch,
        "remote": deploy_res["remote"],
        "verified": deploy_res.get("verified"),
    }


async def _first_commit_sha(service: str, branch: str) -> str | None:
    """Earliest (root/seed) commit SHA on `branch`, or None if the repo is empty.

    The seed commit is the initial import done by scripts/init-vm.sh: the state
    of the service BEFORE any patch landed. We take the root commit (no parents);
    if history has several roots we pick the earliest one rev-list returns.
    """
    res = await _git(service, ["rev-list", "--max-parents=0", branch], check=False)
    if not res.ok:
        return None
    shas = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    return shas[-1] if shas else None


async def list_commits(service: str, branch: str | None = None, n: int = 50) -> dict:
    """Structured commit history for one service, newest first.

    Used to disambiguate vague rollback requests: each entry carries the full and
    short SHA, ISO date, author, subject, and an `is_seed` flag marking the first
    (unpatched) commit. History is scoped to the service's subpath when it shares
    a repo, so a sub-service only shows its own commits.
    """
    s = get_settings()
    requested_service = service
    service = canonical_service_name(service, s)
    branch = branch or s.patcher_default_branch
    sub = service_subpath(service, s)
    args = [
        "log",
        branch,
        f"-n{n}",
        "--pretty=format:%H%x09%h%x09%ad%x09%an%x09%s",
        "--date=iso-strict",
    ]
    if sub:
        args += ["--", sub]
    res = await _git(service, args, check=False)
    if not res.ok:
        raise PatcherError(
            f"git log failed: {(res.stderr or res.stdout).strip()}"
        )
    seed = await _first_commit_sha(service, branch)
    commits: list[dict] = []
    for line in res.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 5:
            continue
        full, short, date, author, subject = parts
        commits.append(
            {
                "sha": full,
                "short_sha": short,
                "date": date,
                "author": author,
                "subject": subject,
                "is_seed": full == seed,
            }
        )
    return {
        "requested_service": requested_service,
        "service": service,
        "branch": branch,
        "seed_commit": seed,
        "commits": commits,
    }


async def rollback_to(
    service: str, target_commit: str | None = None, branch: str | None = None
) -> dict:
    """Restore a service to the state of `target_commit` and deploy it.

    `target_commit=None` means the first (seed) commit -> drop ALL patches. Like
    `rollback`, this is "forward, not backward": we don't rewrite history, we
    create a new commit whose tree matches `target_commit` and deploy it the
    normal way. Scoped to the service's subpath, so rolling back one sub-service
    never touches its siblings sharing the repo.
    """
    s = get_settings()
    requested_service = service
    service = canonical_service_name(service, s)
    branch = branch or s.patcher_default_branch
    if target_commit is not None and not re.match(r"^[0-9a-fA-F]{4,64}$", target_commit):
        raise PatcherError(f"invalid commit sha: {target_commit!r}")
    # Make sure we are on the target branch; create it if it does not exist yet.
    sw = await _git(service, ["switch", branch], check=False)
    if not sw.ok:
        await _git(service, ["switch", "-c", branch])
    # Resolve / validate the target.
    to_seed = target_commit is None
    if to_seed:
        target_commit = await _first_commit_sha(service, branch)
        if not target_commit:
            raise PatcherError("no commits found for service (repo unseeded?)")
    else:
        verify = await _git(
            service, ["rev-parse", "--verify", f"{target_commit}^{{commit}}"], check=False
        )
        if not verify.ok:
            raise PatcherError(f"commit not found: {target_commit}")
    short_target = target_commit[:12]
    sub = service_subpath(service, s)
    pathspec = [sub] if sub else ["."]
    # Restore tracked files in the subpath to their state at the target commit.
    co = await _git(service, ["checkout", target_commit, "--", *pathspec], check=False)
    if not co.ok:
        raise PatcherError(f"git checkout failed: {(co.stderr or co.stdout).strip()}")
    # `checkout <tree> -- path` only rewrites files present in the target; files
    # ADDED in the subpath after the target survive, so remove them explicitly to
    # make the tree an exact match.
    added = await _git(
        service,
        ["diff", "--name-only", "--diff-filter=A", target_commit, "HEAD", "--", *pathspec],
        check=False,
    )
    to_remove = [p for p in added.stdout.splitlines() if p.strip()]
    if to_remove:
        await _git(service, ["rm", "-f", "--", *to_remove], check=False)
    # Nothing staged means the service is already at the target state.
    no_changes = await _git(service, ["diff", "--cached", "--quiet"], check=False)
    if no_changes.ok:
        return {
            "rolled_back_to": target_commit,
            "to_seed": to_seed,
            "requested_service": requested_service,
            "service": service,
            "branch": branch,
            "changed": False,
            "reason": "already at target commit; nothing to roll back",
        }
    label = "seed commit" if to_seed else f"commit {short_target}"
    await _git(service, ["commit", "-m", f"rollback: restore {service} to {label}"])
    # Reuse deploy so the rollback goes through the same push + verification path.
    deploy_res = await deploy(service, branch)
    return {
        "rolled_back_to": target_commit,
        "to_seed": to_seed,
        "requested_service": requested_service,
        "service": service,
        "new_commit_sha": deploy_res["commit_sha"],
        "branch": branch,
        "changed": True,
        "remote": deploy_res["remote"],
        "verified": deploy_res.get("verified"),
    }

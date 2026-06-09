"""Sandboxed git operations for patcher tools, implemented with subprocess calls to the git CLI.

Why subprocess and not GitPython/pygit2: we already need the openssh client
for `git push`, and shelling out to `git` keeps the surface tiny and matches
the operator's mental model (every action you can see in `git reflog`).

https://github.com/modelcontextprotocol/servers-archived/tree/main/src/git 

"""

from __future__ import annotations

import asyncio
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


def canonical_service_name(
    service: str,
    settings: Settings | None = None,
    candidates: list[str] | None = None,
) -> str:
    """Return the patcher workspace repo name for a user-facing service id/name.

    Resolution order: an explicit PATCHER_SERVICE_ALIASES override (case-
    insensitive) wins; otherwise the name is auto-matched against the known
    service repos (`candidates`, defaulting to the materialized workspace) so
    case/separator/suffix differences need no config. An unresolved name is
    returned unchanged, leaving the clone path to report it against the real
    repo list.
    """
    s = settings or get_settings()
    requested = service.strip()
    _validate_service_name(requested)

    aliases = s.patcher_service_aliases
    by_lower = {k.lower(): v for k, v in aliases.items()}
    if requested in aliases:
        canonical = aliases[requested]
    elif requested.lower() in by_lower:
        canonical = by_lower[requested.lower()]
    else:
        known = candidates if candidates is not None else list_workspace_services(s)
        canonical = _match_candidate(requested, known) or requested

    _validate_service_name(canonical)
    return canonical


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
    canonical = canonical_service_name(service, s, candidates=candidates)
    sub = service_subpath(service, s)
    repo_root = (s.patcher_workspace_root / canonical).resolve()
    materialized = canonical in local
    return {
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


def service_root(service: str, settings: Settings | None = None) -> Path:
    """Return the repo working tree path for `service` (clone/push/deploy root)."""
    s = settings or get_settings()
    canonical = canonical_service_name(service, s)
    # Path method to resolve() the path
    return (s.patcher_workspace_root / canonical).resolve()


def service_subpath(service: str, settings: Settings | None = None) -> str:
    """Subdirectory inside the repo this service lives in ("" if it owns the repo).

    Keyed by the *requested* id (not the canonical repo name), so several ids that
    alias to one repo can each scope to their own subdir.
    """
    s = settings or get_settings()
    subs = dict(s.patcher_service_subpaths)
    subs.update({k.lower(): v for k, v in s.patcher_service_subpaths.items()})
    requested = service.strip()
    sub = subs.get(requested, subs.get(requested.lower(), ""))
    return sub.strip("/") if sub else ""


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
        # Prints raw rel_path.
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


# It is a "hidden error detector"

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


# This is because "push successful" doesn't guarantee that the remote contains exactly what
# we expect

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

# Verifies that the code has actually been deployed on the VM. It connects via SSH and checks the actual 
# working directory (the worktree), not just the repository

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



# It's the orchestrator. It performs the git push, then runs the following checks in sequence: 
# 1) was the push successful? 
# 2) did the hook report errors (via _remote_output_has_hook_error)? 
# 3) is the remote ref correct (_verify_remote_branch)? 
# 4) if there's a worktree, is the live code correct (_verify_remote_worktree)? 
# Only if everything passes does it return deployed: True. 
# In practice: "push the code and then prove to me that it's actually in production."

async def deploy(service: str, branch: str | None = None) -> dict:
    """`git push origin <branch>`. The VM's post-receive hook does the rebuild."""
    s = get_settings()
    requested_service = service
    # Normalise aliases to the real service name, but keep the caller's original
    # spelling so we can echo it back in the result.
    service = canonical_service_name(service, s)
    branch = branch or s.patcher_default_branch
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
    }

# Safely undo a deployment: Instead of rewriting history (force-push, risky), 
# create a new revert commit that undoes the changes, then# publishes it using the reuse deploy.

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

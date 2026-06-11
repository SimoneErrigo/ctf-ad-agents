#!/bin/sh
# init-vm.sh — provision the patcher deploy target for one or more services.
# Per service: create a bare repo, install a post-receive hook (checks the pushed
# commit out into the worktree and runs `docker compose up -d --build`), then seed
# and push the first commit (initial deploy).
# Run ON the machine that receives `git push` (the VM), or locally for testing.
# POSIX sh; macOS and Linux.
#
# Usage:
#   init-vm.sh [options] <service>=<source_dir> [<service>=<source_dir> ...]
#
# Options:
#   --remote-root DIR     where bare repos live (default: /srv/git)
#   --worktree-root DIR   if set, each source is COPIED to <root>/<service> and
#                         that copy is the deploy target; else deploy in place
#   --branch NAME         branch to deploy (default: main)
#   --exclude PATH        worktree-relative path kept across deploys, untracked
#                         (repeatable). e.g. --exclude db/data
#   --authorized-key FILE append a public key to ~/.ssh/authorized_keys (repeatable)
#   --force               recreate hook and re-seed even if it exists
#   -h, --help            show this help
#
# Examples:
#   init-vm.sh --remote-root ~/AD-demo/patcher-test/git --exclude db/data \
#              cc-forms=~/AD-demo/patcher-test/CCForms
#   init-vm.sh --remote-root /srv/git --worktree-root /srv/services \
#              --exclude db/data cc-forms=/home/ctf/src/CCForms rce=/home/ctf/src/rce
#
# Service name must match the patcher's canonical repo name (PATCHER_SERVICE_ALIASES)
# and the worktree path must match PATCHER_DEPLOY_WORKTREES so deploy() can verify it.

set -eu

# defaults
REMOTE_ROOT="/srv/git"
WORKTREE_ROOT=""
BRANCH="main"
FORCE=0
EXCLUDES=""    # space-separated worktree-relative paths
AUTH_KEYS=""   # newline-separated key files
SERVICES=""    # newline-separated "name=source" specs
OLDIFS=$IFS

# helpers
log()  { printf '[init-vm] %s\n' "$*"; }
warn() { printf '[init-vm] WARN: %s\n' "$*" >&2; }
die()  { printf '[init-vm] ERROR: %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,30p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'; }

# Expand a leading ~ even when the arg was quoted at the call site.
expand_tilde() {
  case "$1" in
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s\n' "$HOME/${1#~/}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

# Portable realpath: no readlink -f / realpath (absent on stock macOS).
abspath() {
  if [ -d "$1" ]; then
    ( CDPATH= cd -- "$1" && pwd )
  else
    _d=$(dirname -- "$1"); _b=$(basename -- "$1")
    ( CDPATH= cd -- "$_d" && printf '%s/%s\n' "$(pwd)" "$_b" )
  fi
}

# POSIX single-quote a value for safe embedding in the generated hook.
shq() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

# arg parsing
while [ $# -gt 0 ]; do
  case "$1" in
    --remote-root)    [ $# -ge 2 ] || die "--remote-root needs a value"; REMOTE_ROOT=$2; shift 2 ;;
    --worktree-root)  [ $# -ge 2 ] || die "--worktree-root needs a value"; WORKTREE_ROOT=$2; shift 2 ;;
    --branch)         [ $# -ge 2 ] || die "--branch needs a value"; BRANCH=$2; shift 2 ;;
    --exclude)        [ $# -ge 2 ] || die "--exclude needs a value"; EXCLUDES="$EXCLUDES $2"; shift 2 ;;
    --authorized-key) [ $# -ge 2 ] || die "--authorized-key needs a value"; AUTH_KEYS="$AUTH_KEYS
$2"; shift 2 ;;
    --force)          FORCE=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    --)               shift; while [ $# -gt 0 ]; do SERVICES="$SERVICES
$1"; shift; done ;;
    --*)              die "unknown option: $1" ;;
    *)                SERVICES="$SERVICES
$1"; shift ;;
  esac
done

command -v git >/dev/null 2>&1 || die "git is required"
command -v docker >/dev/null 2>&1 || warn "docker not found on this machine's PATH; the deploy step will fail until docker is installed"
[ -n "$SERVICES" ] || { usage; die "no services given"; }

# build exclude pathspec (for the hook) and plain list (for .gitignore)
EXCLUDE_PATHSPEC=""
EXCLUDE_PLAIN=""
for e in $EXCLUDES; do
  EXCLUDE_PATHSPEC="$EXCLUDE_PATHSPEC :(exclude)$e"
  EXCLUDE_PLAIN="$EXCLUDE_PLAIN $e"
done
EXCLUDE_PATHSPEC=$(printf '%s' "$EXCLUDE_PATHSPEC" | sed 's/^ //')

# optional: install authorized keys
if [ -n "$AUTH_KEYS" ]; then
  mkdir -p "$HOME/.ssh"; chmod 700 "$HOME/.ssh"
  touch "$HOME/.ssh/authorized_keys"; chmod 600 "$HOME/.ssh/authorized_keys"
  IFS='
'
  for kf in $AUTH_KEYS; do
    IFS=$OLDIFS
    [ -n "$kf" ] || continue
    kf=$(expand_tilde "$kf")
    [ -f "$kf" ] || die "authorized key file not found: $kf"
    while IFS= read -r line; do
      [ -n "$line" ] || continue
      grep -qxF "$line" "$HOME/.ssh/authorized_keys" 2>/dev/null \
        || printf '%s\n' "$line" >> "$HOME/.ssh/authorized_keys"
    done < "$kf"
    log "added key(s) from $kf to $HOME/.ssh/authorized_keys"
  done
  IFS=$OLDIFS
fi

# resolve roots
REMOTE_ROOT=$(expand_tilde "$REMOTE_ROOT")
mkdir -p "$REMOTE_ROOT"
REMOTE_ROOT=$(abspath "$REMOTE_ROOT")
if [ -n "$WORKTREE_ROOT" ]; then
  WORKTREE_ROOT=$(expand_tilde "$WORKTREE_ROOT")
  mkdir -p "$WORKTREE_ROOT"
  WORKTREE_ROOT=$(abspath "$WORKTREE_ROOT")
fi

# per-service work
make_bare() {  # $1=bare  $2=branch
  if [ -d "$1" ] && [ "$FORCE" -eq 0 ]; then
    log "  bare repo exists: $1 (refreshing hook; use --force to re-seed)"
  else
    mkdir -p "$1"
    git init -q --bare "$1"
  fi
  git --git-dir="$1" symbolic-ref HEAD "refs/heads/$2" 2>/dev/null || true
}

write_hook() {  # $1=bare  $2=worktree  $3=branch  $4=exclude-pathspec
  hook="$1/hooks/post-receive"
  {
    echo '#!/bin/sh'
    echo '# Auto-generated by scripts/init-vm.sh. Regenerate with that script; do not hand-edit.'
    echo 'set -eu'
    echo ''
    echo '# post-receive runs with a minimal PATH that often lacks Docker; add the usual'
    echo '# bin dirs so `docker compose` is found (else the push succeeds but never rebuilds).'
    echo 'export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:/snap/bin:$PATH"'
    echo ''
    printf 'WORK=%s\n' "$(shq "$2")"
    printf 'BRANCH=%s\n' "$(shq "$3")"
    echo 'HOOK_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)'
    echo 'BARE_GIT_DIR=$(CDPATH= cd -- "$HOOK_DIR/.." && pwd)'
    echo ''
    echo '# Clear hook-set GIT_DIR/GIT_WORK_TREE so git below acts on the worktree, not the bare repo.'
    echo 'unset GIT_DIR GIT_WORK_TREE'
    echo ''
    echo '# Deploy pathspec; `:(exclude)<path>` keeps runtime dirs across deploys. Unquoted: split below.'
    printf 'PATHSPEC=". %s"\n' "$4"
    echo ''
    echo 'if [ -d "$WORK/.git" ]; then'
    echo '  NEW_HEAD=$(git --git-dir="$BARE_GIT_DIR" rev-parse "$BRANCH")'
    echo '  git -C "$WORK" fetch -q "$BARE_GIT_DIR" "$BRANCH"'
    echo '  git -C "$WORK" update-ref "refs/heads/$BRANCH" "$NEW_HEAD"'
    echo '  git -C "$WORK" update-ref "refs/remotes/origin/$BRANCH" "$NEW_HEAD"'
    echo '  # shellcheck disable=SC2086'
    echo '  git -C "$WORK" checkout -f HEAD -- $PATHSPEC'
    echo '  # shellcheck disable=SC2086'
    echo '  git -C "$WORK" reset -q HEAD -- $PATHSPEC'
    echo 'else'
    echo '  # shellcheck disable=SC2086'
    echo '  git --work-tree="$WORK" --git-dir="$BARE_GIT_DIR" checkout -f "$BRANCH" -- $PATHSPEC'
    echo 'fi'
    echo ''
    echo '# Rebuild, failing loudly: a post-receive hook cannot reject the push, so the only'
    echo '# failure signal is this stderr text. DEPLOY-FAILED is the sentinel'
    echo '# git_functions._remote_output_has_hook_error scans for.'
    echo 'command -v docker >/dev/null 2>&1 || { echo "post-receive: DEPLOY-FAILED: docker not found (PATH=$PATH)" >&2; exit 1; }'
    echo 'cd "$WORK" || { echo "post-receive: DEPLOY-FAILED: cannot cd $WORK" >&2; exit 1; }'
    echo ''
    echo '# macOS + Docker Desktop: the global credsStore ("desktop") resolves every'
    echo '# registry pull through docker-credential-desktop, which reads the login'
    echo '# keychain. This hook runs in the non-interactive ssh session opened by the'
    echo '# push, where the keychain is unreachable ("user interaction is not allowed")'
    echo '# and the build dies. Point docker at a throwaway config with no credsStore so'
    echo '# public base images pull anonymously. (Private registries would need creds'
    echo '# materialised here instead.) cliPluginsExtraDirs keeps `docker compose` and'
    echo '# buildx resolvable: Docker Desktop installs them under ~/.docker/cli-plugins,'
    echo '# which is no longer searched once DOCKER_CONFIG is redirected.'
    echo 'DOCKER_CONFIG="${TMPDIR:-/tmp}/patcher-docker-config"'
    echo 'mkdir -p "$DOCKER_CONFIG"'
    echo 'cat > "$DOCKER_CONFIG/config.json" <<JSON'
    echo '{ "cliPluginsExtraDirs": ["$HOME/.docker/cli-plugins"] }'
    echo 'JSON'
    echo 'export DOCKER_CONFIG'
    echo ''
    echo 'if docker compose up -d --build; then'
    echo '  echo "post-receive: deploy OK ($WORK @ $(git --git-dir="$BARE_GIT_DIR" rev-parse --short "$BRANCH"))"'
    echo 'else'
    echo '  rc=$?'
    echo '  echo "post-receive: DEPLOY-FAILED: docker compose exited $rc" >&2'
    echo '  exit "$rc"'
    echo 'fi'
  } > "$hook"
  chmod +x "$hook"
}

seed_service() {  # $1=worktree  $2=bare  $3=branch  $4=plain excludes
  wt=$1; bare=$2; br=$3; excl=$4

  # Keep runtime dirs out of git: gitignore them and untrack any already tracked.
  for e in $excl; do
    [ -n "$e" ] || continue
    grep -qxF "$e" "$wt/.gitignore" 2>/dev/null || printf '%s\n' "$e" >> "$wt/.gitignore"
  done

  if [ ! -d "$wt/.git" ]; then
    git -C "$wt" init -q
    git -C "$wt" symbolic-ref HEAD "refs/heads/$br"
  fi
  git -C "$wt" config user.name  >/dev/null 2>&1 || git -C "$wt" config user.name  "patch-agent"
  git -C "$wt" config user.email >/dev/null 2>&1 || git -C "$wt" config user.email "patch-agent@ctf-ad-agents.local"

  for e in $excl; do
    [ -n "$e" ] || continue
    git -C "$wt" rm -r --cached --quiet --ignore-unmatch "$e" >/dev/null 2>&1 || true
  done

  if git -C "$wt" remote get-url origin >/dev/null 2>&1; then
    git -C "$wt" remote set-url origin "$bare"
  else
    git -C "$wt" remote add origin "$bare"
  fi

  git -C "$wt" add -A
  if git -C "$wt" diff --cached --quiet 2>/dev/null && git -C "$wt" rev-parse HEAD >/dev/null 2>&1; then
    log "  nothing new to seed (worktree already committed)"
  else
    git -C "$wt" commit -q -m "seed: initial import of $(basename "$wt")"
  fi

  log "  pushing $br -> $bare (triggers post-receive deploy)"
  if out=$(git -C "$wt" push origin "$br" 2>&1); then
    printf '%s\n' "$out" | sed 's/^/    /'
    case "$out" in
      *DEPLOY-FAILED*|*"command not found"*|*"Cannot connect to the Docker daemon"*)
        warn "the post-receive hook reported a deploy failure (see push output above)" ;;
    esac
  else
    printf '%s\n' "$out" | sed 's/^/    /'
    die "git push failed for $(basename "$wt")"
  fi
}

IFS='
'
for spec in $SERVICES; do
  IFS=$OLDIFS
  [ -n "$spec" ] || continue
  case "$spec" in
    *=*) ;;
    *) die "bad service spec '$spec' (want name=source_dir)" ;;
  esac
  name=${spec%%=*}
  src=${spec#*=}
  [ -n "$name" ] || die "empty service name in spec '$spec'"
  src=$(expand_tilde "$src")
  [ -d "$src" ] || die "source dir for '$name' not found: $src"
  src=$(abspath "$src")

  log "service: $name"
  bare="$REMOTE_ROOT/$name.git"

  if [ -n "$WORKTREE_ROOT" ]; then
    wt="$WORKTREE_ROOT/$name"
    if [ -d "$wt" ] && [ "$FORCE" -eq 0 ]; then
      log "  worktree exists: $wt (keeping; use --force to overwrite from source)"
    else
      log "  copying $src -> $wt"
      mkdir -p "$wt"
      # tar pipe: portable across bsdtar (macOS) / GNU tar (Linux), skips source .git/.
      ( cd "$src" && tar cf - --exclude='./.git' . ) | ( cd "$wt" && tar xf - )
    fi
  else
    wt="$src"
    log "  deploying in place: $wt"
  fi

  make_bare "$bare" "$BRANCH"
  write_hook "$bare" "$wt" "$BRANCH" "$EXCLUDE_PATHSPEC"
  log "  hook installed: $bare/hooks/post-receive"
  seed_service "$wt" "$bare" "$BRANCH" "$EXCLUDE_PLAIN"
  log "  done: $name (bare=$bare worktree=$wt)"
  IFS='
'
done
IFS=$OLDIFS

log "all services provisioned."

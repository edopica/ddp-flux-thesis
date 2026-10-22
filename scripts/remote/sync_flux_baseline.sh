#!/usr/bin/env bash
# =============================================================================
# Sync the historical Flux static-DDP worktree to a remote host.
#
# Usage:
#   scripts/remote/sync_flux_baseline.sh <remote_host>
#
# Defaults:
#   source: ../flux-static-baseline (historical Flux at 8add0e42)
#   target: ~/projects/flux-static-baseline
#
# The source must be a git worktree at EXPECTED_COMMIT and must be clean.
# Override EXPECTED_COMMIT or ALLOW_DIRTY_SOURCE=1 only for a documented
# experiment. This helper never resolves dependencies and never precompiles.
#
# Remote-resolved Manifest.toml files are preserved:
#   --exclude=Manifest.toml        (root and any other Manifest.toml)
#   --exclude=test/Manifest.toml   (explicit, same protection)
# Excluded paths are protected from --delete. The target is not a git
# worktree because .git is excluded, so remote uncommitted changes cannot be
# detected automatically. Verify the target before syncing.
#
# After this script, copy the matching test manifest manually if it changed,
# then precompile on a compute node:
#   FLUX_DIR=~/projects/flux-static-baseline \
#     bash ~/projects/ddp-flux-thesis/scripts/remote/leonardo/precompile-phase3.sh
# =============================================================================
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <remote_host>"
    echo "Example: $0 leonardo"
    exit 1
fi

REMOTE_HOST=$1
REMOTE_PROJECT_DIR="~/projects"
SOURCE_DIR=${SOURCE_DIR:-../flux-static-baseline}
REMOTE_TARGET=${REMOTE_TARGET:-flux-static-baseline}
EXPECTED_COMMIT=${EXPECTED_COMMIT:-8add0e42b7d9388ff365335152126e5b1e9de657}
ALLOW_DIRTY_SOURCE=${ALLOW_DIRTY_SOURCE:-0}

case "$REMOTE_HOST" in
    *[!A-Za-z0-9._-]*|"")
        echo "ERROR: invalid remote host name '$REMOTE_HOST'." >&2
        exit 1
        ;;
esac

HOST_CONF="scripts/remote/hosts/${REMOTE_HOST}.conf"
if [ ! -f "$HOST_CONF" ]; then
    echo "ERROR: unknown remote host '$REMOTE_HOST' (missing $HOST_CONF)." >&2
    echo "Supported hosts: $(cd scripts/remote/hosts && ls *.conf 2>/dev/null | sed 's/\.conf$//' | tr '\n' ' ')" >&2
    exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: historical Flux worktree not found: $SOURCE_DIR" >&2
    exit 1
fi

if ! git -C "$SOURCE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: source is not a git worktree: $SOURCE_DIR" >&2
    exit 1
fi

SOURCE_COMMIT=$(git -C "$SOURCE_DIR" rev-parse HEAD)
if [ "$SOURCE_COMMIT" != "$EXPECTED_COMMIT" ]; then
    echo "ERROR: source commit mismatch: $SOURCE_COMMIT" >&2
    echo "       expected $EXPECTED_COMMIT" >&2
    echo "       set EXPECTED_COMMIT or check out the historical worktree" >&2
    exit 1
fi

SOURCE_DIRTY=$(git -C "$SOURCE_DIR" status --porcelain)
if [ -n "$SOURCE_DIRTY" ]; then
    if [ "$ALLOW_DIRTY_SOURCE" != 1 ]; then
        echo "ERROR: source worktree has uncommitted changes: $SOURCE_DIR" >&2
        printf '%s\n' "$SOURCE_DIRTY" >&2
        echo "       commit/stash them or set ALLOW_DIRTY_SOURCE=1 on purpose" >&2
        exit 1
    fi
    echo "WARNING: ALLOW_DIRTY_SOURCE=1: syncing a dirty worktree" >&2
    printf '%s\n' "$SOURCE_DIRTY" >&2
fi

cat >&2 <<'EOF'
==============================================================================
WARNING: sync_flux_baseline.sh targets the HISTORICAL Flux worktree, not
         ../flux-integration (current) and not ddp_flux.
         Remote Manifest.toml files are preserved and excluded paths survive
         --delete. Uncommitted files on the remote target can still be pruned:
         the remote target is not a git worktree because .git is excluded.
         Verify the remote target before syncing.
==============================================================================
EOF

echo "Syncing $SOURCE_DIR ($SOURCE_COMMIT) to $REMOTE_HOST:$REMOTE_PROJECT_DIR/$REMOTE_TARGET ..."

ssh "$REMOTE_HOST" "mkdir -p $REMOTE_PROJECT_DIR/$REMOTE_TARGET"

rsync -avz --delete \
    --exclude='.git' \
    --exclude='.julia' \
    --exclude='Manifest.toml' \
    --exclude='test/Manifest.toml' \
    "$SOURCE_DIR/" "$REMOTE_HOST:$REMOTE_PROJECT_DIR/$REMOTE_TARGET/"

echo "Sync complete: $REMOTE_HOST:$REMOTE_PROJECT_DIR/$REMOTE_TARGET"
echo "Preserved on the remote target: Manifest.toml and test/Manifest.toml"
echo
echo "NEXT 1: copy the matching test manifest manually (excluded from rsync)."
if [ -f "$SOURCE_DIR/test/Manifest.toml" ]; then
    echo "  rsync -avz '$SOURCE_DIR/test/Manifest.toml' \\"
    echo "    '$REMOTE_HOST:$REMOTE_PROJECT_DIR/$REMOTE_TARGET/test/Manifest.toml'"
else
    echo "  WARNING: no local $SOURCE_DIR/test/Manifest.toml; the remote manifest is preserved as-is."
fi
echo
echo "NEXT 2: precompile on a compute node (never on a login node):"
echo "  FLUX_DIR=\$HOME/projects/$REMOTE_TARGET \\"
echo "    bash \$HOME/projects/ddp-flux-thesis/scripts/remote/leonardo/precompile-phase3.sh"
echo "  On non-Leonardo hosts use the matching scripts/remote/precompile.sh wrapper."

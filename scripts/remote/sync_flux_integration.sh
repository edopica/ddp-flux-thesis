#!/usr/bin/env bash
# =============================================================================
# Sync the separate Flux integration worktree to a remote host.
#
# Usage:
#   scripts/remote/sync_flux_integration.sh <remote_host>
#
# WARNING: this is NOT scripts/remote/sync_code.sh. sync_code.sh targets the
# frozen ddp_flux worktree; Phase 3/4 execute against ../flux-integration.
# Only this helper syncs flux-integration, and no other script calls it
# automatically.
#
# Remote-resolved Manifest.toml files are preserved:
#   --exclude=Manifest.toml        (root and any other Manifest.toml)
#   --exclude=test/Manifest.toml   (explicit, same protection)
# Excluded paths are protected from --delete.
#
# After this script, resolve dependencies if Project.toml changed and
# precompile on the machine that will run the tests:
#   scripts/remote/setup_node.sh <host>
#   scripts/remote/precompile.sh <host>
# =============================================================================
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <remote_host>"
    echo "Example: $0 leonardo"
    exit 1
fi

REMOTE_HOST=$1
REMOTE_PROJECT_DIR="~/projects"
SOURCE_DIR="../flux-integration"

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
    echo "ERROR: flux-integration worktree not found: $SOURCE_DIR" >&2
    exit 1
fi

cat >&2 <<'EOF'
==============================================================================
WARNING: sync_flux_integration.sh does NOT use sync_code.sh's ddp_flux target.
         It syncs ../flux-integration (the Phase 3/4 Flux worktree) instead.
         Remote Manifest.toml files are preserved and excluded paths survive
         --delete.
==============================================================================
EOF

echo "Syncing $SOURCE_DIR to $REMOTE_HOST:$REMOTE_PROJECT_DIR/ ..."

# Ensure the remote project directory exists.
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_PROJECT_DIR"

rsync -avz --delete \
    --exclude='.git' \
    --exclude='.julia' \
    --exclude='Manifest.toml' \
    --exclude='test/Manifest.toml' \
    "$SOURCE_DIR" "$REMOTE_HOST:$REMOTE_PROJECT_DIR/"

echo "Sync complete: $REMOTE_HOST:$REMOTE_PROJECT_DIR/flux-integration"
echo "NEXT: precompile on the run host (compute node on Slurm clusters)."

#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <remote_host>"
    echo "Example: $0 bocconi"
    exit 1
fi

REMOTE_HOST=$1
REMOTE_PROJECT_DIR="~/projects"

echo "Syncing to $REMOTE_HOST:$REMOTE_PROJECT_DIR..."

# Ensure the remote directory exists
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_PROJECT_DIR"

# Sync ddp_flux (the Flux fork). Root Manifest.toml is synced on purpose
# (wiki/devlog/20260716-c6b-hpc-tests.md); test/Manifest.toml is protected so
# the remote-resolved test environment survives. --delete prunes files removed
# locally but never touches excluded paths.
echo "Syncing ddp_flux..."
rsync -avz --delete --exclude='.git' --exclude='.julia' --exclude='test/Manifest.toml' \
    ../ddp_flux "$REMOTE_HOST:$REMOTE_PROJECT_DIR/"

# Sync ddp-flux-thesis (the thesis workspace). .rsyncignore excludes .git,
# .julia, Manifest.toml, and local-only directories; excluded paths are
# protected from --delete.
echo "Syncing ddp-flux-thesis..."
rsync -avz --delete --exclude-from='.rsyncignore' \
    . "$REMOTE_HOST:$REMOTE_PROJECT_DIR/ddp-flux-thesis/"

echo "Sync complete!"

#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <remote_host>"
    echo "Example: $0 hpc"
    exit 1
fi

REMOTE_HOST=$1
REMOTE_PROJECT_DIR="~/projects"

echo "Syncing to $REMOTE_HOST:$REMOTE_PROJECT_DIR..."

# Ensure the remote directory exists
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_PROJECT_DIR"

# Sync ddp_flux (the Flux fork)
echo "Syncing ddp_flux..."
rsync -avz --exclude='.git' --exclude='.julia' \
    ../ddp_flux "$REMOTE_HOST:$REMOTE_PROJECT_DIR/"

# Sync ddp-flux-thesis (the thesis workspace)
echo "Syncing ddp-flux-thesis..."
rsync -avz --exclude-from='.rsyncignore' \
    . "$REMOTE_HOST:$REMOTE_PROJECT_DIR/ddp-flux-thesis/"

echo "Sync complete!"

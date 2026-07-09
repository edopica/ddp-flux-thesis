#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <remote_host>"
    echo "Example: $0 hpc"
    exit 1
fi

REMOTE_HOST=$1
THESIS_DIR="~/projects/ddp-flux-thesis"

echo "Setting up environment on $REMOTE_HOST..."

ssh "$REMOTE_HOST" "cd $THESIS_DIR && bash -l -c \"make env FLUX_REPO_PATH=../ddp_flux\""

echo "Environment setup complete!"

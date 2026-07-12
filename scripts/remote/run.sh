#!/bin/bash
set -e

if [ -z "$2" ]; then
    echo "Usage: $0 <remote_host> <command...>"
    echo "Example: $0 hpc make check"
    exit 1
fi

REMOTE_HOST=$1
shift
COMMAND="$@"
THESIS_DIR="~/projects/ddp-flux-thesis"

# Load remote-specific configuration if it exists
WRAPPER=""
if [ -f "scripts/remote/hosts/${REMOTE_HOST}.conf" ]; then
    source "scripts/remote/hosts/${REMOTE_HOST}.conf"
fi

FULL_COMMAND="$WRAPPER $COMMAND"

echo "Executing on $REMOTE_HOST: $FULL_COMMAND"

ssh "$REMOTE_HOST" "cd $THESIS_DIR && bash -l -c \"$FULL_COMMAND\""

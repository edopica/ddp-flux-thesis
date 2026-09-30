#!/bin/bash
set -e

if [ -z "$2" ]; then
    echo "Usage: $0 <remote_host> <command...>"
    echo "Example: $0 bocconi make check"
    exit 1
fi

REMOTE_HOST=$1
shift
COMMAND="$@"
THESIS_DIR="~/projects/ddp-flux-thesis"

case "$REMOTE_HOST" in
    *[!A-Za-z0-9._-]*|"")
        echo "ERROR: invalid remote host name '$REMOTE_HOST'." >&2
        exit 1
        ;;
esac

# Every supported host has a configuration file in scripts/remote/hosts/.
# Reject unknown names: without a conf the command would run unwrapped on a
# cluster login node instead of a compute node.
HOST_CONF="scripts/remote/hosts/${REMOTE_HOST}.conf"
if [ ! -f "$HOST_CONF" ]; then
    echo "ERROR: unknown remote host '$REMOTE_HOST' (missing $HOST_CONF)." >&2
    echo "Supported hosts: $(cd scripts/remote/hosts && ls *.conf 2>/dev/null | sed 's/\.conf$//' | tr '\n' ' ')" >&2
    exit 1
fi

WRAPPER=""
source "$HOST_CONF"

FULL_COMMAND="$WRAPPER $COMMAND"

echo "Executing on $REMOTE_HOST: $FULL_COMMAND"

ssh "$REMOTE_HOST" "cd $THESIS_DIR && bash -l -c \"$FULL_COMMAND\""

#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <remote_host>"
    echo "Example: $0 hpc"
    exit 1
fi

REMOTE_HOST=$1

echo "Triggering precompilation on $REMOTE_HOST with 8 CPUs..."
NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh "$REMOTE_HOST" make precompile

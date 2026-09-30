#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <remote_host>"
    echo "Example: $0 bocconi"
    exit 1
fi

REMOTE_HOST=$1
THESIS_DIR="~/projects/ddp-flux-thesis"

echo "Setting up environment on $REMOTE_HOST..."
echo "NOTE: 'make env' now only resolves dependencies (no precompilation)."
echo "Follow up with: scripts/remote/precompile.sh $REMOTE_HOST (compiles on a compute node)."

ssh "$REMOTE_HOST" "cd $THESIS_DIR && bash -l -c \"make env FLUX_REPO_PATH=../ddp_flux && make install-mpiexec\""

echo "Environment setup complete!"

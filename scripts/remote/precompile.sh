#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <remote_host>"
    echo "Example: $0 hpc"
    exit 1
fi

REMOTE_HOST=$1

# Precompile BOTH project envs (thesis + Flux test) on a COMPUTE node via srun.
# NEVER precompile on the login node: julia caches bake the host CPU feature
# set (login = graniterapids, compute = icelake-server), so login compiles are
# unusable on compute nodes and vice versa. Any compute node works (gnode01 and
# gnode02 are flag-identical; caches are reused across both).
echo "Triggering precompilation of thesis env + Flux test env on $REMOTE_HOST (compute node, 8 CPUs)..."
NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh "$REMOTE_HOST" "make precompile-all FLUX_REPO_PATH=../ddp_flux"

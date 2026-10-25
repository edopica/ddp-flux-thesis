#!/usr/bin/env bash
# Precompile the Flux test project on a Leonardo compute node.
# Must run inside the caller's srun/salloc allocation (never on the login node).
set -euo pipefail

WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
JULIA_BIN=${JULIA_BIN:-$WORK/epicazio/julia-1.12.6/bin/julia}
JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$WORK/epicazio/julia-depot}
FLUX_DIR=${FLUX_DIR:-$HOME/projects/flux-integration}

export JULIA_DEPOT_PATH
export JULIA_PKG_PRECOMPILE_AUTO=1
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}

host=$(hostname)
case "$host" in
  login*) echo "ERROR: precompile-phase3.sh must run on a compute node (host=$host)" >&2; exit 1 ;;
esac

cd "$FLUX_DIR"
echo "PREcompile host=$(hostname) pwd=$(pwd) julia=$JULIA_BIN"
echo "JULIA_PROJECT=${JULIA_PROJECT:-<unset>} JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"

"$JULIA_BIN" --startup-file=no --project="$FLUX_DIR/test" -e '
println("ACTIVE_PROJECT=", Base.active_project())
using Pkg
Pkg.precompile()
using Flux, CUDA, cuDNN
println(:FLUX_PRECOMPILE_OK)
'

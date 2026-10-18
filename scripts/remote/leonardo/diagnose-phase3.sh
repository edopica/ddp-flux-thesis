#!/usr/bin/env bash
# Forward-only Phase 3 diagnostic on a Leonardo compute node.
set -euo pipefail
WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
JULIA_BIN=${JULIA_BIN:-$WORK/epicazio/julia-1.12.6/bin/julia}
JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$WORK/epicazio/julia-depot}
FLUX_DIR=${FLUX_DIR:-$HOME/projects/flux-integration}
export JULIA_DEPOT_PATH
export JULIA_LOAD_PATH="$THESIS/scripts/static_scaling/julia-env:$FLUX_DIR/test:@stdlib"
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_THREADS=1
exec "$JULIA_BIN" --startup-file=no "$THESIS/scripts/static_scaling/diagnose_phase3_flux.jl" "$@"

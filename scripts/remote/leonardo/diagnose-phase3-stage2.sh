#!/usr/bin/env bash
set -euo pipefail
WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
VENV_PY=${VENV_PY:-$WORK/epicazio/venvs/pytorch-scaling/bin/python}
JULIA_BIN=${JULIA_BIN:-$WORK/epicazio/julia-1.12.6/bin/julia}
JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$WORK/epicazio/julia-depot}
FLUX_DIR=${FLUX_DIR:-$HOME/projects/flux-integration}
FIX=$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2
REF=$THESIS/artifacts/baselines/static-scaling/2026-10-17-phase3/phase3_reference.npz
DIAG=${DIAG:-$THESIS/artifacts/logs/static-scaling/2026-10-17-phase3/stage2_diag.npz}
export JULIA_DEPOT_PATH
export JULIA_LOAD_PATH="$THESIS/scripts/static_scaling/julia-env:$FLUX_DIR/test:@stdlib"
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_THREADS=1
mkdir -p "$(dirname "$DIAG")"
"$VENV_PY" "$THESIS/scripts/static_scaling/diagnose_phase3_torch_stage2.py" \
  --fixture-dir "$FIX" --bundle "$FIX/initial_bundle.npz" --reference "$REF" \
  --out "$DIAG" --device cuda
"$JULIA_BIN" --startup-file=no "$THESIS/scripts/static_scaling/diagnose_phase3_flux_stage2.jl" \
  --fixture-dir "$FIX" --bundle "$FIX/initial_bundle.npz" --reference "$REF" \
  --diag "$DIAG" --device cuda

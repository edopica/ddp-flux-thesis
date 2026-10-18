#!/usr/bin/env bash
# =============================================================================
# Leonardo (CINECA) COMPUTE-NODE acceptance check for the Phase 2 artifacts
# produced by run-phase2-prep.sh.
#
# Steps:
#   1/2 check_dataset.py        (NumPy driver: prepared dataset, schedule,
#                                stats, fixtures, bundle, manifests)
#   2/2 check_fixtures_data.jl  (frozen Julia data check, NPZ-only project)
#
# ALLOCATION (from the thesis root on a login node):
#   bash scripts/remote/run.sh leonardo \
#     "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis \
#      bash scripts/remote/leonardo/run-phase2-check.sh"
#
# Prerequisite (once, login node; plain Pkg operation, no precompile):
#   ssh leonardo 'JULIA_DEPOT_PATH=/leonardo_work/IscrC_UNMASKED/epicazio/julia-depot \
#     JULIA_PKG_PRECOMPILE_AUTO=0 \
#     /leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin/julia \
#     --startup-file=no \
#     --project=\$HOME/projects/ddp-flux-thesis/scripts/static_scaling/julia-env \
#     -e "using Pkg; Pkg.instantiate()"'
#
# Evidence log: $LOG_DIR/remote-check.log (append; scp it back afterwards).
# Exit code: 0 only if both steps and their PHASE2_*_PASS summary lines pass.
# =============================================================================
set -euo pipefail

WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
VENV=${VENV:-$WORK/epicazio/venvs/pytorch-scaling}
VENV_PY=${VENV_PY:-$VENV/bin/python}
PREP=${PREP:-$WORK/epicazio/static-scaling-prepared}
FIX=${FIX:-$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2}
LOG_DIR=${LOG_DIR:-$THESIS/artifacts/logs/static-scaling/2026-10-16-phase2}
ALLOW_LOGIN_NODE=${ALLOW_LOGIN_NODE:-0}

JULIA_PREFIX=${JULIA_PREFIX:-$WORK/epicazio/julia-1.12.6}
JULIA_BIN=${JULIA_BIN:-$JULIA_PREFIX/bin/julia}
JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$WORK/epicazio/julia-depot}
JULIA_ENV=$THESIS/scripts/static_scaling/julia-env

LOG_FILE=$LOG_DIR/remote-check.log
SCRIPTS=$THESIS/scripts/static_scaling

die() { echo "ERROR: $*" >&2; exit 1; }

# ---- compute-node guard -----------------------------------------------------
host=$(hostname)
case "$host" in
  login*)
    if [ "$ALLOW_LOGIN_NODE" != 1 ]; then
      die "running on login node '$host': run under srun on a compute node (or set ALLOW_LOGIN_NODE=1 to override on purpose)"
    fi
    ;;
esac

# ---- environment checks ------------------------------------------------------
[ -x "$VENV_PY" ] || die "venv python not found: $VENV_PY"
[ -f "$PREP/initial_bundle.npz" ] || die "bundle not found: $PREP/initial_bundle.npz (run run-phase2-prep.sh first)"
[ -f "$FIX/fixture_manifest.json" ] || die "fixtures not found: $FIX/fixture_manifest.json (run run-phase2-prep.sh first)"
[ -f "$JULIA_ENV/Manifest.toml" ] || die "julia-env not instantiated: $JULIA_ENV (sync first, then see the instantiate command in the header)"
[ -x "$JULIA_BIN" ] || die "julia not found: $JULIA_BIN"

mkdir -p "$LOG_DIR"

run() {
  echo "=== run-phase2-check: START ==="
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>} SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
  echo "PREP=$PREP"
  echo "FIX=$FIX"
  echo "JULIA_BIN=$JULIA_BIN JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"
  echo "--- versions ---"
  "$VENV_PY" -c "import sys, numpy, PIL, torch; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'pillow', PIL.__version__, 'torch', torch.__version__)"
  "$JULIA_BIN" --version

  py_status=0
  jl_status=0

  echo "=== 1/2 check_dataset.py ==="
  time "$VENV_PY" "$SCRIPTS/check_dataset.py" \
    --prep-dir "$PREP" --fixture-dir "$FIX" \
    --bundle "$PREP/initial_bundle.npz" \
    --bundle-manifest "$PREP/initial_bundle.manifest.json" || py_status=$?

  echo "=== 2/2 check_fixtures_data.jl ==="
  export JULIA_DEPOT_PATH
  time "$JULIA_BIN" --startup-file=no \
    --project="$JULIA_ENV" "$SCRIPTS/check_fixtures_data.jl" \
    --fixture-dir "$FIX" \
    --bundle "$PREP/initial_bundle.npz" \
    --bundle-manifest "$PREP/initial_bundle.manifest.json" \
    --prep-dir "$PREP" || jl_status=$?

  echo "=== summary ==="
  echo "python_check_exit=$py_status julia_check_exit=$jl_status"
  if [ "$py_status" -eq 0 ] && [ "$jl_status" -eq 0 ]; then
    echo "PHASE2_REMOTE_CHECK PASS"
  else
    echo "PHASE2_REMOTE_CHECK FAIL"
  fi
  echo "NEXT (bring evidence home): scp leonardo:$LOG_FILE artifacts/logs/static-scaling/2026-10-16-phase2/"
  [ "$py_status" -eq 0 ] && [ "$jl_status" -eq 0 ]
}

set +e
run 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
echo "=== run-phase2-check: EXIT $status (log: $LOG_FILE) ==="
exit "$status"

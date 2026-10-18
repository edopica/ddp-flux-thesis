#!/usr/bin/env bash
# =============================================================================
# Leonardo (CINECA) compute-node driver for the Phase 3 Step 4 optimizer probe.
#
# Three fresh processes, all CPU-only:
#   1/3 Flux probe          (Julia, Optimisers.jl)
#   2/3 PyTorch probe       (pinned benchmark venv, torch.optim.AdamW)
#   3/3 comparator          (Float64 arithmetic oracle + negative controls)
#
# The probe is independent of training code and does not change any
# tolerance, training schedule, or Phase 3 acceptance checker.
#
# ALLOCATION (from the thesis repo root on a login node):
#   NTASKS=1 CPUS_PER_TASK=8 GRES=none TIME=00:30:00 \
#     scripts/remote/run.sh leonardo \
#     "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis \
#      bash scripts/remote/leonardo/run-phase3-optimizer-probe.sh"
#
# The script never invokes srun/salloc and never precompiles.  The Julia
# runner uses the pinned thesis julia-env plus the synced Flux test project.
# =============================================================================
set -euo pipefail

WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
VENV_PY=${VENV_PY:-$WORK/epicazio/venvs/pytorch-scaling/bin/python}
JULIA_BIN=${JULIA_BIN:-$WORK/epicazio/julia-1.12.6/bin/julia}
FLUX_DIR=${FLUX_DIR:-$HOME/projects/flux-integration}
FIX=${FIX:-$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2}
BUNDLE=${BUNDLE:-$FIX/initial_bundle.npz}
ROOT=${ROOT:-$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-step4}
LOG_DIR=${LOG_DIR:-$ROOT/logs}
RUNNER=${RUNNER:-$THESIS/scripts/remote/leonardo/run-julia-phase3.sh}
PROBE_JL=$THESIS/experiments/static-scaling/probes/probe_optimizer_contract_cross.jl
PROBE_PY=$THESIS/experiments/static-scaling/probes/probe_optimizer_contract_cross.py
COMPARATOR=$THESIS/experiments/static-scaling/probes/compare_optimizer_contract.py
SOURCE_COMMIT=${SOURCE_COMMIT:-}
SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-}
FLUX_COMMIT=${FLUX_COMMIT:-}
FLUX_STATE_SHA256=${FLUX_STATE_SHA256:-}

die() { echo "ERROR: $*" >&2; exit 1; }

host=$(hostname)
case "$host" in
  login*)
    die "running on login node '$host': run under srun on a compute node"
    ;;
esac

[ -x "$VENV_PY" ] || die "venv python not found: $VENV_PY"
[ -x "$JULIA_BIN" ] || die "julia not found: $JULIA_BIN"
[ -f "$RUNNER" ] || die "Julia runner not found: $RUNNER"
[ -f "$PROBE_JL" ] || die "Flux probe not found: $PROBE_JL (sync first)"
[ -f "$PROBE_PY" ] || die "PyTorch probe not found: $PROBE_PY (sync first)"
[ -f "$COMPARATOR" ] || die "comparator not found: $COMPARATOR (sync first)"
[ -f "$BUNDLE" ] || die "bundle not found: $BUNDLE"
[ -d "$FLUX_DIR/test" ] || die "Flux test project not found: $FLUX_DIR/test"

mkdir -p "$ROOT" "$LOG_DIR"

echo "=== run-phase3-optimizer-probe: START ==="
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>} SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}"
echo "THESIS=$THESIS"
echo "FLUX_DIR=$FLUX_DIR"
echo "BUNDLE=$BUNDLE"
echo "ROOT=$ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SOURCE_COMMIT=${SOURCE_COMMIT:-<unset>} SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-<unset>}"
echo "FLUX_COMMIT=${FLUX_COMMIT:-<unset>} FLUX_STATE_SHA256=${FLUX_STATE_SHA256:-<unset>}"
echo "PROBE_JL=$PROBE_JL"
echo "PROBE_PY=$PROBE_PY"
echo "COMPARATOR=$COMPARATOR"

echo "--- versions ---"
"$VENV_PY" - <<'PY' || die "python/torch version probe failed"
import platform
import numpy
import torch
print("python", platform.python_version(), "numpy", numpy.__version__)
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
PY
"$JULIA_BIN" --version || die "julia --version failed"

st_flux=0
st_torch=0
st_cmp=0
pass_flux=0
pass_torch=0
pass_cmp=0

echo "--- 1/3 Flux optimizer probe ---"
set +e
bash "$RUNNER" "$PROBE_JL" --bundle "$BUNDLE" --out "$ROOT/probe_flux.json" \
  2>&1 | tee "$LOG_DIR/probe-flux.log"
st_flux=${PIPESTATUS[0]}
set -e
if [ "$st_flux" -eq 0 ] && grep -q '^PROBE_FLUX WROTE ' "$LOG_DIR/probe-flux.log"; then
  pass_flux=1
fi
echo "--- 1/3 exit=$st_flux pass=$pass_flux ---"

echo "--- 2/3 PyTorch optimizer probe ---"
set +e
"$VENV_PY" "$PROBE_PY" --bundle "$BUNDLE" --out "$ROOT/probe_torch.json" \
  2>&1 | tee "$LOG_DIR/probe-torch.log"
st_torch=${PIPESTATUS[0]}
set -e
if [ "$st_torch" -eq 0 ] && grep -q '^PROBE_TORCH WROTE ' "$LOG_DIR/probe-torch.log"; then
  pass_torch=1
fi
echo "--- 2/3 exit=$st_torch pass=$pass_torch ---"

echo "--- 3/3 comparator ---"
set +e
"$VENV_PY" "$COMPARATOR" \
  --flux "$ROOT/probe_flux.json" \
  --torch "$ROOT/probe_torch.json" \
  --out "$ROOT/comparison.json" \
  2>&1 | tee "$LOG_DIR/comparator.log"
st_cmp=${PIPESTATUS[0]}
set -e
if [ "$st_cmp" -eq 0 ] && grep -qx 'PHASE3_OPTIMIZER_PROBE PASS' "$LOG_DIR/comparator.log"; then
  pass_cmp=1
fi
echo "--- 3/3 exit=$st_cmp pass=$pass_cmp ---"

n_rejected=$(grep -c '^NEGATIVE_CONTROL .* REJECTED' "$LOG_DIR/comparator.log" || true)
n_not_rejected=$(grep -c '^NEGATIVE_CONTROL .* NOT_REJECTED' "$LOG_DIR/comparator.log" || true)

echo "=== summary ==="
echo "flux_exit=$st_flux flux_pass=$pass_flux"
echo "torch_exit=$st_torch torch_pass=$pass_torch"
echo "comparator_exit=$st_cmp comparator_pass=$pass_cmp"
echo "negative_controls_rejected=$n_rejected negative_controls_not_rejected=$n_not_rejected"

sha256sum \
  "$ROOT/probe_flux.json" "$ROOT/probe_flux_arrays.npz" \
  "$ROOT/probe_torch.json" "$ROOT/probe_torch_arrays.npz" \
  "$ROOT/comparison.json" "$LOG_DIR"/*.log \
  > "$LOG_DIR/SHA256SUMS"

if [ "$st_flux" -eq 0 ] && [ "$pass_flux" -eq 1 ] &&
   [ "$st_torch" -eq 0 ] && [ "$pass_torch" -eq 1 ] &&
   [ "$st_cmp" -eq 0 ] && [ "$pass_cmp" -eq 1 ] &&
   [ "$n_not_rejected" -eq 0 ] && [ "$n_rejected" -gt 0 ]; then
  echo "PHASE3_OPTIMIZER_PROBE_DRIVER PASS"
  echo "NEXT (bring evidence home): rsync -avz leonardo:$ROOT/ artifacts/logs/static-scaling/2026-10-17-phase3-step4/"
  exit 0
fi

echo "PHASE3_OPTIMIZER_PROBE_DRIVER FAIL"
exit 1

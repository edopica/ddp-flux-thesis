#!/usr/bin/env bash
# Run the full Flux Phase 3 checker on CPU against (a) the preserved PyTorch
# GPU reference and (b) the new PyTorch CPU reference. Diagnostic only: the
# checkers, tolerances, model math, optimizer, and Flux source are unchanged.
#
# Runs inside an srun on a Leonardo compute node. This script never invokes
# srun itself and refuses to run on a login node.
set -euo pipefail

WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
ROOT=${ROOT:-$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-numerics}
GPU_REF_DIR=${GPU_REF_DIR:-$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-attempt3/reference}
CPU_REF_DIR=${CPU_REF_DIR:-$ROOT/torch_cpu_reference}
FIX=${FIX:-$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2}
BUNDLE=${BUNDLE:-$FIX/initial_bundle.npz}
LOG_DIR=${LOG_DIR:-$ROOT/logs}
RUN_JULIA=$THESIS/scripts/remote/leonardo/run-julia-phase3.sh
CHECK=$THESIS/scripts/static_scaling/check_phase3_flux.jl

host=$(hostname)
case "$host" in
  login*) echo "ERROR: running on login node '$host'; run under srun" >&2; exit 1 ;;
esac

[ -f "$CHECK" ] || { echo "ERROR: checker missing: $CHECK" >&2; exit 1; }
[ -f "$GPU_REF_DIR/phase3_reference.npz" ] || { echo "ERROR: preserved GPU reference missing" >&2; exit 1; }
[ -f "$GPU_REF_DIR/phase3_reference_manifest.json" ] || { echo "ERROR: preserved GPU manifest missing" >&2; exit 1; }
[ -f "$CPU_REF_DIR/phase3_reference.npz" ] || { echo "ERROR: CPU reference missing" >&2; exit 1; }
[ -f "$CPU_REF_DIR/phase3_reference_manifest.json" ] || { echo "ERROR: CPU manifest missing" >&2; exit 1; }

mkdir -p "$LOG_DIR"

run_case() {
  local label=$1 reference=$2 manifest=$3 log=$4
  echo "--- Flux CPU vs $label --- reference=$reference"
  set +e
  bash "$RUN_JULIA" "$CHECK" \
    --fixture-dir "$FIX" \
    --bundle "$BUNDLE" \
    --reference "$reference" \
    --manifest "$manifest" \
    --device cpu \
    --atol 1e-5 --rtol 1e-4 \
    2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  echo "FLUX_CPU_CASE label=$label exit=$status log=$log"
  return 0
}

echo "=== phase3 numerics: Flux CPU checks === host=$host job=${SLURM_JOB_ID:-none} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_case torch_gpu "$GPU_REF_DIR/phase3_reference.npz" \
  "$GPU_REF_DIR/phase3_reference_manifest.json" \
  "$LOG_DIR/flux-cpu-vs-torch-gpu.log"
run_case torch_cpu "$CPU_REF_DIR/phase3_reference.npz" \
  "$CPU_REF_DIR/phase3_reference_manifest.json" \
  "$LOG_DIR/flux-cpu-vs-torch-cpu.log"

{
  sha256sum "$LOG_DIR/flux-cpu-vs-torch-gpu.log" "$LOG_DIR/flux-cpu-vs-torch-cpu.log"
  sha256sum "$GPU_REF_DIR/phase3_reference.npz" "$GPU_REF_DIR/phase3_reference_manifest.json" \
    "$CPU_REF_DIR/phase3_reference.npz" "$CPU_REF_DIR/phase3_reference_manifest.json"
} > "$LOG_DIR/SHA256SUMS-flux-cpu"

echo "FLUX_CPU_SUMMARY done utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FLUX_CPU_DONE"

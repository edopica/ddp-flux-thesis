#!/usr/bin/env bash
# Generate the Phase 3 PyTorch CPU reference and compare it with the
# preserved PyTorch GPU reference. Diagnostic only: no tolerance, model-math,
# optimizer, checker, or Flux source change.
#
# Runs inside an srun on a Leonardo compute node. This script never invokes
# srun itself and refuses to run on a login node.
set -euo pipefail

WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
VENV_PY=${VENV_PY:-$WORK/epicazio/venvs/pytorch-scaling/bin/python}
ROOT=${ROOT:-$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-numerics}
GPU_REF_DIR=${GPU_REF_DIR:-$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-attempt3/reference}
FIX=${FIX:-$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2}
BUNDLE=${BUNDLE:-$FIX/initial_bundle.npz}
OUT=${OUT:-$ROOT/torch_cpu_reference}
LOG_DIR=${LOG_DIR:-$ROOT/logs}
SCRIPTS=$THESIS/scripts/static_scaling

SOURCE_COMMIT=${SOURCE_COMMIT:-3088f61020137912af32d9c969e87ea9ae2a71c3}
SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-82623bbb516a981d6c795792d1433521d25a8ee3bbcc44a2b66f6f4631a452b9}
FLUX_COMMIT=${FLUX_COMMIT:-f7b4520c4bd307c67bbc5ac92a1c68493eccff6e}
FLUX_STATE_SHA256=${FLUX_STATE_SHA256:-e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855}

host=$(hostname)
case "$host" in
  login*) echo "ERROR: running on login node '$host'; run under srun" >&2; exit 1 ;;
esac

[ -x "$VENV_PY" ] || { echo "ERROR: venv python not found: $VENV_PY" >&2; exit 1; }
[ -f "$FIX/data_fixture.npz" ] || { echo "ERROR: fixture missing: $FIX" >&2; exit 1; }
[ -f "$BUNDLE" ] || { echo "ERROR: bundle missing: $BUNDLE" >&2; exit 1; }
[ -f "$GPU_REF_DIR/phase3_reference.npz" ] || { echo "ERROR: preserved GPU reference missing" >&2; exit 1; }
[ -f "$GPU_REF_DIR/phase3_reference_manifest.json" ] || { echo "ERROR: preserved GPU manifest missing" >&2; exit 1; }

mkdir -p "$OUT" "$LOG_DIR"

echo "=== phase3 numerics: PyTorch CPU reference === host=$host job=${SLURM_JOB_ID:-none} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "ROOT=$ROOT"
echo "OUT=$OUT"
echo "LOG_DIR=$LOG_DIR"
echo "GPU_REF_DIR=$GPU_REF_DIR"

echo "--- 1/3 make_phase3_reference.py --device cpu ---"
"$VENV_PY" "$SCRIPTS/make_phase3_reference.py" \
  --fixture-dir "$FIX" \
  --bundle "$BUNDLE" \
  --out-dir "$OUT" \
  --device cpu \
  --force \
  --source-commit "$SOURCE_COMMIT" \
  --source-state-sha256 "$SOURCE_STATE_SHA256" \
  --flux-commit "$FLUX_COMMIT" \
  --flux-state-sha256 "$FLUX_STATE_SHA256" \
  2>&1 | tee "$LOG_DIR/torch-cpu-reference.log"

echo "--- 2/3 check_phase3_reference.py ---"
"$VENV_PY" "$SCRIPTS/check_phase3_reference.py" \
  --fixture-dir "$FIX" \
  --bundle "$BUNDLE" \
  --reference "$OUT/phase3_reference.npz" \
  --manifest "$OUT/phase3_reference_manifest.json" \
  --atol 1e-5 --rtol 1e-4 \
  2>&1 | tee "$LOG_DIR/torch-cpu-reference-check.log"

echo "--- 3/3 compare PyTorch CPU vs preserved PyTorch GPU ---"
set +e
"$VENV_PY" "$SCRIPTS/compare_reference_arrays.py" \
  --ref "$GPU_REF_DIR/phase3_reference.npz" \
  --cand "$OUT/phase3_reference.npz" \
  --bundle "$BUNDLE" \
  --atol 1e-5 --rtol 1e-4 \
  --json-out "$LOG_DIR/torch-cpu-vs-gpu-reference.json" \
  2>&1 | tee "$LOG_DIR/torch-cpu-vs-gpu-reference.log"
COMPARE_EXIT=${PIPESTATUS[0]}
set -e

{
  sha256sum "$OUT/phase3_reference.npz" "$OUT/phase3_reference_manifest.json" \
    "$LOG_DIR/torch-cpu-reference.log" "$LOG_DIR/torch-cpu-reference-check.log" \
    "$LOG_DIR/torch-cpu-vs-gpu-reference.log" "$LOG_DIR/torch-cpu-vs-gpu-reference.json"
  sha256sum "$GPU_REF_DIR/phase3_reference.npz" "$GPU_REF_DIR/phase3_reference_manifest.json"
} > "$LOG_DIR/SHA256SUMS"

echo "CPU_REFERENCE_SUMMARY compare_exit=$COMPARE_EXIT utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "CPU_REFERENCE_DONE"

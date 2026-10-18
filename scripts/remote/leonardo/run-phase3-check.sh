#!/usr/bin/env bash
# =============================================================================
# Leonardo (CINECA) COMPUTE-NODE acceptance driver for the Phase 3
# static-scaling reference and Flux comparison.
#
# Steps (all inside the caller's srun; this script never invokes srun):
#   1/3 make_phase3_reference.py    (PyTorch reference + manifest)
#   2/3 check_phase3_reference.py   (independent numpy-only artifact check)
#   3/3 check_phase3_flux.jl        (Flux vs. reference numerical comparison)
#
# ALLOCATION (from the thesis root on a login node):
#   bash scripts/remote/run.sh leonardo \
#     "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis \
#      bash scripts/remote/leonardo/run-phase3-check.sh"
#
# The Flux load path is the thesis julia-env plus the synced Flux test project:
#   JULIA_LOAD_PATH="$THESIS/scripts/static_scaling/julia-env:$FLUX_DIR/test:@stdlib"
# Precompilation happens on the compute node inside this job; never precompile
# on the login node.
#
# CLI assumptions (Phase 3 contract):
#   make_phase3_reference.py --fixture-dir FIX --bundle BUNDLE --out-dir OUT
#     --device DEVICE [--force] [--source-commit ...] [--source-state-sha256 ...]
#     [--flux-commit ...] [--flux-state-sha256 ...]
#   check_phase3_reference.py --fixture-dir FIX --bundle BUNDLE
#     --reference OUT/phase3_reference.npz
#     --manifest OUT/phase3_reference_manifest.json --atol ATOL --rtol RTOL
#   check_phase3_flux.jl --fixture-dir FIX --bundle BUNDLE --atol ATOL --rtol RTOL
#     plus either --reference-dir OUT or --reference NPZ --manifest JSON.
#   The Flux form is detected from the checker source; the reference form is the
#   fallback when --reference-dir is absent.
#
# Optional source provenance (never required; empty values are not forwarded):
#   SOURCE_COMMIT, SOURCE_STATE_SHA256, FLUX_COMMIT, FLUX_STATE_SHA256
# as env prefixes on scripts/remote/run.sh.
#
# Idempotence: the log is append-only. The reference generator may overwrite
# only its own two files ($REFERENCE, $REFERENCE_MANIFEST); nothing is deleted
# by this driver.
#
# Evidence log: $LOG_DIR/remote-check.log (append; scp it back afterwards).
# Exit code: 0 only if all three steps exit 0 and their PASS summary lines pass.
# =============================================================================
set -euo pipefail

WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
VENV_PY=${VENV_PY:-$WORK/epicazio/venvs/pytorch-scaling/bin/python}
JULIA_BIN=${JULIA_BIN:-$WORK/epicazio/julia-1.12.6/bin/julia}
JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$WORK/epicazio/julia-depot}
FIX=${FIX:-$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2}
BUNDLE=${BUNDLE:-$FIX/initial_bundle.npz}
OUT=${OUT:-$THESIS/artifacts/baselines/static-scaling/2026-10-17-phase3}
LOG_DIR=${LOG_DIR:-$THESIS/artifacts/logs/static-scaling/2026-10-17-phase3}
DEVICE=${DEVICE:-cuda}
FLUX_DIR=${FLUX_DIR:-$HOME/projects/flux-integration}
ATOL=${ATOL:-1e-5}
RTOL=${RTOL:-1e-4}
ALLOW_LOGIN_NODE=${ALLOW_LOGIN_NODE:-0}
SOURCE_COMMIT=${SOURCE_COMMIT:-}
SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-}
FLUX_COMMIT=${FLUX_COMMIT:-}
FLUX_STATE_SHA256=${FLUX_STATE_SHA256:-}

REFERENCE=${REFERENCE:-$OUT/phase3_reference.npz}
REFERENCE_MANIFEST=${REFERENCE_MANIFEST:-$OUT/phase3_reference_manifest.json}

LOG_FILE=$LOG_DIR/remote-check.log
JULIA_ENV=$THESIS/scripts/static_scaling/julia-env
SCRIPTS=$THESIS/scripts/static_scaling

export SOURCE_COMMIT SOURCE_STATE_SHA256 FLUX_COMMIT FLUX_STATE_SHA256

die() { echo "ERROR: $*" >&2; exit 1; }

supports_flag() {
  [ -f "$1" ] && grep -q -- "$2" "$1" 2>/dev/null
}

# ---- compute-node guard -----------------------------------------------------
host=$(hostname)
case "$host" in
  login*)
    if [ "$ALLOW_LOGIN_NODE" != 1 ]; then
      die "running on login node '$host': run under srun on a compute node (or set ALLOW_LOGIN_NODE=1 to override on purpose)"
    fi
    ;;
esac

# ---- environment checks (before touching any output) ------------------------
[ -x "$VENV_PY" ] || die "venv python not found: $VENV_PY"
[ -f "$FIX/data_fixture.npz" ] || die "data fixture not found: $FIX/data_fixture.npz (Phase 2 artifacts missing)"
[ -f "$FIX/fixture_manifest.json" ] || die "fixture manifest not found: $FIX/fixture_manifest.json"
[ -f "$BUNDLE" ] || die "bundle not found: $BUNDLE"
[ -f "$SCRIPTS/make_phase3_reference.py" ] || die "generator not found: $SCRIPTS/make_phase3_reference.py (sync first)"
[ -f "$SCRIPTS/check_phase3_reference.py" ] || die "checker not found: $SCRIPTS/check_phase3_reference.py"
[ -f "$SCRIPTS/check_phase3_flux.jl" ] || die "Flux checker not found: $SCRIPTS/check_phase3_flux.jl"
[ -f "$JULIA_ENV/Manifest.toml" ] || die "julia-env not instantiated: $JULIA_ENV"
[ -d "$FLUX_DIR/test" ] || die "Flux test project not found: $FLUX_DIR/test (sync flux-integration first)"
[ -x "$JULIA_BIN" ] || die "julia not found: $JULIA_BIN"

mkdir -p "$LOG_DIR" "$OUT"

STEP_DIR=$(mktemp -d "$LOG_DIR/.phase3-check.XXXXXX") || die "mktemp failed"
trap 'rm -rf "$STEP_DIR"' EXIT
STEP1_LOG=$STEP_DIR/step1.log
STEP2_LOG=$STEP_DIR/step2.log
STEP3_LOG=$STEP_DIR/step3.log

summary_ok() {
  local file=$1
  local prefix=$2
  local line
  [ -f "$file" ] || return 1
  line=$(grep -E "^${prefix}( |\$)" "$file" 2>/dev/null | tail -n 1 || true)
  [ -n "$line" ] || return 1
  case "$line" in
    "$prefix PASS"*) return 0 ;;
    *) return 1 ;;
  esac
}

run_step() {
  local label=$1
  local logfile=$2
  local status=0
  shift 2
  echo "=== $label ==="
  time "$@" 2>&1 | tee "$logfile"
  status=${PIPESTATUS[0]}
  echo "--- $label exit=$status ---"
  return "$status"
}

run() {
  set +e
  echo "=== run-phase3-check: START ==="
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>} SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
  echo "THESIS=$THESIS"
  echo "FIX=$FIX"
  echo "BUNDLE=$BUNDLE"
  echo "OUT=$OUT"
  echo "REFERENCE=$REFERENCE"
  echo "REFERENCE_MANIFEST=$REFERENCE_MANIFEST"
  echo "LOG_FILE=$LOG_FILE"
  echo "DEVICE=$DEVICE FLUX_DIR=$FLUX_DIR ATOL=$ATOL RTOL=$RTOL"
  echo "SOURCE_COMMIT=${SOURCE_COMMIT:-<unset>} SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-<unset>}"
  echo "FLUX_COMMIT=${FLUX_COMMIT:-<unset>} FLUX_STATE_SHA256=${FLUX_STATE_SHA256:-<unset>}"
  echo "THESIS_GIT=$(git -C "$THESIS" rev-parse --short HEAD 2>/dev/null || echo '<no-git>')"
  echo "FLUX_GIT=$(git -C "$FLUX_DIR" rev-parse --short HEAD 2>/dev/null || echo '<no-git>')"
  echo "--- versions ---"
  "$VENV_PY" - <<'PY' || die "python/torch version probe failed"
import sys
import numpy
print("python", sys.version.split()[0], "numpy", numpy.__version__)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "cuda_available", torch.cuda.is_available())
PY
  case "$DEVICE" in
    cuda*)
      "$VENV_PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
        || die "DEVICE=$DEVICE but torch.cuda.is_available() is false"
      ;;
  esac
  "$JULIA_BIN" --version || die "julia --version failed"
  echo "--- reference overwrite policy ---"
  echo "generator may overwrite only $REFERENCE and $REFERENCE_MANIFEST; $LOG_FILE is append-only"
  echo "--- disk ($OUT) ---"
  df -h "$(dirname "$OUT")"

  local ref_args=(--fixture-dir "$FIX" --bundle "$BUNDLE" --out-dir "$OUT" --device "$DEVICE")
  local force_flag=0
  if supports_flag "$SCRIPTS/make_phase3_reference.py" '--force'; then
    ref_args+=(--force)
    force_flag=1
  fi
  if [ -n "$SOURCE_COMMIT" ]; then ref_args+=(--source-commit "$SOURCE_COMMIT"); fi
  if [ -n "$SOURCE_STATE_SHA256" ]; then ref_args+=(--source-state-sha256 "$SOURCE_STATE_SHA256"); fi
  if [ -n "$FLUX_COMMIT" ]; then ref_args+=(--flux-commit "$FLUX_COMMIT"); fi
  if [ -n "$FLUX_STATE_SHA256" ]; then ref_args+=(--flux-state-sha256 "$FLUX_STATE_SHA256"); fi
  echo "generator_force_flag=$force_flag generator_args=${ref_args[*]}"

  local ref_status=0
  local chk_status=0
  local flux_status=0
  local ref_summary=0
  local chk_summary=0
  local flux_summary=0

  run_step "1/3 make_phase3_reference.py" "$STEP1_LOG" \
    "$VENV_PY" "$SCRIPTS/make_phase3_reference.py" "${ref_args[@]}"
  ref_status=$?
  if [ "$ref_status" -eq 0 ] && summary_ok "$STEP1_LOG" "PHASE3_REFERENCE"; then
    ref_summary=1
  fi

  if [ "$ref_status" -eq 0 ] && [ "$ref_summary" -eq 1 ]; then
    run_step "2/3 check_phase3_reference.py" "$STEP2_LOG" \
      "$VENV_PY" "$SCRIPTS/check_phase3_reference.py" \
      --fixture-dir "$FIX" --bundle "$BUNDLE" \
      --reference "$REFERENCE" --manifest "$REFERENCE_MANIFEST" \
      --atol "$ATOL" --rtol "$RTOL"
    chk_status=$?
    if [ "$chk_status" -eq 0 ] && summary_ok "$STEP2_LOG" "PHASE3_REFERENCE_CHECK"; then
      chk_summary=1
    fi
  else
    echo "=== 2/3 check_phase3_reference.py SKIP (reference generation did not PASS) ==="
    chk_status=1
  fi

  if [ "$chk_status" -eq 0 ] && [ "$chk_summary" -eq 1 ]; then
    local flux_args=(--fixture-dir "$FIX" --bundle "$BUNDLE" --device "$DEVICE" --atol "$ATOL" --rtol "$RTOL")
    if supports_flag "$SCRIPTS/check_phase3_flux.jl" '--reference-dir'; then
      flux_args+=(--reference-dir "$OUT")
    else
      flux_args+=(--reference "$REFERENCE")
      if supports_flag "$SCRIPTS/check_phase3_flux.jl" '--reference-manifest'; then
        flux_args+=(--reference-manifest "$REFERENCE_MANIFEST")
      else
        flux_args+=(--manifest "$REFERENCE_MANIFEST")
      fi
    fi
    echo "flux_checker_args=${flux_args[*]}"
    export JULIA_DEPOT_PATH
    export JULIA_LOAD_PATH="$JULIA_ENV:$FLUX_DIR/test:@stdlib"
    export JULIA_PKG_PRECOMPILE_AUTO=${JULIA_PKG_PRECOMPILE_AUTO:-0}
    export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
    run_step "3/3 check_phase3_flux.jl" "$STEP3_LOG" \
      "$JULIA_BIN" --startup-file=no "$SCRIPTS/check_phase3_flux.jl" "${flux_args[@]}"
    flux_status=$?
    if [ "$flux_status" -eq 0 ] && summary_ok "$STEP3_LOG" "PHASE3_FLUX_CHECK"; then
      flux_summary=1
    fi
  else
    echo "=== 3/3 check_phase3_flux.jl SKIP (reference check did not PASS) ==="
    flux_status=1
  fi

  echo "=== summary ==="
  echo "reference_exit=$ref_status reference_summary=$ref_summary"
  echo "reference_check_exit=$chk_status reference_check_summary=$chk_summary"
  echo "flux_check_exit=$flux_status flux_check_summary=$flux_summary"
  if [ "$ref_status" -eq 0 ] && [ "$ref_summary" -eq 1 ] &&
     [ "$chk_status" -eq 0 ] && [ "$chk_summary" -eq 1 ] &&
     [ "$flux_status" -eq 0 ] && [ "$flux_summary" -eq 1 ]; then
    echo "PHASE3_REMOTE_CHECK PASS"
  else
    echo "PHASE3_REMOTE_CHECK FAIL"
  fi
  echo "NEXT (bring evidence home): scp leonardo:$LOG_FILE artifacts/logs/static-scaling/2026-10-17-phase3/"
  echo "NEXT (bring artifacts home): scp leonardo:$OUT/phase3_reference.npz leonardo:$OUT/phase3_reference_manifest.json artifacts/baselines/static-scaling/2026-10-17-phase3/"

  if [ "$ref_status" -eq 0 ] && [ "$ref_summary" -eq 1 ] &&
     [ "$chk_status" -eq 0 ] && [ "$chk_summary" -eq 1 ] &&
     [ "$flux_status" -eq 0 ] && [ "$flux_summary" -eq 1 ]; then
    return 0
  fi
  return 1
}

set +e
run 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
echo "=== run-phase3-check: EXIT $status (log: $LOG_FILE) ==="
exit "$status"

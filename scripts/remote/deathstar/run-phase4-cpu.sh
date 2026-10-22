#!/usr/bin/env bash
# =============================================================================
# Deathstar CPU/MPI-only Phase 4 static-graph smoke.
#
# Launch from the thesis root:
#   scripts/remote/run.sh deathstar \
#     "bash scripts/remote/deathstar/run-phase4-cpu.sh"
#
# Deathstar has CUDA/cuDNN 6.3.1, which the strict precision gate rejects for
# real GPU runs. This driver never uses the GPUs: it runs `nvidia-smi` once as
# a courtesy check, then clears CUDA_VISIBLE_DEVICES. All training processes use
# DEVICE=cpu and CPU backends (Flux MPI, PyTorch gloo).
#
# Stages (fail-fast, one log per command under LOG_DIR):
#   0. nvidia-smi courtesy check (failure is a warning, not a run failure)
#   a. source identity/hashes, fixture hashes, Julia/PyTorch/MPI versions
#   b. Flux single-process reference (world 1, CPU)
#   c. PyTorch reference via make_phase3_reference.py --device cpu
#   d. split-control for both frameworks (single process)
#   e. Flux DDP for each world in WORLD_SIZES via ~/.julia/bin/mpiexecjl
#   f. PyTorch DDP for each world in WORLD_SIZES via torchrun --standalone
#   g. check_phase4.py; requires PHASE4_CHECK PASS under same-framework
#      tolerances
#   h. COMMANDS.md / PROVENANCE.md plus source and artifact hashes
#
# Runner CLI used here (contract temp/phase4-implementation-contract.md):
#   phase4_flux.jl  --fixture-dir FIX --bundle BUNDLE --mode MODE
#                   --device cpu --backend mpi --world-size W --out OUT
#   phase4_torch.py --fixture-dir FIX --bundle BUNDLE --mode MODE
#                   --device cpu --backend gloo --world-size W
#                   --out-dir OUT (or --out, auto-detected)
#   check_phase4.py --fixture-dir FIX --bundle BUNDLE
#                   --pytorch-reference|--torch-reference FILE
#                   --flux-reference FILE
#                   --ddp-root|--ddp-dir DIR
# Optional --source-* provenance flags are forwarded only when the runner
# source advertises them.
#
# If any log contains "Precompiling packages...", the run fails. Precompile
# first on deathstar with scripts/remote/precompile.sh deathstar.
# =============================================================================
set -euo pipefail

THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
FLUX_DIR=${FLUX_DIR:-$HOME/projects/flux-integration}
JULIA_BIN=${JULIA_BIN:-$(command -v julia 2>/dev/null || true)}
MPIEXECJL=${MPIEXECJL:-$HOME/.julia/bin/mpiexecjl}
if [ -z "${VENV_PY:-}" ]; then
  if [ -x "$HOME/.venvs/pytorch-scaling/bin/python" ]; then
    VENV_PY="$HOME/.venvs/pytorch-scaling/bin/python"
  else
    VENV_PY=python3
  fi
fi
if [ -z "${TORCHRUN_BIN:-}" ]; then
  if [ -n "$VENV_PY" ] && [ -x "$(dirname "$VENV_PY")/torchrun" ]; then
    TORCHRUN_BIN=$(dirname "$VENV_PY")/torchrun
  else
    TORCHRUN_BIN=torchrun
  fi
fi
JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$HOME/.julia}
FIX=${FIX:-$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2}
BUNDLE=${BUNDLE:-$FIX/initial_bundle.npz}
RUN_ID=${RUN_ID:-$(date -u +%Y-%m-%dT%H%M%SZ)-phase4-deathstar-cpu}
OUT=${OUT:-$THESIS/artifacts/logs/static-scaling/$RUN_ID}
LOG_DIR=${LOG_DIR:-$OUT/logs}
DEVICE=${DEVICE:-cpu}
WORLD_SIZES=${WORLD_SIZES:-"2 4"}
TORCH_MASTER_PORT_BASE=${TORCH_MASTER_PORT_BASE:-29600}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
ALLOW_EXISTING_OUT=${ALLOW_EXISTING_OUT:-0}
SOURCE_COMMIT=${SOURCE_COMMIT:-}
SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-}
FLUX_COMMIT=${FLUX_COMMIT:-}
FLUX_STATE_SHA256=${FLUX_STATE_SHA256:-}

SCRIPTS=$THESIS/scripts/static_scaling
FLUX_RUNNER=${FLUX_RUNNER:-$SCRIPTS/phase4_flux.jl}
TORCH_RUNNER=${TORCH_RUNNER:-$SCRIPTS/phase4_torch.py}
CHECKER=${CHECKER:-$SCRIPTS/check_phase4.py}
JULIA_ENV=$SCRIPTS/julia-env
JULIA_LOAD_PATH=${JULIA_LOAD_PATH:-$JULIA_ENV:$FLUX_DIR/test:@stdlib}
FLUX_BACKEND=${FLUX_BACKEND:-mpi}
TORCH_BACKEND=${TORCH_BACKEND:-gloo}
FLUX_REF=${FLUX_REF:-$OUT/reference/flux/cpu/phase4_reference.npz}
FLUX_REF_JSON=${FLUX_REF_JSON:-$OUT/reference/flux/cpu/phase4_reference.json}
TORCH_REF_DIR=${TORCH_REF_DIR:-$OUT/reference/pytorch/cpu}
TORCH_REF=${TORCH_REF:-$TORCH_REF_DIR/phase3_reference.npz}
TORCH_MANIFEST=${TORCH_MANIFEST:-$TORCH_REF_DIR/phase3_reference_manifest.json}
DDP_ROOT=${DDP_ROOT:-$OUT/ddp}
CHECK_DIR=${CHECK_DIR:-$OUT/check}

if [ "$DEVICE" != cpu ]; then
  echo "ERROR: this driver is CPU/MPI-only; DEVICE=$DEVICE is not allowed" >&2
  exit 1
fi
if [ "$FLUX_BACKEND" != mpi ]; then
  echo "ERROR: deathstar Flux backend must be mpi, got $FLUX_BACKEND" >&2
  exit 1
fi
if [ "$TORCH_BACKEND" != gloo ]; then
  echo "ERROR: deathstar PyTorch backend must be gloo, got $TORCH_BACKEND" >&2
  exit 1
fi

# Resolve Julia to the real binary and Sys.BINDIR (handling juliaup shims) so
# mpiexecjl and child Julia processes find ../lib/julia/sys.so.
if [ -n "$JULIA_BIN" ] && command -v "$JULIA_BIN" >/dev/null 2>&1; then
  JULIA_BIN=$(command -v "$JULIA_BIN")
fi
if [ -z "${JULIA_BINDIR:-}" ] && [ -n "$JULIA_BIN" ] && [ -x "$JULIA_BIN" ]; then
  JULIA_BINDIR=$("$JULIA_BIN" --startup-file=no -e 'print(Sys.BINDIR)' 2>/dev/null || dirname "$JULIA_BIN")
fi
if [ -n "${JULIA_BINDIR:-}" ] && [ -x "$JULIA_BINDIR/julia" ]; then
  JULIA_BIN="$JULIA_BINDIR/julia"
fi

is_positive_int() {
  case "${1:-}" in
    ''|*[!0-9]*) return 1 ;;
    *) [ "$1" -gt 0 ] ;;
  esac
}

if [ "$TORCH_MASTER_PORT_BASE" -lt 1024 ] || [ "$TORCH_MASTER_PORT_BASE" -gt 65535 ]; then
  echo "ERROR: TORCH_MASTER_PORT_BASE must be in 1024..65535" >&2
  exit 1
fi
for _world in $WORLD_SIZES; do
  is_positive_int "$_world" || { echo "ERROR: invalid WORLD_SIZES entry '$_world'" >&2; exit 1; }
done
unset _world

# ---- single-process guard ---------------------------------------------------
rank=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-0}}}
if [ "$rank" != 0 ]; then
  echo "ERROR: this driver must run as one process, got rank $rank" >&2
  exit 1
fi

# ---- fresh output root ------------------------------------------------------
case "$OUT" in
  "$THESIS/artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved"*|\
  "$THESIS/artifacts/baselines/static-scaling/2026-10-18-phase3-final"*|\
  "$THESIS/artifacts/baselines/static-scaling/2026-10-18-r1-r2"*)
    echo "ERROR: OUT points at accepted Phase 3 evidence: $OUT" >&2
    exit 1
    ;;
esac
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null || true)" ] && [ "$ALLOW_EXISTING_OUT" != 1 ]; then
  echo "ERROR: OUT exists and is not empty: $OUT" >&2
  echo "       use a fresh OUT/LOG_DIR or set ALLOW_EXISTING_OUT=1 to reuse it" >&2
  exit 1
fi

mkdir -p "$OUT" "$LOG_DIR"
DRIVER_LOG=$LOG_DIR/driver.log
: > "$DRIVER_LOG"

export THESIS FLUX_DIR JULIA_BIN MPIEXECJL VENV_PY TORCHRUN_BIN JULIA_DEPOT_PATH
export JULIA_BINDIR FIX BUNDLE RUN_ID OUT LOG_DIR DEVICE WORLD_SIZES
export TORCH_MASTER_PORT_BASE OMP_NUM_THREADS JULIA_NUM_THREADS JULIA_LOAD_PATH JULIA_ENV
export SOURCE_COMMIT SOURCE_STATE_SHA256 FLUX_COMMIT FLUX_STATE_SHA256
export SCRIPTS FLUX_RUNNER TORCH_RUNNER CHECKER FLUX_BACKEND TORCH_BACKEND
export FLUX_REF FLUX_REF_JSON TORCH_REF_DIR TORCH_REF TORCH_MANIFEST
export DDP_ROOT CHECK_DIR
export JULIA_PKG_PRECOMPILE_AUTO=0

# No GPU use: clear the device list before any training process starts.
export CUDA_VISIBLE_DEVICES=

FLUX_BASH='unset JULIA_PROJECT; export JULIA_BINDIR="$1" JULIA_DEPOT_PATH="$2" JULIA_LOAD_PATH="$3" JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_THREADS="$4"; shift 4; exec "$@"'
TORCH_BASH='export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS="$1" MKL_NUM_THREADS="$1"; shift; exec "$@"'
TORCH_DDP_BASH='export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS="$1" MKL_NUM_THREADS="$1" MASTER_PORT="$2"; shift 2; exec "$@"'

COMMANDS_TRANSCRIPT=$LOG_DIR/commands-transcript.txt
STAGE_STATUS=$LOG_DIR/stage-status.txt
START_UTC=
FAILED=0
FAIL_STAGE=
FAIL_REASON=
FLUX_PROV_ARGS=()
TORCH_PROV_ARGS=()
PY_PROV_ARGS=()
TORCH_OUT_FLAG=
CHECK_TORCH_FLAG=
CHECK_DDP_FLAG=

utc_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

quote_cmd() { printf '%q ' "$@"; }

supports_flag() {
  local file=$1
  local flag=$2
  [ -f "$file" ] || return 1
  if grep -q -- "$flag" "$file" 2>/dev/null; then
    return 0
  fi
  local common="$(dirname "$file")/phase4_common.jl"
  if [ "$(basename "$file")" = "phase4_flux.jl" ] && [ -f "$common" ]; then
    if grep -q -- "$flag" "$common" 2>/dev/null || grep -q -- "\"${flag#--}\"" "$common" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

choose_flag() {
  local file=$1
  shift
  local flag
  for flag in "$@"; do
    if supports_flag "$file" "$flag"; then
      echo "$flag"
      return 0
    fi
  done
  return 1
}

check_runner_flags() {
  local file=$1
  shift
  local missing=()
  local flag
  for flag in "$@"; do
    if ! supports_flag "$file" "$flag"; then
      missing+=("$flag")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "MISSING flags in $file: ${missing[*]}"
    return 1
  fi
  return 0
}

provenance_args() {
  local file=$1
  if [ -n "$SOURCE_COMMIT" ] && supports_flag "$file" --source-commit; then
    printf '%s\n' --source-commit "$SOURCE_COMMIT"
  fi
  if [ -n "$SOURCE_STATE_SHA256" ] && supports_flag "$file" --source-state-sha256; then
    printf '%s\n' --source-state-sha256 "$SOURCE_STATE_SHA256"
  fi
  if [ -n "$FLUX_COMMIT" ] && supports_flag "$file" --flux-commit; then
    printf '%s\n' --flux-commit "$FLUX_COMMIT"
  fi
  if [ -n "$FLUX_STATE_SHA256" ] && supports_flag "$file" --flux-state-sha256; then
    printf '%s\n' --flux-state-sha256 "$FLUX_STATE_SHA256"
  fi
}

record_status() {
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$STAGE_STATUS"
}

set_pass() { record_status "$1" PASS "$2"; }
set_skip() { record_status "$1" SKIP "$2"; }
set_fail() {
  if [ "$FAILED" -eq 0 ]; then
    FAILED=1
    FAIL_STAGE=$1
    FAIL_REASON=$2
  fi
  record_status "$1" FAIL "$2"
}

run_cmd() {
  local stage=$1
  local label=$2
  local logfile=$3
  shift 3
  local display start_utc end_utc status
  display=$(quote_cmd "$@")
  start_utc=$(utc_now)

  echo "=== STAGE $stage :: $label START $start_utc ==="
  echo "COMMAND: $display"
  echo "LOG: $logfile"
  {
    echo "=== STAGE $stage :: $label START $start_utc ==="
    echo "COMMAND: $display"
    echo "LOG: $logfile"
  } > "$logfile"

  set +e
  "$@" 2>&1 | tee -a "$logfile"
  status=${PIPESTATUS[0]}
  set -e

  end_utc=$(utc_now)
  echo "=== STAGE $stage :: $label END $end_utc EXIT $status ===" | tee -a "$logfile"
  {
    echo "STAGE $stage $label"
    echo "START_UTC $start_utc"
    echo "END_UTC $end_utc"
    echo "EXIT $status"
    echo "LOG $logfile"
    echo "COMMAND $display"
    echo
  } >> "$COMMANDS_TRANSCRIPT"

  if grep -Fq 'Precompiling packages...' "$logfile"; then
    echo "STAGE $stage :: $label FAIL: 'Precompiling packages...' found in $logfile" | tee -a "$logfile"
    return 97
  fi
  return "$status"
}

is_executable_or_on_path() {
  local value=$1
  if [ -x "$value" ]; then return 0; fi
  command -v "$value" >/dev/null 2>&1
}

preflight() {
  local missing=0
  local path
  local -a required=(
    "$FIX/data_fixture.npz"
    "$FIX/fixture_manifest.json"
    "$FIX/schedule_fixture.npz"
    "$BUNDLE"
    "$SCRIPTS/make_phase3_reference.py"
    "$FLUX_RUNNER"
    "$TORCH_RUNNER"
    "$CHECKER"
    "$JULIA_ENV/Project.toml"
    "$JULIA_ENV/Manifest.toml"
  )
  for path in "${required[@]}"; do
    if [ ! -f "$path" ]; then
      echo "MISSING file: $path"
      missing=1
    fi
  done
  if ! is_executable_or_on_path "$JULIA_BIN"; then
    echo "MISSING executable: julia ($JULIA_BIN)"
    missing=1
  fi
  if ! is_executable_or_on_path "$VENV_PY"; then
    echo "MISSING executable: python ($VENV_PY)"
    missing=1
  fi
  if ! is_executable_or_on_path "$MPIEXECJL"; then
    echo "MISSING executable: mpiexecjl ($MPIEXECJL)"
    missing=1
  fi
  if ! is_executable_or_on_path "$TORCHRUN_BIN"; then
    echo "MISSING executable: torchrun ($TORCHRUN_BIN)"
    missing=1
  fi
  if [ ! -d "$FLUX_DIR/test" ]; then
    echo "MISSING directory: $FLUX_DIR/test (run scripts/remote/sync_flux_integration.sh deathstar)"
    missing=1
  fi

  if ! check_runner_flags "$FLUX_RUNNER" \
    --fixture-dir --bundle --mode --device --backend --world-size --out; then
    missing=1
  fi
  if ! check_runner_flags "$TORCH_RUNNER" \
    --fixture-dir --bundle --mode --device --backend --world-size; then
    missing=1
  fi
  TORCH_OUT_FLAG=$(choose_flag "$TORCH_RUNNER" --out-dir --out || true)
  if [ -z "$TORCH_OUT_FLAG" ]; then
    echo "MISSING flag in $TORCH_RUNNER: --out-dir or --out"
    missing=1
  fi
  CHECK_TORCH_FLAG=$(choose_flag "$CHECKER" --pytorch-reference --torch-reference || true)
  if [ -z "$CHECK_TORCH_FLAG" ]; then
    echo "MISSING flag in $CHECKER: --pytorch-reference or --torch-reference"
    missing=1
  fi
  CHECK_DDP_FLAG=$(choose_flag "$CHECKER" --ddp-root --ddp-dir || true)
  if [ -z "$CHECK_DDP_FLAG" ]; then
    echo "MISSING flag in $CHECKER: --ddp-root or --ddp-dir"
    missing=1
  fi

  if [ -n "$TORCH_OUT_FLAG" ]; then
    echo "TORCH_OUT_FLAG=$TORCH_OUT_FLAG"
  fi
  if [ -n "$CHECK_TORCH_FLAG" ]; then
    echo "CHECK_TORCH_FLAG=$CHECK_TORCH_FLAG CHECK_DDP_FLAG=$CHECK_DDP_FLAG"
  fi
  return "$missing"
}

stage_a() {
  run_cmd a "source-and-tool-metadata" "$LOG_DIR/a-source-metadata.log" bash -c '
set -euxo pipefail
date -u +%Y-%m-%dT%H:%M:%SZ
hostname
uname -a
nproc
echo "THESIS=$THESIS"
echo "FLUX_DIR=$FLUX_DIR"
echo "FIX=$FIX"
echo "BUNDLE=$BUNDLE"
echo "OUT=$OUT"
echo "LOG_DIR=$LOG_DIR"
echo "DEVICE=$DEVICE FLUX_BACKEND=$FLUX_BACKEND TORCH_BACKEND=$TORCH_BACKEND"
echo "WORLD_SIZES=$WORLD_SIZES"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} (GPUs are not used)"
echo "SOURCE_COMMIT=${SOURCE_COMMIT:-<unset>} SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-<unset>}"
echo "FLUX_COMMIT=${FLUX_COMMIT:-<unset>} FLUX_STATE_SHA256=${FLUX_STATE_SHA256:-<unset>}"
if command -v git >/dev/null 2>&1; then
  echo "THESIS_GIT=$(git -C "$THESIS" rev-parse HEAD 2>/dev/null || echo "<no-git>")"
  echo "FLUX_GIT=$(git -C "$FLUX_DIR" rev-parse HEAD 2>/dev/null || echo "<no-git>")"
else
  echo "THESIS_GIT=<no-git-command>"
  echo "FLUX_GIT=<no-git-command>"
fi
echo "PARENT_COMMAND=$(ps -o args= -p "$PPID" 2>/dev/null || echo "<unavailable>")"
echo "--- julia ---"
"$JULIA_BIN" --version
echo "--- mpiexecjl ---"
"$MPIEXECJL" --project="$FLUX_DIR/test" --version
echo "--- python/torch ---"
"$VENV_PY" - <<"PY"
import sys
print("python", sys.version.split()[0])
import numpy
print("numpy", numpy.__version__)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cudnn", torch.backends.cudnn.version())
PY
echo "--- hash manifest ---"
{
  echo "# static_scaling sources"
  find "$THESIS/scripts/static_scaling" -maxdepth 1 -type f \( -name "*.jl" -o -name "*.py" \) -print0 | sort -z | xargs -0 sha256sum
  echo "# remote drivers"
  for f in "$THESIS/scripts/remote/deathstar/run-phase4-cpu.sh" "$THESIS/scripts/remote/sync_flux_integration.sh"; do
    if [ -f "$f" ]; then sha256sum "$f"; else echo "MISSING $f"; fi
  done
  echo "# launchers"
  sha256sum "$MPIEXECJL" "$JULIA_BIN"
  echo "# inputs"
  sha256sum "$FIX/data_fixture.npz" "$FIX/fixture_manifest.json" "$FIX/schedule_fixture.npz" "$BUNDLE"
  echo "# environments"
  for f in "$JULIA_ENV/Project.toml" "$JULIA_ENV/Manifest.toml" "$FLUX_DIR/Project.toml" "$FLUX_DIR/Manifest.toml" "$FLUX_DIR/test/Project.toml" "$FLUX_DIR/test/Manifest.toml"; do
    if [ -f "$f" ]; then sha256sum "$f"; else echo "MISSING $f"; fi
  done
} | tee "$LOG_DIR/source-sha256.txt"
'
}

stage_b() {
  local -a args=(
    --fixture-dir "$FIX"
    --bundle "$BUNDLE"
    --mode reference
    --device "$DEVICE"
    --backend "$FLUX_BACKEND"
    --world-size 1
    --out "$OUT"
  )
  args+=("${FLUX_PROV_ARGS[@]}")
  run_cmd b "flux-reference" "$LOG_DIR/b-flux-reference.log" \
    bash -c "$FLUX_BASH" _ "$JULIA_BINDIR" "$JULIA_DEPOT_PATH" "$JULIA_LOAD_PATH" "$JULIA_NUM_THREADS" \
    "$JULIA_BIN" --startup-file=no "$FLUX_RUNNER" "${args[@]}" || return $?
  run_cmd b "verify-flux-reference" "$LOG_DIR/b-verify-flux-reference.log" bash -c '
set -euo pipefail
test -f "$1"
test -f "$2"
sha256sum "$1" "$2"
' _ "$FLUX_REF" "$FLUX_REF_JSON" || return $?
  return 0
}

stage_c() {
  local -a args=(
    --fixture-dir "$FIX"
    --bundle "$BUNDLE"
    --out-dir "$TORCH_REF_DIR"
    --device "$DEVICE"
    --force
  )
  args+=("${PY_PROV_ARGS[@]}")
  run_cmd c "pytorch-reference" "$LOG_DIR/c-pytorch-reference.log" \
    bash -c "$TORCH_BASH" _ "$OMP_NUM_THREADS" \
    "$VENV_PY" "$SCRIPTS/make_phase3_reference.py" "${args[@]}" || return $?
  run_cmd c "verify-pytorch-reference" "$LOG_DIR/c-verify-pytorch-reference.log" bash -c '
set -euo pipefail
test -f "$1"
test -f "$2"
sha256sum "$1" "$2"
' _ "$TORCH_REF" "$TORCH_MANIFEST" || return $?
  return 0
}

stage_d() {
  local -a flux_args=(
    --fixture-dir "$FIX"
    --bundle "$BUNDLE"
    --mode split-control
    --device "$DEVICE"
    --backend "$FLUX_BACKEND"
    --world-size 1
    --out "$OUT"
  )
  flux_args+=("${FLUX_PROV_ARGS[@]}")
  run_cmd d "flux-split-control" "$LOG_DIR/d-flux-split-control.log" \
    bash -c "$FLUX_BASH" _ "$JULIA_BINDIR" "$JULIA_DEPOT_PATH" "$JULIA_LOAD_PATH" "$JULIA_NUM_THREADS" \
    "$JULIA_BIN" --startup-file=no "$FLUX_RUNNER" "${flux_args[@]}" || return $?

  local -a torch_args=(
    --fixture-dir "$FIX"
    --bundle "$BUNDLE"
    --mode split-control
    --device "$DEVICE"
    --backend "$TORCH_BACKEND"
    --world-size 1
    "$TORCH_OUT_FLAG" "$OUT"
  )
  torch_args+=("${TORCH_PROV_ARGS[@]}")
  run_cmd d "pytorch-split-control" "$LOG_DIR/d-pytorch-split-control.log" \
    bash -c "$TORCH_BASH" _ "$OMP_NUM_THREADS" \
    "$VENV_PY" "$TORCH_RUNNER" "${torch_args[@]}" || return $?

  local W
  local -a shard_ref_files=()
  for W in $WORLD_SIZES; do
    local -a flux_shard_args=(
      --fixture-dir "$FIX"
      --bundle "$BUNDLE"
      --mode shard-reference
      --device "$DEVICE"
      --backend "$FLUX_BACKEND"
      --world-size "$W"
      --out "$OUT"
    )
    flux_shard_args+=("${FLUX_PROV_ARGS[@]}")
    run_cmd d "flux-shard-reference-w$W" "$LOG_DIR/d-flux-shard-reference-w$W.log" \
      bash -c "$FLUX_BASH" _ "$JULIA_BINDIR" "$JULIA_DEPOT_PATH" "$JULIA_LOAD_PATH" "$JULIA_NUM_THREADS" \
      "$JULIA_BIN" --startup-file=no "$FLUX_RUNNER" "${flux_shard_args[@]}" || return $?

    local -a torch_shard_args=(
      --fixture-dir "$FIX"
      --bundle "$BUNDLE"
      --mode shard-reference
      --device "$DEVICE"
      --backend "$TORCH_BACKEND"
      --world-size "$W"
      "$TORCH_OUT_FLAG" "$OUT"
    )
    torch_shard_args+=("${TORCH_PROV_ARGS[@]}")
    run_cmd d "pytorch-shard-reference-w$W" "$LOG_DIR/d-pytorch-shard-reference-w$W.log" \
      bash -c "$TORCH_BASH" _ "$OMP_NUM_THREADS" \
      "$VENV_PY" "$TORCH_RUNNER" "${torch_shard_args[@]}" || return $?

    shard_ref_files+=(
      "$OUT/shard_reference/flux/$DEVICE/w$W/phase4_reference.npz"
      "$OUT/shard_reference/flux/$DEVICE/w$W/phase4_reference.json"
      "$OUT/shard_reference/pytorch/$DEVICE/w$W/phase4_reference.npz"
      "$OUT/shard_reference/pytorch/$DEVICE/w$W/phase4_reference.json"
    )
  done

  run_cmd d "verify-split-outputs" "$LOG_DIR/d-verify-split-outputs.log" bash -c '
set -euo pipefail
for f in "$@"; do
  test -f "$f" || { echo "MISSING $f"; exit 1; }
done
sha256sum "$@"
' _ \
    "$OUT/split/flux/$DEVICE/split_control.npz" \
    "$OUT/split/flux/$DEVICE/split_control.json" \
    "$OUT/split/pytorch/$DEVICE/split_control.npz" \
    "$OUT/split/pytorch/$DEVICE/split_control.json" \
    "${shard_ref_files[@]}" || return $?
  return 0
}

stage_e() {
  local W
  for W in $WORLD_SIZES; do
    local -a args=(
      --fixture-dir "$FIX"
      --bundle "$BUNDLE"
      --mode ddp
      --device "$DEVICE"
      --backend "$FLUX_BACKEND"
      --world-size "$W"
      --out "$OUT"
    )
    args+=("${FLUX_PROV_ARGS[@]}")
    run_cmd e "flux-ddp-w$W" "$LOG_DIR/e-flux-ddp-w$W.log" \
      bash -c "$FLUX_BASH" _ "$JULIA_BINDIR" "$JULIA_DEPOT_PATH" "$JULIA_LOAD_PATH" "$JULIA_NUM_THREADS" \
      "$MPIEXECJL" --project="$FLUX_DIR/test" -n "$W" \
      "$JULIA_BIN" --startup-file=no "$FLUX_RUNNER" "${args[@]}" || return $?
    run_cmd e "verify-flux-ddp-w$W" "$LOG_DIR/e-verify-flux-ddp-w$W.log" bash -c '
set -euo pipefail
test -d "$1" || { echo "MISSING directory $1"; exit 1; }
find "$1" -maxdepth 1 -type f -print | sort
' _ "$DDP_ROOT/flux/w$W" || return $?
  done
  return 0
}

stage_f() {
  local W PORT
  for W in $WORLD_SIZES; do
    PORT=$((TORCH_MASTER_PORT_BASE + W))
    if [ "$PORT" -gt 65535 ]; then
      echo "ERROR: MASTER_PORT $PORT for world $W exceeds 65535"
      return 1
    fi
    local -a args=(
      --fixture-dir "$FIX"
      --bundle "$BUNDLE"
      --mode ddp
      --device "$DEVICE"
      --backend "$TORCH_BACKEND"
      --world-size "$W"
      "$TORCH_OUT_FLAG" "$OUT"
    )
    args+=("${TORCH_PROV_ARGS[@]}")
    run_cmd f "pytorch-ddp-w$W" "$LOG_DIR/f-pytorch-ddp-w$W.log" \
      bash -c "$TORCH_DDP_BASH" _ "$OMP_NUM_THREADS" "$PORT" \
      "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$W" \
      "$TORCH_RUNNER" "${args[@]}" || return $?
    run_cmd f "verify-pytorch-ddp-w$W" "$LOG_DIR/f-verify-pytorch-ddp-w$W.log" bash -c '
set -euo pipefail
test -d "$1" || { echo "MISSING directory $1"; exit 1; }
find "$1" -maxdepth 1 -type f -print | sort
' _ "$DDP_ROOT/pytorch/w$W" || return $?
  done
  return 0
}

stage_g() {
  local -a args=(
    --fixture-dir "$FIX"
    --bundle "$BUNDLE"
    "$CHECK_TORCH_FLAG" "$TORCH_REF"
    --flux-reference "$FLUX_REF"
    "$CHECK_DDP_FLAG" "$DDP_ROOT"
  )
  run_cmd g "phase4-checker" "$LOG_DIR/g-phase4-checker.log" \
    "$VENV_PY" "$CHECKER" "${args[@]}" || return $?
  run_cmd g "checker-verdict" "$LOG_DIR/g-checker-verdict.log" bash -c '
set -euo pipefail
grep -Eq "^PHASE4_CHECK PASS( |$)" "$1" || { echo "MISSING PHASE4_CHECK PASS in $1"; exit 1; }
test -f "$2" || { echo "MISSING report $2"; exit 1; }
test -f "$3" || { echo "MISSING report $3"; exit 1; }
test -f "$4" || { echo "MISSING report $4"; exit 1; }
test -f "$5" || { echo "MISSING report $5"; exit 1; }
sha256sum "$2" "$3" "$4" "$5"
' _ "$LOG_DIR/g-phase4-checker.log" "$CHECK_DIR/phase4_report.txt" "$CHECK_DIR/phase4_report.json" "$CHECK_DIR/phase4_check_report.json" "$CHECK_DIR/phase4_check_summary.md" || return $?
  return 0
}

write_artifact_hashes() {
  local out=$LOG_DIR/artifact-sha256.txt
  if [ -d "$OUT" ]; then
    {
      find "$OUT" -path "$LOG_DIR" -prune -o -type f \
        \( -name '*.npz' -o -name '*.json' -o -name '*.txt' -o -name '*.md' \) -print0 \
        | sort -z | xargs -0 -r sha256sum
    } > "$out"
  else
    : > "$out"
  fi
}

write_metadata() {
  local end_utc final driver_sha
  end_utc=$(utc_now)
  final=FAIL
  if [ "$FAILED" -eq 0 ]; then final=PASS; fi

  driver_sha=$(sha256sum "$0" | cut -d' ' -f1)
  write_artifact_hashes

  {
    echo "# Commands - static-scaling Phase 4 deathstar CPU/MPI smoke"
    echo
    echo "Run: deathstar workstation, driver \`$0\`."
    echo "No result in this file is inferred: every listed command was executed by the driver."
    echo "GPUs were not used. \`nvidia-smi\` ran only as a courtesy check."
    echo
    echo "| Item | Value |"
    echo "|---|---|"
    echo "| Host | \`$(hostname)\` |"
    echo "| Start (UTC) | \`$START_UTC\` |"
    echo "| End (UTC) | \`$end_utc\` |"
    echo "| Driver SHA-256 | \`$driver_sha\` |"
    echo "| Device / backends | \`$DEVICE\` / Flux \`$FLUX_BACKEND\`, PyTorch \`$TORCH_BACKEND\` |"
    echo "| World sizes | \`$WORLD_SIZES\` |"
    echo "| CUDA_VISIBLE_DEVICES | \`${CUDA_VISIBLE_DEVICES:-<unset>}\` |"
    echo "| OUT | \`$OUT\` |"
    echo "| LOG_DIR | \`$LOG_DIR\` |"
    echo "| Status | **$final** |"
    echo
    echo "## Stage status"
    echo
    echo "| Stage | Status | Note |"
    echo "|---|---|---|"
    if [ -f "$STAGE_STATUS" ]; then
      while IFS=$'\t' read -r stage status note; do
        [ -n "$stage" ] || continue
        printf '| %s | %s | %s |\n' "$stage" "$status" "$note"
      done < "$STAGE_STATUS"
    fi
    echo
    echo "## Environment"
    echo
    echo '```text'
    echo "THESIS=$THESIS"
    echo "FLUX_DIR=$FLUX_DIR"
    echo "VENV_PY=$VENV_PY"
    echo "TORCHRUN_BIN=$TORCHRUN_BIN"
    echo "JULIA_BIN=$JULIA_BIN"
    echo "JULIA_BINDIR=$JULIA_BINDIR"
    echo "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"
    echo "JULIA_LOAD_PATH=$JULIA_LOAD_PATH"
    echo "MPIEXECJL=$MPIEXECJL"
    echo "FIX=$FIX"
    echo "BUNDLE=$BUNDLE"
    echo "FLUX_REF=$FLUX_REF"
    echo "TORCH_REF=$TORCH_REF"
    echo "DDP_ROOT=$DDP_ROOT"
    echo "CHECK_DIR=$CHECK_DIR"
    echo "SOURCE_COMMIT=${SOURCE_COMMIT:-<unset>}"
    echo "SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-<unset>}"
    echo "FLUX_COMMIT=${FLUX_COMMIT:-<unset>}"
    echo "FLUX_STATE_SHA256=${FLUX_STATE_SHA256:-<unset>}"
    echo '```'
    echo
    echo "## Exact command transcript"
    echo
    echo '```text'
    if [ -s "$COMMANDS_TRANSCRIPT" ]; then
      cat "$COMMANDS_TRANSCRIPT"
    else
      echo "(no commands executed)"
    fi
    echo '```'
    echo
    echo "## Evidence produced"
    echo
    echo "- Stage logs: \`$LOG_DIR/*.log\`"
    echo "- Source and input hashes: \`$LOG_DIR/source-sha256.txt\`"
    echo "- Artifact hashes: \`$LOG_DIR/artifact-sha256.txt\`"
    echo "- Flux reference: \`$FLUX_REF\`"
    echo "- PyTorch reference: \`$TORCH_REF\`"
    echo "- DDP root: \`$DDP_ROOT\`"
    echo "- Checker reports: \`$CHECK_DIR/phase4_report.txt\`, \`$CHECK_DIR/phase4_report.json\`"
    echo
    echo "## Copy evidence home (not executed by the driver)"
    echo
    echo "The evidence root is on deathstar under \`$OUT\`. From the operator workstation:"
    echo '```bash'
    echo "rsync -avz deathstar:'$OUT/' artifacts/logs/static-scaling/2026-10-18-phase4-deathstar-cpu/"
    echo '```'
  } > "$LOG_DIR/COMMANDS.md"

  {
    echo "# Provenance - static-scaling Phase 4 deathstar CPU/MPI smoke"
    echo
    echo "All values below were captured by the driver from the live run; no result is estimated."
    echo "Missing optional provenance is printed as \`<unset>\`, not invented."
    echo
    echo "| Item | Value |"
    echo "|---|---|"
    echo "| Host | \`$(hostname)\` |"
    echo "| Start (UTC) | \`$START_UTC\` |"
    echo "| End (UTC) | \`$end_utc\` |"
    echo "| Driver | \`$0\` |"
    echo "| Driver SHA-256 | \`$driver_sha\` |"
    echo "| CUDA_VISIBLE_DEVICES | \`${CUDA_VISIBLE_DEVICES:-<unset>}\` |"
    echo "| Status | **$final** |"
    echo
    echo "## Source identity (provided by the operator)"
    echo
    echo "| Field | Value |"
    echo "|---|---|"
    echo "| \`SOURCE_COMMIT\` | \`${SOURCE_COMMIT:-<unset>}\` |"
    echo "| \`SOURCE_STATE_SHA256\` | \`${SOURCE_STATE_SHA256:-<unset>}\` |"
    echo "| \`FLUX_COMMIT\` | \`${FLUX_COMMIT:-<unset>}\` |"
    echo "| \`FLUX_STATE_SHA256\` | \`${FLUX_STATE_SHA256:-<unset>}\` |"
    echo
    echo "## Source and input SHA-256"
    echo
    if [ -s "$LOG_DIR/source-sha256.txt" ]; then
      echo '```text'
      cat "$LOG_DIR/source-sha256.txt"
      echo '```'
    else
      echo "NOT AVAILABLE: stage a did not produce \`source-sha256.txt\`."
    fi
    echo
    echo "## Tool versions"
    echo
    echo "The complete tool/version transcript is \`$LOG_DIR/a-source-metadata.log\`."
    echo "The courtesy GPU check is \`$LOG_DIR/00-nvidia-smi.log\`; it is not training evidence."
    echo
    if [ -f "$LOG_DIR/a-source-metadata.log" ]; then
      echo '```text'
      grep -E '^(python |numpy |torch |cuda_available |cudnn |julia version |HYDRA|MPICH|mpiexec)' \
        "$LOG_DIR/a-source-metadata.log" || true
      echo '```'
    else
      echo "NOT AVAILABLE: \`a-source-metadata.log\` was not produced."
    fi
    echo
    echo "## Artifact SHA-256"
    echo
    if [ -s "$LOG_DIR/artifact-sha256.txt" ]; then
      echo '```text'
      cat "$LOG_DIR/artifact-sha256.txt"
      echo '```'
    else
      echo "NOT AVAILABLE: no artifact hash file was produced."
    fi
    echo
    echo "## Failure record"
    echo
    if [ "$FAILED" -eq 0 ]; then
      echo "No stage failed."
    else
      echo "Stage \`$FAIL_STAGE\` failed: $FAIL_REASON"
      echo "The log for the failing command is referenced in \`COMMANDS.md\`."
    fi
  } > "$LOG_DIR/PROVENANCE.md"

  printf '%s\n' "PHASE4_DEATHSTAR_CHECK $final" > "$LOG_DIR/status.txt"
}

run() {
  START_UTC=$(utc_now)
  : > "$COMMANDS_TRANSCRIPT"
  : > "$STAGE_STATUS"

  echo "=== run-phase4-cpu (deathstar): START $START_UTC ==="
  echo "host=$(hostname)"
  echo "THESIS=$THESIS"
  echo "FLUX_DIR=$FLUX_DIR"
  echo "FIX=$FIX"
  echo "BUNDLE=$BUNDLE"
  echo "OUT=$OUT"
  echo "LOG_DIR=$LOG_DIR"
  echo "DEVICE=$DEVICE FLUX_BACKEND=$FLUX_BACKEND TORCH_BACKEND=$TORCH_BACKEND"
  echo "WORLD_SIZES=$WORLD_SIZES"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} (GPUs are not used)"
  echo "SOURCE_COMMIT=${SOURCE_COMMIT:-<unset>} SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-<unset>}"
  echo "FLUX_COMMIT=${FLUX_COMMIT:-<unset>} FLUX_STATE_SHA256=${FLUX_STATE_SHA256:-<unset>}"

  if [ -n "$SOURCE_COMMIT$SOURCE_STATE_SHA256$FLUX_COMMIT$FLUX_STATE_SHA256" ]; then
    mapfile -t FLUX_PROV_ARGS < <(provenance_args "$FLUX_RUNNER")
    mapfile -t TORCH_PROV_ARGS < <(provenance_args "$TORCH_RUNNER")
    mapfile -t PY_PROV_ARGS < <(provenance_args "$SCRIPTS/make_phase3_reference.py")
  fi

  echo "=== STAGE 0 :: nvidia-smi courtesy check (GPUs will not be used) ==="
  if run_cmd 0 "nvidia-smi-courtesy" "$LOG_DIR/00-nvidia-smi.log" nvidia-smi; then
    set_pass 0 "nvidia-smi courtesy check; GPUs not used"
  else
    echo "WARNING: nvidia-smi failed; continuing because this driver never uses GPUs"
    record_status 0 WARN "nvidia-smi failed; GPUs not used"
  fi

  echo "=== STAGE 1 :: preflight ==="
  set +e
  preflight > "$LOG_DIR/01-preflight.log" 2>&1
  pre_status=$?
  set -e
  cat "$LOG_DIR/01-preflight.log"
  if [ "$pre_status" -eq 0 ]; then
    set_pass 1 "preflight"
  else
    set_fail 1 "preflight checks failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if stage_a; then set_pass a "source, versions, hashes"; else set_fail a "source/metadata stage failed"; fi
  else
    set_skip a "preflight failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if stage_b; then set_pass b "Flux CPU reference"; else set_fail b "Flux reference failed"; fi
  else
    set_skip b "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if stage_c; then set_pass c "PyTorch CPU reference"; else set_fail c "PyTorch reference stage failed"; fi
  else
    set_skip c "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if stage_d; then set_pass d "split-control both frameworks"; else set_fail d "split-control failed"; fi
  else
    set_skip d "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if stage_e; then set_pass e "Flux DDP worlds: $WORLD_SIZES"; else set_fail e "Flux DDP failed"; fi
  else
    set_skip e "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if stage_f; then set_pass f "PyTorch DDP worlds: $WORLD_SIZES"; else set_fail f "PyTorch DDP failed"; fi
  else
    set_skip f "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if stage_g; then set_pass g "checker PHASE4_CHECK PASS"; else set_fail g "checker failed"; fi
  else
    set_skip g "earlier stage failed"
  fi

  precomp_hits=$(grep -Fl 'Precompiling packages...' "$LOG_DIR"/*.log 2>/dev/null | grep -v '/driver.log$' || true)
  if [ -n "$precomp_hits" ]; then
    echo "ERROR: 'Precompiling packages...' found in stage logs:"
    echo "$precomp_hits"
    set_fail h "precompilation marker found in stage logs"
  fi

  write_metadata
  if [ ! -s "$LOG_DIR/COMMANDS.md" ] || [ ! -s "$LOG_DIR/PROVENANCE.md" ]; then
    FAILED=1
    FAIL_STAGE=h
    FAIL_REASON="metadata files are missing or empty"
    printf '%s\n' "PHASE4_DEATHSTAR_CHECK FAIL" > "$LOG_DIR/status.txt" 2>/dev/null || true
  fi

  echo "=== summary ==="
  if [ -f "$STAGE_STATUS" ]; then
    while IFS=$'\t' read -r stage status note; do
      [ -n "$stage" ] || continue
      echo "STAGE $stage $status $note"
    done < "$STAGE_STATUS"
  fi
  if [ "$FAILED" -eq 0 ]; then
    echo "PHASE4_DEATHSTAR_CHECK PASS"
    echo "OUT=$OUT"
    echo "LOG_DIR=$LOG_DIR"
    return 0
  fi
  echo "PHASE4_DEATHSTAR_CHECK FAIL stage=$FAIL_STAGE reason=$FAIL_REASON"
  echo "OUT=$OUT"
  echo "LOG_DIR=$LOG_DIR"
  return 1
}

set +e
run 2>&1 | tee -a "$DRIVER_LOG"
status=${PIPESTATUS[0]}
set -e
echo "=== run-phase4-cpu (deathstar): EXIT $status (log: $DRIVER_LOG) ==="
exit "$status"

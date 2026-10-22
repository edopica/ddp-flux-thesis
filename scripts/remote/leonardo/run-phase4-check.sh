#!/usr/bin/env bash
# =============================================================================
# Leonardo (CINECA) COMPUTE-NODE Phase 4 static-graph distributed correctness
# driver (Flux NCCL + PyTorch NCCL).
#
# ALLOCATION (from the thesis root on a login node):
#   bash scripts/remote/run.sh leonardo \
#     "srun -n1 -c32 --chdir=\$HOME/projects/ddp-flux-thesis \
#      bash scripts/remote/leonardo/run-phase4-check.sh"
#
# The outer step is one task with 32 CPUs; this driver launches a nested srun
# for every stage. Nested steps use --overlap so they can share the outer
# step's CPU allocation; stages run sequentially, never concurrently. The
# driver never precompiles and never calls sync_code.sh or
# sync_flux_integration.sh. If flux-integration is stale, run
# scripts/remote/sync_flux_integration.sh leonardo manually, then precompile
# with scripts/remote/leonardo/precompile-phase3.sh on the compute node.
#
# Stages (fail-fast, one log per command under LOG_DIR):
#   a. source identity/hashes, fixture/reference hashes, GPU/Slurm/NCCL/CUDA
#      versions
#   b. Flux single-GPU reference (world 1)
#   c. PyTorch single-GPU reference via make_phase3_reference.py, then a
#      SHA-256 bitwise identity check against the preserved reference
#   d. split-control for both frameworks (single GPU)
#   e. Flux DDP for each world in WORLD_SIZES
#   f. PyTorch DDP via torchrun for each world in WORLD_SIZES
#   g. check_phase4.py over all worlds; requires PHASE4_CHECK PASS
#   h. COMMANDS.md / PROVENANCE.md plus source and artifact hashes
#
# Runner CLI used here (contract temp/phase4-implementation-contract.md):
#   phase4_flux.jl  --fixture-dir FIX --bundle BUNDLE --mode MODE
#                   --device DEVICE --backend BACKEND --world-size W --out OUT
#   phase4_torch.py --fixture-dir FIX --bundle BUNDLE --mode MODE
#                   --device DEVICE --backend BACKEND --world-size W
#                   --out-dir OUT (or --out, auto-detected)
#   check_phase4.py --fixture-dir FIX --bundle BUNDLE
#                   --pytorch-reference|--torch-reference FILE
#                   --flux-reference FILE
#                   --ddp-root|--ddp-dir DIR
# Optional --source-* provenance flags are forwarded only when the runner
# source advertises them. Missing required flags fail preflight instead of
# guessing beyond the contract.
#
# Never pass the `env` exec token to srun on Leonardo; variables are set
# inside `bash -c 'export ...; exec ...'`. If any log contains
# "Precompiling packages...", the run fails.
#
# Shared writes are performed only by this driver (single task). Runners are
# responsible for rank-0-only shared writes of their own artifacts.
# =============================================================================
set -euo pipefail

WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
VENV_PY=${VENV_PY:-$WORK/epicazio/venvs/pytorch-scaling/bin/python}
if [ -z "${TORCHRUN_BIN:-}" ]; then
  if [ -x "$(dirname "$VENV_PY")/torchrun" ]; then
    TORCHRUN_BIN=$(dirname "$VENV_PY")/torchrun
  else
    TORCHRUN_BIN=torchrun
  fi
fi
JULIA_BIN=${JULIA_BIN:-$WORK/epicazio/julia-1.12.6/bin/julia}
JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$WORK/epicazio/julia-depot}
FLUX_DIR=${FLUX_DIR:-$HOME/projects/flux-integration}
FIX=${FIX:-$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2}
BUNDLE=${BUNDLE:-$FIX/initial_bundle.npz}
PY_REF=${PY_REF:-$THESIS/artifacts/baselines/static-scaling/2026-10-18-phase3-final/phase3_reference.npz}
PY_MANIFEST=${PY_MANIFEST:-$THESIS/artifacts/baselines/static-scaling/2026-10-18-phase3-final/phase3_reference_manifest.json}
OUT=${OUT:-$WORK/epicazio/static-scaling-evidence/2026-10-18-phase4}
LOG_DIR=${LOG_DIR:-$OUT/logs}
DEVICE=${DEVICE:-cuda}
WORLD_SIZES=${WORLD_SIZES:-"1 2 4"}
TOTAL_CPUS=${TOTAL_CPUS:-${SLURM_CPUS_PER_TASK:-32}}
REF_CPUS=${REF_CPUS:-8}
TORCH_MASTER_PORT_BASE=${TORCH_MASTER_PORT_BASE:-29600}
ALLOW_LOGIN_NODE=${ALLOW_LOGIN_NODE:-0}
ALLOW_EXISTING_OUT=${ALLOW_EXISTING_OUT:-0}
SOURCE_COMMIT=${SOURCE_COMMIT:-}
SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-}
FLUX_COMMIT=${FLUX_COMMIT:-}
FLUX_STATE_SHA256=${FLUX_STATE_SHA256:-}

SCRIPTS=$THESIS/scripts/static_scaling
FLUX_RUNNER=${FLUX_RUNNER:-$SCRIPTS/phase4_flux.jl}
TORCH_RUNNER=${TORCH_RUNNER:-$SCRIPTS/phase4_torch.py}
CHECKER=${CHECKER:-$SCRIPTS/check_phase4.py}
FLUX_REF=${FLUX_REF:-$OUT/reference/flux/$DEVICE/phase4_reference.npz}
FLUX_REF_JSON=${FLUX_REF_JSON:-$OUT/reference/flux/$DEVICE/phase4_reference.json}
TORCH_REF_DIR=${TORCH_REF_DIR:-$OUT/reference/pytorch/$DEVICE}
TORCH_REF=${TORCH_REF:-$TORCH_REF_DIR/phase3_reference.npz}
TORCH_MANIFEST=${TORCH_MANIFEST:-$TORCH_REF_DIR/phase3_reference_manifest.json}
DDP_ROOT=${DDP_ROOT:-$OUT/ddp}
CHECK_DIR=${CHECK_DIR:-$OUT/check}
JULIA_ENV=$SCRIPTS/julia-env
JULIA_LOAD_PATH=${JULIA_LOAD_PATH:-$JULIA_ENV:$FLUX_DIR/test:@stdlib}
JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}

if [ -z "${BACKEND:-}" ]; then
  if [ "$DEVICE" = cuda ]; then BACKEND=nccl; else BACKEND=mpi; fi
fi

case "$DEVICE" in
  cpu|cuda) ;;
  *) echo "ERROR: DEVICE must be cpu or cuda, got '$DEVICE'" >&2; exit 1 ;;
esac

case "$BACKEND" in
  nccl)
    FLUX_BACKEND=${FLUX_BACKEND:-nccl}
    TORCH_BACKEND=${TORCH_BACKEND:-nccl}
    ;;
  mpi|gloo)
    FLUX_BACKEND=${FLUX_BACKEND:-mpi}
    TORCH_BACKEND=${TORCH_BACKEND:-gloo}
    ;;
  *)
    echo "ERROR: BACKEND must be nccl, mpi, or gloo, got '$BACKEND'" >&2
    exit 1
    ;;
esac

if [ "$DEVICE" = cpu ] && [ "$TORCH_BACKEND" = nccl ]; then
  echo "ERROR: DEVICE=cpu with TORCH_BACKEND=nccl is invalid" >&2
  exit 1
fi
if [ "$DEVICE" = cuda ] && [ "$BACKEND" = mpi ]; then
  echo "ERROR: DEVICE=cuda with BACKEND=mpi is not supported by this driver" >&2
  exit 1
fi

if [ -z "${REQUIRE_PY_REF_IDENTITY:-}" ]; then
  if [ "$DEVICE" = cuda ]; then REQUIRE_PY_REF_IDENTITY=1; else REQUIRE_PY_REF_IDENTITY=0; fi
fi
if [ "$TORCH_BACKEND" = nccl ]; then
  TORCH_NCCL_DEBUG=${TORCH_NCCL_DEBUG:-INFO}
else
  TORCH_NCCL_DEBUG=${TORCH_NCCL_DEBUG:-}
fi
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

is_positive_int() {
  case "${1:-}" in
    ''|*[!0-9]*) return 1 ;;
    *) [ "$1" -gt 0 ] ;;
  esac
}

is_positive_int "$TOTAL_CPUS" || { echo "ERROR: TOTAL_CPUS must be a positive integer" >&2; exit 1; }
is_positive_int "$REF_CPUS" || { echo "ERROR: REF_CPUS must be a positive integer" >&2; exit 1; }
if [ "$REF_CPUS" -gt "$TOTAL_CPUS" ]; then REF_CPUS=$TOTAL_CPUS; fi
if [ "$TORCH_MASTER_PORT_BASE" -lt 1024 ] || [ "$TORCH_MASTER_PORT_BASE" -gt 65535 ]; then
  echo "ERROR: TORCH_MASTER_PORT_BASE must be in 1024..65535" >&2
  exit 1
fi
for _world in $WORLD_SIZES; do
  is_positive_int "$_world" || { echo "ERROR: invalid WORLD_SIZES entry '$_world'" >&2; exit 1; }
  if [ "$DEVICE" = cuda ] && [ "$_world" -gt 4 ]; then
    echo "ERROR: world size $_world exceeds the 4 allocated A100 GPUs" >&2
    exit 1
  fi
done
unset _world

# ---- compute-node and single-task guards -----------------------------------
host=$(hostname)
case "$host" in
  login*)
    if [ "$ALLOW_LOGIN_NODE" != 1 ]; then
      echo "ERROR: running on login node '$host': run under srun on a compute node (or set ALLOW_LOGIN_NODE=1 to override on purpose)" >&2
      exit 1
    fi
    echo "WARNING: ALLOW_LOGIN_NODE=1: running on login node '$host'" >&2
    ;;
esac

if [ "${SLURM_NTASKS:-1}" != "1" ] || [ "${SLURM_PROCID:-0}" != "0" ]; then
  echo "ERROR: this driver must run as one task (srun -n1); got SLURM_NTASKS=${SLURM_NTASKS:-<unset>} SLURM_PROCID=${SLURM_PROCID:-<unset>}" >&2
  exit 1
fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "$ALLOW_LOGIN_NODE" != 1 ]; then
  echo "ERROR: SLURM_JOB_ID is unset; launch inside the documented srun allocation" >&2
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
STAGE=${STAGE:-}
if [ -n "$STAGE" ]; then
  ALLOW_EXISTING_OUT=1
fi
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null || true)" ] && [ "$ALLOW_EXISTING_OUT" != 1 ]; then
  echo "ERROR: OUT exists and is not empty: $OUT" >&2
  echo "       use a fresh OUT/LOG_DIR or set ALLOW_EXISTING_OUT=1 to reuse it" >&2
  exit 1
fi
mkdir -p "$OUT" "$LOG_DIR"
DRIVER_LOG=$LOG_DIR/driver.log
if [ "$ALLOW_EXISTING_OUT" != 1 ]; then
  : > "$DRIVER_LOG"
fi

export WORK THESIS VENV_PY TORCHRUN_BIN JULIA_BIN JULIA_DEPOT_PATH FLUX_DIR FIX BUNDLE
export PY_REF PY_MANIFEST OUT LOG_DIR DEVICE BACKEND FLUX_BACKEND TORCH_BACKEND
export WORLD_SIZES TOTAL_CPUS REF_CPUS TORCH_MASTER_PORT_BASE TORCH_NCCL_DEBUG
export OMP_NUM_THREADS JULIA_LOAD_PATH JULIA_NUM_THREADS JULIA_ENV
export SOURCE_COMMIT SOURCE_STATE_SHA256 FLUX_COMMIT FLUX_STATE_SHA256
export SCRIPTS FLUX_RUNNER TORCH_RUNNER CHECKER FLUX_REF FLUX_REF_JSON
export TORCH_REF_DIR TORCH_REF TORCH_MANIFEST DDP_ROOT CHECK_DIR
export JULIA_PKG_PRECOMPILE_AUTO=0

# Bash prelude for Flux stages: no `env` token, all variables set inside bash.
FLUX_BASH='unset JULIA_PROJECT; export JULIA_DEPOT_PATH="$1" JULIA_LOAD_PATH="$2" JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_THREADS="$3"; shift 3; exec "$@"'
# Bash prelude for single-process PyTorch stages.
TORCH_ENV_BASH='export OMP_NUM_THREADS="$1" MKL_NUM_THREADS="$1"; shift; exec "$@"'
# Bash prelude for torchrun stages (unique MASTER_PORT per world).
TORCH_DDP_BASH='export OMP_NUM_THREADS="$1" MKL_NUM_THREADS="$1" TORCH_NCCL_DEBUG="$2" MASTER_PORT="$3"; shift 3; exec "$@"'

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

die() { echo "ERROR: $*" >&2; exit 1; }

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
  if [ "$REQUIRE_PY_REF_IDENTITY" = 1 ]; then
    required+=("$PY_REF" "$PY_MANIFEST")
  fi
  for path in "${required[@]}"; do
    if [ ! -f "$path" ]; then
      echo "MISSING file: $path"
      missing=1
    fi
  done
  for path in "$VENV_PY" "$TORCHRUN_BIN" "$JULIA_BIN"; do
    if [ ! -x "$path" ]; then
      echo "MISSING executable: $path"
      missing=1
    fi
  done
  if [ ! -d "$FLUX_DIR/test" ]; then
    echo "MISSING directory: $FLUX_DIR/test (run scripts/remote/sync_flux_integration.sh leonardo)"
    missing=1
  fi
  if ! command -v srun >/dev/null 2>&1; then
    echo "MISSING executable: srun"
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
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-<unset>}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}"
echo "SLURM_PROCID=${SLURM_PROCID:-<unset>}"
echo "THESIS=$THESIS"
echo "FLUX_DIR=$FLUX_DIR"
echo "FIX=$FIX"
echo "BUNDLE=$BUNDLE"
echo "PY_REF=$PY_REF"
echo "PY_MANIFEST=$PY_MANIFEST"
echo "OUT=$OUT"
echo "LOG_DIR=$LOG_DIR"
echo "DEVICE=$DEVICE BACKEND=$BACKEND FLUX_BACKEND=$FLUX_BACKEND TORCH_BACKEND=$TORCH_BACKEND"
echo "WORLD_SIZES=$WORLD_SIZES TOTAL_CPUS=$TOTAL_CPUS REF_CPUS=$REF_CPUS"
echo "SOURCE_COMMIT=${SOURCE_COMMIT:-<unset>} SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-<unset>}"
echo "FLUX_COMMIT=${FLUX_COMMIT:-<unset>} FLUX_STATE_SHA256=${FLUX_STATE_SHA256:-<unset>}"
if command -v git >/dev/null 2>&1; then
  echo "THESIS_GIT=$(git -C "$THESIS" rev-parse HEAD 2>/dev/null || echo "<no-git>")"
  echo "FLUX_GIT=$(git -C "$FLUX_DIR" rev-parse HEAD 2>/dev/null || echo "<no-git>")"
else
  echo "THESIS_GIT=<no-git-command>"
  echo "FLUX_GIT=<no-git-command>"
fi
echo "PROCESS_COMMAND=$(tr "\0" " " < /proc/$$/cmdline 2>/dev/null || echo "<unavailable>")"
echo "PARENT_COMMAND=$(ps -o args= -p "$PPID" 2>/dev/null || echo "<unavailable>")"
echo "--- srun ---"
srun --version
if [ "$DEVICE" = cuda ]; then
  echo "--- nvidia-smi ---"
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
fi
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
try:
    print("nccl", torch.cuda.nccl.version())
except Exception as exc:
    print("nccl unavailable:", exc)
PY
echo "--- julia ---"
"$JULIA_BIN" --version
echo "--- hash manifest ---"
{
  echo "# static_scaling sources"
  find "$THESIS/scripts/static_scaling" -maxdepth 1 -type f \( -name "*.jl" -o -name "*.py" \) -print0 | sort -z | xargs -0 sha256sum
  echo "# remote drivers"
  for f in "$THESIS/scripts/remote/leonardo/run-phase4-check.sh" "$THESIS/scripts/remote/leonardo/precompile-phase3.sh"; do
    if [ -f "$f" ]; then sha256sum "$f"; else echo "MISSING $f"; fi
  done
  echo "# inputs"
  sha256sum "$FIX/data_fixture.npz" "$FIX/fixture_manifest.json" "$FIX/schedule_fixture.npz" "$BUNDLE"
  for f in "$PY_REF" "$PY_MANIFEST"; do
    if [ -f "$f" ]; then sha256sum "$f"; else echo "MISSING $f"; fi
  done
  echo "# environments"
  for f in "$JULIA_ENV/Project.toml" "$JULIA_ENV/Manifest.toml" "$FLUX_DIR/test/Project.toml" "$FLUX_DIR/test/Manifest.toml"; do
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
    srun --overlap --mpi=pmi2 -n1 -c"$REF_CPUS" --chdir="$THESIS" \
    bash -c "$FLUX_BASH" _ "$JULIA_DEPOT_PATH" "$JULIA_LOAD_PATH" "$JULIA_NUM_THREADS" \
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
    srun --overlap -n1 -c"$REF_CPUS" --chdir="$THESIS" \
    bash -c "$TORCH_ENV_BASH" _ "$OMP_NUM_THREADS" \
    "$VENV_PY" "$SCRIPTS/make_phase3_reference.py" "${args[@]}" || return $?
  run_cmd c "verify-pytorch-reference" "$LOG_DIR/c-verify-pytorch-reference.log" bash -c '
set -euo pipefail
test -f "$1"
test -f "$2"
sha256sum "$1" "$2"
' _ "$TORCH_REF" "$TORCH_MANIFEST" || return $?
  # Identity is checked on the reference NPZ (the arrays). The generated
  # manifest can legitimately differ in generation-time metadata.
  if [ "$REQUIRE_PY_REF_IDENTITY" = 1 ]; then
    run_cmd c "pytorch-reference-identity" "$LOG_DIR/c-reference-identity.log" bash -c '
set -euo pipefail
actual=$(sha256sum "$1" | cut -d" " -f1)
expected=$(sha256sum "$2" | cut -d" " -f1)
echo "preserved=$2"
echo "preserved_sha256=$expected"
echo "generated=$1"
echo "generated_sha256=$actual"
test "$actual" = "$expected"
' _ "$TORCH_REF" "$PY_REF" || return $?
  else
    echo "SKIP c::pytorch-reference-identity (REQUIRE_PY_REF_IDENTITY=0)"
  fi
  return 0
}

stage_d() {
  if [ "$ALLOW_EXISTING_OUT" = 1 ] && [ -f "$OUT/split/flux/$DEVICE/split_control.npz" ] && [ -f "$OUT/split/flux/$DEVICE/split_control.json" ]; then
    echo "REUSE d::flux-split-control ($OUT/split/flux/$DEVICE/split_control.npz)"
  else
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
      srun --overlap --mpi=pmi2 -n1 -c"$REF_CPUS" --chdir="$THESIS" \
      bash -c "$FLUX_BASH" _ "$JULIA_DEPOT_PATH" "$JULIA_LOAD_PATH" "$JULIA_NUM_THREADS" \
      "$JULIA_BIN" --startup-file=no "$FLUX_RUNNER" "${flux_args[@]}" || return $?
  fi

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
    srun --overlap -n1 -c"$REF_CPUS" --chdir="$THESIS" \
    bash -c "$TORCH_ENV_BASH" _ "$OMP_NUM_THREADS" \
    "$VENV_PY" "$TORCH_RUNNER" "${torch_args[@]}" || return $?

  local W
  local -a shard_ref_files=()
  for W in $WORLD_SIZES; do
    if [ "$ALLOW_EXISTING_OUT" = 1 ] && [ -f "$OUT/shard_reference/flux/$DEVICE/w$W/phase4_reference.npz" ] && [ -f "$OUT/shard_reference/flux/$DEVICE/w$W/phase4_reference.json" ]; then
      echo "REUSE d::flux-shard-reference-w$W ($OUT/shard_reference/flux/$DEVICE/w$W/phase4_reference.npz)"
    else
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
        srun --overlap --mpi=pmi2 -n1 -c"$REF_CPUS" --chdir="$THESIS" \
        bash -c "$FLUX_BASH" _ "$JULIA_DEPOT_PATH" "$JULIA_LOAD_PATH" "$JULIA_NUM_THREADS" \
        "$JULIA_BIN" --startup-file=no "$FLUX_RUNNER" "${flux_shard_args[@]}" || return $?
    fi

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
      srun --overlap -n1 -c"$REF_CPUS" --chdir="$THESIS" \
      bash -c "$TORCH_ENV_BASH" _ "$OMP_NUM_THREADS" \
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
  local W CPUS
  for W in $WORLD_SIZES; do
    CPUS=$((TOTAL_CPUS / W))
    if [ "$CPUS" -lt 1 ]; then
      echo "ERROR: world size $W needs more than $TOTAL_CPUS CPUs"
      return 1
    fi
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
      srun --overlap --mpi=pmi2 -n"$W" -c"$CPUS" --chdir="$THESIS" \
      bash -c "$FLUX_BASH" _ "$JULIA_DEPOT_PATH" "$JULIA_LOAD_PATH" "$JULIA_NUM_THREADS" \
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
  local W CPUS PORT
  for W in $WORLD_SIZES; do
    CPUS=$((TOTAL_CPUS / W))
    if [ "$CPUS" -lt 1 ]; then
      echo "ERROR: world size $W needs more than $TOTAL_CPUS CPUs"
      return 1
    fi
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
      srun --overlap -n1 -c"$CPUS" --chdir="$THESIS" \
      bash -c "$TORCH_DDP_BASH" _ "$OMP_NUM_THREADS" "$TORCH_NCCL_DEBUG" "$PORT" \
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
    --fp32-absolute-1e3
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
sha256sum "$2" "$3"
' _ "$LOG_DIR/g-phase4-checker.log" "$CHECK_DIR/phase4_report.txt" "$CHECK_DIR/phase4_report.json" || return $?
  return 0
}

write_artifact_hashes() {
  local out=$LOG_DIR/artifact-sha256.txt
  if [ -d "$OUT" ]; then
    {
      find "$OUT" -path "$LOG_DIR" -prune -o -type f \
        \( -name '*.npz' -o -name '*.json' -o -name '*.txt' \) -print0 \
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
    echo "# Commands - static-scaling Phase 4 remote check"
    echo
    echo "Run: Leonardo compute node, driver \`$0\`."
    echo "No result in this file is inferred: every listed command was executed by the driver."
    echo
    echo "| Item | Value |"
    echo "|---|---|"
    echo "| Host | \`$(hostname)\` |"
    echo "| SLURM job | \`${SLURM_JOB_ID:-<unset>}\` |"
    echo "| SLURM node list | \`${SLURM_JOB_NODELIST:-<unset>}\` |"
    echo "| Start (UTC) | \`$START_UTC\` |"
    echo "| End (UTC) | \`$end_utc\` |"
    echo "| Driver SHA-256 | \`$driver_sha\` |"
    echo "| Device / backend | \`$DEVICE\` / \`$BACKEND\` |"
    echo "| Flux / PyTorch backend | \`$FLUX_BACKEND\` / \`$TORCH_BACKEND\` |"
    echo "| World sizes | \`$WORLD_SIZES\` |"
    echo "| CPUs | \`$TOTAL_CPUS\` (reference stages \`$REF_CPUS\`) |"
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
    echo "JULIA_BIN=$JULIA_BIN"
    echo "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"
    echo "JULIA_LOAD_PATH=$JULIA_LOAD_PATH"
    echo "FIX=$FIX"
    echo "BUNDLE=$BUNDLE"
    echo "PY_REF=$PY_REF"
    echo "PY_MANIFEST=$PY_MANIFEST"
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
    echo '```bash'
    echo "rsync -avz leonardo:'$LOG_DIR/' artifacts/logs/static-scaling/2026-10-18-phase4/logs/"
    echo "rsync -avz leonardo:'$CHECK_DIR/' artifacts/logs/static-scaling/2026-10-18-phase4/check/"
    echo "rsync -avz leonardo:'$DDP_ROOT/' artifacts/logs/static-scaling/2026-10-18-phase4/ddp/"
    echo "rsync -avz leonardo:'$OUT/split/' artifacts/logs/static-scaling/2026-10-18-phase4/split/"
    echo "rsync -avz leonardo:'$OUT/reference/' artifacts/baselines/static-scaling/2026-10-18-phase4/reference/"
    echo '```'
  } > "$LOG_DIR/COMMANDS.md"

  {
    echo "# Provenance - static-scaling Phase 4 remote check"
    echo
    echo "All values below were captured by the driver from the live run; no result is estimated."
    echo "Missing optional provenance is printed as \`<unset>\`, not invented."
    echo
    echo "| Item | Value |"
    echo "|---|---|"
    echo "| Host | \`$(hostname)\` |"
    echo "| SLURM job | \`${SLURM_JOB_ID:-<unset>}\` |"
    echo "| SLURM node list | \`${SLURM_JOB_NODELIST:-<unset>}\` |"
    echo "| Start (UTC) | \`$START_UTC\` |"
    echo "| End (UTC) | \`$end_utc\` |"
    echo "| Driver | \`$0\` |"
    echo "| Driver SHA-256 | \`$driver_sha\` |"
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
    echo
    if [ -f "$LOG_DIR/a-source-metadata.log" ]; then
      echo '```text'
      grep -E '^(python |numpy |torch |cuda_available |cudnn |nccl |julia version |slurm )' \
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

  printf '%s\n' "PHASE4_REMOTE_CHECK $final" > "$LOG_DIR/status.txt"
}

should_run_stage() {
  local s=$1
  [ -z "${STAGE:-}" ] && return 0
  case ",${STAGE// /,}," in
    *,"$s",*) return 0 ;;
    *) return 1 ;;
  esac
}

run() {
  START_UTC=$(utc_now)
  if [ "$ALLOW_EXISTING_OUT" != 1 ]; then
    : > "$COMMANDS_TRANSCRIPT"
  fi
  : > "$STAGE_STATUS"

  echo "=== run-phase4-check: START $START_UTC ==="
  echo "host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>} SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
  echo "THESIS=$THESIS"
  echo "FLUX_DIR=$FLUX_DIR"
  echo "FIX=$FIX"
  echo "BUNDLE=$BUNDLE"
  echo "OUT=$OUT"
  echo "LOG_DIR=$LOG_DIR"
  echo "DEVICE=$DEVICE BACKEND=$BACKEND FLUX_BACKEND=$FLUX_BACKEND TORCH_BACKEND=$TORCH_BACKEND"
  echo "WORLD_SIZES=$WORLD_SIZES"
  echo "SOURCE_COMMIT=${SOURCE_COMMIT:-<unset>} SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-<unset>}"
  echo "FLUX_COMMIT=${FLUX_COMMIT:-<unset>} FLUX_STATE_SHA256=${FLUX_STATE_SHA256:-<unset>}"
  echo "REQUIRE_PY_REF_IDENTITY=$REQUIRE_PY_REF_IDENTITY"

  if [ -n "$SOURCE_COMMIT$SOURCE_STATE_SHA256$FLUX_COMMIT$FLUX_STATE_SHA256" ]; then
    mapfile -t FLUX_PROV_ARGS < <(provenance_args "$FLUX_RUNNER")
    mapfile -t TORCH_PROV_ARGS < <(provenance_args "$TORCH_RUNNER")
    mapfile -t PY_PROV_ARGS < <(provenance_args "$SCRIPTS/make_phase3_reference.py")
  fi

  echo "=== STAGE 0 :: preflight ==="
  set +e
  preflight > "$LOG_DIR/00-preflight.log" 2>&1
  pre_status=$?
  set -e
  cat "$LOG_DIR/00-preflight.log"
  if [ "$pre_status" -eq 0 ]; then
    set_pass 0 "preflight"
  else
    set_fail 0 "preflight checks failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if ! should_run_stage a; then set_pass a "source, versions, hashes (reused)"; elif stage_a; then set_pass a "source, versions, hashes"; else set_fail a "source/metadata stage failed"; fi
  else
    set_skip a "preflight failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if ! should_run_stage b; then set_pass b "Flux single-GPU reference (reused)"; elif stage_b; then set_pass b "Flux single-GPU reference"; else set_fail b "Flux reference failed"; fi
  else
    set_skip b "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if ! should_run_stage c; then set_pass c "PyTorch single-GPU reference and identity (reused)"; elif stage_c; then set_pass c "PyTorch single-GPU reference and identity"; else set_fail c "PyTorch reference stage failed"; fi
  else
    set_skip c "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if ! should_run_stage d; then set_pass d "split-control both frameworks (reused)"; elif stage_d; then set_pass d "split-control both frameworks"; else set_fail d "split-control failed"; fi
  else
    set_skip d "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if ! should_run_stage e; then set_pass e "Flux DDP worlds: $WORLD_SIZES (reused)"; elif stage_e; then set_pass e "Flux DDP worlds: $WORLD_SIZES"; else set_fail e "Flux DDP failed"; fi
  else
    set_skip e "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if ! should_run_stage f; then set_pass f "PyTorch DDP worlds: $WORLD_SIZES (reused)"; elif stage_f; then set_pass f "PyTorch DDP worlds: $WORLD_SIZES"; else set_fail f "PyTorch DDP failed"; fi
  else
    set_skip f "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if ! should_run_stage g; then set_pass g "checker PHASE4_CHECK PASS (reused)"; elif stage_g; then set_pass g "checker PHASE4_CHECK PASS"; else set_fail g "checker failed"; fi
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
    printf '%s\n' "PHASE4_REMOTE_CHECK FAIL" > "$LOG_DIR/status.txt" 2>/dev/null || true
  fi

  echo "=== summary ==="
  if [ -f "$STAGE_STATUS" ]; then
    while IFS=$'\t' read -r stage status note; do
      [ -n "$stage" ] || continue
      echo "STAGE $stage $status $note"
    done < "$STAGE_STATUS"
  fi
  if [ "$FAILED" -eq 0 ]; then
    echo "PHASE4_REMOTE_CHECK PASS"
    echo "OUT=$OUT"
    echo "LOG_DIR=$LOG_DIR"
    return 0
  fi
  echo "PHASE4_REMOTE_CHECK FAIL stage=$FAIL_STAGE reason=$FAIL_REASON"
  echo "OUT=$OUT"
  echo "LOG_DIR=$LOG_DIR"
  return 1
}

set +e
run 2>&1 | tee -a "$DRIVER_LOG"
status=${PIPESTATUS[0]}
set -e
echo "=== run-phase4-check: EXIT $status (log: $DRIVER_LOG) ==="
exit "$status"

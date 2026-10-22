#!/usr/bin/env bash
# =============================================================================
# Leonardo COMPUTE-NODE correctness driver for the current-vs-historical Flux
# static-DDP revision comparison.
#
# ALLOCATION (from the thesis root on a login node):
#   bash scripts/remote/run.sh leonardo \
#     "srun -n1 -c32 --chdir=\$HOME/projects/ddp-flux-thesis \
#      bash scripts/remote/leonardo/run-static-revision-check.sh"
#
# The outer step is one task with 32 CPUs. Every run uses a nested
# `srun --overlap` because stages are sequential and fail fast.
#
# This driver NEVER syncs and NEVER precompiles. Sync the thesis worktrees and
# precompile on a compute node before running it. If any log contains
# "Precompiling packages...", the run fails.
#
# Variants (same thesis runner, different Flux worktree and protocol):
#   current     $HOME/projects/flux-integration        --distributed-api guarded
#   historical  $HOME/projects/flux-static-baseline    --distributed-api static
#
# The gate is the FULL accepted Phase 4 checker, not a Flux-only subset:
#   check_phase4.py --frameworks flux,pytorch -> PHASE4_CHECK PASS categories=202
# PyTorch DDP, reference, split-control, and shard-reference evidence is reused
# read-only from ACCEPTED_PHASE4_ROOT. Each variant gets a fresh composite root:
#   $OUT/checkroot/<variant>/ddp/flux        -> $OUT/<variant>/ddp/flux
#   $OUT/checkroot/<variant>/ddp/pytorch     -> $PY_DDP
#   $OUT/checkroot/<variant>/shard_reference -> $ACCEPTED_SHARD_REF
# Symlinks only: accepted files are never copied, regenerated, or modified.
#
# Evidence is written only under a fresh $WORK output root. Accepted Phase 3/4
# baselines and campaign 61314790 are refused.
#
# Interface dependencies (implemented by the runner/checker agents):
#   phase4_flux.jl   --distributed-api guarded|static
#   check_phase4.py  --flux-collective-protocol guarded|static
# =============================================================================
set -euo pipefail

WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
FLUX_CURRENT=${FLUX_CURRENT:-$HOME/projects/flux-integration}
FLUX_HISTORICAL=${FLUX_HISTORICAL:-$HOME/projects/flux-static-baseline}
FIX=${FIX:-$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2}
BUNDLE=${BUNDLE:-$FIX/initial_bundle.npz}
REF=${REF:-$WORK/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/flux/cuda/phase4_reference.npz}
ACCEPTED_PHASE4_ROOT=${ACCEPTED_PHASE4_ROOT:-$WORK/epicazio/static-scaling-evidence/2026-10-18-phase4}
PY_REF=${PY_REF:-$ACCEPTED_PHASE4_ROOT/reference/pytorch/cuda/phase3_reference.npz}
PY_DDP=${PY_DDP:-$ACCEPTED_PHASE4_ROOT/ddp/pytorch}
ACCEPTED_SHARD_REF=${ACCEPTED_SHARD_REF:-$ACCEPTED_PHASE4_ROOT/shard_reference}
FLUX_SPLIT=${FLUX_SPLIT:-$ACCEPTED_PHASE4_ROOT/split/flux/cuda/split_control.npz}
PY_SPLIT=${PY_SPLIT:-$ACCEPTED_PHASE4_ROOT/split/pytorch/cuda/split_control.npz}
JOB_ID=${SLURM_JOB_ID:-manual}
OUT=${OUT:-$WORK/epicazio/static-scaling-evidence/2026-10-19-revision-check-$JOB_ID}
OUT=$(readlink -m "$OUT")
WORLD_SIZES=${WORLD_SIZES:-"1 2 4"}
TOTAL_CPUS=${TOTAL_CPUS:-${SLURM_CPUS_PER_TASK:-32}}
RUN_TIMEOUT=${RUN_TIMEOUT:-1800}
ALLOW_LOGIN_NODE=${ALLOW_LOGIN_NODE:-0}
ALLOW_EXISTING_OUT=${ALLOW_EXISTING_OUT:-0}
ALLOW_UNEXPECTED_FLUX=${ALLOW_UNEXPECTED_FLUX:-0}
ALLOW_DIRTY_FLUX=${ALLOW_DIRTY_FLUX:-0}
SOURCE_COMMIT=${SOURCE_COMMIT:-}
EXPECTED_CURRENT_COMMIT=${EXPECTED_CURRENT_COMMIT:-f7b4520c4bd307c67bbc5ac92a1c68493eccff6e}
EXPECTED_HISTORICAL_COMMIT=${EXPECTED_HISTORICAL_COMMIT:-8add0e42b7d9388ff365335152126e5b1e9de657}

SCRIPTS=$THESIS/scripts/static_scaling
RUNNER=$SCRIPTS/phase4_flux.jl
CHECKER=$SCRIPTS/check_phase4.py
JULIA_BIN=${JULIA_BIN:-$WORK/epicazio/julia-1.12.6/bin/julia}
JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$WORK/epicazio/julia-depot}
JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
DRIVER_PATH=$(readlink -f "$0")
VENV_PY=${VENV_PY:-$WORK/epicazio/venvs/pytorch-scaling/bin/python}
LOG_DIR=$OUT/logs
COMMANDS_TRANSCRIPT=$LOG_DIR/commands-transcript.txt
STAGE_STATUS=$LOG_DIR/stage-status.txt
DRIVER_LOG=$LOG_DIR/driver.log

THESIS_COMMIT=unknown
THESIS_SCRIPTS_STATE_SHA256=unknown
CURRENT_COMMIT=unknown
CURRENT_GIT_STATE_SHA256=unknown
CURRENT_SNAPSHOT_SHA256=unknown
HISTORICAL_COMMIT=unknown
HISTORICAL_GIT_STATE_SHA256=unknown
HISTORICAL_SNAPSHOT_SHA256=unknown
GPU_DRIVER=unavailable
SRUN_VERSION=unavailable
JULIA_VERSION=unavailable

is_positive_int() {
  case "${1:-}" in
    ''|*[!0-9]*) return 1 ;;
    *) [ "$1" -gt 0 ] ;;
  esac
}

utc_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
quote_cmd() { printf '%q ' "$@"; }

die() { echo "ERROR: $*" >&2; exit 1; }

refuse_protected_out() {
  case "$OUT" in
    "$THESIS/artifacts/baselines/"*|\
    "$THESIS/artifacts/logs/static-scaling/2026-10-17-phase3-attempt3-preserved"*|\
    "$THESIS/artifacts/logs/static-scaling/2026-10-18-phase3-final"*|\
    "$THESIS/artifacts/logs/static-scaling/2026-10-18-r1-r2"*|\
    "$THESIS/artifacts/logs/static-scaling/2026-10-18-phase4"*|\
    "$THESIS/artifacts/logs/static-scaling/2026-10-18-timing-pilot"*|\
    "$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-attempt3-preserved"*|\
    "$WORK/epicazio/static-scaling-evidence/2026-10-18-phase3-final"*|\
    "$WORK/epicazio/static-scaling-evidence/2026-10-18-r1-r2"*|\
    "$WORK/epicazio/static-scaling-evidence/2026-10-18-phase4"*|\
    "$WORK/epicazio/static-scaling-evidence/2026-10-18-timing-pilot"*|\
    *campaign-61314790*)
      echo "ERROR: OUT points at accepted evidence or campaign 61314790: $OUT" >&2
      exit 1
      ;;
  esac
}

# ---- guards ----------------------------------------------------------------
host=$(hostname)
case "$host" in
  login*)
    if [ "$ALLOW_LOGIN_NODE" != 1 ]; then
      echo "ERROR: running on login node '$host': run under srun on a compute node (or set ALLOW_LOGIN_NODE=1 on purpose)" >&2
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

is_positive_int "$TOTAL_CPUS" || { echo "ERROR: TOTAL_CPUS must be a positive integer" >&2; exit 1; }
for _world in $WORLD_SIZES; do
  is_positive_int "$_world" || { echo "ERROR: invalid WORLD_SIZES entry '$_world'" >&2; exit 1; }
  if [ "$_world" -gt 4 ]; then
    echo "ERROR: world size $_world exceeds the 4 allocated GPUs" >&2
    exit 1
  fi
  if [ "$((TOTAL_CPUS / _world))" -lt 1 ]; then
    echo "ERROR: world size $_world needs more than $TOTAL_CPUS CPUs" >&2
    exit 1
  fi
done
unset _world

# ---- fresh output root -----------------------------------------------------
refuse_protected_out
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null || true)" ] && [ "$ALLOW_EXISTING_OUT" != 1 ]; then
  echo "ERROR: OUT exists and is not empty: $OUT" >&2
  echo "       use a fresh OUT or set ALLOW_EXISTING_OUT=1 to reuse it" >&2
  exit 1
fi
mkdir -p "$OUT" "$LOG_DIR" "$OUT/check" "$OUT/source/current" "$OUT/source/historical"
mkdir -p "$OUT/source/thesis/scripts/static_scaling" "$OUT/source/thesis/scripts/remote/leonardo"

export WORK THESIS FLUX_CURRENT FLUX_HISTORICAL FIX BUNDLE REF ACCEPTED_PHASE4_ROOT
export PY_REF PY_DDP ACCEPTED_SHARD_REF FLUX_SPLIT PY_SPLIT
export OUT WORLD_SIZES TOTAL_CPUS RUN_TIMEOUT JOB_ID
export SCRIPTS RUNNER CHECKER JULIA_BIN JULIA_DEPOT_PATH JULIA_NUM_THREADS VENV_PY
export LOG_DIR COMMANDS_TRANSCRIPT STAGE_STATUS DRIVER_LOG DRIVER_PATH
export JULIA_PKG_PRECOMPILE_AUTO=0

supports_flag() {
  local file=$1
  local flag=$2
  [ -f "$file" ] || return 1
  if grep -q -- "$flag" "$file" 2>/dev/null; then
    return 0
  fi
  local common
  common="$(dirname "$file")/phase4_common.jl"
  if [ "$(basename "$file")" = "phase4_flux.jl" ] && [ -f "$common" ]; then
    grep -q -- "$flag" "$common" 2>/dev/null
    return $?
  fi
  return 1
}

snapshot_state_sha256() {
  local dir=$1
  (
    cd "$dir"
    find . -type f -not -path '*/__pycache__/*' -print0 \
      | sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1
  )
}

git_state_sha256() {
  local repo=$1
  if git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    ( cd "$repo" && git status --porcelain=v1 | sha256sum | cut -d' ' -f1 )
  else
    echo unknown
  fi
}

sha256_file() {
  local path=$1
  if [ -f "$path" ]; then
    sha256sum "$path" | cut -d' ' -f1
  else
    echo missing
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
  local path commit dirty

  local -a required=(
    "$FIX/data_fixture.npz"
    "$FIX/fixture_manifest.json"
    "$FIX/schedule_fixture.npz"
    "$BUNDLE"
    "$REF"
    "$PY_REF"
    "$FLUX_SPLIT"
    "$PY_SPLIT"
    "$RUNNER"
    "$CHECKER"
    "$SCRIPTS/julia-env/Project.toml"
    "$SCRIPTS/julia-env/Manifest.toml"
  )
  for path in "${required[@]}"; do
    if [ ! -f "$path" ]; then
      echo "MISSING file: $path"
      missing=1
    fi
  done

  for path in "$PY_DDP" "$ACCEPTED_SHARD_REF"; do
    if [ ! -d "$path" ]; then
      echo "MISSING accepted PyTorch evidence directory: $path"
      missing=1
    fi
  done
  for _world in $WORLD_SIZES; do
    for path in "$PY_DDP/w$_world" "$ACCEPTED_SHARD_REF/flux/cuda/w$_world" "$ACCEPTED_SHARD_REF/pytorch/cuda/w$_world"; do
      if [ ! -d "$path" ]; then
        echo "MISSING accepted PyTorch world evidence: $path"
        missing=1
      fi
    done
  done
  unset _world

  for path in "$JULIA_BIN" "$VENV_PY"; do
    if [ ! -x "$path" ]; then
      echo "MISSING executable: $path"
      missing=1
    fi
  done
  if ! command -v srun >/dev/null 2>&1; then
    echo "MISSING executable: srun"
    missing=1
  fi

  for path in "$FLUX_CURRENT" "$FLUX_HISTORICAL"; do
    if [ ! -d "$path" ]; then
      echo "MISSING Flux worktree: $path"
      missing=1
    elif [ ! -f "$path/test/Project.toml" ]; then
      echo "MISSING $path/test/Project.toml"
      missing=1
    fi
    if [ ! -f "$path/Manifest.toml" ]; then
      echo "MISSING $path/Manifest.toml"
      missing=1
    fi
    if [ ! -f "$path/test/Manifest.toml" ]; then
      echo "NOTE: $path/test/Manifest.toml is absent; the test project resolves from the depot (same as current campaign)"
    fi
  done

  if ! supports_flag "$RUNNER" --distributed-api; then
    echo "MISSING flag in $RUNNER: --distributed-api (interface dependency)"
    missing=1
  fi
  if ! grep -q -- '--flux-collective-protocol' "$CHECKER" 2>/dev/null; then
    echo "MISSING flag in $CHECKER: --flux-collective-protocol (interface dependency)"
    missing=1
  fi

  CURRENT_COMMIT=$EXPECTED_CURRENT_COMMIT
  HISTORICAL_COMMIT=$EXPECTED_HISTORICAL_COMMIT

  if git -C "$FLUX_CURRENT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    commit=$(git -C "$FLUX_CURRENT" rev-parse HEAD)
    if [ "$commit" != "$EXPECTED_CURRENT_COMMIT" ] && [ "$ALLOW_UNEXPECTED_FLUX" != 1 ]; then
      echo "UNEXPECTED current Flux commit: $commit (expected $EXPECTED_CURRENT_COMMIT)"
      missing=1
    fi
    CURRENT_COMMIT=$commit
    dirty=$(git -C "$FLUX_CURRENT" status --porcelain)
    if [ -n "$dirty" ] && [ "$ALLOW_DIRTY_FLUX" != 1 ]; then
      echo "DIRTY current Flux worktree: $FLUX_CURRENT"
      printf '%s\n' "$dirty"
      missing=1
    fi
  else
    echo "NOTE: $FLUX_CURRENT is not a git worktree (remote sync excludes .git);"
    echo "      recording expected current commit $EXPECTED_CURRENT_COMMIT"
  fi

  if git -C "$FLUX_HISTORICAL" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    commit=$(git -C "$FLUX_HISTORICAL" rev-parse HEAD)
    if [ "$commit" != "$EXPECTED_HISTORICAL_COMMIT" ] && [ "$ALLOW_UNEXPECTED_FLUX" != 1 ]; then
      echo "UNEXPECTED historical Flux commit: $commit (expected $EXPECTED_HISTORICAL_COMMIT)"
      missing=1
    fi
    HISTORICAL_COMMIT=$commit
    dirty=$(git -C "$FLUX_HISTORICAL" status --porcelain)
    if [ -n "$dirty" ] && [ "$ALLOW_DIRTY_FLUX" != 1 ]; then
      echo "DIRTY historical Flux worktree: $FLUX_HISTORICAL"
      printf '%s\n' "$dirty"
      missing=1
    fi
  else
    echo "NOTE: $FLUX_HISTORICAL is not a git worktree (remote sync excludes .git);"
    echo "      recording expected historical commit $EXPECTED_HISTORICAL_COMMIT"
  fi

  return "$missing"
}

stage_sources() {
  run_cmd a "source-snapshot" "$LOG_DIR/a-source-snapshot.log" bash -c '
set -euo pipefail
cp -a "$SCRIPTS/." "$OUT/source/thesis/scripts/static_scaling/"
cp -a "$DRIVER_PATH" "$OUT/source/thesis/scripts/remote/leonardo/$(basename "$DRIVER_PATH")"
mkdir -p "$OUT/source/current/scripts/static_scaling" "$OUT/source/historical/scripts/static_scaling"
cp -a "$SCRIPTS/." "$OUT/source/current/scripts/static_scaling/"
cp -a "$SCRIPTS/." "$OUT/source/historical/scripts/static_scaling/"
cp -a "$FLUX_CURRENT/src" "$FLUX_CURRENT/ext" "$OUT/source/current/"
cp -a "$FLUX_CURRENT/Project.toml" "$OUT/source/current/Project.toml"
cp -a "$FLUX_CURRENT/Manifest.toml" "$OUT/source/current/Manifest.toml"
cp -a "$FLUX_HISTORICAL/src" "$FLUX_HISTORICAL/ext" "$OUT/source/historical/"
cp -a "$FLUX_HISTORICAL/Project.toml" "$OUT/source/historical/Project.toml"
cp -a "$FLUX_HISTORICAL/Manifest.toml" "$OUT/source/historical/Manifest.toml"
if [ -f "$FLUX_CURRENT/test/Manifest.toml" ]; then
  mkdir -p "$OUT/source/current/test"
  cp -a "$FLUX_CURRENT/test/Manifest.toml" "$OUT/source/current/test/Manifest.toml"
fi
if [ -f "$FLUX_HISTORICAL/test/Manifest.toml" ]; then
  mkdir -p "$OUT/source/historical/test"
  cp -a "$FLUX_HISTORICAL/test/Manifest.toml" "$OUT/source/historical/test/Manifest.toml"
fi
find "$OUT/source" -type f -not -path "*/__pycache__/*" -print0 \
  | sort -z | xargs -0 -r sha256sum > "$OUT/source-sha256.txt"
' _ || return $?

  THESIS_COMMIT=$(git -C "$THESIS" rev-parse HEAD 2>/dev/null || echo "${SOURCE_COMMIT:-unknown}")
  THESIS_SCRIPTS_STATE_SHA256=$(snapshot_state_sha256 "$OUT/source/thesis/scripts/static_scaling")
  CURRENT_SNAPSHOT_SHA256=$(snapshot_state_sha256 "$OUT/source/current")
  HISTORICAL_SNAPSHOT_SHA256=$(snapshot_state_sha256 "$OUT/source/historical")
  [ -n "$THESIS_SCRIPTS_STATE_SHA256" ] || THESIS_SCRIPTS_STATE_SHA256=unknown
  [ -n "$CURRENT_SNAPSHOT_SHA256" ] || CURRENT_SNAPSHOT_SHA256=unknown
  [ -n "$HISTORICAL_SNAPSHOT_SHA256" ] || HISTORICAL_SNAPSHOT_SHA256=unknown
  if git -C "$FLUX_CURRENT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    CURRENT_GIT_STATE_SHA256=$(git_state_sha256 "$FLUX_CURRENT")
  else
    CURRENT_GIT_STATE_SHA256=$CURRENT_SNAPSHOT_SHA256
  fi
  if git -C "$FLUX_HISTORICAL" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    HISTORICAL_GIT_STATE_SHA256=$(git_state_sha256 "$FLUX_HISTORICAL")
  else
    HISTORICAL_GIT_STATE_SHA256=$HISTORICAL_SNAPSHOT_SHA256
  fi

  {
    echo "THESIS_COMMIT=$THESIS_COMMIT"
    echo "THESIS_SCRIPTS_STATE_SHA256=$THESIS_SCRIPTS_STATE_SHA256"
    echo "CURRENT_COMMIT=$CURRENT_COMMIT"
    echo "CURRENT_GIT_STATE_SHA256=$CURRENT_GIT_STATE_SHA256"
    echo "CURRENT_SNAPSHOT_SHA256=$CURRENT_SNAPSHOT_SHA256"
    echo "HISTORICAL_COMMIT=$HISTORICAL_COMMIT"
    echo "HISTORICAL_GIT_STATE_SHA256=$HISTORICAL_GIT_STATE_SHA256"
    echo "HISTORICAL_SNAPSHOT_SHA256=$HISTORICAL_SNAPSHOT_SHA256"
  } > "$OUT/source-identity.txt"

  {
    date -u +%Y-%m-%dT%H:%M:%SZ
    hostname
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
    echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
    echo "SLURM_NTASKS=${SLURM_NTASKS:-<unset>}"
    echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}"
    echo "SLURM_PROCID=${SLURM_PROCID:-<unset>}"
    echo "THESIS=$THESIS"
    echo "FLUX_CURRENT=$FLUX_CURRENT"
    echo "FLUX_HISTORICAL=$FLUX_HISTORICAL"
    echo "FIX=$FIX"
    echo "BUNDLE=$BUNDLE"
    echo "REF=$REF"
    echo "OUT=$OUT"
    echo "WORLD_SIZES=$WORLD_SIZES"
    echo "TOTAL_CPUS=$TOTAL_CPUS"
    "$JULIA_BIN" --version 2>&1 || true
    srun --version 2>&1 || true
    nvidia-smi -L 2>&1 || true
    nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 || true
  } > "$LOG_DIR/a-metadata.log" 2>&1

  JULIA_VERSION=$("$JULIA_BIN" --version 2>&1 | head -n 1 || true)
  SRUN_VERSION=$(srun --version 2>&1 | head -n 1 || true)
  GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)
  [ -n "$JULIA_VERSION" ] || JULIA_VERSION=unavailable
  [ -n "$SRUN_VERSION" ] || SRUN_VERSION=unavailable
  [ -n "$GPU_DRIVER" ] || GPU_DRIVER=unavailable
  export JULIA_VERSION SRUN_VERSION GPU_DRIVER

  return 0
}

build_provenance_args() {
  PROV_ARGS=()
  if supports_flag "$RUNNER" --source-commit && [ "$THESIS_COMMIT" != unknown ]; then
    PROV_ARGS+=(--source-commit "$THESIS_COMMIT")
  fi
  if supports_flag "$RUNNER" --source-state-sha256 && [ "$THESIS_SCRIPTS_STATE_SHA256" != unknown ]; then
    PROV_ARGS+=(--source-state-sha256 "$THESIS_SCRIPTS_STATE_SHA256")
  fi
  if supports_flag "$RUNNER" --flux-commit && [ "$FLUX_COMMIT" != unknown ]; then
    PROV_ARGS+=(--flux-commit "$FLUX_COMMIT")
  fi
  if supports_flag "$RUNNER" --flux-state-sha256 && [ "$FLUX_GIT_STATE_SHA256" != unknown ]; then
    PROV_ARGS+=(--flux-state-sha256 "$FLUX_GIT_STATE_SHA256")
  fi
  return 0
}

stage_variant_ddp() {
  local variant=$1
  local flux_root=$2
  local distributed_api=$3
  local W CPUS
  local -a args
  local -a prov_args

  if [ "$variant" = current ]; then
    FLUX_COMMIT=$CURRENT_COMMIT
    FLUX_GIT_STATE_SHA256=$CURRENT_GIT_STATE_SHA256
    FLUX_SNAPSHOT_SHA256=$CURRENT_SNAPSHOT_SHA256
  else
    FLUX_COMMIT=$HISTORICAL_COMMIT
    FLUX_GIT_STATE_SHA256=$HISTORICAL_GIT_STATE_SHA256
    FLUX_SNAPSHOT_SHA256=$HISTORICAL_SNAPSHOT_SHA256
  fi

  build_provenance_args
  prov_args=("${PROV_ARGS[@]}")

  for W in $WORLD_SIZES; do
    CPUS=$((TOTAL_CPUS / W))
    args=(
      --fixture-dir "$FIX"
      --bundle "$BUNDLE"
      --mode ddp
      --device cuda
      --backend nccl
      --world-size "$W"
      --out "$OUT/$variant"
      --distributed-api "$distributed_api"
    )
    args+=("${prov_args[@]}")

    run_cmd "$variant" "ddp-w$W" "$LOG_DIR/$variant-ddp-w$W.log" \
      timeout "$RUN_TIMEOUT" \
      srun --overlap --mpi=pmi2 --kill-on-bad-exit=1 -n"$W" -c"$CPUS" --chdir="$THESIS" \
      bash -c "$FLUX_BASH" _ "$JULIA_DEPOT_PATH" "$SCRIPTS/julia-env:$flux_root/test:@stdlib" "$JULIA_NUM_THREADS" \
      "$JULIA_BIN" --startup-file=no "$RUNNER" "${args[@]}" || return $?

    run_cmd "$variant" "verify-ddp-w$W" "$LOG_DIR/$variant-verify-ddp-w$W.log" \
      bash -c '
set -euo pipefail
test -s "$1" || { echo "MISSING rank-0 summary: $1"; exit 1; }
sha256sum "$1"
find "$(dirname "$1")" -maxdepth 1 -type f -print | sort
' _ "$OUT/$variant/ddp/flux/w$W/summary.json" || return $?
  done
  return 0
}

prepare_checkroot() {
  local variant=$1
  local root=$OUT/checkroot/$variant
  local flux_ddp=$OUT/$variant/ddp/flux

  run_cmd "$variant" "prepare-checkroot" "$LOG_DIR/$variant-prepare-checkroot.log" \
    bash -c '
set -euo pipefail
root=$1
flux_ddp=$2
pytorch_ddp=$3
shard_ref=$4
test -d "$flux_ddp" || { echo "MISSING real Flux DDP directory: $flux_ddp"; exit 1; }
test -d "$pytorch_ddp" || { echo "MISSING accepted PyTorch DDP directory: $pytorch_ddp"; exit 1; }
test -d "$shard_ref" || { echo "MISSING accepted shard-reference directory: $shard_ref"; exit 1; }
rm -rf "$root"
mkdir -p "$root/ddp"
ln -s "$flux_ddp" "$root/ddp/flux"
ln -s "$pytorch_ddp" "$root/ddp/pytorch"
ln -s "$shard_ref" "$root/shard_reference"
test -d "$root/ddp/flux"
test -d "$root/ddp/pytorch"
test -d "$root/shard_reference"
ls -l "$root" "$root/ddp"
' _ "$root" "$flux_ddp" "$PY_DDP" "$ACCEPTED_SHARD_REF" || return $?
  return 0
}

stage_check() {
  local variant=$1
  local protocol=$2
  local check_log=$LOG_DIR/$variant-check.log
  local -a args

  prepare_checkroot "$variant" || return $?

  args=(
    --fp32-absolute-1e3
    --fixture-dir "$FIX"
    --bundle "$BUNDLE"
    --frameworks flux,pytorch
    --flux-reference "$REF"
    --pytorch-reference "$PY_REF"
    --ddp-root "$OUT/checkroot/$variant/ddp"
    --device cuda
    --world-sizes "${WORLD_SIZES// /,}"
    --out-dir "$OUT/check/$variant"
    --flux-collective-protocol "$protocol"
    --split-control "flux=$FLUX_SPLIT" "pytorch=$PY_SPLIT"
  )

  run_cmd "$variant" "check" "$check_log" \
    "$VENV_PY" "$CHECKER" "${args[@]}" || return $?

  run_cmd "$variant" "check-verdict" "$LOG_DIR/$variant-check-verdict.log" \
    bash -c '
set -euo pipefail
expected="PHASE4_CHECK PASS categories=202"
if ! grep -Eq "^PHASE4_CHECK PASS categories=202( |$)" "$1"; then
  actual=$(grep -E "^PHASE4_CHECK (PASS|FAIL)" "$1" | tail -n 1 || true)
  [ -n "$actual" ] || actual="<no PHASE4_CHECK line>"
  echo "EXPECTED: $expected"
  echo "ACTUAL:   $actual"
  exit 1
fi
test -s "$2" || { echo "MISSING report $2"; exit 1; }
sha256sum "$2"
' _ "$check_log" "$OUT/check/$variant/phase4_report.txt" || return $?
  return 0
}

write_package_versions() {
  "$VENV_PY" - "$FLUX_CURRENT/Manifest.toml" "$FLUX_HISTORICAL/Manifest.toml" \
    > "$LOG_DIR/package-versions.json" <<'PY'
import json
import sys

try:
    import tomllib
except ImportError:
    tomllib = None

NAMES = ("Flux", "Optimisers", "Functors", "CUDA", "NCCL", "cuDNN", "MPI")


def versions(path):
    if tomllib is None:
        return {"error": "tomllib unavailable"}
    try:
        with open(path, "rb") as handle:
            document = tomllib.load(handle)
    except Exception as exc:
        return {"error": "%s: %s" % (type(exc).__name__, exc)}
    result = {}
    for name in NAMES:
        entry = document.get("deps", {}).get(name)
        if entry is None:
            entry = document.get(name)
        if isinstance(entry, list):
            entry = entry[0] if entry else None
        if isinstance(entry, dict):
            result[name] = entry.get("version")
    return result


print(json.dumps({"current": versions(sys.argv[1]),
                  "historical": versions(sys.argv[2])},
                 indent=2, sort_keys=True))
PY
}

write_provenance_json() {
  "$VENV_PY" - "$OUT/provenance.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ["OUT"])
OUT = Path(sys.argv[1])


def sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path):
    path = Path(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(b"\0")
        file_digest = sha256(file)
        if file_digest is None:
            return None
        digest.update(file_digest.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def identity():
    path = ROOT / "source-identity.txt"
    result = {}
    if path.is_file():
        for raw in path.read_text().splitlines():
            if "=" in raw:
                key, value = raw.split("=", 1)
                result[key] = value
    return result


def package_versions():
    path = ROOT / "logs" / "package-versions.json"
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
    return {}


def commands():
    path = ROOT / "logs" / "commands-transcript.txt"
    result = []
    if path.is_file():
        for raw in path.read_text().splitlines():
            if raw.startswith("COMMAND "):
                result.append(raw[len("COMMAND "):])
    return result


ident = identity()
fixture = Path(os.environ["FIX"])
reference = Path(os.environ["REF"])
py_reference = Path(os.environ["PY_REF"])
py_ddp = Path(os.environ["PY_DDP"])
shard_reference = Path(os.environ["ACCEPTED_SHARD_REF"])
flux_split = Path(os.environ["FLUX_SPLIT"])
py_split = Path(os.environ["PY_SPLIT"])
accepted_root = os.environ["ACCEPTED_PHASE4_ROOT"]
data = {
    "schema": "static-revision-check-provenance-v1",
    "job": os.environ.get("SLURM_JOB_ID", ""),
    "node": os.uname().nodename,
    "created_utc": os.popen("date -u +%Y-%m-%dT%H:%M:%SZ").read().strip(),
    "driver": os.environ.get("DRIVER_PATH"),
    "driver_sha256": sha256(os.environ["DRIVER_PATH"]),
    "thesis": {
        "root": os.environ["THESIS"],
        "commit": ident.get("THESIS_COMMIT"),
        "scripts_state_sha256": ident.get("THESIS_SCRIPTS_STATE_SHA256"),
    },
    "variants": {
        "current": {
            "flux_root": os.environ["FLUX_CURRENT"],
            "commit": ident.get("CURRENT_COMMIT"),
            "git_state_sha256": ident.get("CURRENT_GIT_STATE_SHA256"),
            "snapshot_sha256": ident.get("CURRENT_SNAPSHOT_SHA256"),
            "distributed_api": "guarded",
        },
        "historical": {
            "flux_root": os.environ["FLUX_HISTORICAL"],
            "commit": ident.get("HISTORICAL_COMMIT"),
            "git_state_sha256": ident.get("HISTORICAL_GIT_STATE_SHA256"),
            "snapshot_sha256": ident.get("HISTORICAL_SNAPSHOT_SHA256"),
            "distributed_api": "static",
        },
    },
    "fixture": {
        "dir": str(fixture),
        "data_fixture_sha256": sha256(fixture / "data_fixture.npz"),
        "fixture_manifest_sha256": sha256(fixture / "fixture_manifest.json"),
        "schedule_fixture_sha256": sha256(fixture / "schedule_fixture.npz"),
    },
    "bundle": {
        "path": os.environ["BUNDLE"],
        "sha256": sha256(os.environ["BUNDLE"]),
    },
    "reference": {
        "path": str(reference),
        "sha256": sha256(reference),
    },
    "gate": {
        "checker": os.environ["CHECKER"],
        "frameworks": ["flux", "pytorch"],
        "required_verdict": "PHASE4_CHECK PASS categories=202",
    },
    "checkroots": {
        "current": str(Path(os.environ["OUT"]) / "checkroot" / "current"),
        "historical": str(Path(os.environ["OUT"]) / "checkroot" / "historical"),
    },
    "reused_accepted_evidence": {
        "accepted_phase4_root": accepted_root,
        "read_only": True,
        "pytorch_rerun": False,
        "pytorch_reference": {
            "path": str(py_reference),
            "sha256": sha256(py_reference),
        },
        "pytorch_ddp": {
            "path": str(py_ddp),
            "tree_sha256": tree_sha256(py_ddp),
        },
        "shard_reference": {
            "path": str(shard_reference),
            "tree_sha256": tree_sha256(shard_reference),
        },
        "flux_split": {
            "path": str(flux_split),
            "sha256": sha256(flux_split),
        },
        "pytorch_split": {
            "path": str(py_split),
            "sha256": sha256(py_split),
        },
        "statement": (
            "Accepted Phase 4 PyTorch DDP, reference, split-control, and "
            "shard-reference evidence was reused read-only; PyTorch was not "
            "rerun in this revision check."
        ),
    },
    "world_sizes": os.environ["WORLD_SIZES"].split(),
    "total_cpus": int(os.environ["TOTAL_CPUS"]),
    "tool_versions": {
        "julia": os.environ.get("JULIA_VERSION", "unavailable"),
        "srun": os.environ.get("SRUN_VERSION", "unavailable"),
        "gpu_driver": os.environ.get("GPU_DRIVER", "unavailable"),
    },
    "packages": package_versions(),
    "commands": commands(),
}
OUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
}

write_metadata() {
  local end_utc final driver_sha
  local py_ref_sha flux_split_sha py_split_sha
  end_utc=$(utc_now)
  driver_sha=$(sha256_file "$0")
  py_ref_sha=$(sha256_file "$PY_REF")
  flux_split_sha=$(sha256_file "$FLUX_SPLIT")
  py_split_sha=$(sha256_file "$PY_SPLIT")
  [ -n "$py_ref_sha" ] || py_ref_sha=missing
  [ -n "$flux_split_sha" ] || flux_split_sha=missing
  [ -n "$py_split_sha" ] || py_split_sha=missing

  write_package_versions || true
  write_provenance_json || {
    FAILED=1
    FAIL_STAGE=${FAIL_STAGE:-meta}
    FAIL_REASON=${FAIL_REASON:-provenance.json generation failed}
  }

  final=FAIL
  if [ "$FAILED" -eq 0 ]; then final=PASS; fi

  {
    find "$OUT" -path "$LOG_DIR" -prune -o -type f \
      \( -name '*.npz' -o -name '*.json' -o -name '*.txt' \) -print0 \
      | sort -z | xargs -0 -r sha256sum
  } > "$LOG_DIR/artifact-sha256.txt" 2>/dev/null || true

  {
    echo "# Commands - Flux static-DDP revision correctness check"
    echo
    echo "Run: Leonardo compute node, driver \`$0\`."
    echo "Every command below was executed by the driver; no result is inferred."
    echo
    echo "| Item | Value |"
    echo "|---|---|"
    echo "| Host | \`$(hostname)\` |"
    echo "| SLURM job | \`${SLURM_JOB_ID:-<unset>}\` |"
    echo "| SLURM node list | \`${SLURM_JOB_NODELIST:-<unset>}\` |"
    echo "| Start (UTC) | \`$START_UTC\` |"
    echo "| End (UTC) | \`$end_utc\` |"
    echo "| Driver SHA-256 | \`$driver_sha\` |"
    echo "| OUT | \`$OUT\` |"
    echo "| World sizes | \`$WORLD_SIZES\` |"
    echo "| CPUs | \`$TOTAL_CPUS\` |"
    echo "| Gate | full \`check_phase4.py --frameworks flux,pytorch\`; required \`PHASE4_CHECK PASS categories=202\` |"
    echo "| PyTorch evidence | accepted Phase 4 evidence reused read-only; PyTorch not rerun |"
    echo "| Status | **$final** |"
    echo
    echo "## Protocols"
    echo
    echo "| Variant | Flux worktree | Distributed API |"
    echo "|---|---|---|"
    echo "| current | \`$FLUX_CURRENT\` | \`guarded\` |"
    echo "| historical | \`$FLUX_HISTORICAL\` | \`static\` |"
    echo
    echo "## Composite check roots"
    echo
    echo "Each root is fresh; its entries are symlinks. Accepted files are never copied or modified."
    echo
    echo "| Variant | Flux DDP | Accepted PyTorch DDP | Accepted shard reference |"
    echo "|---|---|---|---|"
    echo "| current | \`$OUT/checkroot/current/ddp/flux\` | \`$PY_DDP\` | \`$ACCEPTED_SHARD_REF\` |"
    echo "| historical | \`$OUT/checkroot/historical/ddp/flux\` | \`$PY_DDP\` | \`$ACCEPTED_SHARD_REF\` |"
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
    echo "## Evidence"
    echo
    echo "- Stage logs: \`$LOG_DIR/*.log\`"
    echo "- Source hashes: \`$OUT/source-sha256.txt\`"
    echo "- Source identity: \`$OUT/source-identity.txt\`"
    echo "- Provenance: \`$OUT/provenance.json\`"
    echo "- Current checker reports: \`$OUT/check/current/phase4_report.txt\`"
    echo "- Historical checker reports: \`$OUT/check/historical/phase4_report.txt\`"
    echo "- Status: \`$LOG_DIR/status.txt\`"
  } > "$LOG_DIR/COMMANDS.md"

  {
    echo "# Provenance - Flux static-DDP revision correctness check"
    echo
    echo "All values were captured from the live run; missing optional values are printed as \`<unset>\`."
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
    echo "## Source identity"
    echo
    echo '```text'
    if [ -s "$OUT/source-identity.txt" ]; then
      cat "$OUT/source-identity.txt"
    else
      echo "NOT AVAILABLE: source-identity.txt was not produced."
    fi
    echo '```'
    echo
    echo "## Inputs"
    echo
    echo '```text'
    echo "FIX=$FIX"
    echo "BUNDLE=$BUNDLE sha256=$(sha256_file "$BUNDLE")"
    echo "REF=$REF sha256=$(sha256_file "$REF")"
    echo "data_fixture_sha256=$(sha256_file "$FIX/data_fixture.npz")"
    echo "fixture_manifest_sha256=$(sha256_file "$FIX/fixture_manifest.json")"
    echo "schedule_fixture_sha256=$(sha256_file "$FIX/schedule_fixture.npz")"
    echo '```'
    echo
    echo "## Reused accepted evidence (read-only; PyTorch was NOT rerun)"
    echo
    echo '```text'
    echo "ACCEPTED_PHASE4_ROOT=$ACCEPTED_PHASE4_ROOT"
    echo "PY_REF=$PY_REF sha256=$py_ref_sha"
    echo "PY_DDP=$PY_DDP tree_sha256=(see provenance.json)"
    echo "ACCEPTED_SHARD_REF=$ACCEPTED_SHARD_REF tree_sha256=(see provenance.json)"
    echo "FLUX_SPLIT=$FLUX_SPLIT sha256=$flux_split_sha"
    echo "PY_SPLIT=$PY_SPLIT sha256=$py_split_sha"
    echo '```'
    echo
    echo "The accepted Phase 4 PyTorch DDP, reference, split-control, and shard-reference"
    echo "artifacts were reused read-only through symlinks. This driver did not run PyTorch"
    echo "and did not regenerate or modify those artifacts."
    echo
    echo "## Tool and package versions"
    echo
    echo '```text'
    echo "JULIA_VERSION=$JULIA_VERSION"
    echo "SRUN_VERSION=$SRUN_VERSION"
    echo "GPU_DRIVER=$GPU_DRIVER"
    if [ -s "$LOG_DIR/package-versions.json" ]; then
      cat "$LOG_DIR/package-versions.json"
    else
      echo "package-versions.json unavailable"
    fi
    echo '```'
    echo
    echo "## Source and artifact SHA-256"
    echo
    if [ -s "$OUT/source-sha256.txt" ]; then
      echo '```text'
      cat "$OUT/source-sha256.txt"
      echo '```'
    else
      echo "NOT AVAILABLE: source-sha256.txt was not produced."
    fi
    echo
    if [ -s "$LOG_DIR/artifact-sha256.txt" ]; then
      echo '```text'
      cat "$LOG_DIR/artifact-sha256.txt"
      echo '```'
    fi
    echo
    echo "## Failure record"
    echo
    if [ "$FAILED" -eq 0 ]; then
      echo "No stage failed."
    else
      echo "Stage \`$FAIL_STAGE\` failed: $FAIL_REASON"
    fi
  } > "$LOG_DIR/PROVENANCE.md"

  if [ "$FAILED" -eq 0 ] \
    && [ -s "$LOG_DIR/COMMANDS.md" ] \
    && [ -s "$LOG_DIR/PROVENANCE.md" ] \
    && [ -s "$OUT/provenance.json" ]; then
    printf '%s\n' "STATIC_REVISION_CHECK PASS" > "$LOG_DIR/status.txt"
  else
    if [ "$FAILED" -eq 0 ]; then
      FAILED=1
      FAIL_STAGE=f
      FAIL_REASON="metadata files are missing or empty"
    fi
    printf '%s\n' "STATIC_REVISION_CHECK FAIL stage=$FAIL_STAGE reason=$FAIL_REASON" > "$LOG_DIR/status.txt"
  fi
}

FLUX_BASH='unset JULIA_PROJECT; export JULIA_DEPOT_PATH="$1" JULIA_LOAD_PATH="$2" JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_THREADS="$3" OMP_NUM_THREADS="$3" OPENBLAS_NUM_THREADS="$3" MKL_NUM_THREADS="$3"; shift 3; exec "$@"'

run() {
  START_UTC=$(utc_now)
  FAILED=0
  FAIL_STAGE=
  FAIL_REASON=
  PROV_ARGS=()
  FLUX_COMMIT=unknown
  FLUX_GIT_STATE_SHA256=unknown
  FLUX_SNAPSHOT_SHA256=unknown
  : > "$COMMANDS_TRANSCRIPT"
  : > "$STAGE_STATUS"

  echo "=== run-static-revision-check: START $START_UTC ==="
  echo "host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>} SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
  echo "THESIS=$THESIS"
  echo "FLUX_CURRENT=$FLUX_CURRENT"
  echo "FLUX_HISTORICAL=$FLUX_HISTORICAL"
  echo "FIX=$FIX"
  echo "BUNDLE=$BUNDLE"
  echo "REF=$REF"
  echo "OUT=$OUT"
  echo "WORLD_SIZES=$WORLD_SIZES TOTAL_CPUS=$TOTAL_CPUS"

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
    if stage_sources; then set_pass a "source snapshot and metadata"; else set_fail a "source snapshot failed"; fi
  else
    set_skip a "preflight failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if stage_variant_ddp current "$FLUX_CURRENT" guarded; then
      set_pass b "current guarded DDP worlds: $WORLD_SIZES"
    else
      set_fail b "current guarded DDP failed"
    fi
  else
    set_skip b "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if stage_variant_ddp historical "$FLUX_HISTORICAL" static; then
      set_pass c "historical static DDP worlds: $WORLD_SIZES"
    else
      set_fail c "historical static DDP failed"
    fi
  else
    set_skip c "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if stage_check current guarded; then
      set_pass d "current PHASE4_CHECK PASS"
    else
      set_fail d "current checker failed"
    fi
  else
    set_skip d "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if stage_check historical static; then
      set_pass e "historical PHASE4_CHECK PASS"
    else
      set_fail e "historical checker failed"
    fi
  else
    set_skip e "earlier stage failed"
  fi

  precomp_hits=$(grep -Fl 'Precompiling packages...' "$LOG_DIR"/*.log 2>/dev/null | grep -v '/driver.log$' || true)
  if [ -n "$precomp_hits" ]; then
    echo "ERROR: 'Precompiling packages...' found in stage logs:"
    echo "$precomp_hits"
    set_fail f "precompilation marker found in stage logs"
  else
    set_pass f "no precompilation marker in stage logs"
  fi

  write_metadata
  if [ "$FAILED" -eq 0 ] && ! grep -Fq "STATIC_REVISION_CHECK PASS" "$LOG_DIR/status.txt" 2>/dev/null; then
    FAILED=1
    FAIL_STAGE=${FAIL_STAGE:-meta}
    FAIL_REASON=${FAIL_REASON:-status.txt does not report PASS}
  fi

  echo "=== summary ==="
  if [ -f "$STAGE_STATUS" ]; then
    while IFS=$'\t' read -r stage status note; do
      [ -n "$stage" ] || continue
      echo "STAGE $stage $status $note"
    done < "$STAGE_STATUS"
  fi
  if [ "$FAILED" -eq 0 ]; then
    echo "STATIC_REVISION_CHECK PASS"
    echo "OUT=$OUT"
    return 0
  fi
  echo "STATIC_REVISION_CHECK FAIL stage=$FAIL_STAGE reason=$FAIL_REASON"
  echo "OUT=$OUT"
  return 1
}

set +e
run 2>&1 | tee -a "$DRIVER_LOG"
status=${PIPESTATUS[0]}
set -e
echo "=== run-static-revision-check: EXIT $status (log: $DRIVER_LOG) ==="
exit "$status"

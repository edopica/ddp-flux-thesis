#!/usr/bin/env bash
# =============================================================================
# Leonardo COMPUTE-NODE timing driver for the current-vs-historical Flux
# static-DDP revision comparison.
#
# ALLOCATION (from the thesis root on a login node):
#   bash scripts/remote/run.sh leonardo \
#     "srun -n1 -c32 --chdir=\$HOME/projects/ddp-flux-thesis \
#      bash scripts/remote/leonardo/run-static-revision-timing.sh"
#
# The driver precompiles both Flux variants ON THE COMPUTE NODE, then runs five
# repeats of the fixture timing workload. Order is balanced across revisions
# and worlds. PyTorch is not run.
#
# Variants:
#   current     $HOME/projects/flux-integration        W>1 --distributed-api guarded
#   historical  $HOME/projects/flux-static-baseline    W>1 --distributed-api static
#   W=1         both variants use the default plain single-GPU baseline
#
# Evidence is written only under a fresh $WORK output root. Accepted Phase 3/4
# baselines and campaign 61314790 are refused.
#
# Interface dependencies (implemented by the checker agents):
#   timing_flux.jl           --distributed-api guarded|static
#   check_timing.py          --frameworks flux --flux-guard STRING --driver-marker STRING
#   check_revision_timing.py --root OUT, with current/historical subdirectories.
#   Each record must carry source.source_sha256; timing_flux.jl emits source
#   identity but not the hash map, so this driver merges it after each run.
# =============================================================================
set -euo pipefail

WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
FLUX_CURRENT=${FLUX_CURRENT:-$HOME/projects/flux-integration}
FLUX_HISTORICAL=${FLUX_HISTORICAL:-$HOME/projects/flux-static-baseline}
FIX=${FIX:-$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2}
BUNDLE=${BUNDLE:-$FIX/initial_bundle.npz}
REF=${REF:-$WORK/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/flux/cuda/phase4_reference.npz}
PHASE4_SOURCE_MANIFEST=${PHASE4_SOURCE_MANIFEST:-$WORK/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/source-sha256.txt}
JOB_ID=${SLURM_JOB_ID:-manual}
ROOT=${ROOT:-$WORK/epicazio/static-scaling-evidence/2026-10-19-revision-timing-$JOB_ID}
OUT=${OUT:-$ROOT}
ROOT=$(readlink -m "$ROOT")
OUT=$(readlink -m "$OUT")
REPEATS=${REPEATS:-5}
UPDATES=${UPDATES:-300}
WARMUP=${WARMUP:-20}
TOTAL_CPUS=${TOTAL_CPUS:-${SLURM_CPUS_PER_TASK:-32}}
RUN_TIMEOUT=${RUN_TIMEOUT:-1200}
ALLOW_LOGIN_NODE=${ALLOW_LOGIN_NODE:-0}
ALLOW_EXISTING_OUT=${ALLOW_EXISTING_OUT:-0}
ALLOW_UNEXPECTED_FLUX=${ALLOW_UNEXPECTED_FLUX:-0}
ALLOW_DIRTY_FLUX=${ALLOW_DIRTY_FLUX:-0}
SOURCE_COMMIT=${SOURCE_COMMIT:-}
EXPECTED_CURRENT_COMMIT=${EXPECTED_CURRENT_COMMIT:-f7b4520c4bd307c67bbc5ac92a1c68493eccff6e}
EXPECTED_HISTORICAL_COMMIT=${EXPECTED_HISTORICAL_COMMIT:-8add0e42b7d9388ff365335152126e5b1e9de657}

CURRENT_MARKER="STATIC_TIMING_REVISION-CURRENT PASS"
HISTORICAL_MARKER="STATIC_TIMING_REVISION-HISTORICAL PASS"
GUARD_CURRENT="presence + nonfinite loss/gradient rejection inside timed window"
GUARD_HISTORICAL="no presence or nonfinite distributed validation"
GUARD_SINGLE="not applicable: single-GPU baseline"

SCRIPTS=$THESIS/scripts/static_scaling
RUNNER=$SCRIPTS/timing_flux.jl
CHECK_TIMING=$SCRIPTS/check_timing.py
CHECK_REVISION=$SCRIPTS/check_revision_timing.py
PRECOMPILE=$THESIS/scripts/remote/leonardo/precompile-phase3.sh
JULIA_BIN=${JULIA_BIN:-$WORK/epicazio/julia-1.12.6/bin/julia}
JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$WORK/epicazio/julia-depot}
JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "$WORK/epicazio/venvs/pytorch-scaling/bin/python" ]; then
    PYTHON_BIN=$WORK/epicazio/venvs/pytorch-scaling/bin/python
  else
    PYTHON_BIN=python3
  fi
fi
DRIVER_PATH=$(readlink -m "$0")
LOG_DIR=$OUT/logs
COMMANDS_TRANSCRIPT=$LOG_DIR/commands-transcript.txt
STAGE_STATUS=$LOG_DIR/stage-status.txt
DRIVER_LOG=$OUT/driver.log

THESIS_COMMIT=unknown
THESIS_SCRIPTS_STATE_SHA256=unknown
CURRENT_COMMIT=unknown
CURRENT_GIT_STATE_SHA256=unknown
CURRENT_SNAPSHOT_SHA256=unknown
HISTORICAL_COMMIT=unknown
HISTORICAL_GIT_STATE_SHA256=unknown
HISTORICAL_SNAPSHOT_SHA256=unknown
FIXTURE_SHA256=unknown
BUNDLE_SHA256=unknown
REF_SHA256=unknown
GPU_DRIVER=unavailable
SRUN_VERSION=unavailable
JULIA_VERSION=unavailable
FAILED=0
FAIL_STAGE=
FAIL_REASON=

is_positive_int() {
  case "${1:-}" in
    ''|*[!0-9]*) return 1 ;;
    *) [ "$1" -gt 0 ] ;;
  esac
}

utc_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
quote_cmd() { printf '%q ' "$@"; }
die() { echo "ERROR: $*" >&2; exit 1; }

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
is_positive_int "$REPEATS" || { echo "ERROR: REPEATS must be a positive integer" >&2; exit 1; }
is_positive_int "$UPDATES" || { echo "ERROR: UPDATES must be a positive integer" >&2; exit 1; }
is_positive_int "$WARMUP" || { echo "ERROR: WARMUP must be a positive integer" >&2; exit 1; }
for _world in 1 2 4; do
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
  echo "       use a fresh ROOT/OUT or set ALLOW_EXISTING_OUT=1 to reuse it" >&2
  exit 1
fi
mkdir -p "$OUT" "$LOG_DIR" "$OUT/current" "$OUT/historical"

export WORK THESIS FLUX_CURRENT FLUX_HISTORICAL FIX BUNDLE REF PHASE4_SOURCE_MANIFEST
export JOB_ID ROOT OUT REPEATS UPDATES WARMUP TOTAL_CPUS RUN_TIMEOUT
export SCRIPTS RUNNER CHECK_TIMING CHECK_REVISION PRECOMPILE
export JULIA_BIN JULIA_DEPOT_PATH JULIA_NUM_THREADS PYTHON_BIN DRIVER_PATH
export LOG_DIR COMMANDS_TRANSCRIPT STAGE_STATUS DRIVER_LOG
export CURRENT_MARKER HISTORICAL_MARKER GUARD_CURRENT GUARD_HISTORICAL GUARD_SINGLE
export JULIA_PKG_PRECOMPILE_AUTO=0

supports_flag() {
  local file=$1
  local flag=$2
  [ -f "$file" ] || return 1
  grep -q -- "$flag" "$file" 2>/dev/null
}

phase4_drift_check() {
  if [ ! -f "$PHASE4_SOURCE_MANIFEST" ]; then
    echo "PHASE4_DRIFT_ERROR missing accepted Phase 4 source manifest: $PHASE4_SOURCE_MANIFEST"
    return 1
  fi
  "$PYTHON_BIN" - "$PHASE4_SOURCE_MANIFEST" "$SCRIPTS" <<'PY'
import hashlib
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
scripts = Path(sys.argv[2])
shared = ("phase4_common.jl", "phase4_torch.py", "precision.jl",
          "model_flux.jl", "model_torch.py", "dataset_adapter.py",
          "replay.jl")
phase4 = {}
for raw in manifest.read_text(errors="replace").splitlines():
    parts = raw.split(None, 1)
    if len(parts) == 2 and len(parts[0]) == 64:
        phase4[Path(parts[1]).name] = parts[0]
present = [name for name in shared
           if name in phase4 and (scripts / name).is_file()]
if not present:
    print("PHASE4_DRIFT_ERROR accepted Phase 4 source manifest has no "
          "shared-helper hashes usable for drift")
    raise SystemExit(1)
drift = [name for name in present
         if hashlib.sha256((scripts / name).read_bytes()).hexdigest()
         != phase4[name]]
if not drift:
    print("PHASE4_DRIFT_ERROR accepted Phase 4 source manifest matches every "
          "shared helper; check_timing.py requires non-empty drift")
    raise SystemExit(1)
print("PHASE4_DRIFT_OK " + ",".join(drift))
PY
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

preflight() {
  local missing=0
  local path commit dirty

  local -a required=(
    "$FIX/data_fixture.npz"
    "$FIX/fixture_manifest.json"
    "$FIX/schedule_fixture.npz"
    "$BUNDLE"
    "$RUNNER"
    "$CHECK_TIMING"
    "$CHECK_REVISION"
    "$PRECOMPILE"
    "$PHASE4_SOURCE_MANIFEST"
    "$SCRIPTS/julia-env/Project.toml"
    "$SCRIPTS/julia-env/Manifest.toml"
  )
  for path in "${required[@]}"; do
    if [ ! -f "$path" ]; then
      echo "MISSING file: $path"
      missing=1
    fi
  done
  if [ ! -f "$CHECK_REVISION" ]; then
    echo "DEPENDENCY: scripts/static_scaling/check_revision_timing.py is not present yet"
  fi

  for path in "$JULIA_BIN" "$PYTHON_BIN"; do
    if ! command -v "$path" >/dev/null 2>&1 && [ ! -x "$path" ]; then
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

  if ! phase4_drift_check; then
    echo "ERROR: the accepted Phase 4 source manifest cannot satisfy the"
    echo "       check_timing.py non-empty shared_helper_drift requirement."
    missing=1
  fi

  if ! supports_flag "$RUNNER" --distributed-api; then
    echo "MISSING flag in $RUNNER: --distributed-api (interface dependency)"
    missing=1
  fi
  if ! supports_flag "$CHECK_TIMING" --frameworks; then
    echo "MISSING flag in $CHECK_TIMING: --frameworks (interface dependency)"
    missing=1
  fi
  if ! supports_flag "$CHECK_TIMING" --flux-guard; then
    echo "MISSING flag in $CHECK_TIMING: --flux-guard (interface dependency)"
    missing=1
  fi
  if ! supports_flag "$CHECK_TIMING" --driver-marker; then
    echo "MISSING flag in $CHECK_TIMING: --driver-marker (interface dependency)"
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

snapshot_variant() {
  local variant=$1
  local flux_root=$2
  local vdir=$OUT/$variant
  mkdir -p "$vdir/source" "$vdir/test" || return 1
  cp -a "$SCRIPTS/." "$vdir/source/" || return 1
  rm -rf "$vdir/source/__pycache__" || return 1
  cp -a "$flux_root/src" "$flux_root/ext" "$vdir/source/" || return 1
  cp -a "$flux_root/Project.toml" "$vdir/source/Project.toml" || return 1
  cp -a "$flux_root/Manifest.toml" "$vdir/source/Manifest.toml" || return 1
  if [ -f "$flux_root/test/Manifest.toml" ]; then
    cp -a "$flux_root/test/Manifest.toml" "$vdir/test/Manifest.toml" || return 1
  else
    printf '%s\n' "NO_TEST_MANIFEST: test/Project.toml resolves from the depot" \
      > "$vdir/test/MANIFEST-STATUS.txt" || return 1
  fi
  cp -a "$THESIS/scripts/static_scaling/julia-env/Manifest.toml" "$vdir/julia-manifest.toml" || return 1
  cp -a "$flux_root/Project.toml" "$vdir/flux-project.toml" || return 1
  cp -a "$flux_root/Manifest.toml" "$vdir/flux-manifest.toml" || return 1
  find "$vdir/source" -type f -not -path '*/__pycache__/*' -print0 \
    | sort -z | xargs -0 -r sha256sum > "$vdir/source-sha256.txt" || return 1
}

write_source_hashes_json() {
  local vdir=$1
  "$PYTHON_BIN" - "$vdir/source" "$vdir/source-hashes.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
hashes = {}
for path in sorted(source.iterdir()):
    if path.is_file() and path.suffix in (".jl", ".py"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[path.name] = digest
Path(sys.argv[2]).write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
PY
}

augment_record() {
  local out=$1
  local hashes=$2
  "$PYTHON_BIN" - "$out" "$hashes" <<'PY'
import json
import sys
from pathlib import Path

record_path = Path(sys.argv[1])
hashes_path = Path(sys.argv[2])
record = json.loads(record_path.read_text())
hashes = json.loads(hashes_path.read_text())
source = record.setdefault("source", {})
source["source_sha256"] = hashes
record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
}

stage_prepare() {
  : > "$OUT/current/driver.log"
  : > "$OUT/historical/driver.log"
  snapshot_variant current "$FLUX_CURRENT" || return 1
  snapshot_variant historical "$FLUX_HISTORICAL" || return 1
  write_source_hashes_json "$OUT/current" || return 1
  write_source_hashes_json "$OUT/historical" || return 1

  THESIS_COMMIT=$(git -C "$THESIS" rev-parse HEAD 2>/dev/null || echo "${SOURCE_COMMIT:-unknown}")
  THESIS_SCRIPTS_STATE_SHA256=$(snapshot_state_sha256 "$THESIS/scripts/static_scaling")
  CURRENT_SNAPSHOT_SHA256=$(snapshot_state_sha256 "$OUT/current/source")
  HISTORICAL_SNAPSHOT_SHA256=$(snapshot_state_sha256 "$OUT/historical/source")
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
  FIXTURE_SHA256=$(sha256_file "$FIX/data_fixture.npz")
  BUNDLE_SHA256=$(sha256_file "$BUNDLE")
  REF_SHA256=$(sha256_file "$REF")
  [ -n "$THESIS_SCRIPTS_STATE_SHA256" ] || THESIS_SCRIPTS_STATE_SHA256=unknown
  [ -n "$CURRENT_SNAPSHOT_SHA256" ] || CURRENT_SNAPSHOT_SHA256=unknown
  [ -n "$HISTORICAL_SNAPSHOT_SHA256" ] || HISTORICAL_SNAPSHOT_SHA256=unknown
  [ -n "$FIXTURE_SHA256" ] || FIXTURE_SHA256=missing
  [ -n "$BUNDLE_SHA256" ] || BUNDLE_SHA256=missing
  [ -n "$REF_SHA256" ] || REF_SHA256=missing

  {
    echo "THESIS_COMMIT=$THESIS_COMMIT"
    echo "THESIS_SCRIPTS_STATE_SHA256=$THESIS_SCRIPTS_STATE_SHA256"
    echo "CURRENT_COMMIT=$CURRENT_COMMIT"
    echo "CURRENT_GIT_STATE_SHA256=$CURRENT_GIT_STATE_SHA256"
    echo "CURRENT_SNAPSHOT_SHA256=$CURRENT_SNAPSHOT_SHA256"
    echo "HISTORICAL_COMMIT=$HISTORICAL_COMMIT"
    echo "HISTORICAL_GIT_STATE_SHA256=$HISTORICAL_GIT_STATE_SHA256"
    echo "HISTORICAL_SNAPSHOT_SHA256=$HISTORICAL_SNAPSHOT_SHA256"
    echo "FIXTURE_SHA256=$FIXTURE_SHA256"
    echo "BUNDLE_SHA256=$BUNDLE_SHA256"
    echo "REF_SHA256=$REF_SHA256"
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
    echo "REPEATS=$REPEATS UPDATES=$UPDATES WARMUP=$WARMUP"
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
}

log_run() {
  local label=$1
  local logfile=$2
  local vdriver=$3
  shift 3
  local display start_utc end_utc rc
  local -a tee_args=(-a "$logfile")
  if [ -n "$vdriver" ]; then
    tee_args+=(-a "$vdriver")
  fi

  display=$(quote_cmd "$@")
  start_utc=$(utc_now)

  : > "$logfile"
  {
    echo "=== $label START $start_utc ==="
    echo "COMMAND: $display"
    echo "LOG: $logfile"
  } | tee "${tee_args[@]}"

  set +e
  "$@" 2>&1 | tee "${tee_args[@]}"
  rc=${PIPESTATUS[0]}
  set -e

  end_utc=$(utc_now)
  echo "=== $label END $end_utc EXIT $rc ===" | tee "${tee_args[@]}"
  {
    echo "STAGE $label"
    echo "START_UTC $start_utc"
    echo "END_UTC $end_utc"
    echo "EXIT $rc"
    echo "COMMAND $display"
    echo
  } >> "$COMMANDS_TRANSCRIPT"

  if [ -n "$vdriver" ] && grep -Fq 'Precompiling packages...' "$logfile"; then
    echo "ERROR: 'Precompiling packages...' found in $logfile" | tee "${tee_args[@]}"
    return 97
  fi
  return "$rc"
}

precompile_variant() {
  local variant=$1
  local flux_root=$2
  local vdir=$OUT/$variant
  local plog=$LOG_DIR/precompile-$variant.log
  local vdriver=$vdir/driver.log
  local start_utc end_utc rc

  start_utc=$(utc_now)
  : > "$plog"
  {
    echo "=== PRECOMPILE $variant START $start_utc ==="
    echo "COMMAND: FLUX_DIR=$flux_root bash $PRECOMPILE"
  } | tee -a "$plog" "$vdriver"

  set +e
  (
    unset JULIA_PROJECT JULIA_LOAD_PATH
    export FLUX_DIR="$flux_root"
    export JULIA_DEPOT_PATH="$JULIA_DEPOT_PATH"
    export JULIA_NUM_THREADS="$JULIA_NUM_THREADS"
    export JULIA_PKG_PRECOMPILE_AUTO=1
    bash "$PRECOMPILE"
  ) 2>&1 | tee -a "$plog" "$vdriver"
  rc=${PIPESTATUS[0]}
  set -e

  end_utc=$(utc_now)
  echo "=== PRECOMPILE $variant END $end_utc EXIT $rc ===" | tee -a "$plog" "$vdriver"
  {
    echo "STAGE precompile-$variant"
    echo "START_UTC $start_utc"
    echo "END_UTC $end_utc"
    echo "EXIT $rc"
    echo "COMMAND FLUX_DIR=$flux_root bash $PRECOMPILE"
    echo
  } >> "$COMMANDS_TRANSCRIPT"

  if [ "$rc" -ne 0 ]; then
    return "$rc"
  fi
  if ! grep -Fq 'FLUX_PRECOMPILE_OK' "$plog"; then
    echo "ERROR: FLUX_PRECOMPILE_OK missing from $plog" | tee -a "$plog" "$vdriver"
    return 1
  fi
  return 0
}

runtime_env_variant() {
  local variant=$1
  local flux_root=$2
  local vdir=$OUT/$variant
  local log=$vdir/runtime-env.txt
  printf '%s\n' "COMMAND: capture runtime-env -> $log" >> "$vdir/driver.log"
  {
    date -u +%Y-%m-%dT%H:%M:%SZ
    hostname
    echo "variant=$variant"
    echo "FLUX_DIR=$flux_root"
    echo "JULIA_BIN=$JULIA_BIN"
    echo "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"
    echo "JULIA_LOAD_PATH=$SCRIPTS/julia-env:$flux_root/test:@stdlib"
    "$JULIA_BIN" --version
    (
      unset JULIA_PROJECT
      export JULIA_DEPOT_PATH="$JULIA_DEPOT_PATH"
      export JULIA_LOAD_PATH="$SCRIPTS/julia-env:$flux_root/test:@stdlib"
      export JULIA_PKG_PRECOMPILE_AUTO=0
      export JULIA_NUM_THREADS=1
      "$JULIA_BIN" --startup-file=no --project="$flux_root/test" \
        -e 'using Pkg; Pkg.status(); foreach(println, Base.load_path())'
    )
  } > "$log" 2>&1 || echo "RUNTIME_ENV_CAPTURE_FAILED" >> "$log"
  if grep -Fq 'Precompiling packages...' "$log"; then
    echo "ERROR: 'Precompiling packages...' in $log" | tee -a "$OUT/$variant/driver.log"
    return 1
  fi
  return 0
}

write_package_versions() {
  "$PYTHON_BIN" - "$FLUX_CURRENT/Manifest.toml" "$FLUX_HISTORICAL/Manifest.toml" \
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

write_variant_provenance() {
  local variant=$1
  local flux_root=$2
  local distributed_api=$3
  local guard_policy=$4
  local vdir=$OUT/$variant

  VARIANT="$variant" FLUX_ROOT="$flux_root" DISTRIBUTED_API="$distributed_api" \
    GUARD_POLICY="$guard_policy" "$PYTHON_BIN" - "$vdir/provenance.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

VDIR = Path(os.environ["OUT"]) / os.environ["VARIANT"]
OUT = Path(sys.argv[1])
ROOT = Path(os.environ["OUT"])


def sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def command_lines():
    path = VDIR / "driver.log"
    result = []
    if path.is_file():
        for raw in path.read_text(errors="replace").splitlines():
            if raw.startswith("COMMAND: "):
                result.append(raw[len("COMMAND: "):])
    return result


def parse_source_manifest(path):
    result = {}
    if not path.is_file():
        return result
    for raw in path.read_text(errors="replace").splitlines():
        parts = raw.split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            result[Path(parts[1]).name] = parts[0]
    return result


def package_versions():
    path = ROOT / "logs" / "package-versions.json"
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
    return {}


def tree_hashes(base):
    base = Path(base)
    result = {}
    if not base.is_dir():
        return result
    for path in sorted(base.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            result[str(path.relative_to(VDIR / "source"))] = sha256(path)
    return result


ident = identity()
source_dir = VDIR / "source"
computed_hashes = {}
for path in sorted(source_dir.iterdir()) if source_dir.is_dir() else []:
    if path.is_file() and path.suffix in (".jl", ".py"):
        computed_hashes[path.name] = sha256(path)
record_hash_path = VDIR / "source-hashes.json"
record_hashes = computed_hashes
if record_hash_path.is_file():
    try:
        record_hashes = json.loads(record_hash_path.read_text())
    except (OSError, ValueError):
        record_hashes = computed_hashes
timing_hashes = record_hashes
flux_hashes = {}
for sub in ("src", "ext"):
    flux_hashes.update(tree_hashes(source_dir / sub))
for name in ("Project.toml", "Manifest.toml"):
    path = source_dir / name
    if path.is_file():
        flux_hashes[name] = sha256(path)
flux_tree_sha256 = hashlib.sha256(
    json.dumps(flux_hashes, sort_keys=True).encode()).hexdigest()

phase4_manifest = Path(os.environ.get("PHASE4_SOURCE_MANIFEST", ""))
phase4_all = parse_source_manifest(phase4_manifest)
shared_helpers = ("phase4_common.jl", "phase4_torch.py", "precision.jl",
                  "model_flux.jl", "model_torch.py", "dataset_adapter.py",
                  "replay.jl")
phase4_hashes = {name: phase4_all[name] for name in shared_helpers
                 if name in phase4_all}
drift = sorted(name for name, digest in phase4_hashes.items()
               if name in timing_hashes and timing_hashes[name] != digest)
fixture = Path(os.environ["FIX"])
bundle = Path(os.environ["BUNDLE"])
reference = Path(os.environ["REF"])
variant = os.environ["VARIANT"]
distributed_api = os.environ["DISTRIBUTED_API"]
guard = os.environ["GUARD_POLICY"]

qualifications = [
    "Timing artifact only; no numerical-equivalence claim.",
    "PyTorch was not run in this revision timing campaign.",
    "World 1 is the plain single-GPU baseline and has no distributed collectives.",
    "Current and historical runs share the thesis runner and differ in the Flux worktree and distributed protocol.",
]
if not phase4_hashes:
    qualifications.append(
        "The accepted Phase 4 source manifest was unavailable; no Phase 4 helper hashes were compared.")
elif not drift:
    qualifications.append(
        "No shared-helper drift against the accepted Phase 4 source snapshot was found.")

data = {
    "schema": "static-timing-provenance-v1",
    "revision_schema": "static-timing-revision-v1",
    "variant": variant,
    "distributed_api": distributed_api,
    "guard_policy": guard,
    "job": os.environ.get("SLURM_JOB_ID", ""),
    "node": os.uname().nodename,
    "created_utc": os.popen("date -u +%Y-%m-%dT%H:%M:%SZ").read().strip(),
    "thesis": {
        "root": os.environ["THESIS"],
        "commit": ident.get("THESIS_COMMIT"),
        "scripts_state_sha256": ident.get("THESIS_SCRIPTS_STATE_SHA256"),
    },
    "flux": {
        "root": os.environ["FLUX_ROOT"],
        "commit": ident.get("CURRENT_COMMIT" if variant == "current" else "HISTORICAL_COMMIT"),
        "git_state_sha256": ident.get("CURRENT_GIT_STATE_SHA256" if variant == "current" else "HISTORICAL_GIT_STATE_SHA256"),
        "snapshot_sha256": ident.get("CURRENT_SNAPSHOT_SHA256" if variant == "current" else "HISTORICAL_SNAPSHOT_SHA256"),
        "tree_sha256": flux_tree_sha256,
        "source_sha256": flux_hashes,
    },
    "fixture": {
        "dir": str(fixture),
        "data_fixture_sha256": sha256(fixture / "data_fixture.npz"),
        "fixture_manifest_sha256": sha256(fixture / "fixture_manifest.json"),
        "schedule_fixture_sha256": sha256(fixture / "schedule_fixture.npz"),
    },
    "bundle": {"path": str(bundle), "sha256": sha256(bundle)},
    "reference": {"path": str(reference), "sha256": sha256(reference)},
    "phase4_source_manifest": str(phase4_manifest) if phase4_hashes else None,
    "timing_source_sha256": timing_hashes,
    "record_source_sha256": {
        "path": str(record_hash_path),
        "sha256": sha256(record_hash_path),
        "source_sha256": record_hashes,
    },
    "phase4_source_sha256": phase4_hashes,
    "shared_helper_drift": drift,
    "execution_policy": {
        "flux": {
            "precision": "FP32 strict FMA, TF32 off",
            "cudnn_deterministic": False,
            "algorithm_selection": "strict FMA candidate filter; no determinism filter",
            "nonfinite_guard": guard,
            "guard_policy": guard,
            "distributed_api": distributed_api,
            "ddp_backend": "NCCL",
            "single_process_w1": "default single-GPU baseline",
        },
        "pytorch": {
            "ran": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "nonfinite_guard": "outside timed loop",
            "note": "PyTorch not run in this revision timing campaign",
        },
    },
    "flags": {
        "repeats": int(os.environ["REPEATS"]),
        "updates": int(os.environ["UPDATES"]),
        "warmup": int(os.environ["WARMUP"]),
        "world_sizes": [1, 2, 4],
        "total_cpus": int(os.environ["TOTAL_CPUS"]),
        "distributed_api_w1": "default single-GPU baseline",
        "distributed_api_wgt1": distributed_api,
    },
    "commands": command_lines(),
    "qualifications": qualifications,
}
OUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
}

write_root_provenance() {
  "$PYTHON_BIN" - "$OUT/provenance.json" "$OUT/current/provenance.json" "$OUT/historical/provenance.json" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["OUT"])
data = {
    "schema": "static-timing-revision-v1",
    "job": os.environ.get("SLURM_JOB_ID", ""),
    "node": os.uname().nodename,
    "root": str(root),
    "current": json.loads(Path(sys.argv[2]).read_text()),
    "historical": json.loads(Path(sys.argv[3]).read_text()),
}
Path(sys.argv[1]).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
}

run_one() {
  local variant=$1
  local distributed_api=$2
  local repeat=$3
  local world=$4
  local flux_root variant_commit variant_state
  local vdir=$OUT/$variant
  local out=$vdir/flux-w$world-r$repeat.json
  local log=$LOG_DIR/$variant-w$world-r$repeat.log
  local cpus=$((TOTAL_CPUS / world))
  local -a args

  if [ "$variant" = current ]; then
    flux_root=$FLUX_CURRENT
    variant_commit=$CURRENT_COMMIT
    variant_state=$CURRENT_GIT_STATE_SHA256
  else
    flux_root=$FLUX_HISTORICAL
    variant_commit=$HISTORICAL_COMMIT
    variant_state=$HISTORICAL_GIT_STATE_SHA256
  fi
  args=(
    --fixture-dir "$FIX"
    --world-size "$world"
    --updates "$UPDATES"
    --warmup "$WARMUP"
    --out "$out"
    --source-commit "$THESIS_COMMIT"
    --source-state-sha256 "$THESIS_SCRIPTS_STATE_SHA256"
    --flux-commit "$variant_commit"
    --flux-state-sha256 "$variant_state"
  )
  if [ "$world" -gt 1 ]; then
    args+=(--distributed-api "$distributed_api")
  fi

  log_run "$variant-w$world-r$repeat" "$log" "$vdir/driver.log" \
    timeout "$RUN_TIMEOUT" \
    srun --overlap --mpi=pmi2 --kill-on-bad-exit=1 -n"$world" -c"$cpus" --chdir="$THESIS" \
    bash -c "$FLUX_BASH" _ "$JULIA_DEPOT_PATH" "$SCRIPTS/julia-env:$flux_root/test:@stdlib" "$JULIA_NUM_THREADS" \
    "$JULIA_BIN" --startup-file=no "$RUNNER" "${args[@]}" || return $?

  if [ ! -s "$out" ]; then
    echo "ERROR: missing JSON output: $out" | tee -a "$log" "$vdir/driver.log"
    return 1
  fi
  if ! augment_record "$out" "$vdir/source-hashes.json"; then
    echo "ERROR: failed to record source hashes in $out" | tee -a "$log" "$vdir/driver.log"
    return 1
  fi
  {
    echo "POSTPROCESS $out source.source_sha256=$vdir/source-hashes.json"
    echo
  } >> "$COMMANDS_TRANSCRIPT"
  printf '%s\n' "COMMAND: augment $out with source.source_sha256" >> "$vdir/driver.log"
  return 0
}

run_variant_worlds() {
  local variant=$1
  local distributed_api=$2
  local repeat=$3
  local worlds=$4
  local world
  for world in $worlds; do
    run_one "$variant" "$distributed_api" "$repeat" "$world" || return $?
  done
  return 0
}

check_timing_variant() {
  local variant=$1
  local guard=$2
  local marker=$3
  local vdir=$OUT/$variant
  local log=$LOG_DIR/check-timing-$variant.log
  local -a args
  args=(
    "$vdir"
    --frameworks flux
    --flux-guard "$guard"
    --driver-marker "$marker"
    --repeats "$REPEATS"
    --fixture-sha256 "$FIXTURE_SHA256"
    --bundle-sha256 "$BUNDLE_SHA256"
  )
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    args+=(--job-id "$SLURM_JOB_ID")
  fi

  log_run "check-timing-$variant" "$log" "$vdir/driver.log" \
    "$PYTHON_BIN" "$CHECK_TIMING" "${args[@]}" || return $?
  grep -Fq 'TIMING_ARTIFACT_CHECK PASS' "$log"
}

check_revision() {
  local log=$LOG_DIR/check-revision-timing.log
  local -a args
  args=(
    --root "$OUT"
    --repeats "$REPEATS"
    --fixture-sha256 "$FIXTURE_SHA256"
    --bundle-sha256 "$BUNDLE_SHA256"
    --current-flux-commit "$CURRENT_COMMIT"
    --historical-flux-commit "$HISTORICAL_COMMIT"
  )
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    args+=(--job-id "$SLURM_JOB_ID")
  fi

  log_run "check-revision-timing" "$log" "" \
    "$PYTHON_BIN" "$CHECK_REVISION" "${args[@]}" || return $?
  grep -Fq 'REVISION_TIMING PASS' "$log"
}

write_metadata() {
  local end_utc final driver_sha
  end_utc=$(utc_now)
  driver_sha=$(sha256_file "$DRIVER_PATH")

  {
    echo "# current"
    cat "$OUT/current/source-sha256.txt"
    echo "# historical"
    cat "$OUT/historical/source-sha256.txt"
  } > "$OUT/source-sha256.txt" 2>/dev/null || true

  write_package_versions || true
  write_variant_provenance current "$FLUX_CURRENT" guarded "$GUARD_CURRENT" || true
  write_variant_provenance historical "$FLUX_HISTORICAL" static "$GUARD_HISTORICAL" || true
  write_root_provenance || true

  if [ ! -s "$OUT/current/provenance.json" ] || [ ! -s "$OUT/historical/provenance.json" ]; then
    if [ "$FAILED" -eq 0 ]; then
      FAILED=1
      FAIL_STAGE=meta
      FAIL_REASON="per-variant provenance.json is missing or empty"
    fi
  fi
  if [ ! -s "$OUT/provenance.json" ] && [ "$FAILED" -eq 0 ]; then
    FAILED=1
    FAIL_STAGE=meta
    FAIL_REASON="root provenance.json is missing or empty"
  fi

  final=FAIL
  if [ "$FAILED" -eq 0 ]; then final=PASS; fi

  {
    echo "# Commands - Flux static-DDP revision timing"
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
    echo "| Repeats / updates / warmup | \`$REPEATS\` / \`$UPDATES\` / \`$WARMUP\` |"
    echo "| CPUs | \`$TOTAL_CPUS\` |"
    echo "| Status | **$final** |"
    echo
    echo "## Revision protocol and guard policy"
    echo
    echo "| Variant | Flux worktree | W>1 distributed API | Guard policy |"
    echo "|---|---|---|---|"
    echo "| current | \`$FLUX_CURRENT\` | \`guarded\` | \`$GUARD_CURRENT\` |"
    echo "| historical | \`$FLUX_HISTORICAL\` | \`static\` | \`$GUARD_HISTORICAL\` |"
    echo "| W=1 (both) | - | default | \`$GUARD_SINGLE\` |"
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
    echo "- Per-variant records: \`$OUT/current/\` and \`$OUT/historical/\`"
    echo "- Per-variant driver logs: \`$OUT/current/driver.log\`, \`$OUT/historical/driver.log\`"
    echo "- Precompile logs: \`$LOG_DIR/precompile-current.log\`, \`$LOG_DIR/precompile-historical.log\`"
    echo "- Timing checker logs: \`$LOG_DIR/check-timing-current.log\`, \`$LOG_DIR/check-timing-historical.log\`"
    echo "- Revision checker log: \`$LOG_DIR/check-revision-timing.log\`"
    echo "- Status: \`$LOG_DIR/status.txt\`"
  } > "$OUT/COMMANDS.md"

  {
    echo "# Provenance - Flux static-DDP revision timing"
    echo
    echo "All values were captured from the live run; missing optional values are printed as \`<unset>\`."
    echo "This campaign is timing-only and makes no numerical-equivalence claim."
    echo
    echo "| Item | Value |"
    echo "|---|---|"
    echo "| Host | \`$(hostname)\` |"
    echo "| SLURM job | \`${SLURM_JOB_ID:-<unset>}\` |"
    echo "| SLURM node list | \`${SLURM_JOB_NODELIST:-<unset>}\` |"
    echo "| Start (UTC) | \`$START_UTC\` |"
    echo "| End (UTC) | \`$end_utc\` |"
    echo "| Driver | \`$DRIVER_PATH\` |"
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
    echo "BUNDLE=$BUNDLE sha256=$BUNDLE_SHA256"
    echo "REF=$REF sha256=$REF_SHA256"
    echo '```'
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
    echo "## Source SHA-256"
    echo
    if [ -s "$OUT/source-sha256.txt" ]; then
      echo '```text'
      cat "$OUT/source-sha256.txt"
      echo '```'
    else
      echo "NOT AVAILABLE: source-sha256.txt was not produced."
    fi
    echo
    echo "## Failure record"
    echo
    if [ "$FAILED" -eq 0 ]; then
      echo "No stage failed."
    else
      echo "Stage \`$FAIL_STAGE\` failed: $FAIL_REASON"
    fi
  } > "$OUT/PROVENANCE.md"

  if [ "$FAILED" -eq 0 ] \
    && [ -s "$OUT/COMMANDS.md" ] \
    && [ -s "$OUT/PROVENANCE.md" ] \
    && [ -s "$OUT/provenance.json" ] \
    && [ -s "$OUT/current/provenance.json" ] \
    && [ -s "$OUT/historical/provenance.json" ]; then
    printf '%s\n' "STATIC_REVISION_TIMING PASS" > "$LOG_DIR/status.txt"
  else
    if [ "$FAILED" -eq 0 ]; then
      FAILED=1
      FAIL_STAGE=meta
      FAIL_REASON="metadata files are missing or empty"
    fi
    printf '%s\n' "STATIC_REVISION_TIMING FAIL stage=$FAIL_STAGE reason=$FAIL_REASON" > "$LOG_DIR/status.txt"
  fi
}

FLUX_BASH='unset JULIA_PROJECT; export JULIA_DEPOT_PATH="$1" JULIA_LOAD_PATH="$2" JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_THREADS="$3" OMP_NUM_THREADS="$3" OPENBLAS_NUM_THREADS="$3" MKL_NUM_THREADS="$3"; shift 3; exec "$@"'

run() {
  START_UTC=$(utc_now)
  export START_UTC
  : > "$COMMANDS_TRANSCRIPT"
  : > "$STAGE_STATUS"

  echo "=== run-static-revision-timing: START $START_UTC ==="
  echo "host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>} SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
  echo "THESIS=$THESIS"
  echo "FLUX_CURRENT=$FLUX_CURRENT"
  echo "FLUX_HISTORICAL=$FLUX_HISTORICAL"
  echo "FIX=$FIX"
  echo "BUNDLE=$BUNDLE"
  echo "OUT=$OUT"
  echo "REPEATS=$REPEATS UPDATES=$UPDATES WARMUP=$WARMUP TOTAL_CPUS=$TOTAL_CPUS"
  if [ "$UPDATES" != 300 ] || [ "$WARMUP" != 20 ]; then
    echo "WARNING: check_revision_timing.py expects updates=300 and warmup=20; this run uses $UPDATES/$WARMUP"
  fi
  echo "GUARD_CURRENT=$GUARD_CURRENT"
  echo "GUARD_HISTORICAL=$GUARD_HISTORICAL"

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
    if stage_prepare; then set_pass a "source snapshots and metadata"; else set_fail a "source snapshot failed"; fi
  else
    set_skip a "preflight failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if precompile_variant current "$FLUX_CURRENT"; then
      set_pass b "current precompile FLUX_PRECOMPILE_OK"
    else
      set_fail b "current precompile failed"
    fi
  else
    set_skip b "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if precompile_variant historical "$FLUX_HISTORICAL"; then
      set_pass c "historical precompile FLUX_PRECOMPILE_OK"
    else
      set_fail c "historical precompile failed"
    fi
  else
    set_skip c "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if runtime_env_variant current "$FLUX_CURRENT" \
      && runtime_env_variant historical "$FLUX_HISTORICAL"; then
      set_pass d "runtime environment captured"
    else
      set_fail d "runtime environment capture failed"
    fi
  else
    set_skip d "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    local repeat
    for ((repeat=1; repeat<=REPEATS; repeat++)); do
      if (( repeat % 2 )); then
        if ! run_variant_worlds current guarded "$repeat" "1 2 4"; then
          set_fail e "current timing failed at repeat $repeat"
          break
        fi
        if ! run_variant_worlds historical static "$repeat" "4 2 1"; then
          set_fail e "historical timing failed at repeat $repeat"
          break
        fi
      else
        if ! run_variant_worlds historical static "$repeat" "1 2 4"; then
          set_fail e "historical timing failed at repeat $repeat"
          break
        fi
        if ! run_variant_worlds current guarded "$repeat" "4 2 1"; then
          set_fail e "current timing failed at repeat $repeat"
          break
        fi
      fi
    done
    if [ "$FAILED" -eq 0 ]; then
      set_pass e "timing repeats 1..$REPEATS"
    fi
  else
    set_skip e "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    printf '%s\n' "$CURRENT_MARKER" | tee -a "$OUT/current/driver.log"
    printf '%s\n' "$HISTORICAL_MARKER" | tee -a "$OUT/historical/driver.log"
    printf '%s\n' "TIMING_DRIVER_EXIT=0" | tee -a "$OUT/current/driver.log" "$OUT/historical/driver.log"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if check_timing_variant current "$GUARD_CURRENT" "$CURRENT_MARKER"; then
      set_pass f "current TIMING_ARTIFACT_CHECK PASS"
    else
      printf '%s\n' "STATIC_TIMING_REVISION-CURRENT CHECK_FAIL" | tee -a "$OUT/current/driver.log"
      set_fail f "current timing checker failed"
    fi
  else
    set_skip f "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if check_timing_variant historical "$GUARD_HISTORICAL" "$HISTORICAL_MARKER"; then
      set_pass g "historical TIMING_ARTIFACT_CHECK PASS"
    else
      printf '%s\n' "STATIC_TIMING_REVISION-HISTORICAL CHECK_FAIL" | tee -a "$OUT/historical/driver.log"
      set_fail g "historical timing checker failed"
    fi
  else
    set_skip g "earlier stage failed"
  fi

  if [ "$FAILED" -eq 0 ]; then
    if check_revision; then
      set_pass h "revision timing checker PASS"
    else
      set_fail h "revision timing checker failed"
    fi
  else
    set_skip h "earlier stage failed"
  fi

  precomp_hits=$(grep -Fl 'Precompiling packages...' "$LOG_DIR"/*-w*-r*.log 2>/dev/null || true)
  if [ -n "$precomp_hits" ]; then
    echo "ERROR: 'Precompiling packages...' found in timed logs:"
    echo "$precomp_hits"
    set_fail i "precompilation marker found in timed logs"
  elif [ "$FAILED" -eq 0 ]; then
    set_pass i "no precompilation marker in timed logs"
  else
    set_skip i "earlier stage failed"
  fi

  write_metadata
  if [ "$FAILED" -eq 0 ] && ! grep -Fq "STATIC_REVISION_TIMING PASS" "$LOG_DIR/status.txt" 2>/dev/null; then
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
    echo "STATIC_REVISION_TIMING PASS"
    echo "OUT=$OUT"
    return 0
  fi
  echo "STATIC_REVISION_TIMING FAIL stage=$FAIL_STAGE reason=$FAIL_REASON"
  echo "OUT=$OUT"
  return 1
}

trap 'rc=$?; echo "TIMING_DRIVER_EXIT=$rc"' EXIT
: > "$DRIVER_LOG"
exec > >(tee -a "$DRIVER_LOG") 2>&1

set +e
run
status=$?
set -e
exit "$status"

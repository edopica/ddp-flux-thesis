#!/usr/bin/env bash
# =============================================================================
# Leonardo (CINECA) COMPUTE-NODE driver for the Phase 3 Step 2 precision
# experiment: current cuDNN math vs explicit CUDNN_FMA_MATH.
#
# Three fresh Julia processes, one per case (descriptor/algorithm caches are
# process-wide and keyed by math type, so a process is never reused):
#
#   cuda_current : --device cuda --precision current
#   cuda_fma     : --device cuda --precision fma
#   cpu_current  : --device cpu  --precision current   (numerical control)
#
# The unchanged PyTorch stage-2 diagnostic (stage2_diag.npz) is the reference.
# The PyTorch reference generator is NOT rerun, and no reference file is
# modified. Tolerances remain atol=1e-5, rtol=1e-4 inside the diagnostic.
#
# INPUTS
#   FIX       Phase 2 fixture directory (data_fixture.npz, initial_bundle.npz)
#   REF       preserved phase3_reference.npz (unchanged PyTorch reference)
#   DIAG      preserved stage2_diag.npz (unchanged PyTorch stage-2 arrays)
#
# Defaults prefer the preserved copies inside the synced thesis tree. The
# synchronization intentionally excludes the large preserved directories, so
# the fallback is the immutable evidence root created during preservation:
#   $WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-attempt3/
# Override REF or DIAG explicitly if that layout changes.
#
# OUTPUT
#   OUT/logs/cuda_current.log, cuda_fma.log, cpu_current.log
#
# PASS RULE (printed in the output):
#   * cuda_fma: conv1, conv2, proj tolerance violations == 0, and both
#     conv1 and conv2 max_abs <= 1e-4 (CPU-control level; the CPU run
#     reports conv1 max_abs around 7.6e-6);
#   * cuda_fma requested and effective conv1/conv2/proj math type is
#     CUDNN_FMA_MATH;
#   * cuda_current: conv1 OR conv2 violations > 0 and the same array's
#     max_abs > 1e-4 (the reproduced failure is retained; cuDNN's
#     heuristic algorithm choice may place the TF32 algorithm on either
#     convolution in a given process, as observed on 2026-10-17);
#   * cuda_current requested/effective math type is CUDNN_DEFAULT_MATH;
#   * cpu_current control: conv1, conv2, proj violations == 0.
#   Tolerances and the reference are not modified.
#
# ALLOCATION (from the thesis root on a login node):
#   GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
#     "srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/run-phase3-precision-experiment.sh"
#
# The script never invokes srun/salloc itself and never precompiles.
# =============================================================================
set -euo pipefail

WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
FIX=${FIX:-$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2}
EVIDENCE=${EVIDENCE:-$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-attempt3}
OUT=${OUT:-$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-step2-precision}
LOG_DIR=${LOG_DIR:-$OUT/logs}
RUNNER=${RUNNER:-$THESIS/scripts/remote/leonardo/run-julia-phase3.sh}
DIAGNOSE=$THESIS/scripts/static_scaling/diagnose_phase3_flux_stage2.jl

PRESERVED_DIAG=$THESIS/artifacts/logs/static-scaling/2026-10-17-phase3-attempt3-preserved/diag/stage2_diag.npz
PRESERVED_REF=$THESIS/artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/phase3_reference.npz
if [ ! -f "$PRESERVED_DIAG" ] && [ -f "$EVIDENCE/diag/stage2_diag.npz" ]; then
  PRESERVED_DIAG=$EVIDENCE/diag/stage2_diag.npz
fi
if [ ! -f "$PRESERVED_REF" ] && [ -f "$EVIDENCE/reference/phase3_reference.npz" ]; then
  PRESERVED_REF=$EVIDENCE/reference/phase3_reference.npz
fi
DIAG=${DIAG:-$PRESERVED_DIAG}
REF=${REF:-$PRESERVED_REF}

die() { echo "ERROR: $*" >&2; exit 1; }

host=$(hostname)
case "$host" in
  login*)
    die "running on login node '$host': run under srun on a compute node"
    ;;
esac

[ -x "$RUNNER" ] || die "runner not found/executable: $RUNNER"
[ -f "$DIAGNOSE" ] || die "diagnostic not found: $DIAGNOSE (sync first)"
[ -f "$FIX/data_fixture.npz" ] || die "fixture not found: $FIX/data_fixture.npz"
[ -f "$FIX/initial_bundle.npz" ] || die "bundle not found: $FIX/initial_bundle.npz"
[ -f "$REF" ] || die "preserved reference not found: $REF (set REF=...)"
[ -f "$DIAG" ] || die "preserved stage-2 diagnostic not found: $DIAG (set DIAG=...)"

mkdir -p "$LOG_DIR"

echo "=== run-phase3-precision-experiment: START ==="
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>} SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
echo "THESIS=$THESIS"
echo "FIX=$FIX"
echo "REF=$REF"
echo "DIAG=$DIAG"
echo "OUT=$OUT"
echo "LOG_DIR=$LOG_DIR"
echo "RUNNER=$RUNNER"
echo "DIAGNOSE=$DIAGNOSE"

run_case() {
  local case_name=$1
  shift
  local log=$LOG_DIR/${case_name}.log
  echo "=== PRECISION_CASE ${case_name} START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  set +e
  bash "$RUNNER" "$DIAGNOSE" \
    --fixture-dir "$FIX" \
    --bundle "$FIX/initial_bundle.npz" \
    --reference "$REF" \
    --diag "$DIAG" \
    "$@" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  echo "PRECISION_CASE ${case_name} COMPLETE exit=${status} log=${log}"
  return "$status"
}

status_current=0
status_fma=0
status_cpu=0
run_case cuda_current --device cuda --precision current || status_current=$?
run_case cuda_fma --device cuda --precision fma || status_fma=$?
run_case cpu_current --device cpu --precision current || status_cpu=$?

CURRENT_LOG=$LOG_DIR/cuda_current.log
FMA_LOG=$LOG_DIR/cuda_fma.log
CPU_LOG=$LOG_DIR/cpu_current.log

# FLUX_STAGE2 <name> max_abs=... max_rel=... violations=...
stage2_field() {
  local file=$1 name=$2 metric=$3
  awk -v n="$name" -v m="$metric" '
    $1 == "FLUX_STAGE2" && $2 == n {
      for (i = 3; i <= NF; i++) {
        split($i, a, "=")
        if (a[1] == m) print a[2]
      }
    }' "$file" | tail -n 1
}

# FLUX_STAGE2_CONFIG key=value ...
config_field() {
  local file=$1 key=$2
  awk -v k="$key" '
    $1 == "FLUX_STAGE2_CONFIG" {
      for (i = 2; i <= NF; i++) {
        split($i, a, "=")
        if (a[1] == k) print a[2]
      }
    }' "$file" | tail -n 1
}

num_le() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a <= b) }'; }
num_gt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'; }

ok=1
check_file() {
  local file=$1 label=$2
  if [ ! -s "$file" ]; then
    echo "PRECISION_CHECK ${label} FAIL missing_or_empty_log=${file}"
    ok=0
    return
  fi
  echo "PRECISION_CHECK ${label} log_present=${file}"
}

check_zero() {
  local label=$1 value=$2
  if [ -z "$value" ]; then
    echo "PRECISION_CHECK ${label} FAIL missing_value"
    ok=0
  elif [ "$value" = "0" ]; then
    echo "PRECISION_CHECK ${label} PASS violations=${value}"
  else
    echo "PRECISION_CHECK ${label} FAIL violations=${value}"
    ok=0
  fi
}

check_file "$CURRENT_LOG" cuda_current_log
check_file "$FMA_LOG" cuda_fma_log
check_file "$CPU_LOG" cpu_current_log

echo "PRECISION_RULE cuda_fma: conv1/conv2/proj violations==0 AND conv1 max_abs<=1e-4 AND conv2 max_abs<=1e-4"
echo "PRECISION_RULE cuda_current: (conv1 OR conv2) violations>0 AND same array max_abs>1e-4 (failure retained)"
echo "PRECISION_RULE cpu_current: conv1/conv2/proj violations==0 (control)"

for case_name in cuda_current cuda_fma cpu_current; do
  case "$case_name" in
    cuda_current) log=$CURRENT_LOG ;;
    cuda_fma) log=$FMA_LOG ;;
    cpu_current) log=$CPU_LOG ;;
  esac
  [ -s "$log" ] || continue
  printf 'PRECISION_CONFIG case=%s precision=%s cublas_math_mode=%s cudnn_requested=%s cudnn_effective_conv1=%s cudnn_effective_conv2=%s cudnn_effective_proj=%s cudnn_version=%s CUDNN_FMA_MATH=%s\n' \
    "$case_name" \
    "$(config_field "$log" precision)" \
    "$(config_field "$log" cublas_math_mode)" \
    "$(config_field "$log" cudnn_requested)" \
    "$(config_field "$log" cudnn_effective_conv1)" \
    "$(config_field "$log" cudnn_effective_conv2)" \
    "$(config_field "$log" cudnn_effective_proj)" \
    "$(config_field "$log" cudnn_version)" \
    "$(config_field "$log" CUDNN_FMA_MATH)"
  for name in conv1 conv2 proj; do
    printf 'PRECISION_RESULT case=%s array=%s max_abs=%s max_rel=%s violations=%s\n' \
      "$case_name" "$name" \
      "$(stage2_field "$log" "$name" max_abs)" \
      "$(stage2_field "$log" "$name" max_rel)" \
      "$(stage2_field "$log" "$name" violations)"
  done
done

for case_status in "cuda_current=$status_current" "cuda_fma=$status_fma" "cpu_current=$status_cpu"; do
  name=${case_status%%=*}
  code=${case_status##*=}
  if [ "$code" = "0" ]; then
    echo "PRECISION_CHECK case=${name} exit PASS"
  else
    echo "PRECISION_CHECK case=${name} exit FAIL exit=${code}"
    ok=0
  fi
done

if [ "$status_cpu" = "0" ] && [ -s "$CPU_LOG" ]; then
  for name in conv1 conv2 proj; do
    check_zero "cpu_current.${name}" "$(stage2_field "$CPU_LOG" "$name" violations)"
  done
fi

if [ "$status_fma" = "0" ] && [ -s "$FMA_LOG" ]; then
  for name in conv1 conv2 proj; do
    check_zero "cuda_fma.${name}" "$(stage2_field "$FMA_LOG" "$name" violations)"
  done
  for name in conv1 conv2; do
    fma_abs=$(stage2_field "$FMA_LOG" "$name" max_abs)
    if [ -z "$fma_abs" ]; then
      echo "PRECISION_CHECK cuda_fma.${name}_max_abs FAIL missing_value"
      ok=0
    elif num_le "$fma_abs" "1e-4"; then
      echo "PRECISION_CHECK cuda_fma.${name}_max_abs PASS max_abs=${fma_abs} <= 1e-4"
    else
      echo "PRECISION_CHECK cuda_fma.${name}_max_abs FAIL max_abs=${fma_abs} > 1e-4"
      ok=0
    fi
  done
  for key in cudnn_requested cudnn_effective_conv1 cudnn_effective_conv2 cudnn_effective_proj; do
    value=$(config_field "$FMA_LOG" "$key")
    case "$value" in
      CUDNN_FMA_MATH*)
        echo "PRECISION_CHECK cuda_fma.${key} PASS value=${value}"
        ;;
      *)
        echo "PRECISION_CHECK cuda_fma.${key} FAIL value=${value:-<missing>}"
        ok=0
        ;;
    esac
  done
fi

if [ "$status_current" = "0" ] && [ -s "$CURRENT_LOG" ]; then
  failure_array=
  for name in conv1 conv2; do
    current_violations=$(stage2_field "$CURRENT_LOG" "$name" violations)
    current_abs=$(stage2_field "$CURRENT_LOG" "$name" max_abs)
    if [ -n "$current_violations" ] && [ -n "$current_abs" ] &&
       num_gt "$current_violations" "0" && num_gt "$current_abs" "1e-4"; then
      failure_array=$name
      failure_violations=$current_violations
      failure_abs=$current_abs
      break
    fi
  done
  if [ -n "$failure_array" ]; then
    echo "PRECISION_CHECK cuda_current.failure_retained PASS array=${failure_array} violations=${failure_violations} max_abs=${failure_abs}"
  else
    echo "PRECISION_CHECK cuda_current.failure_retained FAIL no conv1/conv2 array with violations>0 and max_abs>1e-4"
    ok=0
  fi
  for key in cudnn_requested cudnn_effective_conv1 cudnn_effective_conv2 cudnn_effective_proj; do
    value=$(config_field "$CURRENT_LOG" "$key")
    case "$value" in
      CUDNN_DEFAULT_MATH*)
        echo "PRECISION_CHECK cuda_current.${key} PASS value=${value}"
        ;;
      *)
        echo "PRECISION_CHECK cuda_current.${key} FAIL value=${value:-<missing>}"
        ok=0
        ;;
    esac
  done
fi

echo "=== summary ==="
echo "cuda_current_exit=$status_current"
echo "cuda_fma_exit=$status_fma"
echo "cpu_current_exit=$status_cpu"
if [ "$ok" = "1" ]; then
  echo "PHASE3_PRECISION_EXPERIMENT PASS"
  exit 0
fi
echo "PHASE3_PRECISION_EXPERIMENT FAIL"
exit 1

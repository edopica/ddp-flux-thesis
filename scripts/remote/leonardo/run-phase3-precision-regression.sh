#!/usr/bin/env bash
# =============================================================================
# Leonardo (CINECA) COMPUTE-NODE driver for the focused precision regression.
#
# Three fresh Julia processes (descriptor and algorithm caches are
# process-wide):
#
#   fma     : check_precision_math.jl --mode fma      must PASS (exit 0)
#   current : check_precision_math.jl --mode current  must FAIL with a
#             recorded convolution discrepancy (negative control)
#   policy  : test_precision_policy.jl                must PASS (exit 0)
#
# The FMA case checks the accepted policy: supported CUDA.jl/cuDNN.jl
# package versions, cuBLAS PEDANTIC_MATH, cuDNN CUDNN_FMA_MATH for the
# stage-2 conv1/conv2/proj descriptors, FP32-level convolution
# forward/data-gradient/weight-gradient, FP32-level classifier matmul
# against Float64 CPU references, and an end-to-end autotuned
# Flux.CrossCor NNlib-path convolution whose recorded algorithm choices
# must all be CUDNN_FMA_MATH with fallback=false.
#
# The current case uses the stock cuDNN.jl mapping
# (PEDANTIC_MATH -> CUDNN_DEFAULT_MATH) and is expected to fail on at least
# one convolution item. It is not part of the accepted pass path.
#
# The policy case is the GPU-free ten-case strict-FMA/version regression;
# it runs in its own process, must print CHECK_PRECISION_POLICY PASS, and
# must report ten passing case lines with no failing case line.
#
# OUTPUT
#   $LOG_DIR/fma.log, $LOG_DIR/current.log, $LOG_DIR/policy.log
#
# ALLOCATION (from the thesis root on a login node):
#   GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
#     "srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/run-phase3-precision-regression.sh"
#
# The script never invokes srun/salloc itself and never precompiles.
# =============================================================================
set -euo pipefail

WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
OUT=${OUT:-$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-step2-precision}
LOG_DIR=${LOG_DIR:-$OUT/logs}
RUNNER=${RUNNER:-$THESIS/scripts/remote/leonardo/run-julia-phase3.sh}
CHECK=$THESIS/scripts/static_scaling/check_precision_math.jl
POLICY=$THESIS/scripts/static_scaling/test_precision_policy.jl

die() { echo "ERROR: $*" >&2; exit 1; }

host=$(hostname)
case "$host" in
  login*)
    die "running on login node '$host': run under srun on a compute node"
    ;;
esac

[ -x "$RUNNER" ] || die "runner not found/executable: $RUNNER"
[ -f "$CHECK" ] || die "regression not found: $CHECK (sync first)"
[ -f "$POLICY" ] || die "policy regression not found: $POLICY (sync first)"
mkdir -p "$LOG_DIR"

echo "=== run-phase3-precision-regression: START ==="
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>} SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
echo "THESIS=$THESIS"
echo "CHECK=$CHECK"
echo "POLICY=$POLICY"
echo "OUT=$OUT"
echo "LOG_DIR=$LOG_DIR"
echo "RUNNER=$RUNNER"

run_case() {
  local case_name=$1
  local script=$2
  shift 2
  local log=$LOG_DIR/${case_name}.log
  echo "=== PRECISION_REGRESSION_CASE ${case_name} START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  set +e
  bash "$RUNNER" "$script" "$@" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  echo "PRECISION_REGRESSION_CASE ${case_name} COMPLETE exit=${status} log=${log}"
  return "$status"
}

status_fma=0
status_current=0
status_policy=0
run_case fma "$CHECK" --mode fma || status_fma=$?
run_case current "$CHECK" --mode current || status_current=$?
run_case policy "$POLICY" || status_policy=$?

FMA_LOG=$LOG_DIR/fma.log
CURRENT_LOG=$LOG_DIR/current.log
POLICY_LOG=$LOG_DIR/policy.log

ok=1

echo "=== summary ==="
algo_choices_line=$(grep -E '^CHECK_PRECISION_MATH_ALGO_CHOICES .* (PASS|FAIL)$' "$FMA_LOG" | tail -n 1 || true)
if [ "$status_fma" = "0" ] && [ -s "$FMA_LOG" ] &&
   grep -q '^CHECK_PRECISION_MATH PASS$' "$FMA_LOG" &&
   echo "$algo_choices_line" | grep -q ' PASS$'; then
  echo "PRECISION_REGRESSION_CHECK fma PASS exit=0 algo_choices=\"${algo_choices_line}\""
else
  echo "PRECISION_REGRESSION_CHECK fma FAIL exit=${status_fma} (expected exit 0, CHECK_PRECISION_MATH PASS, and CHECK_PRECISION_MATH_ALGO_CHOICES PASS; got algo_choices=\"${algo_choices_line}\")"
  ok=0
fi

# The negative control must be the deterministic pinned-algorithm path
# (`conv_fixed_*`), not the autotuned NNlib diagnostic (`conv_nnlib_fwd`).
conv_fail_line=$(grep -E '^CHECK_PRECISION_MATH_ITEM conv_fixed[^ ]*_(fwd|dx|dw) .* FAIL$' "$CURRENT_LOG" | head -n 1 || true)
if [ "$status_current" != "0" ] && [ -s "$CURRENT_LOG" ] &&
   grep -q '^CHECK_PRECISION_MATH FAIL$' "$CURRENT_LOG" &&
   [ -n "$conv_fail_line" ]; then
  echo "PRECISION_REGRESSION_CHECK current PASS exit=${status_current} conv_fail=\"${conv_fail_line}\""
else
  echo "PRECISION_REGRESSION_CHECK current FAIL exit=${status_current} (expected nonzero exit, CHECK_PRECISION_MATH FAIL, and a conv discrepancy)"
  ok=0
fi

policy_pass_count=$(grep -c -E '^CHECK_PRECISION_POLICY case=.* PASS$' "$POLICY_LOG" 2>/dev/null || true)
policy_fail_count=$(grep -c -E '^CHECK_PRECISION_POLICY case=.* FAIL$' "$POLICY_LOG" 2>/dev/null || true)
policy_pass_count=${policy_pass_count:-0}
policy_fail_count=${policy_fail_count:-0}
if [ "$status_policy" = "0" ] && [ -s "$POLICY_LOG" ] &&
   grep -q '^CHECK_PRECISION_POLICY PASS$' "$POLICY_LOG" &&
   [ "$policy_pass_count" -ge 10 ] && [ "$policy_fail_count" -eq 0 ]; then
  echo "PRECISION_REGRESSION_CHECK policy PASS exit=0 cases=${policy_pass_count} fail=${policy_fail_count}"
else
  echo "PRECISION_REGRESSION_CHECK policy FAIL exit=${status_policy} (expected exit 0, CHECK_PRECISION_POLICY PASS, >=10 passing case lines, and zero FAIL case lines; got cases=${policy_pass_count} fail=${policy_fail_count})"
  ok=0
fi

if [ "$ok" = "1" ]; then
  echo "PHASE3_PRECISION_REGRESSION PASS"
  exit 0
fi
echo "PHASE3_PRECISION_REGRESSION FAIL"
exit 1

#!/usr/bin/env bash
set -uo pipefail
WORK=/leonardo_work/IscrC_UNMASKED
THESIS=$HOME/projects/ddp-flux-thesis
EVID=$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-attempt3
OUT=$EVID/provenance/environment_probe_math_mode.txt
exec > >(tee "$OUT") 2>&1
echo "# Compute-node cuDNN math-mode mapping probe"
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
bash "$THESIS/scripts/remote/leonardo/run-julia-phase3.sh" -e '
using CUDA, cuDNN
println("cudnnGetVersion_raw=", cuDNN.cudnnGetVersion())
println("cuDNN.version()=", cuDNN.version())
println("cuDNN.math_mode(PEDANTIC_MATH)=", cuDNN.math_mode(CUDA.PEDANTIC_MATH))
println("cuDNN.math_mode(DEFAULT_MATH)=", cuDNN.math_mode(CUDA.DEFAULT_MATH))
println("cuDNN.math_mode(FAST_MATH)=", cuDNN.math_mode(CUDA.FAST_MATH))
CUDA.math_mode!(CUDA.PEDANTIC_MATH)
println("after_CUDA.math_mode!(PEDANTIC_MATH): CUDA.math_mode()=", CUDA.math_mode())
println("after_CUDA.math_mode!(PEDANTIC_MATH): cuDNN.math_mode()=", cuDNN.math_mode())
println("after_CUDA.math_mode!(PEDANTIC_MATH): CUDA.math_precision()=", CUDA.math_precision())
'
echo "--- probe exit=$? ---"

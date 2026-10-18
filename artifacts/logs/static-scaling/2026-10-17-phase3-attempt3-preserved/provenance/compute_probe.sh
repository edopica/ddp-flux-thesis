#!/usr/bin/env bash
set -uo pipefail
WORK=/leonardo_work/IscrC_UNMASKED
THESIS=$HOME/projects/ddp-flux-thesis
EVID=$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-attempt3
OUT=$EVID/provenance/environment_probe.txt
exec > >(tee "$OUT") 2>&1
echo "# Compute-node environment probe (Phase 3 attempt-3 evidence preservation)"
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv,noheader
echo "--- python/torch runtime ---"
"$WORK/epicazio/venvs/pytorch-scaling/bin/python" - <<'PY'
import sys
import numpy
import torch
import torchvision
print("python=" + sys.version.split()[0])
print("numpy=" + numpy.__version__)
print("torch=" + torch.__version__)
print("torchvision=" + torchvision.__version__)
print("torch.version.cuda=" + str(torch.version.cuda))
print("torch.version.cudnn=" + str(getattr(torch.version, "cudnn", None)))
print("torch.backends.cudnn.version=" + str(torch.backends.cudnn.version()))
print("torch.cuda.nccl.version=" + str(torch.cuda.nccl.version()))
print("torch.cuda.is_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("device0=" + torch.cuda.get_device_name(0))
    print("capability=" + str(torch.cuda.get_device_capability(0)))
print("cudnn.allow_tf32=" + str(torch.backends.cudnn.allow_tf32))
print("cuda.matmul.allow_tf32=" + str(torch.backends.cuda.matmul.allow_tf32))
print("float32_matmul_precision=" + torch.get_float32_matmul_precision())
PY
echo "--- julia runtime ---"
bash "$THESIS/scripts/remote/leonardo/run-julia-phase3.sh" -e '
using CUDA, cuDNN
println("CUDA.jl_version=", pkgversion(CUDA))
println("cuDNN.jl_version=", pkgversion(cuDNN))
println("CUDA.runtime_version=", CUDA.runtime_version())
println("CUDA.driver_version=", CUDA.driver_version())
println("cuDNN.library_version=", cuDNN.version())
println("CUDA.math_mode=", CUDA.math_mode())
println("cuDNN.math_mode=", cuDNN.math_mode())
'
echo "--- probe exit=$? ---"

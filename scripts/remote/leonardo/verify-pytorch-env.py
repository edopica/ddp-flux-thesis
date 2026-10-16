#!/usr/bin/env python3
"""Minimal environment verification for the Leonardo PyTorch environment.

Prints a block of key=value lines and exits non-zero if a check fails.
Scope: environment verification only. ENV_SMOKE is a single 4-element
multiply/add kernel; there is no autograd, training, dataset, or benchmark.

Run on a compute node with a GPU allocated, e.g.:
  srun -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_dbg \
    -N1 -n1 --cpus-per-task=4 --gpus-per-task=1 --time=00:05:00 \
    bash -lc '<venv>/bin/python <venv>/verify-pytorch-env.py'
"""

import platform
import sys


def emit(key, value):
    print(f"{key}={value}")


try:
    import torch
    import torchvision
except Exception as exc:
    print(f"import_error={type(exc).__name__}: {exc}")
    print("ENV_VERIFY=FAIL")
    sys.exit(1)

errors = []

emit("python_version", platform.python_version())
emit("python_executable", sys.executable)
emit("torch_version", torch.__version__)
emit("torch_cuda_version", torch.version.cuda)
emit("torchvision_version", torchvision.__version__)
emit("cuda_is_available", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("ENV_VERIFY=FAIL")
    print("ENV_VERIFY_ERROR=cuda_is_available=False")
    sys.exit(1)

try:
    cuda_driver_version = torch._C._cuda_getDriverVersion()
except Exception as exc:
    cuda_driver_version = f"unavailable ({type(exc).__name__}: {exc})"
emit("cuda_driver_version", cuda_driver_version)

try:
    import subprocess

    nvidia_smi_driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[0]
except Exception as exc:
    nvidia_smi_driver = f"unavailable ({type(exc).__name__}: {exc})"
emit("nvidia_smi_driver_version", nvidia_smi_driver)

emit("cuda_device_count", torch.cuda.device_count())
emit("cuda_device_name_0", torch.cuda.get_device_name(0))

capability = torch.cuda.get_device_capability(0)
emit("cuda_device_capability_0", capability)
if capability != (8, 0):
    errors.append(f"cuda_device_capability_0={capability}, expected (8, 0)")

emit("cuda_device_total_memory_0", torch.cuda.get_device_properties(0).total_memory)
emit("cudnn_version", torch.backends.cudnn.version())
emit("nccl_version", torch.cuda.nccl.version())

# ENV_SMOKE: one tiny multiply/add kernel, 4 elements: 4 * (1 * 1.5 + 0.5) = 8.
x = torch.ones(4, device="cuda")
total = (x * 1.5 + 0.5).sum().item()
torch.cuda.synchronize()
smoke_pass = total == 8.0
emit("ENV_SMOKE", f"sum={total} expected=8.0 {'PASS' if smoke_pass else 'FAIL'}")
if not smoke_pass:
    errors.append(f"ENV_SMOKE sum={total}, expected 8.0")

if errors:
    print("ENV_VERIFY=FAIL")
    for error in errors:
        print(f"ENV_VERIFY_ERROR={error}")
    sys.exit(1)

print("ENV_VERIFY=PASS")

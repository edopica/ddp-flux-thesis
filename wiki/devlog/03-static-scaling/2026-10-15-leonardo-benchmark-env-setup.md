# Leonardo benchmark environment setup: PyTorch venv and Tiny ImageNet-200

Date: 2026-10-15
Host: `leonardo` (CINECA); setup and dataset download on a login node, GPU
environment verification on one A100-SXM-64GB compute node (driver 535.274.02)
Scope: static-network GPU scaling (scope 3), environment setup only
Draft experiment: [static-network GPU scaling proposal](../../../reports/static-graph-gpu-scaling-proposal-detailed.md) (discussion draft; no experiment has run)
Context: [Leonardo cluster](../../context/leonardo.md)

## Goal

Prepare, but do not run, the benchmark environment for the static-network GPU
scaling proposal: the Python/PyTorch runtime for the PyTorch side and the Tiny
ImageNet-200 dataset. No test, benchmark, or training ran, and no model or
training code was written.

## Outcome

The PyTorch virtual environment and the Tiny ImageNet-200 dataset are staged on
`$WORK` (`/leonardo_work/IscrC_UNMASKED`). The compute-node environment
verification passed (`ENV_VERIFY=PASS`, exit 0) and the dataset passed every
layout and count check
(`TINY_IMAGENET_OK train=100000 val=10000 test=10000 classes=200`). No Julia,
no precompile, and no commit occurred. The environment is verified only at the
import and single-kernel level.

## PyTorch environment

Created with `uv` on the login node. All paths live on `$WORK`, never in the
50 GB `$HOME`:

| Item | Value |
|---|---|
| uv | 0.13.1, `$WORK/epicazio/tools/uv/uv` (astral.sh installer) |
| Python | CPython 3.11.17, uv-managed under `$WORK/epicazio/uv-python` |
| uv cache | `$WORK/epicazio/uv-cache` |
| venv | `$WORK/epicazio/venvs/pytorch-scaling` (5.9 GB; the uv cache is also 5.9 GB) |
| Pins | `torch==2.9.1+cu126`, `torchvision==0.24.1+cu126`, index `https://download.pytorch.org/whl/cu126` |
| Resolved freeze | 28 lines, `$VENV/requirements-freeze.txt` |
| Remote records | `$VENV/ENV-RECORD.md`, `$VENV/verify-pytorch-env.py` |

Freeze highlights: `numpy 2.4.6`, `pillow 12.3.0`, `nvidia-cudnn-cu12 9.10.2.21`,
`nvidia-nccl-cu12 2.27.5`, `triton 3.5.1`.

The login-node import check passed (`torch 2.9.1+cu126`, `torchvision
0.24.1+cu126`, `cuda 12.6`; `torch.cuda.is_available()` is expected `False` on
login nodes).

### Compute-node verification (environment smoke only)

One `srun` job: `boost_qos_dbg`, 1 task, 1 A100, 5 min; exit 0. The smoke is a
single 4-element multiply/add kernel; no autograd, dataset, training, or
benchmark is involved.

```bash
ssh leonardo "srun -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_dbg -N1 -n1 --cpus-per-task=4 --gpus-per-task=1 --time=00:05:00 bash -lc '/leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/verify-pytorch-env.py'"
```

Raw output block:

```text
python_version=3.11.17
python_executable=/leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python
torch_version=2.9.1+cu126
torch_cuda_version=12.6
torchvision_version=0.24.1+cu126
cuda_is_available=True
nvidia_smi_driver_version=535.274.02
cuda_device_count=1
cuda_device_name_0=NVIDIA A100-SXM-64GB
cuda_device_capability_0=(8, 0)
cuda_device_total_memory_0=68099571712
cudnn_version=91002
nccl_version=(2, 27, 5)
ENV_SMOKE=sum=8.0 expected=8.0 PASS
ENV_VERIFY=PASS
```

## Dataset: Tiny ImageNet-200

- Source: `https://cs231n.stanford.edu/tiny-imagenet-200.zip`.
- Archive: `$WORK/epicazio/datasets/downloads/tiny-imagenet-200.zip`, 248,100,043
  bytes, SHA-256
  `6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821`
  (self-computed; the official page publishes no checksum). The community MD5
  `90528d7ca1a48142e341f4ef8d21d0de` cross-check MATCHes. `unzip -t` reports no
  errors.
- Extracted official layout: `$WORK/epicazio/datasets/tiny-imagenet-200/`
  (249,024,074 bytes apparent; `du -sh` 76M in the setup log).
- Verified counts: train 100,000 JPEG in 200 classes (500 each) plus 200
  `*_boxes.txt`; val 10,000 JPEG with 10,000 annotation lines (50 per class,
  every referenced file exists); test 10,000 JPEG; `wnids.txt` 200 entries.
- Metadata record: `$WORK/epicazio/datasets/tiny-imagenet-200-metadata.md`.
- The validation split remains in the official flat layout (`val/images/` plus
  `val/val_annotations.txt`); no reorganization was performed.

## Reusable scripts

- `scripts/remote/leonardo/setup-pytorch-env.sh` - login-node setup,
  idempotent, with a `--reinstall` flag; a re-run verifies the existing install.
- `scripts/remote/leonardo/verify-pytorch-env.py` - prints the verification
  block above.
- `scripts/remote/leonardo/requirements-pytorch-scaling.txt` - top-level pins
  plus the full resolved freeze.
- `scripts/remote/leonardo/download-tiny-imagenet.sh` - idempotent
  download/extract/verify; a second run skips the download and the extraction
  (the verification always re-runs).

The scripts are staged to `~/leonardo-setup/`; all run on a login node only,
because compute nodes have no internet.

## Evidence

- `artifacts/logs/leonardo/2026-10-15-pytorch-env-setup.log` (284 lines;
  includes the login-node import check, both GPU verification runs, and an
  idempotency rerun).
- `artifacts/logs/leonardo/2026-10-15-pytorch-env-freeze.txt` (28 lines).
- `artifacts/logs/leonardo/2026-10-15-tiny-imagenet-setup.log` (384 lines;
  three runs including the idempotent skips).

## Deviations and findings

- The uv installer needed `UV_INSTALL_DIR` exported before the `curl | sh`
  pipeline, because an environment prefix binds only to `curl`; fixed in
  `setup-pytorch-env.sh`.
- torch 2.9.1 lacks `torch._C._cuda_getDriverVersion`;
  `verify-pytorch-env.py` falls back to
  `nvidia-smi --query-gpu=driver_version` (recorded as 535.274.02).
- `words.txt` is the full WordNet index (82,115 entries), not 200 lines; all 200
  Tiny ImageNet `wnids` are present in it.
- The `research-web` subagent invocation for the environment pin verification
  failed with an HTTP 403 `SUBSCRIPTION_REQUIRED`; direct web search completed
  the verification. See
  [failure record](../../failures/2026-10-15-research-agent-model-unavailable.md).
- No test, benchmark, or training ran; no model or training code was written.

**Exact next action:** write the shared experiment specification and the
training code for both frameworks once the supervisor confirms the open
decisions (AdamW hyperparameters and the batch size via the later memory pilot).
The environment is ready, but no model or training code exists yet and no
experiment has run.

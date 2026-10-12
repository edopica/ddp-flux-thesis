# Leonardo: CUDA artifacts and driverless precompile traps

Date: 2026-10-04
Host: `leonardo` (CINECA). All commands via `scripts/remote/run.sh leonardo`.
No product code changed.

## Failure 1: CUDA artifact download fails on compute nodes

First GPU precompile died with `Unable to automatically download/install
artifact 'CUDA_Runtime' ... Could not connect to server` (later, the same for
`CUDA_Compiler`, `CUDNN`, `NCCL`). Exit 1. Log:
`artifacts/logs/leonardo/2026-10-04-precompile-test-env-gpu2.log`.

Cause: Leonardo compute nodes have no outbound HTTPS; login nodes do. The
`cuda`-tagged JLL artifacts are selected from the *visible driver* at
load/precompile time, so the earlier driverless login resolve/instantiate
(`Pkg.resolve(); Pkg.instantiate()`, 98 artifacts) never selected or downloaded
them. A `LocalPreferences.toml` pin cannot help: Julia reads preferences only
for direct dependencies, and `CUDA_Runtime_jll` is indirect in the test project
(`Base.get_preferences` returns an empty dict; verified on the login node).

Correction: fetch the four artifacts on the login node with explicit platform
tags using the new script `scripts/remote/leonardo/fetch-cuda-artifacts.jl`
(CUDA_Compiler cuda=12, CUDNN 12.0, NCCL 12.9, CUDA_Runtime 12.9). Exit 0; log:
`artifacts/logs/leonardo/2026-10-04-fetch-cuda-artifacts.log`. The precompile
on a GPU node then completed (278 deps, 221 s, exit 0).

## Failure 2: precompile with no driver bakes the wrong CUDA state

The first compute precompile used `GRES=gpu:0` and emitted the
`CUDA.jl's JLLs were precompiled without an NVIDIA driver present` warning.
The next run failed with `CUDA error (code 3, CUDA_ERROR_NOT_INITIALIZED)`,
`cuda_functional=false` despite four working A100s. Log:
`artifacts/logs/leonardo/2026-10-04-verify-stack.log`.

Correction: wipe `$DEPOT/compiled/v1.12` and re-precompile with GPUs allocated
(`GRES=gpu:4`). `cuda_functional=true`, `cudnn_functional=true`, exit 0. Log:
`artifacts/logs/leonardo/2026-10-04-verify-stack-gpu.log`.

## Failure 3: `cd ... && srun ...` escapes the salloc wrapper

Command `run.sh leonardo "cd ~/projects/flux-integration && srun ... julia ..."`
produced `srun: fatal: Can not execute julia` and exit 1. `salloc` executes
only the first command of the list (`cd`), so everything after `&&` ran outside
the allocation without the wrapper's `PATH`/`JULIA_DEPOT_PATH`. Log:
`artifacts/logs/conditional-wrapper/phaseJ-gpu-focused-nccl4-leonardo-bad-wrapper-2026-10-04.log`.

Correction: keep the wrapper command single and set the working directory with
`srun --chdir=<absolute path>`. The rerun passed.

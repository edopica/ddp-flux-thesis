# Phase J: two NCCL harness failures on deathstar

Date: 2026-10-04

Both failed commands were GPU test runs on `deathstar`, not product defects.
No production source changed. Full detail:
`artifacts/logs/conditional-wrapper/phaseJ-metadata.md`.

## Failure 1: cuDNN present but not loaded

Command:

```bash
env FLUX_TEST_DISTRIBUTED_BACKEND=nccl mpiexecjl --project=test -n 2 \
  julia --startup-file=no --project=test \
  test/ext_distributed/cuda_conditional_distributedtest.jl
```

Exit 1 in ~9 s with
`CUDA devices are not functional and force_cuda is set to true. This is caused
by backend: NCCLBackend.` from `FluxMPIExt.__initialize`
(`ext/FluxMPIExt/FluxMPIExt.jl:34`).

Cause: `FluxMPIExt` checks `MLDataDevices.functional(CUDADevice)`, which
MLDataDevices 1.17.10 defines only in its cuDNN extension. Adding `cuDNN` to
`test/Project.toml` makes it resolvable but does not load it; `FluxCUDAExt`'s
`Base.require(Main, :cuDNN)` runs only at precompile time and does not replay
from cache. Correction: the NCCL branch of
`test/ext_distributed/distributed_setup.jl` now imports `cuDNN`. Probe log:
`phaseJ-ext-probe-deathstar-2026-10-04.log`.

## Failure 2: Float32 tolerance below one ULP

After the cuDNN fix, the first full focused run failed one assertion in
`device freeze and thaw preserve values and protocol`:

```
Float32[-0.18800005, …] ≈ Float32[-0.18800004, …] (atol=1.0e-8)
```

Exit 1; 9 passed / 1 failed per rank in that testset.

Cause: `eps(Float32(0.188)) = 1.4901161e-8`, so `atol=1e-8` is 0.67 ULP. The
wrapper averages gradients in Float32 through NCCL; the independent reference
uses an analytic Float64 mean, so a 1-ULP difference is expected. Correction:
`NUMERIC_ATOL = ON_GPU ? 1.0e-6 : 1.0e-8` for reference comparisons.

The pre-fix failure was reproduced deliberately with the old tolerance to keep
a stored log:
`phaseJ-gpu-focused-nccl2-ULP-repro-deathstar-2026-10-04.log` (exit 1, same
evaluated values on both ranks). The corrected run passed 217/217 per rank.

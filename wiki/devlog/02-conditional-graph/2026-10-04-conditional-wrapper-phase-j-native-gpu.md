# Conditional-graph wrapper Phase J: native CUDA/NCCL validation complete

Date: 2026-10-04
Host: `deathstar` (3× TITAN RTX, driver 555.42.02), 2 ranks
Flux worktree: `/home/kurapica/Projects/ddp_flux/flux-integration`, branch
`ddp/integration-m1-spike`, HEAD `55171e65`, dirty (Phase J changed only tests)
Predecessor: [implementation and GPU attempt](2026-10-04-conditional-wrapper-phase-j-native-gpu-attempt.md)
Metadata: `artifacts/logs/conditional-wrapper/phaseJ-metadata.md`

## Outcome

The native CUDA/NCCL gate **passes at two ranks**. The focused device-generic
regression runs on real `CuArray` parameters over NCCL with
`CUDA.allowscalar(false)`:

- `phaseJ-gpu-focused-nccl2-deathstar-2026-10-04.log`: 6 testsets,
  **217/217 assertions per rank**, exit 0, real 52.2 s; rank 0 on device 1,
  rank 1 on device 2.
- Complete dedicated suite, MPI only: `Distributed | 7 7 2m03.6s`, exit 0.
- Stretch goal: complete dedicated suite with `FLUX_TEST_DISTRIBUTED_NCCL=true`,
  all 7 child files under both backends: `Distributed | 14 14 5m01.9s`, exit 0.

No product failure was found. No production source changed. The only test-side
changes are the new file, the cuDNN load in the shared NCCL setup, and a
device-aware reference tolerance.

## Blockers resolved

1. **cuDNN must be loaded, not merely resolvable.** The previous session added
   `cuDNN` to `test/Project.toml`, but the native run still failed with
   `CUDA devices are not functional and force_cuda is set`. The probe
   (`phaseJ-ext-probe-deathstar-2026-10-04.log`) shows `FluxCUDAExt` loads on
   `import CUDA` but does **not** load cuDNN from the precompile cache;
   `MLDataDevices.functional(CUDADevice)` is false until `cuDNN` is imported,
   because MLDataDevices 1.17.10 defines that method only in its cuDNN
   extension. Fix: `test/ext_distributed/distributed_setup.jl` imports `cuDNN`
   in the NCCL branch, before `DistributedUtils.initialize`.
2. **Float32 reference tolerance.** The freeze/thaw test compared Adam moments
   with `atol=1e-8`, below one ULP at 0.188 (`eps(Float32) = 1.49e-8`). The
   wrapper averages in Float32 over NCCL; the independent reference computes an
   analytic Float64 mean. A deterministic 1-ULP difference
   (`-0.18800005` vs `-0.18800004`) failed the assertion. Fix: a device-aware
   `NUMERIC_ATOL` (1e-6 on device, 1e-8 on host) for reference comparisons;
   structural and bitwise checks stay exact. The pre-fix failure is preserved
   by a deliberate reproduction run
   (`phaseJ-gpu-focused-nccl2-ULP-repro-deathstar-2026-10-04.log`, exit 1,
   same values on both ranks).

## Interpretation

The focused testset proves, on real device buffers, that the wrapper keeps
communication outside AD, averages over all ranks, keeps the presence vector on
the host as `Vector{Float32}`, allocates gradient buffers as `CuArray`s, leaves
globally absent leaves and their Adam state bitwise untouched, preserves caller
gradients, handles tied and nested parameters, pairs transpose gradients in
parent space, and honors whole-wrapper freeze/thaw with the ordered collective
protocol. The full-suite NCCL pass additionally shows the existing CPU-array
child tests run unchanged when the NCCL backend is selected (host buffers use
the MPI fallback).

## Deferred / untested

4-rank NCCL weighting (no 4 GPUs on either host), MPI with `CuArray`
parameters, device movement with preserved state, fresh-process checkpoint
continuation with device restoration, partial freeze/thaw on device arrays,
non-square tied transpose on device, and isbits GPU analogues. Reasons per case
are in the metadata table.

## Next action

Phase K (section 12): deterministic toy dataset first, then the MNIST
presentation run, using the 2-rank GPU configuration that passed here.

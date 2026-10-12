# Phase J native CUDA/NCCL validation — metadata

Date: 2026-10-04
Hosts: `deathstar` (3× NVIDIA TITAN RTX, driver 555.42.02, 2 ranks) and
`leonardo` (CINECA, 4× NVIDIA A100-SXM-64GB, driver 535.274.02, 4 ranks)
Ranks: 2 on deathstar; 4 on leonardo (the 4-rank/NCCL weighting case was
closed on 2026-10-04; see the extension section below)
Flux worktree: `~/projects/flux-integration` on deathstar (manual rsync of the
local dirty branch `ddp/integration-m1-spike` at HEAD
`55171e659b2f425da980cbc83177bf5c38f0a7a1`; nothing committed or pushed)
Julia: 1.12.6 (juliaup)
Packages: Flux 0.16.11, Optimisers 0.4.9, Functors 0.5.3, MPI 0.20.27,
Zygote 0.7.13, CUDA 6.2.2, NCCL 0.1.3, cuDNN 6.2.2 (CUDNN_jll 9.26.0+0),
GPUCompiler 1.23.0, MLDataDevices 1.17.10
Backend: `NCCLBackend` over real `CuArray` parameters; CPU checks use
`MPIBackend` (MPICH artifact `1a3503e2…`)

## Status per case

| Case | Testset | Status | Evidence |
|---|---|---|---|
| J1 rank-to-device mapping | `NCCL rank-to-device mapping uses distinct physical devices` | **pass** (5/rank) | focused gate log; ranks 0/1 on devices 1/2 |
| J1 rank-dependent branches, real AD, both update APIs, locally absent grads, caller-grad preservation, changing participation, globally absent steps 3/4 with nonzero Adam history, device grad buffers + host presence vector, independent analytic Adam reference, ordered collective trace | `rank-dependent conditional branches on device arrays` | **pass** (171/rank) | focused gate log |
| J2 tied device parameter reduces once, identity preserved | `tied device parameter reduces once and keeps identity` | **pass** (11/rank) | focused gate log |
| J2 nested device traversal excludes non-trainable arrays | `nested device traversal excludes non-trainable arrays` | **pass** (12/rank) | focused gate log |
| J2 transpose gradients pair in parent space on device arrays | `transpose gradients pair in parent space on device arrays` | **pass** (8/rank) | focused gate log |
| J2 whole-wrapper device freeze/thaw keeps values, moments, protocol | `device freeze and thaw preserve values and protocol` | **pass** (10/rank) after the Float32 tolerance correction | focused gate log; pre-correction failure preserved in the ULP repro log |
| J2 complete dedicated suite, MPI pass | all 7 child files | **pass** (`Distributed \| 7 7 2m03.6s`) | `phaseJ-remote-suite-mpi2-2026-10-04.log` |
| J2 complete dedicated suite, MPI + NCCL passes (stretch) | all 7 child files × 2 backends | **pass** (`Distributed \| 14 14 5m01.9s`) | `phaseJ-remote-suite-nccl2-2026-10-04.log` |
| J1/J2 focused gate total | 217 assertions/rank (5+171+11+12+8+10) | **pass**, exit 0, real 52.2 s | `phaseJ-gpu-focused-nccl2-deathstar-2026-10-04.log` |
| J1/J2 focused gate at **4 ranks** (Leonardo, A100) | same 217 assertions/rank; ranks on devices `[1, 2, 3, 0]` | **pass**, exit 0, srun elapsed 3:17.72 (incl. startup) | `phaseJ-gpu-focused-nccl4-leonardo-2026-10-04.log` |
| J2 complete dedicated suite at **4 ranks**, MPI + NCCL (Leonardo) | all 7 child files × 2 backends | **pass** (`Distributed \| 14 14 5m55.7s`), exit 0 | `phaseJ-remote-suite-nccl4-leonardo-2026-10-04.log` |

Product failures: **none**.

Harness/environment findings (no production change):

1. `FluxMPIExt.__initialize` checks `MLDataDevices.functional(CUDADevice)`,
   which MLDataDevices 1.17.10 defines only in its cuDNN extension
   (`ext/cuDNNExt.jl:37`); without cuDNN loaded the fallback is `false` and
   `force_cuda` errors even though CUDA works. Adding cuDNN to
   `test/Project.toml` is necessary but not sufficient: the test process must
   load it. `FluxCUDAExt`'s `Base.require(Main, :cuDNN)` does not take effect
   from the precompile cache. Fixed in the test harness only by importing
   `cuDNN` in the NCCL branch of `test/ext_distributed/distributed_setup.jl`.
   Probe: `phaseJ-ext-probe-deathstar-2026-10-04.log` (before/after states).
2. The freeze/thaw test compared Float32 Adam moments with `atol=1e-8`, which
   is 0.67 ULP at 0.188 (1 ULP = 1.49e-8); the wrapper (NCCL Float32 average)
   and the reference (analytic Float64 mean) can differ by exactly 1 ULP.
   Fixed by a device-aware `NUMERIC_ATOL = 1e-6` on device / `1e-8` on host;
   structural and bitwise checks stay exact. Reproduction with the pre-fix
   tolerance: `phaseJ-gpu-focused-nccl2-ULP-repro-deathstar-2026-10-04.log`
   (exit 1, same `-0.18800005 ≈ -0.18800004`).

Untested / deferred (with reason):

| Case | Reason |
|---|---|
| MPI with CuArray parameters | Separate backend configuration (plan J2); not exercised |
| Device movement with preserved state | Not implemented in this focused file (plan J2 lists it) |
| Fresh-process checkpoint continuation with explicit device restoration | Not implemented; CPU/MPI continuation exists (Phase H) |
| Partial freeze/thaw on device arrays | Focused GPU file covers whole-wrapper freeze only; partial freeze is CPU/MPI only |
| Transpose of non-square tied device arrays | Focused case uses a square parent; non-square not exercised |
| Isbits GPU analogue | No meaningful GPU analogue; isbits fixtures remain CPU tests (plan J2) |

## Exact commands and results

All remote commands run from the thesis repo root through
`scripts/remote/run.sh deathstar` unless noted; workdir on deathstar is
`~/projects/flux-integration`.

| # | Command | Exit | Result | Log |
|---|---|---|---|---|
| 1 | `ssh deathstar "nvidia-smi --query-gpu=index,name,memory.used,driver_version --format=csv"` | 0 | 3× TITAN RTX, 555.42.02, 6/6/16 MiB idle | (inline) |
| 2 | sync `flux-integration` excluding `test/Manifest.toml`, root `Manifest.toml`, `Manifest.toml.phaseF-bak`; re-scp `.phaseJ_resolve_env.jl` | 0 | remote hashes match local | (inline) |
| 3 | `JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no --project=test .phaseJ_resolve_env.jl` | 0 | cuDNN 6.2.2 + CUDNN_jll 9.26.0+0, 733.5 MiB artifact; 6 deps precompiled, 29 s; 298 already precompiled | `phaseJ-precompile-deathstar-2026-10-04.log` |
| 4 | `julia --project=test -e 'using cuDNN, CUDA; @show cuDNN.functional() CUDA.functional()'` | 0 | `cuDNN.functional() = true`, `CUDA.functional() = true` | `phaseJ-cudnn-verify-deathstar-2026-10-04.log` |
| 5 | `julia --project=test .phaseJ_ext_probe.jl` (diagnostic) | 0 | `FluxCUDAExt loaded: true`; cuDNN loaded after `import CUDA`: false; `functional(CUDADevice)`: false; after explicit `import cuDNN`: true | `phaseJ-ext-probe-deathstar-2026-10-04.log` |
| 6 | `env FLUX_TEST_DISTRIBUTED_BACKEND=nccl mpiexecjl --project=test -n 2 julia --startup-file=no --project=test test/ext_distributed/cuda_conditional_distributedtest.jl` | 0 | 217/217 per rank; 6 testsets; devices `[1, 2]`; real 52.2 s | `phaseJ-gpu-focused-nccl2-deathstar-2026-10-04.log` |
| 7 | `env JULIA_MPI_TEST_NPROCS=2 FLUX_TEST_DISTRIBUTED_TIMEOUT=600 julia --startup-file=no --project=test test/ext_distributed/runtests.jl` | 0 | `Distributed \| 7 7 2m03.6s` | `phaseJ-remote-suite-mpi2-2026-10-04.log` |
| 8 | `env JULIA_MPI_TEST_NPROCS=2 FLUX_TEST_DISTRIBUTED_NCCL=true FLUX_TEST_DISTRIBUTED_TIMEOUT=1200 julia --startup-file=no --project=test test/ext_distributed/runtests.jl` | 0 | `Distributed \| 14 14 5m01.9s` (MPI + NCCL) | `phaseJ-remote-suite-nccl2-2026-10-04.log` |
| 9 | same as #6 with `.phaseJ_ulp_repro.jl` (`NUMERIC_ATOL = 1.0e-8`) | 1 | 9/10 freeze testset, 1-ULP moment mismatch (intended reproduction) | `phaseJ-gpu-focused-nccl2-ULP-repro-deathstar-2026-10-04.log` |
| 10 | `julia --project=test -e 'using Pkg; Pkg.status()'` | 0 | final package list incl. cuDNN 6.2.2 | `phaseJ-versions-final-deathstar-2026-10-04.log` |

Earlier failing iterations (`force_cuda` before the cuDNN-load fix) reused the
focused log path and were overwritten by the passing run; the exact stack and
diagnosis are preserved in `temp/phaseJ_handoff.md` section 6 and
`wiki/devlog/2026-10-04-conditional-wrapper-phase-j-native-gpu-attempt.md`.

## Source and test hashes at test time

| Path | md5 |
|---|---|
| `test/ext_distributed/cuda_conditional_distributedtest.jl` | `dabdaddb77301d93f83707c1c930bc6f` |
| `test/ext_distributed/distributed_setup.jl` | `d23c82a82bae581a0af6f86de06f2ace` |
| `test/Project.toml` | `ca91c533eaef89202158d612207172f8` |
| `test/ext_distributed/runtests.jl` | `ed1a027d07888731db44bafc15692a90` |
| `src/distributed/public_api.jl` | `9bfa981384eaf88750ec3d19e8bb76bd` |
| `NEWS.md` | `5aaab49b3457cb06b9749b234cc6f3b0` |
| `test/ext_distributed/synchronized_distributedtest.jl` | `12f7d570ff965bfced468994299f378a` |

The first two files changed in Phase J (`cuda_conditional_distributedtest.jl`
is new; `distributed_setup.jl` gained the cuDNN import). No production source
changed in Phase J; `src/distributed/public_api.jl`, `NEWS.md`, and
`synchronized_distributedtest.jl` carry the uncommitted Phase H changes.

## Leonardo 4-rank extension (2026-10-04)

Fresh bring-up on CINECA Leonardo: Julia 1.12.6 on `$WORK`
(`$WORK/epicazio/julia-1.12.6`, depot `$WORK/epicazio/julia-depot`), 4×
A100-SXM-64GB, driver 535.274.02, MPICH (`srun --mpi=pmi2`), Slurm QoS
`boost_qos_dbg`. Host config: `scripts/remote/hosts/leonardo.conf`. All
commands through `scripts/remote/run.sh leonardo`.

| # | Command (inner `srun` shown) | Exit | Result | Log |
|---|---|---|---|---|
| L1 | srun -n 1 -c 8: `Pkg.precompile()` on a GPU-allocated node | 0 | 278 deps in 221 s (CUDA 6.2.2, NCCL 0.1.3, cuDNN 6.2.2, FluxCUDAExt, FluxMPINCCLExt) | `2026-10-04-precompile-test-env-gpu3.log` |
| L2 | srun -n 1 -c 4: `verify-stack.sh` | 0 | `cuda_functional=true`, `cudnn_functional=true`, `ndevices=4` | `2026-10-04-verify-stack-gpu.log` |
| L3 | srun -n 4 -c 2: `smoke-mpi4.sh` | 0 | ranks 0-3, `MPI 4-rank smoke OK` | `2026-10-04-smoke-mpi4.log` |
| L4 | srun -n 4 -c 4: `smoke-nccl4.sh` | 0 | devices `[1, 2, 3, 0]`, allreduce 10.0 = expected | `2026-10-04-smoke-nccl4.log` |
| L5 | `FLUX_TEST_DISTRIBUTED_BACKEND=nccl` srun -n 4 -c 4: focused file | 0 | **217/217 per rank**; devices `[1, 2, 3, 0]` | `phaseJ-gpu-focused-nccl4-leonardo-2026-10-04.log` |
| L6 | `JULIA_MPI_TEST_NPROCS=4 FLUX_TEST_DISTRIBUTED_NCCL=true` srun -n 1 -c 8: `runtests.jl` | 0 | `Distributed \| 14 14 5m55.7s` | `phaseJ-remote-suite-nccl4-leonardo-2026-10-04.log` |

Bring-up findings (no product change):

- Compute nodes have no outbound HTTPS. The driverless login node does not
  select the `cuda`-tagged JLL artifacts, so they must be fetched on the login
  node before the first GPU precompile:
  `scripts/remote/leonardo/fetch-cuda-artifacts.jl` (CUDA_Compiler, CUDNN,
  NCCL, CUDA_Runtime; tags 12/12.0/12.9/12.9).
- Precompiling with `GRES=gpu:0` bakes CUDA.jl's "precompiled without an NVIDIA
  driver" state; the CUDA stack must be precompiled on a GPU-allocated node.
- In a `run.sh` command, `cd ... && srun ...` splits the wrapper: the part
  after `&&` runs outside the salloc without the Julia PATH/depot. Use
  `srun --chdir=<abs path>`.
- `--gres=gpu:4` (all GPUs visible, rank `r` → `mod1(r + 2, 4)`), not
  `--gpus-per-task=1`, for the focused regression (`CUDA.ndevices() >= ranks`).

## Remote hygiene

- Root workspace `Manifest.toml` regenerated on deathstar; Phase F backup kept
  at `~/projects/flux-integration/Manifest.toml.phaseF-bak`.
- Temporary remote files `.phaseJ_ext_probe.jl` and `.phaseJ_ulp_repro.jl`
  removed after use.
- Post-run `nvidia-smi`: GPUs idle (6/6/16 MiB); no stray julia/mpiexec
  processes.
- Leonardo: no queued/running jobs and no stray julia processes after the runs
  (`squeue -u epicazio` empty); all allocations released. Logs collected under
  `artifacts/logs/leonardo/`.

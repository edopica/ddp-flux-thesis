# Leonardo bring-up: 4-rank native NCCL validation

Date: 2026-10-04
Host: `leonardo` (CINECA), 4× NVIDIA A100-SXM-64GB, driver 535.274.02
Flux worktree: `~/projects/flux-integration` (manual rsync of the local dirty
branch `ddp/integration-m1-spike` at HEAD `55171e65`; nothing committed)
Julia: 1.12.6 on `$WORK` (depot `$WORK/epicazio/julia-depot`)
Predecessor: [Phase J native GPU validation](2026-10-04-conditional-wrapper-phase-j-native-gpu.md)
Metadata: `artifacts/logs/conditional-wrapper/phaseJ-metadata.md`

## Outcome

The previously deferred **4-rank/NCCL weighting case passes**, closing the
Phase J untested list. On one 4×A100 node:

- Focused device regression (`CUDA.allowscalar(false)`, real `CuArray`
  parameters): **217/217 assertions per rank**, exit 0; ranks mapped to
  devices `[1, 2, 3, 0]`. srun elapsed 3:17.72 including startup.
- Full dedicated suite at 4 ranks under **both** MPI and NCCL:
  `Distributed | 14 14 5m55.7s`, exit 0.
- 4-rank MPI and NCCL smokes pass; NCCL smoke allreduce 10.0 = expected.
- `verify-stack`: `CUDA.functional() = true`, `cuDNN.functional() = true`,
  `ndevices = 4`, MPI library MPICH.

No product failure and no production source change. The local Flux worktree,
`flux-integration`, is unchanged by this session.

## Bring-up path

Leonardo had no Julia workflow. Installed Julia 1.12.6 on `$WORK`, created
`scripts/remote/hosts/leonardo.conf` (salloc wrapper: account `iscrc_unmasked`,
partition `boost_usr_prod`, QoS `boost_qos_dbg`, `-N1 --gres=gpu:4`,
`SLURM_MPI_TYPE=pmi2`, `JULIA_DEPOT_PATH`/`PATH` on `$WORK`), synced
`flux-integration` and the thesis repo, resolved the test environment on the
login node, and precompiled on a GPU-allocated compute node
(278 dependencies, 221 s). Full commands: `wiki/context/leonardo.md`.

Three traps, all recorded in
`wiki/failures/2026-10-04-leonardo-cuda-artifact-prefetch.md`:

1. **Compute nodes have no internet; the driverless login does not select the
   CUDA artifacts.** `CUDA_Runtime_jll` picks its artifact from the visible
   driver. A login-side `Pkg.instantiate()` therefore downloads nothing
   `cuda`-tagged, and the first GPU precompile fails trying to download
   `CUDA_Runtime`, `CUDNN`, `NCCL`, and `CUDA_Compiler`. Fixed with
   `scripts/remote/leonardo/fetch-cuda-artifacts.jl`, which installs the four
   artifacts with explicit platform tags on the login node.
2. **Precompiling with `GRES=gpu:0` bakes the no-driver CUDA state.** CUDA.jl
   must first see a driver: precompile the CUDA stack under `GRES=gpu:4`.
3. **`cd ... && srun ...` escapes the wrapper.** `salloc` executes only the
   first command; everything after `&&` runs outside the allocation without
   `PATH`/`JULIA_DEPOT_PATH` (`srun: Can not execute julia`). Use
   `srun --chdir=<absolute path>`.

A `LocalPreferences.toml` pin for `CUDA_Runtime_jll` was tried first but is
ignored because the package is an indirect dependency (Julia only reads
preferences for direct dependencies); it was removed without effect.

## Interpretation

The Phase J wrapper contract holds at four ranks on A100s with a different
MPI toolchain (MPICH via `srun --mpi=pmi2`) and a different CUDA runtime than
deathstar: rank-ordered collectives, global-mean weighting by `1/num_ranks`,
host presence metadata, tied/nested/transpose cases, and whole-wrapper
freeze/thaw all pass per rank and the full existing suite is green under both
backends. The deferred list is now: MPI with `CuArray` parameters, device
movement with preserved state, fresh-process checkpoint continuation with
device restoration, partial device freeze/thaw, non-square tied transpose, and
isbits GPU analogues.

## Next action

Phase K (plan section 12): deterministic toy dataset first, then the MNIST
presentation run. Both 2-rank (deathstar) and 4-rank (leonardo) native
configurations are now validated for the comparison.

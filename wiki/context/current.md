# Current context

Date: 2026-07-30
Active checkpoint: C8 - Correctness battery
Active Flux branch: `master` (at `d583411f`, merging `ddp/docs-examples` + `upsteam-pr`)
Flux repo path: `/home/kurapica/Projects/ddp_flux/ddp_flux`  
Preserved reference branch: `upsteam-pr`

## What is known

- The local Flux.jl fork exists.
- The first implementation priority was understanding the current distributed code (C0-C6b done).
- CPU/MPI verification and upstream PR prep is mostly done.
- C7 is complete: an end-to-end example `scripts/examples/train_ddp.jl` was successfully created, executed on HPC using `make example-ddp`, and added as a documentation guide (`distributed.md`) to the Flux upstream docs.
- The `resolve_unused_parameters!` API in Flux was updated to `!!` in recent upstream PR work, and our example scripts are aligned with this.
- HPC precompilation works smoothly with `scripts/remote/precompile.sh hpc`.

## What was done last

- **C8.2 Collective Invariants (2026-07-30):**
  - Rewrote `test/ext_distributed/common_distributedtest.jl` with five test blocks:
    1. Data Type Coverage Matrix (MPI: Float32/Float64 on `Array`; NCCL: Float16/Float32 on `CuArray`)
    2. Nonzero Root Broadcast (root=1)
    3. Sum vs. Average Convention (nworkers==2: [2.0]/[4.0] → avg [3.0], sum [6.0])
    4. Buffer Reuse Leakage (fill(rank,4) → mutate → fill(rank*2,4) → allreduce again)
    5. NCCL vs MPI Math Equivalence (large Float32 CuArray, NCCL allreduce vs MPI allreduce via `backend.mpi_backend`, max abs diff < 1e-6)
  - **HPC verification:** MPI/CPU 36/36 PASS (2 ranks, test project env); NCCL/GPU 37/37 PASS (2 ranks, thesis project env, `FLUX_TEST_DISTRIBUTED_NCCL=true`).
  - **Test env resolution issue (documented, not fixed):** Adding CUDA/NCCL to `test/Project.toml` triggers a `StridedViews` conflict (explicit transitive pin 0.4.6 vs CUDA's 0.5 requirement). Workaround: run NCCL tests against the thesis project env (`~/projects/ddp-flux-thesis`), which has CUDA/NCCL precompiled. `test/Project.toml` was reverted to its original state.

## Commands that pass

- `make example-ddp` (HPC and local)
- `make precompile`
- MPI/CPU `common_distributedtest.jl` (36/36, 2 ranks)
- NCCL/GPU `common_distributedtest.jl` (37/37, 2 ranks, thesis env)
- (All previous C1-C6b checks)

## Open questions

- `test/Project.toml` cannot resolve with CUDA/NCCL added (StridedViews conflict). Fix needed before CI can run the NCCL path from the test project; candidate fix: investigate which transitive dep pins StridedViews 0.4.6 and loosen it, or restructure the test env.
- Deadlock exit condition must be verified on HPC (local MPI broken)
- N=0 `@test_throws ArgumentError` — source may not yet throw; test documents expected behavior, source fix may need separate PR

## Next exact action

Proceed to the next C8 sub-item: Functors traversal ordering test in `synchronized_distributedtest.jl`, or multi-step descent equivalence in `optimizer_distributedtest.jl`.

# Current context

Date: 2026-07-29
Active checkpoint: C8 - Correctness battery
Active Flux branch: `master` (at `d583411f`, merging `ddp/docs-examples` + `upsteam-pr`)
Flux repo path: `/home/kurapica/Projects/ddp_flux/ddp_flux`  
Preserved reference branch: `upsteam-pr`

## What is known

- The local Flux.jl fork exists.
- The first implementation priority was understanding the current distributed code (C0-C6b done).
- CPU/MPI verification and upstream PR prep is mostly done.
- C7 is complete: an end-to-end example `scripts/examples/train_ddp.jl` was successfully created, executed on HPC using `make example-ddp`, and added as a documentation guide (`distributed.md`) to the Flux upstream docs.
- The example explicitly handles a conditional graph (using `resolve_unused_parameters!!`) and operates conditionally on NCCL if CUDA is present.
- The `resolve_unused_parameters!` API in Flux was updated to `!!` in recent upstream PR work, and our example scripts are aligned with this.
- HPC precompilation works smoothly with `scripts/remote/precompile.sh hpc`.

## What was done last

- **C8.0/C8.1 harness implementation (2026-07-29):**
  - Fixed launcher `@test true` → `proc.exitcode == 0` in `test/ext_distributed/runtests.jl`
  - Created `test/ext_distributed/helper.jl` with `run_with_enforced_exit()`, `set_rank_seed!()`, `compare_structures()`
  - Rewrapped all 6 distributed test files to use `run_with_enforced_exit()` pattern
  - Added watchdog timeout (120s) + SIGUSR1 stack dump mechanism
  - CI YAML `distributed_ci.yml`: 2-rank fast + 4-rank edge jobs, direct `mpiexecjl` invocation
  - Fixed cyclic padding bug: `public_api.jl:273` now uses `mod1(i, total_size)` instead of unrestricted range
  - Added N=1, N=2, N=0 test cases to `data_distributedtest.jl`
  - Created correctness matrix at `wiki/testing/c8-correctness-matrix.md`
  - **All changes uncommitted in Flux working tree** — 9 modified + 2 new files
- **Deadlock verification:** Deferred — local MPICH segfaults with Julia 1.12/MPI.jl (see `wiki/failures/`)

## Commands that pass

- `make example-ddp` (HPC and local)
- `make precompile`
- (All previous C1-C6b checks)

## Open questions

- Deadlock exit condition must be verified on HPC (local MPI broken)
- N=0 `@test_throws ArgumentError` — source may not yet throw; test documents expected behavior, source fix may need separate PR

## Next exact action

Commit the Flux working tree changes, then run deadlock verification on HPC or CI.

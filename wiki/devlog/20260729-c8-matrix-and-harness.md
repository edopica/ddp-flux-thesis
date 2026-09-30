# Devlog: 2026-07-29 (C8 Correctness Harness)

## Context

Starting the C8 correctness battery phase. The objective is to establish unambiguous correctness proofs for Distributed Data Parallel (DDP) training in Flux, including MPI baselines, NCCL native operations, asymmetric data sharding, and conditional model synchronization. 

## Actions Taken

1. **Repository Alignment:** Updated `C8.md` plan to use `master` branch exclusively rather than `ddp/tests`, aligning with `README.md` and `wiki/context/current.md`.
2. **Correctness Matrix:** Created `wiki/testing/c8-correctness-matrix.md` to map out all required testing invariants (data sharding, collectives, equivalences, conditional usage) to their respective behaviors, execution environments, and historical guard constraints.
3. **Behavior Formalization:** Formally established the expected public behavior conventions for DDP within the matrix (e.g. `ArgumentError` on empty datasets, cyclic repitition on `N < world_size`, explicit mathematical scaling for loss reductions, and freezing optimizer state for entirely unused parameters).
4. **CI Preparation:** Prepared the blueprint for the distributed testing launcher and the CI workflow pipeline. 
5. **C8.0/C8.1 Harness Implementation (afternoon):**
   - Fixed launcher `@test true` → `proc.exitcode == 0` in `test/ext_distributed/runtests.jl`
   - Implemented `test/ext_distributed/helper.jl` with `run_with_enforced_exit()`, `set_rank_seed!()`, `print_test_header()`, `compare_structures()`
   - Rewrapped all 6 `_distributedtest.jl` files to use `run_with_enforced_exit()` pattern
   - Added watchdog timeout + SIGUSR1 stack dump mechanism on timeout
   - CI YAML (`distributed_ci.yml`): 2-rank MPI fast + 4-rank MPI edge jobs, direct `mpiexecjl` invocation
   - **Cyclic padding fix:** `public_api.jl:273` — `append!(indices, 1:(...))` → `append!(indices, [mod1(i, total_size) for i in 1:(...)])`
   - **New tests in `data_distributedtest.jl`:** N=1 (cyclic repeat), N=2 (valid indices), N=0 (`@test_throws ArgumentError`)
6. **Deadlock verification:** Deferred — local MPICH segfaults with Julia 1.12/MPI.jl. Harness logic verified by inspection. Documented in `wiki/failures/20260729-mpi-setup-segfault.md`. Will reattempt on Bocconi.
7. **Project.toml cleanup:** Reverted accidental MPI/MPIPreferences hard-dep additions from setup attempt.

## Next Steps

- Commit the Flux working tree changes (harness + cyclic padding + N<world_size tests)
- Run deadlock verification on Bocconi where MPI.jl is properly configured
- Begin C8.2: strengthen collective tests (sum-vs-average convention, buffer reuse, Functors traversal)

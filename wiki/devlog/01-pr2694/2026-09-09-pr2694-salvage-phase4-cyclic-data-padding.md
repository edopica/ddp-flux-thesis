# 2026-09-09 — PR #2694 salvage phase 4: complete cyclic data padding (RED + GREEN)

Follows phases 1-2 (`...-phases-1-2.md`), phase 3a (`...-phase3a-launcher-guard-tests.md`), and phase 3b (`...-phase3b-pmi-guardrail.md`). Phase 4 of `temp/pr_salvage_plan.md` is now complete. Delegated to subagents; orchestrator verified the end state.

## What was done

Worktree `/home/kurapica/Projects/ddp_flux/flux-pr2694-salvage`, branch `ddp/pr2694-salvage`:

- Commit `e64165d3` "Add cyclic data-padding test matrix for DistributedDataContainer (phase 4 RED)" — tests only.
- Commit `4eb5708e` "Complete cyclic data padding in DistributedDataContainer (phase 4 GREEN)" — `src/distributed/public_api.jl` only.

### 4a RED — test matrix (`test/ext_distributed/data_distributedtest.jl`)

Rewrote the data test as a five-size cyclic-padding matrix driven by a shared spec helper:

```
per   = cld(N, W)                  # every rank receives `per` observations
total = per * W                    # padded length of the global index list
full  = [mod1(i, N) for i in 1:total]
Rank `r` (0-based) gets full[(r*per + 1):((r+1)*per)]
```

For each non-empty N ∈ {1, 2, 7, 10} under both rank counts (W = 2 and W = 4 at runtime, via `FLUX_TEST_DISTRIBUTED_BACKEND=mpi` + `mpiexec -n`), every rank asserts:

- the spec list properties: length `total`, all indices in `1:N`, `full[1:N] == collect(1:N)` (every original index occurs before padding), and the tail equals the cyclic repeats;
- the identity container (`data = Float32.(1:N)`) yields exactly `cld(N, tworkers)` observations whose values equal the rank's expected cyclic chunk (exact local indices);
- the allreduce aggregate of a random container's local sums equals `sum(data[full])` — the aggregate includes exactly the documented duplicates.

N = 0 asserts `@test_throws ArgumentError` on construction.

Against the old padding (`append!(indices, 1:(total_padded - total_size))`), the matrix is RED exactly where the old code is wrong:

- `-n 4`: N = 1 fails on every rank — the buggy code appends `1:3` to a 1-element dataset, producing invalid indices 2 and 3; the identity container throws `BoundsError`, and the caught-error ranks feed `NaN32` into the allreduce so the duplicate-aggregate test fails too. Top-level: `42 pass / 1 fail / 2 error` and `44 pass / 1 fail`. EXIT 1.
- `-n 2`: at two ranks the padding tail stays within range, so all sizes pass; N = 0 also happens to pass because Julia's `Iterators.partition(Int[], 0)` throws an `ArgumentError` on this version. EXIT 0.

Logs: `artifacts/logs/pr2694-salvage/phase4-data-RED-n2.log`, `artifacts/logs/pr2694-salvage/phase4-data-RED-n4.log`.

### 4b GREEN — implementation (`src/distributed/public_api.jl`)

`__construct_distributed_data_container` now:

1. rejects `numobs(data) == 0` with a clear `ArgumentError` before any partitioning (replaces the old incidental failure);
2. uses `size_per_worker = cld(total_size, split_across)` instead of `Int(ceil(total_size / split_across))`;
3. builds the whole padded index list as `indices = [mod1(i, total_size) for i in 1:total_padded]`, which keeps the original indices `1:N` first and then repeats them cyclically, so every index stays in `1:N` regardless of how the worker count compares to the dataset size;
4. keeps the deterministic rank-contiguous partitioning (`Iterators.partition` + per-rank slice) unchanged, so every shard is length `cld(N, W)`.

The `DistributedDataContainer` docstring now documents that the dataset is not modified (only the internal index sequence is padded), shards are equal length `cld(N, workers)`, padding duplicates observations, metrics over sharded/padded data count the duplicates and must account for them, and an empty dataset is rejected with `ArgumentError`.

### Verification

`FLUX_TEST_DISTRIBUTED_BACKEND=mpi mpiexec -n {2,4} julia --project=test --startup-file=no test/ext_distributed/data_distributedtest.jl`

- `-n 2`: `DistributedDataContainer cyclic padding` 45/45 pass per rank, EXIT 0.
- `-n 4`: same 45/45 per rank, EXIT 0.
- No `Fail`/`Error`/`BoundsError` in either GREEN log.

Logs: `artifacts/logs/pr2694-salvage/phase4-data-GREEN-n2.log`, `artifacts/logs/pr2694-salvage/phase4-data-GREEN-n4.log`.

## Scope boundaries honored

- Only the data-padding defect + empty-data rejection were addressed. No optimizer changes, no `DistributedOptimizerState`, no NCCL claims, no new guide.
- The divergent commit `9d4ddc52` was NOT cherry-picked; only the implementation idea (mod1 cyclic repeat + empty guard) was manually ported into the salvage history.
- Test file follows the fa3bf228 harness (ENV backend, `distributed_setup.jl`, no ARGS, no helper.jl).

## State after session

`flux-pr2694-salvage` @ `4eb5708e` [`ddp/pr2694-salvage`], clean. Logs in `ddp-flux-thesis/artifacts/logs/pr2694-salvage/`.

## Next exact action

Phase 5 of `temp/pr_salvage_plan.md` — retain and polish Carlo's test changes: rename "Sharding & Asymmetry" CI job to "MPI DDP Correctness (4 Ranks)"; remove unused `FLUX_TEST_DISTRIBUTED_MPI` from jobs that invoke the runner directly; remove the commented NCCL job; phrase output handling as "explicitly inherit child streams"; avoid claiming a perfect process supervisor; preserve the optional NCCL route without claiming CPU-buffer fallback tests validate native NCCL. When replying to Carlo, explain that `distributed_setup.jl` centralizes setup instead of `test/test_module.jl` because each MPI rank is a fresh standalone Julia process.

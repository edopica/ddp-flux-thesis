# Phase H independent review

## Result

No blocking findings in the Phase H corrections within the documented CPU/MPI scope.
The combined dedicated suites passed at two and four ranks.
Phase I API acceptance remains a user decision.

The review covered the plan, compatibility audit, four issue reports, production diff, synchronization tests, checkpoint probe, and native Optimisers semantics.
Flux remains at `55171e659b2f425da980cbc83177bf5c38f0a7a1` with the existing three modified files.
This review changed no Flux source or tests.

## Conclusions

- Direct assignment to `Leaf.state` retains scalar replacements and leaf identity. Native Optimisers uses that identity for tied-gradient accumulation.
- The regressions use rank-distinct histories, a nonzero root, independent native references, and both update APIs.
- Rules and frozen flags remain local by explicit contract. The tests demonstrate divergence before configuration alignment and correct continuation afterward.
- Explicit `.tree` migration meets the plan. The audit identifies field, indexing, traversal, return-value, and historical-checkpoint incompatibilities.
- Serialization and JLD2 continuation use data-only payloads and fresh backends. The frozen leaf has nonzero history.
- The hook comparison identifies the required whole-model boundary before gradient pruning.

The existing non-finite-loss risk remains in `src/train.jl:375-378`.
A rank can skip its update while another rank enters collectives. The audit correctly limits its supported path to consistent participation.
Historical per-leaf checkpoint conversion and native GPU/NCCL behavior remain outside the evidence.
The NEWS entry still needs an issue or PR link before upstream submission.

## New evidence

All commands returned exit 0. No timeout or failed command occurred.
The suite times are the reported testset times, not separately measured wall times.

| Run | Result | Time |
|---|---|---|
| Dedicated suite, two ranks | 6/6 child files | 1 m 58.7 s |
| Dedicated suite, four ranks | 6/6 child files | 2 m 22.8 s |
| Compatibility/write, two ranks | 61/61 assertions per rank | 17.582 s |
| Fresh-process resume, two ranks | 41/41 assertions per rank | 15.822 s |
| Ambiguity check | 0 ambiguities | Not measured |

Both suites include the conditional tests and all synchronization regressions.
The new synchronization testsets passed 160, 288, and 32 assertions per rank at both rank counts.
The file also contains 19 earlier top-level assertions, for 499 assertions per rank.
Checkpoint jobs passed 204 assertions in total.
The local two-rank run precompiled `FluxEnzymeExt` once. The test watchdog did not expire.

Logs: `temp/phase_h/review-2026-10-02-suite-mpi{2,4}.log` and `review-2026-10-02-continuation-{write,resume}-mpi2.log`.
The continuation logs record versions, package paths, expanded MPI commands, and this production source SHA-256:
`2984210b84e136da9bc9948c3b5928877f1f6c10f870f8db841ad2d4ab6136f0`.

## Exact commands

Working directory: `/home/kurapica/Projects/ddp_flux/ddp-flux-thesis`.
These commands record this review. A repeat run needs fresh evidence filenames.

```bash
ls temp/phase_h /tmp/opencode && git -C ../flux-integration diff --check && git diff --check
set -o pipefail && JULIA_MPI_TEST_NPROCS=2 timeout --signal=TERM --kill-after=10s 600s julia --startup-file=no --project=../flux-integration/test ../flux-integration/test/ext_distributed/runtests.jl mpi 2>&1 | tee temp/phase_h/review-2026-10-02-suite-mpi2.log
set -o pipefail && JULIA_MPI_TEST_NPROCS=4 timeout --signal=TERM --kill-after=10s 600s julia --startup-file=no --project=../flux-integration/test ../flux-integration/test/ext_distributed/runtests.jl mpi 2>&1 | tee temp/phase_h/review-2026-10-02-suite-mpi4.log
timeout --signal=TERM --kill-after=10s 420s julia --startup-file=no --project=../flux-integration/test temp/phase_h/04-run-migration.jl temp/phase_h/review-2026-10-02-continuation
julia --startup-file=no --project=../flux-integration/test -e 'using Flux, Test; ambiguities = Test.detect_ambiguities(Flux.DistributedUtils; recursive=false); println("ambiguities=", length(ambiguities)); @assert isempty(ambiguities)'
git diff --check && git -C ../flux-integration diff --check && git -C ../flux-integration status --short
```

## Exact next action

Conduct Phase I user review of the wrapper API, synchronization contract, migration policy, remaining limitations, and combined CPU/MPI evidence.

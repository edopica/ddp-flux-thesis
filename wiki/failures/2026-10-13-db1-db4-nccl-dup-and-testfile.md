# 2026-10-13 — DB-1/DB-4 campaign failures: suite log-test defect and NCCL Duplicated-route timeout

Two commands failed during the DB-1/DB-4 GREEN campaign (ADR-0008). Both are
test-tooling defects, not product defects. Each was diagnosed, fixed, and
re-run; the final gates are GREEN (dedicated suite 8/8 files at 2 and 4 MPI
ranks and 16/16 with NCCL; harness 12 PASS + 1 UNTESTED at 2 NCCL ranks).

Recorded per the repository rule to log failed commands.

## 1. Dedicated suite mpi2 attempt 1 — extra log record in the new testset

Command (deathstar, from the thesis root):

```
scripts/remote/run.sh deathstar "JULIA_MPI_TEST_NPROCS=2 FLUX_TEST_DISTRIBUTED_NCCL=false \
  julia --startup-file=no --project=../flux-integration/test \
  ../flux-integration/test/ext_distributed/runtests.jl"
```

Result: 7/8 child files passed; `nonfinite_loss_distributedtest.jl` failed 1 of
10 assertions in `no trainable parameters still votes exactly once per step` at
`nonfinite_loss_distributedtest.jl:303`. The strict
`@test_logs (:warn, r"setup found no trainable parameters")` captured an extra
`LogRecord(Warn, "trainable(x) should now return a NamedTuple with the field
names, not a Tuple", Optimisers, :interface, ...)` because
`Optimisers.trainable(::NoParamsModel) = ()` returned a legacy Tuple. Log:
`artifacts/logs/conditional-wrapper/numerical-guards/suite-deathstar/suite-mpi2-attempt1-dup-warning-failure.log`.

Fix: `Optimisers.trainable(::NoParamsModel) = NamedTuple()` and
`match_mode = :any` on the `@test_logs`. Re-run:
`suite-deathstar/suite-mpi2-r2.log`, `Distributed | 8 8 2m27.2s`, exit 0.

## 2. NCCL harness attempt 1 — `dup-train-one` startup watchdog

Command (deathstar, from the thesis root):

```
scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group loss --ranks 2 --backend nccl \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-loss-nccl2"
```

Result: 12 PASS + 1 FAIL. `dup-train-one` hit the 420 s startup watchdog (exit
15, 421.8 s elapsed). No rank printed `GUARD_PROBE_READY`: the
`Enzyme.Duplicated` eager-route availability check never completed. The
termination stacks show both rank PIDs (554143, 554144) terminated with signal
15 while waiting in the Julia scheduler (`pthread_cond_wait` -> `uv_cond_wait`
-> `ijl_task_get_next`, `scheduler.c:523`) after
`Allocations: 1,563,431,388`. Metadata reasons: `missing CASE_RESULT rank=1`,
`rank 0: expected 9 collectives, got 0`, `rank 1: expected 9 collectives, got
0`. Artifacts retained: `green-deathstar-loss-nccl2/`.

Resolution: the route is recorded UNTESTED under CUDA/NCCL. The probe now
reports it explicitly instead of retrying the hanging availability check:
`CASE_RESULT ... status=UNTESTED reason="Duplicated route unavailable: ... the
availability check exceeded the startup watchdog (420 s) in the
green-deathstar-loss-nccl2 attempt; MPI exercises this route"`. The route
remains exercised and PASS under MPI at 2 ranks.

The same pass fixed a latent probe bug: the probe's own agreement vote used
`BACKEND.comm`, which is an NCCL communicator under NCCL; it now uses the MPI
communicator (`BACKEND.mpi_backend.comm` under NCCL). Under MPI the two are the
same communicator, so no MPI result (RED or GREEN) is affected.

Re-run: `green-deathstar-loss-nccl2-r2/`, 12 PASS + 1 UNTESTED, exit 0
(`raw-dup-train-one-nccl2-fault1.log` records the explicit UNTESTED status).

## Impact

No product defect, no unintended Flux source change, no commit. Final suite and
harness results are in
`artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md`; the
resolution is summarized in
[audit 002](../audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md#resolution-2026-10-13)
and the [fix devlog](../devlog/02-conditional-graph/2026-10-13-db1-db4-coordinated-nonfinite-loss.md).

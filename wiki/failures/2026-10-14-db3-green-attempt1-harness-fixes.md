# 2026-10-14 — DB-3 campaign harness/test defects (fixed)

Six harness or test defects were found and fixed during the DB-3 GREEN campaign
(ADR-0009). None is a production defect. Each item below states the symptom,
root cause, fix, and evidence path; all failing runs are retained and not
rewritten. The final gates are GREEN: gradient harness 19/19 (mpi2), 8/8
(mpi4), 17 PASS + 2 UNTESTED (nccl2); focused test exit 0 at mpi2, mpi4, and
nccl2; full suite 9/9 files at mpi2 and mpi4, 18/18 at nccl2 (MPI + NCCL);
single-process `train` 137/137.

Recorded per the repository rule to log failed commands.

## 1. Probe `run_grad_no_trainable` MethodError (first RED batch)

Symptom: the first RED probe batch could not run the `grad-no-trainable` case.
The v1 summary records `MethodError: no method matching
(::var"#run_grad_no_trainable##0#run_grad_no_trainable##1")(::EmptyModel,
::Int64)` on both ranks, with the closest candidate taking three arguments
(`conditional_graph_numerical_guard_probe.jl:1159`): the loss closure received
a 1-tuple `(i,)` while its signature takes `(m, i, x)`.

Root cause: the no-trainable call site passed the wrong argument shape.

Fix: call the closure with `data_range(1:3)`.

Evidence: probe md5 at the failure `ee6e6c264a77ef79cbd26ba43d131f3e`; the v1
run `artifacts/logs/conditional-wrapper/numerical-guards/red-deathstar-gradient-mpi2/`
is superseded and retained (`raw-grad-no-trainable-mpi2.log`). The final RED run
is `red-deathstar-gradient-mpi2-v2/` with the probe at
`ce16af4ab7932c5eb746f7b141e37f1c`.

## 2. Probe `CorruptBackend` injection window

Symptom: `grad-asym-vote` rejected during its setup updates instead of at the
poisoned update. The controlled backend injected the poisoned value on every
length-`C_LEN` buffer from the first update, so the pre-fault valid updates were
already invalid.

Root cause: the injection had no arming gate.

Fix: add an `arm::Ref{Bool}` gate that is set only before the poisoned update.

Evidence: superseded v1 run
`artifacts/logs/conditional-wrapper/numerical-guards/red-deathstar-gradient-mpi2/`
(`raw-grad-asym-vote-mpi2.log`); final
`red-deathstar-gradient-mpi2-v2/raw-grad-asym-vote-mpi2.log`.

## 3. Driver expected schedule for `grad-audit-zero`

Symptom: the driver compared the case against a wrong collective schedule.
Steps 1-2 are ordinary valid steps (tied buffer + `c` reduce); only step 3
reduces the tied buffer alone.

Root cause: the expected per-step schedule encoded the wrong collective sets.

Fix: corrected expected value to
`EAGER_STEP, EAGER_STEP, [:vote, :presence, :grad_a, :gradvote]`.

Evidence: superseded v1 run
`red-deathstar-gradient-mpi2/` (`raw-grad-audit-zero-mpi2-fault1.log`); final
driver md5 `9dba4ae76367e58f12467f180a42381f` (baseline
`49c7e0aacdb48661f044231b1fc6c5f1`); final RED run
`red-deathstar-gradient-mpi2-v2/`.

## 4. Driver fault rank for `grad-update-both` and `grad-caller`

Symptom: no rank injected the fault in those two cases, so they failed for the
wrong reason (missing stage metadata rather than rejection). Both cases inject
via the `FAULT_RANK` environment variable, but the driver table passed
`fault_rank -1`.

Root cause: wrong table entries.

Fix: corrected to fault rank 0.

Evidence: superseded v1 run
`red-deathstar-gradient-mpi2/` (`raw-grad-update-both-mpi2.log`,
`raw-grad-caller-mpi2.log`); final
`red-deathstar-gradient-mpi2-v2/raw-grad-update-both-mpi2.log` and
`raw-grad-caller-mpi2.log`.

## 5. Focused test healthy reference partial gradient NamedTuple

Symptom: the focused test's healthy-reference testset passed the partial
gradient NamedTuple `(a = ga,)` to native `Optimisers.update!` for a two-leaf
model; native Optimisers raised `FieldError` because the gradient NamedTuple
did not match the model leaves.

Root cause: missing second leaf entry in the test fixture.

Fix: pass `(a = ga, c = nothing)`.

Evidence: final focused test
`test/ext_distributed/nonfinite_gradient_distributedtest.jl` md5
`06eef21d92c78ca85f718b325ddbe8c6`; GREEN focused consoles
`green-deathstar-gradient-focused-{mpi2,mpi4,nccl2}/console.log`. No separate
failing log was retained for this local iteration.

## 6. `DeviceLikeArray` mock `all` behavior (suite attempt 1)

Symptom: the first full-suite mpi2 attempt passed 8 of 9 files and failed only
in `conditional_distributedtest.jl` (child exit 1). The mock iterated the
buffer elementwise for `all(isfinite, ...)`, tripping the device scalar-access
counter (8 accesses), and then hit a method ambiguity with Base's
`all(f::Function, a::AbstractArray; dims)`.

Root cause: the mock's array behavior did not mirror a device reduction.

Fix: add `Base.all(f::Function, x::DeviceLikeArray) = all(f, x.data)`, so the
mock performs one raw-storage reduction like a real device.

Evidence: the failed attempt is preserved as
`artifacts/logs/conditional-wrapper/numerical-guards/suite-deathstar-db3/mpi2-attempt1.log`;
the rerun `suite-deathstar-db3/mpi2.log` is 9/9 files.

## Impact

No production defect, no unintended Flux source change, no commit. Final gate
results and commands are in
[COMMANDS.md](../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#db-3-campaign-2026-10-14-deathstar);
the resolution is summarized in
[ADR-0009](../decisions/ADR-0009-post-reduction-nonfinite-gradient-rejection.md)
and the
[fix devlog](../devlog/02-conditional-graph/2026-10-14-db3-gradient-rejection.md).

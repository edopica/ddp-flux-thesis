# DB-3 post-reduction non-finite-gradient rejection

Date: 2026-10-14

Implemented and gate-accepted the post-reduction non-finite-gradient guard from
[ADR-0009](../../decisions/ADR-0009-post-reduction-nonfinite-gradient-rejection.md)
and the [DB-3 fix plan](../../../reports/conditional-graph-db3-fix-plan.md).
All gradient-path gates are GREEN on `deathstar`; no commit was made. The
`flux-integration` worktree is dirty on HEAD
`314aef2022333fa8f37035ed10cdc8e2360fa475`.

## Implementation

- `src/distributed/public_api.jl`: the guard lives in `_sync_gradients`, after
  the complete gradient-reduction loop and before `_build!`. The `np == 0`
  early return stays before the presence exchange, so a model without trainable
  parameters still issues zero collectives.
- `_reduced_buf_finite(buf::AbstractArray) = all(isfinite, buf)` scans each
  globally present canonical buffer after the reductions. The coordinated vote
  uses the existing `_any_rank_invalid` helper (one host `Float32` 0/1 invalid
  flag, `avg` allreduce, threshold `> 0`) and runs whenever `np > 0`, even when
  every gradient is globally absent.
- Rejection throws `DomainError(nothing, _NONFINITE_GRADIENT_ERROR)` on every
  rank. The message contains "non-finite synchronized gradient" and "No
  optimizer update occurred".
- `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl`: the NCCL extension overrides
  `_reduced_buf_finite(::CuArray)`. `NCCL.Allreduce!` returns after it enqueues
  the collective on the task-local CUDA stream, so the override calls
  `CUDA.synchronize()` and then runs the device reduction `all(isfinite, buf)`.
  No host transfer and no scalar indexing.
- Collective-count rule encoded by the updated test files: every nonempty
  update (eager or explicit) with `np > 0` gains exactly one trailing host
  `Float32` length-1 `avg` collective (the gradient vote); `np == 0` stays
  zero; a loss-rejected step keeps its single loss vote; gradient-rejected
  steps count as full steps.
- `test/ext_distributed/nonfinite_gradient_distributedtest.jl` (new, 848 lines,
  15 testsets) covers rejection, state preservation, control flow, and
  collective order on the device-generic paths. The existing conditional and
  CUDA conditional test files updated their exact collective expectations;
  `NEWS.md` records the default rejection and the extra metadata collective.

## RED (deathstar, pre-fix production)

- Gradient harness mpi2, `red-deathstar-gradient-mpi2-v2/`:
  `PASS=1 FAIL=18`. The single PASS is `grad-no-trainable`, a no-regression
  case that passes RED by design. The v1 run
  (`red-deathstar-gradient-mpi2/`) is superseded because its probe had the
  `run_grad_no_trainable` `MethodError`; it is retained.
- Focused test mpi2, `red-deathstar-focused-mpi2/console.log`: first testset
  `7 pass / 7 fail / 1 error`; the run aborts at the first failing testset.

## GREEN (deathstar)

| Gate | Result |
|---|---|
| Gradient harness mpi2 | 19/19 PASS |
| Gradient harness mpi4 | 8/8 PASS |
| Gradient harness nccl2 | 17 PASS + 2 UNTESTED (`grad-overflow`, `grad-tied-overflow`) |
| Focused test mpi2 / mpi4 / nccl2 | exit 0 (three runs) |
| Loss-group regression mpi2 | 13/13 PASS |
| Loss-group regression mpi4 | 8/8 PASS |
| Combined `--group all` mpi2 | 32/32 PASS |
| Combined `--group all` mpi4 | 16/16 PASS |
| Combined `--group all` nccl2 | 29 PASS + 3 UNTESTED (`dup-train-one`, `grad-overflow`, `grad-tied-overflow`) |
| Full suite mpi2 | 9/9 files (2m53.8s) |
| Full suite mpi4 | 9/9 files (3m09.9s) |
| Full suite nccl2 (MPI + NCCL passes) | 18/18 (7m37.9s) |
| Single-process `train` group | 137/137 PASS |
| `detect_ambiguities(Flux.DistributedUtils; recursive=false)` | 0 |
| Deadlock regression | 246/246 PASS |
| Phase K toy demo mpi 2,4 | 13/13 PASS |
| `Precompiling packages...` lines | 0 in every suite log |

Notes:

- The two NCCL UNTESTED cases are the native overflow fixtures. They are
  MPI-only, and the controlled injection covers the post-reduction device path;
  they are recorded UNTESTED, not PASS.
- The first full-suite mpi2 attempt failed only in
  `conditional_distributedtest.jl` (the `DeviceLikeArray` `all` ambiguity); the
  log is preserved as `suite-deathstar-db3/mpi2-attempt1.log`.
- The Phase K toy demo ran after the `scripts/examples/conditional_graph/common.jl`
  trace-check update for the extra vote.
- The combined `--group all` gates cover Phase D: an invalid loss stops before
  the gradient path, and a finite loss with invalid gradients stops at the
  gradient vote, with the healthy conditional reference retained.
- The nccl2 `--group all` run survived an SSH disconnect mid-run; its
  `summary.tsv` completed on disk and the result was verified afterwards.
- Every failing-then-fixed run is retained on disk; nothing was deleted or
  rewritten.

## Harness and test defects found and fixed

All six are harness or test defects; none is a production defect. Full record:
[failure note](../../failures/2026-10-14-db3-green-attempt1-harness-fixes.md).

1. Probe `run_grad_no_trainable` called the loss with a 1-tuple while the
   closure takes `(m, i, x)` -> `MethodError`; fixed by using
   `data_range(1:3)` (probe md5 at the failure
   `ee6e6c264a77ef79cbd26ba43d131f3e`).
2. Probe `CorruptBackend` injected on every length-`C_LEN` buffer from the
   first update, so `grad-asym-vote` rejected during its setup updates; fixed
   with an `arm::Ref{Bool}` gate set only before the poisoned update.
3. Driver expected schedule for `grad-audit-zero` was wrong: steps 1-2 are
   ordinary valid steps (tied buffer + `c` reduce), and only step 3 reduces the
   tied buffer alone; corrected to
   `EAGER_STEP, EAGER_STEP, [:vote, :presence, :grad_a, :gradvote]`.
4. Driver passed `fault_rank -1` for `grad-update-both` and `grad-caller`,
   which inject via `FAULT_RANK`, so no rank injected; corrected to fault
   rank 0.
5. The focused test's healthy-reference testset passed the partial gradient
   NamedTuple `(a = ga,)` to native `Optimisers.update!` for a two-leaf model
   -> `FieldError`; fixed to `(a = ga, c = nothing)`.
6. `conditional_distributedtest.jl` `DeviceLikeArray` mock iterated
   elementwise for `all(isfinite, ...)`, tripping the device scalar-access
   counter (8 accesses), and then hit a method ambiguity with Base's
   `all(f::Function, a::AbstractArray; dims)`; fixed with
   `Base.all(f::Function, x::DeviceLikeArray) = all(f, x.data)`, so the mock
   mirrors a device reduction. This was the only failed gate in suite attempt 1
   (`suite-deathstar-db3/mpi2-attempt1.log`).

## Source identity

Flux (`flux-integration` worktree, HEAD
`314aef2022333fa8f37035ed10cdc8e2360fa475`):

| File | md5 (final; baseline) |
|---|---|
| `src/distributed/public_api.jl` | `d21c73f8885b35ae8558a0f0d4e43196` (`e4ed1f8d4e2aed25d99654ad05d54919`) |
| `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl` | `841c662047cbe86cd082ceeb226b6231` (`330999cee29c78f7ecebe6e7e2610c13`) |
| `NEWS.md` | `83e69bb91cc5edd110c7de9af2b0b87a` |
| `test/ext_distributed/conditional_distributedtest.jl` | `54a3f8da8d677c35e5de3ed34130e34a` (`c26ef49b070bf2c446b827c64cc313c0`) |
| `test/ext_distributed/cuda_conditional_distributedtest.jl` | `c470cc3f3e4370b4d8b6404077684987` (`dabdaddb77301d93f83707c1c930bc6f`) |
| `test/ext_distributed/nonfinite_loss_distributedtest.jl` | `0e2680635752bc4f9ed168376458e470` (`b8d861fb0967d525eab6e0a7973b1bea`) |
| `test/ext_distributed/nonfinite_gradient_distributedtest.jl` | `06eef21d92c78ca85f718b325ddbe8c6` (new, 848 lines, 15 testsets) |

Thesis (HEAD `0ea21f79b20092ead8ca07f33f34f372a4ee3e70`):

| File | md5 (final; baseline) |
|---|---|
| `scripts/checks/conditional_graph_numerical_guard_check.jl` | `9dba4ae76367e58f12467f180a42381f` (`49c7e0aacdb48661f044231b1fc6c5f1`) |
| `scripts/checks/conditional_graph_numerical_guard_probe.jl` | `ce16af4ab7932c5eb746f7b141e37f1c` (`e5d3dd019e833e83d3c6707f9fbf7485`) |
| `scripts/examples/conditional_graph/common.jl` | `2ec29fb283cf45e3e33c684e454e2422` |
| `scripts/checks/conditional_deadlock_check.jl` | `467498a568ead33d0915d1b66a00d43f` |
| `scripts/checks/conditional_deadlock_probe.jl` | `86945681c0f8c57ec6da8d79c1fdf11c` |
| `notebook/phase_k_workflow_helpers.jl` | `cbca216033752b9fa2abf827257d5183` |

## Artifacts

All under `artifacts/logs/conditional-wrapper/numerical-guards/` unless noted:

- RED: `red-deathstar-gradient-mpi2-v2/`,
  `red-deathstar-gradient-mpi2/` (superseded v1, retained),
  `red-deathstar-focused-mpi2/console.log`.
- GREEN harness: `green-deathstar-gradient-mpi2/`,
  `green-deathstar-gradient-mpi4/`, `green-deathstar-gradient-nccl2/`;
  focused consoles
  `green-deathstar-gradient-focused-{mpi2,mpi4,nccl2}/console.log`.
- Loss regression: `green-deathstar-loss-db3-mpi2/`,
  `green-deathstar-loss-db3-mpi4/`.
- Suite: `suite-deathstar-db3/{mpi2,mpi4,nccl2,train}.log`; the failed first
  attempt `suite-deathstar-db3/mpi2-attempt1.log`.
- Other gates: `deadlock-check-db3.log`; Phase K toy demo
  `artifacts/logs/conditional-wrapper/demo/db3-toy-2026-10-14-mpi/`.
- Exact commands, file hashes, and the gate table:
  [COMMANDS.md, DB-3 campaign](../../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#db-3-campaign-2026-10-14-deathstar).

## Limitations

- The native overflow fixtures are MPI-only; under NCCL they are recorded
  UNTESTED, and the controlled injection covers the post-reduction device path.
- The contract is the synchronized-gradient boundary. Optimizer-internal
  overflow is not covered, and the added overhead remains unmeasured (plan
  section 8).
- A rejected step preserves parameters and optimizer state but does not roll
  back forward-pass side effects (ADR-0009, ADR-0008).

## Exact next action

Review the DB-3 working-tree diff and evidence, then decide on committing to
`flux-integration` and preparing the upstream contribution/review.

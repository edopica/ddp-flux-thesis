# ADR-0009 - Post-reduction rejection of non-finite synchronized gradients in distributed updates

Date: 2026-10-14
Status: Accepted (implemented 2026-10-14)

Related records:

- Audit 002, finding DB-3:
  [002-conditional-wrapper-deferred-behaviors.md](../audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md).
- Approved fix plan:
  [conditional-graph-db3-fix-plan.md](../../reports/conditional-graph-db3-fix-plan.md).
- Complementary decision, the DB-1/DB-4 loss vote:
  [ADR-0008-coordinated-nonfinite-loss-rejection.md](ADR-0008-coordinated-nonfinite-loss-rejection.md).
  ADR-0008 guards the scalar loss before any optimizer collective. This ADR
  guards the synchronized gradients after the reductions. The two decisions
  protect different boundaries and both apply to an eager training step.

Source references are relative to the `flux-integration` worktree.

## Context

Audit 002 section 4.3 records the DB-3 runtime evidence: a finite scalar loss can
produce a NaN or Inf gradient. A source-wide scan in the audit found no
finiteness check on gradients in `src/distributed/` or in the MPI and NCCL
extensions. Audit 002 records that the gradient buffers built by `_walk!` are
allreduced unconditionally. A NaN gradient from one rank therefore contaminates
the averaged gradient on every rank. The inner optimizer then writes NaN values
into the parameters and the Adam moments. No exception occurs, and the run
continues on non-finite parameters. The DB-3 row of the deferred-behavior table
records this consequence as silent corruption. The audit's backlog item A2
requests rank-consistent non-finite-gradient handling.

The same boundary has two more non-finite sources. First, the MPI
implementation sums the buffers before it divides by the worker count. Finite
inputs can therefore overflow during the sum, even when their mathematical
average is representable. Second, local tied-gradient accumulation can overflow
before any collective starts.

Plan section 5 requires the DB-3 ADR before the source edits. Plan section 7
(Phase A) starts with this record. This ADR precedes the production change.

## Decision

Add one coordinated gradient-validity vote inside the shared wrapper boundary
`_sync_gradients`, after the complete gradient-reduction loop and before
`_build!` (plan section 4.1). Both `Optimisers.update` and `Optimisers.update!`
already call this function before the inner optimizer.

1. Scan the `buf` field of every canonical `_SyncBuf` whose globally agreed
   `present[i]` is true, with a device-generic finiteness reduction:
   `all(isfinite, buf)`. An empty buffer is vacuously valid. Do not traverse the
   original gradient tree again. The canonical buffers already encode
   tied-gradient accumulation, copies into parameter-compatible storage, and
   zero contributions from locally absent gradients.
2. Combine the local results into one invalid flag. The encoding is `1f0` for
   invalid and `0f0` for valid. Run one host-metadata vote with the existing
   `_any_rank_invalid` helper (`src/distributed/public_api.jl:703-707`). The
   helper uses `DistributedUtils.allreduce!` with `DistributedUtils.avg`. The
   threshold is `> 0`, never `> 0.5`, because one invalid rank of four must
   reject the update.
3. When any rank is invalid, throw `DomainError` on every rank. The message
   states that a synchronized gradient is non-finite and that no optimizer
   update occurred. The message does not attribute a source rank, because a
   post-reduction check cannot know which rank produced the invalid value.

If buffers exist but every gradient is globally absent, the local flag is valid
and the vote still runs. This keeps one validation stage for every nonempty
synchronization plan.

## Implementation summary (2026-10-14)

The guard is implemented in the `flux-integration` worktree on HEAD
`314aef2022333fa8f37035ed10cdc8e2360fa475` (uncommitted).

- Both the scan and the vote live in `_sync_gradients`, after the complete
  gradient-reduction loop and before `_build!`. The `np == 0` early return stays
  before the presence exchange, so a model without trainable parameters still
  issues zero collectives.
- The per-buffer check is
  `_reduced_buf_finite(buf::AbstractArray) = all(isfinite, buf)` in
  `src/distributed/public_api.jl`. The scan is
  `invalid |= !_reduced_buf_finite(bufs[i].buf)` over the globally present
  canonical buffers. The vote runs whenever `np > 0`, even when every gradient
  is globally absent.
- The NCCL extension overrides `_reduced_buf_finite(::CuArray)`. Because
  `NCCL.Allreduce!` returns after it enqueues the collective on the task-local
  CUDA stream, the override calls `CUDA.synchronize()` and then runs the device
  reduction. The gradient buffer never leaves the device; only the scalar
  finiteness result (`Bool`) transfers to the host to drive the validity vote,
  and no scalar indexing is involved.
- Coordination uses the existing `_any_rank_invalid` helper: one host `Float32`
  0/1 flag through `DistributedUtils.allreduce!` with `DistributedUtils.avg`
  and the `> 0` threshold.
- Rejection throws `DomainError(nothing, _NONFINITE_GRADIENT_ERROR)` on every
  rank. The message contains "non-finite synchronized gradient" and "No
  optimizer update occurred".

## Collective schedule

Schedule from plan section 4.4. Let `P` be the number of globally present
canonical buffers.

| Entry path | Schedule for a nonempty buffer plan |
|---|---|
| Explicit update | presence → `P` gradient reductions → gradient vote |
| Eager training with the loss fix | loss vote → presence → `P` gradient reductions → gradient vote |
| Gradient rejection | the same complete schedule, then error on every rank |
| Globally absent gradients | presence → gradient vote. All inner leaves keep skip semantics |
| No trainable buffers | zero collectives. The existing early return is unchanged |

The gradient vote does not depend on the number or the content of the present
buffers. No rank-local error or early return precedes it on any nonempty plan.
The vote runs for every nonempty plan (`np > 0`), including the case where all
gradients are globally absent; `np == 0` keeps the existing early return and
zero collectives. The collective-count expectations in the updated test files
encode this rule: every nonempty update (eager or explicit) gains exactly one
trailing host `Float32` length-1 `avg` collective, a loss-rejected step keeps
its single loss vote, and gradient-rejected steps count as full steps.

## Why the check follows reduction

The inner optimizer consumes the synchronized values, not the original local
gradients. The finiteness scan covers three mechanisms at once: NaN propagation
through the average, local tied-gradient accumulation, and non-finite results
that the reduction itself generates. A pre-reduction check alone cannot catch
finite-input sum overflow in the MPI averaging path. The contract concerns the
final synchronized values, not a complete audit of every input or intermediate
operation.

## No early throw

Every rank completes the identical collective schedule and throws only after
the schedule ends. A rank that throws after its finiteness scan, but before the
vote, leaves its peers in an unmatched collective. The post-reduction position
of the throw keeps every collective matched, including on rejected updates.
Eager paths also run the loss vote from ADR-0008 before the presence exchange,
so a non-finite loss never reaches this vote.

## Preservation guarantee

Rejection occurs before the inner optimizer call. Parameters, existing
optimizer state, and caller-owned gradients keep their pre-update values. The
fresh communication buffers can contain non-finite values, and the wrapper
discards them. Both `update` and `update!` satisfy this guarantee.

The guarantee has limits. Rejection does not roll back forward-pass side
effects, such as BatchNorm statistics, RNG advancement, or user mutations. It
does not protect against overflow inside the optimizer rule itself. It does not
repair pre-existing invalid parameters, optimizer state, or rank-local
configuration.

## Device rules (CUDA and NCCL)

- The finiteness reduction runs on the device. Only the scalar validity Bool
  transfers to the host.
- The implementation must not use `Array(buffer)`, `collect(buffer)`, host
  iteration, or scalar indexing.
- The NCCL extension makes the ordering explicit with a CUDA stream
  synchronization before the device finiteness reduction. `NCCL.Allreduce!`
  returns after it enqueues the collective on the task-local CUDA stream. The
  Julia method return alone does not prove that the device writes completed.
- Device-specific code stays in `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl`. Core
  Flux does not import CUDA.
- `CUDA.allowscalar(false)` stays enabled in the focused tests.

## Scope and boundaries

The initial policy is mandatory rejection. No public opt-out and no recovery
policy enter this change. Plan section 8 can consider a separate decision if the
measured cost is unacceptable.

The guard covers every entry point that calls `_sync_gradients`:

- `Optimisers.update!` and `Optimisers.update` with a `DistributedOptimizerState`.
- `Flux.update!` where it dispatches to the same wrapper methods.
- Explicit `withgradient` plus either wrapper update method.
- Eager `train!`, `trainstep!`, and `trainstep_withgradient!` through those methods.

DB-3 does not correct a rank that skips the wrapper because of a non-finite
loss. That case belongs to DB-1 and DB-4, and ADR-0008 handles it. A rank that
never enters `_sync_gradients` cannot participate in the gradient vote.

## Compatibility contract

- Frozen present leaves keep the existing participation rule. The vote includes
  them, so an invalid frozen gradient rejects the update.
- Globally absent leaves still rebuild as `ZeroTangent` and keep the Adam skip
  semantics.
- Mutable tied parameters keep one canonical buffer.
- Isbits parameters keep the existing per-occurrence buffers.
- Models with no trainable parameters keep zero collectives.
- Empty numeric buffers are valid and cannot cause a spurious rejection.

Plan section 4.1 defines these points.

## Consequences

- The implementation adds one finiteness scan and one scalar host collective to
  every nonempty wrapper update.
- A CUDA finiteness scan adds kernel launches and one device-to-host
  synchronization per buffer.
- The overhead is unmeasured. Plan section 8 defines the post-correctness
  comparison and forbids a performance threshold in this plan.
- Exact collective counts in the dedicated suite, the harness, and the
  demonstration parsers must increase by one per nonempty update. The order
  checks stay.
- Error semantics change for distributed update paths. A non-finite
  synchronized gradient now raises `DomainError` instead of a silent update.
  The plan also requires a NEWS entry (plan section 6).

## Alternatives considered

1. Pre-reduction-only finiteness check: rejected. A pre-reduction check cannot
   catch finite inputs that overflow during the sum, and the contract concerns
   the final synchronized values.
2. Per-parameter votes: rejected. They add one collective per parameter and
   place the decision at the wrong boundary. Gradient validity is a property of
   the synchronized update.
3. A local throw after the finiteness scan: rejected. The throwing rank exits
   while its peers wait in the gradient vote. The collectives then
   desynchronize.
4. Host copies of CUDA buffers: rejected. Full-buffer device-to-host copies
   break the device rules and add transfer cost.
5. A public opt-out in the initial policy: rejected. The initial policy is
   mandatory. If the measured cost is unacceptable, a separate decision can
   consider batched device reductions or an explicit opt-out (plan section 8).
6. Repair or rollback of the rejected step: rejected. No recovery policy enters
   this change. The wrapper rejects the update and leaves recovery to the
   caller.

## Review response (2026-10-14)

The [DB-3 review](../../reports/conditional-graph-db3-review.md) requested two P2
corrections and one P3 documentation correction. All three are implemented and
gate-verified on deathstar; production guard code did not change. The corrected
hashes below supersede the campaign hashes recorded above; the campaign tables
remain the DB-3 campaign record. Response devlog:
[2026-10-14-db3-review-response.md](../devlog/02-conditional-graph/2026-10-14-db3-review-response.md).

- R1 (thesis): `notebook/phase_k_workflow_helpers.jl` pins the collective
  protocol per bundle (`legacy-5/7` base 3 no vote, `guarded-6/8` base 4 with
  the trailing vote); a missing or unknown pin throws. The new
  `scripts/checks/phase_k_collective_protocol_check.jl` passed on deathstar:
  both tutorials, the guarded DB-3 toy bundle, and three negative controls
  (`PHASE_K_PROTOCOL_SUMMARY PASS`, exit 0).
- R2 (test only): `test/ext_distributed/nonfinite_gradient_distributedtest.jl`
  uses a nonzero `a` target and the correct step-3 weighting, and asserts
  nonzero `a` moments before the absence step. GREEN: healthy testset 59/59 per
  rank at mpi2, mpi4, and nccl2 (exit 0). RED control with the old weighting:
  53 pass / 6 fail / 59 per rank, exit 1.
- R3 (docs only): the wrapper docstrings, `NEWS.md`, the NCCL extension comment,
  and this ADR's implementation summary now state the all-rank vote,
  coordinated `DomainError`, preservation guarantee, nonempty-plan rule, and
  scalar device-to-host transfer consistently. No executable line changed
  (verified by reverting the prose edits and reproducing the campaign md5s);
  the synced sources precompiled successfully on deathstar.

Corrected hashes (old → new; `flux-integration` worktree, still dirty, no
commit):

| File | Old | New |
|---|---|---|
| `src/distributed/public_api.jl` | `d21c73f8885b35ae8558a0f0d4e43196` | `48ea1fcd3c0629d1773b91ae9d6f1677` |
| `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl` | `841c662047cbe86cd082ceeb226b6231` | `3bca62aa63f2b1d18e2f7f4eb630d221` |
| `NEWS.md` | `83e69bb91cc5edd110c7de9af2b0b87a` | `f9cd8bf52caf18a99d0b2b76cb13ee2b` |
| `test/ext_distributed/nonfinite_gradient_distributedtest.jl` | `06eef21d92c78ca85f718b325ddbe8c6` | `b0566261df12deb00cd69fc6d99220e9` (862 lines) |

The first R1 run failed on an incomplete remote 4-rank tutorial bundle and a
one-dimensional `(1,)` shape capture; both fixes and the transcribed console
are in the
[failure note](../failures/2026-10-14-db3-review-response-r1-shape-and-artifacts.md).

## Evidence and links

Design records:

- Audit 002:
  [DB-3 observation](../audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md),
  section 4.3, now with the 2026-10-14 resolution note.
- DB-3 fix plan:
  [conditional-graph-db3-fix-plan.md](../../reports/conditional-graph-db3-fix-plan.md),
  sections 1-4 and 9.
- Complementary decision:
  [ADR-0008-coordinated-nonfinite-loss-rejection.md](ADR-0008-coordinated-nonfinite-loss-rejection.md).
- Fix devlog:
  [2026-10-14-db3-gradient-rejection.md](../devlog/02-conditional-graph/2026-10-14-db3-gradient-rejection.md).
- Campaign commands, identities, and the full gate table:
  [COMMANDS.md, DB-3 campaign (2026-10-14, deathstar)](../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#db-3-campaign-2026-10-14-deathstar).
- Harness/test defect record:
  [2026-10-14-db3-green-attempt1-harness-fixes.md](../failures/2026-10-14-db3-green-attempt1-harness-fixes.md).
- Source anchors: `src/distributed/public_api.jl:735-792` (`_sync_gradients`;
  scan and vote at `:776-782`), `src/distributed/public_api.jl:703-707`
  (`_any_rank_invalid`), and
  `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl:92-95` (`_reduced_buf_finite(::CuArray)`).
  Line numbers refer to the inspected source and can change.

Source identity (final `flux-integration` worktree, HEAD
`314aef2022333fa8f37035ed10cdc8e2360fa475`):

| File | md5 (final; baseline) |
|---|---|
| `src/distributed/public_api.jl` | `d21c73f8885b35ae8558a0f0d4e43196` (`e4ed1f8d4e2aed25d99654ad05d54919`) |
| `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl` | `841c662047cbe86cd082ceeb226b6231` (`330999cee29c78f7ecebe6e7e2610c13`) |
| `NEWS.md` | `83e69bb91cc5edd110c7de9af2b0b87a` |
| `test/ext_distributed/nonfinite_gradient_distributedtest.jl` | `06eef21d92c78ca85f718b325ddbe8c6` (new, 848 lines, 15 testsets) |
| `test/ext_distributed/conditional_distributedtest.jl` | `54a3f8da8d677c35e5de3ed34130e34a` (`c26ef49b070bf2c446b827c64cc313c0`) |
| `test/ext_distributed/cuda_conditional_distributedtest.jl` | `c470cc3f3e4370b4d8b6404077684987` (`dabdaddb77301d93f83707c1c930bc6f`) |
| `test/ext_distributed/nonfinite_loss_distributedtest.jl` | `0e2680635752bc4f9ed168376458e470` (`b8d861fb0967d525eab6e0a7973b1bea`) |

Acceptance evidence: all runs are on `deathstar`; artifacts are under
`artifacts/logs/conditional-wrapper/numerical-guards/` unless noted.

RED (pre-fix production):

| Gate | Result | Artifact |
|---|---|---|
| Gradient harness mpi2 | 18 FAIL + 1 PASS (`grad-no-trainable` is a no-regression case and passes RED by design) | `red-deathstar-gradient-mpi2-v2/` |
| Focused test mpi2 | first testset 7 pass / 7 fail / 1 error; the run aborts at the first failing testset | `red-deathstar-focused-mpi2/console.log` |

The v1 harness run (`red-deathstar-gradient-mpi2/`) is superseded because its
probe had the `run_grad_no_trainable` `MethodError`; it is retained.

GREEN:

| Gate | Result | Artifact |
|---|---|---|
| Gradient harness mpi2 | 19/19 PASS | `green-deathstar-gradient-mpi2/` |
| Gradient harness mpi4 | 8/8 PASS | `green-deathstar-gradient-mpi4/` |
| Gradient harness nccl2 | 17 PASS + 2 UNTESTED (`grad-overflow`, `grad-tied-overflow`; the native overflow fixtures are MPI-only and the controlled injection covers the device path) | `green-deathstar-gradient-nccl2/` |
| Focused test mpi2 / mpi4 / nccl2 | exit 0 | `green-deathstar-gradient-focused-{mpi2,mpi4,nccl2}/console.log` |
| Loss-group regression mpi2 / mpi4 | 13/13 / 8/8 PASS | `green-deathstar-loss-db3-mpi2/`, `green-deathstar-loss-db3-mpi4/` |
| Combined `--group all` mpi2 / mpi4 | 32/32 / 16/16 PASS | `green-deathstar-all-mpi2/`, `green-deathstar-all-mpi4/` |
| Combined `--group all` nccl2 | 29 PASS + 3 UNTESTED (`dup-train-one`, `grad-overflow`, `grad-tied-overflow`) | `green-deathstar-all-nccl2/` |
| Dedicated suite mpi2 / mpi4 | 9/9 files PASS (2m53.8s / 3m09.9s) | `suite-deathstar-db3/mpi2.log`, `suite-deathstar-db3/mpi4.log` |
| Dedicated suite nccl2 (MPI + NCCL passes) | 18/18 PASS (7m37.9s) | `suite-deathstar-db3/nccl2.log` |
| Single-process `train` group | 137/137 PASS | `suite-deathstar-db3/train.log` |
| `detect_ambiguities(Flux.DistributedUtils; recursive=false)` | 0 | COMMANDS.md |
| Deadlock regression | 246/246 PASS | `deadlock-check-db3.log` |
| Phase K toy demo mpi 2,4 | 13/13 PASS | `artifacts/logs/conditional-wrapper/demo/db3-toy-2026-10-14-mpi/` |
| `Precompiling packages...` lines | 0 in every suite log | COMMANDS.md |

The first full-suite mpi2 attempt failed in `conditional_distributedtest.jl`
(mock `all` ambiguity) and is preserved as
`suite-deathstar-db3/mpi2-attempt1.log`.

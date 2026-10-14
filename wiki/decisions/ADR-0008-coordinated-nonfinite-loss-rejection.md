# ADR-0008 - Coordinated non-finite-loss rejection in eager distributed training

Date: 2026-10-13
Status: Accepted

Related records:

- Audit 002, findings DB-1 and DB-4:
  [002-conditional-wrapper-deferred-behaviors.md](../audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md).
- Approved fix plan:
  [conditional-graph-db1-db4-fix-plan.md](../../reports/conditional-graph-db1-db4-fix-plan.md).
- Separate gradient-guard plan:
  [conditional-graph-db3-fix-plan.md](../../reports/conditional-graph-db3-fix-plan.md).

Source references are relative to the `flux-integration` worktree.

## Context

DB-1: one rank's non-finite scalar loss makes `Flux.train!` hang. The offending rank
skips its optimizer update and throws `DomainError` afterwards. Every other rank
blocks in the distributed update's presence collective with no timeout.

DB-4: a manual `trainstep!` loop with one skipped update silently shifts gradient
pairing and then hangs. `trainstep!` does not throw on a non-finite loss, so the
skipping rank continues one step ahead. Later collectives pair one rank's step *k*
with another rank's step *k+1*.

Root cause: `_eager_step!` makes a rank-local decision before any collective
(`src/train.jl:375-379`):

```julia
v, gs = Flux.withgradient(m -> loss(m, batch...), adtype, model)
isfinite(_loss_scalar(v)) && _update!(opt_state, model, gs[1])
return v, gs[1]
```

The offending rank never calls `_update!`, so it never enters `_sync_gradients`
(`src/distributed/public_api.jl:698-737`) and issues no collective for that step.
A guard inside `Optimisers.update!` alone cannot fix the defect because the
offending rank never reaches the update.

Plan section 1 records the same mechanism. Audit 002 sections 4.1 and 4.4 contain
the runtime-confirmed hang and shifted-pairing evidence. Plan section 5 requires
this ADR before the source edits.

## Decision

Add one coordinated loss-validity vote to every eager distributed training step
(plan sections 3.1 and 3.2).

1. Define a private hook in `Flux.Train`:

   ```julia
   _loss_update_allowed(opt_state, scalar_loss) = isfinite(scalar_loss)
   ```

   The default preserves the existing single-process skip behavior. Call the hook
   unconditionally after `withgradient` and before `_update!` in `_eager_step!`.

2. Extend the hook for `DistributedOptimizerState` in `src/distributed/public_api.jl`:
   - Encode local validity as a one-element host `Float32` buffer: invalid is `1f0`,
     valid is `0f0`.
   - Reduce the flag with `DistributedUtils.allreduce!` and `DistributedUtils.avg`.
   - Reject the step when the reduced value is positive. The threshold is `> 0`,
     never `> 0.5`; one invalid rank out of four must reject the step.
   - On rejection, throw `DomainError` on every rank before the presence exchange,
     the gradient reductions, or the inner optimizer update.

The error message states that at least one rank reported a non-finite scalar loss
and that no optimizer update occurred. It must not claim that the local loss was
invalid on every rank.

## Collective schedule

Schedule from plan section 3.3. Let `P` be the number of globally present canonical
gradient buffers.

| Case | Required schedule |
|---|---|
| Finite loss, before DB-3 | loss vote → presence → `P` gradient reductions |
| Finite loss, with DB-3 | loss vote → presence → `P` gradient reductions → gradient vote |
| Any non-finite loss | loss vote → error on every rank |
| No trainable buffers, finite loss | loss vote → existing empty-update behavior |
| No trainable buffers, invalid loss | loss vote → error on every rank |

The loss vote is always first, and it does not depend on the number of trainable
buffers. No rank-local early return or error may precede the vote for a returned,
validly typed scalar loss.

## Why local skipping and a wrapper-only guard are insufficient

- Local skipping: the offending rank never enters the update wrapper, so it issues
  no collective for the step. Other ranks block in the presence `allreduce!`
  (DB-1), or a manual loop pairs their gradients with the skipping rank's later
  iteration (DB-4).
- Wrapper-only guard: a check inside `Optimisers.update!` or `_sync_gradients` runs
  only on ranks that reach the update. The offending rank does not reach it, so the
  check cannot produce an all-rank decision. A guard at the update boundary always
  arrives after the participation mismatch exists.

The hook in `_eager_step!` is the first shared point where every rank can agree
before any optimizer collective.

## Compatibility contract

- Ordinary, non-distributed optimizer states keep the local skip behavior;
  `_loss_update_allowed` returns `isfinite(scalar_loss)` by default.
- Distributed `trainstep!` now raises instead of returning after a rejected loss.
- Distributed `trainstep_withgradient!` returns neither the loss nor the gradient
  on rejection.
- Finite objectives keep their existing return values, including tuple and
  named-tuple auxiliary outputs.
- A non-finite auxiliary value alone does not reject a step; the vote looks only at
  the scalar loss.
- An explicit `withgradient` call followed by `update!` has no scalar-loss argument
  and stays out of scope.

Plan section 2 defines these points as the compatibility contract.

## State-preservation boundary

- A rejection performs no optimizer update. Parameters and optimizer state keep
  their pre-update values.
- Forward-pass side effects, such as BatchNorm statistics, RNG advancement, or user
  mutations, are not rolled back.
- Tests use pure losses and compare parameters and optimizer state against
  snapshots from before the rejected step.
- Documentation must not promise that the entire model is unchanged.

Plan section 2 defines this boundary.

## Preconditions and exclusions

- All ranks must enter the same API step and use consistent model, data, and
  configuration structure.
- The fix provides no recovery from process failure, a thrown forward or backward
  exception, or rank-local early termination.
- The Reactant compiled path stays outside this eager-path contract.
- No automatic retry, silent skip, communicator abort, or recovery policy enters
  this change.
- DB-3 (non-finite gradients) is a separate decision, covered by
  `reports/conditional-graph-db3-fix-plan.md`; this ADR does not include it.

## Consequences

- One extra host-metadata collective (the loss vote) occurs on every eager
  distributed training step.
- Exact collective counts in the dedicated distributed suite increase by the vote;
  tests must update those expectations without weakening order checks.
- Error semantics change for the distributed eager APIs (`train!`, `trainstep!`,
  `trainstep_withgradient!`); NEWS documents the change.
- Historical deferred-probe artifacts remain valid for the pre-fix source. They
  characterize the old behavior and are not rewritten.

## Alternatives considered

1. Keep the local skip and add a guard only in the wrapper: rejected; the offending
   rank never enters the wrapper, so the mismatch and the deadlock remain.
2. Throw locally before the vote: rejected; the offending rank would exit while
   peers block in a collective that will never complete.
3. Use a majority or `> 0.5` threshold: rejected; it is wrong for more than two
   ranks. One invalid rank out of four must reject the step.
4. Traverse the gradient tree and vote per parameter: rejected; it adds one
   collective per parameter and places the decision at the wrong boundary. Loss
   validity is a per-step property.
5. Abort the communicator on rejection: rejected; no recovery policy is in scope,
   and an abort destroys the job instead of reporting the condition.

## Evidence and links

- Audit 002:
  [DB-1 and DB-4 observations](../audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md),
  sections 4.1 and 4.4.
- DB-1/DB-4 fix plan:
  [conditional-graph-db1-db4-fix-plan.md](../../reports/conditional-graph-db1-db4-fix-plan.md),
  sections 1, 2, 3.1-3.3, and 5.
- Historical RED probes in `scripts/checks/conditional_graph_deferred_probe.jl`,
  modes `nan-loss-hang`, `nan-loss-all`, and `step-desync`.
- Acceptance evidence will come from the numerical-guard harness:
  `scripts/checks/conditional_graph_numerical_guard_probe.jl` and
  `scripts/checks/conditional_graph_numerical_guard_check.jl` (plan section 4.1).
- Source anchors: `src/train.jl:375-379` (`_eager_step!`),
  `src/distributed/public_api.jl:698-737` (`_sync_gradients`), and
  `src/distributed/public_api.jl:461-472` (inner optimizer update).

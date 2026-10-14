# DB-1 and DB-4: coordinated non-finite-loss rejection

Date: 2026-10-13
Status: proposed implementation plan. No implementation or runtime validation occurred during planning.

## 1. Goal and failure mechanism

Prevent a non-finite scalar loss on one rank from causing deadlock or cross-step gradient pairing.
The fix covers the eager distributed paths of `Flux.train!`, `Flux.trainstep!`, and `Flux.trainstep_withgradient!`.

The current `_eager_step!` differentiates the loss and then makes a local decision:

```julia
isfinite(_loss_scalar(v)) && _update!(opt_state, model, gs[1])
```

A rank with a non-finite loss skips every update collective.
Other ranks enter the wrapper and wait, or pair their gradients with a later iteration.
A guard inside `Optimisers.update!` alone cannot correct this failure because the offending rank never calls it.

Evidence:

- [Audit 002, DB-1/DB-2/DB-4](../wiki/audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md).
- `../flux-integration/src/train.jl:220-245,334-378,425-429`.
- `../flux-integration/src/distributed/public_api.jl:461-472,698-737`.
- `scripts/checks/conditional_graph_deferred_probe.jl` modes `nan-loss-hang`, `nan-loss-all`, and `step-desync`.

Line references describe the inspected source and can change during implementation.

## 2. Required behavior

On every eager distributed training step:

1. Compute the loss and gradient through the existing AD path.
2. Extract the scalar objective with `_loss_scalar`.
3. Participate in one all-rank loss-validity vote, even when the local objective is non-finite.
4. If any rank reports an invalid objective, raise `DomainError` on every rank.
5. Otherwise, call the existing `_update!` path.

The error occurs before presence exchange, gradient reduction, or an inner optimizer update for the rejected step.
Finite ranks must also raise the error. They must not return successfully and continue to the next batch.

The diagnostic must state that at least one rank reported a non-finite scalar loss and that no optimizer update occurred.
It must not claim that the local loss was invalid on every rank.
Identifying the offending rank or the dataset index is optional and requires no extra collective in the initial fix.

### Compatibility contract

- Ordinary, non-distributed optimizer states retain their existing local skip behavior.
- Distributed `trainstep!` changes behavior: it raises an error instead of returning after a rejected loss.
- Distributed `trainstep_withgradient!` follows the same error policy and returns neither result on rejection.
- Finite scalar objectives retain their existing return values, including tuple and named-tuple auxiliary outputs.
- Non-finite auxiliary values alone do not reject a step.
- Explicit `withgradient` plus `update!` has no scalar-loss argument. This fix cannot inspect that loss.
- The gradient safeguard is separate: [DB-3 plan](conditional-graph-db3-fix-plan.md).

### State-preservation boundary

No optimizer update occurs on rejection. Trainable parameters and optimizer state retain their pre-update values.
Tests use pure losses and compare these values with snapshots from before the rejected step.
Forward-pass side effects, such as BatchNorm statistics, RNG advancement, or user mutations, are not rolled back.
Documentation must avoid an unconditional promise that the entire model is unchanged.

### Preconditions and exclusions

All ranks must enter the same training API step and use consistent structure and configuration.
This fix does not recover from process failure, a thrown forward/backward exception, or rank-local early termination.
Reactant's compiled path is outside this eager-path contract.
Keep existing unsupported-path behavior and state the scope explicitly.
No automatic retry, silent skip, communicator abort, or recovery policy enters this change.

## 3. Implementation design

### 3.1 Add a dispatch hook before the local branch

`src/Flux.jl` includes `train.jl` before `distributed/public_api.jl`.
The training module cannot reference `DistributedOptimizerState` in an early method signature.

Define a private generic hook in `Flux.Train`, with the optimizer state as its first argument.
For example, `_loss_update_allowed(opt_state, scalar_loss)` returns the existing local `isfinite` result by default.
Call that hook unconditionally after `withgradient` and before `_update!`.

Conceptual shape, not a patch:

```julia
v, gs = Flux.withgradient(m -> loss(m, batch...), adtype, model)
_loss_update_allowed(opt_state, _loss_scalar(v)) &&
    _update!(opt_state, model, gs[1])
return v, gs[1]
```

Define the distributed specialization after `DistributedOptimizerState` exists, in `distributed/public_api.jl`.
Use a qualified method definition or explicit import consistent with Julia's extension rules.
The specialization returns `true` after a successful global vote, or throws after a failed global vote.

This hook preserves the shared AD call, `_update!` dispatch, and `Duplicated` model unwrapping.
Do not duplicate `_eager_step!`, reimplement the optimizer update, or add separate guards to each public training function.

### 3.2 Use one host-metadata vote

Encode the local invalid flag as a one-element host `Float32` buffer: invalid is `1f0`, valid is `0f0`.
Reduce it through `DistributedUtils.allreduce!` with `DistributedUtils.avg`.
Any positive result means that at least one rank rejected the objective.
Do not use a `> 0.5` threshold: one invalid rank out of four must reject the step.

This protocol reuses the backend support already used for presence metadata.
For NCCL, host metadata uses the backend's MPI fallback.
The scalar-loss vote introduces no full-gradient or parameter transfer.

Use a small private invalid-flag helper if the DB-3 implementation shares this protocol.
Keep the helper limited to metadata agreement. The loss and gradient call sites retain distinct messages and policy boundaries.
When implemented independently, either fix can introduce the helper first.

### 3.3 Collective schedule

Let `P` be the number of globally present canonical gradient buffers.

| Case | Required schedule |
|---|---|
| Finite loss, before DB-3 | loss vote → presence → `P` gradient reductions |
| Finite loss, with DB-3 | loss vote → presence → `P` gradient reductions → gradient vote |
| Any non-finite loss | loss vote → error on every rank |
| No trainable buffers, finite loss | loss vote → existing empty-update behavior |
| No trainable buffers, invalid loss | loss vote → error on every rank |

The loss vote must not depend on the number of trainable buffers.
No rank-local early return or error can precede that vote for a returned, validly typed scalar loss.

## 4. RED-first tests

### 4.1 Build bounded success regressions

The historical deferred driver treats demonstrated hangs and corruption as successful characterization.
Preserve that driver and its saved artifacts as historical evidence.
Create a separate success-regression harness with these proposed files:

- `scripts/checks/conditional_graph_numerical_guard_probe.jl`.
- `scripts/checks/conditional_graph_numerical_guard_check.jl`.

Both plans share this harness. Its interface must support `--group loss|gradient|all`, `--ranks`, `--backend`, and `--outdir`.
The parent driver launches MPI children, records the exact child command, and applies separate startup and execution watchdogs.
A timeout always fails a success regression.
Each rank must print completion, rejection status, snapshot status, and collective trace markers.
Only rank 0 writes shared result files. The parent can capture interleaved child output.

Avoid the old logger's `Float32 => presence` classification.
Loss metadata, presence metadata, and Float32 gradients can share a dtype and length.
Classify expected stages by the scenario's explicit protocol schedule, and compare raw collective ordinals, shapes, and operations.
Do not read a GPU buffer with `first(buf)` for logging.

Run the new tests against the unmodified source and retain the expected failures.
The asymmetric-loss cases must fail because of missing completion or wrong participation, not a startup failure.

### 4.2 CPU acceptance matrix

| Case | Required assertion |
|---|---|
| One rank returns NaN through `train!` | Every rank catches `DomainError`. The rejected step has only the loss vote |
| One rank returns positive or negative Inf | Same rejection contract |
| Every rank returns NaN | DB-2 remains a clean stop, now with a matched loss vote |
| Manual `trainstep!` loop | Every rank stops at the injected step. No later batch executes |
| `trainstep_withgradient!` | Same rejection and no returned gradient result |
| Fault on rank 0 and on a non-root rank | Root identity does not change the policy |
| One invalid rank out of four | Any-positive vote rejects. No majority rule applies |
| Pure loss after two valid Adam updates | Parameters, moments, scalar counters, and tied identities match snapshots |
| Finite scalar with non-finite auxiliary output | Step succeeds and return structure remains intact |
| Tuple/named-tuple with invalid first element | Step rejects on all ranks |
| Finite conditional training | Per-step parameters and Adam state match the native reference |
| Model without trainable parameters | Invalid loss still raises on every rank |
| Non-distributed optimizer state | Existing skip/return and `train!` error semantics remain intact |

Include the supported `Duplicated` eager route if its dependencies are available in the test environment.
Record an unavailable route as untested rather than silently excluding it from a claim.

Use an optional coordinated recovery subcase: every rank catches the error and performs one known-good step.
That step must match a reference that omits the rejected update.
This proves that the rejected step leaves no unmatched collective. It does not add an automatic recovery API.

## 5. Production files and documentation

Expected Flux changes:

- `src/train.jl`: generic hook, shared call site, comments, and public non-finite-loss documentation.
- `src/distributed/public_api.jl`: distributed specialization and private vote helper.
- `test/train.jl`: single-process compatibility coverage where existing coverage is insufficient.
- A focused `test/ext_distributed/nonfinite_loss_distributedtest.jl`: supported MPI/NCCL regressions.
- Existing conditional tests: update exact collective expectations without weakening order or value assertions.
- `NEWS.md`: distributed eager APIs now raise collectively on a non-finite scalar loss.

Before source edits, write a focused ADR that records DB-1/DB-4, the hook boundary, and the coordinated-error policy.
The ADR must explain why local skipping and a wrapper-only fix are insufficient.
Update audit 002 with a dated resolution and evidence links after acceptance, preserving the original observations.
Correct the workaround table: `train!` alone is not a workaround, and inspecting a returned `trainstep!` loss is too late.

## 6. Execution sequence and commands

### Phase A: baseline and failure evidence

1. Inspect the Flux worktree status and existing diff before editing.
2. Record HEAD, modified source hashes, Julia/package versions, and the launcher backend.
3. Write the ADR and bounded success-regression harness.
4. Run the CPU RED cases against the current source.

### Phase B: implementation and CPU gates

1. Add the generic hook and distributed specialization.
2. Run the focused two-rank loss cases.
3. Run the four-rank cases, including one invalid non-root rank.
4. Run single-process training regressions and method-ambiguity checks.
5. Run the dedicated distributed suite at two and four CPU ranks.

Planned harness commands, from the thesis root, after the new files exist:

```bash
julia --startup-file=no --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl --group loss --ranks 2 --backend mpi --outdir artifacts/logs/conditional-wrapper/numerical-guards/loss-mpi2
julia --startup-file=no --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl --group loss --ranks 4 --backend mpi --outdir artifacts/logs/conditional-wrapper/numerical-guards/loss-mpi4
```

Existing full-suite entry point, from the thesis root:

```bash
JULIA_MPI_TEST_NPROCS=2 FLUX_TEST_DISTRIBUTED_NCCL=false julia --startup-file=no --project=../flux-integration/test ../flux-integration/test/ext_distributed/runtests.jl
JULIA_MPI_TEST_NPROCS=4 FLUX_TEST_DISTRIBUTED_NCCL=false julia --startup-file=no --project=../flux-integration/test ../flux-integration/test/ext_distributed/runtests.jl
```

These are planned commands, not recorded successful runs.
Use fresh artifact directories for repeats and record any timeout overrides with their reasons.

### Phase C: native GPU gate

After CPU gates pass, run focused two-rank native CUDA/NCCL cases with `CUDA.allowscalar(false)`.
Include one-rank NaN loss, all-rank NaN loss, manual-loop rejection, and a healthy conditional reference case.
Load the remote skill and matching server page before any remote execution.
Resolve the exact remote command from that workflow and record it with the artifacts.
Use compute-node precompilation when the server requires it.

### Phase D: combined acceptance

Run `--group all` after both fixes exist.
A bad loss with bad gradients must fail at the loss vote, before any gradient collective.
A finite loss with bad gradients must reach the DB-3 guard.

## 7. Completion criteria

- Every required CPU case passes at its stated rank count, with no watchdog timeout.
- The focused native GPU cases pass before the fix claims CUDA/NCCL support.
- Every rejected step has matched collective traces and no optimizer-state mutation.
- Healthy distributed trajectories and single-process behavior remain correct.
- Documentation states the eager-path scope and forward-side-effect boundary.
- ADR, commands, versions, hashes, raw logs, and failures are recorded.
- Audit and context updates distinguish completed evidence from remaining coverage.

**Exact first implementation action:** record the current Flux diff, then write the ADR and the two-rank asymmetric-loss success regression.

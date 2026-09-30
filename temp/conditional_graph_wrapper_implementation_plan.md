# Conditional-Graph Wrapper Implementation Plan

Phase E review correction (2026-09-30): **both P2 gaps corrected; correction gate GREEN; committed at `55171e65`**. The device-adaptation test now uses Adam, transforms exactly the two nonzero state arrays, asserts types/values/backend identity/source immutability, and updates through the adapted state against a native reference. The M1.1 API checks are restored (`Chain tuple-gradient diagnostic`, `Enzyme Duplicated model setup and update`, shared-gradient tied subcase, whole-wrapper `freeze!`/`thaw!`), and the freeze test uses Adam with retained states and moment assertions. Correction gate at two ranks: focused conditional **367/367 per rank**, optimizer 6/6, dedicated suite `Distributed \| 6 6 2m02.6s`, 0 ambiguities, `git diff --check` clean. Evidence: devlog "Corrections applied" and "Commit" sections, `artifacts/logs/conditional-wrapper/phaseE-correction-*`. Current next action: Phase F (two- and four-rank gates).

Date: 2026-09-16

Status: Approved implementation direction

Phase E update (2026-09-30): **Phase E complete and review-corrected — GREEN, committed at `55171e65`**. `test/ext_distributed/conditional_distributedtest.jl` gained 15 testsets (+560/−0, tests only): the M1.1 real-AD and public-API cases with no stronger Phase D equivalent, the two Phase E items (stateful freeze with collective-count assertions, Functor device adaptation with backend retention and state-array traversal), and the restored M1.1 API checks (`Chain` tuple-gradient diagnostic, `Enzyme Duplicated` setup/update). M1.1 testsets already superseded by stronger D5 testsets were not duplicated. Two-rank GREEN: focused conditional 367/367 per rank across 30 testsets, focused optimizer 6/6, full dedicated suite `Distributed | 6 6 2m02.6s`, 0 ambiguities, `git diff --check` clean. All RED iterations were new-test defects; no production change. Next action: Phase F (two- and four-rank CPU/MPI gates; the four-rank case must prove weighting). Details: `wiki/devlog/2026-09-30-conditional-wrapper-phase-e-m11-battery.md`, `artifacts/logs/conditional-wrapper/phaseE-metadata.md`.

Independent review (2026-09-30): **D5 reopened for two P2 test gaps, then corrected**. The absent Adam parameter now takes prior active steps (nonzero moments) before becoming absent; caller gradients are rank-distinct with per-rank snapshots, an independent global mean, and a genuinely read-only gradient. Targeted RED runs show each corrected testset fails under the defect it targets (aliased buffer mutates the caller gradient to the global mean; absent leaf fed a zero gradient changes value and full Adam state). Final two-rank gate GREEN: conditional 232/232 per rank, optimizer 6/6, dedicated suite 6/6. Next action: Phase E. Review and correction details are in the D5 devlog.

Review update (2026-09-30): **Phase D5 complete — committed at `d6046103`**. Eight focused testsets pin the retained contracts (BatchNorm running statistics, Transpose/Adjoint parent space, caller-gradient non-mutation on both paths, immutable `update!` returns and wrapper identity, fresh `update` wrapper without input mutation, globally absent Adam state, locally absent global mean, empty trainable models, array-container shape, composition guards). The gate found one real defect: `_buildmap` flattened multidimensional array containers (`size` `(4,)` vs `(2, 2)`); `_buildmap`/`_foreach_child` now use `CartesianIndices`. RED → GREEN at two MPI ranks: focused conditional 224/224 per rank, focused optimizer 6/6, full dedicated suite `Distributed | 6 6 1m38.2s`, 0 ambiguities, `git diff --check` clean. Next action: Phase E (restore the M1.1 battery). Evidence and exact commands: `wiki/devlog/2026-09-30-conditional-wrapper-phase-d5-regression-gate.md`.

Revision: 2026-09-30. Phases D1-D3 are committed at `cbdd3edc`; the corrected D4 and D5 are committed at `d6046103`; the review-corrected Phase E is committed at `55171e65` (none pushed). Phase D and Phase E (including the review correction) are complete; Phases F-H remain.

Current entry point: section 12, Phase F (Phase E is committed at `55171e65` on `ddp/integration-m1-spike`; commit/push authorization remains separate). Sections 11 and Phase A describe completed branch integration, not commands to repeat.

Primary implementation worktree: `/home/kurapica/Projects/ddp_flux/flux-integration`

Primary implementation branch: `ddp/integration-m1-spike`

PR #2694 head to incorporate: `fee2c310a32958ca2f44c5ce5c1ddbef3db87c75`

Preserved M1.1 reference: `3370b910`

## 1. Purpose

This document is the implementation handoff for conditional-graph support in Flux distributed training.

The next model must use this document as the source of implementation intent. It must not infer the design from old branches alone.

The first implementation will use a `DistributedOptimizerState` wrapper. The wrapper will synchronize the complete gradient tree before Optimisers.jl applies per-parameter updates.

The implementation will target CPU arrays and the MPI backend first. Native NCCL execution will start only after the CPU/MPI gates pass.

This implementation is separate from PR #2694. PR #2694 deliberately removed automatic unused-parameter support to keep that PR reviewable.

The wrapper remains an experiment until the compatibility review is complete. The experiment will provide concrete evidence for a possible Optimisers.jl API change.

## 2. User Decision

The user approved these decisions:

1. Complete the wrapper implementation before proposing an Optimisers.jl change.
2. Keep development on `ddp/integration-m1-spike` in the `flux-integration` worktree.
3. Incorporate all current PR #2694 changes into that branch.
4. Preserve Optimisers.jl skip semantics for globally absent gradients.
5. Use the completed wrapper to evaluate a shared Flux/Lux solution with Carlo Lucibello.

Do not add this feature to PR #2694. Do not push this feature without a separate review decision.

## 3. Project Context

### 3.1 Repository layout

The project uses one thesis repository and three Flux worktrees.

| Path | Purpose | Rule |
|---|---|---|
| `/home/kurapica/Projects/ddp_flux/ddp-flux-thesis` | Plans, scripts, reports, logs, and project context | Store evidence and this plan here |
| `/home/kurapica/Projects/ddp_flux/ddp_flux` | Dirty C8 work on old `master` | Keep frozen |
| `/home/kurapica/Projects/ddp_flux/flux-pr2694-salvage` | PR #2694 worktree | Do not add conditional-graph code here |
| `/home/kurapica/Projects/ddp_flux/flux-integration` | Conditional-graph implementation worktree | Make wrapper changes here |

The `ddp_flux` worktree contains unrelated C8 changes. Do not merge it and do not restore files from it.

### 3.2 Important commits

| Commit | Meaning |
|---|---|
| `404ff37d` | PR #2694 merge base with upstream `master` |
| `fa3bf228` | Distributed test-harness base used by the first M1 work |
| `04ddeaa0` | First `DistributedOptimizerState` wrapper spike |
| `3370b910` | M1.1 corrective wrapper spike |
| `5c6ea561` | PR scope reset that removed unused-parameter support |
| `c64f695f` | PR #2694 head after the first complete salvage pass |
| `69703025` | First Windows routing-test correction |
| `fee2c310` | Current PR #2694 head and final Windows assertion correction |

The implementation branch is at `cbdd3edc`. The PR branch remains at `fee2c310`.

Correction to the original plan: `fee2c310` and `3370b910` are on separate branches from `fa3bf228`. A fast-forward was not possible.

Phase A used a user-approved reset to `fee2c310`. The archive preserves `3370b910`. Do not repeat that reset.

Phase C consists of checkpoint `109aad9c`, composition regression `a18392dd`, and direct-nesting fix `f5b87975`.

### 3.3 Current PR state

PR #2694 is open and mergeable at `fee2c310`.

The two-rank and four-rank MPI jobs pass. The Windows correction is in live CI.

The Julia nightly failure is unrelated package drift. The Buildkite failure is also present on other PRs.

Do not mix PR review work with this feature. If PR #2694 receives a new commit, record that commit before the next wrapper rebase.

## 4. The Failure That This Work Must Correct

### 4.1 Normal Optimisers.jl behavior

`Optimisers.setup(rule, model)` creates a state tree. Each trainable parameter maps to an `Optimisers.Leaf`.

`Optimisers.update!` uses two conceptual passes.

The first pass collects available gradients. It stops at `nothing` and `ChainRulesCore.AbstractZero` branches.

The second pass updates optimizer leaves. It calls `apply!` only for leaves with collected gradients.

This behavior is correct for one process. An absent gradient means that Optimisers.jl skips the parameter update.

The skip also preserves optimizer state. Adam does not change its moments or its decay state for that parameter.

### 4.2 Current Flux distributed behavior

The current Flux implementation puts communication in the per-leaf method:

```julia
function Optimisers.apply!(opt::DistributedOptimizer, state, x, y)
    y_avg = DistributedUtils.allreduce!(opt.backend, y, DistributedUtils.avg)
    return Optimisers.apply!(opt.opt, state, x, y_avg)
end
```

Optimisers.jl calls this method only when the local gradient exists. A local `nothing` gradient prevents the call.

The resulting collective sequence depends on the local control-flow path. Different ranks can then call different collectives.

### 4.3 Deadlock example

Consider a model with a shared trunk and three optional branches.

Rank 0 uses `trunk`, `a`, and `b`. Rank 1 uses `trunk` and `c`.

The ranks issue these collectives:

```text
Rank 0                    Rank 1
allreduce(trunk)          allreduce(trunk)
allreduce(a)              allreduce(c)
allreduce(b)              [update complete]
```

Rank 0 blocks on the third collective. Rank 1 never calls that collective.

### 4.4 Silent-corruption example

Rank 0 uses `trunk` and `a`. Rank 1 uses `trunk` and `b`.

If `a` and `b` have compatible shapes, MPI pairs their collectives by call order.

No deadlock occurs. The reduced value mixes the gradient of `a` with the gradient of `b`.

This failure is silent. A test that detects only hangs is not sufficient.

### 4.5 Existing failure evidence

The thesis repository contains a three-mode detector:

- `scripts/checks/conditional_deadlock_probe.jl`
- `scripts/checks/conditional_deadlock_check.jl`
- `make verify-conditional-deadlock`

The detector has these modes:

| Mode | Rank behavior | Current result |
|---|---|---|
| `deadlock` | Unequal collective counts | Rank 0 hangs |
| `corruption` | Equal counts for different parameters | Wrong values without a hang |
| `control` | Equal parameter use | Correct values |

The 2026-09-14 run passed 12 characterization assertions. Exit code 0 means that the detector reproduced the current bug.

Evidence is in `artifacts/logs/conditional-deadlock/detector-2026-09-14.log`.

## 5. Required Gradient Semantics

The implementation must preserve three distinct cases.

| Local state | Global state | Required contribution | Optimizer result |
|---|---|---|---|
| Gradient present | Present | Local gradient | Apply the global mean |
| Gradient absent | Present on another rank | Zero | Apply the global mean |
| Gradient absent | Absent on every rank | No gradient | Skip the optimizer step |

Do not replace every missing gradient with zero.

A zero gradient is not equal to an absent gradient for a stateful optimizer. Adam updates its internal state after a zero gradient.

For a globally absent gradient, the rebuilt tree must contain `ZeroTangent` or an equivalent skip value.

For a locally absent but globally present gradient, the rank must contribute a device-compatible zero buffer.

Every rank must make the same global-presence decision before parameter reductions start.

## 6. Why the Manual Helper Is Not the Solution

The old `resolve_unused_parameters!!` helper replaced local `nothing` values with zero arrays.

The helper did not run automatically in `Flux.train!`. Users had to remember a separate call.

The helper also lacked global-presence information. It did not preserve skip semantics for a globally absent parameter.

The original helper test used constructed gradients. It did not prove a real rank-dependent forward pass.

PR #2694 removed the helper, its test, its documentation, and its NEWS claim.

Do not restore `resolve_unused_parameters!!`. Do not provide two competing mechanisms.

## 7. Chosen Architecture

### 7.1 High-level flow

The distributed rule remains an `Optimisers.AbstractRule`:

```julia
struct DistributedOptimizer{B <: AbstractFluxDistributedBackend} <: AbstractRule
    backend::B
    opt
end
```

A specialized setup method returns a state wrapper:

```julia
struct DistributedOptimizerState{B <: AbstractFluxDistributedBackend, T}
    backend::B
    tree::T
end
```

The wrapper owns the whole-update boundary:

```text
local model gradient
        |
        v
DistributedOptimizerState update
        |
        +-- inspect complete model and gradient trees
        +-- compute global gradient presence
        +-- reduce globally present gradients in model order
        +-- preserve globally absent skip values
        |
        v
ordinary Optimisers.update! on the inner tree
```

### 7.2 Setup contract

`Optimisers.setup(::DistributedOptimizer, model)` must call setup for the inner rule:

```julia
DistributedOptimizerState(backend, Optimisers.setup(inner_rule, model))
```

The inner setup remains responsible for these behaviors:

- optimizer-state construction,
- tied mutable parameter identity,
- isbits leaf behavior,
- frozen leaf state,
- optimizer-specific initialization.

The wrapper must not reimplement optimizer rules.

### 7.3 Update contract

The specialized `update!` method must:

1. Reject higher-order gradient arguments.
2. Build a synchronized gradient tree.
3. Call `Optimisers.update!` on the inner tree.
4. Return the model from the inner update.
5. Return the same wrapper state for the in-place path.

The specialized `update` method must:

1. Reject higher-order gradient arguments.
2. Build a synchronized gradient tree.
3. Call `Optimisers.update` on the inner tree.
4. Wrap the returned inner state.
5. Return the model from the inner update.

### 7.4 Communication boundary

All collectives must execute outside automatic differentiation.

Do not put collectives in the loss function. Do not add a custom optimizer for the training path.

The final inner call must use the public `Optimisers.update` or `Optimisers.update!` API.

### 7.5 Composition rule

`DistributedOptimizer` must be the outermost rule for this first implementation.

This form is valid:

```julia
DistributedOptimizer(backend, OptimiserChain(...))
```

This form is invalid:

```julia
OptimiserChain(..., DistributedOptimizer(backend, Adam()))
```

Nested use must throw an `ArgumentError`. Silent per-leaf fallback is not acceptable.

## 8. Synchronization Algorithm

### 8.1 Required inputs and output

The synchronization function receives:

- the distributed backend,
- the complete model,
- one local gradient tree.

It returns an Optimisers-compatible gradient tree. The output represents the global mean.

The function must not mutate the caller gradient. It must not mutate the model.

### 8.2 Traversal source of truth

The model defines the collective order. The local gradient tree must not define that order.

The traversal must use public Optimisers and Functors behavior:

- descend through `Optimisers.trainable(model_node)`,
- stop at `Optimisers.isnumeric(model_node)`,
- pair model and gradient nodes structurally,
- use `Functors.functor(typeof(model_node), gradient_node)` where applicable.

This traversal excludes non-trainable arrays such as BatchNorm running statistics.

Both passes must use the same model-defined child order. The order of a named `trainable` subset must not change occurrence order.

For named children, select fields by name and retain functor field order. Pair gradients with those structural field names.

For positional containers, retain their shape and index order. Do not flatten an array container during reconstruction.

Use one normalized selection rule in both passes. Do not independently traverse raw `trainable` values and functor children.

### 8.3 Structural gradient pairing

Do not replay model `KeyPath` values against the gradient tree.

Gradient structures can differ from model structures. `Transpose` and `Adjoint` are important examples.

Use the same parent-space concept that Optimisers.jl uses for gradient pairing.

Treat `nothing` and `ChainRulesCore.AbstractZero` as absent gradient subtrees.

If a child is absent from the gradient container, treat that child as locally absent.

### 8.4 Buffer rules

Allocate one writable synchronization buffer for each unique mutable trainable parameter.

Allocate one buffer for each isbits trainable occurrence. Optimisers.jl does not identity-cache isbits values.

Use `similar(parameter)` for mutable parameter buffers. Copy the local gradient with `copyto!`.

Initialize a missing local contribution with zero. Do not use `collect(gradient)`.

Do not alias a caller-owned gradient array. The in-place all-reduce mutates that array.

### 8.5 Tied parameter accumulation

A mutable parameter can occur more than once in the model.

Optimisers.jl gives all occurrences one shared `Leaf`. It accumulates the gradient contribution from each occurrence.

The synchronization walk must match this behavior before the all-reduce.

For the first occurrence, allocate the buffer and record its index. For later occurrences, add the local contribution to that buffer.

After the all-reduce, emit the synchronized buffer only at the first occurrence. Emit `ZeroTangent` at later occurrences.

This rule prevents the inner Optimisers traversal from adding the same global gradient more than once.

### 8.6 Explicit occurrence plan

M1.1 uses a counter during the rebuild pass. That counter is unsafe for mixed tied and isbits parameters.

Use an explicit occurrence plan instead.

The walk must record one plan entry for every trainable occurrence. Each entry must identify its synchronization buffer.

A plan entry needs enough information to distinguish these cases:

- first occurrence of a mutable parameter,
- later occurrence of a tied mutable parameter,
- one isbits occurrence.

One possible private representation is:

```julia
struct _SyncOccurrence
    buffer_index::Int
    emit::Bool
end
```

The exact type is not mandatory. The invariant is mandatory.

The rebuild pass must consume one occurrence-plan entry for each numeric model occurrence. It must not infer buffer indices from occurrence counts.

Collection and reconstruction must consume occurrences in the same canonical model order (section 8.2).

A sequential occurrence plan is safe only with that shared order. Otherwise, each entry needs an explicit structural association.

After reconstruction, require exactly one consumed entry per trainable occurrence. No entry can remain unused.

Structural associations identify model occurrences only. Do not use them to index the raw gradient tree instead of structural gradient pairing.

### 8.7 Presence protocol

Build one presence flag for each synchronization buffer:

```text
0.0f0 = no local gradient contribution
1.0f0 = one or more local gradient contributions
```

Reduce the complete flag vector once with `DistributedUtils.avg`.

The global-present condition is `flag > 0`. Do not use `flag > 0.5`.

For more than two ranks, one active rank produces a fraction less than or equal to `0.5`.

### 8.8 Host-safe presence metadata

Use a CPU `Vector{Float32}` for the presence vector in the first implementation.

This vector contains metadata, not parameter gradients. The NCCL backend already falls back to MPI for CPU buffers.

The CPU vector avoids host scalar indexing into a GPU boolean array. It also avoids a device synchronization for control flow.

Keep parameter buffers on the parameter device. Do not copy gradient data to the CPU.

Document this split:

- presence metadata uses the CPU/MPI path,
- parameter gradients use the selected backend and their existing device.

### 8.9 Collective order

Every rank must execute this sequence:

1. One presence-vector all-reduce.
2. One gradient all-reduce for each globally present buffer.
3. Gradient all-reduces in ascending buffer index.

Every rank obtains the same global-presence vector. Therefore, every rank selects the same gradient collectives.

Do not use local presence to skip a collective after the presence-vector reduction.

### 8.10 Rebuild rules

The rebuilt gradient tree must mirror the model structure that Optimisers.jl accepts.

Use these leaf results:

| Occurrence state | Rebuilt value |
|---|---|
| First occurrence and globally present | Synchronized buffer |
| Later tied occurrence | `ZeroTangent()` |
| Globally absent | `ZeroTangent()` |
| Non-trainable child | `nothing` |

Container nodes without trainable descendants can use `nothing` where Zygote uses `nothing`.

### 8.11 Empty model behavior

If the model has no trainable parameters, return the input gradient tree.

Do not issue a presence collective for an empty synchronization plan.

This behavior must be identical on all ranks because the model structure must be identical.

## 9. M1.1 Evidence and Remaining Defects

### 9.1 What M1.1 already proved

Commit `3370b910` passed the complete MPI suite at two and four ranks.

Its conditional test contains real AD and these cases:

- rank-dependent branches,
- analytical global means,
- four-rank weighting,
- Adam state comparisons,
- globally absent parameters,
- BatchNorm trainable semantics,
- transpose and adjoint gradients,
- tied mutable parameters,
- isbits parameters,
- optimizer composition rejection,
- caller-gradient non-mutation,
- immutable model returns,
- higher-order rejection,
- Enzyme diagnostics,
- frozen parameters,
- collective counts,
- `Flux.setup` and `Flux.train!`.

Use these tests as source material. Do not assume that their combined coverage closes the defects below.

### 9.2 Mixed tied and isbits defect

M1.1 deduplicates mutable buffers but increments its rebuild counter for every numeric occurrence.

The counter can exceed the number of unique buffers after a tied occurrence. A later isbits occurrence can then use the wrong index.

Add one model that mixes these properties in this order:

1. First occurrence of mutable parameter `p`.
2. Second occurrence of the same `p`.
3. An isbits trainable parameter.
4. A distinct mutable parameter `q`.

The test must prove values, state updates, and collective counts.

### 9.3 GPU presence-control defect

M1.1 can allocate the presence vector with `similar` from a parameter buffer.

A GPU parameter can produce a GPU presence vector. `pres .> 0` then produces a GPU boolean vector.

Host code later reads `present[i]`. CUDA scalar-indexing checks can reject that operation.

The new implementation must keep presence decisions on the host. Add a static or mock-device test before native NCCL work.

### 9.4 Legacy `trainable` compatibility

Optimisers.jl supports older `trainable` definitions that return tuples for struct children. Its internal compatibility path maps those values back to model fields.

M1.1 does not fully prove equivalent behavior. Add a custom layer with an old tuple-returning `trainable` method.

The synchronization traversal must select the same parameters as `Optimisers.setup`. It must preserve the same warning behavior where practical.

Do not call the private `Optimisers._trainable` function.

Optimisers 0.4.9 defines this behavior in `src/interface.jl:186-195`:

- Named selections merge into model children whose unselected values are `nothing`. Existing model-field order remains unchanged.
- Legacy tuple selections use `c in tr` over model children. This is value membership, not identity membership.

Use these definitions as read-only references. Implement equivalent selection through public `Optimisers.trainable` and Functors APIs.

Add equal-valued, distinct mutable fields to the legacy regression. An identity-only replacement can select different parameters from native setup.

Native setup already emits the legacy warning. Preserve that warning without adding a warning on every distributed update.

### 9.5 Reordered named trainable defect

Phase C collects buffers in `trainable` order but reconstructs gradients in functor order.

For model fields `(a, b)`, a named selection `(b=b, a=a)` reverses collection order. Isbits reconstruction still consumes buffers by position.

Independent, equal-shaped isbits parameters can therefore receive each other's gradients without any tied parameter.

Add a separate regression with different gradient values for `a` and `b`. Compare against a native Optimisers global-mean reference.

The occurrence-plan correction must also correct traversal order. Replacing a counter with a sequential list alone is insufficient.

### 9.6 Wrapper state compatibility

The wrapper changes the shape returned by setup.

Before the wrapper:

```text
state.layer.weight
```

After the wrapper:

```text
state.tree.layer.weight
```

This change is the main compatibility cost. Do not hide it with broad `getproperty` forwarding in the first implementation.

The compatibility review must examine checkpoints, state inspection, `adjust!`, freezing, device movement, and serialization.

## 10. Scope

### 10.1 In scope

- Whole-model synchronization through `DistributedOptimizerState`.
- Real rank-dependent conditional models.
- CPU arrays and MPI execution.
- Two-rank and four-rank correctness evidence.
- Descent and Adam comparisons.
- Globally absent skip semantics.
- Tied, isbits, wrapped, and immutable model values.
- Existing Flux training APIs.
- Public Optimisers.jl interfaces only.
- Clear errors for unsupported composition and higher-order gradients.
- A compatibility report for the wrapper state.

### 10.2 Out of scope for the first CPU/MPI gate

- Performance optimization.
- Reuse of caller gradient buffers.
- Gradient bucketing.
- Overlap of communication and computation.
- Native NCCL correctness claims.
- Higher-order distributed gradients.
- Enzyme `Duplicated` updates.
- Nested ownership of `DistributedOptimizer` inside optimizer chains.
- A new Optimisers.jl API.
- A new Lux.jl implementation.
- The dirty-master `no_sync` experiment.
- Unrelated PR #2694 corrections.

Do not expand scope because an adjacent cleanup is convenient.

## 11. Branch Integration Procedure

Historical procedure: Phase A is complete. The original fast-forward premise was incorrect (section 3.2).

Do not run the branch-movement commands in this section. Phase D starts at `f5b87975` without branch movement.

### 11.1 Preconditions

Before any branch movement, make sure that these worktrees are clean:

```bash
git -C /home/kurapica/Projects/ddp_flux/flux-integration status --short
git -C /home/kurapica/Projects/ddp_flux/flux-pr2694-salvage status --short
```

The expected output is empty for both commands.

Record these commit IDs:

```bash
git -C /home/kurapica/Projects/ddp_flux/flux-integration rev-parse HEAD
git -C /home/kurapica/Projects/ddp_flux/flux-pr2694-salvage rev-parse HEAD
```

The expected values are `3370b910...` and `fee2c310...`.

If either worktree is dirty, stop. Do not remove or overwrite unknown changes.

### 11.2 Preserve the old spike

The branch `archive/pr2694-unused-gradient-spike` already points to `3370b910`.

Make sure that this reference still exists:

```bash
git -C /home/kurapica/Projects/ddp_flux/flux-integration show-ref --verify refs/heads/archive/pr2694-unused-gradient-spike
```

Do not move the archive branch.

### 11.3 Incorporate the PR head

From the `flux-integration` worktree, fast-forward the implementation branch:

```bash
git merge --ff-only ddp/windows-path-fix
```

The new branch head must be `fee2c310`.

This fast-forward removes the wrapper from the working tree because the PR scope-reset commits removed it.

This temporary removal is expected. The archive branch preserves the implementation reference.

### 11.4 Port policy

Use `3370b910` as a read-only reference. Reapply the design as new commits after `fee2c310`.

Do not revert `5c6ea561`. A complete revert restores old documentation and the obsolete manual helper.

Do not replace the complete `public_api.jl` file with its old version. That action removes later PR fixes.

Port only the required optimizer blocks and tests. Use `apply_patch` for manual edits.

## 12. Implementation Phases

### Phase A: Record the integrated baseline

Goal: prove that the implementation branch contains the exact PR head and no wrapper code.

Actions:

1. Fast-forward to `fee2c310`.
2. Make sure that the worktree is clean.
3. Make sure that `DistributedOptimizerState` is absent.
4. Run the standalone routing test.
5. Run the current two-rank distributed suite.
6. Store the baseline logs in a new conditional-graph artifact directory.

Pass conditions:

- HEAD equals `fee2c310`.
- The routing test passes 17/17.
- The current distributed suite passes 5/5.
- The worktree remains clean.

Do not run the full main suite on the local 14 GiB host. That route previously exhausted memory through Reactant/XLA instantiation.

### Phase B: Add a fast RED corruption test

Goal: prove the product error without a deliberate hang.

Add `test/ext_distributed/conditional_distributedtest.jl` with one focused real-AD case.

The test model must contain two equal-shaped conditional branches. Rank 0 must use branch A and rank 1 must use branch B.

The test must calculate the expected global mean for each branch. It must assert both updated parameter values.

The current per-leaf implementation must fail quickly with wrong values. It must not wait for a timeout.

Run the child test directly under two MPI ranks. Capture the failure and exit code.

Commit only the RED test. Do not include production code in this commit.

Pass condition for this phase: the new assertion fails for the expected numerical reason.

### Phase C: Add the wrapper boundary

Goal: move communication from per-leaf `apply!` to whole-model `update` methods.

Add these elements to `src/distributed/public_api.jl`:

- `DistributedOptimizerState`,
- its Functors declaration,
- specialized setup methods,
- specialized `update` and `update!` methods,
- composition errors in per-leaf `init` and `apply!`,
- higher-order gradient errors,
- state delegation for adjustment and freezing,
- `synchronize!!` support for the wrapper.

Add a minimal synchronization implementation that makes the RED test pass.

Do not restore the old manual helper. Do not add user documentation in this phase.

Run the focused test again. Store GREEN evidence.

### Phase D: Correct traversal and presence handling

Goal: correct traversal and presence handling through focused RED/GREEN steps before the broad Phase E test expansion.

Baseline: `ddp/integration-m1-spike` at `f5b87975`, with Phase C review complete. Phase D does not require another reset or merge.

#### Shared contract

- The model defines occurrence order. Local gradient presence never defines that order.
- Both passes use the same normalized selection and canonical child order (section 8.2).
- Each numeric trainable occurrence has an explicit buffer index and an emission decision.
- Mutable aliases share one buffer. Each isbits occurrence has its own buffer.
- Repeated mutable occurrences contribute locally but emit the global gradient only once.
- Gradient pairing uses the model type and gradient structure, not raw gradient KeyPath lookup.
- Presence metadata is a CPU `Vector{Float32}`. Parameter buffers retain their device.
- Global absence emits `ZeroTangent` and preserves optimizer state.
- The inner public Optimisers update performs every optimizer-specific state transition.

Read the active Optimisers package before implementation. Record its version and source path with the evidence.

The Phase C evidence used Optimisers 0.4.9. Reference `src/interface.jl` for setup, structural gradient pairing, and legacy selection.

Do not call private Optimisers functions. Do not add performance work, GPU tests, user documentation, or a new public API.

#### D1: Canonical traversal order

Status 2026-09-28: implemented in the `flux-integration` working tree (uncommitted). `_walk!` now uses `_foreach_child`/`_trainable_child` (same normalized selection as `_build!`). Regression: `@testset "canonical traversal order for reversed named trainable"` in `test/ext_distributed/conditional_distributedtest.jl` (`IsoVec` isbits leaf, `ReversedTrainable`, `RecordingBackend`, native global-mean reference). RED → GREEN evidence under `artifacts/logs/conditional-wrapper/phaseD-d1-*`; devlog `wiki/devlog/2026-09-28-conditional-wrapper-phase-d1-canonical-traversal.md`.

1. Add a custom model with two independent, equal-shaped isbits trainable arrays.
2. Return a named `trainable` selection in reverse model-field order.
3. Use different gradient values for each field and rank.
4. Assert parameter values against an independent native Optimisers global-mean reference.
5. Assert collective order and count through a recording backend.
6. Run the focused regression before production changes.
7. Record the expected wrong-value RED result.
8. Normalize child selection into model order for both passes.
9. Run the regression again and record GREEN evidence.

Use a minimal deterministic fixture for this structural regression. Keep the existing real-AD conditional regression as a separate control.

The recording backend must distinguish equal-shaped buffers by their input values or explicit expected sequence, not shape alone.

#### D2: Explicit occurrence mapping

Status 2026-09-28: implemented in the `flux-integration` working tree (uncommitted). `_walk!` now pushes a `_SyncOccurrence(buffer_index, emit)` entry per numeric occurrence; `_build!` consumes the plan through a cursor and `_sync_gradients` asserts complete consumption (`ArgumentError` otherwise). Regression: `MixedTiedIsbits` (`p`, tied `p2`, isbits `c`, distinct mutable `q`) with three Adam steps and a native reference plus an equal-valued isbits guard in `test/ext_distributed/conditional_distributedtest.jl`; the fixture is type-preserving (`c::IsoVec{4}` + reconstruction `convert`, `isbits(model.c)` asserted each step). RED (before the fix): 33 passed / 5 failed per rank, all on isbits `c`; GREEN: 38/38. Evidence under `artifacts/logs/conditional-wrapper/phaseD-d2-*`; devlog `wiki/devlog/2026-09-28-conditional-wrapper-phase-d2-occurrence-plan.md`.

1. Add a model in this order: mutable `p`, the same `p`, an isbits parameter, distinct mutable `q`.
2. Give each occurrence a distinguishable local gradient contribution.
3. Assert summed tied contributions against a native Optimisers reference.
4. Assert parameter values and Adam state after multiple steps.
5. Assert one presence reduction plus three gradient reductions when all buffers are globally present.
6. Include a step where one tied occurrence is absent but the other contributes.
7. Record RED evidence before the occurrence-mapping correction.
8. Replace inferred buffer indices with explicit occurrence-plan entries.
9. Assert complete plan consumption during reconstruction.
10. Run D1 and D2 again and record GREEN evidence.

Also cover equal-valued isbits occurrences. They must remain independent rather than share a buffer through equality or identity caching.

#### D3: CPU presence metadata

Status 2026-09-29: complete and reviewed. Commit `cbdd3edc` contains D1-D3. Presence flags use a CPU `Vector{Float32}`. Parameter buffers retain device-like placement. The independent two-rank run passed D3 16/16 per rank, with zero ambiguities. The mock presence fraction is simulated. Four-rank and native CUDA/NCCL evidence remain deferred. Evidence: `wiki/devlog/2026-09-29-conditional-wrapper-phase-d3-cpu-presence-metadata.md`.

1. Add a device-like parameter fixture that exposes device-derived metadata allocation without GPU hardware.
2. Record every collective buffer type and operation.
3. Assert that the presence buffer is exactly `Vector{Float32}`.
4. Assert that parameter buffers retain the fixture's device-like placement.
5. Record the allocation or scalar-access RED result before production changes.
6. Allocate presence metadata directly on the CPU.
7. Retain one presence reduction with `avg` and the threshold `> 0`.
8. Run the fixture and focused MPI regressions again.

The fixture must exercise `_sync_gradients`, not a separate copy of the allocation expression.

A deterministic mock backend can supply a presence fraction of `0.25f0` to exercise the threshold.

This mock does not replace the four-rank weighting gate in Phase F. It does not prove native CUDA or NCCL behavior.

#### D4: Legacy trainable selection

Status 2026-09-29 (corrected after review P1): implemented in the `flux-integration` working tree (uncommitted, based on committed `cbdd3edc`). The synchronization walk reads the selection from the state tree returned by `Optimisers.setup` (`state.tree`), the retained setup-time record, instead of recomputing `trainable` on every update: `_sync_gradients(backend, tree, model, grads)`; `_walk!`/`_build!` pair each model child with its tree node through the shared structural `_child` helper; a numeric child is trainable iff its tree node is an `Optimisers.Leaf`, and an empty tree node (`()`) skips the subtree. `_trainable_child` is removed; no warning is emitted in the walk. The first attempt (`_trainable_child(ch::NamedTuple, tr::Tuple, k) = ch[k] in tr`, value membership per update) failed review: after the first update made equal-valued fields diverge, a setup-selected field was silently dropped from later updates although its `Leaf` remained (review characterization `phaseD-d4-review-multistep.jl`: step 2 wrapper `b=0.45` vs native `-0.10`, 2 collectives vs 3; both update paths). Regression: the `LegacyTupleTrainable` one-step native-reference testset plus a new multi-step testset `legacy tuple trainable selection persists across updates` (two steps, both `Optimisers.update!` and `Optimisers.update`, stateful `Momentum`, native global-mean reference, parameter + `state.tree == ref_state` + cumulative collective-count assertions, cross-rank equality). RED (first defect, before the legacy predicate existed): exit 1, 33 s, D4 12 passed / 3 failed / 5 errored, first error `MethodError: haskey(::Tuple{Vector{Float64}}, ::Symbol)`. GREEN (first fix): exit 0, 44 s, D4 20/20 per rank. RED2 (second-step defect, corrected tests on the buggy source): exit 1, 37.8 s, new testset 24 passed / 6 failed per rank. GREEN2 (final): exit 0, 48 s, control 5/5 + D1 10/10 + guard 11/11 + D2 38/38 + D3 16/16 + D4 20/20 + D4-multistep 30/30 per rank; full 2-rank suite 6/6 (`Distributed | 6 6 1m12.0s`); 0 ambiguities. Evidence under `artifacts/logs/conditional-wrapper/phaseD-d4-*`; devlog `wiki/devlog/2026-09-29-conditional-wrapper-phase-d4-legacy-trainable.md`.

1. Add a struct with a tuple-returning `trainable` method and an excluded array field.
2. Add equal-valued, distinct mutable fields to expose value-membership semantics.
3. Compare selected leaves against public native `Optimisers.setup` behavior.
4. Compare updates and collective counts against the same selected parameters.
5. Record the legacy-selection RED result before production changes.
6. Read the selected children from the state tree returned by `Optimisers.setup`; keep canonical model order through the model's functor children.
7. Preserve the native setup warning without repeated update warnings.
8. Compare parameters, optimizer state, and collective counts against a native reference over at least two updates, for both `Optimisers.update!` and `Optimisers.update`, so a selection that changes between steps fails.
9. Run D1-D4 again and record GREEN evidence.

If an earlier normalization change also fixes D4, record that dependency. Demonstrate RED against the unchanged Phase C baseline separately.

Do not weaken a regression or introduce an artificial defect to produce RED evidence.

#### D5: Regression gate

Status 2026-09-30: complete and committed at `d6046103` (together with the corrected D4). Eight testsets added to `test/ext_distributed/conditional_distributedtest.jl` (BatchNorm running statistics, transpose/adjoint parent space, caller-gradient non-mutation for both update paths, immutable `update!` return + wrapper identity, fresh `update` wrapper without input mutation, globally absent Adam state, locally absent global mean, empty trainable model, array-container shape). The gate found and corrected one defect: `_buildmap`/`_foreach_child` now map array containers over `CartesianIndices`, so a 2×2 `Matrix` child keeps its shape instead of being flattened to a length-4 vector. RED: focused 2-rank conditional exit 1, array-container 18/20, `(4,) == (2, 2)`; GREEN: 224/224 per rank, focused optimizer 6/6, full dedicated 2-rank suite `Distributed | 6 6 1m38.2s`, 0 ambiguities, `git diff --check` clean. Evidence: `artifacts/logs/conditional-wrapper/phaseD-d5-*`; devlog `wiki/devlog/2026-09-30-conditional-wrapper-phase-d5-regression-gate.md`.

Status update after independent review (2026-09-30): the review reopened D5 for two P2 coverage gaps, both now corrected in the same working tree. (1) The absent-Adam testset now runs two active steps on `absent` first, asserts nonzero moments and a moved value, snapshots value and full optimizer state, then runs three globally absent steps and asserts exact preservation of the snapshot (a reset to fresh state would fail because the test also asserts `snap_state != fresh.absent.state`). (2) The caller-gradient testset now uses rank-distinct gradients on both update paths and in the read-only case, compares every caller gradient with its local snapshot, and checks each updated model against an independent global mean; the writable `view` was replaced by a `ReadOnlyGradient` wrapper whose `setindex!` throws. Targeted RED A: with a temporary aliasing defect (`buf = g` for the length-2 gradients), the corrected testset fails `gs.w == snap` with the global mean `[0.4, -0.075]` on both ranks, and the read-only case errors on unaliasing — earlier testsets all pass; exit 1, 1 m 10.8 s. Targeted RED B: with a temporary absent-leaf defect (`_build!` emits the zero buffer instead of `ZeroTangent`), the corrected absent testset fails `model.absent == snap_value` and `state.tree.absent.state == snap_state` with changed moments and step products — earlier testsets and the caller testset all pass; exit 1, 1 m 14.3 s. Production source was restored byte-for-byte after both injections (md5 `06c2b464a5d9c1c784e07cb7752a0e7e`) and no defect was shipped. Final GREEN: focused conditional 232/232 per rank, focused optimizer 6/6, full dedicated 2-rank suite `Distributed | 6 6 1m45.2s`, 0 ambiguities, `git diff --check` clean. Combined uncommitted D4 + D5 diff relative to `cbdd3edc`: 2 files, +633/−66 (`src` +67/−65 unchanged, test +567/−1). Evidence: `artifacts/logs/conditional-wrapper/phaseD-d5-correction-*-2026-09-30.log`, `phaseD-d5-metadata.md`; devlog corrections section. The whole Phase D4 + D5 diff was committed as `d6046103` (2 files, +633/−66, not pushed) after the corrections.

Retain or add focused assertions for these contracts:

- BatchNorm running statistics remain excluded.
- Transpose and Adjoint gradients pair in parent space.
- Caller gradients remain unchanged for both update paths.
- `update!` propagates immutable model returns and retains wrapper identity.
- `update` returns a new wrapper without mutating the input model or state.
- Globally absent parameters and Adam state remain unchanged after a prior active step.
- Locally absent parameters receive the global mean when another rank contributes.
- Empty trainable models perform no collectives.
- Array containers retain their shape during gradient reconstruction.
- Phase C composition guards remain GREEN.

Run the focused conditional and optimizer files at two MPI ranks after each related correction.

After D1-D4 pass, run the complete dedicated two-rank suite. Do not run the full local main suite.

Run the ambiguity check and `git diff --check`. The ambiguity count must remain zero.

Four-rank suite execution remains Phase F. Native GPU/NCCL execution remains outside this phase.

#### Evidence and completion

Use `artifacts/logs/conditional-wrapper/phaseD-*` for logs and command metadata.

Record exact commands, source state, package versions, exit codes, elapsed times, and assertion totals for each RED/GREEN pair.

For mock tests, label simulated presence results explicitly. For MPI tests, record actual rank counts and collective traces.

Each RED result must identify the intended defect. Fixture errors and dependency failures do not count as RED evidence.

Keep tests and their corresponding fixes reviewable as separate changes. Commit or push only with explicit user authorization.

Phase D is complete only after D1-D5 pass. Completion does not imply Phase E, F, G, or H completion.

### Phase E: Restore and improve the M1.1 test battery

Status 2026-09-30: complete, review-corrected, and GREEN, committed at Flux `55171e65` (not pushed). Ported 13 M1.1 testsets with no stronger Phase D equivalent plus the two Phase E items (stateful freeze with collective-count assertions; Functor device adaptation with state-array traversal and backend retention) and the restored M1.1 API checks (`Chain` tuple-gradient diagnostic, `Enzyme Duplicated` setup/update). Two-rank GREEN: focused conditional 367/367 per rank (30 testsets), optimizer 6/6, dedicated suite 6/6, 0 ambiguities, `git diff --check` clean. Tests only: +560/−0. Evidence: `artifacts/logs/conditional-wrapper/phaseE-metadata.md` and `phaseE-correction-*`; devlog `wiki/devlog/2026-09-30-conditional-wrapper-phase-e-m11-battery.md`.

Goal: regain the proven M1.1 cases without restoring its known defects.

Use `3370b910:test/ext_distributed/conditional_distributedtest.jl` as reference.

Port each testset deliberately. Keep test names that still describe the contract.

Retain the focused Phase D regressions. Expand their real-AD coverage without duplicating fixtures unnecessarily.

Add the remaining tests:

- partial freeze on all ranks with collective-count assertions,
- wrapper traversal through device adaptation if the existing public API supports it.

Run the conditional file after each related group. Do not wait until all testsets are present.

### Phase F: Run CPU/MPI gates

Goal: prove correctness at two and four ranks.

Run the focused child file first. Then run the complete dedicated suite.

Required rank counts:

- two ranks,
- four ranks.

The four-rank analytical case must prove weighting. One rank can use branch A while three ranks use branch B.

Record exact command lines, elapsed times, test totals, and exit codes.

### Phase G: Convert the external detector

Goal: make the external detector pass only after the error is corrected.

Update `scripts/checks/conditional_deadlock_check.jl` and its probe expectations.

After the correction:

- `deadlock` mode must finish on both ranks,
- `corruption` mode must report correct branch values,
- `control` mode must remain correct,
- collective traces must show one presence reduction plus ordered gradient reductions.

Change the final summary. Exit code 0 must mean that conditional-graph support works.

Update the Makefile comments. They currently state that exit code 0 confirms the bug.

Keep the old log as historical evidence. Store new logs under a different name.

### Phase H: Compatibility review

Goal: decide whether the wrapper fits Flux and Optimisers workflows.

Review these interfaces:

- `Flux.setup`,
- `Flux.train!`,
- `Optimisers.setup`,
- `Optimisers.update`,
- `Optimisers.update!`,
- `Optimisers.adjust` and `adjust!`,
- `Optimisers.freeze!` and `thaw!`,
- `DistributedUtils.synchronize!!`,
- Functors traversal and device movement,
- checkpoint serialization,
- direct state inspection.

Document every state-shape incompatibility. Do not add forwarding methods only to make tests convenient.

Prepare a short comparison between the wrapper and an Optimisers.jl pre-update hook.

### Phase I: CPU implementation review checkpoint

Stop after the CPU/MPI gates and compatibility review.

Do not start native NCCL tests until the user reviews these results:

- production diff,
- test matrix,
- wrapper compatibility report,
- remaining limitations,
- exact evidence paths,
- proposed Optimisers.jl discussion points.

## 13. Detailed Test Matrix

### 13.1 Core distributed semantics

| Case | Required proof |
|---|---|
| Same parameter used on all ranks | Result equals the global-mean reference |
| Parameter used on one rank | Missing ranks contribute zero |
| Parameter used on some ranks | Weight equals active-rank contribution divided by all workers |
| Parameter absent on all ranks | Parameter and optimizer state remain unchanged |
| Different branches with equal shapes | No silent cross-parameter reduction |
| Different branches with unequal counts | No deadlock |

### 13.2 Optimizer semantics

| Optimizer | Required proof |
|---|---|
| `Descent` | Parameters match an analytical reference |
| `Adam` | Parameters, moments, variance, and decay state match a local global-mean reference |
| Frozen leaf | The parameter and leaf state remain unchanged |
| Thawed leaf | Updates resume with the expected state |

### 13.3 Model traversal

| Model property | Required proof |
|---|---|
| Nested custom layer | Trainable leaves synchronize in model order |
| BatchNorm | `γ` and `β` synchronize, while `μ` and `σ²` do not |
| `Transpose` | Parent-space gradient is correct |
| `Adjoint` | Parent-space gradient is correct |
| Tied mutable parameter | One global reduction and one effective inner gradient |
| Isbits parameter | One independent buffer per occurrence |
| Mixed tied/isbits model | Correct buffer mapping after tied deduplication |
| Reordered named `trainable` | Independent isbits fields retain their own gradients in canonical model order |
| Old tuple `trainable` | Same selected leaves as `Optimisers.setup` |
| Equal-valued legacy fields | Selection matches native value-membership behavior, not identity-only selection |
| Array containers | Rebuilt gradient containers retain model shape and index order |
| Empty trainable model | No collective and no error |

### 13.4 API behavior

| API | Required proof |
|---|---|
| `Flux.setup` | Returns `DistributedOptimizerState` |
| `Optimisers.setup` | Returns the same wrapper contract |
| `Flux.train!` | Executes a complete distributed step |
| `Optimisers.update!` | Updates mutable models and returns the inner model result |
| `Optimisers.update` | Returns a new wrapper and a new model where required |
| `adjust` | Changes the inner rule without losing the backend |
| `adjust!` | Changes the inner tree in place |
| `freeze!` | Delegates to the inner tree |
| `thaw!` | Delegates to the inner tree |
| `synchronize!!` | Synchronizes inner optimizer state |
| Nested distributed rule | Throws a clear `ArgumentError` |
| Higher-order gradients | Throw a clear `ArgumentError` |
| Enzyme `Duplicated` update | Throws the documented error |

### 13.5 Mutation and placement

| Property | Required proof |
|---|---|
| Caller gradient | Byte or value equal after update |
| Mutable parameter buffer | Uses `similar(parameter)` |
| Gradient copy | Uses `copyto!` |
| Presence metadata | Resides on the CPU |
| Parameter gradient | Remains on the parameter device |
| CUDA scalar indexing | No host scalar access to GPU arrays |

### 13.6 Collective protocol

Use `CountingBackend` or an equivalent logging wrapper.

For each step, the expected count is:

```text
1 presence collective + N globally present unique synchronization buffers
```

Globally absent parameters do not receive gradient collectives.

Tied mutable parameters count once. Isbits occurrences count independently.

Every rank must report the same count and order.

## 14. Commands

### 14.1 Local baseline commands

Run these commands from `/home/kurapica/Projects/ddp_flux/flux-integration`:

```bash
git status --short --branch
git rev-parse HEAD
julia --startup-file=no --project=test test/distributed_routing.jl
JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi
```

The direct runner owns child process creation. Do not wrap this command in another `mpiexec`.

### 14.2 Focused conditional commands

Use the dedicated runner with a file selector only if its current argument contract supports that selector.

Otherwise, run the child file with `mpiexecjl`:

```bash
FLUX_TEST_DISTRIBUTED_BACKEND=mpi mpiexecjl -n 2 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl
FLUX_TEST_DISTRIBUTED_BACKEND=mpi mpiexecjl -n 4 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl
```

Before use, inspect `test/ext_distributed/distributed_setup.jl`. Match its backend-selection contract exactly.

Do not guess whether the child reads `ARGS` or `ENV`. PR #2694 centralized this setup.

### 14.3 Full CPU/MPI gates

```bash
JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi
JULIA_MPI_TEST_NPROCS=4 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi
```

Expected suite count after the conditional file returns: 6 child files.

### 14.4 Thesis detector

Run this command from `/home/kurapica/Projects/ddp_flux/ddp-flux-thesis`:

```bash
make verify-conditional-deadlock FLUX_REPO_PATH=../flux-integration
```

First, change the Makefile target to use `--project=$(FLUX_REPO_PATH)/test` for the driver.

This change makes the detector use the requested Flux test environment. It avoids a hidden dependency on the thesis manifest.

### 14.5 Static checks

```bash
git diff --check
julia --startup-file=no --project=test -e 'using Flux, Test; Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)'
```

Record the ambiguity count. The M1.1 result was zero.

### 14.6 HPC commands

Do not run these commands until all local CPU/MPI gates pass.

Load the `remote` skill before any HPC command.

After each source sync, precompile on a compute node:

```bash
scripts/remote/sync_code.sh hpc
scripts/remote/precompile.sh hpc
```

Then run CPU/MPI gates with make variables as command arguments:

```bash
NTASKS=2 CPUS_PER_TASK=4 scripts/remote/run.sh hpc "make c8-mpi FLUX_REPO_PATH=../flux-integration"
NTASKS=4 CPUS_PER_TASK=4 scripts/remote/run.sh hpc "make c8-mpi-4 FLUX_REPO_PATH=../flux-integration"
```

If a test prints `Precompiling packages...`, stop the test. Run compute-node precompilation again.

## 15. File Change Plan

### 15.1 Flux production code

Primary file:

- `src/distributed/public_api.jl`

Expected changes:

- replace per-leaf communication with wrapper-owned communication,
- add `DistributedOptimizerState`,
- add setup and update methods,
- add private synchronization helpers,
- add delegation methods,
- update docstrings and restrictions.

Keep the implementation in this file for the first correctness pass. Do not create a new source file only to shorten the diff.

### 15.2 Flux tests

Primary files:

- `test/ext_distributed/conditional_distributedtest.jl`,
- `test/ext_distributed/optimizer_distributedtest.jl`.

Possible supporting file:

- `test/ext_distributed/helper.jl`, only if a reusable backend wrapper belongs there.

Do not change the runner unless the new test exposes a runner defect.

### 15.3 Thesis repository

Expected later changes:

- convert `scripts/checks/conditional_deadlock_check.jl` to a success regression,
- update `scripts/checks/conditional_deadlock_probe.jl` verdicts,
- update the Makefile target and comments,
- store RED and GREEN logs,
- update `wiki/context/current.md`,
- add a dated devlog,
- update ADR-0007 status or add a new ADR after architecture approval.

Do not rewrite the onboarding report before the implementation stabilizes.

## 16. Commit Plan

Keep commits small enough to review independently.

Recommended sequence:

1. `Add conditional-graph distributed corruption regression (RED)`
2. `Synchronize whole-model gradients before optimizer updates`
3. `Handle tied, isbits, and legacy trainable traversal`
4. `Expand conditional distributed optimizer correctness tests`
5. `Convert conditional deadlock detector to a success regression`
6. `Document wrapper compatibility and CPU/MPI evidence`

Do not commit generated logs to the Flux repository. Store evidence in the thesis repository.

Do not push any implementation commit until the user requests a push.

## 17. Evidence Requirements

Create a new directory such as:

```text
artifacts/logs/conditional-wrapper/
```

Store these files:

- baseline two-rank suite log,
- RED silent-corruption log,
- focused GREEN two-rank log,
- focused GREEN four-rank log,
- complete two-rank suite log,
- complete four-rank suite log,
- updated detector log,
- ambiguity-check log,
- diff-check result,
- command metadata with commit IDs.

Each metadata record must contain:

- date and time,
- host,
- Flux commit,
- thesis commit or working-tree state,
- exact command,
- exit code,
- elapsed time,
- test totals.

Do not report a pass without a stored exit code or clear command result.

## 18. Stop Conditions

Stop implementation and report the result if one of these conditions occurs:

1. The model structure differs across ranks.
2. The implementation needs a private Optimisers.jl function.
3. The wrapper cannot preserve `Flux.train!` behavior.
4. The wrapper loses an immutable model value returned by `update!`.
5. A tied parameter needs different optimizer states for its occurrences.
6. Presence handling needs parameter gradients on the CPU.
7. A collective count differs across ranks.
8. A new method ambiguity appears.
9. The PR branch gains changes that conflict with the implementation.
10. Native NCCL work starts before CPU/MPI gates pass.

Do not hide a failure with a timeout increase. Do not disable scalar-indexing checks.

## 19. Review Questions After CPU/MPI Completion

The implementation review must answer these questions.

### 19.1 Flux wrapper fit

1. Is the `state.tree` compatibility cost acceptable?
2. Can all normal state operations delegate without surprising behavior?
3. Does Functors traversal preserve backend ownership and inner state movement?
4. Does serialization reconstruct the wrapper correctly?
5. Is outermost-only composition acceptable for the first release?

### 19.2 Optimisers.jl alternative

1. Does Optimisers.jl need a public pre-update gradient transformation hook?
2. Can that hook preserve the two-pass update contract?
3. Can Flux and Lux use the same hook without a distributed dependency in Optimisers.jl?
4. Can the hook expose the complete model, state tree, and gradient tree?
5. Can the hook preserve tied leaves without exposing private implementation details?

### 19.3 Distributed contract

1. Must all ranks have identical model and optimizer-state structures?
2. Must freezing decisions be identical across ranks?
3. Is a CPU/MPI presence collective acceptable for NCCL training?
4. Does global absence require exact Optimisers skip semantics for every rule?
5. Does the public API need a diagnostic for rank-structure mismatches?

These questions belong in the Carlo discussion after the wrapper produces evidence.

## 20. Acceptance Criteria

The CPU/MPI wrapper implementation is complete only when all criteria pass.

### Architecture

- Communication occurs before the inner Optimisers update.
- Communication remains outside AD.
- The implementation uses public Optimisers.jl APIs.
- The inner optimizer performs all optimizer-specific state transitions.
- Nested distributed optimizer composition fails clearly.

### Correctness

- Rank-dependent branches do not deadlock.
- Equal-shaped branches do not corrupt each other.
- Global means match analytical references.
- Globally absent gradients skip optimizer updates.
- Adam state matches the independent reference.
- Tied and isbits parameters work in the same model.
- Named `trainable` field order does not change occurrence-to-buffer mapping.
- Legacy tuple selection matches public native setup behavior.
- Non-trainable arrays do not receive collectives.
- Caller gradients remain unchanged.
- Immutable model returns propagate to the caller.

### Protocol

- Every rank issues collectives in the same order.
- Every rank issues the same collective count.
- Presence metadata uses one collective per step.
- Gradient collectives occur only for globally present buffers.

### Tests

- Focused two-rank conditional tests pass.
- Focused four-rank conditional tests pass.
- Complete two-rank distributed suite passes.
- Complete four-rank distributed suite passes.
- Updated external detector passes as a success regression.
- The ambiguity count is zero.
- `git diff --check` passes.

### Documentation

- The wrapper compatibility cost is documented.
- Known limitations are documented.
- Exact commands and results are stored.
- The current context and dated devlog are updated.

## 21. Known Limitations After Acceptance

Acceptance of the CPU/MPI implementation does not prove these properties:

- native NCCL execution,
- CUDA scalar-indexing safety under real hardware,
- good performance for large models,
- communication overlap,
- higher-order gradients,
- Enzyme `Duplicated` update support,
- nested distributed rule composition,
- stable public state shape,
- compatibility with every external Optimisers state consumer.

State these limits in every review summary.

## 22. Exact First Action

Start in `/home/kurapica/Projects/ddp_flux/flux-integration`.

Make sure that commit `cbdd3edc` (D1-D3) is present and that the working tree contains the corrected D4 changes (`src/distributed/public_api.jl` and `test/ext_distributed/conditional_distributedtest.jl`) with no conflicting changes.

If the active Optimisers version changed, read its source again. The D1-D4 evidence used Optimisers 0.4.9 (`/home/kurapica/.julia/packages/Optimisers/tMaaf`).

Review the corrected D4 selection (state tree, section 12 D4) and its RED2/GREEN2 evidence, then implement Phase D5: retain or add the focused contract assertions and run the focused conditional and optimizer files at two MPI ranks, followed by the complete dedicated two-rank suite.

Do not run the full local main suite. Four-rank gates remain Phase F. Commit or push only with explicit user authorization.

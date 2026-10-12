# 2026-09-29 - Conditional-graph wrapper Phase D4: legacy tuple-returning `trainable`

## Goal

Phase D4 of `temp/conditional_graph_wrapper_implementation_plan.md`: make the
gradient-synchronization walk select exactly the children that public native
`Optimisers.setup` selects for the legacy no-names tuple `trainable` API, in
model field order, including its value-membership rule, and preserve the native
legacy warning without repeating it on every distributed update.

## Defect

Optimisers 0.4.9 still supports old Flux-style `trainable` methods that return a
plain `Tuple` of selected fields. Its setup normalizes the selection before
walking the model (`Optimisers._trainable`, `src/interface.jl:186-196`):

- named selections merge into the functor children in model field order
  (`nothing` at unselected fields),
- the legacy no-names tuple selection maps each functor child `c` to
  `c in tr ? c : nothing`, i.e. selection is by **value membership**, not by
  identity, so equal-valued distinct fields are selected together,
- it emits a warning with `maxlog=3`.

The wrapper's selection predicate only knew the named/container shapes. For
`tr::Tuple` it fell through to `haskey(tr, k)` with a field-name `Symbol`, so the
first wrapper update of a legacy model failed with:

```text
MethodError: no method matching haskey(::Tuple{Vector{Float64}}, ::Symbol)
```

Even with a `haskey`-based workaround, an identity-only selection would pick
different parameters than native setup for equal-valued fields.

## Fix

> **Superseded.** This first fix (value-membership predicate reevaluated per update)
> failed an independent review for a second-update defect. The corrected fix is in
> the "Correction after P1 review" section at the end; the final code reads the
> selection from the state tree and `_trainable_child` no longer exists.

`src/distributed/public_api.jl` (uncommitted, on top of committed `cbdd3edc`):

- both traversal passes now call `_trainable_child(ch, tr, k)`, so the selection
  rule sees the parent's functor children together with the public
  `Optimisers.trainable(x)` result;
- the new method
  `_trainable_child(ch::NamedTuple, tr::Tuple, k) = ch[k] in tr` adds the legacy
  path with the exact native expression (`in`), no private Optimisers API and no
  identity test;
- the named/container methods keep their previous rules, reshaped to dispatch on
  `(ch, tr)`; the generic fallback is unchanged;
- both passes evaluate the predicate in model field order, so canonical order and
  occurrence-plan alignment hold for legacy models too;
- no warning is emitted in the synchronization walk: native setup already warns
  once; `_sync_gradients` stays silent on every update.

## Regression

`test/ext_distributed/conditional_distributedtest.jl` gained fixture
`LegacyTupleTrainable` and testset `legacy tuple trainable selection matches
native setup`:

- `a` and `b` are equal-valued, distinct mutable arrays; `excluded` is a distinct
  array with a different value; `Optimisers.trainable(m) = (m.a,)` is the legacy
  tuple selection. Native setup therefore gives leaves to `a` and `b` and none to
  `excluded`; the wrapper tree must be `==` to that native tree.
- The wrapper update is compared against a plain `Optimisers.update` reference
  built from explicit `avg` allreduces of the same gradients; `excluded` is
  asserted untouched by both paths.
- `RecordingBackend` asserts one presence reduction (`(2,)`, `Vector{Float32}`,
  `avg`) followed by one gradient reduction per selected buffer in model order
  (`a`, then `b`), distinguished by value because the shapes are equal, plus
  cross-rank count equality.
- `@test_logs` asserts the legacy warning around wrapper setup (emitted by native
  setup) and **no warnings at all** around the wrapper update.

## RED evidence

Run from `/home/kurapica/Projects/ddp_flux/flux-integration` against committed
`cbdd3edc` (2-argument `_trainable_child`):

```bash
FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no --project=test \
  test/ext_distributed/conditional_distributedtest.jl
```

Exit 1, 33 s. Per rank: control 5/5, D1 10/10, equal-valued guard 11/11,
D2 38/38, D3 16/16, D4 **12 passed / 3 failed / 5 errored / 20**. The first error
is the intended defect: `MethodError: no method matching
haskey(::Tuple{Vector{Float64}}, ::Symbol)` thrown by the wrapper update through
`_trainable_child`. The remaining failures are downstream of the same throw:
`model.a`/`model.b` stayed `[1.0, …]` (reference `[0.85, …]` / `[0.45, …]`),
`length(rec.log) == 3` evaluated `0 == 3`, and `rec.log` indexing raised
`BoundsError`. No fixture error, no dependency error.

Log: `phaseD-d4-RED-mpi2-2026-09-29.log`.

## GREEN evidence

Environment: Julia 1.12.6, Optimisers 0.4.9
(`/home/kurapica/.julia/packages/Optimisers/tMaaf`), host omarchy. Flux HEAD
remains `cbdd3edc`; D4 changes uncommitted.

| Check | Command | Result |
|---|---|---|
| Focused conditional, 2 ranks | same as RED | exit 0, 44 s: control 5/5, D1 10/10, guard 11/11, D2 38/38, D3 16/16, D4 20/20 per rank |
| Focused optimizer, 2 ranks | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl` | exit 0, 11 s, 6/6 per rank |
| Full 2-rank suite | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | exit 0, 65 s, `Distributed \| 6 6 1m01.5s` |
| Ambiguities | `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` | 0 (Optimisers 0.4.9) |
| Whitespace | `git diff --check` | clean (no output) |

## Scope notes

- D4 only. D5 (focused regression gate) remains open. Four-rank execution remains
  Phase F.
- No GPU/NCCL, detector, Bocconi, or main-suite runs. No mock or simulated value is
  used in the D4 testset.
- Optimisers 0.4.9 is the only version recorded; `_trainable` behavior for other
  versions was not verified in this phase.
- No commit. Tests and fix stay reviewable as separate changes in the worktree;
  commit/push only with explicit user authorization.

## Evidence

`artifacts/logs/conditional-wrapper/`:

- `phaseD-d4-metadata.md`,
- `phaseD-d4-RED-mpi2-2026-09-29.log`,
- `phaseD-d4-GREEN-mpi2-2026-09-29.log`,
- `phaseD-d4-GREEN-optimizer-mpi2-2026-09-29.log`,
- `phaseD-d4-GREEN-suite-nprocs2-2026-09-29.log`,
- `phaseD-d4-ambiguities-2026-09-29.log`,
- `phaseD-d4-diff-check-2026-09-29.log`.

## Next action

Phase D5: add the focused regression gate for the retained contracts (BatchNorm
running statistics, Transpose/Adjoint parent-space pairing, caller-gradient
non-mutation, immutable model returns, globally absent skip semantics, locally
absent global mean, empty trainable models, array container shape, Phase C
composition guards), then run the complete dedicated two-rank suite. Request user
authorization before committing D4.

## Independent review: D4 reopened

The review found one blocking correctness defect (P1) at `src/distributed/public_api.jl:663`.
Both traversal passes reevaluate legacy value membership for every update.
Native Optimisers evaluates that membership during setup and retains the resulting state tree during updates.

The existing fixture starts with distinct arrays `a == b`, so native setup selects both fields.
Different gradients make their values unequal after the first update.
The wrapper then excludes `b` from the second update, despite its existing `Optimisers.Leaf`.
Native Optimisers continues to update `b`.

An independent two-rank run reproduced this defect for `Optimisers.update!` and `Optimisers.update`.
Each path produced these values:

| Step | Wrapper `b[1]` | Native `b[1]` | Wrapper collectives |
|---|---|---|---|
| 1 | 0.45 | 0.45 | 3 |
| 2 | 0.45 | -0.10 | 2 |

The existing conditional file still passed all testsets: 5, 10, 11, 38, 16, and 20 assertions per rank.
The added characterization passed 16 assertions per rank because it explicitly asserts the defect.
Exit 0 therefore records successful reproduction, not D4 correctness. Elapsed time: 29.042 seconds.
The stored full-suite log supports the reported 6/6 result, but the suite lacks this second-step assertion.
Flux `git diff --check` passed. Flux source and tests remain unchanged by this review.
Source state: `cbdd3edcb5fdeaa791968768a5d7c660e5817244` plus the uncommitted D4 diff.

Exact command from `/home/kurapica/Projects/ddp_flux/flux-integration`:

```bash
set -o pipefail && { time FLUX_TEST_DISTRIBUTED_BACKEND=mpi /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 julia --startup-file=no --project=test /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/logs/conditional-wrapper/phaseD-d4-review-multistep.jl; } 2>&1 | tee /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/logs/conditional-wrapper/phaseD-d4-review-mpi2-2026-09-29.log
git diff --check && git status --short --branch && git rev-parse HEAD
```

Next action: preserve setup-time selection through the existing state tree or explicit structural metadata.
Add a multi-step native-reference regression for both update paths, including collective counts and optimizer state.
Then run the D1-D4 regressions before D5.

## Correction after P1 review

The review's suggested correction is implemented in the same uncommitted working tree
(Flux HEAD still `cbdd3edc`). The walk no longer computes a `trainable` predicate at
all: it pairs the model with the state tree returned by `Optimisers.setup`, which is
the immutable record of the setup-time selection.

### Final design

- `_sync_gradients(backend, tree, model, grads)` takes the inner state tree and passes
  it to both passes. `Optimisers.update!`/`update` pass `s.tree`.
- `_walk!`/`_build!` visit the model's functor children in canonical order and look up
  the corresponding tree node through the shared structural `_child` helper (the former
  `_child_grad`, now used for gradient nodes and tree nodes alike):
  - a numeric child is trainable iff its tree node is an `Optimisers.Leaf`; otherwise it
    is skipped (not selected at setup);
  - an empty tree node (`()`) means setup selected no leaf in that subtree and the
    subtree is skipped; otherwise the children are paired recursively;
  - the child's gradient node is read with the same helper, so the D1-D3 pairing
    semantics (`nothing`/Zero handling, Transpose/Adjoint, Dict/Array/Tuple containers)
    are unchanged.
- `_trainable_child` and its value-membership rule are deleted. The state tree cannot
  drift as parameter values change, so a second (or any later) update keeps exactly the
  leaves setup chose. No warning is emitted in the walk.
- The D3 test, which calls `_sync_gradients` directly, now builds
  `tree = Optimisers.setup(Optimisers.Descent(1.0), model)` and passes it.

### RED2 (corrected tests on the unchanged buggy source)

The new testset `legacy tuple trainable selection persists across updates` runs two
steps through both update paths with stateful `Momentum`, distinct per-rank/per-step
gradients, and a native global-mean reference. Exit 1, 37.8 s; prior testsets unchanged
(5/5, 10/10, 11/11, 38/38, 16/16, 20/20); new testset **24 passed / 6 failed / 30** per
rank, 3 intended failures per path at step 2:

| Assertion (step 2) | Wrapper (buggy) | Native reference |
|---|---|---|
| `model.b` | `[0.55, …]` | `[-0.605, …]` |
| momentum state `state.tree.b.state` | `[0.45, …]` | `[1.155, …]` |
| collectives | 5 | 6 |

Log: `phaseD-d4-RED2-mpi2-2026-09-29.log`.

### GREEN2

Exit 0: focused conditional 48 s (control 5/5, D1 10/10, guard 11/11, D2 38/38, D3
16/16, D4 20/20, D4-review 30/30 per rank); focused optimizer 6/6, 10 s; full dedicated
2-rank suite `Distributed | 6 6 1m12.0s`, 74 s; ambiguities 0; `git diff --check` clean.
Logs: `phaseD-d4-GREEN2-*`, `phaseD-d4-RED2-*`, and rerun `phaseD-d4-ambiguities-*` /
`phaseD-d4-diff-check-*`. Final uncommitted D4 diff: 2 files, +244/−64 (`src` +62/−63,
test +182/−1).

The general contract this establishes is stronger than the legacy case alone: any
selection decision that optimizers make at setup time is honored for the whole
training run, because the wrapper defers to the native state tree instead of
re-deriving selection from mutable state.

Next action: Phase D5 (focused regression gate for the retained contracts, then the
complete dedicated two-rank suite). Request user authorization before committing D4.

## Independent follow-up review

The corrected D4 diff has no blocking findings. The setup state tree now controls selection in both synchronization passes.
Both public update paths pass `s.tree` to the synchronization function.
The change removes value-dependent reselection and preserves canonical model order.
The new Momentum regression covers both update paths, parameter values, optimizer state, and collective counts across two steps.
This closes the previous P1 finding.

The independent two-rank run passed all seven testsets: 5, 10, 11, 38, 16, 20, and 30 assertions per rank.
Exit code: 0. Elapsed time: 31.201 seconds. Optimisers version: 0.4.9.
The ambiguity check returned zero. Flux `git diff --check` passed.
The stored GREEN2 full-suite log supports the reported 6/6 result. This review reran the focused conditional file only.

Exact commands from `/home/kurapica/Projects/ddp_flux/flux-integration`:

```bash
set -o pipefail && { time FLUX_TEST_DISTRIBUTED_BACKEND=mpi /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl; } 2>&1 | tee /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/logs/conditional-wrapper/phaseD-d4-review2-mpi2-2026-09-29.log
git diff --check && julia --startup-file=no --project=test -e 'using Flux, Test, Optimisers; println("Optimisers=", pkgversion(Optimisers)); a = Test.detect_ambiguities(Flux.DistributedUtils; recursive=false); println("ambiguities=", length(a)); @assert isempty(a)'
```

The earlier `phaseD-d4-review-multistep.jl` artifact asserts the old defect and remains historical evidence.
The new permanent regression asserts correct behavior.
Flux source and tests remain unchanged by this review.

Next action: implement the Phase D5 regression gate and run the complete dedicated two-rank suite.

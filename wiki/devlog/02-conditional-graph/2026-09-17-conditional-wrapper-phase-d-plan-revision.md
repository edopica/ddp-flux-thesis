# 2026-09-17 - Phase D plan revision

## Scope

The user requested a Phase D plan revision after approval of the Phase C composition correction.
This session changed documentation only. It did not change Flux source or tests.
Phase D is planned, not implemented. Flux remains at `f5b87975`.

## Design

The revised plan requires both traversal passes to use the same normalized model order.
Named `trainable` order cannot change occurrence order. An occurrence list alone does not enforce this requirement.

The active reference is Optimisers 0.4.9 at `/home/kurapica/.julia/packages/Optimisers/tMaaf/src/interface.jl`.
Lines 186-195 describe named selection and legacy tuple selection.
Named selection retains model-field order. Legacy tuple selection uses value membership, not identity membership.
The implementation must use public APIs rather than call these private reference methods.

## Sequence

1. D1: reversed-order named `trainable` regression and canonical traversal order.
2. D2: mixed tied/isbits regression and explicit occurrence-to-buffer mapping.
3. D3: device-like metadata regression and CPU presence allocation.
4. D4: legacy tuple regression and native-compatible field selection.
5. D5: focused contract checks, two-rank suite, ambiguity check, and whitespace check.

Each correction requires focused RED/GREEN evidence. Earlier corrections can also fix later regressions.
In that case, the plan requires separate baseline RED evidence rather than an artificial defect.

The plan retains four-rank gates in Phase F. Mock presence fractions do not replace actual four-rank evidence.
Native GPU/NCCL work remains deferred. No performance work enters Phase D.

## Consistency

Updated algorithm requirements, defect descriptions, the test matrix, acceptance criteria, and the implementation entry point.
Phase E now retains and expands Phase D regressions rather than deferring their first implementation.
Corrected the false ancestry claim about `fee2c310` and `3370b910`.
Marked Phase A branch commands as historical. They must not run again.
Updated the README next action and current context. No progress-table status changed.

## Validation

Reviewed the revised sequence, algorithm requirements, test matrix, and entry instructions for consistency.

These whitespace commands passed with exit code 0 from the thesis repository:

```bash
git diff --check -- README.md wiki/context/current.md
git diff --no-index --check -- /dev/null temp/conditional_graph_wrapper_implementation_plan.md
git diff --no-index --check -- /dev/null wiki/devlog/02-conditional-graph/2026-09-17-conditional-wrapper-phase-d-plan-revision.md
```

`git status --short --branch` in the Flux worktree showed a clean `ddp/integration-m1-spike` branch.
`git rev-parse --short HEAD` returned `f5b87975`.

No Julia, MPI, remote, or GPU tests ran for this documentation-only revision.

## Next Action

At Flux baseline `f5b87975`, record the active Optimisers version and source path.
Then add the Phase D1 reversed-order named `trainable` regression.
Record its numerical RED result before production changes.

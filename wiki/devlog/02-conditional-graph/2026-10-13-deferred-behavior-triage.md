# Deferred-behavior scope discussion

Date: 2026-10-13

The user requested a discussion after a tutor meeting.
The goal is to fix important, bounded defects and track other work through GitHub issues.

## Evidence and provisional recommendation

The discussion used audit 002 and the current source in `../flux-integration`.
`src/train.jl:375-378` still contains the rank-local loss guard.
`src/distributed/public_api.jl:698-737` still synchronizes gradients without finiteness checks.

- Fix DB-1 and DB-4 together through a coordinated loss decision before the local update guard.
- Prefer an error on every rank over silent skipping for this initial fix.
- Discuss DB-3 separately. Gradient checks require device-aware reductions and a clear API policy.
- Include focused NCCL coverage for any fix that claims GPU support.
- Defer general unequal-participation handling and optional configuration or structure diagnostics.
- Retain the accepted checkpoint contract and the correct DB-2 behavior.
- Clarify the auxiliary-output contract and unsafe workarounds in the documentation.

Two opinion reviewers returned assessments. They agreed on DB-1/DB-4 priority but disagreed on immediate DB-3 scope.
The third reviewer was unavailable because its configured model was not found.
No shell command, test, remote job, source edit, or GitHub issue creation occurred.
The user has not accepted a final scope.

## Exact next action

Agree on the DB-1/DB-4 error policy and whether DB-3 enters the immediate fix batch.

## Follow-up: detailed plans requested

The user emphasized DB-3's silent corruption and requested detailed plans for both fixes.
The revised recommendation includes DB-3 with a mandatory post-reduction safeguard.

Plans:

- [DB-1/DB-4](../../../reports/conditional-graph-db1-db4-fix-plan.md).
- [DB-3](../../../reports/conditional-graph-db3-fix-plan.md).

Source inspection established the training-module include order, the common wrapper update boundary, and MPI's sum-before-division implementation.
The historical audit driver expects defect signatures, so the plans specify a separate bounded success-regression harness.
Its command interface is proposed and does not yet exist.
The plans require device-stream ordering checks and scalar-indexing-disabled NCCL cases.
No implementation, numerical test, or remote job occurred in this planning follow-up.

Commands run from the thesis root:

```bash
git status --short
git diff --check
```

The status command succeeded and showed existing documentation changes and untracked artifacts.
Those artifacts remain untouched.
`git diff --check` passed for tracked changes. The new plans also received a source-consistency and wording review.

**Revised exact next action:** record the Flux worktree diff, then write the DB-1/DB-4 ADR and bounded asymmetric-loss success regression.

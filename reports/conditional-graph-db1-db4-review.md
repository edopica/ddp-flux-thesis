# DB-1/DB-4 implementation review

Date: 2026-10-13. Follow-up review: 2026-10-14.

## Final review verdict — 2026-10-14

**R1, R2, and the NamedTuple coverage gap are closed.**
No blocking findings remain for the documented DB-1/DB-4 eager-path scope.

The final driver reads NUL-separated argv elements from `/proc/<pid>/cmdline`.
It requires a separate probe filename argument and the exact adjacent pair `("--campaign", token)`.
It does not interpret token characters as regex syntax or accept token prefixes.
Both termination passes select PIDs through this exact matcher.
The header documents the uniqueness requirement for operator-supplied tokens.

The independent local cleanup selftest passed:

```text
SELFTEST_CLEANUP PASS prefix_ok=true metachar_ok=true final_empty=true
```

This closes the reproduced prefix and metacharacter collisions.
The final driver hash is `49c7e0aacdb48661f044231b1fc6c5f1`.
The probe, dedicated test file, and production hashes remain unchanged and match the recorded identity table.

The inspected `green3-*` summaries report MPI 2 ranks 13/13, MPI 4 ranks 8/8, and NCCL 2 ranks 12 PASS plus 1 UNTESTED.
The previous suite results remain applicable because the final correction changes only the external driver.
R2 and NamedTuple closure retain the evidence from the v2 follow-up.

Remaining exclusions are unchanged: NCCL `Duplicated` is untested, and DB-3 plus combined `--group all` acceptance remain pending.
The fix does not roll back forward-pass side effects or cover Reactant's compiled path.

**Exact next action:** decide whether to implement the separate DB-3 gradient guard before combined `--group all` acceptance.

## V2 follow-up verdict (superseded by final closure)

R2 and the minor NamedTuple coverage gap are closed.
R1 remains P2 because campaign selection does not match tokens literally or require a complete argument match.
The production assessment remains unchanged: no correctness defect found within the stated eager-path contract.

### R1 remains open: token prefix and regular-expression collisions

Current location: `scripts/checks/conditional_graph_numerical_guard_check.jl:53-75`.

`campaign_pattern(token)` inserts the token directly into a regular expression without a trailing argument boundary.
Cleanup for `campA` therefore also selects `campA-rerun`.
The documented environment override accepts arbitrary strings, so `camp.1` also matches `campX1`.
Both `pkill` calls use this pattern, including cleanup at normal campaign completion.

A local reproduction started two synthetic children with distinct tokens, `review-<unique>` and `review-<unique>-rerun`.
The actual `campaign_pids` function selected both PIDs for the first token:

```text
PREFIX_ISOLATION_REPRO crossmatch=true own_pid=247945 other_pid=247946 selected=[247945, 247946]
```

The reproduction terminated only its own children through their process handles.
It did not call `reap_campaign` or send pattern-based signals.
The saved selftest uses non-prefix tokens, so its PASS result does not detect this defect.
The Hydra cleanup explanation for the concurrent campaign remains valid but does not resolve token matching.

Required correction:

1. Match the campaign as a complete literal argument, or use owned process groups.
2. If regex selection remains, escape or restrict token characters and require argument boundaries.
3. Add isolation cases for prefix-related tokens and tokens with regex characters.
4. Require user-supplied tokens to be unique across concurrent campaigns.

### R2 closed: conditional eager trajectory

Current locations: `conditional_graph_numerical_guard_probe.jl:166-218,262-285,620-665`, under `scripts/checks/`.

The new case trains rank-distinct branches through eager `train!`.
It compares all parameters, both Adam moments, and decay products against the native reference after each step.
Its schedule includes global absence after active steps and subsequent reactivation.
The driver requires per-rank state, absence, and reactivation markers, plus the expected ordered collective metadata.
The saved MPI 2/4 and NCCL 2 logs contain every required marker with `ok=true`.

### NamedTuple coverage closed

Location: `../flux-integration/test/ext_distributed/nonfinite_loss_distributedtest.jl:289-328`.

The new test covers an outer NamedTuple on both the finite and asymmetric invalid paths.
It checks the return structure, update, rejection, one-vote schedule, and parameter/state snapshots.
The saved v2 suites pass at MPI 2/4 ranks and NCCL 2 ranks.

### Follow-up evidence

All five inspected hashes match the v2 identity table in `artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md`.
These include both production files, the probe, the driver, and the dedicated test file.
The v2 suite logs report 8/8, 8/8, and 16/16 files passed.
This follow-up ran local cleanup-selection reproductions only, with no new distributed training or remote execution.
Exact commands: `wiki/devlog/02-conditional-graph/2026-10-14-db1-db4-v2-review.md`.

**Exact next action:** make campaign selection literal and argument-exact, then run the expanded cleanup isolation selftest.

## Original review verdict (2026-10-13)

The production hook matches the planned design. This review found no production correctness defect within the stated eager-path contract.
Two P2 findings require corrections in the harness and acceptance coverage before the implementation receives full plan acceptance.
The saved GREEN results remain valid for the cases that ran.

Reviewed Flux worktree: `../flux-integration`, HEAD `55171e659b2f425da980cbc83177bf5c38f0a7a1`, with uncommitted changes.
Source and test hashes match the GREEN column in `artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md`.

## Findings

### R1 — P2: cleanup can terminate another campaign

Location: `scripts/checks/conditional_graph_numerical_guard_check.jl:154-170,467-468`.

`reap_orphans()` selects processes by the probe filename through `pgrep -f` and `pkill -f`.
It does not restrict the selection to descendants or the process group of this driver.
The driver calls this function after a timeout and at normal completion.

If two campaigns overlap under the same account, the first completed campaign can terminate the other campaign's MPI children.
Different backends and output directories provide no isolation because both campaigns use the same probe filename.
The subsequent `pkill -9` can also terminate a probe that started during the one-second delay.

Required correction:

1. Track each launched job through an owned process group or another job-specific identifier.
2. Restrict termination and orphan cleanup to that job.
3. Add an isolation check with two concurrent jobs, including a timeout in one job.

This finding comes from source inspection. The review did not execute the broad cleanup against concurrent processes.

### R2 — P2: the healthy gate omits the required conditional trajectory

Location: `scripts/checks/conditional_graph_numerical_guard_probe.jl:524-543` and `:144-145`.

The plan requires finite conditional training with per-step parameter and Adam-state comparisons (section 4.2).
Its NCCL gate also requires a healthy conditional reference case (section 6, Phase C).

`run_healthy()` uses the same objective on every rank. Its loss always uses all three fields, including both tied occurrences.
It runs all four distributed steps before comparison with the native reference.
The comparison covers parameters, first moments, and decay products, but omits second moments.
Thus, its PASS result does not establish the required conditional trajectory or complete per-step Adam-state agreement.

The existing conditional suites do cover branch absence and full Adam state through explicit `withgradient` plus `Optimisers.update!`.
For example, `../flux-integration/test/ext_distributed/cuda_conditional_distributedtest.jl:148-205` uses that path.
Those calls bypass `_loss_update_allowed`, so they do not close this integration gap.
The new dedicated loss tests use a single trainable parameter and also do not close it.

Required correction:

1. Add a finite conditional case through `trainstep!` or `trainstep_withgradient!`.
2. Include rank-distinct branches, global absence after nonzero Adam history, and reactivation.
3. Compare parameters, both moments, and decay products against an independent reference after every step.
4. Assert the loss vote, presence exchange, and applicable gradient reductions in order.
5. Run the case at two and four MPI ranks, then two native NCCL ranks.

## Minor coverage note

The auxiliary cases return an outer `Tuple` with a nested `NamedTuple`.
They do not cover an outer `NamedTuple` whose first field is the scalar objective.
The plan names both return forms. A focused distributed case for each validity outcome will close this small gap.
The shared `_loss_scalar` implementation already handles both types, so this is not evidence of a production defect.

## Production assessment

- `src/train.jl:385-402`: the hook runs after differentiation and before `_update!` for both eager step APIs.
- The default hook retains the previous local `isfinite` decision for ordinary optimizer states.
- `src/distributed/public_api.jl:703-723`: one host `Float32` flag uses `avg` with an any-positive threshold.
- Every rank throws `DomainError` after an invalid vote, before presence exchange or gradient reduction.
- The vote does not depend on the number of trainable buffers.
- The existing update path retains the `Duplicated` model unwrapping.
- NCCL host metadata uses the existing MPI fallback in `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl:70-73`.
- The `trainstep!` documentation states the eager-path and forward-side-effect boundaries.

Flux paths in this section are relative to `../flux-integration`.

## Evidence assessment

The saved summaries and suite logs support the reported counts:

| Recorded gate | Result |
|---|---|
| Loss harness, MPI 2 ranks | 13 PASS |
| Loss harness, MPI 4 ranks | 8 PASS |
| Loss harness, NCCL 2 ranks | 12 PASS, 1 UNTESTED |
| Dedicated suite, MPI 2 / 4 ranks | 8 / 8 files PASS at each rank count |
| Dedicated suite, MPI plus NCCL | 16 / 16 files PASS |
| Single-process train suite | 137 / 137 PASS |
| Non-recursive DistributedUtils ambiguity scan | 0 |

The raw four-rank asymmetric log shows rank 2 as the sole offender.
All four ranks complete collective 9, catch `DomainError`, and retain their snapshots after two valid updates.
The raw NCCL healthy log shows host vote/presence buffers and device gradient buffers.
These inspected logs support the protocol and backend claims for their actual scenarios.

The NCCL `Duplicated` route remains explicitly UNTESTED after its availability timeout.
DB-3 and the combined `--group all` gate remain pending by design.
Neither limitation is a newly discovered defect in this review.

An independent local two-rank attempt failed during `using Flux`, before any test assertion.
Julia reported a CUDA dependency `KeyError` from `Base.Precompilation.scan_deps!`.
The review therefore adds no new passing runtime evidence.
The exact command and failure are in `wiki/failures/2026-10-13-db1-db4-review-local-loading.md`.
Both repositories passed `git diff --check` before review documentation edits.

## Original next action (superseded by the v2 follow-up)

Correct the job-scoped cleanup in R1, then add and run the conditional eager-path acceptance case in R2 before DB-3 work.

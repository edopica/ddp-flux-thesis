# Devlog index

Chronological development log, grouped by project scope. Entries are dated and
kept in place; the indexes below list them newest-first within each scope.

| Scope | Period | Folder | Contents |
|---|---|---|---|
| 1. PR #2694 | 2026-07-02 → 2026-09-16 | [`01-pr2694/`](01-pr2694/) | C0–C8 upstream DDP work: audit, reference loop, launch, model sync, data sharding, gradient/optimizer sync, end-to-end example, correctness battery; M0/M1 integration spike; PS1–PS8 PR #2694 salvage; Windows routing fix; early remote and documentation tooling of that era |
| 2. Conditional graph | 2026-09-14 → present | [`02-conditional-graph/`](02-conditional-graph/) | Conditional-graph onboarding; `DistributedOptimizerState` wrapper Phases A–K; native CUDA/NCCL validation; Leonardo bring-up; timing stretch; notebooks; implementation and deferred-behavior reports |
| 3. Static-network GPU scaling | 2026-10-15 → present | [`03-static-scaling/`](03-static-scaling/) | Tiny ImageNet-200/PyTorch benchmark environment setup for the static-network Flux-vs-PyTorch scaling proposal; the 2026-10-15 scaling-discussion entries remain in scope 2 |

An entry belongs to the scope of the work it records, not to the date alone.
Remote/infrastructure entries follow the project they supported: the
2026-09-03 precompile experiment (C8/PR testing) is in scope 1; the Leonardo
bring-up and remote-tooling entries (Phase J/K support) are in scope 2.
The 2026-10-15 scaling-discussion entries (proposal drafting) remain in scope
2; scope 3 starts with the benchmark environment setup.

## Scope 1 — PR #2694 (upstream DDP)

- [2026-09-16](01-pr2694/2026-09-16-pr2694-windows-routing-fix.md) — Windows-only routing-test failure fixed with direct AST argument equality; PR head `fee2c310`.
- [2026-09-10](01-pr2694/2026-09-10-pr2694-review-fixes.md) — Phases 2–7 code-review fixes: NCCL `force` guard bypass, main-suite distributed opt-in routing, padding wording, in-scope corrections.
- [2026-09-10](01-pr2694/2026-09-10-pr2694-salvage-phase7-verification.md) — Phase 7 verification gates: all six PASS, including OpenMPI topology emulation via rootless podman.
- [2026-09-10](01-pr2694/2026-09-10-pr2694-salvage-phase6-docs-consistency.md) — Phase 6 docs & consistency sweep (`gpu.md`, `NEWS.md`, typo, hygiene).
- [2026-09-10](01-pr2694/2026-09-10-rootless-podman-ci-parity.md) — Rootless-podman CI parity: hwloc topology-file emulation for the OpenMPI slot condition.
- [2026-09-09](01-pr2694/2026-09-09-pr2694-salvage-phase5-polish-carlo-tests.md) — Phase 5: CI labels and harness wording polish.
- [2026-09-09](01-pr2694/2026-09-09-pr2694-salvage-phase4-cyclic-data-padding.md) — Phase 4: cyclic data padding in `DistributedDataContainer` (RED + GREEN).
- [2026-09-09](01-pr2694/2026-09-09-pr2694-salvage-phase3b-pmi-guardrail.md) — Phase 3b: narrowed PMI guardrail implementation (GREEN).
- [2026-09-09](01-pr2694/2026-09-09-pr2694-salvage-phase3a-launcher-guard-tests.md) — Phase 3a: PMI launcher-guard decision-matrix tests (RED baseline).
- [2026-09-09](01-pr2694/2026-09-09-pr2694-salvage-phases-1-2.md) — Phases 1–2: salvage worktree setup and scope-reset commit.
- [2026-09-08](01-pr2694/2026-09-08-m1.1-corrective-spike.md) — M1.1 corrective spike after the M1 gate reopen (7 blocking findings fixed).
- [2026-09-08](01-pr2694/2026-09-08-m0-m1-integration-spike.md) — M0/M1: integration workspace isolation and automatic unused-gradient spike.
- [2026-09-03](01-pr2694/2026-09-03-precompile-experiment.md) — Bocconi compute-only precompile workflow (see [ADR-0005](../decisions/ADR-0005-precompile-compute-only.md)).
- [2026-08-17](01-pr2694/2026-08-17-pr2694-restore.md) — PR #2694 frozen-branch restore and rebase onto upstream master.
- [2026-08-01](01-pr2694/2026-08-01.md) — C8.3 NCCL verification on Bocconi.
- [2026-07-31](01-pr2694/2026-07-31.md) — C8.2 finalization, Julia 1.12 fixes, full MPI suite.
- [2026-07-30](01-pr2694/2026-07-30.md) — C8.2 collective-invariant tests for `bcast!`/`reduce!`/`allreduce!`.
- [2026-07-29](01-pr2694/20260729-c8-matrix-and-harness.md) — C8 correctness matrix and distributed test harness.
- [2026-07-29](01-pr2694/20260729-branch-cleanup.md) — DDP branches merged into master; `upsteam-pr` preserved.
- [2026-07-28](01-pr2694/20260728-c7-fixes.md) — C7 blindspot fixes and Bocconi GPU validation.
- [2026-07-22](01-pr2694/20260722-c7-end-to-end.md) — C7 end-to-end example and upstream distributed guide.
- [2026-07-16](01-pr2694/20260716-c6b-hpc-tests.md) — C6b: polished upstream PR commits and Bocconi cluster tests.
- [2026-07-16](01-pr2694/20260716-wiki-rename.md) — `openwiki` → `wiki` rename and terminology update.
- [2026-07-15](01-pr2694/20260715-conditional-testing.md) — Enhanced conditional-graph testing for the PR-era helper.
- [2026-07-14](01-pr2694/20260714-openwiki-manual-update.md) — Manual wiki status update.
- [2026-07-13](01-pr2694/20260713-c5-c6-grad-sync.md) — C5/C6 gradient and optimizer-state synchronization.
- [2026-07-13](01-pr2694/20260713-c4-data-sharding.md) — C4 distributed data sharding and padding fix.
- [2026-07-11](01-pr2694/20260711-c3-sync-model.md) — C3 model broadcast and parameter verification.
- [2026-07-10](01-pr2694/2026-07-10.md) — PMI mismatch guardrails.
- [2026-07-09](01-pr2694/20260709-c2-launch.md) — C2 distributed launch skeleton.
- [2026-07-09](01-pr2694/20260709-ssh-refactor.md) — SSH workflow refactoring and deathstar C2 verification.
- [2026-07-09](01-pr2694/20260709-deathstar.md) — Deathstar node setup.
- [2026-07-07](01-pr2694/2026-07-07.md) — C1 deterministic single-process reference loop.
- [2026-07-07](01-pr2694/2026-07-07-remote.md) — Remote execution workflow setup.
- [2026-07-05](01-pr2694/2026-07-05.md) — C0 audit continuation (DistributedUtils, CPU/MPI smoke).
- [2026-07-02](01-pr2694/2026-07-02.md) — C0 audit kickoff.

## Scope 2 — Conditional graph

- [2026-10-14](02-conditional-graph/2026-10-14-db3-review-response.md) — DB-3 review response complete: R1 pins the collective protocol per bundle (5/7 legacy, 6/8 guarded with the trailing vote) and passes on deathstar; R2 corrects the focused healthy reference (59/59 per rank GREEN, old-weighting RED control); R3 corrects the public contract documentation with no executable line changed.

- [2026-10-14](02-conditional-graph/2026-10-14-db3-gradient-rejection.md) — DB-3 post-reduction non-finite-gradient rejection implemented and gate-accepted (ADR-0009); RED retained, GREEN at 2/4 MPI ranks and 2 NCCL ranks; harness/test defects fixed.

- [2026-10-14 review](02-conditional-graph/2026-10-14-db1-db4-v2-review.md) — R2 and NamedTuple coverage closed. R1 was closed in round 2 by exact-argv campaign matching and the extended isolation selftest ([fix devlog](02-conditional-graph/2026-10-13-db1-db4-coordinated-nonfinite-loss.md#review-response-r1-round-2), `green3` re-gates).

- [2026-10-13 review](02-conditional-graph/2026-10-13-db1-db4-review.md) — DB-1/DB-4 production hook matches the plan; the two P2 harness/coverage corrections are implemented and re-gated in the fix devlog's review response (v2).

- [2026-10-13](02-conditional-graph/2026-10-13-db1-db4-coordinated-nonfinite-loss.md) — DB-1/DB-4 coordinated non-finite-loss fix: hook and loss vote implemented, GREEN at 2/4 MPI ranks and 2-rank NCCL; audit resolution; review-response v2 re-gate (campaign-scoped cleanup, conditional healthy reference, NamedTuple coverage).
- [2026-10-13](02-conditional-graph/2026-10-13-numerical-guard-red-harness.md) — DB-1/DB-4 numerical-guard RED harness: 13/13 expected failures on the unmodified source.
- [2026-10-13](02-conditional-graph/2026-10-13-deferred-behavior-triage.md) — Deferred-behavior scope triage: DB-1/DB-4 bounded fix proposed, DB-3 separate.
- [2026-10-07](02-conditional-graph/2026-10-07-wiki-reorganization.md) — Repo-wide wiki maintenance: devlog/audits grouped by scope; links rewritten; stale facts corrected.
- [2026-10-06](02-conditional-graph/2026-10-06-timing-corrections.md) — Phase K timing corrections and Leonardo re-measurement (training-only 1.888594, total 1.397335).
- [2026-10-06](02-conditional-graph/2026-10-06-phase-k-workflow-notebook-implementation.md) — Distributed workflow notebook implemented and executed in artifact mode.
- [2026-10-06](02-conditional-graph/2026-10-06-phase-k-two-notebook-plan.md) — Two-notebook plan (workflow tutorial + benchmark).
- [2026-10-06](02-conditional-graph/2026-10-06-phase-k-review-notebook.md) — Phase K implementation review and GPU recap notebook outline.
- [2026-10-06](02-conditional-graph/2026-10-06-phase-k-network-figure.md) — K2 network/routing figure with provenance metadata.
- [2026-10-06](02-conditional-graph/2026-10-06-deferred-behaviors-report.md) — Deferred-behavior report (audit 002): 8/8 modes, 83 assertions.
- [2026-10-06](02-conditional-graph/2026-10-06-conditional-graph-implementation-report.md) — Implementation report from per-leaf failure to tested wrapper.
- [2026-10-05](02-conditional-graph/2026-10-05-phase-k-timing-stretch.md) — Historical timing stretch: 1×A100 vs 4×A100 NCCL (instrumented 1.480203, superseded).
- [2026-10-05](02-conditional-graph/2026-10-05-phase-k-visualization-plan.md) — Reference plots and live-chart stretch plan.
- [2026-10-05](02-conditional-graph/2026-10-05-phase-k-k2-mnist-implementation.md) — K2 task-conditioned MNIST/Fashion-MNIST implementation and evidence.
- [2026-10-05](02-conditional-graph/2026-10-05-phase-k-k2-readiness.md) — Deathstar readiness probe before the K2 campaign.
- [2026-10-05](02-conditional-graph/2026-10-05-leonardo-k1-k2-4rank-nccl.md) — Leonardo K1/K2 CUDA/NCCL at 4 ranks; required matrix complete.
- [2026-10-04](02-conditional-graph/2026-10-04-conditional-wrapper-phase-k-toy-implementation.md) — K1 toy implementation and CPU/NCCL evidence.
- [2026-10-04](02-conditional-graph/2026-10-04-conditional-wrapper-phase-k-prompt.md) — Phase K isolated implementation prompt.
- [2026-10-04](02-conditional-graph/2026-10-04-leonardo-node-readiness.md) — Leonardo node readiness check.
- [2026-10-04](02-conditional-graph/2026-10-04-leonardo-bringup-4rank-nccl.md) — Leonardo bring-up; 4-rank NCCL weighting case closed.
- [2026-10-04](02-conditional-graph/2026-10-04-conditional-wrapper-phase-j-native-gpu.md) — Phase J native CUDA/NCCL validation complete at 2 ranks.
- [2026-10-04](02-conditional-graph/2026-10-04-conditional-wrapper-phase-j-native-gpu-attempt.md) — Phase J implementation and first GPU attempt (harness issue).
- [2026-10-02](02-conditional-graph/2026-10-02-conditional-wrapper-phase-i-planning.md) — Phase I agenda and post-1.0 performance goals.
- [2026-10-02](02-conditional-graph/2026-10-02-conditional-wrapper-phase-h-review.md) — Independent Phase H review; no blocking findings.
- [2026-10-01](02-conditional-graph/2026-10-01-conditional-wrapper-phase-h-compatibility.md) — Phase H compatibility review and corrections.
- [2026-09-30](02-conditional-graph/2026-09-30-remote-tooling-bocconi-rename.md) — Remote tooling cleanup, `hpc` → `bocconi` rename, per-server docs.
- [2026-09-30](02-conditional-graph/2026-09-30-conditional-wrapper-phase-g-detector.md) — Phase G: detector converted to a success regression.
- [2026-09-30](02-conditional-graph/2026-09-30-conditional-wrapper-phase-f-cpu-mpi-gates.md) — Phase F CPU/MPI gates GREEN at 2 and 4 ranks on deathstar.
- [2026-09-30](02-conditional-graph/2026-09-30-conditional-wrapper-phase-e-m11-battery.md) — Phase E: M1.1 battery restored, review-corrected; committed `55171e65`.
- [2026-09-30](02-conditional-graph/2026-09-30-conditional-wrapper-phase-d5-regression-gate.md) — Phase D5 regression gate; array-container shape defect found and fixed.
- [2026-09-29](02-conditional-graph/2026-09-29-conditional-wrapper-phase-d4-legacy-trainable.md) — Phase D4 legacy tuple `trainable`; corrected to read selection from the setup tree.
- [2026-09-29](02-conditional-graph/2026-09-29-conditional-wrapper-phase-d3-cpu-presence-metadata.md) — Phase D3 CPU presence metadata with device-like buffers.
- [2026-09-28](02-conditional-graph/2026-09-28-conditional-wrapper-phase-d2-occurrence-plan.md) — Phase D2 explicit occurrence mapping for mixed tied/isbits parameters.
- [2026-09-28](02-conditional-graph/2026-09-28-conditional-wrapper-phase-d1-canonical-traversal.md) — Phase D1 canonical traversal order.
- [2026-09-22](02-conditional-graph/2026-09-22-supervisor-meeting-report.md) — Supervisor meeting note (compact Italian report).
- [2026-09-21](02-conditional-graph/2026-09-21-conditional-graph-report-italian.md) — Compact Italian conditional-graph design report.
- [2026-09-17](02-conditional-graph/2026-09-17-conditional-wrapper-phase-d-plan-revision.md) — Phase D plan revision.
- [2026-09-17](02-conditional-graph/2026-09-17-conditional-wrapper-phase-c-review-followup.md) — Phase C review follow-up: direct-nesting composition guard.
- [2026-09-17](02-conditional-graph/2026-09-17-conditional-wrapper-phase-c-wrapper-boundary.md) — Phase C wrapper boundary; committed `109aad9c`.
- [2026-09-16](02-conditional-graph/2026-09-16-conditional-wrapper-phase-b-red-test.md) — Phase B focused corruption regression (RED).
- [2026-09-16](02-conditional-graph/2026-09-16-conditional-wrapper-phase-a-baseline.md) — Phase A integrated baseline at PR head `fee2c310`.
- [2026-09-16](02-conditional-graph/2026-09-16-conditional-wrapper-plan.md) — Approved wrapper implementation plan.
- [2026-09-14](02-conditional-graph/2026-09-14-conditional-graph-optimisers-notebook.md) — Optimisers.jl tutorial notebook.
- [2026-09-14](02-conditional-graph/2026-09-14-conditional-graph-onboarding.md) — Onboarding: report, deadlock/corruption detector, notebook.

## Scope 3 — Static-network GPU scaling

- [2026-10-16](03-static-scaling/2026-10-16-od1-od5-od6-recorded.md) — OD-1, OD-5, OD-6 recorded: architecture/initialization frozen at 11,271,432 parameters, per-channel training-split normalization statistics, saved-augmentation triples; Phase 2 unblocked.
- [2026-10-16](03-static-scaling/2026-10-16-phase1-review-response.md) — Phase 1 review response: R1-R4 corrected in the specification and config; focused acceptance probes pass and reproduce exactly under independent verification.
- [2026-10-16](03-static-scaling/2026-10-16-phase1-review.md) — Phase 1 shared-specification review: four defects (classifier axes, kernel orientation, AdamW constructor, beta-power mapping); local probes and source evidence.
- [2026-10-15](03-static-scaling/2026-10-15-phase1-shared-specification.md) — Phase 1 shared specification drafted: config, readable spec, and open-decision table; independent verification of parameter counts and source pins.
- [2026-10-15](03-static-scaling/2026-10-15-leonardo-benchmark-env-setup.md) — Leonardo benchmark environment setup (not run): PyTorch 2.9.1+cu126 venv and Tiny ImageNet-200 verified; environment smoke PASS; no test, benchmark, or training ran.

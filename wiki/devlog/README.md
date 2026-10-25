# Devlog index

Latest static-scaling decision: [2026-10-19 OD-14 benchmark policy](03-static-scaling/2026-10-19-od14-benchmark-policy.md).

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

- [2026-10-25](03-static-scaling/2026-10-25-cache-diagnostic.md) — Bounded W=1 cache diagnostic PASS: job 62241616, four runs, remote/local verification and 118 checksums; Flux releases ~48.4–48.5k MiB after GC.

- [2026-10-25](03-static-scaling/2026-10-25-memory-campaign-completed-62195500.md) — Memory campaign completed (job `62195500`, complete 390-update campaign, 3 repeats): all nine batch-256 framework/world cases stayed below the OD-8 whole-process sampled peak limit (`51,955.8 MiB`, 80% of `64,944.8 MiB`) across all 390 measured updates. Worst-rank peaks: `flux-current` W=1 `49,856 MiB` (76.77%, margin 2,099.8 MiB, delta +64 MiB vs preflight), W=2 `37,338 MiB` (57.49%), W=4 `34,282 MiB` (52.79%); `flux-historical` W=1 `49,760 MiB` (76.62%), W=2 `41,978 MiB` (64.64%), W=4 `39,370 MiB` (60.62%); `pytorch` W=1 `10,894 MiB`, W=2 `6,138 MiB`, W=4 `3,994 MiB`. Memory in Flux is essentially flat over 390 updates (+32–128 MiB across all cases); 91.3% of the Flux W=1 peak is resident before the measured window (CUDA cache and runtime). Local independent checker passed: `FULL_CAMPAIGN_ARTIFACT_CHECK PASS`.

- [2026-10-25](03-static-scaling/2026-10-25-memory-campaign-readiness.md) — Memory campaign launch readiness (local review only): preflight source and current campaign scripts are byte-identical, local gates re-pass, and the copied decision passes `ready`; conditional GO for the three-repeat 390-update measurement, NO-GO for a final campaign-length memory claim. The 20-update preflight leaves Flux W=1 only 2.1–2.2 GiB below the 80% gate; the campaign checker reports but does not re-gate 390-update memory, and the peak is 91% pre-window CUDA cache. Do not sync: the decision binds remote state `1ddd2d20…`, while the local tree at review was `7f1c9518…` at HEAD `976db1f`. Set `REPEATS=3`; run the reclaim diagnostic before a live-tensor claim. The campaign was subsequently launched as job `62195500`.

- [2026-10-24](03-static-scaling/2026-10-24-memory-preflight-62167742.md) — Memory preflight PASS (job `62167742`, preflight only): all nine batch-256 framework/world cases pass OD-8 with a deliberately small 4-CPU allocation; Flux W=1 is tightest at `49,792 MiB` (76.67%) current and `49,728 MiB` (76.57%) historical, only 2.1–2.2 GiB below the 80% gate, while W=2/4 and PyTorch are lower. Every rank had 5–18 training-window samples; max trace gaps 0.29–0.44 s. The 390-update campaign was not run, and this PASS does not establish campaign headroom.

- [2026-10-24](03-static-scaling/2026-10-24-memory-harness-prepared.md) — Memory benchmark harness prepared (local only): campaign repeats now sample `nvidia-smi` process memory with per-repeat raw traces and rank registration; both full-dataset runners record per-rank wall-clock training windows; the checker validates windows, enforces campaign sample density (one sample/second, two-sample floor), a 5 s gap/slack limit, strictly increasing timestamps, and aggregate span plus internal-gap coverage on the simultaneous samples themselves; it reports per-rank peak/median, startup and whole-process peaks, repeat spread, and a simultaneous-sum aggregate, and writes a `memory` section plus a nine-row `FULL_MEMORY` table. The thesis pin is local HEAD `660e363e…`. A second review added duplicate-timestamp and aggregate-start/middle-gap rejections. Local gates pass (`FULL_CAMPAIGN_SELFTEST PASS checks=37`, 8/8 launcher tests, `py_compile`, `bash -n`, `JULIA_PARSE_OK`, `git diff --check`); no remote job ran and no memory result exists.

- [2026-10-24](03-static-scaling/2026-10-24-memory-rerun-plan.md) — Memory benchmark rerun plan: reuse the accepted full-dataset three-arm W=1/2/4 batch-256 campaign; enable the existing 0.25 s sampler for campaign repeats, add per-rank training-window timestamps, validate coverage, run the nine-case preflight, then 20 warmup + 390 updates with three fresh-process repeats; report per-rank sampled peaks, worst rank, repeat spread, and a simultaneous-sum aggregate, with startup separated from training. Planning only; no code or remote state changed.

- [2026-10-22](03-static-scaling/2026-10-22-presentation-revision.md) — Supervisor-facing static-scaling deck rebuilt: candidate-comparison slide added, executive summary made number-free, architecture and campaign-integrity slides removed, timing protocol consolidated, internal phase/commit/job/OD/Slurm labels stripped, one candidate palette applied (static Flux red, guarded Flux yellow, PyTorch blue), and charts enlarged to fill the slide; 27 pages, benchmark data and claim boundaries unchanged.

- [2026-10-19](03-static-scaling/2026-10-19-full-campaign-launcher-hardening.md) — Full-dataset preflight/campaign launcher and checker hardened: `all`/`preflight`/`campaign` modes behind `CONFIRM_CAMPAIGN=1`, pinned thesis/Flux identity checks, compute-node precompile, 9-config OD-8 preflight with `nvidia-smi` peak sampling, K=195/390 campaign with 5 repeats, and untimed diagnostics; the checker validates exact records, runtime versions, `eta_sha256`, and `NUMERICAL_EQUIVALENCE_CLAIM none` (`FULL_CAMPAIGN_SELFTEST PASS checks=21`). No preflight or campaign ran; the OD-8 memory risk remains and local historical Flux state drift is a preflight risk ([launcher](../../scripts/remote/leonardo/run-static-full-campaign.sh), [checker](../../scripts/static_scaling/check_full_campaign.py)).

- [2026-10-19](03-static-scaling/2026-10-19-full-dataset-loader-validation.md) — Full-dataset mmap loader implemented for PyTorch, Flux current `f7b4520c`, and Flux historical `8add0e42`; job `61450014` PASS (`FULL_VALIDATION PASS`, 14:09) at W=1,2,4 CPU contract checks (exact order/shard/tail/onehot/bundle; transformed `max_abs_diff=0.0`), unchanged fixture smoke PASS, and three 1-update/batch-512/W=1 GPU dry runs with finite loss `5.492467880249023`. Flux device-used 83–86% flags an OD-8 memory-margin risk; no campaign ran and no performance claim is made ([results](../../artifacts/logs/static-scaling/2026-10-19-full-loader-validation-61450014/RESULTS.md), [failure note](../failures/2026-10-19-full-loader-validation-driver.md)).

- [2026-10-19](03-static-scaling/2026-10-19-timing-review-response.md) — Timing review response: the four P2 findings are answered; the checker now validates schema/precision/job/node/fixture/bundle/state hashes, all 36 source hashes, driver markers, and `provenance.json` policy/qualifications (`TIMING_ARTIFACT_SELFTEST PASS checks=13`); the two loaded shared helpers differ from the Phase 4 accepted snapshot (recorded, not asserted equivalent). The 300-update final-state snapshot diagnostic is complete (job `61366015`, 6/6 records and 6/6 snapshots) and descriptive only: `param` `max_abs` up to `0.0333` and `rel_l2` up to `0.00224`, `m` `rel_l2` up to `0.0485`, `v` `rel_l2` up to `0.000333`, `beta_power` exact; `NUMERICAL_EQUIVALENCE_CLAIM none`. The Phase 6 numerical-acceptance policy (OD-14) remains open ([response](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/REVIEW-RESPONSE.md), [numerics results](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-61366015/RESULTS.md), [failed attempts](../failures/2026-10-19-timing-numerics-attempts.md)).

- [2026-10-19](03-static-scaling/2026-10-19-fixture-timing-campaign.md) — Fixture GPU timing campaign PASS: job `61314790` (`static-timing-campaign`, `lrdn3064`, `boost_qos_lprod`) printed `FLUX_PRECOMPILE_OK`, `STATIC_TIMING_campaign PASS`, and `TIMING_DRIVER_EXIT=0` with 30/30 records (5 repeats × 6 configurations) and no failed runs; checker `TIMING_ARTIFACT_CHECK PASS`. Flux is flat across W=1/2/4 (`5.483/5.715/5.664 s` median; efficiency `1.0000/0.4797/0.2420`), while PyTorch scales `1.5602x` at W=2 and `1.8060x` at W=4. Bounded training-only fixture pilot; not full-dataset, end-to-end, memory-pilot, or learning-curve evidence. Supersedes canceled partial `61305371`; smoke `61305319` remains preflight ([results](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/RESULTS.md)).

- [2026-10-18](03-static-scaling/2026-10-18-phase4-distributed-correctness.md) — Phase 4 GPU accepted under absolute error `<1e-3`, 202 categories PASS. CPU deferred. GPU timing pilot submitted.

- [2026-10-18](03-static-scaling/2026-10-18-r1-r2-precision-policy-fixes.md) — R1/R2 precision-policy fixes implemented, hardened after independent review, and remotely verified, with full Phase 3 recheck PASS: strict FMA is fail-closed with `PrecisionPolicyError`, CUDA.jl/cuDNN.jl 6.2.2 versions, override methods, and configured state are enforced, and the precision regression now has a 10-case synthetic `policy` suite; final hardened precompile `61187225`, focused Leonardo job `61188854` (`lrdn1845`, exit 0) with `PHASE3_PRECISION_REGRESSION PASS`, `fallback=0`, and policy 10/10, and full recheck `61191281` (`lrdn2911`, exit 0) reproducing all four PASS summaries, 337/337 exact reference, 32/32 `CUDNN_FMA_MATH`/`fallback=0`, and zero violations. The GPU-free policy test passes on deathstar under CUDA.jl/cuDNN.jl 6.3.1 (synthetic only). The default `precompile.sh` cannot run on Leonardo because the root manifest is excluded and compute nodes are offline; use `precompile-phase3.sh`. Intermediate jobs `61074887`/`61077382`/`61091402` are superseded history. [Review response](../../reports/static-scaling-phase3-acceptance-review-response.md).
- [2026-10-18](03-static-scaling/2026-10-18-phase3-final-acceptance.md) — **Phase 3 accepted**: final matched single-GPU Flux/PyTorch driver (job `60956582`, `lrdn0165`, exit 0) prints `PHASE3_REFERENCE PASS`, `PHASE3_REFERENCE_CHECK PASS`, `PHASE3_FLUX_CHECK PASS`, and `PHASE3_REMOTE_CHECK PASS` under the frozen FP32/TF32-off policy and OD-10 rule; all 32 cuDNN choices are `CUDNN_FMA_MATH(3)`/no fallback, the reference is 337/337 exact-equal to the preserved attempt-3 archive, remote/local SHA-256 matches 15/15, and the precision regression (`60954547`) and optimizer probe (`60955690`) pass. Precision cause was `PEDANTIC_MATH -> CUDNN_DEFAULT_MATH` plus the Float32 autotuner math-type skip; explicit `CUDNN_FMA_MATH` and the FMA-aware chooser corrected it. Benchmarks are not part of Phase 3; next is Phase 4 distributed correctness at 2 and 4 ranks ([results](../../artifacts/logs/static-scaling/2026-10-18-phase3-final/RESULTS.md), [review response](../../reports/static-scaling-phase3-review-response.md)).
- [2026-10-18](03-static-scaling/2026-10-18-phase3-step5-decisions.md) — Phase 3 Step 5 complete: hardened Phase 2 checkers pass on the real Leonardo `$PREP`/fixtures (job `60951247`, `lrdn0171`; `PHASE2_DATASET_CHECK`/`PHASE2_JULIA_DATA_CHECK`/`PHASE2_REMOTE_CHECK` PASS, 113 checks, driver exit 0), and OD-2/OD-3/OD-4/OD-11 are recorded with user authority at the values exercised by Steps 3c/4; Phase 2 remote prerequisite closed; Step 6 subsequently passed (see the 2026-10-18 final-acceptance entry above) ([provenance](../../artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/PROVENANCE.md)).

- [2026-10-17](03-static-scaling/2026-10-17-phase3-step4-optimizer-probe.md) — Phase 3 Step 4 optimizer-contract probe PASS: CPU-only Leonardo job `60948431` (`lrdn0171`) prints `PHASE3_OPTIMIZER_PROBE PASS` with 69 checks and 34/34 negative controls rejected; the probe asserts the real 21/41 per-leaf AdamW grouping in both frameworks, controlled nonzero-moment updates against a Float64 oracle, tiny-gradient epsilon placement, and schedule boundaries at `1`/`W`/`W+1`/`T`, without changing training code or Phase 3 tolerances ([results](../../artifacts/logs/static-scaling/2026-10-17-phase3-step4/RESULTS.md)).
- [2026-10-17](03-static-scaling/2026-10-17-phase3-step3c-od10-acceptance.md) — Phase 3 Step 3c acceptance PASS: approved OD-10 revision applies the base rule (`1e-5`/`1e-4`) plus only `grads.u1 atol=1e-3`, `grads.u3 atol=2e-3`, `m.u3 atol=2e-4`; driver job `60942933` on `lrdn0129` prints all four PASS summaries and exits 0, all 32 cuDNN choices remain `CUDNN_FMA_MATH(3)`/no fallback, and the new reference is again 337/337 exact-equal with the independent checker PASS ([results](../../artifacts/logs/static-scaling/2026-10-17-phase3-step3c/RESULTS.md)).
- [2026-10-17](03-static-scaling/2026-10-17-phase3-numerics-calibration.md) — Phase 3 numerical calibration: new PyTorch CPU reference (job `60912525`) and Flux CPU checks (job `60912664`) show the gradient/first-moment residual is reproduced by a PyTorch CPU-vs-GPU control (grads.u1 `3.02e-4`, grads.u3 `4.10e-4`, m.u3 `5.06e-5`) and is within ~1.6x of the Flux GPU residual; ordinary FP32 backend/kernel variation, not a Flux backward defect. Proposes, but does not apply, category floors `grads.u1 1e-3`, `grads.u3 2e-3`, `m.u3 2e-4` (rtol `1e-4`, ≥2x margin) ([results](../../artifacts/logs/static-scaling/2026-10-17-phase3-numerics/RESULTS.md)).
- [2026-10-17](03-static-scaling/2026-10-17-phase3-step3b-fma-algo-chooser.md) — Phase 3 Step 3b: world-age-safe FMA `math_mode`/algorithm-chooser overrides plus perf-cache clearing; precision regression PASS (job 60910394) and all 32 full-model choices `CUDNN_FMA_MATH(3)`/no fallback, so activations, logits, loss, params, and `v.u3` now PASS; `grads.u1` (3.03e-4), `grads.u3` (6.67e-4), and `m.u3` (6.58e-5) still fail on FP32 backward FFT/Winograd/direct algorithm rounding (job 60910786). Step 4 blocked on a user decision ([results](../../artifacts/logs/static-scaling/2026-10-17-phase3-step3b/RESULTS.md)).
- [2026-10-17](03-static-scaling/2026-10-17-phase3-step3-full-comparison-fail.md) — Phase 3 Step 3 full comparison FAIL (superseded by Step 3b): new reference array-identical to preserved (337/337), but Flux failed activations (`7.65e-3`, 2,705,938 violations) and downstream categories with unchanged tolerances; requested/effective `CUDNN_FMA_MATH(3)` did not change the full-model kernels because the Float32 autotuner ignored the descriptor math type. Step 4 blocked on the precision-enforcement fix ([failure](../failures/2026-10-17-phase3-step3-fma-full-model.md)).
- [2026-10-17](03-static-scaling/2026-10-17-phase3-step2-precision-experiment.md) — Phase 3 Step 2 precision experiment PASS: stock `DEFAULT_MATH` retains the conv failure (conv1 `3.57e-3`, conv2 `3.65e-3`) while explicit `CUDNN_FMA_MATH` is at CPU level (`0`, `1.9e-6`, zero violations); reusable `configure_precision!` wired into the checkers; pinned-algorithm regression PASS (FMA ~1e-7..1e-6 vs current ~2.9e-4).
- [2026-10-17](03-static-scaling/2026-10-17-phase2-acceptance-fixes.md) — Phase 2 acceptance fixes verified: R1-R3 corrected, local `check_all.jl` 81 checks exit 0 (was 75), Python positive 63 checks, and `negative_tests_local.py` `NEGTEST PASS`; remote re-check on the real `$PREP` pending Leonardo SSH access ([failure](../failures/2026-10-17-leonardo-ssh-publickey.md)).
- [2026-10-17](03-static-scaling/2026-10-17-phase2-implementation-review.md) — Phase 2 implementation review: three P2 acceptance gaps (prepared-file hashes, Julia fixture integrity, 4-of-20 GroupNorm coverage); no numerical defect in the saved fixtures.
- [2026-10-16](03-static-scaling/2026-10-16-phase2-artifacts-complete.md) — Phase 2 complete and independently verified: dataset, 50-epoch schedule, 62-array bundle (`bundle_hash=a7632ca9…`), and four shared fixtures generated on Leonardo; prep/check jobs 60758615/60758839 COMPLETED 0:0 with all PASS markers; local operator check 75 checks exit 0; independent verification PASS; full failure history (env token, QoS, `std64` masked exit, linear bias) indexed.
- [2026-10-16](03-static-scaling/2026-10-16-phase2-prep-retry4-make-fixtures-bug.md) — Retry-4: steps 1-3 PASS, fixtures blocked by the linear-bias broadcast bug; fixed driver aborted with `PHASE2_PREP FAIL`; resolved in retry-5.
- [2026-10-16](03-static-scaling/2026-10-16-od1-od5-od6-recorded.md) — OD-1, OD-5, OD-6 recorded: architecture/initialization frozen at 11,271,432 parameters, per-channel training-split normalization statistics, saved-augmentation triples; Phase 2 unblocked.
- [2026-10-16](03-static-scaling/2026-10-16-phase1-review-response.md) — Phase 1 review response: R1-R4 corrected in the specification and config; focused acceptance probes pass and reproduce exactly under independent verification.
- [2026-10-16](03-static-scaling/2026-10-16-phase1-review.md) — Phase 1 shared-specification review: four defects (classifier axes, kernel orientation, AdamW constructor, beta-power mapping); local probes and source evidence.
- [2026-10-15](03-static-scaling/2026-10-15-phase1-shared-specification.md) — Phase 1 shared specification drafted: config, readable spec, and open-decision table; independent verification of parameter counts and source pins.
- [2026-10-15](03-static-scaling/2026-10-15-implementation-phases.md) — Nine-phase implementation plan created from the detailed proposal and environment evidence; deliverables and acceptance conditions per phase; remaining decisions preserved.
- [2026-10-15](03-static-scaling/2026-10-15-leonardo-benchmark-env-setup.md) — Leonardo benchmark environment setup (not run): PyTorch 2.9.1+cu126 venv and Tiny ImageNet-200 verified; environment smoke PASS; no test, benchmark, or training ran.

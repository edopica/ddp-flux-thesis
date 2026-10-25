# Wiki Quickstart

**2026-10-25 update:** the full 390-update memory campaign and four-run W=1
cache diagnostic both pass. Flux releases over 97% of post-GC process residency
at the diagnostic boundary; operational training peaks remain unchanged.
See [current context](context/current.md) and the
[memory interpretation](../reports/static-scaling-memory-interpretation.md)
for results and why per-step cache release is not the baseline default.
Earlier next-action notes below are historical.

**2026-10-19 update:** [OD-14](../experiments/static-scaling/OD-14-benchmark-acceptance.md) is recorded.
The fixture timing remains a qualified descriptive result, with no long-run numerical-equivalence claim.
The next action is the Phase 5 larger-batch memory pilot. See [current context](context/current.md).

Static-network GPU scaling **Phase 3 is accepted (2026-10-18)**. The final matched single-GPU Flux/PyTorch run (driver job `60956582`, `lrdn0165`, exit 0) passed `PHASE3_REFERENCE PASS`, `PHASE3_REFERENCE_CHECK PASS`, `PHASE3_FLUX_CHECK PASS`, and `PHASE3_REMOTE_CHECK PASS` under the frozen rule: base `atol=1e-5`/`rtol=1e-4` plus only `grads.u1 atol=1e-3`, `grads.u3 atol=2e-3`, and `m.u3 atol=2e-4`, each with `rtol=1e-4`. All 32 cuDNN choices are `CUDNN_FMA_MATH(3)` with no fallback; the reference is 337/337 exact-equal to the preserved attempt-3 archive; the independent NumPy checker passes; remote/local SHA-256 matches 15/15 ([results](../artifacts/logs/static-scaling/2026-10-18-phase3-final/RESULTS.md), [provenance](../artifacts/logs/static-scaling/2026-10-18-phase3-final/PROVENANCE.md), [review response](../reports/static-scaling-phase3-review-response.md)).
The precision cause was stock `PEDANTIC_MATH -> CUDNN_DEFAULT_MATH` (TF32 permitted) plus cuDNN.jl 6.2.2's Float32 autotuner math-type skip; explicit `CUDNN_FMA_MATH` and the FMA-aware chooser in `scripts/static_scaling/precision.jl` corrected it. Focused probes on the final snapshot pass: precision regression `60954547` (`PHASE3_PRECISION_REGRESSION PASS`) and optimizer/schedule probe `60955690` (`PHASE3_OPTIMIZER_PROBE PASS`, 34/34 negative controls rejected); precompile `60954207` printed `FLUX_PRECOMPILE_OK`. The hardened Phase 2 remote prerequisite remains PASS on job `60951247` (113 checks, exit 0; [evidence](../artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/PROVENANCE.md)).
Benchmarks are **not** part of Phase 3. The acceptance-review R1/R2 fixes are hardened and verified: precompile `61187225`, focused regression `61188854` with 10/10 policy cases, and the hardened helper reproduced the accepted result in full recheck `61191281` (337/337 exact reference, 32/32 FMA, `fallback=0`, zero violations). R3 is recorded as OD-14 for Phase 6. Next action: start Phase 4 distributed correctness at two and four ranks per `temp/static-graph-gpu-scaling-implementation-phases.md` section 6. Leonardo access currently works; a transient network drop occurred during the superseded intermediate recheck but the job completed, and the deathstar policy log was recovered after a transient DNS interruption ([failure](failures/2026-10-17-leonardo-ssh-publickey.md)). See [current.md](context/current.md), the [hardening evidence](../artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-hardening/), and the [final-acceptance devlog](devlog/03-static-scaling/2026-10-18-phase3-final-acceptance.md).

Phase K complete on deathstar and Leonardo (2026-10-05): the K2 task-conditioned MNIST/Fashion-MNIST presentation passes CPU/MPI at 2 and 4 ranks (13/13 driver cells) and CUDA/NCCL at 2 ranks (6/6) on deathstar, plus the K2 smoke (8/8) and K1 toy regression (8/8).
The Leonardo 4-rank NCCL cells now PASS: K1 toy root `k1-2026-10-05-leonardo-nccl4` (8/8) and K2 root `k2-2026-10-05-leonardo-nccl4` (6/6), all driver exit 0 with `PHASE_K_SUMMARY ... backend=nccl ranks=4 PASS`.
Per-step comparison against an independent native reference, ordered collective traces, fresh-process continuation, and the DDP/reference learning-curve and learning-rate figures all pass (41 PNGs plus `figures-metadata.txt` in `reports/figures/phase-k-k2/`); one checkpoint-digest defect was found, fixed, and pinned by regression. No Flux source changed.
The optional live timing stretch is complete and corrected (2026-10-06): correctness preflight PASS and 5/5 timed repeats each mode for 1×A100 vs 4×A100 NCCL on Leonardo; ratio-of-medians training-only speedup 1.888594 and total speedup 1.397335; the historical instrumented 1.480203 is superseded ([report](../reports/conditional-graph-phase-k-timing-stretch.md), [devlog](devlog/02-conditional-graph/2026-10-06-timing-corrections.md)).
The deferred-behavior report is complete ([audit 002](audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md)).
Next action: implement `notebook/conditional_graph_benchmark.ipynb` from the corrected timing artifacts. See [current.md](context/current.md), the [K1/K2 Leonardo devlog](devlog/02-conditional-graph/2026-10-05-leonardo-k1-k2-4rank-nccl.md), and the [K2 devlog](devlog/02-conditional-graph/2026-10-05-phase-k-k2-mnist-implementation.md).
Earlier phases are summarized in [current.md](context/current.md) and the [devlog index](devlog/README.md).

Flux.jl DDP Thesis — a thesis-control workspace for stabilizing, testing, documenting, and evaluating Distributed Data Parallel training in Flux.jl.

**Current checkpoint (conditional graph):** Phase K required acceptance matrix complete (2026-10-05): deathstar K2 CPU/MPI at 2 and 4 ranks (13/13) and native CUDA/NCCL at 2 ranks (6/6); Leonardo 4-rank CUDA/NCCL K1 `k1-2026-10-05-leonardo-nccl4` (8/8) and K2 `k2-2026-10-05-leonardo-nccl4` (6/6) PASS, driver exit 0; 41 figures in `reports/figures/phase-k-k2/`. The optional post-Phase K live timing stretch is complete and corrected (training-only 1.888594, total 1.397335; historical instrumented 1.480203 superseded; `reports/conditional-graph-phase-k-timing-stretch.md`). The deferred-behavior report is complete (`wiki/audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md`). Flux has uncommitted changes on `55171e65`.
**Active implementation branch:** `ddp/integration-m1-spike` at `55171e65` (`../flux-integration` worktree, not pushed).
**Next action (conditional graph):** implement `notebook/conditional_graph_benchmark.ipynb` from the corrected `artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/` artifacts (two-notebook plan implementation item 6).

## Repository overview

This repository is the control center for a 12-checkpoint plan (C0–C12) to bring Flux.jl's DDP support from audit through correctness verification to production-ready examples. A local Flux.jl fork at `../ddp_flux` (`/home/kurapica/Projects/ddp_flux/ddp_flux`) is developed alongside this repo.

Key areas:

| Directory | Purpose |
|---|---|
| `scripts/setup/` | C0 environment and auditing utilities |
| `scripts/checks/` | MPI health and smoke tests |
| `scripts/remote/` | Cluster deployment and synchronization |
| `scripts/reference/` | C1 deterministic single-process baseline |
| `scripts/launch/` | C2 distributed launch skeleton |
| `scripts/sync/` | C3 model broadcast and C5/C6 gradient/optimizer synchronization |
| `scripts/data/` | C4 distributed data sharding verification |
| `artifacts/baselines/` | Reference loop baseline data |
| `artifacts/logs/` | Audit and test output logs |
| `notebook/` | Feature demonstration notebooks |

## Getting started

```bash
make env                  # Set up Julia environment
make install-mpiexec      # Install MPI launcher
make check                # Verify environment
make health-check         # MPI/PMI validation
make smoke-cpu            # 2-process MPI smoke test
```

Common reproducible commands: `make env`, `make install-mpiexec`, `make check`, `make health-check`, `make smoke-cpu`, `make audit`, `make reference`, `make reference-verify`, `make launch-2`, `make launch-4`, `make sync-model`, `make verify-sync`, `make verify-data`, `make verify-gradients`.
The `Makefile` default `FLUX_REPO_PATH` still names the old fork path `../Flux.jl`; pass the real path explicitly, for example `make env FLUX_REPO_PATH=../ddp_flux` or `make audit FLUX_REPO_PATH=../ddp_flux`.

## Documentation sections

### Context
Current project state, remote node guides, and operational knowledge.

- **[current.md](context/current.md)** — Always start here. Current checkpoint, what is known, what was done last, commands that pass/fail, and next action.
- **[remote-nodes.md](context/remote-nodes.md)** — Remote execution index: workflow, tools, and server list.
- **[bocconi.md](context/bocconi.md)** — Bocconi SLURM cluster: precompile rules and usage.
- **[deathstar.md](context/deathstar.md)** — Non-SLURM remote node usage (courtesy machine, use with care).
- **[leonardo.md](context/leonardo.md)** — CINECA GPU cluster: operational since 2026-10-04 (SSH, Slurm, Julia workflow, 4-rank NCCL verified).

### Decisions
Architecture Decision Records explaining why the codebase is structured the way it is.

- **[ADR-0001](decisions/ADR-0001-repo-layout.md)** — Repository layout and thesis-control workspace design.
- **[ADR-0002](decisions/ADR-0002-baseline-ddp-path.md)** — Choosing DDP as the baseline distributed training path.
- **[ADR-0003](decisions/ADR-0003-mpi-pmi-guardrails.md)** — MPI/PMI mismatch guardrails and health checks.
- **[ADR-0004](decisions/ADR-0004-scripts-reorganization.md)** — Scripts directory reorganization into functional subdirectories.
- **[ADR-0005](decisions/ADR-0005-precompile-compute-only.md)** — Precompile only on compute nodes (Bocconi CPU-cache incompatibility).
- **[ADR-0006](decisions/ADR-0006-integration-baseline.md)** — PR #2694 integration baseline and isolated worktree policy.
- **[ADR-0007](decisions/ADR-0007-defer-unused-parameter-support.md)** — Defer unused-parameter support from the upstream PR; conditional graph as separate work.

### Devlog
Chronological development log entries for each checkpoint and milestone, grouped by project scope.

- **[01-pr2694/](devlog/01-pr2694/)** — C0–C8 upstream DDP work, M0/M1 integration spike, PS1–PS8 PR #2694 salvage, Windows routing fix.
- **[02-conditional-graph/](devlog/02-conditional-graph/)** — conditional-graph onboarding, wrapper Phases A–K, native CUDA/NCCL validation, timing, notebooks.
- **[03-static-scaling/](devlog/03-static-scaling/)** — static-network GPU scaling: Tiny ImageNet-200/PyTorch benchmark environment setup; Phase 2 data, schedule, bundle, and fixture pipeline (complete and verified; acceptance checks hardened 2026-10-17).

Complete dated index: **[devlog/README.md](devlog/README.md)**.

### Audits
Codebase audit notes, grouped by project scope. Complete index: **[audits/README.md](audits/README.md)**.

- **[01-pr2694/](audits/01-pr2694/)** — C0 audit of Flux.jl's distributed training implementation.
- **[02-conditional-graph/](audits/02-conditional-graph/)** — Phase H compatibility audit (001) and deferred-behavior audit (002).

### Failures
Post-mortem analysis of significant failures encountered during development. All records live in **[failures/](failures/)**; the [template](failures/0000-template.md) is `0000-template.md`. Recent records cover the conditional detector, the Phase H defects, the NCCL harness, Leonardo artifact prefetch, the Phase K campaigns, the static-scaling Phase 2 prep failures (Leonardo env token/QoS, `std64` masked exit, linear-fixture bias), and the 2026-10-17 Leonardo SSH publickey blocker.

### Templates
Templates for new documentation entries.

- **[devlog-template](templates/devlog-template.md)**
- **[experiment-template](templates/experiment-template.md)**

### Testing
CI diagnostics guidance and distributed-test campaign notes.

- **[ci-diagnostics.md](testing/ci-diagnostics.md)** — Best practices and anti-pattern examples from the PR #2694 CI failure investigation (harness output capture, MPI slots, docker parity repro).
- **[c8-correctness-matrix.md](testing/c8-correctness-matrix.md)** — C8 correctness matrix (frozen `ddp_flux` master scope).

### Reports
Wiki-local reports; thesis-facing drafts live in the top-level `reports/` directory.

- **[c8-negative-demonstrations.md](reports/c8-negative-demonstrations.md)** — C8 negative demonstrations for the frozen master scope.

## For agents

When working in this repository:

1. Read `README.md` and `wiki/context/current.md` first.
2. Check the latest devlog entry for recent activity.
3. Consult relevant ADRs before making architectural changes.
4. Record all reproducible commands and their outcomes.
5. Update context, devlog, and progress table at session end.
6. File failure reports for any command that does not pass.

## Key conventions

- **Do not optimize before correctness tests pass.**
- DDP communication stays outside automatic differentiation for the baseline.
- Use Optimisers.jl for optimizer setup; Flux/Functors-compatible traversal for nested models.
- Every MPI collective must be called by every rank in the same order.
- Rank 0 is the only rank writing shared output files (unless explicitly designed otherwise).
- Scale gradients explicitly and document the convention.
- Treat "loss decreases" as a signal, not proof of correctness.

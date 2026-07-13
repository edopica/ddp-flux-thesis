# OpenWiki Quickstart

Flux.jl DDP Thesis — a thesis-control workspace for stabilizing, testing, documenting, and evaluating Distributed Data Parallel training in Flux.jl.

**Current checkpoint:** C6 — Optimisers.jl integration (Done) → C7 — End-to-end two-GPU example  
**Active Flux branch:** `ddp/optimizer`

## Repository overview

This repository is the control center for a 12-checkpoint plan (C0–C12) to bring Flux.jl's DDP support from audit through correctness verification to production-ready examples. A local Flux.jl fork at `../Flux.jl` (`/home/kurapica/Projects/ddp_flux/ddp_flux`) is developed alongside this repo.

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

All reproducible commands (through C6): `make env`, `make install-mpiexec`, `make check`, `make health-check`, `make smoke-cpu`, `make audit`, `make reference`, `make reference-verify`, `make launch-2`, `make launch-4`, `make sync-model`, `make verify-sync`, `make verify-data`, `make verify-gradients`.

## Documentation sections

### Context
Current project state, remote node guides, and operational knowledge.

- **[current.md](context/current.md)** — Always start here. Current checkpoint, what is known, what was done last, commands that pass/fail, and next action.
- **[remote-nodes.md](context/remote-nodes.md)** — HPC cluster and remote execution workflow (sync, setup, run).
- **[deathstar.md](context/deathstar.md)** — Non-SLURM remote node usage (courtesy machine, use with care).

### Decisions
Architecture Decision Records explaining why the codebase is structured the way it is.

- **[ADR-0001](decisions/ADR-0001-repo-layout.md)** — Repository layout and thesis-control workspace design.
- **[ADR-0002](decisions/ADR-0002-baseline-ddp-path.md)** — Choosing DDP as the baseline distributed training path.
- **[ADR-0003](decisions/ADR-0003-mpi-pmi-guardrails.md)** — MPI/PMI mismatch guardrails and health checks.
- **[ADR-0004](decisions/ADR-0004-scripts-reorganization.md)** — Scripts directory reorganization into functional subdirectories.

### Devlog
Chronological development log entries for each checkpoint and milestone.

- **[2026-07-02](devlog/2026-07-02.md)** — C0 audit kickoff.
- **[2026-07-05](devlog/2026-07-05.md)** — C0 audit continuation.
- **[2026-07-07](devlog/2026-07-07.md)** — C1 deterministic reference loop.
- **[2026-07-07-remote](devlog/2026-07-07-remote.md)** — Remote execution workflow setup.
- **[20260709-c2-launch](devlog/20260709-c2-launch.md)** — C2 distributed launch skeleton.
- **[20260709-ssh-refactor](devlog/20260709-ssh-refactor.md)** — SSH workflow refactoring.
- **[20260709-deathstar](devlog/20260709-deathstar.md)** — Deathstar node setup.
- **[2026-07-10](devlog/2026-07-10.md)** — PMI mismatch guardrails.
- **[20260711-c3-sync-model](devlog/20260711-c3-sync-model.md)** — C3 model broadcast and verification.
- **[20260713-c4-data-sharding](devlog/20260713-c4-data-sharding.md)** — C4 distributed data sharding and padding fix.
- **[20260713-c5-c6-grad-sync](devlog/20260713-c5-c6-grad-sync.md)** — C5/C6 gradient synchronization and Optimisers.jl integration.

### Audits
Codebase audit notes.

- **[000-current-flux-distributed](audits/000-current-flux-distributed.md)** — C0 audit of Flux.jl's distributed training implementation.

### Failures
Post-mortem analysis of significant failures encountered during development.

- **[20260709-ssh-deathstar](failures/20260709-ssh-deathstar.md)** — SSH authentication failure on deathstar node.
- **[0000-template](failures/0000-template.md)** — Failure report template.

### Templates
Templates for new documentation entries.

- **[devlog-template](templates/devlog-template.md)**
- **[experiment-template](templates/experiment-template.md)**

## For agents

When working in this repository:

1. Read `README.md` and `openwiki/context/current.md` first.
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

# Deferred-behavior probe metadata — 2026-10-06

Local two-rank CPU/MPI characterization of `DistributedOptimizerState` deferred
behaviors. Report: `wiki/audits/002-conditional-wrapper-deferred-behaviors.md`.

## Host and runtime

- Host: `omarchy` (local workstation), MPICH at `/opt/mpich/bin/mpiexec`.
- Julia 1.12.6.
- Packages (per probe `VERSIONS` marker): Flux 0.16.11, MPI 0.20.27,
  Optimisers 0.4.9.
- Launcher: `MPI.mpiexec()` from the Flux test environment, 2 ranks.
- Project: `/home/kurapica/Projects/ddp_flux/flux-integration/test`.

## Source state

Implementation worktree `/home/kurapica/Projects/ddp_flux/flux-integration`:

- Branch `ddp/integration-m1-spike`, HEAD `55171e659b2f425da980cbc83177bf5c38f0a7a1`.
- Uncommitted (Phase H/J): `NEWS.md`, `src/distributed/public_api.jl`,
  `test/Project.toml`, `test/ext_distributed/distributed_setup.jl`,
  `test/ext_distributed/runtests.jl`,
  `test/ext_distributed/synchronized_distributedtest.jl`, plus untracked
  `test/ext_distributed/cuda_conditional_distributedtest.jl`.

Working-tree blob hashes (`git hash-object`):

| Path | Blob |
|---|---|
| `src/train.jl` | `05d57e90b925621c1d96af330394a6818a03f19e` |
| `src/distributed/public_api.jl` | `67e52161048e8c2d46b17b4c42368610abb1a288` |
| `src/distributed/backend.jl` | `56f6e5a9932eeeec97ae095353bef3be11581f87` |
| `ext/FluxMPIExt/FluxMPIExt.jl` | `ba4ae846a576f0c5ac130db8c0da52dba3dc2770` |
| `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl` | `f9c41488d19e84752b533326b145ac38d3557a3a` |
| `test/ext_distributed/conditional_distributedtest.jl` | `ba13b67d5a62f15540be362a3379da7f4a8133d2` |

Probe scripts (this repository, sha256):

| Path | SHA-256 |
|---|---|
| `scripts/checks/conditional_graph_deferred_probe.jl` | `60e490f41ec61fc6b9259fa20996c465982cb69ebec12c16e1df4b4ce38b702c` |
| `scripts/checks/conditional_graph_deferred_check.jl` | `44aa970b03a68cbaa507484b67bd129a4ed14e31ce912c28a5c6fdf753f5d75a` |

## Exact command

```bash
julia --startup-file=no --project=../flux-integration/test \
    scripts/checks/conditional_graph_deferred_check.jl
```

Environment overrides: `DEFERRED_STARTUP_TIMEOUT` (default 420 s),
`DEFERRED_HANG_TIMEOUT` (default 60 s), `DEFERRED_OUTDIR`.
The driver launches `MPI.mpiexec() -n 2 julia --startup-file=no
--project=<flux test dir> scripts/checks/conditional_graph_deferred_probe.jl <mode>`
per mode, with a shared stdout/stderr temp handle, a startup watchdog, and an
external hang watchdog. Hang modes: after the timeout the driver
`SIGTERM`/`SIGKILL`s `mpiexec` and reaps leftover probe PIDs.

## Results

Run 2026-10-06 11:12: **8/8 modes PASS, 0 failures, 83 PASS assertions**,
driver exit 0. Summary log
`artifacts/logs/conditional-wrapper/deferred/check-summary-2026-10-06.log`.

| Mode | Timeout | Exit | Runtime evidence (raw log) |
|---|---|---|---|
| `control` | none | 0 | native Adam shadow `maxdiff=0.0`; 12/12 collectives per rank |
| `nan-loss-all` | none | 0 | both ranks `DomainError` at item 3; state equals the two-step shadow (`maxdiff=0.0`, `state_ok=true`); 6/6 collectives |
| `grad-nonfinite` | none | 0 | rank 0 finite gradient, rank 1 NaN gradient, both `w_finite=false`, `wmom_finite=false`, `exc=none`; 3/3 collectives |
| `config-eta` | none | 0 | each rank matches its own rule reference (`self_maxdiff=0.0`); `cross_delta=0.14991810934487604`; 12/12 collectives |
| `config-frozen` | none | 0 | rank 1 `unchanged=true`; `cross_delta=0.20012763731269412`; 12/12 collectives (frozen still reduces) |
| `nan-loss-hang` | update | 15 | rank 0: 7 BEGIN / 6 END, blocked in presence collective #7; rank 1: `DomainError`, held |
| `early-stop` | update | 15 | rank 0: 10 BEGIN / 9 END, blocked in presence #10; rank 1 stopped at step 4, held |
| `step-desync` | update | 15 | rank 1: `delta_vs_correct=0.04994879241581662`, `delta_vs_shifted=0.0`; rank 0: 13 BEGIN / 12 END, blocked in presence #13 |

Raw logs: `artifacts/logs/conditional-wrapper/deferred/raw-<mode>-mpi2-2026-10-06.log`.

## Probe defect iterations (test tooling only, no product defect)

Before the final GREEN run, the new driver/probe pair had four defects, each
fixed and re-run: rank-interleaved marker slicing, two regex-prefix mismatches
(`CONFIG_ETA ... cross_delta`, `DESYNC_RESULT ... delta_vs_shifted`), a
`stdout=path`/`stderr=path` double-open that truncated hung-run logs, and a
config-eta shadow that applied the local instead of the global-mean gradient.
No Flux source was modified at any point. Recorded in
`wiki/failures/2026-10-06-deferred-probe-driver-defects.md`.

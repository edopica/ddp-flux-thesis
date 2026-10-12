# Conditional-graph wrapper Phase F: CPU/MPI gates at two and four ranks

Date: 2026-09-30

## Goal

Run the plan's Phase F gates: the focused conditional child file and the
complete dedicated suite at two and four MPI ranks, on the CPU-only remote
machine `deathstar`, with the four-rank analytical case proving weighting
(one rank branch A, three ranks branch B).

## Synchronization decision

`scripts/remote/sync_code.sh` syncs `../ddp_flux` (the frozen dirty C8
worktree), which is not the conditional-wrapper worktree. Per the plan and the
user's instruction, I did not use it. I ran two manual rsyncs instead:

- local `/home/kurapica/Projects/ddp_flux/flux-integration/` (clean at
  `55171e65`) -> `deathstar:~/projects/flux-integration/`, excluding `.git`,
  `.julia`, and `test/Manifest.toml`;
- local thesis workspace -> `deathstar:~/projects/ddp-flux-thesis/` with
  `--exclude-from=.rsyncignore` (the remote thesis scripts were stale before
  this sync).

The remote file hash matched the local one:
`conditional_distributedtest.jl` md5 `c26ef49b070bf2c446b827c64cc313c0`.

## Environment

Deathstar is a non-Slurm workstation, so `precompile.sh` would compile where
tests run. I ran `Pkg.instantiate(); Pkg.precompile()` directly in
`~/projects/flux-integration` with `--project=test`: exit 0, real 2 m 7.5 s,
128 dependencies precompiled in 115 s. Julia 1.12.6. All runs are CPU/MPI
only; `nvidia-smi` showed 3 healthy TITAN RTX GPUs but no GPU work was done.

## Results (all exit 0)

| Gate | Command environment | Wall | Result |
|---|---|---|---|
| Focused conditional, 2 ranks | `mpiexecjl --project=test -n 2`, `FLUX_TEST_DISTRIBUTED_BACKEND=mpi` | 1 m 3.6 s | 30 testsets, **367/367 per rank** |
| Focused conditional, 4 ranks | `mpiexecjl --project=test -n 4` | 1 m 7.9 s | 30 testsets, **367/367 per rank** |
| Focused optimizer, 2 ranks | `mpiexecjl --project=test -n 2` | 8.5 s | composition guard 6/6 per rank |
| Full dedicated suite, 2 ranks | `JULIA_MPI_TEST_NPROCS=2 julia --project=test runtests.jl mpi` | 1 m 35.4 s | `Distributed \| 6 6 1m33.5s` |
| Full dedicated suite, 4 ranks | `JULIA_MPI_TEST_NPROCS=4 ...` | 1 m 45.6 s | `Distributed \| 6 6 1m43.6s` |

The suite runs used the direct runner (it owns child creation) and not
`make c8-mpi`, which would wrap the parent in another `mpiexec`. Both suite
logs contain "Running Distributed Tests with 2/4 processes" and zero
`Fail`/`Error` lines. `FLUX_TEST_DISTRIBUTED_TIMEOUT=600` was set for the two
suite runs only; the actual child times never approached the default 120 s
watchdog (suite totals 1 m 33.5 s / 1 m 43.6 s), so the override did not mask
any failure.

## Four-rank weighting proof

The first testset (`equal-shaped conditional branches do not cross-reduce`)
gives rank 0 branch `a` and ranks 1-3 branch `b`, and integrates the analytic
global means alongside the updates: `ga = 2*(a - ta)/4` (1/4 weight) and
`gb = 2*(b - tb)*3/4` (3/4 weight). Every rank asserts both updated parameter
values. It passed 5/5 on all four ranks, so the four-rank weighting is proven
rather than assumed.

## Static checks

- `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` -> 0,
  exit 0 (re-run locally after the remote gates).
- `git diff --check` clean; worktree clean at
  `55171e659b2f425da980cbc83177bf5c38f0a7a1`.

## Notes and limitations

- After the gates, the local machine briefly lost VPN/DNS access to deathstar
  while re-querying the remote package versions and launcher path. All test
  output had already been streamed to the thesis repository, so no gate
  evidence was lost. The user restored the VPN and the metadata query was
  completed: Optimisers 0.4.9, MPI 0.20.27, Flux 0.16.11, launcher
  `/home/picazio/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec`
  (same artifact hash as local). The three remote source/test md5s match the
  local worktree exactly. No test was re-run and no test failed.
- No source or test changes were made in this phase; Flux stayed at
  `55171e65`, tests-only for Phase E.
- No native GPU/NCCL run, no detector conversion (Phase G), no compatibility
  review (Phase H).

## Evidence

`artifacts/logs/conditional-wrapper/phaseF-precompile-deathstar-2026-09-30.log`,
`phaseF-conditional-mpi2-deathstar-2026-09-30.log`,
`phaseF-conditional-mpi4-deathstar-2026-09-30.log`,
`phaseF-optimizer-mpi2-deathstar-2026-09-30.log`,
`phaseF-suite-nprocs2-deathstar-2026-09-30.log`,
`phaseF-suite-nprocs4-deathstar-2026-09-30.log`,
`phaseF-ambiguities-2026-09-30.log`,
`phaseF-diff-check-2026-09-30.log`,
`phaseF-versions-deathstar-2026-09-30.log`,
`phaseF-metadata.md`.

## Next action

Phase G: convert `scripts/checks/conditional_deadlock_check.jl` and the probe
verdicts into a success regression, update the Makefile target/comments, and
store new detector logs under a new name (keep the old characterization log).

# Phase F metadata: CPU/MPI gates at two and four ranks

Date: 2026-09-30 (runs 17:44-17:56 CEST)
Host: `deathstar` (Ubuntu workstation; Ryzen Threadripper, 64 threads, 125 GiB
RAM; 3x NVIDIA TITAN RTX present and healthy per `nvidia-smi`, but all Phase F
runs are CPU/MPI only)
Flux worktree on deathstar: `/home/picazio/projects/flux-integration`
Local source of truth: `/home/kurapica/Projects/ddp_flux/flux-integration`
Flux branch: `ddp/integration-m1-spike`
Flux commit: `55171e659b2f425da980cbc83177bf5c38f0a7a1` ("Add Phase E conditional
wrapper tests and restore M1.1 API checks"), clean worktree
Julia: 1.12.6 (juliaup; verified on deathstar)
Launcher: `/home/picazio/.julia/bin/mpiexecjl` ->
`/home/picazio/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec`
(MPICH/Hydra; same artifact hash as the local machine)
Packages (verified on deathstar via `pkgversion`): Optimisers 0.4.9, MPI 0.20.27,
Flux 0.16.11. Optimisers 0.4.9 matches the local evidence environment
(`/home/kurapica/.julia/packages/Optimisers/tMaaf`). The implementation uses
public Optimisers APIs only.
Thesis working tree: dirty (this evidence, wiki, README, plan updates).

## Synchronization (manual; `sync_code.sh` syncs the wrong worktree)

`scripts/remote/sync_code.sh` syncs `../ddp_flux` (the frozen dirty C8 master),
not the conditional-wrapper worktree. To run Phase F against the correct
project, two manual rsyncs were used:

```bash
rsync -a --delete --exclude='.git' --exclude='.julia' \
  --exclude='test/Manifest.toml' \
  /home/kurapica/Projects/ddp_flux/flux-integration/ \
  deathstar:~/projects/flux-integration/

rsync -a --delete --exclude-from='.rsyncignore' \
  /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/ \
  deathstar:~/projects/ddp-flux-thesis/
```

Remote source hashes verified after the sync (and re-verified after the gates):
`src/distributed/public_api.jl` md5 `06c2b464a5d9c1c784e07cb7752a0e7e`
(matches the D5 correction record and the local hash),
`test/ext_distributed/conditional_distributedtest.jl` md5
`c26ef49b070bf2c446b827c64cc313c0`, `test/ext_distributed/optimizer_distributedtest.jl`
md5 `9b2749dea1d96a99a8a419bfbe0b02b8` — all equal to the local worktree.

## Precompile (deathstar is the run machine; no login/compute split)

```bash
cd ~/projects/flux-integration
julia --startup-file=no --project=test -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

- Exit 0, real 2 m 7.5 s; 128 dependencies precompiled in 115 s (Enzyme,
  Mooncake, Zygote, FluxMPIExt, FluxEnzymeExt, ...). Remaining packages
  already precompiled.
- Log: `phaseF-precompile-deathstar-2026-09-30.log`.

## Focused conditional file

```bash
cd ~/projects/flux-integration
time env FLUX_TEST_DISTRIBUTED_BACKEND=mpi \
  $HOME/.julia/bin/mpiexecjl --project=test -n 2 \
  julia --startup-file=no --project=test \
  test/ext_distributed/conditional_distributedtest.jl

time env FLUX_TEST_DISTRIBUTED_BACKEND=mpi \
  $HOME/.julia/bin/mpiexecjl --project=test -n 4 \
  julia --startup-file=no --project=test \
  test/ext_distributed/conditional_distributedtest.jl
```

| Ranks | Exit | Wall | Testsets/rank | Per rank | Failures |
|---|---|---|---|---|---|
| 2 | 0 | 1 m 3.6 s | 30 | 367/367 | 0 |
| 4 | 0 | 1 m 7.9 s | 30 | 367/367 | 0 |

Logs: `phaseF-conditional-mpi2-deathstar-2026-09-30.log`,
`phaseF-conditional-mpi4-deathstar-2026-09-30.log`.

Four-rank weighting proof: the first testset
(`equal-shaped conditional branches do not cross-reduce`,
`conditional_distributedtest.jl:27-69`) gives rank 0 the `a` branch and ranks
1-3 the `b` branch. The expected global means embedded in the test are
`ga = 2*(a - ta)/4` (one active rank, 1/4 weight) and
`gb = 2*(b - tb)*3/4` (three active ranks, 3/4 weight), and the updated values
are asserted on every rank. It passed 5/5 on all four ranks, so the four-rank
analytical weighting is proven. All other testsets also passed at 4 ranks
(367/367 per rank), including the mixed tied/isbits, tied real-AD, Adam
reference, absent-Adam, collective-protocol, and Phase E freeze/adaptation
cases.

## Focused optimizer file, two ranks

```bash
time env FLUX_TEST_DISTRIBUTED_BACKEND=mpi \
  $HOME/.julia/bin/mpiexecjl --project=test -n 2 \
  julia --startup-file=no --project=test \
  test/ext_distributed/optimizer_distributedtest.jl
```

- Exit 0, 8.5 s, composition guard 6/6 per rank.
- Log: `phaseF-optimizer-mpi2-deathstar-2026-09-30.log`.

## Complete dedicated suite (direct runner; not wrapped in another mpiexec)

```bash
time env JULIA_MPI_TEST_NPROCS=2 FLUX_TEST_DISTRIBUTED_TIMEOUT=600 \
  julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi

time env JULIA_MPI_TEST_NPROCS=4 FLUX_TEST_DISTRIBUTED_TIMEOUT=600 \
  julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi
```

| Ranks | Exit | Wall | Suite total | Child files |
|---|---|---|---|---|
| 2 | 0 | 1 m 35.4 s | `Distributed \| 6 6 1m33.5s` | 6/6 |
| 4 | 0 | 1 m 45.6 s | `Distributed \| 6 6 1m43.6s` | 6/6 |

Logs: `phaseF-suite-nprocs2-deathstar-2026-09-30.log`,
`phaseF-suite-nprocs4-deathstar-2026-09-30.log`. Both logs contain
"Running Distributed Tests with 2/4 processes" and zero `Fail`/`Error` lines.

`FLUX_TEST_DISTRIBUTED_TIMEOUT=600` was set only for these two runs: the
default per-child watchdog is 120 s, and this was the first suite run on a new
machine (test-file JIT is not covered by package precompilation). The run did
not approach the watchdog (suite 1 m 33.5 s / 1 m 43.6 s), so the override did
not hide any failure; it only avoided a machine-warmup flake.

## Static checks (local)

```bash
julia --startup-file=no --project=test -e \
  'using Flux, Test, Optimisers; a = Test.detect_ambiguities(Flux.DistributedUtils; recursive=false); println("ambiguities = ", length(a)); @assert isempty(a)'
git diff --check
```

- Ambiguities 0, exit 0 (`phaseF-ambiguities-2026-09-30.log`).
- `git diff --check` clean, worktree clean, HEAD
  `55171e659b2f425da980cbc83177bf5c38f0a7a1`
  (`phaseF-diff-check-2026-09-30.log`).

## Connectivity note

Directly after the suite runs, an attempt to record the remote package
versions, launcher artifact path, and remote source hashes failed because the
VPN/DNS to `deathstar` dropped (all gate logs had already been streamed to the
thesis repository, so no gate evidence was lost). The user re-established the
VPN, and the metadata query was completed at 18:05 CEST: packages, launcher
path, and the three source hashes above are now verified. The updated output is
in `phaseF-versions-deathstar-2026-09-30.log` (this file was overwritten with
the complete query). No test was re-run.

One incidental observation: neither the local nor the remote flux-integration
worktree persists a `test/Manifest.toml`; Julia 1.12 resolves this environment
in memory from `test/Project.toml` (`[sources] Flux = {path = ".."}`) and the
depot, which is why `Pkg.instantiate()`/`precompile()` succeed and the gates
run identically on both machines. This matches the pre-existing local setup.

## Acceptance mapping (plan section 20)

- Focused two-rank conditional tests pass: 367/367 per rank, exit 0.
- Focused four-rank conditional tests pass: 367/367 per rank, exit 0.
- Complete two-rank distributed suite passes: 6/6, exit 0.
- Complete four-rank distributed suite passes: 6/6, exit 0.
- Four-rank analytical weighting: proven by the first testset at 4 ranks.
- Ambiguity count zero; `git diff --check` clean.
- Protocol order/counts: covered by the in-file recording/collective testsets
  at both 2 and 4 ranks.
- Not in Phase F: detector conversion (Phase G), compatibility review
  (Phase H), native GPU/NCCL.

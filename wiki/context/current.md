# Current context

Date: 2026-09-03
Active checkpoint: C8 - Correctness battery (precompile experiment on branch `experiment/precompile`)
Active Flux branch: `master` (at `b633cdc7`, dirty: C8.2 test expansion in progress)
Flux repo path: `/home/kurapica/Projects/ddp_flux/ddp_flux`  
Preserved reference branch: `upsteam-pr`
PR #2694 head: `bc40df25` (rebased `upsteam-pr` on upstream `404ff37d`, 2 commits, mergeable)

## What is known

- The local Flux.jl fork exists.
- The first implementation priority was understanding the current distributed code (C0-C6b done).
- CPU/MPI verification and upstream PR prep is mostly done.
- C7 is complete: an end-to-end example `scripts/examples/train_ddp.jl` was successfully created, executed on HPC using `make example-ddp`, and added as a documentation guide (`distributed.md`) to the Flux upstream docs.
- HPC GPUs are reachable via `salloc --gres=gpu:2` + `mpiexecjl` (NOT via `run.sh` which wraps in `bash -l` that resets env).
- HPC login node (slnode01) ≠ GPU compute node (gnode01); salloc grants allocation but `bash -c` runs on login node; use `mpiexecjl`/`srun` to launch on GPU nodes.
- **Precompilation (2026-09-03, branch `experiment/precompile`):** julia caches bake the host CPU feature set. Login node (graniterapids) and compute nodes (gnode01/02, icelake-server, mutually compatible) do NOT share caches. Old workflow precompiled on the login node (~10 min per Flux change, wasted) and the first compute-node test after a source change silently recompiled for ~4 min inside the test watchdog -> "timeouts on precomp". New workflow: precompile ONLY on compute nodes via `srun` (`scripts/remote/precompile.sh hpc`, now runs `make precompile-all FLUX_REPO_PATH=../ddp_flux`), `make env` never precompiles, `JULIA_PKG_PRECOMPILE_AUTO=0` in hpc.conf. Validated end-to-end on hpc: warm 2-rank suite 4m25 with zero in-test compilation; ADR-0005 + devlog `2026-09-03-precompile-experiment.md`. CPU-target pinning (`JULIA_CPU_TARGET`/`-C`) was tested and rejected (env var ignored at load; gnode images carry `pconfig` absent on login).

## What was done last (2026-09-03) — precompile experiment

- Phase A audit + Phase B baseline + Phase C mitigation experiments + Phase D
  implementation + Phase E end-to-end validation on hpc (details: `wiki/devlog/2026-09-03-precompile-experiment.md`, ADR-0005, branch `experiment/precompile`).
- Local uncommitted changes on `experiment/precompile`: Makefile (env/precompile/precompile-flux-test/precompile-all/install-mpiexec), scripts/remote/{precompile.sh,setup_node.sh,hosts/hpc.conf}, README.md, wiki/{context/remote-nodes.md, decisions/ADR-0005-*, devlog/2026-09-03-*}.

## What was done before (2026-08-17) — PR #2694 frozen-branch restore + rebase

- Pushed the frozen branch `upsteam-pr` to the PR head `ddp/upstream-pr`,
  restoring the CI-enabling `test/runtests.jl` default, the `distributed.md`
  guide, and its `docs/make.jl` registration.
- Rebased the PR onto current upstream master (`404ff37d`); resolved the only
  conflict (`NEWS.md`) by merging our distributed section with upstream's
  Unreleased bullets.
- PR now mergeable (`mergeable: true`); CI running.
- Local backups: `backup/pr2694-old-head` (`df95f602`),
  `backup/upsteam-pr-pre-rebase` (`504e816e`).
- Details: `wiki/devlog/2026-08-17-pr2694-restore.md`.

## What was done before (2026-08-01) — C8.3 NCCL verification

- **Fixed `check_cross_rank_sync` in `helper.jl`:**
  - Was calling `MPI.Bcast!` with `backend.comm` (NCCL.Communicator), not MPI.Comm.
  - Switched to `DistributedUtils.bcast!(backend, ref; root=0)` which dispatches correctly for both MPI and NCCL backends.
- **NCCL+GPU verification on HPC (gnode01, 2 ranks, 2x A100):**
  - `end_to_end_distributedtest.jl nccl`: **78/78 PASS**
  - `data_distributedtest.jl nccl`: **116/116 PASS**
  - `reduce_distributedtest.jl nccl`: **1/1 PASS**
- **Pre-existing NCCL failures (not C8.3 related):**
  - `common_distributedtest.jl`: 11/17 (6 errors — pre-existing NCCL comm pattern mismatches)
  - `synchronized_distributedtest.jl`: 4/4? (needs investigation)
  - `optimizer_distributedtest.jl`: fails at line 101 (pre-existing)
  - `deadlock_distributedtest.jl`: ProcessExited(1) (known issue)

## Commands that pass

- `mpiexecjl -n 2 julia --project=... end_to_end_distributedtest.jl mpi` (local, 78/78)
- `mpiexecjl -n 4 julia --project=... end_to_end_distributedtest.jl mpi` (local, 78/78)
- `FLUX_TEST_DISTRIBUTED_NCCL=true salloc --gres=gpu:2 ... mpiexecjl -n 2 julia --project=... end_to_end_distributedtest.jl nccl` (HPC, 78/78)
- `make example-ddp` (HPC and local)
- HPC precompile workflow (validated 2026-09-03): `scripts/remote/sync_code.sh hpc` -> `scripts/remote/setup_node.sh hpc` (resolve-only, no compile) -> `scripts/remote/precompile.sh hpc` (compute node, ~2-10 min) -> `NTASKS=2 CPUS_PER_TASK=4 scripts/remote/run.sh hpc "make c8-mpi FLUX_REPO_PATH=../ddp_flux"` (warm, 4m25, 7/8 PASS)
- `make precompile` (local)

## Open questions

- `test/Project.toml` cannot resolve with CUDA/NCCL added (StridedViews conflict). Fix needed before CI can run the NCCL path from the test project.
- Pre-existing NCCL test failures in common/synchronized/optimizer/deadlock tests need root cause analysis.
- Deathstar GPU state: GPU0000:43:00.0 unavailable; GPU tests cannot run on deathstar.
- Optional hardening for the precompile workflow: pre-flight `using Flux` warm probe / hash stamp so a stale cache cannot silently recompile inside the test watchdog.

## Next exact action

- Review + commit branch `experiment/precompile` (Makefile, scripts/remote, docs) if approved; then apply the "precompile.sh hpc after every sync" rule to the next C8/HPC test session.

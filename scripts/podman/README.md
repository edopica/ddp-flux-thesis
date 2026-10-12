# Running the PR #2694 distributed CI suite under rootless podman

Agent-oriented runbook. The container runtime is **rootless podman** — no
docker, no sudo, no group membership needed. The image `ddp-ci-repro:1.12.7`
and depot volume `ddp-ci-depot` are already migrated.

## Rules

- Use `scripts/podman/run-ci-parity.sh` for **every** container command. It
  emulates the GitHub runner topology (4 vCPU over 2 physical cores) by
  mounting an hwloc topology file and pinning `taskset -c 0-3`. This is
  required for the 4-rank job to exercise the OpenMPI slot condition.
- Do **not** use `docker`, `sudo`, `newgrp`, plain `podman run`, or
  `--cpuset-cpus` for the suite. Rootless podman cannot do cpuset; the wrapper
  exists for that reason.
- Run the suite against the **staging copy** only. Never modify
  `/home/kurapica/Projects/ddp_flux/flux-pr2694-salvage` or `ddp_flux`.
- Capture real exit codes (do not let `tee` mask them) and save logs under
  `temp/docker_ci_fix/podman/`.

## Setup — stage the current PR content

```bash
cd /home/kurapica/Projects/ddp_flux/ddp-flux-thesis
rsync -a --delete --exclude .git \
  /home/kurapica/Projects/ddp_flux/flux-pr2694-salvage/ \
  /tmp/opencode/docker_ci/repo-salvage/
```

## Job `mpi_fast` — 2 ranks, plain system launcher

```bash
scripts/podman/run-ci-parity.sh julia --project=test -e \
  'using Pkg; Pkg.add("MPIPreferences"); using MPIPreferences; MPIPreferences.use_system_binary()'
scripts/podman/run-ci-parity.sh julia --project=test -e 'using Pkg; Pkg.instantiate()'
scripts/podman/run-ci-parity.sh julia --project=test -e 'using MPI; MPI.install_mpiexecjl(force=true)'
scripts/podman/run-ci-parity.sh env JULIA_MPI_TEST_NPROCS=2 FLUX_TEST_DISTRIBUTED_TIMEOUT=1200 \
  julia --project=test test/ext_distributed/runtests.jl
```

## Job `mpi_edge` — 4 ranks, `--use-hwthread-cpus`

```bash
scripts/podman/run-ci-parity.sh julia --project=test -e \
  'using MPIPreferences; MPIPreferences.use_system_binary(mpiexec=`mpiexec --use-hwthread-cpus`)'

# probe step (workflow equivalent)
scripts/podman/run-ci-parity.sh bash -lc '
  mpiexec --version
  nproc; taskset -cp $$
  julia --project=test -e "using MPI; println(MPI.mpiexec()); println(MPI.MPI_LIBRARY)"
  mpiexec --use-hwthread-cpus -n 4 true && echo "4-rank launch OK"'

scripts/podman/run-ci-parity.sh env JULIA_MPI_TEST_NPROCS=4 FLUX_TEST_DISTRIBUTED_TIMEOUT=1200 \
  julia --project=test test/ext_distributed/runtests.jl
```

## Restore — leave the staging repo as found

```bash
scripts/podman/run-ci-parity.sh julia --project=test -e \
  'using MPIPreferences; MPIPreferences.use_system_binary()'
```

## Expected results

- Both harness runs: exit 0, `Distributed | 5 5` (5/5), ~40-65 s each after
  precompilation. No `not enough slots` banner.
- 2-rank probe: `nproc` = 4, affinity `0-3`.
- 4-rank probe: launcher `` `mpiexec --use-hwthread-cpus` ``, `MPI_LIBRARY =>
  OpenMPI`, `4-rank launch OK`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ERROR: file /home/runner/.julia/bin/mpiexecjl already exists` | Persistent depot (CI runners are fresh). Use `MPI.install_mpiexecjl(force=true)`. |
| `nproc` = 12 inside the container | Wrapper not used (topology not emulated). Use `scripts/podman/run-ci-parity.sh`. |
| `error: podman image 'ddp-ci-repro:1.12.7' not found` | Image missing from podman storage; escalate to the orchestrator (rollback copy is still in docker storage). |
| First run compiles for minutes | Precompile after a source change; wait, do not kill. The per-file watchdog is `FLUX_TEST_DISTRIBUTED_TIMEOUT` (1200 s). |
| `--cpuset-cpus` / `taskset` outside podman | Do not use. See `wiki/failures/2026-09-10-podman-taskset-cpuset.md`. |

## Overrides

`REPO` (staging path), `IMAGE`, `DEPOT`, `CPUS`, `REGEN=1` (regenerate the
topology XML), `NATIVE_CPUSET=1` (after the optional cpuset delegation — see
`temp/docker_ci_fix/RESULTS-podman.md` Plan B).

## References

- Full migration evidence and exact commands: `temp/docker_ci_fix/RESULTS-podman.md`
- Findings: `wiki/devlog/01-pr2694/2026-09-10-rootless-podman-ci-parity.md`
- Failure record (why taskset/cpuset cannot be used): `wiki/failures/2026-09-10-podman-taskset-cpuset.md`
- Runner source and topology generator: `scripts/podman/run-ci-parity.sh`, `scripts/podman/topo-gen.c`

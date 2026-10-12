# 2026-09-10 — Rootless podman CI parity validation (rootful-docker → rootless-podman migration)

Goal: produce raw validation evidence that the PR #2694 distributed-test CI can run under rootless podman, and that the CI's four-rank OpenMPI slot condition (`--use-hwthread-cpus`) is exercised, not bypassed.

Environment: rootless podman 6.1.0; image `ddp-ci-repro:1.12.7` (Ubuntu 24.04, OpenMPI 4.1.6, Julia 1.12.7); depot volume `ddp-ci-depot`; staging `/tmp/opencode/docker_ci/repo-salvage`; container `--userns=keep-id` (runner uid 1000 = host uid 1000); host 12 CPUs / 6 cores / 2 threads-per-core, CPUs 0-3 = 2 physical cores × 2 threads (GitHub runner topology). Evidence: `temp/docker_ci_fix/podman/`.

## Finding 1 — outside `taskset` does not restrict the container

`taskset -c 0-3 podman run ... taskset -cp 1` → container PID 1 affinity `0-11`; `nproc` = 12 (not 4). Cause: rootless podman's `conmon` is forked by the systemd user manager (parent PID 925), not by the taskset-restricted client, so the affinity mask is not inherited. Log: `podman-R0-topology-cpuset.log`.

## Finding 2 — `taskset` inside the container restricts affinity but OpenMPI ignores it

`podman run ... taskset -c 0-3 bash -c 'nproc; taskset -cp $$'` → 4 / `0-3` (`podman-R0b-topology-inside-taskset.log`). But with affinity 0-3, plain `mpiexec -n 4 true` still exits 0: OpenMPI 4.1.6 counts slots from the hwloc machine topology, not from the process affinity mask. So `taskset` alone cannot reproduce GitHub's "not enough slots" failure.

## Finding 3 — real cgroup cpuset restriction is impossible rootless

- `podman run --cpuset-cpus=0-3` → runc error, `cpuset.cpus: no such file or directory` (exit 127).
- `user@1000.service/cgroup.controllers` = `cpu memory pids` (no cpuset); writing `+cpuset` to `cgroup.subtree_control` → ENOENT.
- `systemd-run --user --scope -p AllowedCPUs=0-3` is silently ignored (`nproc` 12, cpuset effective `0-11`).
Log: `podman-R0c-cpuset-unavailable.log`.

## Finding 4 — working emulation: OpenMPI `hwloc_base_topo_file`

OpenMPI exposes MCA `hwloc_base_topo_file` ("read local topology from file instead of directly sensing it"). Generated a restricted topology XML (2 cores / 4 PUs from CPUs 0-3) with a small C program against the container's `libhwloc` (`/tmp/opencode/podman_migrate/topo-gen.c` → `topo-2c4t.xml`). Probe in the container:

- plain `mpiexec -n 4` → "There are not enough slots available…", exit 1;
- plain `mpiexec -n 2` → exit 0;
- `mpiexec --use-hwthread-cpus -n 4` → exit 0.

Log: `podman-R0d-hwloc-topo-emulation.log`. This reproduces the CI condition exactly at the hwloc/OpenMPI level. Deviation from the planned method: the container command wraps the workload with `taskset -c 0-3` internally and sets `OMPI_MCA_hwloc_base_topo_file=/mnt/topo-2c4t.xml`; the topology file is bind-mounted read-only.

## Results (all real exit codes; wall via `SECONDS`)

| Step | Command (abridged) | Log | Exit | Wall | Result |
|---|---|---|---|---|---|
| R0 | outside taskset topology proof | `podman-R0-topology-cpuset.log` | 0 | 0s | nproc=12, affinity 0-11 — taskset NOT propagated |
| R0b | inside taskset | `podman-R0b-topology-inside-taskset.log` | 0 | - | nproc=4, affinity 0-3 |
| R0c | cpuset availability | `podman-R0c-cpuset-unavailable.log` | - | - | cpuset controller not delegated |
| R0d | hwloc topo emulation probe | `podman-R0d-hwloc-topo-emulation.log` | 0 | - | plain 4 fails, plain 2 ok, hwt 4 ok |
| R1a-literal | exact planned command, outside taskset, 4 ranks | `podman-R1a-literal-outside-taskset-4rank.log` | 0 | 44s | **5/5 Pass** — negative control NOT reproduced (invalidates taskset-only parity) |
| R1a | plain prefs + topo + inside taskset, 4 ranks | `podman-R1a-negative-4rank-cpuset.log` | **1** | 8s | 0 passed / 5 failed; "not enough slots" ×5 |
| R2b-1 | `MPIPreferences.use_system_binary(mpiexec=`mpiexec --use-hwthread-cpus`)` | `podman-R2b-1-cmd-prefs-config.log` | 0 | 4s | prefs written |
| R2b-2 | `MPI.mpiexec()` identity | `podman-R2b-2-mpi-identity.log` | 0 | 10s | `mpiexec --use-hwthread-cpus`, `MPI_LIBRARY => OpenMPI` |
| R2b-3 | cmd prefs + topo + inside taskset, 4 ranks | `podman-R2b-3-positive-4rank-cpuset.log` | 0 | 59s | **5/5 Pass**, no slot errors |
| R3-0 | restore plain prefs | `podman-R3-0-restore-plain-prefs.log` | 0 | 3s | `mpiexec = "mpiexec"` |
| R3 | plain prefs + topo + inside taskset, 2 ranks | `podman-R3-control-2rank-cpuset.log` | 0 | 51s | **5/5 Pass** |
| R4-1 | re-apply cmd prefs (CI order) | `podman-R4-1-cmd-prefs-config.log` | 0 | 3s | prefs written |
| R4-2 | CI probe (version, topology, launcher, slots) | `podman-R4-2-probe.log` | 0 | 11s | "4-rank launch OK" |
| R4-3 | CI 4-rank job, `FLUX_TEST_DISTRIBUTED_TIMEOUT=1200` | `podman-R4-3-CI-4rank-cpuset.log` | 0 | 63s | **5/5 Pass**, no slot errors |
| R5 | restore plain prefs + md5 verify | `podman-R5-restore-prefs.log` | 0 | 4s | md5 `183acda9c034313d0b27f63abe08f2a5` identical to backup |

Staging left as found: only `test/LocalPreferences.toml` was written during the runs and it is byte-identical to the pre-session backup; `Manifest.toml`/`test/Project.toml` mtimes predate the session.

## Verdict

- Taskset-based cpuset parity **does not work** in this rootless-podman setup (Findings 1-3). The literal negative control passes 5/5.
- The hwloc topology-file emulation **does** reproduce the CI slot condition: negative fails with the OpenMPI banner, positive passes 5/5, 2-rank control passes 5/5, full CI 4-rank order passes 5/5.
- The rootless-podman migration is validated for the distributed suite under an emulated 2-core/4-thread topology. Real cgroup cpuset parity remains impossible without root/delegation; a rootful-docker run (gate 4) is still the only way to test with a real cpuset.

## Next action

Decide whether to (a) accept the hwloc-emulation evidence as podman migration validation, (b) run the same suite in a rootful environment (e.g., a VM or delegated cpuset) for true cpuset parity, or (c) formalize the emulation as a reusable script in `scripts/`.

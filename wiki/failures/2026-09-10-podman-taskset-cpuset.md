# Failure - rootless podman ignores `taskset` for cpuset parity

Date: 2026-09-10
Checkpoint: PR #2694 salvage phase 7 / gate 4 podman migration validation
Command: `taskset -c 0-3 podman run --rm --userns=keep-id ... ddp-ci-repro:1.12.7 nproc`
Backend: OpenMPI 4.1.6 (system) inside rootless podman 6.1.0
Rank count: 4
Hardware: host 12 CPUs (6 cores × 2 threads); CPUs 0-3 = 2 physical cores × 2 threads

## Symptom

The planned topology restriction did not apply. The container saw all 12 CPUs (`nproc` = 12, PID 1 affinity `0-11`), so the intended negative control (plain `mpiexec -n 4` failing with "not enough slots") could not be reproduced; the literal negative-control command passed 5/5.

## Full error or log excerpt

`temp/docker_ci_fix/podman/podman-R0-topology-cpuset.log`:

```
12
pid 1's current affinity list: 0-11
...
no cgroup cpuset file
EXIT=0
```

`temp/docker_ci_fix/podman/podman-R1a-literal-outside-taskset-4rank.log`:

```
Distributed   |    5      5  41.8s
EXIT=0
```

## Minimal reproduction

```bash
taskset -c 0-3 podman run --rm ddp-ci-repro:1.12.7 taskset -cp 1
# pid 1's current affinity list: 0-11
```

Secondary blockers (same session, `podman-R0c-cpuset-unavailable.log`):

```
podman run --cpuset-cpus=0-3 ...  -> runc: cpuset.cpus: no such file or directory (exit 127)
user@1000.service/cgroup.controllers -> cpu memory pids        # no cpuset
echo +cpuset > .../cgroup.subtree_control -> ENOENT
systemd-run --user --scope -p AllowedCPUs=0-3 -> nproc 12, cpuset.cpus.effective 0-11
```

## Suspected cause

1. Rootless podman starts `conmon` from the systemd user manager (observed parent PID 925), so the container process tree does not inherit the affinity of the `taskset`-wrapped podman client.
2. Even with affinity restricted inside the container (`nproc` = 4), OpenMPI 4.1.6 counts slots from the hwloc machine topology, not from the affinity mask; plain `mpiexec -n 4` still succeeds.
3. The cpuset cgroup controller is not delegated to the user session, so no real cpuset restriction can be created.

## Fix attempted

- `taskset` inside the container: restricts affinity but does not change OpenMPI slot counting.
- `--cpuset-cpus`, direct `cgroup.subtree_control` write, `systemd-run --user -p AllowedCPUs`: all unavailable without root.
- Working emulation: `OMPI_MCA_hwloc_base_topo_file` + hwloc-generated topology XML restricted to CPUs 0-3 (2 cores / 4 PUs). Plain `-n 4` then fails with the exact OpenMPI "not enough slots" banner (exit 1); `--use-hwthread-cpus -n 4` and plain `-n 2` succeed.

## Outcome

Suite validation completed with the emulation: negative control 0/5 with banner ×5 (exit 1); positive 4-rank 5/5 (exit 0); 2-rank control 5/5 (exit 0); CI-order 4-rank job 5/5 (exit 0). True cgroup-cpuset parity remains untested (rootless limitation).

## Thesis lesson

`taskset` and cgroup cpusets are not interchangeable for MPI slot detection: OpenMPI/hwloc reads the machine topology, not the affinity mask. Emulating a smaller machine requires either a cgroup cpuset (needs delegation/root) or an explicit hwloc topology file. Always verify the restriction inside the container (`taskset -cp 1`, `nproc`) before trusting a negative control.

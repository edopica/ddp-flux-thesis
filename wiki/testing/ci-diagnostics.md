# CI Diagnostics Guide — best practices and failure examples

Source: the PR #2694 4-rank CI failure (2026-08 → 2026-09-06). Evidence:
`temp/docker_ci/RESULTS.md`, `temp/handoff-2026-09-05-pr2694-ci-investigation.md`,
CI logs in `temp/ci-full-logs-32582844912/`.

This guide exists because CI diagnostics will become routine as the project
proceeds. It is split into: (1) the incident in one paragraph, (2) what we did
wrong — each anti-pattern with the rule it produces, (3) a best-practices
checklist, (4) copy-paste templates, (5) pointers to primary evidence.

## 1. The incident in one paragraph

PR #2694's CI job "MPI DDP Sharding & Asymmetry (4 Ranks)" failed on
GitHub-hosted ubuntu-latest: all six distributed test files exited 1 in under a
second at `mpiexec -n 4`, while the sibling `-n 2` job passed. Local runs and
deathstar (both with MPI.jl jll binaries, 64-core host) passed 6/6. Root cause:
OpenMPI defaults its launchable slot count to the number of **physical cores**;
standard GitHub runners expose 4 vCPU over 2 physical cores, so `mpiexec -n 4`
aborts before launching anything ("There are not enough slots available..."). The
harness ran children with `run(cmd, wait=false)`, which sends child I/O to
devnull, so the error text was structurally invisible in CI. The failure was
reproduced locally only when the docker container was restricted with
`--cpuset-cpus=0-3` (hwloc then sees 2 physical cores), and diagnosed via a
direct child launch with stderr attached.

## 2. What we did wrong (anti-patterns → rules)

### 2.1 The harness swallowed child output
`runtests.jl:36` used `proc = run(cmd, wait=false)` with no pipeline. In Julia
1.12, `wait=false` directs the child's I/O streams to devnull. Consequence:
every CI failure is an exit code with no text. The passing 2-rank job suffered
the same blindness; nothing distinguished "fine" from "lucky".

Rule: **a test harness must capture child stdout/stderr and print them when the
child exits non-zero.** Never launch children whose streams you cannot read back
on failure. See template 4.1.

### 2.2 We assumed `mpiexec -n 4` can launch on ubuntu-latest
Nobody verified the runner can host 4 ranks. `nproc` reports 4, but OpenMPI
slots default to physical cores (2), not hardware threads (4). GitHub does not
disclose runner core/thread topology — it must be measured, not assumed.

Rule: **probe the infrastructure before running the real workload.** A
`mpiexec -n <N> true` step in the job costs seconds and would have caught this
instantly. Record topology (`lscpu`, `nproc`, hwloc view) in every job log.

### 2.3 We tested where CI is not
Green on local (MPICH_jll) and deathstar (OpenMPI_jll 4.1.9, MPItrampoline
launch, 64 cores) built confidence that did not transfer: CI uses **system**
OpenMPI 4.1.6 (Debian) with direct dlopen (`MPIPreferences.use_system_binary()`)
on a 2-physical-core VM. Three deltas (flavor, version, topology) — any of them
could matter, and none was exercised before CI.

Rule: **keep an environment-delta table** (MPI flavor, version, launcher stack,
CPU topology, prefs mode) for every place you run. "Green elsewhere" is a
hypothesis, not proof. The env closest to CI is the one that decides.

### 2.4 No CI-faithful local repro path existed at failure time
Docker was off-limits locally, deathstar was down, and diagnostic CI runs were
forbidden. Diagnosis therefore had to wait for an environment that could fail.

Rule: **build the parity repro environment early** (see 4.3) and know its
limits. Docker parity finding from this incident: `--cpus=4` throttles CPU time
but does **not** shrink the hwloc topology (container still sees all host PUs →
slots unchanged → false green). `--cpuset-cpus=0-3` confines the topology and
was the condition that reproduced CI exactly.

### 2.5 CI runs were treated as expensive instead of as probes
Each re-run costs ~half a day of wall time and was (rightly) gated. But the
workflow had no diagnostic mode: no `workflow_dispatch` inputs, no
infrastructure probe step, no artifact upload of child logs. A single manual
"diagnose" job would have shortened the incident by weeks.

Rule: **make diagnosis a first-class, cheap path**: `workflow_dispatch` with
inputs, a probe step that runs before tests, artifacts on failure, and local
download of raw logs (`gh run view --log`) before they expire.

### 2.6 What we did right (keep doing)
- Timing signatures: 6/6 failures in ≤1 s, content-independent, n-dependent ⇒
  launch-level failure, not test logic. This reasoning was correct and steered
  everything that followed.
- Evidence discipline: exact commands recorded, logs saved, runtests.jl pinned
  by md5, handoffs written between sessions.
- Ranking hypotheses with falsifiability instead of speculative code fixes.

## 3. Best-practices checklist

### 3.1 Harness (Julia)
- [ ] Children launched with captured stdout/stderr (pipeline to buffers/files).
- [ ] On non-zero child exit: print captured output and per-file elapsed time.
- [ ] Watchdog distinguishes "watchdog killed" from "child failed" in the report.
- [ ] Log line per file before launch ("Running X with N processes") and a Test
      Summary that can be diffed across runs.
- [ ] Precompile is a separate step from the timed test; log files show no
      "Precompiling..." inside the timed step.

### 3.2 CI job (GitHub Actions)
- [ ] Infra probe step before tests: MPI identity, versions, topology, slots
      probe for the exact `-n` used (`mpiexec -n N true`).
- [ ] Environment recording step: `mpirun --version`, `ompi_info` key lines,
      `nproc` + physical core count, `MPI.MPI_LIBRARY`/`MPI.mpiexec()` from
      MPI.jl, `OMPI_*`/`PMI*`/`SLURM_*` env vars, prefs mode
      (system binary vs jll).
- [ ] Artifacts uploaded on failure (full raw logs), and raw logs saved locally
      right after the run (they expire after ~90 days).
- [ ] A `workflow_dispatch` diagnostic job exists (inputs: nprocs, backend,
      extra flags) so probes do not require commits.
- [ ] Control runs kept: the smallest and largest rank counts on the same job
      (2-rank green vs 4-rank red was the decisive contrast).
- [ ] Comments in the workflow explain every non-obvious flag (e.g., why
      `--use-hwthread-cpus` is present).

### 3.3 MPI specifics
- [ ] Remember: slots ≠ `nproc`. OpenMPI defaults slots to physical cores when
      no hostfile/RM is present. On unknown runner topology either pass
      `--use-hwthread-cpus` (count hardware threads; preferred — keeps a real
      limit and is self-documenting) or `--oversubscribe` (ignore limits), or
      clamp `-n` to verified slots. Document the choice in the workflow.
- [ ] Record MPI flavor/version/launcher stack per run; system MPI (direct
      dlopen) and jll MPItrampoline stacks are different worlds.
- [ ] When a run dies in ≤ ~1-2 s across all files: launch/env-level problem —
      check slots, launcher, PMIx before touching test code.
- [ ] Hangs (not crashes) are where DDP ordering bugs live: collectives must be
      called in the same order by every rank; debug those with timestamps per
      rank, not by reading code.

### 3.4 Reproduction discipline
- [ ] Reproduce with stderr visible in a parity environment **before** fixing
      code. No speculative fixes.
- [ ] One variable per experiment; record exact commands; save logs into
      `temp/` evidence dirs with an index (RESULTS.md pattern).
- [ ] Pin artifacts by hash where byte-identity matters (runtests.jl md5
      precedent).
- [ ] Know your parity limits in writing (docker cpuset vs `--cpus`, runner
      topology not disclosed, etc.).

## 4. Copy-paste templates

### 4.1 Julia harness: capture child output, print on failure

```julia
# Pattern — replace bare `run(cmd, wait=false)` + `wait(proc)`.
out = PipeBuffer(); err = PipeBuffer()
proc = run(pipeline(cmd, stdout=out, stderr=err), wait=false)
timed_out = false
while process_running(proc)
    sleep(0.5)
    if time() - t0 > timeout_s
        kill(proc); timed_out = true
    end
end
wait(proc)
if proc.exitcode != 0
    # This block is the whole point: without it, CI shows only an exit code.
    println("=== child failed (exit $(proc.exitcode))$(timed_out ? " — WATCHDOG" : "") ===")
    print(String(take!(out))); print(String(take!(err)))
end
```

### 4.2 GitHub Actions: infra probe step (run before tests)

```yaml
- name: Diagnose MPI environment
  run: |
    echo "== topology =="; nproc; lscpu | grep -E "^(CPU\(s\)|Core\(s\) per socket|Thread\(s\) per core|Socket\(s\))"
    echo "== MPI =="; mpirun --version | head -2; ompi_info | grep -iE "Open MPI:|Thread support" || true
    echo "== MPI.jl identity =="; julia --project=test -e 'using MPI; println("lib=", MPI.MPI_LIBRARY); println("mpiexec=", MPI.mpiexec())'
    echo "== slots probe for n=$JULIA_MPI_TEST_NPROCS =="
    mpiexec -n "$JULIA_MPI_TEST_NPROCS" true && echo "launch OK" || echo "LAUNCH FAILS at n=$JULIA_MPI_TEST_NPROCS"
```

### 4.3 Docker parity quick reference
Image `ddp-ci-repro:1.12.7` (ubuntu:24.04, apt OpenMPI 4.1.6-7ubuntu2, Julia
1.12.7) with repo at `/home/runner/Flux.jl` and depot volume `ddp-ci-depot`.
Full runbook and Dockerfile: `temp/docker_ci/RESULTS.md`, `temp/docker_ci/Dockerfile`.

- CPU-time throttling only (NOT topology-faithful): `--cpus=4`.
- Topology-faithful (needed to reproduce OpenMPI slots behavior): `--cpuset-cpus=0-3`.
- Direct child with stderr (the diagnostic that ended the investigation):
  `mpiexec -n 4 julia --color=yes --project=/home/runner/Flux.jl/test --startup-file=no test/ext_distributed/common_distributedtest.jl mpi`

## 5. Primary evidence

| Item | Path |
|---|---|
| Reproduction results + root cause | `temp/docker_ci/RESULTS.md` |
| Full investigation handoff (superseded hypotheses, env reference) | `temp/handoff-2026-09-05-pr2694-ci-investigation.md` |
| Real CI logs (failing step verbatim) | `temp/ci-full-logs-32582844912/full-job.log` |
| Captured child stderr (the slots banner) | `temp/docker_ci/child-common-cpuset-stderr.log` |
| Dockerfile / image id | `temp/docker_ci/Dockerfile`, `temp/docker_ci/image-id.txt` |
| Related guardrail work | `wiki/decisions/ADR-0003-mpi-pmi-guardrails.md` |

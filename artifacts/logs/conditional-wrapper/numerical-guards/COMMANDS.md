# DB-1/DB-4 numerical-guard campaign commands

Date: 2026-10-13. Scope: coordinated non-finite-loss rejection for the eager
distributed paths (`Flux.train!`, `Flux.trainstep!`, `Flux.trainstep_withgradient!`).
Plan: `reports/conditional-graph-db1-db4-fix-plan.md`. Decision:
`wiki/decisions/ADR-0008-coordinated-nonfinite-loss-rejection.md`.

## Source identity

| Item | RED (before fix) | GREEN (after fix) |
|---|---|---|
| Flux worktree HEAD | `55171e659b2f425da980cbc83177bf5c38f0a7a1` (dirty) | same HEAD, further dirty (no commit) |
| `src/train.jl` md5 | `eea4e545d3eb2118c3a5b23653c7a71a` | `42b85187852f455bfab1dcd8113d3a26` |
| `src/distributed/public_api.jl` md5 | `9bfa981384eaf88750ec3d19e8bb76bd` | `e4ed1f8d4e2aed25d99654ad05d54919` |
| New test file md5 | n/a | `test/ext_distributed/nonfinite_loss_distributedtest.jl` `5a0a41eaec5922b0fb944c645f49c4c8` |
| Probe md5 | n/a | `scripts/checks/conditional_graph_numerical_guard_probe.jl` `ea7e639e5c7b7d8efeaaeb4756a1fcbe` |
| Driver md5 | n/a | `scripts/checks/conditional_graph_numerical_guard_check.jl` (see run metadata) |

Local toolchain: Julia 1.12.6, MPICH Hydra 5.0.1, Flux 0.16.11, Optimisers 0.4.9,
MPI 0.20.27. Remote (deathstar): Julia 1.12.6; workspace manifest resolved
2026-10-04: CUDA 6.2.2, NCCL 0.1.3, cuDNN entry 6.2.2, Flux 0.16.11,
Optimisers 0.4.9, MPI 0.20.27.

## RED evidence (local, unmodified Flux source)

```
julia --startup-file=no --project=../flux-integration/test \
  scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group loss --ranks 2 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/red-local-loss-mpi2
```

Result: 13 cases, all FAIL for the expected reasons (9 asymmetric timeouts with
unmatched `COLLECTIVE_BEGIN`, `no-trainable` wrong participation, and
`train-all`/`healthy`/`aux-nonfinite-ok` missing VOTE stages). No startup
failure; no orphans. Artifacts: `red-local-loss-mpi2/`.

## GREEN evidence (deathstar)

Sync (root and test manifests are deliberately excluded so the remote-resolved
workspace manifest survives):

```
rsync -avz --delete --exclude-from='.rsyncignore' --exclude='artifacts/' \
  . deathstar:~/projects/ddp-flux-thesis/
rsync -avz --delete --exclude='.git' --exclude='Manifest.toml' \
  --exclude='test/Manifest.toml' ../flux-integration deathstar:~/projects/
```

Precompile (plain host, no scheduler):

```
scripts/remote/run.sh deathstar "make precompile-all FLUX_REPO_PATH=../flux-integration"
```

Harness gates (exact commands; run from the thesis root):

```
scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group loss --ranks 2 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-loss-mpi2"

scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group loss --ranks 4 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-loss-mpi4"

scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group loss --ranks 2 --backend nccl \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-loss-nccl2-r2"
```

| Gate | Result |
|---|---|
| Harness loss mpi2 | 13/13 PASS |
| Harness loss mpi4 (incl. 1 invalid rank of 4) | 8/8 PASS |
| Harness loss nccl2 | 12 PASS + 1 UNTESTED (`dup-train-one`; see failure note) |
| Harness loss nccl2 attempt 1 | 12 PASS + 1 FAIL (`dup-train-one` startup watchdog 420 s) — retained |
| Single-process `train` | 137/137 PASS |
| `detect_ambiguities(Flux.DistributedUtils)` | 0 |
| Dedicated suite mpi2 | 8/8 files PASS |
| Dedicated suite mpi4 | 8/8 files PASS |
| Dedicated suite nccl2 (mpi + nccl passes) | 16/16 PASS |

Suite / unit commands:

```
scripts/remote/run.sh deathstar "JULIA_MPI_TEST_NPROCS=2 FLUX_TEST_DISTRIBUTED_NCCL=false \
  julia --startup-file=no --project=../flux-integration/test \
  ../flux-integration/test/ext_distributed/runtests.jl"

scripts/remote/run.sh deathstar "JULIA_MPI_TEST_NPROCS=4 FLUX_TEST_DISTRIBUTED_NCCL=false \
  julia --startup-file=no --project=../flux-integration/test \
  ../flux-integration/test/ext_distributed/runtests.jl"

scripts/remote/run.sh deathstar "JULIA_MPI_TEST_NPROCS=2 FLUX_TEST_DISTRIBUTED_NCCL=true \
  julia --startup-file=no --project=../flux-integration/test \
  ../flux-integration/test/ext_distributed/runtests.jl"

scripts/remote/run.sh deathstar "FLUX_TEST_REACTANT=false julia --startup-file=no \
  --project=../flux-integration/test ../flux-integration/test/runtests.jl train"

scripts/remote/run.sh deathstar 'julia --startup-file=no --project=../flux-integration/test \
  -e ...detect_ambiguities(Flux.DistributedUtils; recursive=false)...'
```

No run printed `Precompiling packages...` (checked in every console log).

## v2 review-response campaign (2026-10-13)

External review of the v1 campaign
([report](../../../../reports/conditional-graph-db1-db4-review.md)) found two P2
harness issues and one minor coverage gap. All three are fixed and re-gated with
no Flux source change. The v1 sections above remain the historical record; this
section supersedes their evidence.

Harness changes:

- **Campaign-scoped cleanup (R1).** The filename-wide `reap_orphans()` is
  replaced by `reap_campaign(token)`. Every probe child now carries
  `--campaign TOKEN` (default `<pid>-<time_ns>`, override
  `NUMERICAL_GUARD_CAMPAIGN`), and cleanup matches only probe processes whose
  command line contains that token, so concurrent campaigns cannot reap each
  other. New `--selftest-cleanup` mode checks the isolation without launching
  MPI.
- **Conditional `healthy` gate (R2).** The case now trains a new `CondModel`
  (`w` len 2 always on; `u` len 3 branch U; `v` len 4 branch V) for four
  rank-distinct eager `train!` steps: (1) all branches; (2) even ranks U / odd
  ranks V; (3) all ranks U only, V globally absent after nonzero Adam history;
  (4) all branches, V reactivated. Each step compares parameters and the
  complete Adam leaf state (moment 1, moment 2, counter) against a native host
  reference updated with the global averaged gradient and prints
  `STEP_STATE_OK`; step 3 asserts `ABSENCE_FROZEN` and step 4 `REACTIVATED`.
  The driver requires every marker on every rank. Expected collective schedule
  per full step: `vote, presence, grad_w(2), grad_u(3), grad_v(4)`. The
  partially present step keeps both gradient reductions (the locally absent
  branch is zero-filled); only the globally absent step drops `grad_v`.
- **NamedTuple loss coverage.** `nonfinite_loss_distributedtest.jl` gained a
  NamedTuple-loss testset (now 9 testsets): a finite outer NamedTuple
  round-trips, and a poisoned `(loss = NaN32, acc = 1.0)` rejects on every rank
  with only the vote.

### Cleanup self-test and concurrency check (deathstar)

```
scripts/remote/run.sh deathstar "julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --selftest-cleanup"
```

Output (`selftest-cleanup-deathstar.log`):

```
SELFTEST_CLEANUP PASS pids_a=563275 pids_b=563276 killed_a=563275 tokA=selftest-A-563235 tokB=selftest-B-563235
```

Concurrency command pattern: campA (`--case healthy`, token `campA`, 1 s
execution watchdog) and campB (full loss mpi2, token `campB`, default watchdog)
launched at the same time on deathstar:

```
NUMERICAL_GUARD_PHASE=green NUMERICAL_GUARD_CAMPAIGN=campA \
  NUMERICAL_GUARD_EXECUTION_TIMEOUT=1 \
  julia --startup-file=no --project=../flux-integration/test \
  scripts/checks/conditional_graph_numerical_guard_check.jl --case healthy \
  --group loss --ranks 2 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/cleanup-concurrency-deathstar/campA &

NUMERICAL_GUARD_PHASE=green NUMERICAL_GUARD_CAMPAIGN=campB \
  julia --startup-file=no --project=../flux-integration/test \
  scripts/checks/conditional_graph_numerical_guard_check.jl --group loss \
  --ranks 2 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/cleanup-concurrency-deathstar/campB &
wait
```

Evidence (`cleanup-concurrency-deathstar-attempt2/`, renamed in round 2):
campB `summary: PASS=13 FAIL=0 UNTESTED=0` (its children were not killed by
campA); campA verdict FAIL with reason `watchdog timeout (execution) after
40.3s`, and both campA ranks reached `GUARD_PROBE_READY` before the kill;
`pgrep -f conditional_graph_numerical_guard_probe.jl` was empty after both
campaigns.
Documented deviation: the campA console has no `reaped orphan pids` line
because MPICH Hydra terminates its own ranks once `stop_probe` kills the
launcher, so `reap_campaign(campA)` found zero survivors; the token-scoped kill
itself is proven by the selftest. Recorded in the
[failure note](../../../../wiki/failures/2026-10-13-db1-db4-healthy-probe-fielderror.md).

### Harness gates (green2, deathstar)

```
scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group loss --ranks 2 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green2-deathstar-loss-mpi2"

scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group loss --ranks 4 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green2-deathstar-loss-mpi4"

scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group loss --ranks 2 --backend nccl \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green2-deathstar-loss-nccl2"
```

### Dedicated suite (v2, deathstar)

```
scripts/remote/run.sh deathstar "JULIA_MPI_TEST_NPROCS=2 FLUX_TEST_DISTRIBUTED_NCCL=false \
  julia --startup-file=no --project=../flux-integration/test \
  ../flux-integration/test/ext_distributed/runtests.jl"

scripts/remote/run.sh deathstar "JULIA_MPI_TEST_NPROCS=4 FLUX_TEST_DISTRIBUTED_NCCL=false \
  julia --startup-file=no --project=../flux-integration/test \
  ../flux-integration/test/ext_distributed/runtests.jl"

scripts/remote/run.sh deathstar "JULIA_MPI_TEST_NPROCS=2 FLUX_TEST_DISTRIBUTED_NCCL=true \
  julia --startup-file=no --project=../flux-integration/test \
  ../flux-integration/test/ext_distributed/runtests.jl"
```

Console logs: `suite-deathstar-v2/mpi2.log`, `mpi4.log`, `nccl2.log`.

### Attempt 1 failure and fix

The first v2 attempt failed `healthy` in all three gates with
`FieldError: type Array has no field 'u'` (`CuArray` under NCCL). Cause:
`cond_local_loss` passed `m.u`/`m.v` to `ul`/`vl`, which already take the
model. Fix: `ul(m, x)`/`vl(m, x)`. The failed attempt is retained as
`green2-attempt1-healthy-bug-deathstar-loss-mpi2|mpi4|nccl2/` and
`cleanup-concurrency-deathstar-attempt1/`. Full record:
[failure note](../../../../wiki/failures/2026-10-13-db1-db4-healthy-probe-fielderror.md).

### Results

| Gate | Result |
|---|---|
| Harness green2 loss mpi2 | 13/13 PASS (healthy 37.3 s) |
| Harness green2 loss mpi4 | 8/8 PASS (healthy 38.1 s) |
| Harness green2 loss nccl2 | 12 PASS + 1 UNTESTED (`dup-train-one`; healthy 60.5 s) |
| Suite v2 mpi2 | 8/8 files PASS (2m28.2s) |
| Suite v2 mpi4 | 8/8 files PASS (2m42.8s) |
| Suite v2 nccl2 (MPI + NCCL passes) | 16/16 PASS (6m12.7s) |
| `Precompiling packages...` lines | 0 in every v2 console log |

Healthy markers (all `ok=true`): mpi2 8 `STEP_STATE_OK` + 2 `ABSENCE_FROZEN` +
2 `REACTIVATED`; mpi4 16 + 4 + 4; nccl2 8 + 2 + 2.

### Source identity (v2)

| Item | md5 |
|---|---|
| Probe `scripts/checks/conditional_graph_numerical_guard_probe.jl` | `e5d3dd019e833e83d3c6707f9fbf7485` |
| Driver `scripts/checks/conditional_graph_numerical_guard_check.jl` | `cd51c41fa77d99155bf9b80ba3799448` |
| Test file `test/ext_distributed/nonfinite_loss_distributedtest.jl` | `b8d861fb0967d525eab6e0a7973b1bea` |
| Flux `src/train.jl` | `42b85187852f455bfab1dcd8113d3a26` (unchanged) |
| Flux `src/distributed/public_api.jl` | `e4ed1f8d4e2aed25d99654ad05d54919` (unchanged) |

Deathstar was left clean after the campaign (no `julia`/`mpiexec`, GPUs idle).

## v3 review-response round 2 (2026-10-14)

The v2 campaign-token cleanup still interpolated the token into an unbounded
regular expression without an argument boundary (`campaign_pattern`), so
cleanup for `campA` also selected `campA-rerun`, and `camp.1` also selected
`campX1`; the reviewer reproduced the cross-match with the real
`campaign_pids()`
([reproduction](../../../../wiki/devlog/02-conditional-graph/2026-10-14-db1-db4-v2-review.md)).
The v1/v2 sections above remain the historical record; this section supersedes
their evidence.

Fix (driver only; no Flux source, probe, or test-file change):

- Regex/pgrep/pkill matching is removed entirely. `campaign_pids(token)` scans
  `/proc/<pid>/cmdline`, requires the probe filename as a separate argv
  element, and requires the exact adjacent pair `("--campaign", token)`;
  `reap_campaign` TERMs then KILLs only those pids.
- The driver header documents that operator-supplied
  `NUMERICAL_GUARD_CAMPAIGN` values must be unique across concurrent runs; the
  default `<pid>-<time_ns>` is unique.
- The selftest was extended to four real idle Julia children:
  `t1=selftestA-<pid>`, `t2=t1*"-rerun"` (prefix), `t3=selftest.<pid>` (regex
  metacharacter), and `t4=selftestX<pid>` (would collide with a `selftest.`
  wildcard); the ordered reaps prove `prefix_ok` and `metachar_ok`.

The re-gates below ran on deathstar with the final driver on 2026-10-14.

Source identity (v3):

| Item | md5 |
|---|---|
| Driver `scripts/checks/conditional_graph_numerical_guard_check.jl` | `49c7e0aacdb48661f044231b1fc6c5f1` |
| Probe `scripts/checks/conditional_graph_numerical_guard_probe.jl` | `e5d3dd019e833e83d3c6707f9fbf7485` (unchanged) |
| Test file `test/ext_distributed/nonfinite_loss_distributedtest.jl` | `b8d861fb0967d525eab6e0a7973b1bea` (unchanged) |
| Flux `src/train.jl` | `42b85187852f455bfab1dcd8113d3a26` (unchanged) |
| Flux `src/distributed/public_api.jl` | `e4ed1f8d4e2aed25d99654ad05d54919` (unchanged) |

### Cleanup selftest v2 (deathstar)

```
scripts/remote/run.sh deathstar "julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --selftest-cleanup"
```

Output (`selftest-cleanup-deathstar-v2.log`):

```
SELFTEST_CLEANUP PASS prefix_ok=true metachar_ok=true final_empty=true t1=selftestA-585549 pids1=585589 killed1=585589 t2=selftestA-585549-rerun pids2=585593 killed2=585593 t3=selftest.585549 pids3=585594 killed3=585594 t4=selftestX585549 pids4=585595 killed4=585595
```

### Harness gates (green3, deathstar)

```
scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group loss --ranks 2 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green3-deathstar-loss-mpi2"

scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group loss --ranks 4 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green3-deathstar-loss-mpi4"

scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group loss --ranks 2 --backend nccl \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green3-deathstar-loss-nccl2"
```

Consoles: `console-v2/mpi2.log`, `console-v2/mpi4.log`, `console-v2/nccl2.log`
(the v3 re-gates overwrote the v2 harness consoles).

| Gate | Result |
|---|---|
| Harness green3 loss mpi2 | 13/13 PASS (healthy 36.0 s) |
| Harness green3 loss mpi4 | 8/8 PASS (healthy 39.9 s) |
| Harness green3 loss nccl2 | 12 PASS + 1 UNTESTED (`dup-train-one`; healthy 60.5 s) |

Healthy markers (all `ok=true`): mpi2 8 `STEP_STATE_OK` + 2 `ABSENCE_FROZEN` +
2 `REACTIVATED`; mpi4 16/4/4; nccl2 8/2/2.

One transient infra note: the first mpi4 attempt failed with `Connection to
10.1.60.14 port 22 timed out` before anything started; an identical rerun
passed 8/8. No artifact was produced by the failed attempt.

### Cleanup concurrency check (deathstar, v3)

The v2 command pattern above was rerun unchanged with the final driver (same
tokens, same outdirs): campA (`--case healthy`, token `campA`, 1 s execution
watchdog) and campB (full loss mpi2, token `campB`, default watchdog) launched
at the same time.

Evidence (`cleanup-concurrency-deathstar/`): campB `summary: PASS=13 FAIL=0
UNTESTED=0`; campA verdict FAIL with reason `watchdog timeout (execution) after
37.7s`, and both campA ranks reached `GUARD_PROBE_READY` before the kill; no
probe process remained afterwards and the GPUs were idle. The no-`reaped
orphan pids`-line mechanism is unchanged: MPICH Hydra terminates its own ranks
once `stop_probe` kills the launcher, so `reap_campaign(campA)` found zero
survivors; the exact-argv kill itself is proven by the extended selftest.

Rename mapping: the previous successful regex-cleanup concurrency evidence is
now `cleanup-concurrency-deathstar-attempt2/` (v2 run; campB 13/13, campA
timeout 40.3 s); `cleanup-concurrency-deathstar/` holds the round-2 exact-argv
run. `cleanup-concurrency-deathstar-attempt1/` remains the pre-fix FieldError
attempt.

Suite v2 results remain authoritative (the dedicated test file is unchanged):
`suite-deathstar-v2/` mpi2 8/8, mpi4 8/8, nccl2 16/16 was not rerun in round 2.

## Artifact map

v1:

- `red-local-loss-mpi2/`: local RED driver artifacts (summary, metadata, raw logs).
- `green-deathstar-loss-mpi2/`, `green-deathstar-loss-mpi4/`,
  `green-deathstar-loss-nccl2-r2/`: green driver artifacts fetched from deathstar.
- `green-deathstar-loss-nccl2/`: retained first NCCL attempt (dup route timeout).
- `console/`: driver console logs.
- `suite-deathstar/`: suite, train, and ambiguity console logs, including
  `suite-mpi2-attempt1-dup-warning-failure.log` (pre-fix test defect).
- `baseline-metadata.md`: pre-edit source identity.

v2 review response:

- `green2-deathstar-loss-mpi2/`, `green2-deathstar-loss-mpi4/`,
  `green2-deathstar-loss-nccl2/`: final review-response driver artifacts.
- `green2-attempt1-healthy-bug-deathstar-loss-mpi2|mpi4|nccl2/`: retained
  attempt-1 `healthy` failures (probe `FieldError`).
- `suite-deathstar-v2/`: v2 suite console logs.
- `selftest-cleanup-deathstar.log`: v2 campaign-scoped cleanup selftest output.
- `cleanup-concurrency-deathstar-attempt2/`: v2 concurrent campA/campB check
  with the regex matcher (renamed from `cleanup-concurrency-deathstar/` in
  round 2).
- `cleanup-concurrency-deathstar-attempt1/`: retained concurrent check before
  the probe fix (campA FieldError, campB `healthy` FAIL).

v3 review response round 2:

- `green3-deathstar-loss-mpi2/`, `green3-deathstar-loss-mpi4/`,
  `green3-deathstar-loss-nccl2/`: round-2 driver artifacts (final driver).
- `selftest-cleanup-deathstar-v2.log`: extended cleanup selftest output
  (prefix, regex metacharacter, and wildcard-collision tokens).
- `cleanup-concurrency-deathstar/`: round-2 exact-argv concurrent campA/campB
  check.
- `console-v2/`: round-2 green3 harness driver console logs (overwrote the v2
  harness consoles).

## Known limitations recorded with this campaign

- `dup-train-one` under CUDA/NCCL is UNTESTED: the `Enzyme.Duplicated` route
  availability check did not complete within the 420 s startup watchdog on the
  first attempt. The route is exercised and passes under MPI (2 ranks); the v2
  and v3 re-runs keep the explicit UNTESTED verdict.
- DB-3 (synchronized-gradient guard) is not implemented; `--group gradient|all`
  prints `GROUP_UNAVAILABLE` and exits 2. Phase D of both plans is pending after
  v2/v3.
- Forward-pass side effects are not rolled back; only parameters and optimizer
  state are preserved for a rejected step (ADR-0008).
- The campA console of the concurrent cleanup check has no `reaped orphan pids`
  line: MPICH Hydra tears down its ranks when the driver kills the launcher, so
  the final reap found no survivors. The token-scoped kill is proven by
  `--selftest-cleanup`; see the failure note.

## DB-3 campaign (2026-10-14, deathstar)

DB-3 post-reduction non-finite-gradient rejection for the conditional wrapper.
Plan: `reports/conditional-graph-db3-fix-plan.md`. Decision:
[ADR-0009](../../../../wiki/decisions/ADR-0009-post-reduction-nonfinite-gradient-rejection.md).
The v1-v3 loss sections above remain the historical record for DB-1/DB-4; this
section records the DB-3 source change and gates. Every failing-then-fixed run
is retained on disk; nothing was deleted or rewritten.

Collective-count rule encoded by the updated test files and helpers: every
nonempty update (eager or explicit) with `np > 0` gains exactly one trailing
host `Float32` length-1 `avg` collective (the gradient vote); `np == 0` stays
zero; a loss-rejected step keeps its single loss vote; gradient-rejected steps
count as full steps.

### Source identity (DB-3)

| Item | Baseline | Final |
|---|---|---|
| Flux worktree HEAD | file baselines below (pre-DB-3) | `314aef2022333fa8f37035ed10cdc8e2360fa475` (dirty; DB-3 changes uncommitted) |
| `src/distributed/public_api.jl` md5 | `e4ed1f8d4e2aed25d99654ad05d54919` | `d21c73f8885b35ae8558a0f0d4e43196` |
| `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl` md5 | `330999cee29c78f7ecebe6e7e2610c13` | `841c662047cbe86cd082ceeb226b6231` |
| `NEWS.md` md5 | - | `83e69bb91cc5edd110c7de9af2b0b87a` |
| `test/ext_distributed/conditional_distributedtest.jl` md5 | `c26ef49b070bf2c446b827c64cc313c0` | `54a3f8da8d677c35e5de3ed34130e34a` |
| `test/ext_distributed/cuda_conditional_distributedtest.jl` md5 | `dabdaddb77301d93f83707c1c930bc6f` | `c470cc3f3e4370b4d8b6404077684987` |
| `test/ext_distributed/nonfinite_loss_distributedtest.jl` md5 | `b8d861fb0967d525eab6e0a7973b1bea` | `0e2680635752bc4f9ed168376458e470` |
| `test/ext_distributed/nonfinite_gradient_distributedtest.jl` md5 | n/a | `06eef21d92c78ca85f718b325ddbe8c6` (new, 848 lines, 15 testsets) |
| Driver `scripts/checks/conditional_graph_numerical_guard_check.jl` md5 | `49c7e0aacdb48661f044231b1fc6c5f1` | `9dba4ae76367e58f12467f180a42381f` |
| Probe `scripts/checks/conditional_graph_numerical_guard_probe.jl` md5 | `e5d3dd019e833e83d3c6707f9fbf7485` | `ce16af4ab7932c5eb746f7b141e37f1c` |
| `scripts/examples/conditional_graph/common.jl` md5 | - | `2ec29fb283cf45e3e33c684e454e2422` |
| `scripts/checks/conditional_deadlock_check.jl` md5 | - | `467498a568ead33d0915d1b66a00d43f` |
| `scripts/checks/conditional_deadlock_probe.jl` md5 | - | `86945681c0f8c57ec6da8d79c1fdf11c` |
| `notebook/phase_k_workflow_helpers.jl` md5 | - | `cbca216033752b9fa2abf827257d5183` |

Thesis HEAD for the final harness: `0ea21f79b20092ead8ca07f33f34f372a4ee3e70`.
Local toolchain: Julia 1.12.6, MPICH Hydra 5.0.1, Flux 0.16.11, Optimisers
0.4.9, MPI 0.20.27. Remote (deathstar): Julia 1.12.6; CUDA 6.2.2, NCCL 0.1.3,
cuDNN entry 6.2.2, Flux 0.16.11, Optimisers 0.4.9, MPI 0.20.27.

### Sync and precompile (deathstar)

```
rsync -az --delete --exclude-from='.rsyncignore' --exclude='artifacts/' \
  . deathstar:~/projects/ddp-flux-thesis/
rsync -az --delete --exclude='.git' --exclude='Manifest.toml' \
  --exclude='test/Manifest.toml' ../flux-integration deathstar:~/projects/
scripts/remote/run.sh deathstar "make precompile-all FLUX_REPO_PATH=../flux-integration"
```

### RED evidence (deathstar, pre-fix production)

Gradient harness mpi2:

```
scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=red julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group gradient --ranks 2 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/red-deathstar-gradient-mpi2-v2"
```

Focused test mpi2:

```
scripts/remote/run.sh deathstar "FLUX_TEST_DISTRIBUTED_BACKEND=mpi ~/.julia/bin/mpiexecjl -n 2 \
  julia --project=../flux-integration/test --startup-file=no \
  ../flux-integration/test/ext_distributed/nonfinite_gradient_distributedtest.jl"
```

Result: `red-deathstar-gradient-mpi2-v2/` `PASS=1 FAIL=18`; the single PASS is
`grad-no-trainable` (a no-regression case that passes RED by design). Focused
first testset `7 pass / 7 fail / 1 error` in
`red-deathstar-focused-mpi2/console.log`; the run aborts at the first failing
testset. The superseded v1 run `red-deathstar-gradient-mpi2/` is retained
(probe `MethodError`; see the [failure note](../../../../wiki/failures/2026-10-14-db3-green-attempt1-harness-fixes.md)).

### GREEN gates (deathstar)

Gradient harness:

```
scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group gradient --ranks 2 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-gradient-mpi2"

scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group gradient --ranks 4 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-gradient-mpi4"

scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group gradient --ranks 2 --backend nccl \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-gradient-nccl2"
```

Focused test:

```
scripts/remote/run.sh deathstar "FLUX_TEST_DISTRIBUTED_BACKEND=mpi ~/.julia/bin/mpiexecjl -n 2 \
  julia --project=../flux-integration/test --startup-file=no \
  ../flux-integration/test/ext_distributed/nonfinite_gradient_distributedtest.jl"

scripts/remote/run.sh deathstar "FLUX_TEST_DISTRIBUTED_BACKEND=mpi ~/.julia/bin/mpiexecjl -n 4 \
  julia --project=../flux-integration/test --startup-file=no \
  ../flux-integration/test/ext_distributed/nonfinite_gradient_distributedtest.jl"

scripts/remote/run.sh deathstar "FLUX_TEST_DISTRIBUTED_BACKEND=nccl ~/.julia/bin/mpiexecjl -n 2 \
  julia --project=../flux-integration/test --startup-file=no \
  ../flux-integration/test/ext_distributed/nonfinite_gradient_distributedtest.jl"
```

Loss-group regression:

```
scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group loss --ranks 2 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-loss-db3-mpi2"

scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group loss --ranks 4 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-loss-db3-mpi4"
```

Combined `--group all` (Phase D):

```
scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group all --ranks 2 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-all-mpi2"

scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group all --ranks 4 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-all-mpi4"

scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group all --ranks 2 --backend nccl \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-all-nccl2"
```

Dedicated suite:

```
scripts/remote/run.sh deathstar "JULIA_MPI_TEST_NPROCS=2 FLUX_TEST_DISTRIBUTED_NCCL=false \
  FLUX_TEST_DISTRIBUTED_TIMEOUT=900 julia --startup-file=no \
  --project=../flux-integration/test ../flux-integration/test/ext_distributed/runtests.jl"

scripts/remote/run.sh deathstar "JULIA_MPI_TEST_NPROCS=4 FLUX_TEST_DISTRIBUTED_NCCL=false \
  FLUX_TEST_DISTRIBUTED_TIMEOUT=900 julia --startup-file=no \
  --project=../flux-integration/test ../flux-integration/test/ext_distributed/runtests.jl"

scripts/remote/run.sh deathstar "JULIA_MPI_TEST_NPROCS=2 FLUX_TEST_DISTRIBUTED_NCCL=true \
  FLUX_TEST_DISTRIBUTED_TIMEOUT=900 julia --startup-file=no \
  --project=../flux-integration/test ../flux-integration/test/ext_distributed/runtests.jl"
```

Single-process `train` group:

```
scripts/remote/run.sh deathstar "FLUX_TEST_REACTANT=false julia --startup-file=no \
  --project=../flux-integration/test ../flux-integration/test/runtests.jl train"
```

Deadlock regression:

```
scripts/remote/run.sh deathstar "make verify-conditional-deadlock FLUX_REPO_PATH=../flux-integration"
```

Phase K toy demo (mpi 2 and 4):

```
scripts/remote/run.sh deathstar "julia --startup-file=no --project=../flux-integration/test \
  scripts/checks/conditional_graph_demo_check.jl \
  --root artifacts/logs/conditional-wrapper/demo/db3-toy-2026-10-14-mpi \
  --backend mpi --ranks 2,4"
```

Ambiguities:

```
scripts/remote/run.sh deathstar 'julia --startup-file=no --project=../flux-integration/test \
  -e "...detect_ambiguities(Flux.DistributedUtils; recursive=false)..."'
```

### Results

| Gate | Result |
|---|---|
| Gradient harness mpi2 | 19/19 PASS (`green-deathstar-gradient-mpi2/`) |
| Gradient harness mpi4 | 8/8 PASS (`green-deathstar-gradient-mpi4/`) |
| Gradient harness nccl2 | 17 PASS + 2 UNTESTED (`grad-overflow`, `grad-tied-overflow`; `green-deathstar-gradient-nccl2/`) |
| Focused test mpi2 / mpi4 / nccl2 | exit 0 (`green-deathstar-gradient-focused-{mpi2,mpi4,nccl2}/console.log`) |
| Loss-group regression mpi2 / mpi4 | 13/13 / 8/8 PASS (`green-deathstar-loss-db3-mpi2/`, `-mpi4/`) |
| Combined `--group all` mpi2 | 32/32 PASS (`green-deathstar-all-mpi2/`) |
| Combined `--group all` mpi4 | 16/16 PASS (`green-deathstar-all-mpi4/`) |
| Combined `--group all` nccl2 | 29 PASS + 3 UNTESTED (`dup-train-one`, `grad-overflow`, `grad-tied-overflow`; `green-deathstar-all-nccl2/`) |
| Dedicated suite mpi2 | 9/9 files PASS, 2m53.8s (`suite-deathstar-db3/mpi2.log`) |
| Dedicated suite mpi4 | 9/9 files PASS, 3m09.9s (`suite-deathstar-db3/mpi4.log`) |
| Dedicated suite nccl2 (MPI + NCCL passes) | 18/18 PASS, 7m37.9s (`suite-deathstar-db3/nccl2.log`) |
| Single-process `train` group | 137/137 PASS (`suite-deathstar-db3/train.log`) |
| `detect_ambiguities(Flux.DistributedUtils; recursive=false)` | 0 |
| Deadlock regression | 246/246 PASS (`deadlock-check-db3.log`) |
| Phase K toy demo mpi 2,4 | 13/13 PASS (`artifacts/logs/conditional-wrapper/demo/db3-toy-2026-10-14-mpi/`) |
| `Precompiling packages...` lines | 0 in every suite log |
| Full-suite mpi2 attempt 1 | FAIL in `conditional_distributedtest.jl` (mock `all` ambiguity); retained (`suite-deathstar-db3/mpi2-attempt1.log`) |

Deathstar was left clean after the campaign (no `julia`/`mpiexec`, GPUs idle).

### Artifact map (DB-3)

- `red-deathstar-gradient-mpi2-v2/`: final RED gradient harness artifacts.
- `red-deathstar-gradient-mpi2/`: superseded v1 RED run (probe `MethodError`).
- `red-deathstar-focused-mpi2/`: first RED focused-test console.
- `green-deathstar-gradient-mpi2/`, `green-deathstar-gradient-mpi4/`,
  `green-deathstar-gradient-nccl2/`: GREEN gradient harness artifacts.
- `green-deathstar-gradient-focused-mpi2/`, `-mpi4/`, `-nccl2/`: focused-test
  consoles.
- `green-deathstar-loss-db3-mpi2/`, `green-deathstar-loss-db3-mpi4/`:
  loss-group regression under the DB-3 source.
- `green-deathstar-all-mpi2/`, `green-deathstar-all-mpi4/`,
  `green-deathstar-all-nccl2/`: combined Phase D acceptance (loss + gradient
  case tables). The nccl2 run continued through an SSH disconnect; the result
  was verified from its completed `summary.tsv`.
- `suite-deathstar-db3/`: suite (`mpi2.log`, `mpi4.log`, `nccl2.log`), train
  regressions (`train.log`), and the failed first attempt
  (`mpi2-attempt1.log`).
- `deadlock-check-db3.log`: deadlock regression.
- `artifacts/logs/conditional-wrapper/demo/db3-toy-2026-10-14-mpi/`: Phase K
  toy demo after the `common.jl` trace-check update.

### Harness/test defects fixed during this campaign

Six harness or test defects were found and fixed; none is a production defect.
Full record:
[2026-10-14-db3-green-attempt1-harness-fixes.md](../../../../wiki/failures/2026-10-14-db3-green-attempt1-harness-fixes.md).
Summary: probe `run_grad_no_trainable` arity (`data_range(1:3)`),
`CorruptBackend` arming (`arm::Ref{Bool}`), `grad-audit-zero` expected schedule,
`grad-update-both`/`grad-caller` fault rank (-1 -> 0), focused-test healthy
reference `(a = ga, c = nothing)`, and the `DeviceLikeArray`
`Base.all(f::Function, x::DeviceLikeArray) = all(f, x.data)` override.

## DB-3 review response campaign (2026-10-14, deathstar)

Response to the [DB-3 review](../../../../reports/conditional-graph-db3-review.md):
R1 thesis protocol pin, R2 focused healthy reference, R3 contract docs. The old
hashes in the DB-3 campaign section above remain the DB-3 campaign record; the
new hashes below describe the response state. Production guard code did not
change; no commit was made.

### Source identity (DB-3 review response)

| Item | Before response | After response |
|---|---|---|
| `notebook/phase_k_workflow_helpers.jl` md5 | `cbca216033752b9fa2abf827257d5183` | `b570d0777109a41eee5bb1be3b4d53d5` |
| `scripts/checks/phase_k_collective_protocol_check.jl` md5 | n/a | `923b7f49c20fa763a0e4727d27191331` (new) |
| `test/ext_distributed/nonfinite_gradient_distributedtest.jl` md5 | `06eef21d92c78ca85f718b325ddbe8c6` | `b0566261df12deb00cd69fc6d99220e9` (862 lines) |
| `src/distributed/public_api.jl` md5 | `d21c73f8885b35ae8558a0f0d4e43196` | `48ea1fcd3c0629d1773b91ae9d6f1677` |
| `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl` md5 | `841c662047cbe86cd082ceeb226b6231` | `3bca62aa63f2b1d18e2f7f4eb630d221` |
| `NEWS.md` md5 | `83e69bb91cc5edd110c7de9af2b0b87a` | `f9cd8bf52caf18a99d0b2b76cb13ee2b` |

The `src/distributed/public_api.jl`, `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl`,
and `NEWS.md` changes are docstrings/comments/prose only; no executable line
changed (verified by reverting the prose edits and reproducing the DB-3 campaign
md5s). The corrected content: the all-rank gradient-validity vote, coordinated
`DomainError` before any optimizer update, preservation guarantee, nonempty-plan
rule, and scalar CUDA device-to-host validity transfer.

### Sync and precompile (deathstar)

```
rsync -az --delete --exclude-from='.rsyncignore' --exclude='artifacts/' . deathstar:~/projects/ddp-flux-thesis/
rsync -az --delete --exclude='.git' --exclude='Manifest.toml' --exclude='test/Manifest.toml' ../flux-integration deathstar:~/projects/
scripts/remote/run.sh deathstar "make precompile-all FLUX_REPO_PATH=../flux-integration"
```

The thesis sync excludes `artifacts/`, so remote-only artifacts survive. The
first R1 run failed on an incomplete remote 4-rank tutorial bundle; the two
missing files (local sizes `trajectory.jls` 148521848 bytes,
`validation-train.tsv` 108911 bytes) were uploaded without `--delete`.

### R1 protocol acceptance (deathstar)

```
scripts/remote/run.sh deathstar "julia --startup-file=no scripts/checks/phase_k_collective_protocol_check.jl"
```

Exit 0:

```
PHASE_K_PROTOCOL tutorial-2rank PASS
PHASE_K_PROTOCOL tutorial-4rank PASS
PHASE_K_PROTOCOL db3-toy-mpi2 PASS
PHASE_K_PROTOCOL guarded-pinned-legacy-5/7 PASS
PHASE_K_PROTOCOL guarded-without-protocol PASS
PHASE_K_PROTOCOL tutorial-2rank-pinned-guarded-6/8 PASS
PHASE_K_PROTOCOL_SUMMARY PASS
```

The guarded checks ran against the real DB-3 guarded artifacts
`artifacts/logs/conditional-wrapper/demo/db3-toy-2026-10-14-mpi` (mpi2, 16 steps,
6/8 traces with the trailing vote); the negatives prove a guarded trace pinned
legacy, a missing pin, and a legacy trace pinned guarded all throw. The first
run also failed on the one-dimensional `(1,)` shape capture; both first-run
failures and their fixes are in the
[failure note](../../../../wiki/failures/2026-10-14-db3-review-response-r1-shape-and-artifacts.md)
(the failed console is transcribed there because the local log was overwritten
by the successful rerun).

### R2 focused test (deathstar)

```
scripts/remote/run.sh deathstar "FLUX_TEST_DISTRIBUTED_BACKEND=mpi ~/.julia/bin/mpiexecjl -n 2 julia --project=../flux-integration/test --startup-file=no ../flux-integration/test/ext_distributed/nonfinite_gradient_distributedtest.jl"

scripts/remote/run.sh deathstar "FLUX_TEST_DISTRIBUTED_BACKEND=mpi ~/.julia/bin/mpiexecjl -n 4 julia --project=../flux-integration/test --startup-file=no ../flux-integration/test/ext_distributed/nonfinite_gradient_distributedtest.jl"

scripts/remote/run.sh deathstar "FLUX_TEST_DISTRIBUTED_BACKEND=nccl ~/.julia/bin/mpiexecjl -n 2 julia --project=../flux-integration/test --startup-file=no ../flux-integration/test/ext_distributed/nonfinite_gradient_distributedtest.jl"
```

| Run | Result |
|---|---|
| mpi2 | exit 0; healthy testset 59/59 per rank |
| mpi4 (4 ranks) | exit 0; healthy testset 59/59 per rank |
| nccl2 (`allowscalar(false)`) | exit 0; healthy testset 59/59 per rank |

RED control: a temporary copy `red_weighting.jl` with the old
`ga = local_ga ./ tworkers` failed the healthy testset: 53 pass / 6 fail / 59
per rank, exit 1, at the model-`a` comparison and both `a`-moment comparisons.
The temp copy was deleted; the real file md5 is
`b0566261df12deb00cd69fc6d99220e9` locally and remotely.

### Results

| Gate | Result |
|---|---|
| R1 protocol check | exit 0; `PHASE_K_PROTOCOL_SUMMARY PASS` |
| R2 focused mpi2 / mpi4 / nccl2 | exit 0; healthy 59/59 per rank on each |
| R2 RED control mpi2 | 53 pass / 6 fail / 59 per rank, exit 1 |
| `Precompiling packages...` lines | 0 in every gate log |

Deathstar was left clean (no `julia`/`mpiexec`, GPUs idle 6/6/16 MiB); the RED
temp file was removed.

### Artifact map (DB-3 review response)

All under
`artifacts/logs/conditional-wrapper/numerical-guards/db3-review-response/`:

- `r1-protocol-check.log`: final R1 acceptance run (the local log was
  overwritten by the successful rerun).
- `focused-mpi2.log`, `focused-mpi4.log`, `focused-nccl2.log`: R2 GREEN runs.
- `red-mpi2-old-weighting.log`: R2 RED control.

Full response record:
[devlog](../../../../wiki/devlog/02-conditional-graph/2026-10-14-db3-review-response.md).

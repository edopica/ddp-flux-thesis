# DB-1/DB-4 coordinated non-finite-loss fix

Date: 2026-10-13

Implemented and gate-accepted the coordinated non-finite-loss rejection from
[ADR-0008](../../decisions/ADR-0008-coordinated-nonfinite-loss-rejection.md)
and the [DB-1/DB-4 fix plan](../../../reports/conditional-graph-db1-db4-fix-plan.md).
Every loss-path gate is GREEN; DB-3 remains a separate plan and no commit was
made. Flux HEAD stays `55171e659b2f425da980cbc83177bf5c38f0a7a1`, further dirty
in the `flux-integration` worktree.

## Implementation

- `src/train.jl`: private hook `Flux.Train._loss_update_allowed(opt_state,
  scalar_loss)`, default `isfinite(scalar_loss)`, so ordinary optimizer states
  keep the pre-existing local skip.
- `_eager_step!` calls the hook unconditionally after `withgradient` and before
  `_update!`:
  `_loss_update_allowed(opt_state, _loss_scalar(v)) && _update!(...)`.
- `src/distributed/public_api.jl`: `Train._loss_update_allowed` specialization
  for `DistributedOptimizerState` via `_any_rank_invalid`: one host `Float32`
  0/1 flag reduced with `DistributedUtils.allreduce!` and
  `DistributedUtils.avg`, rejection at `> 0` (never `> 0.5`).
- On rejection every rank throws `DomainError` before the presence exchange,
  the gradient reductions, and the inner optimizer update. The message states
  that at least one rank reported a non-finite scalar loss and that no
  optimizer update occurred.
- `test/ext_distributed/nonfinite_loss_distributedtest.jl` (new, included by
  `runtests.jl`): device-generic, 8 testsets covering asymmetric NaN through
  `train!` (the harness adds Inf and -Inf), all-rank NaN against a native Adam
  shadow, manual `trainstep!` rejection, `trainstep_withgradient!`, non-finite
  auxiliary outputs in both directions, a model without trainable parameters,
  and a known-good step after a rejection. Each testset ends with a broadcast
  that confirms every rank observed the same collective count.
- `test/train.jl` (+14 lines): single-process compatibility for the unchanged
  local skip and the returned gradient in `trainstep_withgradient!`.
- `NEWS.md`: distributed eager APIs now raise collectively on a non-finite
  scalar loss; one extra host-metadata collective per step.

Source identity (GREEN column of
[COMMANDS.md](../../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md)):
`src/train.jl` md5 `42b85187852f455bfab1dcd8113d3a26`,
`src/distributed/public_api.jl` md5 `e4ed1f8d4e2aed25d99654ad05d54919`,
new test file md5 `5a0a41eaec5922b0fb944c645f49c4c8`.

## RED (local, unmodified source)

The 2-rank MPI harness run of the previous session
([devlog](2026-10-13-numerical-guard-red-harness.md)) is the RED baseline:
13/13 cases FAIL for the expected reasons (9 asymmetric timeouts with an
unmatched `COLLECTIVE_BEGIN`, `no-trainable` wrong participation, and missing
VOTE stages in `train-all`/`healthy`/`aux-nonfinite-ok`), no startup failure,
no orphans. Artifacts:
`artifacts/logs/conditional-wrapper/numerical-guards/red-local-loss-mpi2/`.

This is the only local test run for this fix. All subsequent gates ran on
`deathstar` per the user's instruction; the exact commands are in COMMANDS.md.

## GREEN (deathstar)

| Gate | Result |
|---|---|
| Harness loss mpi2 | 13/13 PASS |
| Harness loss mpi4 (one invalid non-root rank of four included) | 8/8 PASS |
| Harness loss nccl2 | 12 PASS + 1 UNTESTED (`dup-train-one`) |
| Dedicated suite mpi2 | 8/8 files PASS |
| Dedicated suite mpi4 | 8/8 files PASS |
| Dedicated suite nccl2 (MPI + NCCL passes) | 16/16 PASS |
| Single-process `train` regressions | 137/137 PASS |
| `detect_ambiguities(Flux.DistributedUtils)` | 0 |
| `Precompiling packages...` lines | 0 in every console log |

Case-level summaries: `green-deathstar-loss-mpi2/summary.tsv`,
`green-deathstar-loss-mpi4/summary.tsv`,
`green-deathstar-loss-nccl2-r2/summary.tsv`. Suite and unit logs:
`suite-deathstar/suite-mpi2-r2.log`, `suite-mpi4.log`, `suite-nccl2.log`,
`train-regressions.log`, `ambiguities.log`.

## Failures encountered and fixed

1. First dedicated-suite mpi2 run: 7/8 files passed;
   `nonfinite_loss_distributedtest.jl` failed 1 of 10 assertions at line 303
   because `Optimisers.trainable(::NoParamsModel) = ()` returned a legacy Tuple,
   which emitted an extra deprecation warning and broke the strict
   `@test_logs`. Fixed to `NamedTuple()` and `match_mode = :any`; re-run 8/8.
   Logs: `suite-mpi2-attempt1-dup-warning-failure.log`, `suite-mpi2-r2.log`.
2. First NCCL harness attempt: `dup-train-one` hit the 420 s startup watchdog
   (12 PASS + 1 FAIL). The `Enzyme.Duplicated` availability check does not
   complete under CUDA/NCCL; the probe now reports the route UNTESTED by
   policy, and it remains exercised under MPI. The same pass fixed a latent
   probe bug that used `BACKEND.comm` (an NCCL communicator under NCCL) instead
   of the MPI `COMM` in its own agreement vote. Retained:
   `green-deathstar-loss-nccl2/`; re-run: `green-deathstar-loss-nccl2-r2/`
   (12 PASS + 1 UNTESTED). Full record:
   [failure note](../../failures/2026-10-13-db1-db4-nccl-dup-and-testfile.md).

## Review response (second pass)

The external review ([report](../../../reports/conditional-graph-db1-db4-review.md))
found two P2 harness issues and one minor coverage gap. All three are fixed and
re-gated with no Flux source change; the v1 results above remain valid for the
cases that ran. Exact commands, identities, and counts:
[COMMANDS.md](../../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#v2-review-response-campaign-2026-10-13).

1. **R1 — campaign-scoped cleanup.** The driver's filename-wide
   `reap_orphans()` is replaced by `reap_campaign(token)`. Every probe child
   carries `--campaign TOKEN` (default `<pid>-<time_ns>`, override
   `NUMERICAL_GUARD_CAMPAIGN`), and cleanup matches only processes whose
   command line contains the probe filename and the token. New
   `--selftest-cleanup` mode proves the isolation without MPI:
   `SELFTEST_CLEANUP PASS pids_a=563275 pids_b=563276 killed_a=563275 ...`
   ([log](../../../artifacts/logs/conditional-wrapper/numerical-guards/selftest-cleanup-deathstar.log)).
   Under concurrency, campA (`healthy`, 1 s execution watchdog, token `campA`)
   timed out without affecting campB (full loss mpi2, token `campB`): campB is
   `PASS=13 FAIL=0 UNTESTED=0`, campA's ranks reached `GUARD_PROBE_READY`
   before the kill, and no probe process remained afterwards. The campA console
   has no `reaped orphan pids` line because MPICH Hydra tears down its ranks
   once the launcher is killed, so the token-scoped kill is proven by the
   selftest; recorded in the
   [failure note](../../failures/2026-10-13-db1-db4-healthy-probe-fielderror.md).
2. **R2 — conditional healthy gate.** `healthy` now runs a four-step
   rank-distinct conditional eager `train!` trajectory on a new `CondModel`
   (`w` len 2 always on; `u` len 3 branch U; `v` len 4 branch V): all branches;
   even-U/odd-V; U only with V globally absent after nonzero Adam history; all
   branches with V reactivated. Each step compares parameters and the complete
   Adam leaf state (moment 1, moment 2, counter) against a native host
   reference updated with the global averaged gradient and prints
   `STEP_STATE_OK`; step 3 asserts `ABSENCE_FROZEN` and step 4 `REACTIVATED`,
   required on every rank. A full step schedules `vote, presence, grad_w(2),
   grad_u(3), grad_v(4)`; the locally absent branch is zero-filled, and only
   the globally absent step drops `grad_v`.
3. **NamedTuple loss coverage.** `nonfinite_loss_distributedtest.jl` gained a
   testset (now 9 testsets): a finite outer NamedTuple round-trips and a
   poisoned `(loss = NaN32, acc = 1.0)` rejects on every rank with only the
   vote.

The first attempt failed `healthy` in all three gates because `cond_local_loss`
passed `m.u`/`m.v` to `ul`/`vl` (which already take the model),
`FieldError: type Array has no field 'u'` (`CuArray` under NCCL), crashing in
warmup before `GUARD_PROBE_READY`. Fixed to `ul(m, x)`/`vl(m, x)`; the failed
attempt is retained as `green2-attempt1-healthy-bug-*` and
`cleanup-concurrency-deathstar-attempt1`. Full record:
[failure note](../../failures/2026-10-13-db1-db4-healthy-probe-fielderror.md).

Final v2 results (all on deathstar):

| Gate | Result |
|---|---|
| Harness green2 loss mpi2 | 13/13 PASS (healthy 37.3 s; 8 `STEP_STATE_OK`, 2 `ABSENCE_FROZEN`, 2 `REACTIVATED`) |
| Harness green2 loss mpi4 | 8/8 PASS (healthy 38.1 s; 16/4/4 markers) |
| Harness green2 loss nccl2 | 12 PASS + 1 UNTESTED (`dup-train-one`; healthy 60.5 s; 8/2/2 markers) |
| Suite v2 mpi2 | 8/8 files PASS (2m28.2s) |
| Suite v2 mpi4 | 8/8 files PASS (2m42.8s) |
| Suite v2 nccl2 (MPI + NCCL passes) | 16/16 PASS (6m12.7s) |
| `Precompiling packages...` lines | 0 in every v2 log |

Flux source is unchanged (`src/train.jl`
`42b85187852f455bfab1dcd8113d3a26`, `src/distributed/public_api.jl`
`e4ed1f8d4e2aed25d99654ad05d54919`); the v2 harness hashes are probe
`e5d3dd019e833e83d3c6707f9fbf7485`, driver
`cd51c41fa77d99155bf9b80ba3799448`, and test file
`b8d861fb0967d525eab6e0a7973b1bea`. Deathstar was left clean (no
`julia`/`mpiexec`, GPUs idle).

## Review response (R1 round 2)

The 2026-10-14 follow-up review kept R1 open: the v2 cleanup interpolated the
campaign token into an unbounded regular expression without an argument
boundary, so cleanup for `campA` also selected `campA-rerun`, and `camp.1` also
selected `campX1`. The reviewer reproduced the cross-match with the real
`campaign_pids()`
([devlog](2026-10-14-db1-db4-v2-review.md)). Round 2 fixes the driver only;
no Flux source, probe, or test file changed.

- **Exact-argv fix.** Regex/pgrep/pkill matching is removed.
  `campaign_pids(token)` scans `/proc/<pid>/cmdline`, requires the probe
  filename as a separate argv element, and requires the exact adjacent pair
  `("--campaign", token)`; `reap_campaign` TERMs then KILLs only those pids.
  The driver header documents that operator-supplied `NUMERICAL_GUARD_CAMPAIGN`
  values must be unique across concurrent runs (the default `<pid>-<time_ns>`
  is unique).
- **Extended selftest.** Four real idle Julia children:
  `t1=selftestA-<pid>`, `t2=t1*"-rerun"` (prefix), `t3=selftest.<pid>` (regex
  metacharacter), `t4=selftestX<pid>` (would collide with a `selftest.`
  wildcard); the ordered reaps prove `prefix_ok=true` and `metachar_ok=true`
  ([log](../../../artifacts/logs/conditional-wrapper/numerical-guards/selftest-cleanup-deathstar-v2.log)).
- **Re-gates (deathstar, final driver).** `green3-deathstar-loss-mpi2` 13/13
  PASS; `-mpi4` 8/8 PASS; `-nccl2` 12 PASS + 1 UNTESTED (`dup-train-one`).
  Healthy markers all true: mpi2 8 `STEP_STATE_OK` + 2 `ABSENCE_FROZEN` + 2
  `REACTIVATED`; mpi4 16/4/4; nccl2 8/2/2. Consoles: `console-v2/*`.
- **Concurrency (fresh `cleanup-concurrency-deathstar/`).** campB (full loss
  mpi2, token `campB`) 13/13 PASS while campA (`healthy`, 1 s execution
  watchdog, token `campA`) timed out after 37.7 s with both ranks at
  `GUARD_PROBE_READY`; no leftover probe processes, GPUs idle. The
  no-`reaped orphan pids`-line mechanism (MPICH Hydra terminates ranks with
  the launcher) is unchanged.
- Suite v2 (mpi2 8/8, mpi4 8/8, nccl2 16/16) remains authoritative: the
  dedicated test file is unchanged.

Driver md5 `49c7e0aacdb48661f044231b1fc6c5f1`; probe
`e5d3dd019e833e83d3c6707f9fbf7485` and test file
`b8d861fb0967d525eab6e0a7973b1bea` unchanged; Flux source unchanged
(`src/train.jl` `42b85187852f455bfab1dcd8113d3a26`,
`src/distributed/public_api.jl` `e4ed1f8d4e2aed25d99654ad05d54919`). Exact
commands, identity table, rename mapping, and the transient SSH note:
[COMMANDS.md v3](../../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#v3-review-response-round-2-2026-10-13).

## Limitations

- Eager paths only (`train!`, `trainstep!`, `trainstep_withgradient!`); the
  Reactant compiled path is excluded (ADR-0008).
- A rejected step preserves parameters and optimizer state but does not roll
  back forward-pass side effects (BatchNorm statistics, RNG, user mutations).
- DB-3 (non-finite gradients with a finite loss) is not implemented:
  `--group gradient|all` prints `GROUP_UNAVAILABLE` and exits 2, so the
  combined Phase D acceptance is pending. Plan:
  [conditional-graph-db3-fix-plan.md](../../../reports/conditional-graph-db3-fix-plan.md).
- The `Duplicated` eager route under CUDA/NCCL is untested; it passes under
  MPI at 2 ranks.

## Artifacts

All under `artifacts/logs/conditional-wrapper/numerical-guards/`:

- v1: `red-local-loss-mpi2/`, `green-deathstar-loss-mpi2/`,
  `green-deathstar-loss-mpi4/`, `green-deathstar-loss-nccl2/` (retained first
  attempt), `green-deathstar-loss-nccl2-r2/`, `suite-deathstar/`, `console/`,
  and `baseline-metadata.md`.
- v2: `green2-deathstar-loss-mpi2/`, `green2-deathstar-loss-mpi4/`,
  `green2-deathstar-loss-nccl2/`, `green2-attempt1-healthy-bug-deathstar-loss-*`,
  `suite-deathstar-v2/`, `selftest-cleanup-deathstar.log`,
  `cleanup-concurrency-deathstar-attempt2/` (renamed in round 2), and
  `cleanup-concurrency-deathstar-attempt1/`.
- v3: `green3-deathstar-loss-mpi2/`, `green3-deathstar-loss-mpi4/`,
  `green3-deathstar-loss-nccl2/`, `console-v2/` (the v2 harness consoles were
  overwritten by these), `selftest-cleanup-deathstar-v2.log`, and
  `cleanup-concurrency-deathstar/`.

Exact commands, versions, hashes, results, and limitations:
[COMMANDS.md](../../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md).

Audit 002 records DB-1/DB-4 as resolved, now with the v2 review-response
evidence: [Resolution (2026-10-13)](../../audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md#resolution-2026-10-13).

## Exact next action

Decide whether to implement the DB-3 gradient guard (separate plan) before the
combined `--group all` Phase D acceptance.

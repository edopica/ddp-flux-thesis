# 2026-10-13 — DB-1/DB-4 v2 attempt 1: healthy-probe `FieldError` and cleanup reap-line deviation

The first review-response (v2) campaign failed the new `healthy` case in all
three gates. The failure is a probe-harness defect, not a product defect. The
fix was applied and the campaign re-ran GREEN. This note also records one
expected deviation in the concurrent-cleanup evidence.

Recorded per the repository rule to log failed commands.

## 1. Healthy-probe `FieldError` (all three gates)

Command (mpi2 shown; mpi4 and nccl2 change only `--ranks` and `--backend`; the
pre-fix outputs were retained under the `green2-attempt1-healthy-bug-*` names):

```
scripts/remote/run.sh deathstar "NUMERICAL_GUARD_PHASE=green julia --startup-file=no \
  --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl \
  --group loss --ranks 2 --backend mpi \
  --outdir artifacts/logs/conditional-wrapper/numerical-guards/green2-deathstar-loss-mpi2"
```

Result: `healthy` FAIL in all three gates; no other case failed (the NCCL
`dup-train-one` remains UNTESTED). The probe crashed in warmup
before any rank printed `GUARD_PROBE_READY`, so the driver reported
`expected 19 collectives, got 0` and missing `STEP_STATE_OK` (steps 1-4),
`ABSENCE_FROZEN`, and `REACTIVATED` on every rank.

Exact error (mpi2 raw log):

```
CASE_RESULT rank=1 case=healthy status=FAIL error="FieldError: type Array has no field `u`, available fields: `ref`, `size`"
CASE_RESULT rank=0 case=healthy status=FAIL error="FieldError: type Array has no field `u`, available fields: `ref`, `size`"
Stacktrace:
  [1] _pullback(::Zygote.Context{false}, ::typeof(ul), ::Vector{Float64}, ::Vector{Float64})
    @ Zygote ~/.julia/packages/Zygote/dywlZ/src/compiler/interface2.jl:87
  [5] #_eager_step!##0
    @ ~/projects/flux-integration/src/train.jl:386 [inlined]
```

Under NCCL the same defect reports (raw log):

```
FieldError: type CUDACore.CuArray has no field `u`, available fields: `data`, `maxsize`, `offset`, `dims`
```

**Root cause.** `cond_local_loss` called `ul(m.u, x)` and `vl(m.v, x)`, but
`ul`/`vl` already take the model and read `m.u`/`m.v` themselves; passing the
parameter array made the field access operate on a `Vector`/`CuArray`.

**Fix.** Call `ul(m, x)` and `vl(m, x)`. Probe md5 after the fix:
`e5d3dd019e833e83d3c6707f9fbf7485`. Failed attempt retained as
`green2-attempt1-healthy-bug-deathstar-loss-mpi2/`, `...-mpi4/`, `...-nccl2/`,
and `cleanup-concurrency-deathstar-attempt1/`.

**Re-run.** GREEN: `green2-deathstar-loss-mpi2/` 13/13, `...-mpi4/` 8/8,
`...-nccl2/` 12 PASS + 1 UNTESTED (`dup-train-one`); suite v2 mpi2 8/8, mpi4
8/8, nccl2 16/16 (MPI + NCCL passes). Commands, hashes, and results:
[COMMANDS.md](../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#v2-review-response-campaign-2026-10-13).

## 2. Documented deviation: no `reaped orphan pids` line in the campA console

The concurrent-cleanup check (`cleanup-concurrency-deathstar/`) ran campA
(`--case healthy`, `NUMERICAL_GUARD_EXECUTION_TIMEOUT=1`, token `campA`) and
campB (full loss mpi2, token `campB`) at the same time. campB completed
`PASS=13 FAIL=0 UNTESTED=0`; campA's verdict is FAIL with reason
`watchdog timeout (execution) after 40.3s`, with both ranks reaching
`GUARD_PROBE_READY` before the kill. `pgrep -f
conditional_graph_numerical_guard_probe.jl` was empty after both campaigns.

Deviation: `campA-console.log` has no `reaped orphan pids: ...` line. MPICH
Hydra terminates its own ranks when `stop_probe` kills the launcher, so
`reap_campaign(campA)` (the driver's final cleanup) found zero survivors. The
token-scoped match and kill are themselves proven by `--selftest-cleanup`:
`SELFTEST_CLEANUP PASS pids_a=563275 pids_b=563276 killed_a=563275
tokA=selftest-A-563235 tokB=selftest-B-563235`
(`artifacts/logs/conditional-wrapper/numerical-guards/selftest-cleanup-deathstar.log`).

## Impact

No product defect, no Flux source change, no commit. The attempt-1 defect is
harness-only; the v2 gates are GREEN with the fix. Evidence index:
[COMMANDS.md](../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md);
devlog
[review response](../devlog/02-conditional-graph/2026-10-13-db1-db4-coordinated-nonfinite-loss.md#review-response-second-pass);
audit
[resolution](../audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md#resolution-2026-10-13).

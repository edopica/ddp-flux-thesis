# 2026-09-03 — Precompilation experiment on Bocconi (branch `experiment/precompile`)

## Problem

- Every minor change to the dev'd Flux fork (`src/distributed/*.jl` is part of the Flux
  package itself) seemed to cost a long precompilation on Bocconi; tests sometimes died with
  "timeout on precomp". The workflow precompiled for changes it should not need to.

## Hypotheses (from Phase A audit)

- `make env` (setup_node) ran an explicit full `Pkg.precompile()` on the login node.
- salloc shells stay on the LOGIN node on this cluster; only `srun` reaches compute
  nodes -> caches compiled for the wrong CPU (graniterapids vs icelake-server).
- Tests run under a per-file watchdog (`FLUX_TEST_DISTRIBUTED_TIMEOUT`, 300 s default,
  120 s in several reports); silent on-node recompiles inside that window read as
  "timeout on precomp".

## Baseline measurements (Phase B, all on bocconi, Julia 1.12.6)

| Scenario | Wall time | Notes |
|---|---|---|
| Fresh `make precompile` after sync (login, 4 vCPU) | 359 s | only Flux + 3 exts recompiled (246 s julia) |
| Warm `make precompile` (login) | 195 s | nothing compiled; salloc+julia boot+staleness scan over beegfs |
| Warm 2-rank c8-mpi suite (gnode02) | 296 s | 7/8 PASS, zero on-node compile |
| Same suite right after a one-comment change, no precompile (gnode02) | 527 s | both ranks silently recompiled Flux on-node: first file 38 s -> 4m03/4m42 |
| Login precompile after the change (thesis env) | 360 s | graniterapids target — useless on gnode |
| Login precompile after the change (test env) | 250 s | 6 pkgs incl. 2 Mooncake exts |
| ...followed by warm-ish c8-mpi (gnode02) | 533 s | Flux REcompiled on-node again: login precompiles did not warm compute |
| gnode01 load probe 12 min after gnode02 compiled | 366 s | recompiled Flux+FluxMPIExt (146 s): cross-node miss in that state |

Root causes proven: (1) login vs compute CPU targets; (2) silent load-time recompile of
stale Flux inside the test watchdog; (3) login precompiles do not warm compute nodes.

## Mitigation experiments (Phase C)

| Experiment | Result |
|---|---|
| Precompile under `srun` on gnode02 (thesis env) | 20 deps compiled in 153 s (ELAPSED 200.8 s incl. salloc/boot) |
| Same for Flux test env | 62 deps in 204 s (ELAPSED 222.3 s) |
| Warm c8-mpi on gnode02 after that | suite 4m20, common 36.7 s — WARM, no compile |
| c8-mpi on gnode01 (cross-node, no pin) | suite 4m35, common 35.7 s — reused gnode02 caches (in this state) |
| `JULIA_CPU_TARGET=icelake-server` pin, login load probe | MISS: env var does not pin the loading process; only CLI `-C` |
| Login `-C icelake-server` probe vs gnode-native caches | MISS: gnode images carry `pconfig` (host CPUID feature), absent on login VM |
| Login status-quo probe (`using Flux`, no pin) | silent Precompiling within 2 s (the waste being eliminated) |
| Final: precompile on gnode02, warm suite verify | 7/8 PASS (deadlock = pre-existing fail), 5m09.9, zero "Precompiling" |

Verdict: CPU-target pinning is NOT viable (env var ignored at load; `pconfig` gap).
The fix is process-level: precompile only on compute nodes via `srun`.

## Implementation (Phase D, branch `experiment/precompile`, local changes)

- `Makefile`: `env` no longer precompiles + `JULIA_PKG_PRECOMPILE_AUTO=0`;
  `precompile`/`precompile-flux-test` are srun-guarded with a login-node refusal
  (hostname slnode01) and plain-julia local fallback; new `precompile-all`;
  `install-mpiexec` uses `force=true` (idempotent).
- `scripts/remote/precompile.sh`: now runs `make precompile-all FLUX_REPO_PATH=../ddp_flux`
  under NTASKS=1 CPUS_PER_TASK=8 (julia executes on the compute node).
- `scripts/remote/hosts/bocconi.conf`: exports `JULIA_PKG_PRECOMPILE_AUTO=0` + comments.
- `scripts/remote/setup_node.sh`: messaging updated (resolve-only).
- Docs: `wiki/context/remote-nodes.md`, `README.md`, `ADR-0005`.

## End-to-end validation (Phase E, fixed code)

| Step | Wall | Outcome |
|---|---|---|
| sync_code.sh | 15 s | new Makefile/scripts shipped; Flux snapshot reset (hash change) |
| setup_node.sh (login) | 4m17s | resolve-only, ZERO "Precompiling" lines (r1 failed on non-idempotent mpiexecjl -> force=true; also saw forced JLL install compiles ~22 s, unavoidable) |
| precompile.sh (gnode02, srun) | 10m04s | thesis env 114 deps / 369 s; test env 6 pkgs / 130 s (first pass after sync; steady state ~2-4 min) |
| warm c8-mpi | 4m53s | 7/8 PASS, suite 4m25, common 36.7/37.2 s, no Precompiling |
| stale-cache load probe (`JULIA_PKG_PRECOMPILE_AUTO=0` verified in env) | 3m12s | STILL silently recompiles (Flux 32.0 s + FluxMPIExt 27.5 s, "2 deps in 97 s"), no loud error — documented; workflow discipline is the mitigation |
| sed-restore of probe comment | — | hash back to synced state; load probe 39 s warm WITHOUT re-precompile |

End-to-end cost of the new workflow after a Flux change: sync 15 s + setup (only when
needed) + precompile ~2-10 min + warm tests ~5 min = vs before: ~10 min login
precompiles + 4-9 min cold/racing test runs + watchdog timeout risk.

## Failures recorded (also in Phase E logs)

- `salloc: error: _fork_command: Unable to find command "FLUX_REPO_PATH=../ddp_flux"`
  — env assignments cannot precede the command under salloc (execs argv). Fix: make-arg
  style (`make precompile-all FLUX_REPO_PATH=../ddp_flux`). Documented in remote-nodes.md.
- `ERROR: file /home/3320522/.julia/bin/mpiexecjl already exists` — fixed with
  `MPI.install_mpiexecjl(force=true)`.
- ssh drops during long salloc sessions: use `setsid ... > log 2>&1 < /dev/null &` and poll.
- `srun: error: Unable to allocate resources: User's group not permitted to use this
  partition` when calling bare srun without a full salloc allocation.
- deadlock_distributedtest fails 7/8 in every run — pre-existing, unrelated.

## Next action

- Review branch `experiment/precompile` changes; commit if approved.
- Optionally: harness hardening so a stale Flux is caught before tests start (e.g. a
  pre-flight `using Flux` warm probe or hash stamp), and revisit `FLUX_TEST_DISTRIBUTED_TIMEOUT`.

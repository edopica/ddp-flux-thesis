# DB-1/DB-4 numerical-guard check metadata

- date: 2026-10-13 18:39:04
- phase: green
- group: loss (loss only; DB-3 gradient guard is a separate plan)
- backend: mpi
- ranks: 2
- outdir: /home/picazio/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/numerical-guards/cleanup-concurrency-deathstar/campA
- parent julia: 1.12.6
- child project: /home/picazio/projects/flux-integration/test
- Flux source: HEAD unknown (src/train.jl=42b85187852f455bfab1dcd8113d3a26 src/distributed/public_api.jl=e4ed1f8d4e2aed25d99654ad05d54919)
- child versions (VERSIONS markers): julia=1.12.6 flux=0.16.11 mpi=0.20.27 optimisers=0.4.9
- startup watchdog: 420.0 s (NUMERICAL_GUARD_STARTUP_TIMEOUT)
- execution watchdog: 1.0 s (NUMERICAL_GUARD_EXECUTION_TIMEOUT)
- timeout semantics: a timeout always fails the case, even with plausible markers

## Cases

| case | fault_rank | fault_value | verdict | timeout | exit | elapsed_s | log |
|---|---|---|---|---|---|---|---|
| healthy | -1 | nan | FAIL | execution | 0 | 40.3 | raw-healthy-mpi2.log |

## Per-case commands and reasons

### healthy fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case healthy --backend mpi --ranks 2 --campaign campA`
- timeout: execution
- exit status: 0
- verdict: FAIL
- reason: watchdog timeout (execution) after 40.3s
- reason: missing CASE_RESULT rank=0
- reason: missing CASE_RESULT rank=1
- reason: rank 0: expected 19 collectives, got 0
- reason: rank 1: expected 19 collectives, got 0
- reason: rank 0: missing STEP_STATE_OK step=1
- reason: rank 0: missing STEP_STATE_OK step=2
- reason: rank 0: missing STEP_STATE_OK step=3
- reason: rank 0: missing STEP_STATE_OK step=4
- reason: rank 1: missing STEP_STATE_OK step=1
- reason: rank 1: missing STEP_STATE_OK step=2
- reason: rank 1: missing STEP_STATE_OK step=3
- reason: rank 1: missing STEP_STATE_OK step=4
- reason: rank 0: missing ABSENCE_FROZEN
- reason: rank 0: missing REACTIVATED
- reason: rank 1: missing ABSENCE_FROZEN
- reason: rank 1: missing REACTIVATED

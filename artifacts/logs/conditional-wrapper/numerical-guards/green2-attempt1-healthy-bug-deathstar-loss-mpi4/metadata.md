# DB-1/DB-4 numerical-guard check metadata

- date: 2026-10-13 17:46:40
- phase: green
- group: loss (loss only; DB-3 gradient guard is a separate plan)
- backend: mpi
- ranks: 4
- outdir: /home/picazio/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/numerical-guards/green2-deathstar-loss-mpi4
- parent julia: 1.12.6
- child project: /home/picazio/projects/flux-integration/test
- Flux source: HEAD unknown (src/train.jl=42b85187852f455bfab1dcd8113d3a26 src/distributed/public_api.jl=e4ed1f8d4e2aed25d99654ad05d54919)
- child versions (VERSIONS markers): julia=1.12.6 flux=0.16.11 mpi=0.20.27 optimisers=0.4.9
- startup watchdog: 420.0 s (NUMERICAL_GUARD_STARTUP_TIMEOUT)
- execution watchdog: 120.0 s (NUMERICAL_GUARD_EXECUTION_TIMEOUT)
- timeout semantics: a timeout always fails the case, even with plausible markers

## Cases

| case | fault_rank | fault_value | verdict | timeout | exit | elapsed_s | log |
|---|---|---|---|---|---|---|---|
| train-one | 0 | nan | PASS | none | 0 | 23.8 | raw-train-one-mpi4-fault0.log |
| train-one | 2 | nan | PASS | none | 0 | 25.4 | raw-train-one-mpi4-fault2.log |
| train-one-inf | 3 | inf | PASS | none | 0 | 23.8 | raw-train-one-inf-mpi4-fault3.log |
| loop-one | 3 | nan | PASS | none | 0 | 24.4 | raw-loop-one-mpi4-fault3.log |
| aux-nan-first | 2 | nan | PASS | none | 0 | 25.0 | raw-aux-nan-first-mpi4-fault2.log |
| no-trainable | 2 | nan | PASS | none | 0 | 20.4 | raw-no-trainable-mpi4-fault2.log |
| recover-one | 3 | nan | PASS | none | 0 | 25.6 | raw-recover-one-mpi4-fault3.log |
| healthy | -1 | nan | FAIL | none | 1 | 23.8 | raw-healthy-mpi4.log |

## Per-case commands and reasons

### train-one fault_rank=0 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one --backend mpi --ranks 4 --campaign 564635-3052838537570315 --fault-rank 0 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### train-one fault_rank=2 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one --backend mpi --ranks 4 --campaign 564635-3052838537570315 --fault-rank 2 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### train-one-inf fault_rank=3 fault_value=inf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one-inf --backend mpi --ranks 4 --campaign 564635-3052838537570315 --fault-rank 3 --fault-value inf`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### loop-one fault_rank=3 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case loop-one --backend mpi --ranks 4 --campaign 564635-3052838537570315 --fault-rank 3 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### aux-nan-first fault_rank=2 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case aux-nan-first --backend mpi --ranks 4 --campaign 564635-3052838537570315 --fault-rank 2 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### no-trainable fault_rank=2 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case no-trainable --backend mpi --ranks 4 --campaign 564635-3052838537570315 --fault-rank 2 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### recover-one fault_rank=3 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case recover-one --backend mpi --ranks 4 --campaign 564635-3052838537570315 --fault-rank 3 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### healthy fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case healthy --backend mpi --ranks 4 --campaign 564635-3052838537570315`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: no GUARD_PROBE_READY on all 4 ranks
- reason: rank 0 error: FieldError: type Array has no field `u`, available fields: `ref`, `size`
- reason: missing CASE_RESULT rank=1
- reason: rank 2 error: FieldError: type Array has no field `u`, available fields: `ref`, `size`
- reason: rank 3 error: FieldError: type Array has no field `u`, available fields: `ref`, `size`
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 2 CASE_RESULT status=FAIL
- reason: rank 3 CASE_RESULT status=FAIL
- reason: rank 0: expected 19 collectives, got 0
- reason: rank 1: expected 19 collectives, got 0
- reason: rank 2: expected 19 collectives, got 0
- reason: rank 3: expected 19 collectives, got 0
- reason: rank 0: missing STEP_STATE_OK step=1
- reason: rank 0: missing STEP_STATE_OK step=2
- reason: rank 0: missing STEP_STATE_OK step=3
- reason: rank 0: missing STEP_STATE_OK step=4
- reason: rank 1: missing STEP_STATE_OK step=1
- reason: rank 1: missing STEP_STATE_OK step=2
- reason: rank 1: missing STEP_STATE_OK step=3
- reason: rank 1: missing STEP_STATE_OK step=4
- reason: rank 2: missing STEP_STATE_OK step=1
- reason: rank 2: missing STEP_STATE_OK step=2
- reason: rank 2: missing STEP_STATE_OK step=3
- reason: rank 2: missing STEP_STATE_OK step=4
- reason: rank 3: missing STEP_STATE_OK step=1
- reason: rank 3: missing STEP_STATE_OK step=2
- reason: rank 3: missing STEP_STATE_OK step=3
- reason: rank 3: missing STEP_STATE_OK step=4
- reason: rank 0: missing ABSENCE_FROZEN
- reason: rank 0: missing REACTIVATED
- reason: rank 1: missing ABSENCE_FROZEN
- reason: rank 1: missing REACTIVATED
- reason: rank 2: missing ABSENCE_FROZEN
- reason: rank 2: missing REACTIVATED
- reason: rank 3: missing ABSENCE_FROZEN
- reason: rank 3: missing REACTIVATED
- reason: child exit code 1

# DB-1/DB-4 numerical-guard check metadata

- date: 2026-10-14 12:22:39
- phase: green
- group: loss (loss only; DB-3 gradient guard is a separate plan)
- backend: mpi
- ranks: 2
- outdir: /home/picazio/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/numerical-guards/green3-deathstar-loss-mpi2
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
| train-one | 0 | nan | PASS | none | 0 | 23.2 | raw-train-one-mpi2-fault0.log |
| train-one | 1 | nan | PASS | none | 0 | 22.4 | raw-train-one-mpi2-fault1.log |
| train-one-inf | 1 | inf | PASS | none | 0 | 22.6 | raw-train-one-inf-mpi2-fault1.log |
| train-one-ninf | 1 | ninf | PASS | none | 0 | 22.4 | raw-train-one-ninf-mpi2-fault1.log |
| loop-one | 1 | nan | PASS | none | 0 | 22.6 | raw-loop-one-mpi2-fault1.log |
| loop-withgradient-one | 1 | nan | PASS | none | 0 | 22.6 | raw-loop-withgradient-one-mpi2-fault1.log |
| train-all | -1 | nan | PASS | none | 0 | 23.0 | raw-train-all-mpi2.log |
| aux-nonfinite-ok | -1 | nan | PASS | none | 0 | 22.4 | raw-aux-nonfinite-ok-mpi2.log |
| aux-nan-first | 1 | nan | PASS | none | 0 | 22.8 | raw-aux-nan-first-mpi2-fault1.log |
| no-trainable | 1 | nan | PASS | none | 0 | 19.1 | raw-no-trainable-mpi2-fault1.log |
| recover-one | 1 | nan | PASS | none | 0 | 23.0 | raw-recover-one-mpi2-fault1.log |
| healthy | -1 | nan | PASS | none | 0 | 36.0 | raw-healthy-mpi2.log |
| dup-train-one | 1 | nan | PASS | none | 0 | 56.2 | raw-dup-train-one-mpi2-fault1.log |

## Per-case commands and reasons

### train-one fault_rank=0 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one --backend mpi --ranks 2 --campaign 585738-3119651523052731 --fault-rank 0 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### train-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one --backend mpi --ranks 2 --campaign 585738-3119651523052731 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### train-one-inf fault_rank=1 fault_value=inf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one-inf --backend mpi --ranks 2 --campaign 585738-3119651523052731 --fault-rank 1 --fault-value inf`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### train-one-ninf fault_rank=1 fault_value=ninf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one-ninf --backend mpi --ranks 2 --campaign 585738-3119651523052731 --fault-rank 1 --fault-value ninf`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### loop-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case loop-one --backend mpi --ranks 2 --campaign 585738-3119651523052731 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### loop-withgradient-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case loop-withgradient-one --backend mpi --ranks 2 --campaign 585738-3119651523052731 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### train-all fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-all --backend mpi --ranks 2 --campaign 585738-3119651523052731`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### aux-nonfinite-ok fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case aux-nonfinite-ok --backend mpi --ranks 2 --campaign 585738-3119651523052731`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### aux-nan-first fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case aux-nan-first --backend mpi --ranks 2 --campaign 585738-3119651523052731 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### no-trainable fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case no-trainable --backend mpi --ranks 2 --campaign 585738-3119651523052731 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### recover-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case recover-one --backend mpi --ranks 2 --campaign 585738-3119651523052731 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### healthy fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case healthy --backend mpi --ranks 2 --campaign 585738-3119651523052731`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### dup-train-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case dup-train-one --backend mpi --ranks 2 --campaign 585738-3119651523052731 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

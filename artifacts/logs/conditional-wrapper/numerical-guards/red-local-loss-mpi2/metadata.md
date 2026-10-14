# DB-1/DB-4 numerical-guard check metadata

- date: 2026-10-13 14:57:18
- phase: green
- group: loss (loss only; DB-3 gradient guard is a separate plan)
- backend: mpi
- ranks: 2
- outdir: /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/logs/conditional-wrapper/numerical-guards/red-local-loss-mpi2
- parent julia: 1.12.6
- child project: /home/kurapica/Projects/ddp_flux/flux-integration/test
- Flux source: HEAD 55171e659b2f425da980cbc83177bf5c38f0a7a1 (src/train.jl=eea4e545d3eb2118c3a5b23653c7a71a src/distributed/public_api.jl=9bfa981384eaf88750ec3d19e8bb76bd)
- child versions (VERSIONS markers): julia=1.12.6 flux=0.16.11 mpi=0.20.27 optimisers=0.4.9
- startup watchdog: 420.0 s (NUMERICAL_GUARD_STARTUP_TIMEOUT)
- execution watchdog: 120.0 s (NUMERICAL_GUARD_EXECUTION_TIMEOUT)
- timeout semantics: a timeout always fails the case, even with plausible markers

## Cases

| case | fault_rank | fault_value | verdict | timeout | exit | elapsed_s | log |
|---|---|---|---|---|---|---|---|
| train-one | 0 | nan | FAIL | execution | 15 | 147.1 | raw-train-one-mpi2-fault0.log |
| train-one | 1 | nan | FAIL | execution | 15 | 146.8 | raw-train-one-mpi2-fault1.log |
| train-one-inf | 1 | inf | FAIL | execution | 15 | 144.9 | raw-train-one-inf-mpi2-fault1.log |
| train-one-ninf | 1 | ninf | FAIL | execution | 15 | 146.0 | raw-train-one-ninf-mpi2-fault1.log |
| loop-one | 1 | nan | FAIL | execution | 15 | 147.5 | raw-loop-one-mpi2-fault1.log |
| loop-withgradient-one | 1 | nan | FAIL | execution | 15 | 145.4 | raw-loop-withgradient-one-mpi2-fault1.log |
| train-all | -1 | nan | FAIL | none | 0 | 26.6 | raw-train-all-mpi2.log |
| aux-nonfinite-ok | -1 | nan | FAIL | none | 0 | 25.8 | raw-aux-nonfinite-ok-mpi2.log |
| aux-nan-first | 1 | nan | FAIL | execution | 15 | 145.4 | raw-aux-nan-first-mpi2-fault1.log |
| no-trainable | 1 | nan | FAIL | none | 1 | 23.0 | raw-no-trainable-mpi2-fault1.log |
| recover-one | 1 | nan | FAIL | execution | 15 | 146.2 | raw-recover-one-mpi2-fault1.log |
| healthy | -1 | nan | FAIL | none | 0 | 23.2 | raw-healthy-mpi2.log |
| dup-train-one | 1 | nan | FAIL | execution | 15 | 180.7 | raw-dup-train-one-mpi2-fault1.log |

## Per-case commands and reasons

### train-one fault_rank=0 fault_value=nan

- command: `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 /home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration/test --startup-file=no --color=no /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one --backend mpi --ranks 2 --fault-rank 0 --fault-value nan`
- timeout: execution
- exit status: 15
- verdict: FAIL
- reason: watchdog timeout (execution) after 147.1s
- reason: missing CASE_RESULT rank=1
- reason: rank 0: expected 9 collectives, got 6
- reason: rank 0: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 3 more metadata mismatches
- reason: rank 1: expected 9 collectives, got 7
- reason: rank 1: 7 begins but 6 ends (unmatched collective)
- reason: rank 1: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 4 more metadata mismatches

### train-one fault_rank=1 fault_value=nan

- command: `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 /home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration/test --startup-file=no --color=no /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one --backend mpi --ranks 2 --fault-rank 1 --fault-value nan`
- timeout: execution
- exit status: 15
- verdict: FAIL
- reason: watchdog timeout (execution) after 146.8s
- reason: missing CASE_RESULT rank=0
- reason: rank 0: expected 9 collectives, got 7
- reason: rank 0: 7 begins but 6 ends (unmatched collective)
- reason: rank 0: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 4 more metadata mismatches
- reason: rank 1: expected 9 collectives, got 6
- reason: rank 1: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 3 more metadata mismatches

### train-one-inf fault_rank=1 fault_value=inf

- command: `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 /home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration/test --startup-file=no --color=no /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one-inf --backend mpi --ranks 2 --fault-rank 1 --fault-value inf`
- timeout: execution
- exit status: 15
- verdict: FAIL
- reason: watchdog timeout (execution) after 144.9s
- reason: missing CASE_RESULT rank=0
- reason: rank 0: expected 9 collectives, got 7
- reason: rank 0: 7 begins but 6 ends (unmatched collective)
- reason: rank 0: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 4 more metadata mismatches
- reason: rank 1: expected 9 collectives, got 6
- reason: rank 1: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 3 more metadata mismatches

### train-one-ninf fault_rank=1 fault_value=ninf

- command: `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 /home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration/test --startup-file=no --color=no /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one-ninf --backend mpi --ranks 2 --fault-rank 1 --fault-value ninf`
- timeout: execution
- exit status: 15
- verdict: FAIL
- reason: watchdog timeout (execution) after 146.0s
- reason: missing CASE_RESULT rank=0
- reason: rank 0: expected 9 collectives, got 7
- reason: rank 0: 7 begins but 6 ends (unmatched collective)
- reason: rank 0: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 4 more metadata mismatches
- reason: rank 1: expected 9 collectives, got 6
- reason: rank 1: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 3 more metadata mismatches

### loop-one fault_rank=1 fault_value=nan

- command: `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 /home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration/test --startup-file=no --color=no /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case loop-one --backend mpi --ranks 2 --fault-rank 1 --fault-value nan`
- timeout: execution
- exit status: 15
- verdict: FAIL
- reason: watchdog timeout (execution) after 147.5s
- reason: missing CASE_RESULT rank=0
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 9 collectives, got 7
- reason: rank 0: 7 begins but 6 ends (unmatched collective)
- reason: rank 0: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 4 more metadata mismatches
- reason: rank 1: expected 9 collectives, got 6
- reason: rank 1: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 3 more metadata mismatches

### loop-withgradient-one fault_rank=1 fault_value=nan

- command: `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 /home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration/test --startup-file=no --color=no /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case loop-withgradient-one --backend mpi --ranks 2 --fault-rank 1 --fault-value nan`
- timeout: execution
- exit status: 15
- verdict: FAIL
- reason: watchdog timeout (execution) after 145.4s
- reason: missing CASE_RESULT rank=0
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 9 collectives, got 7
- reason: rank 0: 7 begins but 6 ends (unmatched collective)
- reason: rank 0: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 4 more metadata mismatches
- reason: rank 1: expected 9 collectives, got 6
- reason: rank 1: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 3 more metadata mismatches

### train-all fault_rank=-1 fault_value=nan

- command: `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 /home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration/test --startup-file=no --color=no /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-all --backend mpi --ranks 2`
- timeout: none
- exit status: 0
- verdict: FAIL
- reason: rank 0: expected 9 collectives, got 6
- reason: rank 0: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 3 more metadata mismatches
- reason: rank 1: expected 9 collectives, got 6
- reason: rank 1: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 3 more metadata mismatches

### aux-nonfinite-ok fault_rank=-1 fault_value=nan

- command: `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 /home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration/test --startup-file=no --color=no /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case aux-nonfinite-ok --backend mpi --ranks 2`
- timeout: none
- exit status: 0
- verdict: FAIL
- reason: rank 0: expected 12 collectives, got 9
- reason: rank 0: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 6 more metadata mismatches
- reason: rank 1: expected 12 collectives, got 9
- reason: rank 1: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 6 more metadata mismatches

### aux-nan-first fault_rank=1 fault_value=nan

- command: `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 /home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration/test --startup-file=no --color=no /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case aux-nan-first --backend mpi --ranks 2 --fault-rank 1 --fault-value nan`
- timeout: execution
- exit status: 15
- verdict: FAIL
- reason: watchdog timeout (execution) after 145.4s
- reason: missing CASE_RESULT rank=0
- reason: rank 0: expected 9 collectives, got 7
- reason: rank 0: 7 begins but 6 ends (unmatched collective)
- reason: rank 0: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 4 more metadata mismatches
- reason: rank 1: expected 9 collectives, got 6
- reason: rank 1: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 3 more metadata mismatches

### no-trainable fault_rank=1 fault_value=nan

- command: `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 /home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration/test --startup-file=no --color=no /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case no-trainable --backend mpi --ranks 2 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 0: expected 3 collectives, got 0
- reason: rank 1: expected 3 collectives, got 0
- reason: child exit code 1

### recover-one fault_rank=1 fault_value=nan

- command: `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 /home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration/test --startup-file=no --color=no /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case recover-one --backend mpi --ranks 2 --fault-rank 1 --fault-value nan`
- timeout: execution
- exit status: 15
- verdict: FAIL
- reason: watchdog timeout (execution) after 146.2s
- reason: missing CASE_RESULT rank=0
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 13 collectives, got 16
- reason: rank 0: 16 begins but 15 ends (unmatched collective)
- reason: rank 0: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 10 more metadata mismatches
- reason: rank 1: expected 13 collectives, got 15
- reason: rank 1: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 10 more metadata mismatches

### healthy fault_rank=-1 fault_value=nan

- command: `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 /home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration/test --startup-file=no --color=no /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case healthy --backend mpi --ranks 2`
- timeout: none
- exit status: 0
- verdict: FAIL
- reason: rank 0: expected 16 collectives, got 12
- reason: rank 0: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 6 more metadata mismatches
- reason: rank 1: expected 16 collectives, got 12
- reason: rank 1: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 6 more metadata mismatches

### dup-train-one fault_rank=1 fault_value=nan

- command: `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 /home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/kurapica/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration/test --startup-file=no --color=no /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case dup-train-one --backend mpi --ranks 2 --fault-rank 1 --fault-value nan`
- timeout: execution
- exit status: 15
- verdict: FAIL
- reason: watchdog timeout (execution) after 180.7s
- reason: missing CASE_RESULT rank=0
- reason: rank 0: expected 9 collectives, got 7
- reason: rank 0: 7 begins but 6 ends (unmatched collective)
- reason: rank 0: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 4 more metadata mismatches
- reason: rank 1: expected 9 collectives, got 6
- reason: rank 1: ordinal 1 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 2 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 3 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 3 more metadata mismatches

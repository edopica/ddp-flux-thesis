# DB-1/DB-4/DB-3 numerical-guard check metadata

- date: 2026-10-14 15:57:30
- phase: red
- group: gradient (loss = DB-1/DB-4 loss vote; gradient = DB-3 gradient vote; all = loss cases followed by gradient cases in one campaign)
- backend: mpi
- ranks: 2
- outdir: /home/picazio/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/numerical-guards/red-deathstar-gradient-mpi2
- parent julia: 1.12.6
- child project: /home/picazio/projects/flux-integration/test
- Flux source: HEAD unknown (src/train.jl=42b85187852f455bfab1dcd8113d3a26 src/distributed/public_api.jl=e4ed1f8d4e2aed25d99654ad05d54919)
- child versions (VERSIONS markers): julia=1.12.6 flux=0.16.11 mpi=0.20.27 optimisers=0.4.9
- startup watchdog: 420.0 s (NUMERICAL_GUARD_STARTUP_TIMEOUT)
- execution watchdog: 120.0 s (NUMERICAL_GUARD_EXECUTION_TIMEOUT)
- timeout semantics: a timeout always fails the case, even with plausible markers
- gradient vote: every nonempty `_sync_gradients` runs one extra host Float32 len-1 `avg` collective after the presence and gradient reductions; `np == 0` keeps zero collectives
- rejected steps: every rank completes the full expected schedule before throwing (eager: loss vote -> presence -> P reductions -> gradient vote; explicit: presence -> P reductions -> gradient vote); a loss-rejected eager step issues only the loss vote
- classification: `:vote`, `:presence` and `:gradvote` are metadata-identical (host Float32 len 1; `:presence` when np == 1), so stages are classified by ordinal within the explicit expected schedule
- schedules: eager GuardModel = vote, presence, grad_a, grad_c, gradvote; explicit GuardModel = presence, grad_a, grad_c, gradvote; explicit CondModel = presence, grad_w, grad_u, grad_v, gradvote; eager CondModel = vote, presence, grad_w, grad_u, grad_v, gradvote, or without grad_v when v is globally absent; no-trainable = one loss vote per eager step and zero wrapper collectives; empty buffer = presence, grad_e (len 0), gradvote

## Cases

| case | fault_rank | fault_value | verdict | timeout | exit | elapsed_s | log |
|---|---|---|---|---|---|---|---|
| grad-train-one | 0 | nan | FAIL | none | 1 | 42.5 | raw-grad-train-one-mpi2-fault0.log |
| grad-train-one | 1 | nan | FAIL | none | 1 | 41.3 | raw-grad-train-one-mpi2-fault1.log |
| grad-audit-zero | 1 | nan | FAIL | none | 1 | 41.7 | raw-grad-audit-zero-mpi2-fault1.log |
| grad-late | 1 | nan | FAIL | none | 1 | 45.5 | raw-grad-late-mpi2-fault1.log |
| grad-inf | 1 | inf | FAIL | none | 1 | 41.1 | raw-grad-inf-mpi2-fault1.log |
| grad-ninf | 1 | ninf | FAIL | none | 1 | 41.3 | raw-grad-ninf-mpi2-fault1.log |
| grad-update-both | -1 | nan | FAIL | none | 1 | 41.7 | raw-grad-update-both-mpi2.log |
| grad-overflow | -1 | inf | FAIL | none | 1 | 41.3 | raw-grad-overflow-mpi2.log |
| grad-tied-overflow | -1 | inf | FAIL | none | 1 | 41.5 | raw-grad-tied-overflow-mpi2.log |
| grad-asym-vote | -1 | nan | FAIL | none | 1 | 41.3 | raw-grad-asym-vote-mpi2.log |
| grad-caller | -1 | nan | FAIL | none | 1 | 41.3 | raw-grad-caller-mpi2.log |
| grad-conditional-absence | 0 | nan | FAIL | none | 1 | 41.9 | raw-grad-conditional-absence-mpi2-fault0.log |
| grad-absent-history | -1 | nan | FAIL | none | 0 | 46.4 | raw-grad-absent-history-mpi2.log |
| grad-all-absent | -1 | nan | FAIL | none | 0 | 41.3 | raw-grad-all-absent-mpi2.log |
| grad-no-trainable | -1 | nan | FAIL | none | 1 | 43.1 | raw-grad-no-trainable-mpi2.log |
| grad-empty | -1 | nan | FAIL | none | 0 | 40.9 | raw-grad-empty-mpi2.log |
| grad-frozen | 1 | nan | FAIL | none | 1 | 41.5 | raw-grad-frozen-mpi2-fault1.log |
| grad-recover | 1 | nan | FAIL | none | 1 | 42.9 | raw-grad-recover-mpi2-fault1.log |
| grad-healthy | -1 | nan | FAIL | none | 0 | 47.7 | raw-grad-healthy-mpi2.log |

## Per-case commands and reasons

### grad-train-one fault_rank=0 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-train-one --backend mpi --ranks 2 --campaign 594636-3132074413688452 --fault-rank 0 --fault-value nan`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 15 collectives, got 13
- reason: rank 0: ordinal 6 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 7 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 8 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 5 more metadata mismatches
- reason: rank 1: expected 15 collectives, got 13
- reason: rank 1: ordinal 6 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 7 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 8 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 5 more metadata mismatches
- reason: child exit code 1

### grad-train-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-train-one --backend mpi --ranks 2 --campaign 594636-3132074413688452 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 15 collectives, got 13
- reason: rank 0: ordinal 6 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 7 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 8 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 5 more metadata mismatches
- reason: rank 1: expected 15 collectives, got 13
- reason: rank 1: ordinal 6 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 7 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 8 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 5 more metadata mismatches
- reason: child exit code 1

### grad-audit-zero fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-audit-zero --backend mpi --ranks 2 --campaign 594636-3132074413688452 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ordinal 8 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ordinal 8 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: child exit code 1

### grad-late fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-late --backend mpi --ranks 2 --campaign 594636-3132074413688452 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 15 collectives, got 13
- reason: rank 0: ordinal 6 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 7 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 8 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 5 more metadata mismatches
- reason: rank 1: expected 15 collectives, got 13
- reason: rank 1: ordinal 6 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 7 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 8 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 5 more metadata mismatches
- reason: child exit code 1

### grad-inf fault_rank=1 fault_value=inf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-inf --backend mpi --ranks 2 --campaign 594636-3132074413688452 --fault-rank 1 --fault-value inf`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 12 collectives, got 9
- reason: rank 0: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 3 more metadata mismatches
- reason: rank 1: expected 12 collectives, got 9
- reason: rank 1: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 3 more metadata mismatches
- reason: child exit code 1

### grad-ninf fault_rank=1 fault_value=ninf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-ninf --backend mpi --ranks 2 --campaign 594636-3132074413688452 --fault-rank 1 --fault-value ninf`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 12 collectives, got 9
- reason: rank 0: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 3 more metadata mismatches
- reason: rank 1: expected 12 collectives, got 9
- reason: rank 1: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 3 more metadata mismatches
- reason: child exit code 1

### grad-update-both fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-update-both --backend mpi --ranks 2 --campaign 594636-3132074413688452`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 12 collectives, got 9
- reason: rank 0: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 3 more metadata mismatches
- reason: rank 1: expected 12 collectives, got 9
- reason: rank 1: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 3 more metadata mismatches
- reason: child exit code 1

### grad-overflow fault_rank=-1 fault_value=inf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-overflow --backend mpi --ranks 2 --campaign 594636-3132074413688452`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 12 collectives, got 9
- reason: rank 0: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 3 more metadata mismatches
- reason: rank 1: expected 12 collectives, got 9
- reason: rank 1: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 3 more metadata mismatches
- reason: child exit code 1

### grad-tied-overflow fault_rank=-1 fault_value=inf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-tied-overflow --backend mpi --ranks 2 --campaign 594636-3132074413688452`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 12 collectives, got 9
- reason: rank 0: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 3 more metadata mismatches
- reason: rank 1: expected 12 collectives, got 9
- reason: rank 1: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 3 more metadata mismatches
- reason: child exit code 1

### grad-asym-vote fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-asym-vote --backend mpi --ranks 2 --campaign 594636-3132074413688452`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 12 collectives, got 9
- reason: rank 0: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 3 more metadata mismatches
- reason: rank 1: expected 12 collectives, got 9
- reason: rank 1: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 3 more metadata mismatches
- reason: child exit code 1

### grad-caller fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-caller --backend mpi --ranks 2 --campaign 594636-3132074413688452`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 12 collectives, got 9
- reason: rank 0: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 3 more metadata mismatches
- reason: rank 1: expected 12 collectives, got 9
- reason: rank 1: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 3 more metadata mismatches
- reason: child exit code 1

### grad-conditional-absence fault_rank=0 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-conditional-absence --backend mpi --ranks 2 --campaign 594636-3132074413688452 --fault-rank 0 --fault-value nan`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 10 collectives, got 8
- reason: rank 0: ordinal 5 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=3 buftype=host)
- reason: rank 0: ordinal 6 expected presence (op=avg eltype=Float32 len=3 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 7 expected grad_w (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 1 more metadata mismatches
- reason: rank 1: expected 10 collectives, got 8
- reason: rank 1: ordinal 5 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=3 buftype=host)
- reason: rank 1: ordinal 6 expected presence (op=avg eltype=Float32 len=3 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 7 expected grad_w (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 1 more metadata mismatches
- reason: child exit code 1

### grad-absent-history fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-absent-history --backend mpi --ranks 2 --campaign 594636-3132074413688452`
- timeout: none
- exit status: 0
- verdict: FAIL
- reason: rank 0: expected 23 collectives, got 19
- reason: rank 0: ordinal 7 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=3 buftype=host)
- reason: rank 0: ordinal 8 expected presence (op=avg eltype=Float32 len=3 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 9 expected grad_w (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 10 more metadata mismatches
- reason: rank 1: expected 23 collectives, got 19
- reason: rank 1: ordinal 7 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=3 buftype=host)
- reason: rank 1: ordinal 8 expected presence (op=avg eltype=Float32 len=3 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 9 expected grad_w (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 10 more metadata mismatches

### grad-all-absent fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-all-absent --backend mpi --ranks 2 --campaign 594636-3132074413688452`
- timeout: none
- exit status: 0
- verdict: FAIL
- reason: rank 0: expected 10 collectives, got 7
- reason: rank 0: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 1 more metadata mismatches
- reason: rank 1: expected 10 collectives, got 7
- reason: rank 1: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 1 more metadata mismatches

### grad-no-trainable fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-no-trainable --backend mpi --ranks 2 --campaign 594636-3132074413688452`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 error: MethodError: no method matching (::var'#run_grad_no_trainable##0#run_grad_no_trainable##1')(::EmptyModel, ::Int64) The function `#run_grad_no_trainable##0` exists, but no method is defined for this combination of argument types.  Closest candidates are:   (::var'#run_grad_no_trainable##0#run_grad_no_trainable##1')(::Any, ::Any, !Matched::Any)    @ Main ~/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl:1159 
- reason: rank 1 error: MethodError: no method matching (::var'#run_grad_no_trainable##0#run_grad_no_trainable##1')(::EmptyModel, ::Int64) The function `#run_grad_no_trainable##0` exists, but no method is defined for this combination of argument types.  Closest candidates are:   (::var'#run_grad_no_trainable##0#run_grad_no_trainable##1')(::Any, ::Any, !Matched::Any)    @ Main ~/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl:1159 
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 3 collectives, got 0
- reason: rank 1: expected 3 collectives, got 0
- reason: child exit code 1

### grad-empty fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-empty --backend mpi --ranks 2 --campaign 594636-3132074413688452`
- timeout: none
- exit status: 0
- verdict: FAIL
- reason: rank 0: expected 3 collectives, got 2
- reason: rank 1: expected 3 collectives, got 2

### grad-frozen fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-frozen --backend mpi --ranks 2 --campaign 594636-3132074413688452 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 8 collectives, got 6
- reason: rank 0: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: expected 8 collectives, got 6
- reason: rank 1: ordinal 4 expected gradvote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 5 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 6 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: child exit code 1

### grad-recover fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-recover --backend mpi --ranks 2 --campaign 594636-3132074413688452 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 1
- verdict: FAIL
- reason: rank 0 error: DomainError with NaN: DistributedOptimizer rejected the step: at least one rank reported a non-finite scalar loss. No optimizer update occurred: parameters and optimizer state are unchanged. The local scalar loss is not necessarily the offending value.
- reason: rank 1 error: DomainError with NaN: DistributedOptimizer rejected the step: at least one rank reported a non-finite scalar loss. No optimizer update occurred: parameters and optimizer state are unchanged. The local scalar loss is not necessarily the offending value.
- reason: rank 0 CASE_RESULT status=FAIL
- reason: rank 1 CASE_RESULT status=FAIL
- reason: rank 0: expected 20 collectives, got 13
- reason: rank 0: ordinal 6 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 0: ordinal 7 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 8 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 5 more metadata mismatches
- reason: rank 1: expected 20 collectives, got 13
- reason: rank 1: ordinal 6 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=2 buftype=host)
- reason: rank 1: ordinal 7 expected presence (op=avg eltype=Float32 len=2 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 8 expected grad_a (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 5 more metadata mismatches
- reason: child exit code 1

### grad-healthy fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-healthy --backend mpi --ranks 2 --campaign 594636-3132074413688452`
- timeout: none
- exit status: 0
- verdict: FAIL
- reason: rank 0: expected 23 collectives, got 19
- reason: rank 0: ordinal 7 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=3 buftype=host)
- reason: rank 0: ordinal 8 expected presence (op=avg eltype=Float32 len=3 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 0: ordinal 9 expected grad_w (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 0: ... 10 more metadata mismatches
- reason: rank 1: expected 23 collectives, got 19
- reason: rank 1: ordinal 7 expected vote (op=avg eltype=Float32 len=1 buftype=host) got (op=avg eltype=Float32 len=3 buftype=host)
- reason: rank 1: ordinal 8 expected presence (op=avg eltype=Float32 len=3 buftype=host) got (op=avg eltype=Float64 len=2 buftype=host)
- reason: rank 1: ordinal 9 expected grad_w (op=avg eltype=Float64 len=2 buftype=host) got (op=avg eltype=Float64 len=3 buftype=host)
- reason: rank 1: ... 10 more metadata mismatches

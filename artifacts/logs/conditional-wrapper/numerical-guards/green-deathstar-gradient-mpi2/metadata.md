# DB-1/DB-4/DB-3 numerical-guard check metadata

- date: 2026-10-14 17:02:53
- phase: green
- group: gradient (loss = DB-1/DB-4 loss vote; gradient = DB-3 gradient vote; all = loss cases followed by gradient cases in one campaign)
- backend: mpi
- ranks: 2
- outdir: /home/picazio/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-gradient-mpi2
- parent julia: 1.12.6
- child project: /home/picazio/projects/flux-integration/test
- Flux source: HEAD unknown (src/train.jl=42b85187852f455bfab1dcd8113d3a26 src/distributed/public_api.jl=d21c73f8885b35ae8558a0f0d4e43196)
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
| grad-train-one | 0 | nan | PASS | none | 0 | 41.1 | raw-grad-train-one-mpi2-fault0.log |
| grad-train-one | 1 | nan | PASS | none | 0 | 41.5 | raw-grad-train-one-mpi2-fault1.log |
| grad-audit-zero | 1 | nan | PASS | none | 0 | 41.7 | raw-grad-audit-zero-mpi2-fault1.log |
| grad-late | 1 | nan | PASS | none | 0 | 45.3 | raw-grad-late-mpi2-fault1.log |
| grad-inf | 1 | inf | PASS | none | 0 | 41.5 | raw-grad-inf-mpi2-fault1.log |
| grad-ninf | 1 | ninf | PASS | none | 0 | 41.7 | raw-grad-ninf-mpi2-fault1.log |
| grad-update-both | 0 | nan | PASS | none | 0 | 41.9 | raw-grad-update-both-mpi2-fault0.log |
| grad-overflow | -1 | inf | PASS | none | 0 | 41.3 | raw-grad-overflow-mpi2.log |
| grad-tied-overflow | -1 | inf | PASS | none | 0 | 41.5 | raw-grad-tied-overflow-mpi2.log |
| grad-asym-vote | -1 | nan | PASS | none | 0 | 41.7 | raw-grad-asym-vote-mpi2.log |
| grad-caller | 0 | nan | PASS | none | 0 | 41.3 | raw-grad-caller-mpi2-fault0.log |
| grad-conditional-absence | 0 | nan | PASS | none | 0 | 41.7 | raw-grad-conditional-absence-mpi2-fault0.log |
| grad-absent-history | -1 | nan | PASS | none | 0 | 46.6 | raw-grad-absent-history-mpi2.log |
| grad-all-absent | -1 | nan | PASS | none | 0 | 41.5 | raw-grad-all-absent-mpi2.log |
| grad-no-trainable | -1 | nan | PASS | none | 0 | 40.9 | raw-grad-no-trainable-mpi2.log |
| grad-empty | -1 | nan | PASS | none | 0 | 40.9 | raw-grad-empty-mpi2.log |
| grad-frozen | 1 | nan | PASS | none | 0 | 41.3 | raw-grad-frozen-mpi2-fault1.log |
| grad-recover | 1 | nan | PASS | none | 0 | 41.9 | raw-grad-recover-mpi2-fault1.log |
| grad-healthy | -1 | nan | PASS | none | 0 | 46.2 | raw-grad-healthy-mpi2.log |

## Per-case commands and reasons

### grad-train-one fault_rank=0 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-train-one --backend mpi --ranks 2 --campaign 607363-3136002479023206 --fault-rank 0 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-train-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-train-one --backend mpi --ranks 2 --campaign 607363-3136002479023206 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-audit-zero fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-audit-zero --backend mpi --ranks 2 --campaign 607363-3136002479023206 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-late fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-late --backend mpi --ranks 2 --campaign 607363-3136002479023206 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-inf fault_rank=1 fault_value=inf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-inf --backend mpi --ranks 2 --campaign 607363-3136002479023206 --fault-rank 1 --fault-value inf`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-ninf fault_rank=1 fault_value=ninf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-ninf --backend mpi --ranks 2 --campaign 607363-3136002479023206 --fault-rank 1 --fault-value ninf`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-update-both fault_rank=0 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-update-both --backend mpi --ranks 2 --campaign 607363-3136002479023206 --fault-rank 0 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-overflow fault_rank=-1 fault_value=inf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-overflow --backend mpi --ranks 2 --campaign 607363-3136002479023206`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-tied-overflow fault_rank=-1 fault_value=inf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-tied-overflow --backend mpi --ranks 2 --campaign 607363-3136002479023206`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-asym-vote fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-asym-vote --backend mpi --ranks 2 --campaign 607363-3136002479023206`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-caller fault_rank=0 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-caller --backend mpi --ranks 2 --campaign 607363-3136002479023206 --fault-rank 0 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-conditional-absence fault_rank=0 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-conditional-absence --backend mpi --ranks 2 --campaign 607363-3136002479023206 --fault-rank 0 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-absent-history fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-absent-history --backend mpi --ranks 2 --campaign 607363-3136002479023206`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-all-absent fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-all-absent --backend mpi --ranks 2 --campaign 607363-3136002479023206`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-no-trainable fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-no-trainable --backend mpi --ranks 2 --campaign 607363-3136002479023206`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-empty fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-empty --backend mpi --ranks 2 --campaign 607363-3136002479023206`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-frozen fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-frozen --backend mpi --ranks 2 --campaign 607363-3136002479023206 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-recover fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-recover --backend mpi --ranks 2 --campaign 607363-3136002479023206 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-healthy fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-healthy --backend mpi --ranks 2 --campaign 607363-3136002479023206`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

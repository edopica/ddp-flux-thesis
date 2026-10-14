# DB-1/DB-4/DB-3 numerical-guard check metadata

- date: 2026-10-14 17:28:30
- phase: green
- group: gradient (loss = DB-1/DB-4 loss vote; gradient = DB-3 gradient vote; all = loss cases followed by gradient cases in one campaign)
- backend: mpi
- ranks: 4
- outdir: /home/picazio/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-gradient-mpi4
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
| grad-train-one | 2 | nan | PASS | none | 0 | 47.2 | raw-grad-train-one-mpi4-fault2.log |
| grad-late | 3 | nan | PASS | none | 0 | 51.0 | raw-grad-late-mpi4-fault3.log |
| grad-inf | 3 | inf | PASS | none | 0 | 47.2 | raw-grad-inf-mpi4-fault3.log |
| grad-asym-vote | -1 | nan | PASS | none | 0 | 46.4 | raw-grad-asym-vote-mpi4.log |
| grad-conditional-absence | 2 | nan | PASS | none | 0 | 43.9 | raw-grad-conditional-absence-mpi4-fault2.log |
| grad-recover | 3 | nan | PASS | none | 0 | 47.4 | raw-grad-recover-mpi4-fault3.log |
| grad-healthy | -1 | nan | PASS | none | 0 | 52.4 | raw-grad-healthy-mpi4.log |
| grad-tied-overflow | -1 | inf | PASS | none | 0 | 47.7 | raw-grad-tied-overflow-mpi4.log |

## Per-case commands and reasons

### grad-train-one fault_rank=2 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-train-one --backend mpi --ranks 4 --campaign 613885-3137957661040208 --fault-rank 2 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-late fault_rank=3 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-late --backend mpi --ranks 4 --campaign 613885-3137957661040208 --fault-rank 3 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-inf fault_rank=3 fault_value=inf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-inf --backend mpi --ranks 4 --campaign 613885-3137957661040208 --fault-rank 3 --fault-value inf`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-asym-vote fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-asym-vote --backend mpi --ranks 4 --campaign 613885-3137957661040208`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-conditional-absence fault_rank=2 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-conditional-absence --backend mpi --ranks 4 --campaign 613885-3137957661040208 --fault-rank 2 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-recover fault_rank=3 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-recover --backend mpi --ranks 4 --campaign 613885-3137957661040208 --fault-rank 3 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-healthy fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-healthy --backend mpi --ranks 4 --campaign 613885-3137957661040208`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-tied-overflow fault_rank=-1 fault_value=inf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 4 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-tied-overflow --backend mpi --ranks 4 --campaign 613885-3137957661040208`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

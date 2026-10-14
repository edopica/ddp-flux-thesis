# DB-1/DB-4/DB-3 numerical-guard check metadata

- date: 2026-10-14 17:17:55
- phase: green
- group: loss (loss = DB-1/DB-4 loss vote; gradient = DB-3 gradient vote; all = loss cases followed by gradient cases in one campaign)
- backend: mpi
- ranks: 2
- outdir: /home/picazio/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-loss-db3-mpi2
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
| train-one | 0 | nan | PASS | none | 0 | 28.2 | raw-train-one-mpi2-fault0.log |
| train-one | 1 | nan | PASS | none | 0 | 28.2 | raw-train-one-mpi2-fault1.log |
| train-one-inf | 1 | inf | PASS | none | 0 | 28.2 | raw-train-one-inf-mpi2-fault1.log |
| train-one-ninf | 1 | ninf | PASS | none | 0 | 28.0 | raw-train-one-ninf-mpi2-fault1.log |
| loop-one | 1 | nan | PASS | none | 0 | 28.2 | raw-loop-one-mpi2-fault1.log |
| loop-withgradient-one | 1 | nan | PASS | none | 0 | 28.2 | raw-loop-withgradient-one-mpi2-fault1.log |
| train-all | -1 | nan | PASS | none | 0 | 28.6 | raw-train-all-mpi2.log |
| aux-nonfinite-ok | -1 | nan | PASS | none | 0 | 23.0 | raw-aux-nonfinite-ok-mpi2.log |
| aux-nan-first | 1 | nan | PASS | none | 0 | 23.6 | raw-aux-nan-first-mpi2-fault1.log |
| no-trainable | 1 | nan | PASS | none | 0 | 19.3 | raw-no-trainable-mpi2-fault1.log |
| recover-one | 1 | nan | PASS | none | 0 | 28.6 | raw-recover-one-mpi2-fault1.log |
| healthy | -1 | nan | PASS | none | 0 | 36.1 | raw-healthy-mpi2.log |
| dup-train-one | 1 | nan | PASS | none | 0 | 57.0 | raw-dup-train-one-mpi2-fault1.log |

## Per-case commands and reasons

### train-one fault_rank=0 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one --backend mpi --ranks 2 --campaign 611493-3137319909748722 --fault-rank 0 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### train-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one --backend mpi --ranks 2 --campaign 611493-3137319909748722 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### train-one-inf fault_rank=1 fault_value=inf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one-inf --backend mpi --ranks 2 --campaign 611493-3137319909748722 --fault-rank 1 --fault-value inf`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### train-one-ninf fault_rank=1 fault_value=ninf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one-ninf --backend mpi --ranks 2 --campaign 611493-3137319909748722 --fault-rank 1 --fault-value ninf`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### loop-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case loop-one --backend mpi --ranks 2 --campaign 611493-3137319909748722 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### loop-withgradient-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case loop-withgradient-one --backend mpi --ranks 2 --campaign 611493-3137319909748722 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### train-all fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-all --backend mpi --ranks 2 --campaign 611493-3137319909748722`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### aux-nonfinite-ok fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case aux-nonfinite-ok --backend mpi --ranks 2 --campaign 611493-3137319909748722`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### aux-nan-first fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case aux-nan-first --backend mpi --ranks 2 --campaign 611493-3137319909748722 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### no-trainable fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case no-trainable --backend mpi --ranks 2 --campaign 611493-3137319909748722 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### recover-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case recover-one --backend mpi --ranks 2 --campaign 611493-3137319909748722 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### healthy fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case healthy --backend mpi --ranks 2 --campaign 611493-3137319909748722`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### dup-train-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case dup-train-one --backend mpi --ranks 2 --campaign 611493-3137319909748722 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

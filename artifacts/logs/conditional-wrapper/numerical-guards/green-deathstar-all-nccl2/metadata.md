# DB-1/DB-4/DB-3 numerical-guard check metadata

- date: 2026-10-14 19:22:46
- phase: green
- group: all (loss = DB-1/DB-4 loss vote; gradient = DB-3 gradient vote; all = loss cases followed by gradient cases in one campaign)
- backend: nccl
- ranks: 2
- outdir: /home/picazio/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/numerical-guards/green-deathstar-all-nccl2
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
| train-one | 0 | nan | PASS | none | 0 | 53.6 | raw-train-one-nccl2-fault0.log |
| train-one | 1 | nan | PASS | none | 0 | 53.4 | raw-train-one-nccl2-fault1.log |
| train-one-inf | 1 | inf | PASS | none | 0 | 53.4 | raw-train-one-inf-nccl2-fault1.log |
| train-one-ninf | 1 | ninf | PASS | none | 0 | 53.2 | raw-train-one-ninf-nccl2-fault1.log |
| loop-one | 1 | nan | PASS | none | 0 | 53.0 | raw-loop-one-nccl2-fault1.log |
| loop-withgradient-one | 1 | nan | PASS | none | 0 | 53.0 | raw-loop-withgradient-one-nccl2-fault1.log |
| train-all | -1 | nan | PASS | none | 0 | 53.2 | raw-train-all-nccl2.log |
| aux-nonfinite-ok | -1 | nan | PASS | none | 0 | 48.0 | raw-aux-nonfinite-ok-nccl2.log |
| aux-nan-first | 1 | nan | PASS | none | 0 | 48.3 | raw-aux-nan-first-nccl2-fault1.log |
| no-trainable | 1 | nan | PASS | none | 0 | 40.9 | raw-no-trainable-nccl2-fault1.log |
| recover-one | 1 | nan | PASS | none | 0 | 52.6 | raw-recover-one-nccl2-fault1.log |
| healthy | -1 | nan | PASS | none | 0 | 61.9 | raw-healthy-nccl2.log |
| dup-train-one | 1 | nan | UNTESTED | none | 0 | 48.0 | raw-dup-train-one-nccl2-fault1.log |
| grad-train-one | 0 | nan | PASS | none | 0 | 68.3 | raw-grad-train-one-nccl2-fault0.log |
| grad-train-one | 1 | nan | PASS | none | 0 | 68.5 | raw-grad-train-one-nccl2-fault1.log |
| grad-audit-zero | 1 | nan | PASS | none | 0 | 68.9 | raw-grad-audit-zero-nccl2-fault1.log |
| grad-late | 1 | nan | PASS | none | 0 | 71.9 | raw-grad-late-nccl2-fault1.log |
| grad-inf | 1 | inf | PASS | none | 0 | 68.5 | raw-grad-inf-nccl2-fault1.log |
| grad-ninf | 1 | ninf | PASS | none | 0 | 68.9 | raw-grad-ninf-nccl2-fault1.log |
| grad-update-both | 0 | nan | PASS | none | 0 | 69.5 | raw-grad-update-both-nccl2-fault0.log |
| grad-overflow | -1 | inf | UNTESTED | none | 0 | 67.9 | raw-grad-overflow-nccl2.log |
| grad-tied-overflow | -1 | inf | UNTESTED | none | 0 | 67.3 | raw-grad-tied-overflow-nccl2.log |
| grad-asym-vote | -1 | nan | PASS | none | 0 | 68.7 | raw-grad-asym-vote-nccl2.log |
| grad-caller | 0 | nan | PASS | none | 0 | 68.9 | raw-grad-caller-nccl2-fault0.log |
| grad-conditional-absence | 0 | nan | PASS | none | 0 | 68.5 | raw-grad-conditional-absence-nccl2-fault0.log |
| grad-absent-history | -1 | nan | PASS | none | 0 | 73.1 | raw-grad-absent-history-nccl2.log |
| grad-all-absent | -1 | nan | PASS | none | 0 | 69.1 | raw-grad-all-absent-nccl2.log |
| grad-no-trainable | -1 | nan | PASS | none | 0 | 67.9 | raw-grad-no-trainable-nccl2.log |
| grad-empty | -1 | nan | PASS | none | 0 | 67.3 | raw-grad-empty-nccl2.log |
| grad-frozen | 1 | nan | PASS | none | 0 | 68.9 | raw-grad-frozen-nccl2-fault1.log |
| grad-recover | 1 | nan | PASS | none | 0 | 68.9 | raw-grad-recover-nccl2-fault1.log |
| grad-healthy | -1 | nan | PASS | none | 0 | 74.2 | raw-grad-healthy-nccl2.log |

## Per-case commands and reasons

### train-one fault_rank=0 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 0 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### train-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### train-one-inf fault_rank=1 fault_value=inf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one-inf --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value inf`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### train-one-ninf fault_rank=1 fault_value=ninf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-one-ninf --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value ninf`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### loop-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case loop-one --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### loop-withgradient-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case loop-withgradient-one --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### train-all fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case train-all --backend nccl --ranks 2 --campaign 629190-3143208366662451`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### aux-nonfinite-ok fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case aux-nonfinite-ok --backend nccl --ranks 2 --campaign 629190-3143208366662451`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### aux-nan-first fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case aux-nan-first --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### no-trainable fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case no-trainable --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### recover-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case recover-one --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### healthy fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case healthy --backend nccl --ranks 2 --campaign 629190-3143208366662451`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### dup-train-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case dup-train-one --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: UNTESTED
- reason: none

### grad-train-one fault_rank=0 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-train-one --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 0 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-train-one fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-train-one --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-audit-zero fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-audit-zero --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-late fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-late --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-inf fault_rank=1 fault_value=inf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-inf --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value inf`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-ninf fault_rank=1 fault_value=ninf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-ninf --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value ninf`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-update-both fault_rank=0 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-update-both --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 0 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-overflow fault_rank=-1 fault_value=inf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-overflow --backend nccl --ranks 2 --campaign 629190-3143208366662451`
- timeout: none
- exit status: 0
- verdict: UNTESTED
- reason: none

### grad-tied-overflow fault_rank=-1 fault_value=inf

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-tied-overflow --backend nccl --ranks 2 --campaign 629190-3143208366662451`
- timeout: none
- exit status: 0
- verdict: UNTESTED
- reason: none

### grad-asym-vote fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-asym-vote --backend nccl --ranks 2 --campaign 629190-3143208366662451`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-caller fault_rank=0 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-caller --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 0 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-conditional-absence fault_rank=0 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-conditional-absence --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 0 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-absent-history fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-absent-history --backend nccl --ranks 2 --campaign 629190-3143208366662451`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-all-absent fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-all-absent --backend nccl --ranks 2 --campaign 629190-3143208366662451`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-no-trainable fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-no-trainable --backend nccl --ranks 2 --campaign 629190-3143208366662451`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-empty fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-empty --backend nccl --ranks 2 --campaign 629190-3143208366662451`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-frozen fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-frozen --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-recover fault_rank=1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-recover --backend nccl --ranks 2 --campaign 629190-3143208366662451 --fault-rank 1 --fault-value nan`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

### grad-healthy fault_rank=-1 fault_value=nan

- command: `/home/picazio/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/bin/mpiexec -n 2 /home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia -C native -J/home/picazio/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/sys.so -g1 --startup-file=no --project=/home/picazio/projects/flux-integration/test --startup-file=no --color=no /home/picazio/projects/ddp-flux-thesis/scripts/checks/conditional_graph_numerical_guard_probe.jl --case grad-healthy --backend nccl --ranks 2 --campaign 629190-3143208366662451`
- timeout: none
- exit status: 0
- verdict: PASS
- reason: none

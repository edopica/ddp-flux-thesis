# ADR-0004 - Scripts reorganization

Date: 2026-07-12
Status: Accepted

## Context

By the end of checkpoint C3, the `scripts/` directory had accumulated 15 files in a flat structure. This mixed concerns like Julia experiments, Julia modules (libraries), shell infrastructure, remote deployment scripts, and environment setups without clear separation. Additionally, filenames had become redundant or inconsistent (e.g. `save_reference.jl`, `verify_reference.jl`, `sync_model.jl`, `verify_sync.jl`). There was also a stray `update_wiki.sh` script at the repository root.

## Decision

We reorganized the `scripts/` directory by functional area and workflow stage, rather than keeping a flat structure. We also renamed files to be more concise since the parent directories now provide the context. 

The new layout is:
- `scripts/setup/`: C0 environment & auditing scripts (`check_env.jl`, `setup_gpu.jl`, `audit_distributed.sh`)
- `scripts/checks/`: C0 MPI smoke tests & health checks (`smoke_mpi.jl`, `health_check.jl`)
- `scripts/remote/`: Infrastructure and remote node management (`sync_code.sh`, `setup_node.sh`, `run.sh`, `update_wiki.sh`, and `hosts/`)
- `scripts/reference/`: C1 deterministic baseline (`ReferenceLoop.jl`, `save.jl`, `verify.jl`)
- `scripts/launch/`: C2 distributed launch skeleton (`skeleton.jl`)
- `scripts/sync/`: C3 model broadcast & verification (`broadcast.jl`, `verify.jl`)

## Consequences

- Improved navigability and cleaner filenames.
- Simpler addition of new checkpoints (C4-C12) as they will naturally fall into new subdirectories.
- Git history remains intact via `git mv`.
- Makefile paths and OpenWiki documentation required updates to reflect the new paths.

# AGENTS.md

## Project goal

Stabilize, test, document, and evaluate Distributed Data Parallel training support in Flux.jl.

## Source of truth before every session

Read these files before acting:

1. `README.md`
2. `wiki/context/current.md`
3. Latest file in `wiki/devlog/`
4. Relevant files in `wiki/decisions/`
5. Relevant files in `wiki/audits/`

## Current checkpoint

C0 - Repository and API audit.

## Current priority order

1. Understand DDP concepts, NCCL, and MPI.
2. Understand current Flux distributed implementation.
3. Verify CPU/MPI behavior.
4. Move to GPU/NCCL support only after CPU smoke tests and API audit are clear.

## Non-negotiable rules

- Do not optimize before correctness tests pass.
- All modifications to the `Flux.jl` upstream repository must be strictly motivated by a specific failure mode and fully documented (e.g., via ADRs or Notebooks) explaining the root cause and the fix.
- Keep DDP communication outside automatic differentiation for the baseline.
- Do not implement a custom optimizer for the main training path.
- Use Optimisers.jl for optimizer setup and updates.
- Use Flux/Functors-compatible traversal for nested models.
- Avoid silent GPU-to-CPU transfers.
- Enable CUDA scalar-indexing checks early in GPU work.
- Every collective must be called by every rank in the same order.
- Record exact commands used to reproduce results.
- Rank 0 should be the only rank writing shared output files unless explicitly designed otherwise.
- Do not fabricate results or mark tasks complete without evidence.
- Scale gradients explicitly and document the convention.
- Do not treat “loss decreases” as proof of correctness.
- Profile only after correctness tests pass.

## HPC execution rules (hpc / deathstar)

- Load the `remote` skill before running anything on `hpc` or `deathstar`.
- On `hpc`, precompile ONLY on compute nodes: run `scripts/remote/precompile.sh hpc` after every `sync_code.sh` or Flux source change, and BEFORE running tests. Login-node julia compiles target the wrong CPU (`graniterapids` vs `icelake-server`); a skipped precompile makes the first test run silently recompile for ~2-5 min inside the test watchdog ("timeout on precomp").
- Never run julia that loads the project envs (`--project=...` with `using Flux`) on the HPC login node.
- Pass make variables as make args on `hpc` (`make c8-mpi FLUX_REPO_PATH=../ddp_flux`), not as env prefixes before the command (salloc execs argv directly). Resource overrides go before the *script*: `NTASKS=2 CPUS_PER_TASK=4 scripts/remote/run.sh hpc ...`.
- If a run prints `Precompiling packages...`, a sync happened without a follow-up precompile: stop, precompile, rerun.
- Evidence and rationale: `wiki/decisions/ADR-0005-precompile-compute-only.md`, `wiki/context/remote-nodes.md`, `wiki/devlog/2026-09-03-precompile-experiment.md`.

## Before ending a session

- Update `wiki/context/current.md`.
- Add or update today’s devlog entry.
- Update the README progress table if the status changed.
- Record failures in `wiki/failures/` if any command failed.
- Print the exact next action.

## Wiki

This repository has documentation located in the /wiki directory.

Start here:
- [Wiki quickstart](wiki/quickstart.md)

Wiki includes repository overview, architecture notes, workflows, domain concepts, operations, integrations, testing guidance, and source maps.

When working in this repository, read the Wiki quickstart first, then follow its links to the relevant architecture, workflow, domain, operation, and testing notes.

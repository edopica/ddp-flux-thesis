# AGENTS.md

## Project goal

Stabilize, test, document, and evaluate Distributed Data Parallel training support in Flux.jl.

## Source of truth before every session

Read these files before acting:

1. `README.md`
2. `openwiki/context/current.md`
3. Latest file in `openwiki/devlog/`
4. Relevant files in `openwiki/decisions/`
5. Relevant files in `openwiki/audits/`

## Current checkpoint

C0 - Repository and API audit.

## Current priority order

1. Understand DDP concepts, NCCL, and MPI.
2. Understand current Flux distributed implementation.
3. Verify CPU/MPI behavior.
4. Move to GPU/NCCL support only after CPU smoke tests and API audit are clear.

## Non-negotiable rules

- Do not optimize before correctness tests pass.
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

## Before ending a session

- Update `openwiki/context/current.md`.
- Add or update today’s devlog entry.
- Update the README progress table if the status changed.
- Record failures in `openwiki/failures/` if any command failed.
- Print the exact next action.

## OpenWiki

This repository has documentation located in the /openwiki directory.

Start here:
- [OpenWiki quickstart](openwiki/quickstart.md)

OpenWiki includes repository overview, architecture notes, workflows, domain concepts, operations, integrations, testing guidance, and source maps.

When working in this repository, read the OpenWiki quickstart first, then follow its links to the relevant architecture, workflow, domain, operation, and testing notes.

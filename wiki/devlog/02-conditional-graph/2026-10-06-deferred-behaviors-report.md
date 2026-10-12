# Conditional-graph deferred-behavior report

Date: 2026-10-06

The user requested the deferred-behavior report for the conditional-graph
wrapper. This was the only remaining Phase K item
(`temp/conditional_graph_deferred_behaviors_prompt.md`).

## Scope and source state

Investigation of behavioral limitations of `DistributedOptimizerState`,
without production fixes, metadata optimization, or a new Optimisers API.
Source: `flux-integration` at `55171e65` plus uncommitted Phase H/J changes
(`src/distributed/public_api.jl` blob `67e52161...`, `src/train.jl` blob
`05d57e90...`). Runtime: Julia 1.12.6, Flux 0.16.11, MPI 0.20.27,
Optimisers 0.4.9, 2 CPU/MPI ranks.

## Work performed

- Read the prompt, plan sections I-K/21-23, audit 001, the Phase H reports and
  tests, ADR-0007, the current `train.jl`/`public_api.jl`, and the Optimisers
  0.4.9 source (setup/update/`Leaf` semantics).
- Traced the loss guard (`src/train.jl:240-245,375-385`), the presence and
  gradient collective protocol (`public_api.jl:698-737`), and the frozen/rules
  contract (`public_api.jl:244-269,609,739-750`).
- Classified the recorded Phase J GPU failures and deferred GPU cases from
  `phaseJ-metadata.md` and the failure records; no new GPU run was performed.
- Wrote and ran a new bounded 2-rank CPU/MPI probe and watchdog driver:
  `scripts/checks/conditional_graph_deferred_probe.jl` and
  `scripts/checks/conditional_graph_deferred_check.jl`. The driver saves raw
  logs and asserts marker signatures; hang modes require an unmatched
  `COLLECTIVE_BEGIN` plus the diverging rank's stop/hold marker, so a timeout
  alone is never accepted.

## Results

Final run: **8/8 modes PASS, 83 assertions, 0 failures, driver exit 0**.
Raw logs and summary: `artifacts/logs/conditional-wrapper/deferred/`.
Metadata (hashes, versions, commands): `deferred-metadata.md` there.

- `control`: wrapper matches native Adam exactly (`maxdiff=0.0`), 12/12
  collectives per rank.
- `nan-loss-hang`: rank 0 blocked inside presence collective #7; rank 1 threw
  `DomainError` and held. Runtime-confirmed deadlock window (DB-1).
- `nan-loss-all`: both ranks stopped at item 3; state matches the two-step
  shadow (`maxdiff=0.0`, `state_ok=true`). Benign boundary (DB-2).
- `grad-nonfinite`: finite loss, NaN `w` gradient on rank 1; both ranks left
  with `w=NaN` and `mt=NaN`, no exception (`exc=none`). Silent corruption
  (DB-3).
- `step-desync`: rank 1 matched the shifted-pairing reference exactly
  (`delta_vs_shifted=0.0`) and differed from the correct reference
  (`delta_vs_correct=0.0499`), then rank 0 blocked at collective #13.
  Silent trajectory corruption then hang (DB-4).
- `early-stop`: rank 0 blocked at presence collective #10; rank 1 stopped at
  step 4 (DB-5).
- `config-eta`: each rank matched its own rule reference (`self_maxdiff=0.0`),
  replicas diverged by `cross_delta=0.1499` with no error or hang (DB-6).
- `config-frozen`: rank 1 unchanged, rank 0 moved, `cross_delta=0.2001`,
  12/12 collectives (frozen leaves still reduce) (DB-7).

Accepted-contract boundaries (rules/frozen/backend excluded from
`synchronize!!`, data-only checkpoints, root-only restore unsupported),
source-only unsupported workflows (post-setup model changes, broken ties,
custom state), the GPU-untested list, and the aux-value guard scope are
documented in the report.

## Probe tooling defects (test-only)

The first full driver run exited 1 because of four defects in the new tooling:
marker slicing at the last `PROBE_READY` dropped one rank's late markers; two
assertion regexes had a literal prefix; `stdout=path` and `stderr=path` were
opened separately, so a killed run's stderr overwrote its stdout in the log;
and the `config-eta` shadow applied the local instead of the global-mean
gradient. All four were fixed and re-run; the final run is 0-failure. No Flux
source, test, or witness value changed. Recorded in
`wiki/failures/2026-10-06-deferred-probe-driver-defects.md`.

## Deliverables and updates

- `wiki/audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md` (report).
- Probes and driver, metadata and raw logs (paths above).
- `wiki/context/current.md`, README progress wording, implementation-plan
  section 21 follow-up backlog (A1-A8).
- No commit, push, Flux change, or remote execution.

## Exact next action

Review the deferred-behavior report and decide which findings the
demonstration carries (section 6 conditions; section 7 backlog). No other
Phase K item remains.

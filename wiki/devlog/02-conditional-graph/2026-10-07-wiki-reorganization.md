# Wiki reorganization: devlog and audits grouped by project scope

Date: 2026-10-07
Scope: repository-wide documentation maintenance (kept in scope 2 because it
records work done during the conditional-graph project phase).
Flux worktrees: untouched (`flux-integration` HEAD `55171e65` dirty,
`ddp_flux` dirty master frozen, `flux-pr2694-salvage` on `ddp/windows-path-fix`
at `fee2c310`).
Project source: no Julia, MPI, CUDA, notebook, or Makefile change.

## Outcome

The flat `wiki/devlog/` folder (76 entries) and `wiki/audits/` folder (3
reports) are grouped by project scope:

- `wiki/devlog/01-pr2694/` — 36 entries: C0–C8 upstream DDP work, M0/M1
  integration spike, PS1–PS8 PR #2694 salvage, Windows routing fix, and the
  remote/documentation tooling of that era.
- `wiki/devlog/02-conditional-graph/` — 40 moved entries plus this one:
  conditional-graph onboarding, `DistributedOptimizerState` wrapper Phases
  A–K, native CUDA/NCCL validation, Leonardo bring-up, timing stretch,
  notebooks, and reports.
- `wiki/audits/01-pr2694/` — audit 000 (C0).
- `wiki/audits/02-conditional-graph/` — audits 001 (Phase H compatibility) and
  002 (deferred behaviors).

New indexes: `wiki/devlog/README.md` (scope table plus a complete
newest-first list per scope) and `wiki/audits/README.md`.

## Method

1. Two read-only subagents: one classified every devlog/audit entry by scope
   from file content; one audited the wiki against the codebase and the three
   Flux worktrees (paths, `make` targets, HEADs, status tables).
2. All 79 files moved with `git mv` (tracked files) or `mv` (untracked files);
   no file was renamed.
3. A script rewrote every relative markdown link by resolving it against the
   pre-move location and recomputing the path from the new location, and
   rewrote repo-root-relative `wiki/devlog/...` / `wiki/audits/...` text path
   references in wiki docs, `README.md`, `AGENTS.md`, `reports/`, and
   `scripts/podman/README.md`. Historical evidence under `artifacts/` and
   `temp/` was deliberately left untouched.
4. A link inventory over every markdown file in the repo reports zero broken
   links to moved files; the only remaining broken target is the pre-existing
   `temp/pr_salvage_plan.md -> (...)` placeholder.

## Inconsistencies corrected in the same pass

- `wiki/quickstart.md`: stale next action (`conditional_graph_gpu_recap.ipynb`
  was replaced by the benchmark notebook); Leonardo described as "not
  operational"; ADR index missing 0005–0007; audit index missing 002; devlog
  and failures indexes replaced with the new scope indexes; `../Flux.jl` path
  corrected to `../ddp_flux`; broken `make verify-conditional` removed from the
  reproducible list.
- `wiki/context/remote-nodes.md`: Leonardo row updated (Julia workflow up).
- `wiki/context/current.md`: stale "Active checkpoint" block, resolved open
  questions (CUDA/NCCL resolve, deathstar GPU, NCCL runtime validation,
  `experiment/precompile` branch), and the bottom "Next exact action" updated.
- `README.md`: `CW-E-review` status set to done (review-corrected); PS7 note
  annotated (PR head superseded by PS8 `fee2c310`); PS8 status no longer claims
  "CI running"; C8 marked frozen; guardrail wording narrowed to the implemented
  MPICH/PMIx check; `../Flux.jl` path corrected.
- ADR-0001, ADR-0003, ADR-0004: added short "superseded by" notes for the
  renamed fork path, the narrowed launcher guard, and the `update_wiki.sh` move.

## Verification

- `python3` link inventory over all `*.md`: zero broken links introduced.
- `git status`: all moves visible as renames for tracked files; no content
  file deleted.
- The scope classification was independently performed by a subagent and
  checked against a filename-rule split; the two agree.
- `README.md`, `wiki/quickstart.md`, `wiki/context/current.md`, and
  `wiki/devlog/README.md` cross-check against the current worktree state
  reported by a second subagent.

## Exact next action

Unchanged: implement `notebook/conditional_graph_benchmark.ipynb` from the
corrected `artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/`
artifacts (two-notebook plan implementation item 6).

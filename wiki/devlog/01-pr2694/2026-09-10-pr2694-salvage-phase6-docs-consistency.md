# 2026-09-10 — PR #2694 salvage phase 6: documentation & consistency sweep

Follows phases 1-2 (`...-phases-1-2.md`), phase 3a (`...-phase3a-launcher-guard-tests.md`), phase 3b (`...-phase3b-pmi-guardrail.md`), phase 4 (`...-phase4-cyclic-data-padding.md`), and phase 5 (`...-phase5-polish-carlo-tests.md`). Phase 6 of `temp/pr_salvage_plan.md` (lines 251-267) is now complete. Delegated to three implementation subagents (one for `gpu.md`, one for `NEWS.md`, one for the typo + consistency sweep), then an INDEPENDENT verification subagent audited the result and returned APPROVE; the orchestrator reviewed the end state. No subagent failed.

The edits are **uncommitted working-tree changes** — the user did not explicitly request a commit, so no commit was made.

## What was done

Worktree `/home/kurapica/Projects/ddp_flux/flux-pr2694-salvage`, branch `ddp/pr2694-salvage`, on top of commit `23cae338` (phase 5):

- Exactly 4 files changed, `+15/-11`, uncommitted. No behavioral change (docs/comment/typo/newline only).

### `docs/src/guide/gpu.md` (3 line edits)

- Both launcher occurrences corrected from `mpiexecjl --project=. -n <np> julia <filename>.jl` to `mpiexecjl -n <np> julia --project=. <filename>.jl` (lines ~317 and ~397).
- Line ~349: "split the data uniformly" replaced with equal-length/cyclic-duplication wording — every process gets an equal-length shard; when `N` is not divisible by workers the container pads its index sequence by repeating observations cyclically so each process receives `cld(N, workers)` observations while every index stays in `1:N`; the underlying dataset is not modified; duplicated observations affect epoch accounting and metrics (must be excluded/accounted for); an empty dataset is rejected.
- No unused-gradient instructions and no native-NCCL-testing claims were introduced; the "Known shortcomings" CUDA-aware warning was left untouched.

### `NEWS.md`

- The `### Distributed training fixes` subsection now has **exactly two** bullets, each ending with `([#2694](https://github.com/FluxML/Flux.jl/pull/2694))`:
  1. **Data sharding** — equal-length cyclic sharding + empty-data `ArgumentError`.
  2. PMI guardrail — known MPICH/PMIx launcher mismatch diagnostic + `force=true`.
- The old `**Distributed test fixes**` test-renaming/CI bullet and the vague `**Data padding**` bullet were removed.
- No other NEWS section changed.

### `src/distributed/public_api.jl:342`

- Docstring typo `AbstractFluxDistributedBacked` → `AbstractFluxDistributedBackend`.

### `ext/FluxMPIExt/FluxMPIExt.jl`

- Restored the missing final newline (no semantic change).

## Method (delegated implement → verify)

- Three implementation subagents applied the edits in the `flux-pr2694-salvage` worktree (no commit): one handled `gpu.md`, one handled `NEWS.md`, one handled the `AbstractFluxDistributedBacked` typo plus the consistency sweep.
- An INDEPENDENT verification subagent reproduced the checks against Phase 6 of `temp/pr_salvage_plan.md` (lines 251-267) and returned APPROVE.
- No RED/GREEN cycle was used for phase 6 because it is a documentation/consistency sweep with no semantic change — that decision was the orchestrator's.

## Checks that passed (do not claim more)

- `git diff --check` clean (exit 0).
- Old launcher order absent.
- "split uniformly" absent.
- Both `#2694` links present in `NEWS.md`.
- `docs/src/guide/distributed.md` absent and `docs/make.jl` has no registration for it.
- No stale terms remain (`resolve_unused_parameters`, `unused_parameters`, `DistributedOptimizerState`, `conditional-gradient`, `AbstractFluxDistributedBacked`, `split uniformly`).
- No child distributed test reads `ARGS`.
- Only the 4 expected files are modified; no untracked files.

**Correction (phase 7, 2026-09-10):** the earlier claim that "all changed files end with exactly one newline" was inaccurate. The phase-6 verifier checked only the 4 edited files; 3 already-committed test files (`common_distributedtest.jl`, `optimizer_distributedtest.jl`, `synchronized_distributedtest.jl`) lacked final newlines, and `runtests.jl` had 5 trailing-whitespace lines. This was found by the phase-7 gate-1 static scope check and fixed in commit `caf47145` (8 files, `+23/-19`). See `wiki/devlog/01-pr2694/2026-09-10-pr2694-salvage-phase7-verification.md`.

The full MPI suite and the docs build were NOT run in phase 6; both were run in phase 7 (gates 3 and 5, both PASS).

## Scope boundaries honored

- Narrow edits only: `gpu.md`, `NEWS.md`, one docstring typo, one final newline. Exactly 4 files, `+15/-11`.
- No unused-gradient instructions, no `DistributedOptimizerState` content, no conditional-gradient API, no re-added distributed guide, no native-NCCL-testing claims.
- The "Known shortcomings" CUDA-aware warning in `gpu.md` untouched.
- Changes deliberately left uncommitted (no explicit user request to commit).

## State after session

`flux-pr2694-salvage` @ `23cae338` [`ddp/pr2694-salvage`], **4 modified files uncommitted** (working tree NOT clean). The changes still need a commit decision.

## Next exact action

Phase 7 of `temp/pr_salvage_plan.md` (lines 268-290) — verification gates before pushing:

1. **Static scope checks** — `git diff --check`; inspect `404ff37d...HEAD`; verify no `resolve_unused_parameters!!` / `DistributedOptimizerState` / conditional-gradient API / added distributed guide / NCCL-validation claim remains; verify no child test reads `ARGS`.
2. **PMI guard tests** — focused compatibility-checker tests; default MPICH mismatch/bypass reproduction; system OpenMPI initialization reproduction.
3. **MPI suite** — `JULIA_MPI_TEST_NPROCS=2 julia --project=test test/ext_distributed/runtests.jl` and `JULIA_MPI_TEST_NPROCS=4 ...` full runs.
4. **Docker OpenMPI parity** — rerun the Ubuntu 24.04/OpenMPI reproduction under `--cpuset-cpus=0-3` with `mpiexec --use-hwthread-cpus`; confirm both two-rank and four-rank suites pass with visible child output.
5. **Documentation** — build the docs after the guide registration removal and `gpu.md` edits.
6. **Final review** — inspect status, complete diff, changed-file list, recent commits; confirm the live CI workflow exercises the five data sizes at both rank counts.

Also: decide whether to commit the phase-6 edits (currently uncommitted) before or as part of the Phase 7 gate run.

Note: `docs/src/guide/gpu.md:366-377` (pre-M1 Leaf output) is unrelated to this PR scope and remains deferred.

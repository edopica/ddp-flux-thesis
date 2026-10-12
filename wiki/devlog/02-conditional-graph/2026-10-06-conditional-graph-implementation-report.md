# Conditional graph implementation report

Date: 2026-10-06

## Focus

The user asked for a detailed report that documents how conditional graph
support was implemented. The report prepares the thesis author for a
supervisor progress meeting. The user required the simple-english skill,
explicit code references with snippets, and a short global-role introduction
before each technical component.

## Completed

- Wrote `reports/conditional-graph-implementation.md` (619 lines, 8 sections,
  2 appendices). The report covers:
  - the failure of the per-leaf allreduce at `fee2c310` (deadlock and silent
    corruption) and why zero-filling is not a solution;
  - the whole-model synchronization design and its six-step protocol;
  - every production component in `src/distributed/public_api.jl` with file
    paths, line numbers, and snippets: `DistributedOptimizerState`,
    `setup`/composition guard, `update`/`update!` boundary, `_sync_gradients`
    with its traversal, buffers, occurrence plan, presence protocol, and
    rebuild pass, plus `synchronize!!`;
  - the outside-AD evidence (`src/train.jl:375-379`, `@non_differentiable`);
  - the production and test-harness defects found in phases B to H;
  - the test and demonstration evidence, the file/commit map, exact
    reproduction commands, and the audit 002 limitations.
- Verified all facts against the implementation worktree and the wiki:
  `git diff --numstat fee2c310 -- src test NEWS.md`, `git log --oneline
  fee2c310..55171e65`, and the source at
  `../flux-integration/src/distributed/public_api.jl`.
- Applied the simple-english rules to the report. The self-check found and
  fixed 7 sentences with `would`/`could`, 5 prose semicolons, and 2 long list
  sentences. The final automated pass reports zero violations and no prose
  sentence above 25 words.

## Commands run

```bash
git -C ../flux-integration log --oneline fee2c310..55171e65
git -C ../flux-integration diff --numstat fee2c310 -- src test NEWS.md
python3 <style-check script>   # banned modals, semicolons, sentence length
```

No Julia, MPI, CUDA, or remote command ran. No Flux source or test changed.
No commit was created.

## Results

Report ready for the supervisor meeting. The report states the code footprint
as +384/-13 in `src/distributed/public_api.jl` (working tree), +1,598 lines in
`conditional_distributedtest.jl`, and the exact commit chain from `6065ce8f`
to `55171e65`.

## Failures

None.

## Decisions

- The report keeps one term per concept: "check", "configuration", "run",
  "show". Domain words stay unchanged.
- Code, identifiers, commands, and quoted errors are untouched by the style
  rules.
- The report separates the supported path from the deferred path (audit 002)
  so the supervisor sees the open risks.

## Next action

Unchanged: implement `notebook/conditional_graph_benchmark.ipynb` from the
corrected `timing/stretch-2026-10-06-leonardo/` artifacts (two-notebook plan,
implementation item 6).

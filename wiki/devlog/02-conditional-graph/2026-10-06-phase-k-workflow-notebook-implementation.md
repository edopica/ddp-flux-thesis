# Phase K distributed workflow notebook implementation

Date: 2026-10-06

## Request and outcome

The user asked to implement the distributed workflow notebook from
`temp/conditional-graph-phase-k-notebook-plan.md` (deliverable 1 of the
two-notebook plan). The notebook is implemented, executed twice from fresh
IJulia kernels in artifact mode, exported to HTML, and stored with an evidence
record.

Deliverables:

- `notebook/conditional_graph_distributed_workflow.ipynb` (53 cells, 21 code
  cells, saved four-rank outputs);
- `notebook/phase_k_workflow_helpers.jl` (artifact bundle map, provenance and
  summary tables, Python plotter calls, IJulia display, child-process driver);
- extended `scripts/plots/plot_phase_k_demo.py` (exposure, collectives, and
  continuation figures);
- pinned plotting environment `scripts/plots/requirements-phase-k.txt`
  (matplotlib 3.11.2, numpy 2.5.3; used by all three Phase K plotters);
- execution record and copies under
  `artifacts/logs/conditional-wrapper/notebook/2026-10-06/`.

The benchmark notebook (deliverable 2) is not implemented; it is the next
action.

## Artifact bundle map

The configuration cell selects a complete run bundle, not a rank label:

| Selection | Campaign root | Backend / ranks | Hardware |
|---|---|---|---|
| `tutorial-4rank` (default) | `demo/k2-2026-10-05-leonardo-nccl4` | NCCL / 4 | Leonardo, 4x A100-SXM-64GB, devices `[1,2,3,0]` |
| `tutorial-2rank` | `demo/k2-2026-10-05-deathstar-nccl2` | NCCL / 2 | deathstar, 2x TITAN RTX, devices `[1,2]` |

Both bundles read the shared prepared dataset provenance in
`demo/k2-2026-10-05-deathstar-mpi/prepared/`. `resolve_bundle` validates every
required artifact and raises one error listing all missing paths. The selection
can be overridden with `PHASE_K_TUTORIAL_CONFIG` without editing the notebook.

## Notebook contents

The nine teaching sections follow the plan: provenance; shared-trunk model and
routing; schedule and gradient weighting; backend initialization and the public
`DistributedOptimizer` API; local AD plus the wrapper update outside AD;
reference comparison and the collective protocol; learning curves, configured
learning rate, and predictions; data-only checkpoint continuation; reproduction
commands and demonstrated scope.

Design decisions:

- The kernel never loads Flux/MPI/CUDA and never joins a collective. API
  examples are markdown; an executed cell verifies the exact public API and
  driver anchor lines in the current `../flux-integration` worktree and prints
  them with line numbers.
- Every figure section has a question, a data-derived observation, and an
  interpretation. Bundle-specific numbers are printed from the parsed
  artifacts, so the two-rank execution shows 2-rank exposures, weights, and
  resume counts without editing the notebook; no caption hardcodes four-rank
  values.
- Plotting stays in Python: the Julia helper runs `plot_phase_k_demo.py` and
  `plot_phase_k_network.py` through `uv` with the pinned requirements and
  displays the PNGs through IJulia MIME display.
- The opt-in `run_training` path calls the bounded driver
  (`scripts/checks/conditional_graph_demo_check.jl`) as a child process. The
  kernel still owns no rank.

## Plotter extensions

`plot_phase_k_demo.py` gained three validated figures per pair, all derived
from the saved TSVs:

1. `-exposure.png`: per-step task weight (`counts / ranks`) and cumulative
   exposure; mixed-step weights are annotated.
2. `-collectives.png`: expected collectives per step (`1` presence reduction
   plus one gradient reduction per present leaf: 5 or 7).
3. `-continuation.png`: train-minus-resume DDP validation differences over the
   metric intersection `checkpoint+1..steps`, with exact-match counts, maximum
   differences, and an explicit "all zeros" annotation.

The helper independently recomputes exposure, collective counts, presence-flag
averages, schedule routing, and continuation intersections from the TSVs and
console logs; the notebook's Julia tables and the figure annotations must agree
(and do).

## Execution and acceptance

Fresh IJulia kernels via `jupyter nbconvert --execute`, artifact mode, no GPU:

- four-rank: exit 0; resume markers 472/472; exposures 6400/8960; mixed weights
  1/4 and 3/4; maxima param 5.960e-8, mom1 4.657e-9, mom2 1.455e-11; beta-exact
  false count 0; collective protocol 5/7 verified on every rank-step record.
- two-rank: exit 0; resume markers 236/236; exposures 7680/7680; mixed weights
  1/2 and 1/2; all discrepancy maxima exactly 0.0; beta-exact false count 0.
- HTML export: 12 embedded figures in order, nine sections, no error outputs.
- Missing-artifact path: a fabricated absent run root lists every missing
  required file; an unknown selection key lists the available keys.
- Provenance: Flux wrapper hashes match the current worktree; 10/11 recorded
  script hashes match. The single mismatch is
  `scripts/examples/conditional_graph/reference.jl`, changed after the campaign
  by the additive `reference_update!` helper from the corrected timing driver
  (`reports/conditional-graph-phase-k-timing-stretch.md`); the K2 path calls
  `global_gradient` and is unaffected. The notebook states this explicitly.
- Child-process training path verified directly (not from the notebook kernel):
  `run_training_child` on MPI CPU 2 ranks, exit 0 in 151.8 s, driver 6/6 PASS
  under `notebook/tutorial-2rank-20261006-1710769/`.

Exact commands and stored copies: `artifacts/logs/conditional-wrapper/notebook/2026-10-06/COMMANDS.md`.

## Defects found during implementation

All defects were in the new helper/notebook, not in campaign data or Flux.

- `REPO_ROOT` and `@sprintf` were missing from the notebook namespace on the
  first execution; exported the constant and imported Printf in the bootstrap
  cell.
- The console regex used `resume_match=(\S+)`. The driver sometimes emits two
  `DEMO_STEP` records on one line (`...resume_match=trueDEMO_STEP ...`); the
  greedy capture consumed the next record marker and silently dropped 108 of
  472 resume records. Fixed with `resume_match=(true|false|n/a)`, restoring
  364 -> 472 and 236.
- `local` cannot be used as a generator loop variable in the parser; renamed.
- `validation_summary` initially mixed tuple indices and table columns and
  `discrepancy_summary` had a malformed required-column list; both rewritten.
- `exposure_summary` first counted schedule rows instead of sample identities;
  the prepared schedule row holds a local batch of indices. Fixed by summing
  `pool_indices` lengths per task (6400/8960 and 7680/7680 now cross-check).
- `run_training_child` and the reproduction command passed the prepared
  *directory* instead of `mnist-fashion_prepared.jls`; caught by actually
  running the child path, fixed to the file path.
- The child-run timestamp used `HHmmss` (month + milliseconds in Julia); fixed
  to `HHMMSS`. The recorded pre-fix directory name keeps its original spelling.
- After delivery, the user opened the notebook in a KaTeX-based frontend
  (VS Code) and every table output failed with
  `KaTeX parse error: No such environment: tabular`. `md_table` returned
  `Markdown.MD`, which IJulia publishes with `text/html`, `text/markdown`, and
  `text/latex`; the KaTeX frontend selected `text/latex` and rejected Julia's
  `\begin{tabular}`. `md_table` now returns an `HtmlTable` wrapper with only
  `text/html` and `text/plain` show methods, built through `Markdown.html`.
  Both notebooks were re-executed from fresh kernels and re-exported: all 14
  table outputs per notebook are HTML, and the exports contain zero `tabular`
  or KaTeX blocks.

Details: `wiki/failures/2026-10-06-workflow-notebook-helper-defects.md`.

## Files

- `notebook/conditional_graph_distributed_workflow.ipynb` (new, executed)
- `notebook/phase_k_workflow_helpers.jl` (new)
- `scripts/plots/plot_phase_k_demo.py` (three figures, pinned command)
- `scripts/plots/requirements-phase-k.txt` (new)
- `scripts/plots/plot_phase_k_network.py`, `scripts/plots/plot_phase_k_timing.py`
  (pinned command line only)
- `scripts/plots/README.md` (pinned commands, new figure list)
- `artifacts/logs/conditional-wrapper/notebook/2026-10-06/` (evidence)

## Next action

Implement `notebook/conditional_graph_benchmark.ipynb` from the corrected
`timing/stretch-2026-10-06-leonardo/` campaign (plan implementation item 6),
including the throughput and component views and the ratio-of-medians checks.

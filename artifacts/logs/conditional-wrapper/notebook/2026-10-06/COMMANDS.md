# Phase K distributed workflow notebook — execution record

Date: 2026-10-06
Notebook: `notebook/conditional_graph_distributed_workflow.ipynb`
Helper: `notebook/phase_k_workflow_helpers.jl`
Plotters: `scripts/plots/plot_phase_k_demo.py`, `scripts/plots/plot_phase_k_network.py`
Pinned Python environment: `scripts/plots/requirements-phase-k.txt`
(uv 0.12.10, Python 3.13.13, matplotlib 3.11.2, numpy 2.5.3)

## Environment

```bash
julia --version
# 1.12.6
jupyter kernelspec list
# julia-1.12 is the execution kernel (IJulia)
uv --version
# uv 0.12.10
```

The notebook kernels start in the default `v1.12` environment. They read only
artifacts plus the `flux-integration` source tree; they load neither Flux, MPI,
nor CUDA, and they join no collective.

## Artifact-mode execution, four-rank default

```bash
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1500 \
    notebook/conditional_graph_distributed_workflow.ipynb
```

Outcome: exit 0, 53 cells, 21 executed code cells, 0 error outputs.
Selected bundle `tutorial-4rank` (Leonardo, 4x A100, CUDA/NCCL).

Key verified values:

- campaign status `PASS`; resume markers `472/472` = 4 ranks x 118 steps;
- task exposures 6400 (A) and 8960 (B); mixed-step weights 1/4 and 3/4;
- `peer_param_maxerr` 5.960e-8, `peer_mom1_maxerr` 4.657e-9,
  `peer_mom2_maxerr` 1.455e-11, `peer_beta_exact` false records 0;
- collective protocol: 7 collectives with both tasks present, 5 with one task
  globally absent; presence flags are one CPU `Vector{Float32}` average first;
- recorded source hashes 10/11 match the current worktree; the one changed
  file is `scripts/examples/conditional_graph/reference.jl` (post-campaign
  additive `reference_update!` helper from the timing corrections); Flux
  wrapper hashes match.

## Artifact-mode execution, two-rank selection

```bash
PHASE_K_TUTORIAL_CONFIG=tutorial-2rank \
jupyter nbconvert --to notebook --execute \
    --output /tmp/opencode/phase-k-notebook/conditional_graph_distributed_workflow_2rank.ipynb \
    --ExecutePreprocessor.timeout=1500 \
    notebook/conditional_graph_distributed_workflow.ipynb
```

Outcome: exit 0, 0 error outputs. Selected bundle `tutorial-2rank`
(deathstar, 2x TITAN RTX, CUDA/NCCL).

Key verified values:

- campaign status `PASS`; resume markers `236/236` = 2 ranks x 118 steps;
- task exposures 7680 (A) and 7680 (B); mixed-step weights 1/2 and 1/2;
- `peer_param_maxerr`, `peer_mom1_maxerr`, `peer_mom2_maxerr` exactly 0.0;
  `peer_beta_exact` false records 0;
- captions are bundle-adaptive: the same notebook prints the two-rank numbers
  and the 1/2 + 1/2 weights.

Copies in this directory:

- `workflow-2rank-executed.ipynb` (executed two-rank notebook)
- `workflow-4rank.html` (HTML export of the four-rank default notebook)
- `workflow-2rank.html` (HTML export of the two-rank execution)

## HTML inspection

```bash
jupyter nbconvert --to html --output <path>.html <notebook>.ipynb
```

Outcome: exit 0. Both HTML exports contain 12 embedded figures in section
order (network, routing, exposure, collectives, discrepancy, validation loss,
validation accuracy, validation difference, learning rate, predictions A,
predictions B, continuation), nine level-2 sections in order, and no error
outputs. nbconvert reports the expected "alternative text is missing" warning
for IJulia PNG displays.

## Missing-artifact behavior

`resolve_bundle("unknown-key")` raises:

```
unknown tutorial configuration "unknown-key"; available: tutorial-2rank, tutorial-4rank
```

`bundle_missing_files` on a bundle whose run root is absent lists every
required missing path (metadata, command records, console logs, checkpoint,
trajectory, and the four train/resume tables). No figure falls back to another
run.

## Fresh-training child process (opt-in path)

The notebook default (`run_training = false`) starts no process. The child
path was verified with a direct helper call, not from the notebook kernel:

```julia
include("notebook/phase_k_workflow_helpers.jl")
using .PhaseKWorkflow
bundle = resolve_bundle("tutorial-2rank")
result = run_training_child(bundle; backend = "mpi", ranks = 2, timeout = 1200)
```

```bash
timeout 1500 julia --startup-file=no /tmp/opencode/phase-k-child-run.jl
```

Outcome: `CHILD_EXIT=0 elapsed=151.8`. Campaign root
`artifacts/logs/conditional-wrapper/notebook/tutorial-2rank-20261006-1710769`
with `PHASE_K_SUMMARY dataset=mnist-fashion backend=mpi ranks=2 PASS`; driver
testset `Phase K mnist-fashion campaign (mpi) | 6 6 2m20.5s`; `mpi2-train`
exit 0 in 69.7 s and `mpi2-resume` exit 0 in 66.7 s.

## Plotter commands used inside the notebook

```bash
uv run --offline --with-requirements scripts/plots/requirements-phase-k.txt \
    python3 scripts/plots/plot_phase_k_network.py \
    --out notebook/figures/conditional_graph_distributed_workflow/<key>/network \
    --worlds <ranks>

uv run --offline --with-requirements scripts/plots/requirements-phase-k.txt \
    python3 scripts/plots/plot_phase_k_demo.py \
    --run artifacts/logs/conditional-wrapper/demo/<selected-root> \
    --out notebook/figures/conditional_graph_distributed_workflow/<key> \
    --title "Phase K tutorial: <label>"
```

The plotter writes `figures-metadata.txt` with input and output SHA-256 in the
same directory.

## Post-delivery fix and re-execution

A KaTeX-based frontend (VS Code) rejected every table output with
`No such environment: tabular`, because `md_table` returned `Markdown.MD` and
IJulia also published a `text/latex` representation. `md_table` now returns an
`HtmlTable` wrapper with only `text/html` and `text/plain` show methods
(`showable(MIME"text/latex"(), table) == false`). Both notebooks were
re-executed from fresh kernels with the same commands above and re-exported:
each publishes exactly 14 `text/html` and 14 `text/plain` table outputs, and
the HTML exports contain 14 `<table>` elements, zero `tabular` occurrences, and
zero KaTeX blocks. The resumed-step output now also prints the parameter and
Adam-moment maxima at resumed steps specifically.

## Out of scope

The second deliverable, `notebook/conditional_graph_benchmark.ipynb` (one-A100
versus four-A100 timing), is not implemented in this session. The timing
plotters and the pinned environment are ready for it.

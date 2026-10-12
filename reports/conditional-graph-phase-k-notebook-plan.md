# Phase K: two IJulia notebooks

Date: 2026-10-06
Status: implementation plan. The Python plotting approach is a recommendation.

## Deliverables

1. `notebook/conditional_graph_distributed_workflow.ipynb`: wrapper tutorial, training results, and checkpoint continuation.
2. `notebook/conditional_graph_benchmark.ipynb`: corrected one-A100 versus four-A100 timing experiment and analysis.

Both notebooks use IJulia. Four ranks are the default tutorial configuration. Two ranks are selectable through one configuration cell.
The notebooks read saved artifacts by default. A reader needs no GPU allocation to reproduce the analysis.
Each notebook includes a separate reproduction section with the exact campaign commands.
Fresh training runs use the existing bounded drivers as child processes. The notebook kernel does not join MPI collectives.

This plan replaces the earlier single `conditional_graph_gpu_recap.ipynb` proposal.

## Plotting recommendation

Retain Python, Matplotlib, and NumPy for figures. Call the existing command-line plotters from Julia through `run`.
Display their PNG outputs through IJulia MIME display. Add SVG export for diagrams and line plots where useful.
Use one small Julia helper for paths, subprocess arguments, figure selection, and display.
PythonCall and PyCall are unnecessary for this file-based interface.

The existing scripts contain strict schema checks, reference overlays, timing arithmetic, and provenance output:

- `scripts/plots/plot_phase_k_network.py`
- `scripts/plots/plot_phase_k_demo.py`
- `scripts/plots/plot_phase_k_timing.py`

A Julia rewrite duplicates that work and requires new numerical and visual parity checks.
The tutorial remains Julia-based because its model, AD, optimizer, and reproduction examples use Julia.
The plotting language does not affect measured training time. Figure generation occurs after training.

Implementation will pin Python plotting dependencies in a dedicated environment.
Saved notebook outputs remain readable without Python. Figure regeneration requires that environment.
Julia cells calculate small explanatory tables from the same artifacts, including speedup arithmetic.
Those calculations must agree with the plotter summaries.

## Artifact selection and provenance

The configuration cell selects a complete run bundle, not a rank-count label alone.

| Selection | Artifact root under `artifacts/logs/conditional-wrapper/` |
|---|---|
| Tutorial, four ranks | `demo/k2-2026-10-05-leonardo-nccl4/` |
| Tutorial, two ranks | `demo/k2-2026-10-05-deathstar-nccl2/` |
| Shared dataset provenance | `demo/k2-2026-10-05-deathstar-mpi/prepared/` |
| Benchmark | `timing/stretch-2026-10-06-leonardo/` |

Each bundle identifies train/resume tables, summaries, prepared-data provenance, source hashes, hardware, backend, dtype, and tolerances.
All figures and numerical captions derive from that selection.
Missing required files cause an explicit error. Figure selection never falls back to another run.
The default artifact analysis uses portable tables and images. Training and checkpoint replay use the recorded Julia environment.

## Notebook 1: distributed workflow

### Teaching sequence

1. Purpose, selected run, environment, and provenance.
2. Shared-trunk model and dataset-conditioned routing.
3. Six-step schedule and gradient weighting.
4. Backend initialization, device placement, synchronization, and optimizer setup.
5. Local AD followed by the distributed wrapper update outside AD.
6. Reference comparisons and collective order.
7. Learning curves, configured learning rate, and predictions.
8. Data-only checkpoint and fresh-process continuation.
9. Reproduction commands and demonstrated API scope.

The code examples use the current `../flux-integration` implementation and show the resolved Flux path.
They explain the public `DistributedOptimizer` setup path and the resulting `DistributedOptimizerState`.
They retain returned state/model values from `Optimisers.update!` and use native Optimisers.jl Adam.
The tutorial distinguishes normal setup from the campaign's native-tree wrapping and checkpoint reconstruction.
The explicit fixed-step training loop follows the real driver. Every rank takes each update in the same order.

The weighting equation is:

`g = (1 / R) * sum(g_r for r in 0:R-1)`.

Each local loss is a batch mean. Equal local batch sizes make this the global sample mean.
Locally absent parameters contribute zero when another rank uses them.
Globally absent parameters skip the Adam update and retain their parameters and optimizer history.
The denominator is the total rank count, not the number of active ranks.

### Graph implementation

| Figure | Implementation and data | Required explanation |
|---|---|---|
| Model diagram | Reuse the network plotter. Show 784 → 32 shared trunk and two ten-class heads, with 25,780 parameters. | Dataset identity selects the head. The target class does not select it. |
| Six-step routing grid | Reuse the routing plotter. Select the configured rank count and cross-check the saved schedule. | Steps 1–2: one A rank, remaining B ranks. Step 3: all A. Step 4: all B. Step 5: original split. Step 6: reversed split. |
| Task weights and exposure | Add a compact weight panel and cumulative exposure curves from saved schedule/metric records. | Four-rank mixed steps have weights 1/4 and 3/4. Two-rank mixed steps have weights 1/2 and 1/2. |
| Parameter and Adam discrepancies | Reuse the discrepancy figure. Label replica/reference errors separately and include scalar agreement counts. | Small error supports the recorded reference checks. Preserve exact zeros in labels and tables. |
| Collective protocol | Reuse or extend the schedule figure with ordered trace evidence for representative steps. | One CPU presence reduction precedes ordered gradient reductions. GPU gradients use NCCL. Expected counts are five or seven. |
| Validation loss | Reuse the two-task DDP/reference overlay, against global update index. | Describe initial/final loss and the maximum recorded reference difference. |
| Validation accuracy | Reuse the two-task DDP/reference overlay, with accuracy bounded by 0 and 1. | Report task-specific results and exposure. Loss reduction alone does not establish correctness. |
| Configured learning rate | Reuse the learning-rate plot, with a compact legend for coincident series. | Adam eta is constant at 0.001 in this campaign. This is not an effective step-size estimate. |
| Predictions | Reuse the fixed-sample prediction figure. Add image thumbnails only where saved sample identities can resolve them. | Show true class, predicted class, and task. Keep sample selection deterministic and report errors as well as successes. |
| Checkpoint continuation | Add a focused train/resume difference panel and a match-count table. Use saved resume markers and metric intersections. | Checkpoint after step 2, fresh processes, continuation from step 3. Require parameter and optimizer-state evidence alongside visual overlap. |

The main learning plots show uninterrupted DDP and reference curves.
The continuation section adds resumed curves, which avoids four nearly identical lines throughout the tutorial.
Detailed metric-difference plots can appear in an appendix.

Every graph has three adjacent items: its question, a data-derived observation, and its interpretation.
Captions update with the selected bundle. Four-rank and two-rank task exposures differ.
Their correctness runs also use different hardware. These runs do not form a scaling experiment.

Raw maximum errors do not directly represent the combined `atol`/`rtol` acceptance rule.
A normalized tolerance plot requires elementwise reference magnitudes. Otherwise, show raw errors and the recorded acceptance result separately.
Zero-valued errors need an explicit zero label if a logarithmic display uses a floor.

### API scope

The tutorial explains fixed model structure, identical optimizer configuration, and equal rank participation.
It explains `.tree`, data-only checkpoints, and fresh backends on every rank.
Finite losses do not guarantee finite gradients. `Flux.train!` does not solve asymmetric non-finite losses.
Any future numerical guard needs an all-rank decision before a rank stops or skips an update.
The existing finite K2 evidence supports this presentation. Acceptance of broader deferred behavior remains a separate decision.

## Notebook 2: one-A100 versus four-A100 benchmark

### Experimental definition

The baseline is one process on one A100 with native Adam.
It consumes the same four virtual shards and routed objective as the four-rank DDP run.
Both modes perform 120 updates with global batch 128, for 15,360 samples per repeat.
The corrected campaign contains five repeats per mode.
The notebook explains warm-up, CUDA synchronization, the untimed correctness preflight, and timing boundaries before the results.

### Graph implementation

| Figure | Implementation and data | Required explanation |
|---|---|---|
| Duration distributions | Reuse timing plots, with separate training-only and total panels. Show all five observations plus medians. | Baseline uses `t_train_s`/`t_total_s`. DDP uses `slowest_train_s`/`slowest_rank_s`. Repeats are independent, not paired. |
| Speedup and efficiency | Extend the summary figure or adjacent table. Compute ratios from per-mode medians. | Training speedup is 1.888594. Total speedup is 1.397335. Training efficiency is approximately 47.2% across four GPUs. |
| Throughput | Add samples/s or updates/s from fixed work divided by each repeat duration. Show training-only and total separately. | Global throughput counts the same work once. It does not multiply the global sample count by four. |
| Progress against elapsed time | Reuse recorded steps/samples versus elapsed-time figures. | These are measured record timestamps, including the documented diagnostic intervals. They are not training-only timestamps. |
| Quality against elapsed time | Reuse task-specific loss/accuracy timing plots. Add update-index overlays where needed. | Explain the same-work trajectory and the time needed to reach recorded quality. Label repeat bands as descriptive spread. |
| Timing components | Add per-repeat rank-0 training, validation, writing, and residual components if the schema supports additive reconstruction. | Label these as rank-0 diagnostics. Do not stack them into a claimed slowest-rank decomposition. |

Headline arithmetic:

- Training-only: `median(single.t_train_s) / median(ddp.slowest_train_s)`.
- Total: `median(single.t_total_s) / median(ddp.slowest_rank_s)`.
- Four-GPU training efficiency: `training_speedup / 4`.

The notebook computes these values from the corrected tables and compares them with the saved summary.
The historical 2026-10-05 instrumented campaign is superseded.
The current plotter requires the corrected 15-column `repeats.tsv` schema.

The interpretation states that this is a tiny-model, fixed-work experiment on the recorded node.
Five subsecond repeats show variability. They do not establish general scaling or statistical confidence.
The results do not attribute overhead to communication without component profiling.

## Implementation order and acceptance

1. Define the artifact bundle map and dependency environments.
2. Add the shared Julia subprocess/display helper and data-derived summary tables.
3. Extend the existing Python plots with exposure, continuation, throughput, and optional component views.
4. Build the workflow notebook around the actual API and the selected four-rank artifacts.
5. Add the two-rank selection and its own captions, references, and continuation counts.
6. Build the benchmark notebook from the corrected campaign only.
7. Run both notebooks from fresh IJulia kernels in artifact mode.
8. Run the workflow notebook again with the two-rank configuration.
9. Export HTML and inspect legends, image resolution, formulas, paths, and output order.

Acceptance requires complete selected-run coverage, matching source provenance, correct numerical summaries, and explicit missing-artifact errors.
Summary checks include resume counts of 472/472 at four ranks and 236/236 at two ranks.
Benchmark checks include five repeats per mode and both recorded ratio-of-medians values.
The notebooks must run without remote execution in artifact mode.
Exact implementation commands and outcomes belong in the implementation devlog.

**Exact next action:** implement the artifact bundle map and shared IJulia display helper, then extend the existing Python figures.

## Implementation status (2026-10-06)

Notebook 1 (`notebook/conditional_graph_distributed_workflow.ipynb`) is
implemented, executed from fresh IJulia kernels in artifact mode at both
tutorial rank counts (resume 472/472 and 236/236), and exported to HTML with 12
figures per run. `notebook/phase_k_workflow_helpers.jl` provides the artifact
bundle map and table/summary helpers; `scripts/plots/plot_phase_k_demo.py`
gained the exposure, collectives, and continuation figures; the plotting
environment is pinned in `scripts/plots/requirements-phase-k.txt`. The opt-in
child-process training path was verified (MPI 2 ranks, 6/6 PASS).

Notebook 2 (`notebook/conditional_graph_benchmark.ipynb`) remains the next
implementation item (order item 6 above). Evidence:
`artifacts/logs/conditional-wrapper/notebook/2026-10-06/COMMANDS.md`; devlog
`wiki/devlog/02-conditional-graph/2026-10-06-phase-k-workflow-notebook-implementation.md`.

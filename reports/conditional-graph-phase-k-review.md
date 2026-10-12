# Phase K implementation review and GPU notebook plan

Date: 2026-10-06

## Verdict

The required correctness demonstration is suitable for a GPU recap notebook.
This review found no blocking defect in the inspected K1/K2 correctness path.
The optional timing path needs corrections before a new training-speed comparison.
The stored 1.480203 ratio describes the existing instrumented runs only.

Scope: source inspection, stored K2 GPU evidence, and local figure regeneration.
This review did not run new distributed training or remote commands.
The current Flux wrapper source hash matches both stored GPU campaigns.

## Findings

### P1: The timed baseline includes asymmetric diagnostic transfers

`scripts/examples/conditional_graph_timing.jl:272-283` calls `CGReference.reference_step!` for every baseline update.
`scripts/examples/conditional_graph/reference.jl:59-77` snapshots each globally absent head before and after its update.
`head_state` converts its parameters and both Adam moments to host arrays at lines 46-53.
These operations run inside the timed loop at `conditional_graph_timing.jl:419-421`.
The DDP step at lines 288-298 has no equivalent snapshots.

The six-step schedule contains 40 globally absent steps per 120-update repeat.
The baseline therefore requests 480 array copies to the host per repeat for these checks.
The extra copies and synchronization can bias the comparison in favor of DDP.
The report incorrectly states that the timing path contains no parameter snapshots (timing report, lines 49-50).
The magnitude of this bias remains unmeasured.

Correction: retain these assertions in the untimed correctness path.
Use the native gradient/update path without snapshots in timed baseline steps.
Then repeat the correctness preflight and the timing campaign.
The existing correctness evidence remains valid.

### P2: The timing result does not isolate training-only duration

`scripts/examples/conditional_graph_timing.jl:433-455` includes validation and record writes inside the elapsed interval.
The headline denominator uses the slowest total rank duration.
The numerator uses baseline total duration.
The report discloses validation inclusion, but prompt section 11 requests separate chart overhead and training-only measurements.

`t_train_s` subtracts validation duration but retains the record writes and their file-open/flush cost.
Only rank 0 writes that field to the repeat table.
There is no slowest-rank, training-only duration in the stored summary.
The subsecond repeats make these distinctions relevant.

Correction: measure training blocks separately from validation and record writes.
Use the slowest rank for each distributed training interval.
Report both training-only duration and total duration with explicit labels.

### P2: The speedup figure uses a different summary statistic

`scripts/plots/plot_phase_k_timing.py:411-426` pairs independent repeats by row position and takes the median of their ratios.
The report defines speedup as the ratio of the two median durations.
These statistics can differ after a new campaign or a change in repeat order.
For the stored five-repeat campaign, both happen to equal 1.480203.

Correction: use the ratio of medians for the summary line.
Keep individual durations and their spread visible without implying paired experiments.

## Correctness evidence retained

The implementation follows the main Phase K contracts:

- The host-side model branch evaluates exactly one task head per local batch.
- The saved schedule contains explicit samples and a six-step routing cycle.
- The native reference differentiates the mean of all scheduled shard losses once per update.
- The wrapper update stays outside AD and retains both returned values.
- Each update compares parameters, Adam moments, decay products, and replica values.
- Globally absent heads retain their parameters and Adam history exactly.
- The GPU path requires CUDA arrays and enables scalar-indexing checks.
- The checkpoint contains model data and native optimizer trees, without runtime backends.
- Fresh processes restore the checkpoint and continue against the uninterrupted trajectory.
- Rank 0 writes separate DDP/reference validation records and learning-rate records.

The GPU protocol uses CPU presence metadata through MPI and device gradient buffers through NCCL.
It is not an entirely NCCL-only protocol.

| Stored K2 campaign | Train rank-step records | Resume matches | Maximum parameter error | Final digit accuracy | Final clothing accuracy |
|---|---:|---:|---:|---:|---:|
| deathstar, 2 NCCL ranks | 240 | 236/236 | 0.0 | 0.856 | 0.727 |
| Leonardo, 4 NCCL ranks | 480 | 472/472 | 5.960464e-08 | 0.847 | 0.739 |

Both modes contain every expected rank/step key and exact Adam decay-product agreement.
Each mode contains the expected five-collective and seven-collective steps.
The four-rank maximum errors for Adam moments are 4.656613e-09 and 1.455192e-11.
The declared Float32 tolerances remain `atol=2e-6`, `rtol=1e-5`.

Evidence roots:

- `artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-deathstar-nccl2/`
- `artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4/`

## Proposed notebook

Suggested file: `notebook/conditional_graph_gpu_recap.ipynb`, with an IJulia kernel.
The notebook uses existing modules and the bounded driver as its execution interface.
The default presentation reads stored artifacts, so a reader can reproduce the recap without a GPU allocation.
A separate execution section runs the driver as a child process in the configured GPU environment.
The driver owns MPI process creation and fresh-process resume.

Suggested sections:

1. **Purpose and model:** shared trunk, equal-shaped heads, and dataset-conditioned routing.
2. **Run selection:** two TITAN RTX GPUs or four Leonardo A100 GPUs, with run identity and source hashes.
3. **Gradient semantics:** average over all ranks, including locally absent contributions.
4. **Six-step schedule:** rank/task grid, global absence, reactivation, and per-task exposure.
5. **Training code:** the real AD call and wrapper update from the existing implementation.
6. **Correctness:** parameter and moment discrepancies, scalar agreement, and collective counts.
7. **Learning:** per-task DDP/reference overlays, configured `eta`, and fixed prediction samples.
8. **Continuation:** checkpoint after step 2, fresh processes, resume at step 3, and complete match counts.
9. **Reproduction and scope:** exact commands, environment, immutable output roots, and supported configurations.

The two-rank and four-rank schedules have different task weights.
Their task exposures are 7680/7680 and 6400/8960 respectively.
Each configuration needs its own reference overlay.
The different hardware and objectives prevent a direct two-versus-four-GPU scaling claim from these correctness runs.

The older `notebook/conditional_graph_optimisers.ipynb` documents the historical per-leaf failure and selects the frozen worktree.
The new notebook must select `../flux-integration` and show the resolved Flux path.
Checkpoint resume requires unchanged source hashes and package paths.
A fresh full driver run is the reproducible execution unit after source changes.

The GPU artifact roots reuse a prepared dataset outside their local `prepared/` directories.
The plotter therefore warns about missing adjacent provenance files.
The notebook must link the shared dataset provenance under `k2-2026-10-05-deathstar-mpi/prepared/` explicitly.

## Review commands and outcomes

Stored-evidence audit (exit 0):

```bash
python3 /tmp/opencode/review_phase_k_20261006.py
```

The audit checks summary status, unique rank/step records, resume counts, metrics, collective counts, and the current Flux wrapper hash.
It also recomputes both timing summary statistics.
The script is a local review artifact under `/tmp/opencode`.

Figure regeneration (exit 0, 17 PNGs plus metadata):

```bash
uv run --offline --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_demo.py --run artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-deathstar-nccl2 --run artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4 --out /tmp/opencode/phase-k-review-20261006-figures
```

The plotter passed its schema checks for both campaigns.
Its warnings concern legacy console directories and external prepared-data provenance.
No command failed during this review.

Whitespace checks (exit 0):

```bash
git diff --check -- README.md wiki/context/current.md && git diff --no-index --check /dev/null reports/conditional-graph-phase-k-review.md && git diff --no-index --check /dev/null wiki/devlog/02-conditional-graph/2026-10-06-phase-k-review-notebook.md
```

**Exact next action:** create `notebook/conditional_graph_gpu_recap.ipynb` from the existing two-rank and four-rank correctness artifacts.
The timing corrections remain separate follow-up work before new performance claims.

## Corrections applied (2026-10-06)

All three timing findings were corrected and re-measured in a fresh Leonardo
campaign. The findings above are unchanged.

### P1 closed

Timed and warm-up baseline steps now call the plain native
`CGReference.reference_update!`; the snapshot-based `reference_step!` runs
only in the untimed `--check-trajectory` path. The corrected campaign preflight
still passes on both modes (`single` `maxabs=2.980232e-08`, `ddp`
`maxabs=0.000000e+00`; tolerances unchanged at `atol=2.0e-6`, `rtol=1.0e-5`).
Evidence:
`artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/`
(`CAMPAIGN.md`, `summary-correctness.txt`, `correctness-single/console.log`,
`correctness-ddp/console.log`) and
`reports/conditional-graph-phase-k-timing-stretch.md` section 1.

### P2a closed

Validation and record writes now run in a barrier-guarded diagnostic window
that is excluded from a training-only accumulator. `repeats.tsv` carries
`t_write_s`, `slowest_train_s`, and training throughput columns;
`TIMING_SUMMARY` reports `median_train_s`, and DDP uses slowest-rank values.
Evidence: `baseline/repeats.tsv`, `ddp/repeats.tsv`, `baseline/console.log`,
and `ddp/console.log` `TIMING_SUMMARY` lines in the corrected root, plus
`reports/conditional-graph-phase-k-timing-stretch.md` sections 2 and 5.

### P2b closed

The driver and `plot_phase_k_timing.py` both use the ratio of per-mode median
durations, and `timing-speedup.png` shows individual repeats per mode instead
of paired ratios. Evidence:
`artifacts/logs/leonardo/2026-10-06-timing-r2.log`
(`speedup_train=1.888594 speedup_total=1.397335`),
`reports/figures/phase-k-timing/figures-metadata.txt`, and
`reports/conditional-graph-phase-k-timing-stretch.md` sections 6 and 8.
The historical instrumented 1.480203 is superseded; the old run root is not
replottable with the current 15-column strict schema.

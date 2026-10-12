# Phase K plotting tool

`plot_phase_k_demo.py` reads the Phase K train/resume TSV artifacts described in
`temp/phase_k_k2_interface.md` and writes deterministic presentation PNGs plus a
`figures-metadata.txt` provenance record. It never modifies or invents data: a
malformed header, duplicate key, non-finite number, or inconsistent identity
stops the run before any figure is written.

## Reproducible command

The system `python3` has no `pip`, `numpy`, or `matplotlib`. Use the pinned
environment in `scripts/plots/requirements-phase-k.txt`:

```sh
uv run --offline --with-requirements scripts/plots/requirements-phase-k.txt \
    python3 scripts/plots/plot_phase_k_demo.py \
    --run <ROOT> [--run <ROOT>...] \
    --out <FIG_DIR> \
    [--title TITLE]
```

`--run` is repeatable. Each `ROOT` is a campaign directory that contains one or
more `<config>/train` + `<config>/resume` pairs (for example `mpi2/train`,
`mpi2/resume`, `mpi4/...`, `nccl2/...`). A `ROOT` that directly contains the
`train/` and `resume/` pair is also accepted. Legacy sibling directories such as
`mpi2-train/` that hold only `command.txt`/`console.log` are ignored; the tool
warns when it replaces such a path with the real TSV directory.

## Inputs and validation

Every mode directory must contain `validation-<mode>.tsv`,
`learning-rate-<mode>.tsv`, `predictions-<mode>.tsv`, and `metrics-<mode>.tsv`
with the exact tab-separated headers below. `metadata.txt` is optional and read
when present; `sources.tsv` and `subset_indices.tsv` are optional and read from
the prepared directory when present. Missing optional files produce warnings,
not errors.

| File | Columns |
|---|---|
| `validation-{train,resume}.tsv` | `run_id mode schedule_digest dataset_digest step model task n loss accuracy exposure present checkpoint_step` |
| `learning-rate-{train,resume}.tsv` | `run_id mode schedule_digest step model rank leaf eta` |
| `predictions-{train,resume}.tsv` | `run_id mode schedule_digest step model task sample_index target predicted pixels` |
| `metrics-{train,resume}.tsv` | `step cycle present_a present_b counts_a counts_b loss_ref loss_local_rank0 peer_param_maxerr peer_mom1_maxerr peer_mom2_maxerr peer_beta_exact val_a_loss val_a_acc val_b_loss val_b_acc` |

Checks include:

- exact header match and fixed field count per row;
- unique keys: `(step, model, task)` for validation, `(step, model, rank, leaf)`
  for rates, `(step, model, task, sample_index)` for predictions, `step` for
  metrics;
- finite numbers, boolean `true`/`false` flags, `accuracy` in `[0, 1]`,
  positive `n`, non-negative `exposure`, `counts_a/b >= 0`, canonical six leaves
  (`trunk_weight`, `trunk_bias`, `head_a_weight`, `head_a_bias`,
  `head_b_weight`, `head_b_bias`), and DDP ranks contiguous from 0;
- one `run_id`, `schedule_digest`, `dataset_digest`, `checkpoint_step`, worker
  count, and horizon shared by the trained and resumed modes;
- contiguous step grids: train validation `0..S`, resume validation
  `checkpoint..S`, train metrics/rates `1..S`, resume metrics/rates
  `checkpoint+1..S`, prediction snapshots `{0, S}` (train) and
  `{checkpoint, S}` (resume);
- identical `eta` across all leaves and ranks at every step;
- predictions fixed across snapshots: the same first-20 sample-index order,
  target, and pixel vector for a `(task, sample_index)`;
- when metadata provides `local_batch`, exposure deltas equal
  `counts x local_batch`.

A `peer_beta_exact` value of `false` is plotted as measured and recorded as a
warning; it is not a schema error.

## Figures

Per `<config>` (filename prefix is the parent directory name, plus the campaign
root when several `--run` roots are supplied):

1. `<label>-validation-loss.png` / `<label>-validation-accuracy.png` - DDP vs
   reference overlay per task, train solid and resume dashed with markers.
2. `<label>-validation-difference.png` - per-task DDP-minus-reference difference
   for loss and accuracy, so matching overlays remain visible.
3. `<label>-learning-rate.png` - configured Adam `eta` against global update
   index for every DDP rank and the reference, with distinct linestyles,
   markers, and an explicit legend; y label is `configured Adam learning rate
   eta`.
4. `<label>-discrepancy.png` - `peer_param_maxerr`, `peer_mom1_maxerr`, and
   `peer_mom2_maxerr` on a log scale; `peer_beta_exact` (required true at every
   step); `loss_ref` vs `loss_local_rank0`.
5. `<label>-predictions-task-a.png` / `...-task-b.png` - first 20 fixed
   validation samples, target and predicted labels at the initial and final
   snapshots. Square pixel vectors (K2: 784 values) are reshaped to a row-major
   image grid; two-value toy samples use a feature scatter.
6. `<label>-schedule.png` - per-step task presence and contributing-rank counts,
   cumulative exposure per task, shaded globally absent steps, and the
   dotted checkpoint/resume boundary line.
7. `<label>-exposure.png` - per-step task weight (`counts / ranks`) and the
   cumulative exposure curves, for the selected rank count.
8. `<label>-collectives.png` - expected collectives per step (`1` presence
   reduction plus one gradient reduction per present leaf: 5 or 7).
9. `<label>-continuation.png` - train-minus-resume difference for the DDP
   validation loss and accuracy at the metric intersection
   `checkpoint+1..steps`, with exact-match counts and maximum differences.
   Zero-valued differences are annotated explicitly.

When more than one configuration is loaded, `phase-k-comparison.png` adds final
validation loss and accuracy per task per configuration, annotated with the
maximum final DDP-minus-reference difference.

All PNGs are written at 160 dpi with deterministic layouts (sorted series, no
randomness). `figures-metadata.txt` records the run roots, per-pair labels,
run ids, schedule/dataset digests, every input path with SHA256 and row count,
every output path with SHA256 and size, warnings, and the exact generation
command including a ready-to-copy `uv run` line.

## Exit codes

- `0` - all requested figures written.
- `2` - schema/validation error, missing input, or a load failure; no figures
  are written for the malformed configuration.
- `1` - unexpected Python failure.

## Testing

Synthetic K2-like (784 pixels) and toy-like (2 pixels) fixtures are generated
under `/tmp/opencode` only, never in the repository. The test harness checks
that every expected figure exists and is non-empty, that `figures-metadata.txt`
lists them all, that a corrupted header exits with status 2, and that repeated
runs produce identical PNG bytes:

```sh
python3 /tmp/opencode/phase-k-fixtures/make_fixtures.py
uv run --offline --with-requirements scripts/plots/requirements-phase-k.txt \
    python3 scripts/plots/plot_phase_k_demo.py \
    --run /tmp/opencode/phase-k-fixtures/k2_like --out /tmp/opencode/phase-k-tests/k2
python3 /tmp/opencode/phase-k-fixtures/check_fixture_outputs.py /tmp/opencode/phase-k-tests/k2
```

# Phase K timing-stretch plotting and live viewer

`plot_phase_k_timing.py` reads the timing-stretch artifacts described in
`temp/phase_k_timing_stretch_plan.md`. It never invents metric values: a
malformed header, non-finite number, or inconsistent mode stops the run with
exit status 2 before any figure is written. A measured speedup below 1 is a
valid result and is plotted as measured. Only `matplotlib` and `numpy` are
used.

## Static figures

```sh
uv run --offline --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_timing.py \
    --baseline <BASELINE_DIR> --ddp <DDP_DIR> --out <FIG_DIR> \
    [--title TITLE] [--label offline|replay|live]
```

Each directory must contain `records.tsv`, `steps.tsv`, and `repeats.tsv` with
the exact tab-separated headers below. `metadata.txt` is optional and recorded
when present.

| File | Columns |
|---|---|
| `records.tsv` | `run_id mode repeat step elapsed_s train_s samples_total samples_a samples_b updates_per_s samples_per_s val_a_loss val_a_acc val_b_loss val_b_acc` |
| `steps.tsv` | `mode repeat step elapsed_s train_s samples_total samples_a samples_b` |
| `repeats.tsv` | `run_id mode repeat t_total_s t_train_s t_val_s t_write_s slowest_rank_s slowest_train_s steps samples_total updates_per_s samples_per_s train_updates_per_s train_samples_per_s` |

Figures (160 dpi, deterministic layouts, all curve x axes are elapsed training
time in `train_s`, validation excluded):

1. `timing-updates-vs-elapsed.png` / `timing-samples-vs-elapsed.png` - median
   DDP and baseline curves with per-repeat spread bands and thin per-repeat
   lines.
2. `timing-val-loss-{a,b}.png` / `timing-val-acc-{a,b}.png` - task-specific
   validation metrics at the record-boundary cadence.
3. `timing-repeat-times.png` - per-repeat rank-0 local wall time decomposed
   into stacked `t_train_s` + `t_val_s` + `t_write_s` + remainder segments with
   median lines; DDP `slowest_rank_s` (`x`) and `slowest_train_s` (`+`) markers.
4. `timing-speedup.png` - two panels (training-only and total) that show each
   mode's individual repeat durations as jittered points with median lines. The
   annotated speedup is the ratio of the two median durations (baseline / DDP
   slowest rank); repeats are not paired by index. A 1.0 s reference line and a
   `--label replay` note are drawn in each panel.

`figures-metadata.txt` records every input and output path with SHA256, the
row counts, the `--label` provenance value, and the exact command including a
ready-to-copy `uv run --offline` line. The DDP step curves use rank-0 local
training time; the headline speedup is the ratio of median durations and uses
the slowest-rank training time (`TIMING_SUMMARY median_train_s` for DDP), with
the total-time ratio using `median_total_s` (`slowest_rank_s` for DDP).

## Live viewer

```sh
uv run --offline --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_timing.py \
    --follow --records <DIR>/records.tsv [--baseline <BASELINE_DIR>] \
    --out <FIG_DIR> --interval 2 --status-log <FILE> [--replay]
```

The viewer polls the growing `records.tsv`, prints and appends lines of the
form

```
[viewer] wall=<iso> status=<WAITING|RUNNING|DONE> last_step=<n> record_age_s=<x> updates_per_s=<x>
```

and refreshes `timing-live.png`. `record_age_s` is measured from the local
receipt wall clock. `status` is `DONE` when a sibling `done` marker exists and
the file stopped growing for two intervals, at which point the viewer exits.
A missing file, partial final lines, and an optional saved baseline are
handled. `--replay` labels every viewer output as replay and must not be
presented as a live measurement; the plain `--label` values `live` and
`offline` are also shown in every figure.

## Exit codes

- `0` - all requested figures written, or the viewer reached `DONE`.
- `2` - schema/validation error, missing input, or a load failure.
- `1` - unexpected Python failure.

# Phase K network diagram

`plot_phase_k_network.py` draws the K2 conditional network (`TwoHeadModel`) and
its six-step routing cycle. It is a schematic, not a data plot: the content is
parsed from the repository sources and a missing or ambiguous pattern stops the
run with exit status 2 before any figure is written.

```sh
uv run --offline --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_network.py \
    --out reports/figures/phase-k-network [--worlds 2,4]
```

Parsed sources:

- K2 model dimensions from the `CGModel.build_toy_model(seed_init; indim=...,
  hidden=..., nclasses=...)` line in `conditional_graph_demo.jl`;
- trunk/head layer definitions from `conditional_graph/model.jl`;
- `ROUTING_CYCLE` purposes and rank assignment from `conditional_graph_demo.jl`;
- the `prepare` global-batch default from `conditional_graph_demo.jl`.

Figures (160 dpi, deterministic layouts):

1. `phase-k-k2-network.png` - architecture: input (784), shared trunk
   (`Dense(784 => 32, tanh)`, 25 120 parameters), both heads
   (`Dense(32 => 10)`, 330 parameters each), parameter totals, and one lane per
   task showing the active path and the locally absent head. The footer states
   the DDP presence protocol and the resulting 5-versus-7 collective count.
2. `phase-k-k2-routing.png` - the six routing cycles for `--worlds`, with the
   per-step purpose strings and the resulting global presence.

`figures-metadata.txt` records the parsed model dimensions and parameter
counts, the global batch, every routing cycle, the SHA256 of every consumed
source and produced figure, and the exact command including a ready-to-copy
`uv run --offline` line.

## Exit codes

- `0` - both figures written.
- `2` - a source file is missing, has no match, or has an unexpected match; no
  figures are written.
- `1` - unexpected Python failure.

# Phase K timing corrections and Leonardo re-measurement

Date: 2026-10-06
Host: `leonardo` (CINECA), 4× NVIDIA A100-SXM-64GB, driver 535.274.02
Nodes/jobs: correctness `lrdn1934` (salloc `59514262`), timing `lrdn0009`
(`59514959`), both on `boost_usr_prod`
Flux worktree: `~/projects/flux-integration`, HEAD `55171e65` dirty (git fields
read `unavailable` on the synced tree); no Flux source changed this session; no
commits
Julia: 1.12.6, project `~/projects/flux-integration/test`
Predecessor: [Phase K review and GPU notebook plan](2026-10-06-phase-k-review-notebook.md)
Campaign record: `artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/CAMPAIGN.md`
Supervisor report: `reports/conditional-graph-phase-k-timing-stretch.md`

## Outcome

All three timing findings from the 2026-10-06 review are corrected and
re-measured. Correctness preflight PASS (`single` trajectory
`maxabs=2.980232e-08`, `ddp` `maxabs=0.000000e+00`, tolerances unchanged
`atol=2.0e-6`, `rtol=1.0e-5`); baseline 5/5 and DDP 5/5 timed repeats.
Corrected headline (ratio of medians): training-only speedup
`0.263440 / 0.139490 = 1.888594`; total speedup
`0.438678 / 0.313939 = 1.397335`. All jobs exited 0; 0
`Precompiling packages` lines in all four console logs; no Flux source
changed; no commits. The core correctness matrix is untouched and valid.

## Plan

1. Correct P1 (baseline diagnostic host copies), P2a (training-only timing
   isolation), and P2b (ratio-of-medians summary) in the timing driver,
   reference path, and plotter.
2. Verify the corrected path locally on CPU/MPI, including an independent
   schema and arithmetic check.
3. Re-run the Leonardo correctness preflight and the timing campaign under a
   fresh root.
4. Rewrite the report, add the campaign record, and update README and wiki.

## Review findings

From [the review](../../../reports/conditional-graph-phase-k-review.md):

- P1: the timed baseline called `CGReference.reference_step!`, which snapshots
  each globally absent head's parameters and both Adam moments to host arrays.
  The six-step schedule contains 40 globally absent steps per 120-step repeat,
  so the baseline took 480 `Array` copies per repeat; DDP had no equivalent.
- P2a: timing did not isolate training-only duration. Record writes stayed in
  `t_train_s`, only rank 0 wrote that field, and there was no slowest-rank
  training-only field.
- P2b: the speedup figure used the median of paired per-repeat ratios while the
  report used the ratio of medians.

The last two statistics coincide for the stored campaign, but the baseline
bias was unmeasured.

## Implementation

- `scripts/examples/conditional_graph_timing.jl`: timed and warm-up baseline
  steps now call the plain native `CGReference.reference_update!` with no host
  snapshots. The absent-head assertions remain in the untimed
  `--check-trajectory` correctness path. A barrier-guarded diagnostic window
  now excludes validation and record writes from training time. New
  `t_write_s` and `slowest_train_s` fields (`slowest_train_s` from
  `Allgather` of per-rank training intervals); `repeats.tsv` extended from 11
  to 15 columns; `TIMING_SUMMARY` now reports `median_total_s` and
  `median_train_s`.
- `scripts/checks/conditional_graph_timing_check.jl`: new `--phase timing`
  runs baseline + DDP and writes one `summary-timing.txt`. The driver summary
  reports both `speedup_train` and `speedup_total`, each from the ratio of
  medians.
- `scripts/plots/plot_phase_k_timing.py`: headline uses `ratio_of_medians`;
  `timing-speedup.png` shows per-mode individual repeat durations instead of
  index-paired ratios; strict 15-column schema check (exits 2 on a mismatched
  header).
- `scripts/examples/conditional_graph/reference.jl` is unchanged; the timing
  CLI switched to the existing `reference_update!` for timed steps, and the
  additive `reference_step!` remains the untimed check entry point.

## Local verification (CPU/MPI)

Local smoke on the deathstar prepared K2 artifact was run from the thesis repo
root (steps 8, warm-up 1, repeats 2, record-every 2), exit 0:

```bash
julia --project=../flux-integration/test scripts/checks/conditional_graph_timing_check.jl \
  --root /tmp/opencode/phase-k-timing-corrected-smoke-20261006 \
  --prepared artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-deathstar-mpi/prepared/mnist-fashion_prepared.jls \
  --backend mpi --device cpu --ranks 4 --steps 8 --warmup 1 --repeats 2 \
  --record-every 2 --check-steps 4 --timeout 1200 --phase all --label local-smoke
```

All four children exited 0; the same-host check printed
`TIMING_CHECK_RESULT source=single-vs-ddp PASS maxabs=2.980232e-08 nbad=0` and
`TIMING_RANK_DUMP rank=0..3 equal=true`. The local CPU speedup values are not
meaningful (one CPU process versus four MPI ranks) and only exercise the new
schema and arithmetic.

Independent local audit, exit 0:

```bash
python3 /tmp/opencode/audit_phase_k_timing.py
```

It re-loaded both `repeats.tsv` files, asserted the 15-column header,
finiteness, `t_train_s <= t_total_s`, `t_val_s >= 0`, `t_write_s >= 0`,
`t_train_s + t_val_s + t_write_s <= t_total_s`, the DDP bounds
`t_train_s <= slowest_train_s <= slowest_rank_s`, and recomputed all six
summary values; result: `schema+arithmetic problems: NONE`.

Local plotter checks:

```bash
uv run --offline --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_timing.py \
  --baseline artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/baseline \
  --ddp artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/ddp \
  --out /tmp/opencode/phase-k-new-schema-check
```

wrote 8 figures plus `figures-metadata.txt` and printed
`speedup (ratio of medians, baseline / DDP): training_only=1.888594
total=1.397335`. The same command against the 2026-10-05 root exited 2 with
`repeats.tsv header mismatch` (intended strict-schema behavior).

## Leonardo campaign commands

Correctness preflight `59514262` on `lrdn1934` (driver expansion, quoted from
`artifacts/logs/leonardo/2026-10-06-timing-correctness-r2.log`; invoked through
`scripts/remote/run.sh leonardo`):

```bash
JULIA_DEPOT_PATH=/leonardo_work/IscrC_UNMASKED/epicazio/julia-depot PATH=/leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin:$PATH \
  SLURM_MPI_TYPE=pmi2 JULIA_PKG_PRECOMPILE_AUTO=0 \
  salloc --account=iscrc_unmasked --partition=boost_usr_prod --qos=boost_qos_dbg \
  -N1 --ntasks=4 --cpus-per-task=4 --gres=gpu:4 --time=00:30:00 \
  srun --mpi=pmi2 -n 1 -c 8 --chdir=$HOME/projects/ddp-flux-thesis \
  julia --startup-file=no --project=$HOME/projects/flux-integration/test \
  scripts/checks/conditional_graph_timing_check.jl \
  --root artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo \
  --prepared /leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls \
  --trajectory artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4/nccl4/train/trajectory.jls \
  --backend nccl --device cuda --ranks 4 --steps 12 --warmup 2 --repeats 1 \
  --record-every 4 --check-steps 12 --timeout 3600 --phase correctness \
  --label k2-timing-correctness-r2
```

Timing `59514959` on `lrdn0009` (same wrapper; only the driver arguments
change):

```bash
JULIA_DEPOT_PATH=/leonardo_work/IscrC_UNMASKED/epicazio/julia-depot PATH=/leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin:$PATH \
  SLURM_MPI_TYPE=pmi2 JULIA_PKG_PRECOMPILE_AUTO=0 \
  salloc --account=iscrc_unmasked --partition=boost_usr_prod --qos=boost_qos_dbg \
  -N1 --ntasks=4 --cpus-per-task=4 --gres=gpu:4 --time=00:30:00 \
  srun --mpi=pmi2 -n 1 -c 8 --chdir=$HOME/projects/ddp-flux-thesis \
  julia --startup-file=no --project=$HOME/projects/flux-integration/test \
  scripts/checks/conditional_graph_timing_check.jl \
  --root artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo \
  --prepared /leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls \
  --backend nccl --device cuda --ranks 4 --steps 120 --warmup 3 --repeats 5 \
  --record-every 5 --timeout 3600 --phase timing --label k2-timing-r2
```

Both driver summaries are status PASS: `summary-correctness.txt` and
`summary-timing.txt` in the new root.

## Corrected campaign results

Fixed work per repeat: 120 global updates × global batch 128 = 15360 samples
(`a=6400`, `b=8960`). Baseline = 1 process on 1 A100; DDP = 4 NCCL ranks on
devices `[1,2,3,0]`. Prepared artifact SHA-256 `7f0d38d9...`, schedule digest
`ce6b41aa...`, dataset digest `658e3dd3...` (unchanged).

Baseline `repeats.tsv` (single, 1× A100):

| repeat | t_total_s | t_train_s | t_val_s | t_write_s |
|---|---|---|---|---|
| 1 | 0.577706 | 0.324570 | 0.200927 | 0.051707 |
| 2 | 0.438678 | 0.263440 | 0.158325 | 0.016411 |
| 3 | 0.429985 | 0.263047 | 0.153931 | 0.012529 |
| 4 | 0.430569 | 0.261991 | 0.154793 | 0.013285 |
| 5 | 0.837961 | 0.591889 | 0.233055 | 0.012528 |
| median | **0.438678** | **0.263440** | 0.158325 | 0.013285 |

DDP `repeats.tsv` (4× A100 NCCL, devices `[1,2,3,0]`):

| repeat | slowest_rank_s | t_train_s | slowest_train_s | t_val_s | t_write_s |
|---|---|---|---|---|---|
| 1 | 0.672028 | 0.471516 | 0.471712 | 0.179729 | 0.020052 |
| 2 | 0.313939 | 0.139279 | 0.139490 | 0.163333 | 0.010612 |
| 3 | 0.366796 | 0.193618 | 0.193702 | 0.161043 | 0.011472 |
| 4 | 0.292301 | 0.122781 | 0.122834 | 0.157857 | 0.011060 |
| 5 | 0.290867 | 0.119330 | 0.119353 | 0.161112 | 0.009844 |
| median | **0.313939** | 0.139279 | **0.139490** | 0.161112 | 0.011060 |

Driver lines:

```text
TIMING_SUMMARY mode=single repeats=5 median_total_s=0.438678 min_total_s=0.429985 max_total_s=0.837961 median_train_s=0.263440 min_train_s=0.261991 max_train_s=0.591889 median_updates_per_s=273.548871 median_train_updates_per_s=455.512186
TIMING_SUMMARY mode=ddp repeats=5 median_total_s=0.313939 min_total_s=0.290867 max_total_s=0.672028 median_train_s=0.139490 min_train_s=0.119353 max_train_s=0.471712 median_updates_per_s=382.240102 median_train_updates_per_s=860.276605
TIMING_SPEEDUP backend=nccl ranks=4 median_single_train_s=0.263440 median_ddp_slowest_train_s=0.139490 speedup_train=1.888594 median_single_total_s=0.438678 median_ddp_slowest_total_s=0.313939 speedup_total=1.397335
TIMING_STRETCH_SUMMARY backend=nccl ranks=4 speedup_train=1.888594 speedup_total=1.397335 status=PASS
```

Correctness preflight: `single` `TIMING_CHECK_RESULT source=trajectory PASS
maxabs=2.980232e-08`; `ddp` `TIMING_CHECK_RESULT source=trajectory PASS
maxabs=0.000000e+00`; both child `done` markers PASS. Figures under
`reports/figures/phase-k-timing/` regenerated to 8 PNGs plus
`figures-metadata.txt`.

## Historical versus corrected

- Historical instrumented run (2026-10-05): total speedup 1.480203 (baseline
  median `t_total_s` 0.465324, DDP median `slowest_rank_s` 0.314365, baseline
  median `t_train_s` 0.295540).
- Corrected baseline median `t_train_s` 0.263440 is −0.032100 s (−10.9%)
  against the historical 0.295540, consistent with removing the 480 diagnostic
  host copies per repeat.
- Corrected DDP slowest total 0.313939 is −0.14% against the historical
  0.314365.
- The historical 1.480203 describes the instrumented runs only and is
  superseded by the corrected values above.

## Documentation updated

- `reports/conditional-graph-phase-k-timing-stretch.md` (rewritten; handled
  outside this session's edit scope).
- New campaign record
  `artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/CAMPAIGN.md`.
- `README.md` (review paragraph, Phase K complete paragraph, current
  checkpoint, pending evidence, `CW-K-review` and `CW-K` rows), with the
  historical 1.480203 explicitly marked superseded.
- `wiki/context/current.md` (new top section), `wiki/context/leonardo.md` (new
  status paragraph), and this devlog.

## Failures

No command failed: no `status=FAIL`, no `ERROR: LoadError`, all driver and
child exit codes 0. No `wiki/failures/` entry is needed. The only console
warnings are the ssh post-quantum notice and the srun `--cpus-per-task` note
(8 requested inside a 4-CPU job), both pre-existing and harmless.

The strict plotter exiting 2 on the old 11-column `repeats.tsv` header is
intended behavior, not a failure: the 2026-10-05 campaign roots are no longer
replottable with the current schema-checking plotter.

## Exact next action

Review the deferred findings and decide the demonstration preconditions
(report section 6); then create `notebook/conditional_graph_gpu_recap.ipynb`
from the existing two-rank and four-rank correctness artifacts.

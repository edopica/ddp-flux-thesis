# Phase K timing stretch: 1×A100 versus 4×A100 NCCL on Leonardo (corrected)

Supervisor report. Date: 2026-10-06. Host: `leonardo` (CINECA). This report
supersedes the 2026-10-05 timing report. Scope: optional post-Phase K stretch
from `temp/conditional_graph_phase_k_prompt.md` section 11 and
`temp/phase_k_timing_stretch_plan.md`, rerun after the corrections requested in
`reports/conditional-graph-phase-k-review.md`.

Status: **COMPLETE and CORRECTED.** Correctness preflight PASS; baseline 5/5
and DDP 5/5 timed repeats; driver 3m39.9s. Corrected headline statistics use
the ratio of per-mode median durations. Training-only
`speedup_train = median(single t_train_s) / median(ddp slowest_train_s) =
0.263440 / 0.139490 = 1.888594`. Total
`speedup_total = median(single t_total_s) / median(ddp slowest_rank_s) =
0.438678 / 0.313939 = 1.397335`. The historical instrumented statistic
1.480203 describes the 2026-10-05 runs only and is superseded (section 7). No
Flux source changed; no commits; 0 `Precompiling packages` lines in all four
console logs. Timings are diagnostic measurements of this fixed schedule on
one A100 versus four A100s; they are not a production-speedup claim.

## 1. Scope and corrections

The 2026-10-06 review (`reports/conditional-graph-phase-k-review.md`) found
three issues in the optional timing stretch. This campaign closes them.

### P1: Timed baseline host snapshots

The timed single-GPU baseline called `CGReference.reference_step!`, which
snapshots the parameters and both Adam moments of globally absent heads to CPU
arrays (480 `Array` copies per 120-step repeat) while DDP has no equivalent
operation. Correction: timed and warm-up baseline steps now call the plain
native `CGReference.reference_update!`; the untimed `--check-trajectory` path
keeps the `reference_step!` assertions.

### P2a: Training-only isolation

The timing fields did not isolate training-only duration: record-write time
was retained inside `t_train_s`, and no slowest-rank training-only field
existed. Correction: validation and record writes now run in a barrier-guarded
diagnostic window that is excluded from a new training-only accumulator.
`repeats.tsv` now has columns `run_id mode repeat t_total_s t_train_s t_val_s
t_write_s slowest_rank_s slowest_train_s steps samples_total updates_per_s
samples_per_s train_updates_per_s train_samples_per_s`. `TIMING_SUMMARY`
reports `median_total_s` and `median_train_s`, and DDP uses the slowest-rank
values for both.

### P2b: Speedup summary statistic

The speedup figure used the median of paired per-repeat ratios instead of the
ratio of median durations. Correction: the driver and the figure both use the
ratio of medians, and the figure shows individual repeats per mode, not paired
ratios.

### Driver phase

The driver has a new `--phase timing` mode that runs baseline and DDP together
and writes one `summary-timing.txt`; phase summaries no longer overwrite each
other.

## 2. Method

- Fixed work per repeat: 120 global updates × global batch 128 = 15360
  scheduled samples (`samples_a=6400`, `samples_b=8960` in both modes).
- Warm-up is untimed: `--warmup 3` steps plus one validation call to compile
  both paths; the model and optimizer tree are then discarded and rebuilt from
  the prepared artifact before every timed repeat. The schedule restarts at
  step 1 in every repeat.
- Timed region per repeat: `MPI.Barrier` (DDP) and `CUDA.synchronize()`, then
  `t0`, the fixed-work step loop, then `CUDA.synchronize()` between steps. Each
  `timed_step!` call accumulates training-only nanoseconds around it.
- Validation runs only at record boundaries (`--record-every 5`) and record
  writes happen only after validation, inside a `Barrier`-guarded diagnostic
  window that every rank enters and leaves in the same order. This window is
  excluded from `t_train_s`; its validation and write components are reported
  as `t_val_s` and `t_write_s`.
- The DDP repeat duration used for the speedup is the slowest rank:
  `slowest_rank_s = maximum(Allgather(local_elapsed))` and
  `slowest_train_s = maximum(Allgather(local_train))`.
- Baseline and DDP ran inside the same Slurm timing allocation on the same
  node.
- No in-loop native reference, parameter snapshots, or collective traces are
  present in the timing path. Correctness checking is a separate, untimed
  `--check-trajectory` mode.

## 3. Correctness preflight

The corrected timing code paths were checked before any pacing measurement.

1. Same-host equivalence: a local CPU/MPI smoke run passed before the remote
   run (bounded driver with a scratch root under `/tmp/opencode`). The
   authoritative gate is the Leonardo trajectory check below.
2. Leonardo allocation `59514262` on node `lrdn1934`: `single` trajectory
   check PASS `maxabs=2.980232e-08` (`nbad=0`); `ddp` trajectory check PASS
   `maxabs=0.000000e+00` (all ranks, `nbad=0`). Tolerances were unchanged at
   `atol=2.0e-6, rtol=1.0e-5`. Child elapsed 164.0 s and 56.7 s; driver
   3m41.2s. Console logs:
   `correctness-single/console.log` and `correctness-ddp/console.log` in the
   campaign root.
3. The timed and warm-up baseline steps no longer call `reference_step!`; the
   trajectory assertions run only in the untimed `--check-trajectory` path.

## 4. Hardware, provenance, and exact commands

Both jobs ran on Leonardo `boost_usr_prod` (`--gres=gpu:4`, `cpu=16`), driver
535.274.02, 4× NVIDIA A100-SXM-64GB per node. Baseline is 1 process on GPU
device 0; DDP is 4 NCCL ranks on devices `[1,2,3,0]` with
`CUDA_VISIBLE_DEVICES=0,1,2,3`.

| Job | salloc ID | Node | Purpose | Driver elapsed |
|---|---|---|---|---|
| Correctness preflight | 59514262 | `lrdn1934` | `single` + `ddp` trajectory check | 3m41.2s |
| Timing | 59514959 | `lrdn0009` | baseline + DDP timed repeats | 3m39.9s |

Provenance (`baseline/metadata.txt`, `ddp/metadata.txt`):

| Item | Value |
|---|---|
| Julia | 1.12.6 |
| Project | `~/projects/flux-integration/test` |
| Flux | 0.16.11 (`pathof(Flux)` = `~/projects/flux-integration/src/Flux.jl`) |
| Optimisers / Functors | 0.4.9 / 0.5.3 |
| MPI / Zygote | 0.20.27 (MPICH via `srun --mpi=pmi2`) / 0.7.13 |
| CUDA / NCCL / cuDNN | 6.2.2 / 0.1.3 / 6.2.2 |
| Driver | 535.274.02 |
| Optimizer / eta / dtype | native Adam / 0.001 / Float32 |
| Tolerances | atol = 2.0e-6, rtol = 1.0e-5 (unchanged) |
| Prepared artifact | `/leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls`, SHA-256 `7f0d38d997b14de3905ab9176cd3f914103363398c0bd9be6bf6360595373cd6`, 39804054 bytes |
| Dataset digest | `658e3dd341ca4e4fa9e5e4a05eacf53c0bebca626a05b17a87e917062dec2971` |
| Schedule digest | `ce6b41aa81c9eb9d3c4cc526a25f50e4e302738e47e660db0b4e5ad2aa78bc57` (world 4) |
| `conditional_graph_timing.jl` SHA-256 | `635925650a257ab52ea3b55abd2b41f29e014ca1ed2195bed2be81965fc4bf81` |
| `conditional_graph_timing_check.jl` SHA-256 | `5a46120dd6c111c027d76d43cde58207f98d82c7e92855e87fe94b35d49471c8` |
| `plot_phase_k_timing.py` SHA-256 | `23db4e1007520e5e51ddf44d53757f1cafe166a5c16f5655ada72bd71837443a` |
| `conditional_graph/reference.jl` SHA-256 | `324c735ca9c86cae55e3faeeeefe5759c3c3680c57166928ff82921716604e3a` |
| `flux-integration/src/Flux.jl` SHA-256 | `78710f9b398971609ac86824cb9ce6ab2ce3ab4672139df2142d24d44bacd0ce` |
| `flux-integration/src/distributed/public_api.jl` SHA-256 | `2984210b84e136da9bc9948c3b5928877f1f6c10f870f8db841ad2d4ab6136f0` |
| Uncounted warm-up | `compile_warmup_s` 34.031006 (baseline) / 28.722841 (DDP) |

Each command below was run from the thesis root through
`scripts/remote/run.sh leonardo '<inner srun>'`.

Correctness preflight (job 59514262):

```bash
scripts/remote/run.sh leonardo 'srun --mpi=pmi2 -n 1 -c 8 --chdir=$HOME/projects/ddp-flux-thesis julia --startup-file=no --project=$HOME/projects/flux-integration/test scripts/checks/conditional_graph_timing_check.jl --root artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo --prepared /leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls --trajectory artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4/nccl4/train/trajectory.jls --backend nccl --device cuda --ranks 4 --steps 12 --warmup 2 --repeats 1 --record-every 4 --check-steps 12 --timeout 3600 --phase correctness --label k2-timing-correctness-r2'
```

Timing, baseline and DDP in one phase (job 59514959):

```bash
scripts/remote/run.sh leonardo 'srun --mpi=pmi2 -n 1 -c 8 --chdir=$HOME/projects/ddp-flux-thesis julia --startup-file=no --project=$HOME/projects/flux-integration/test scripts/checks/conditional_graph_timing_check.jl --root artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo --prepared /leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls --backend nccl --device cuda --ranks 4 --steps 120 --warmup 3 --repeats 5 --record-every 5 --timeout 3600 --phase timing --label k2-timing-r2'
```

Local driver logs:
`artifacts/logs/leonardo/2026-10-06-timing-correctness-r2.log` and
`artifacts/logs/leonardo/2026-10-06-timing-r2.log`.

Static figures (exit 0, 8 PNGs plus `figures-metadata.txt`):

```bash
uv run --offline --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_timing.py --baseline artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/baseline --ddp artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/ddp --out reports/figures/phase-k-timing --title "Phase K timing corrections: 1xA100 vs 4xA100 NCCL (Leonardo, 2026-10-06)"
```

The plotter printed `training_only=1.888594 total=1.397335`.

## 5. Results

Same work per repeat: 120 updates, 15360 samples, exposures a=6400 / b=8960.
`t_total_s`, `t_train_s`, `t_val_s`, and `t_write_s` are rank-local values from
rank 0; `t_val_s` and `t_write_s` are always rank 0 because only rank 0 runs
validation and record writes. For DDP, `slowest_rank_s` and `slowest_train_s`
are the maxima over ranks.

### Baseline (`baseline/repeats.tsv`, single, 1× A100)

| repeat | t_total_s | t_train_s | t_val_s | t_write_s |
|---|---|---|---|---|
| 1 | 0.577706 | 0.324570 | 0.200927 | 0.051707 |
| 2 | 0.438678 | 0.263440 | 0.158325 | 0.016411 |
| 3 | 0.429985 | 0.263047 | 0.153931 | 0.012529 |
| 4 | 0.430569 | 0.261991 | 0.154793 | 0.013285 |
| 5 | 0.837961 | 0.591889 | 0.233055 | 0.012528 |
| median | **0.438678** | **0.263440** | **0.158325** | **0.013285** |
| min / max | 0.429985 / 0.837961 | 0.261991 / 0.591889 | 0.153931 / 0.233055 | 0.012528 / 0.051707 |

### DDP (`ddp/repeats.tsv`, 4× A100 NCCL, devices `[1,2,3,0]`)

| repeat | t_total_s (rank 0) | t_train_s (rank 0) | t_val_s (rank 0) | t_write_s (rank 0) | slowest_rank_s | slowest_train_s |
|---|---|---|---|---|---|---|
| 1 | 0.672028 | 0.471516 | 0.179729 | 0.020052 | 0.672028 | 0.471712 |
| 2 | 0.313939 | 0.139279 | 0.163333 | 0.010612 | 0.313939 | 0.139490 |
| 3 | 0.366796 | 0.193618 | 0.161043 | 0.011472 | 0.366796 | 0.193702 |
| 4 | 0.292301 | 0.122781 | 0.157857 | 0.011060 | 0.292301 | 0.122834 |
| 5 | 0.290867 | 0.119330 | 0.161112 | 0.009844 | 0.290867 | 0.119353 |
| median | 0.313939 | 0.139279 | **0.161043** | **0.010612** | **0.313939** | **0.139490** |
| min / max | 0.290867 / 0.672028 | 0.119330 / 0.471516 | 0.157857 / 0.179729 | 0.009844 / 0.020052 | 0.290867 / 0.672028 | 0.119353 / 0.471712 |

### Median throughput

Throughput at the median duration, as reported by `TIMING_SUMMARY`:

| mode | updates/s (total) | updates/s (training-only) |
|---|---|---|
| baseline single | 273.548871 | 455.512186 |
| DDP, slowest rank | 382.240102 | 860.276605 |

Driver summary lines (from the child consoles):

```text
TIMING_SUMMARY mode=single repeats=5 median_total_s=0.438678 min_total_s=0.429985 max_total_s=0.837961 median_train_s=0.263440 min_train_s=0.261991 max_train_s=0.591889 median_updates_per_s=273.548871 median_train_updates_per_s=455.512186
TIMING_SUMMARY mode=ddp repeats=5 median_total_s=0.313939 min_total_s=0.290867 max_total_s=0.672028 median_train_s=0.139490 min_train_s=0.119353 max_train_s=0.471712 median_updates_per_s=382.240102 median_train_updates_per_s=860.276605
```

Final validation at step 120 matched the K2 Leonardo metrics (digits accuracy
0.847, clothing accuracy 0.739). Record and step row counts: 120 records and
600 steps per mode; 4 `TIMING_RANK_DONE mode=ddp` markers.

## 6. Speedup definition

The headline statistics are ratios of per-mode median durations. Repeats are
never paired by row position.

- Training-only:
  `median(single t_train_s) / median(ddp slowest_train_s) = 0.263440 /
  0.139490 = 1.888594`.
- Total:
  `median(single t_total_s) / median(ddp slowest_rank_s) = 0.438678 /
  0.313939 = 1.397335`.

For this tiny fixed schedule the four-rank DDP run was faster than the
single-process baseline by these measured factors. Per-step compute is small,
so the ratios are sensitive to communication and schedule effects. They are
reported as measured, and no correctness claim depends on them.

## 7. Historical comparison

The 2026-10-05 campaign (`stretch-2026-10-05-leonardo`) measured the timed
baseline with `reference_step!`, so its baseline included 480 host snapshot
copies per repeat, and its `t_train_s` still contained record-write time. It
reported `median(t_total_s) / median(slowest_rank_s) = 0.465324 / 0.314365 =
1.480203`. That statistic describes the instrumented runs only and is
superseded by the corrected `1.397335`, which it exceeds by 0.082868 absolute
(about 5.9% relative).

| Statistic | 2026-10-05 instrumented | 2026-10-06 corrected | Change |
|---|---|---|---|
| median baseline `t_total_s` | 0.465324 | 0.438678 | -0.026646 s |
| median baseline `t_train_s` | 0.295540 | 0.263440 | -0.032100 s (-10.9%) |
| median DDP `slowest_rank_s` | 0.314365 | 0.313939 | -0.000426 s (-0.14%) |
| median DDP `slowest_train_s` | not defined | 0.139490 | new field |
| total speedup (ratio of medians) | 1.480203 | 1.397335 | historical superseded |
| training-only speedup (ratio of medians) | not defined | 1.888594 | new statistic |

The baseline `t_train_s` decrease is consistent with removing the diagnostic
host snapshot copies from the timed path. The DDP median total changed by less
than 0.001 s. The old campaign root remains as historical evidence and is not
replottable with the current strict plotter: its `repeats.tsv` has 11 columns,
while the current plotter requires the 15-column schema and exits 2 on the old
files.

## 8. Plots

Static figures in `reports/figures/phase-k-timing/` (8 PNGs plus
`figures-metadata.txt`, whose input SHA-256 rows pin the four TSVs of each
mode):

- `timing-updates-vs-elapsed.png` - completed global updates against elapsed
  time for both modes.
- `timing-samples-vs-elapsed.png` - processed samples (global-batch count)
  against elapsed time for both modes.
- `timing-val-loss-a.png`, `timing-val-acc-a.png` - task A (digits) validation
  loss and accuracy at the record cadence against elapsed time.
- `timing-val-loss-b.png`, `timing-val-acc-b.png` - task B (clothing).
- `timing-repeat-times.png` - per-repeat total and training-only durations for
  each mode.
- `timing-speedup.png` - individual repeat durations per mode and the
  ratio-of-medians speedup for training-only and total time; the panels state
  that individual repeats are shown and are not paired.

The figure command is in section 4. The plotter exited 0 and printed
`training_only=1.888594 total=1.397335`. The figures are labeled `offline` in
`figures-metadata.txt`.

## 9. Interpretation and limitations

- One 120-step repeat of this small model takes well under a second, so five
  repeats leave visible variance: baseline repeat 5 is the slowest baseline
  repeat (0.591889 s training), and DDP repeat 1 is the slowest DDP repeat
  (0.471712 s training-only, slowest rank). The medians are diagnostic, not a
  stable benchmark.
- The baseline evaluates all four scheduled shards serially in one process per
  global update; DDP computes one shard per rank concurrently. The plan warns
  that this tiny model can be communication-bound, so the measured factors are
  properties of this one fixed schedule and are not a general scaling claim.
- The benchmark is schedule- and model-specific. A different workload, batch,
  or rank count can shift or invert the result; a slowdown would have been
  reported as measured.
- Rank-0 record curves can understate the slowest rank; the DDP durations used
  for the speedup are `slowest_rank_s` and `slowest_train_s`.
- Compilation and warm-up (34.031006 s baseline / 28.722841 s DDP) are outside
  every timed repeat.
- The timing allocation ran baseline and DDP sequentially on one node; both
  modes used the same prepared artifact, schedule digest, dtype, optimizer,
  and update budget as the K2 campaign.
- No Flux source was changed; no commit or push was made; `flux-integration`
  was untouched by this stretch.

**Exact next action:** review the deferred findings and decide the
demonstration preconditions (audit 002 section 6); then create
`notebook/conditional_graph_gpu_recap.ipynb` from the existing two-rank and
four-rank correctness artifacts. The optional timing stretch is complete and
corrected.

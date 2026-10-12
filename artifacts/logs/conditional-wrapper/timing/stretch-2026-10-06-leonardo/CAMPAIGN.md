# Phase K timing stretch — leonardo — 2026-10-06 (corrected)

Status: **COMPLETE and CORRECTED.** Correctness preflight PASS; baseline 5/5
timed repeats; DDP 5/5 timed repeats; driver 3m39.9s. All child exits 0. No
Flux source changed; no commits; 0 `Precompiling packages` lines in all four
console logs. Corrected headline statistics are ratios of per-mode medians:
training-only `0.263440 / 0.139490 = 1.888594`; total
`0.438678 / 0.313939 = 1.397335`. Timings are diagnostic measurements of this
fixed schedule on one A100 versus four A100s; they are not a
production-speedup claim.

This root is the corrected rerun after `reports/conditional-graph-phase-k-review.md`.
The 2026-10-05 root remains as historical evidence; its instrumented total
statistic 1.480203 is superseded.

## Allocations, nodes, GPU assignments

| Job | salloc ID | Node | Purpose | Child elapsed | Driver elapsed |
|---|---|---|---|---|---|
| Correctness preflight | 59514262 | `lrdn1934` | `single` + `ddp` trajectory check | 164.0 s / 56.7 s | 3m41.2s |
| Timing | 59514959 | `lrdn0009` | baseline + DDP in one `--phase timing` run | 157.5 s / 61.7 s | 3m39.9s |

Both jobs used `boost_usr_prod`, `--gres=gpu:4`, `cpu=16`, driver 535.274.02,
4× NVIDIA A100-SXM-64GB. Baseline is 1 process on GPU device 0; DDP is 4 NCCL
ranks on devices `[1,2,3,0]` with `CUDA_VISIBLE_DEVICES=0,1,2,3`.

## Correctness preflight (exit 0)

| child | ranks | exit | elapsed | marker | maxabs |
|---|---|---|---|---|---|
| correctness-single | 1 | 0 | 164.0 s | `TIMING_CHECK_RESULT source=trajectory PASS` | `2.980232e-08` (nbad=0) |
| correctness-ddp | 4 | 0 | 56.7 s | `TIMING_CHECK_RESULT source=trajectory PASS` | `0.000000e+00` (all ranks, nbad=0) |

Tolerances were unchanged at `atol=2.0e-6, rtol=1.0e-5`. Run products are
under `correctness/{single,ddp}/`; console logs are `correctness-single/console.log`
and `correctness-ddp/console.log`. A same-host local CPU/MPI smoke run also
passed before the remote run (scratch driver root under `/tmp/opencode`).

## Per-repeat timings

### baseline (`baseline/repeats.tsv`, mode=single, 1 process, 1 GPU, steps=120)

| repeat | t_total_s | t_train_s | t_val_s | t_write_s |
|---|---|---|---|---|
| 1 | 0.577706 | 0.324570 | 0.200927 | 0.051707 |
| 2 | 0.438678 | 0.263440 | 0.158325 | 0.016411 |
| 3 | 0.429985 | 0.263047 | 0.153931 | 0.012529 |
| 4 | 0.430569 | 0.261991 | 0.154793 | 0.013285 |
| 5 | 0.837961 | 0.591889 | 0.233055 | 0.012528 |
| **median** | **0.438678** | **0.263440** | **0.158325** | **0.013285** |
| min / max | 0.429985 / 0.837961 | 0.261991 / 0.591889 | 0.153931 / 0.233055 | 0.012528 / 0.051707 |

### ddp (`ddp/repeats.tsv`, mode=ddp, 4 ranks, 4 GPUs, steps=120)

| repeat | t_total_s (rank 0) | t_train_s (rank 0) | t_val_s (rank 0) | t_write_s (rank 0) | slowest_rank_s | slowest_train_s |
|---|---|---|---|---|---|---|
| 1 | 0.672028 | 0.471516 | 0.179729 | 0.020052 | 0.672028 | 0.471712 |
| 2 | 0.313939 | 0.139279 | 0.163333 | 0.010612 | 0.313939 | 0.139490 |
| 3 | 0.366796 | 0.193618 | 0.161043 | 0.011472 | 0.366796 | 0.193702 |
| 4 | 0.292301 | 0.122781 | 0.157857 | 0.011060 | 0.292301 | 0.122834 |
| 5 | 0.290867 | 0.119330 | 0.161112 | 0.009844 | 0.290867 | 0.119353 |
| **median** | 0.313939 | 0.139279 | **0.161043** | **0.010612** | **0.313939** | **0.139490** |
| min / max | 0.290867 / 0.672028 | 0.119330 / 0.471516 | 0.157857 / 0.179729 | 0.009844 / 0.020052 | 0.290867 / 0.672028 | 0.119353 / 0.471712 |

Throughput at the median duration (`TIMING_SUMMARY` values):

| mode | updates/s (total) | updates/s (training-only) |
|---|---|---|
| baseline single | 273.548871 | 455.512186 |
| ddp, slowest rank | 382.240102 | 860.276605 |

Driver summary lines:

```text
TIMING_SUMMARY mode=single repeats=5 median_total_s=0.438678 min_total_s=0.429985 max_total_s=0.837961 median_train_s=0.263440 min_train_s=0.261991 max_train_s=0.591889 median_updates_per_s=273.548871 median_train_updates_per_s=455.512186
TIMING_SUMMARY mode=ddp repeats=5 median_total_s=0.313939 min_total_s=0.290867 max_total_s=0.672028 median_train_s=0.139490 min_train_s=0.119353 max_train_s=0.471712 median_updates_per_s=382.240102 median_train_updates_per_s=860.276605
TIMING_SPEEDUP backend=nccl ranks=4 median_single_train_s=0.263440 median_ddp_slowest_train_s=0.139490 speedup_train=1.888594 median_single_total_s=0.438678 median_ddp_slowest_total_s=0.313939 speedup_total=1.397335
```

Fixed work: 120 global updates × global batch 128 = 15360 scheduled samples per
repeat (`samples_total` in `steps.tsv` is cumulative; do not sum that column).
Exposures per repeat are identical in both modes: `samples_a=6400`,
`samples_b=8960`. Final validation at step 120 matched the K2 Leonardo
metrics (digits accuracy 0.847, clothing accuracy 0.739).

## Exact commands

Each command was run from the thesis root through
`scripts/remote/run.sh leonardo '<inner srun>'`.

Correctness preflight (job 59514262):

```bash
scripts/remote/run.sh leonardo 'srun --mpi=pmi2 -n 1 -c 8 --chdir=$HOME/projects/ddp-flux-thesis julia --startup-file=no --project=$HOME/projects/flux-integration/test scripts/checks/conditional_graph_timing_check.jl --root artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo --prepared /leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls --trajectory artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4/nccl4/train/trajectory.jls --backend nccl --device cuda --ranks 4 --steps 12 --warmup 2 --repeats 1 --record-every 4 --check-steps 12 --timeout 3600 --phase correctness --label k2-timing-correctness-r2'
```

Timing, baseline + DDP (job 59514959):

```bash
scripts/remote/run.sh leonardo 'srun --mpi=pmi2 -n 1 -c 8 --chdir=$HOME/projects/ddp-flux-thesis julia --startup-file=no --project=$HOME/projects/flux-integration/test scripts/checks/conditional_graph_timing_check.jl --root artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo --prepared /leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls --backend nccl --device cuda --ranks 4 --steps 120 --warmup 3 --repeats 5 --record-every 5 --timeout 3600 --phase timing --label k2-timing-r2'
```

Local driver logs:
`artifacts/logs/leonardo/2026-10-06-timing-correctness-r2.log` and
`artifacts/logs/leonardo/2026-10-06-timing-r2.log`.

Static figures (exit 0, 8 PNGs + `figures-metadata.txt`; printed
`training_only=1.888594 total=1.397335`):

```bash
uv run --offline --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_timing.py --baseline artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/baseline --ddp artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/ddp --out reports/figures/phase-k-timing --title "Phase K timing corrections: 1xA100 vs 4xA100 NCCL (Leonardo, 2026-10-06)"
```

## Provenance

| Item | Value |
|---|---|
| Date | 2026-10-06 |
| Julia | 1.12.6 |
| Project | `~/projects/flux-integration/test` |
| Flux | 0.16.11 (`pathof(Flux)` = `~/projects/flux-integration/src/Flux.jl`) |
| Optimisers / Functors | 0.4.9 / 0.5.3 |
| MPI / Zygote | 0.20.27 (MPICH via `srun --mpi=pmi2`) / 0.7.13 |
| CUDA / NCCL / cuDNN | 6.2.2 / 0.1.3 / 6.2.2 |
| Driver | 535.274.02 (4× NVIDIA A100-SXM-64GB) |
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
| Uncounted warm-up | `compile_warmup_s` 34.031006 (baseline) / 28.722841 (ddp) |

## Root file list

```text
summary-correctness.txt      phase = correctness, status PASS
summary-timing.txt           phase = timing, status PASS
correctness/single/          done, metadata.txt, records.tsv, repeats.tsv, steps.tsv
correctness/ddp/             done, metadata.txt, records.tsv, repeats.tsv, steps.tsv
correctness-single/          command.txt, console.log
correctness-ddp/             command.txt, console.log
baseline/                    command.txt, console.log, done, metadata.txt, records.tsv, repeats.tsv, steps.tsv
ddp/                         command.txt, console.log, done, metadata.txt, records.tsv, repeats.tsv, steps.tsv
```

## Evidence checklist

| Check | Result |
|---|---|
| Correctness `TIMING_CHECK_RESULT ... PASS` (both children) | present; driver exit 0 |
| `correctness/{single,ddp}/done` | `single single status=PASS`, `ddp ddp status=PASS` |
| `baseline/done` / `ddp/done` | `baseline single status=PASS`, `ddp ddp status=PASS` |
| `TIMING_DONE` + `TIMING_SUMMARY repeats=5` | present in both mode consoles |
| `TIMING_RANK_DONE mode=ddp` markers | 4 |
| `repeats.tsv` rows | 5 + header per mode |
| `records.tsv` rows (rank 0 only) | 120 + header per mode |
| `steps.tsv` rows | 600 + header per mode |
| `TIMING_SPEEDUP` training-only / total | 1.888594 / 1.397335 |
| `Precompiling packages` lines | 0 in all four console logs |
| `summary-correctness.txt` / `summary-timing.txt` | both present, `status PASS` |
| Static figures | 8 PNGs + `figures-metadata.txt` in `reports/figures/phase-k-timing/` |

## Caveats

- One 120-step repeat of this small model takes well under a second. The same
  five repeats contain one slow baseline repeat (repeat 5, 0.591889 s
  training) and one slow DDP repeat (repeat 1, 0.471712 s training-only,
  slowest rank). The medians are diagnostic, not a stable benchmark.
- The ratio is schedule- and model-specific; the tiny per-step compute can
  make DDP communication-bound. A different workload, batch, or rank count can
  shift or invert the result.
- Rank-0 record curves can understate the slowest rank; the DDP durations used
  for the speedup are `slowest_rank_s` and `slowest_train_s`.
- The 2026-10-05 root is historical evidence only and is not replottable with
  the current strict plotter: its `repeats.tsv` has 11 columns, while the
  current plotter requires the 14-column schema and exits 2 on the old files.
- No Flux source was changed; no commits were made; `flux-integration` was
  untouched.

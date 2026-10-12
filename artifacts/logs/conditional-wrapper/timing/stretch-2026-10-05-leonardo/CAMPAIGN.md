# Phase K timing stretch — leonardo — 2026-10-05 (FINAL)

Status: **COMPLETE on leonardo. Correctness preflight PASS, baseline 5/5 timed
repeats, DDP 5/5 timed repeats.** All jobs exit 0. No Flux source changed; no
precompile occurred (0 `Precompiling packages` lines in every console log).
Timings are diagnostic measurements of this fixed schedule on one A100 vs four
A100s; they are not a production-speedup claim.

- Correctness: node `lrdn0790`, salloc job `59434349`; `single` trajectory
  check `maxabs=2.980232e-08`, `ddp` `maxabs=0.000000e+00`, both PASS.
- Baseline (1 process, GPU device 0): node `lrdn1990`, salloc job `59434611`.
- DDP (4 ranks, GPUs [1,2,3,0]): node `lrdn1847`, salloc job `59435161`.
- Headline: `speedup = median(t_total_s baseline) / median(slowest_rank_s ddp)
  = 0.465324 / 0.314365 = 1.480203` (a speedup > 1 on this workload).

## Per-repeat timings

### baseline (`baseline/repeats.tsv`, mode=single, 1 process, 1 GPU, steps=120)

| repeat | t_total_s | t_train_s | t_val_s | slowest_rank_s | updates_per_s | samples_per_s |
|---|---|---|---|---|---|---|
| 1 | 0.626333 | 0.467879 | 0.158454 | 0.626333 | 191.591302 | 24523.686699 |
| 2 | 0.439836 | 0.281983 | 0.157853 | 0.439836 | 272.829179 | 34922.134961 |
| 3 | 0.832694 | 0.297053 | 0.535641 | 0.832694 | 144.110578 | 18446.153970 |
| 4 | 0.465324 | 0.295540 | 0.169784 | 0.465324 | 257.884817 | 33009.256576 |
| 5 | 0.444683 | 0.274797 | 0.169886 | 0.444683 | 269.855422 | 34541.494049 |
| **median** | **0.465324** | **0.295540** | **0.169784** | **0.465324** | **257.884817** | **33009.256576** |
| min / max | 0.439836 / 0.832694 | 0.274797 / 0.467879 | 0.157853 / 0.535641 | 0.439836 / 0.832694 | 144.110578 / 272.829179 | 18446.153970 / 34922.134961 |

### ddp (`ddp/repeats.tsv`, mode=ddp, 4 ranks, 4 GPUs, steps=120)

| repeat | t_total_s | t_train_s | t_val_s | slowest_rank_s | updates_per_s | samples_per_s |
|---|---|---|---|---|---|---|
| 1 | 0.714849 | 0.530085 | 0.184764 | 0.714849 | 167.867700 | 21487.065618 |
| 2 | 0.344812 | 0.185500 | 0.159312 | 0.344812 | 348.016031 | 44546.051978 |
| 3 | 0.282049 | 0.129534 | 0.152515 | 0.282049 | 425.458686 | 54458.711835 |
| 4 | 0.314365 | 0.160856 | 0.153509 | 0.314365 | 381.721898 | 48860.402991 |
| 5 | 0.266138 | 0.112892 | 0.153246 | 0.266138 | 450.894077 | 57714.441822 |
| **median** | **0.314365** | **0.160856** | **0.153509** | **0.314365** | **381.721898** | **48860.402991** |
| min / max | 0.266138 / 0.714849 | 0.112892 / 0.530085 | 0.152515 / 0.184764 | 0.266138 / 0.714849 | 167.867700 / 450.894077 | 21487.065618 / 57714.441822 |

Driver summary lines (from child consoles, reproducible by the command strings
in `command.txt` in each directory):

```text
TIMING_SUMMARY mode=single repeats=5 median_total_s=0.465324 min_total_s=0.439836 max_total_s=0.832694 median_updates_per_s=257.884817
TIMING_SUMMARY mode=ddp repeats=5 median_total_s=0.314365 min_total_s=0.266138 max_total_s=0.714849 median_updates_per_s=381.721898
```

Fixed work: 120 global updates × global batch 128 = 15360 scheduled samples per
repeat (each `samples_total` in `steps.tsv` is cumulative; the final row per
repeat is 15360; do not sum that column). Task exposures per repeat are
identical in both modes: `samples_a=6400`, `samples_b=8960` (five repeats:
a=32000, b=44800). DDP repeat 1 is the slowest DDP repeat (0.714849 s,
residual warm-up variance); the baseline's slowest repeat is repeat 3
(0.832694 s), whose spread is dominated by a 0.535641 s validation pass.

Final validation at step 120 (median repeat, rank 0):

| mode | task A loss | task A acc | task B loss | task B acc |
|---|---|---|---|---|
| baseline | 7.657627544e-01 | 0.847000 | 7.740118246e-01 | 0.739000 |
| ddp | 7.657627544e-01 | 0.847000 | 7.740118322e-01 | 0.739000 |

The only difference is Float32 accumulation in task B loss (7.6e-09 absolute, ~9.8e-09 relative).
These values match the K2 Leonardo 4-rank campaign final metrics
(a acc 0.8470, b acc 0.7390).

## Correctness preflight (exit 0)

Driver `--phase correctness`, child logs at
`correctness-single/console.log` and `correctness-ddp/console.log` (the driver's
run directory for the DDP child is `correctness-ddp/`; the run products are
`correctness/ddp/`), run products under `correctness/{single,ddp}/`:

| child | ranks | exit | elapsed | marker | maxabs |
|---|---|---|---|---|---|
| correctness-single | 1 | 0 | 166.1 s | `TIMING_CHECK_RESULT source=trajectory PASS` | `2.980232e-08` (step 12, nbad=0) |
| correctness-ddp | 4 | 0 | 58.2 s | `TIMING_CHECK_RESULT source=trajectory PASS` | `0.000000e+00` (all ranks nbad=0) |

Driver: `TIMING_STRETCH_SUMMARY backend=nccl ranks=4 speedup=na status=PASS`.
`single` is compared to the saved K2 Leonardo trajectory with the same prepared
schedule; `ddp` reproduces the trajectory exactly. The single maximum is the
known Float32 accumulation difference inside `atol=2.0e-6, rtol=1.0e-5`;
tolerances were not changed.

## Live viewer

The local follow viewer ran from 19:08:18 UTC (WAITING) to 19:14:33 UTC (DONE)
while the DDP job ran; it wrote `viewer/timing-live.png` and appended to
`viewer-live.log` (38 KB; `--status-log` and stdout both append there, so each
line appears twice). Status counts: 126 WAITING, 218 RUNNING, 2 DONE; 22
RUNNING lines carry data.

| state | wall (UTC) | line |
|---|---|---|
| first WAITING | 2026-10-05T19:08:18 | `status=WAITING last_step=0 record_age_s=-1.00 updates_per_s=0.000000` |
| first RUNNING | 2026-10-05T19:10:36 | `status=RUNNING last_step=0 record_age_s=0.00 updates_per_s=0.000000` |
| first RUNNING with data | 2026-10-05T19:14:09 | `status=RUNNING last_step=120 record_age_s=0.00 updates_per_s=648.360265` |
| last RUNNING | 2026-10-05T19:14:31 | `status=RUNNING last_step=120 record_age_s=19.79 updates_per_s=1067.708380` |
| DONE | 2026-10-05T19:14:33 | `status=DONE last_step=120 record_age_s=22.01 updates_per_s=1067.708380` |

Deviation, recorded: the ssh pipe delivered the growing `records.tsv` in a late
buffered burst rather than row-by-row, so mid-run `last_step` stayed 0 and all
120 rows arrived ~16 s before the job ended. The viewer still ran live (not a
replay), saw WAITING/RUNNING/DONE, and drew every streamed row. Ordering note:
the viewer was started ~2 min before the stream and the DDP job so that the
WAITING state was captured; the plan's tail-first ordering creates the local
file immediately through the shell redirection, which prevents a WAITING poll.

## Allocation, nodes, GPU assignments, durations

| Job | salloc ID | Node | Elapsed | Start–End (local time) | TRES |
|---|---|---|---|---|---|
| GPU check | 59434335 | lrdn3078 | 00:00:04 | 20:58:44–20:58:48 | 4× A100, cpu=16 |
| correctness | 59434349 | lrdn0790 | 00:04:32 | 20:59:05–21:03:37 | 4× A100, cpu=16 |
| baseline | 59434611 | lrdn1990 | 00:03:40 | 21:04:04–21:07:44 | 4× A100, cpu=16 |
| ddp | 59435161 | lrdn1847 | 00:03:41 | 21:10:44–21:14:25 | 4× A100, cpu=16 |

Julia step wall times: correctness 4m29s, baseline 3m34s, ddp 3m34s.
Driver-reported child elapsed: correctness-single 166.1 s, correctness-ddp
58.2 s (driver total 3m44.8s); baseline 175.6 s (driver total 2m55.9s); ddp
176.6 s (driver total 2m57.2s).

GPU assignments from `metadata.txt`:

- baseline `device 0 0` → single process on device 0; `device_name NVIDIA
  A100-SXM-64GB`; `CUDA_VISIBLE_DEVICES=0,1,2,3` (only device 0 used).
- ddp `device 0 1`, `device 1 2`, `device 2 3`, `device 3 0` → the FluxMPIExt
  rank→device map `[1,2,3,0]`, 4 distinct A100s; `device_name NVIDIA
  A100-SXM-64GB`.
- `srun` reported the benign `Job step's --cpus-per-task value exceeds that of
  job (8 > 4)` warning (same as K1/K2; step ran).

## Exact commands

Targeted source sync (no `--delete`, no full tree, `flux-integration` untouched):

```bash
rsync -avzR ./scripts/examples/conditional_graph_timing.jl \
  ./scripts/checks/conditional_graph_timing_check.jl \
  ./scripts/examples/conditional_graph/reference.jl \
  ./scripts/plots/plot_phase_k_timing.py ./scripts/plots/README.md \
  leonardo:~/projects/ddp-flux-thesis/
```

Remote verification before any job (md5 of the three Julia files matched local
exactly; prepared SHA-256 and size matched):

```bash
ssh leonardo 'md5sum ~/projects/ddp-flux-thesis/scripts/examples/conditional_graph_timing.jl ~/projects/ddp-flux-thesis/scripts/checks/conditional_graph_timing_check.jl ~/projects/ddp-flux-thesis/scripts/examples/conditional_graph/reference.jl'
ssh leonardo 'sha256sum /leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls; wc -c /leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls'
```

md5: `conditional_graph_timing.jl 4eaf7ff6286317f0a2a79ccb6b779cdb`,
`conditional_graph_timing_check.jl a00c31ce60f62236ac03c9ce8eef91bf`,
`reference.jl 6a9422f56138a0ed5e0900e7586bf9bd`.
Prepared: `7f0d38d997b14de3905ab9176cd3f914103363398c0bd9be6bf6360595373cd6`,
39804054 bytes.

GPU check (job 59434335):

```bash
scripts/remote/run.sh leonardo "srun --mpi=pmi2 -n 1 -c 2 bash -lc '/usr/bin/nvidia-smi -L'"
```

Correctness preflight (job 59434349):

```bash
scripts/remote/run.sh leonardo 'srun --mpi=pmi2 -n 1 -c 8 --chdir=$HOME/projects/ddp-flux-thesis julia --startup-file=no --project=$HOME/projects/flux-integration/test scripts/checks/conditional_graph_timing_check.jl --root artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo --prepared /leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls --trajectory artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4/nccl4/train/trajectory.jls --backend nccl --device cuda --ranks 4 --steps 12 --warmup 2 --repeats 1 --record-every 4 --check-steps 12 --timeout 3600 --phase correctness --label k2-timing-correctness'
```

Baseline (job 59434611):

```bash
scripts/remote/run.sh leonardo 'srun --mpi=pmi2 -n 1 -c 8 --chdir=$HOME/projects/ddp-flux-thesis julia --startup-file=no --project=$HOME/projects/flux-integration/test scripts/checks/conditional_graph_timing_check.jl --root artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo --prepared /leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls --backend nccl --device cuda --ranks 4 --steps 120 --warmup 3 --repeats 5 --record-every 5 --timeout 3600 --phase baseline --label k2-timing-baseline'
```

DDP with live stream (job 59435161):

```bash
scripts/remote/run.sh leonardo 'srun --mpi=pmi2 -n 1 -c 8 --chdir=$HOME/projects/ddp-flux-thesis julia --startup-file=no --project=$HOME/projects/flux-integration/test scripts/checks/conditional_graph_timing_check.jl --root artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo --prepared /leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls --backend nccl --device cuda --ranks 4 --steps 120 --warmup 3 --repeats 5 --record-every 5 --timeout 3600 --phase ddp --label k2-timing-ddp'
```

Live stream and viewer (started before the DDP job):

```bash
ssh leonardo "tail -F -n +1 ~/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo/ddp/records.tsv" > /tmp/opencode/timing-live/records.tsv 2>/dev/null &
uv run --offline --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_timing.py --follow --records /tmp/opencode/timing-live/records.tsv --baseline artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo/baseline --out artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo/viewer --interval 2 --status-log artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo/viewer-live.log &
```

Pull-back (exit 0; no forward sync afterward):

```bash
rsync -avz leonardo:~/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo/ artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo/
```

Static figures (exit 0, 8 PNGs + `figures-metadata.txt`, step 7):

```bash
uv run --offline --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_timing.py --baseline artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo/baseline --ddp artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo/ddp --out reports/figures/phase-k-timing --title "Phase K timing stretch: 1xA100 vs 4xA100 NCCL (Leonardo)"
```

## Evidence checklist

| Check | Result |
|---|---|
| `correctness/{single,ddp}/done` + `metadata.txt` + TSVs | present; both `status=PASS` |
| `correctness-{single,ddp}/console.log` `TIMING_CHECK_RESULT ... PASS` | present (maxabs above) |
| `TIMING_STRETCH_SUMMARY ... status=PASS` for correctness | present; driver exit 0 |
| baseline/ddp `TIMING_DONE` and `TIMING_SUMMARY repeats=5` | present; driver exit 0 |
| DDP `TIMING_RANK_DONE mode=ddp` markers | 4 |
| baseline/ddp `repeats.tsv` data rows | 5 each |
| baseline/ddp `records.tsv` rows (rank 0 only) | 120 each (24 per repeat × 5) |
| baseline/ddp `steps.tsv` rows | 600 each |
| `done` markers | `baseline single status=PASS`, `ddp ddp status=PASS` |
| `summary.txt` | present, `status PASS` for the final (`--phase ddp`) invocation; see deviation |
| `Precompiling packages` lines | 0 in all four console logs |
| live viewer log + PNG | present (see Live viewer) |
| figures | 8 PNGs + `figures-metadata.txt` in `reports/figures/phase-k-timing/` |

Deviation: the driver writes one `summary.txt` per invocation at the run root;
each phase overwrote the previous one, so the stored `summary.txt` reflects the
`--phase ddp` run only. Per-phase PASS evidence for correctness and baseline is
in the child `done`/`console.log` files above and in the local driver logs
`artifacts/logs/leonardo/2026-10-05-timing-correctness.log`,
`...-baseline.log`, `...-ddp.log`.

## Provenance

| Item | Value |
|---|---|
| Date | 2026-10-05 |
| Julia | 1.12.6 |
| Project | `~/projects/flux-integration/test` |
| Flux | 0.16.11 (`pathof(Flux)` = `~/projects/flux-integration/src/Flux.jl`) |
| Optimisers / Functors | 0.4.9 / 0.5.3 |
| MPI / Zygote | 0.20.27 (MPICH via `srun --mpi=pmi2`) / 0.7.13 |
| CUDA / NCCL / cuDNN | 6.2.2 / 0.1.3 / 6.2.2 |
| Driver | 535.274.02 (4× NVIDIA A100-SXM-64GB) |
| Optimizer | native Adam, eta = 0.001 |
| Tolerances | atol = 2.0e-6, rtol = 1.0e-5 (Float32, unchanged) |
| Prepared artifact | `/leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls`, SHA-256 `7f0d38d997b14de3905ab9176cd3f914103363398c0bd9be6bf6360595373cd6`, 39804054 bytes |
| Dataset digest | `658e3dd341ca4e4fa9e5e4a05eacf53c0bebca626a05b17a87e917062dec2971` |
| Schedule digest | `ce6b41aa81c9eb9d3c4cc526a25f50e4e302738e47e660db0b4e5ad2aa78bc57` (world 4; identical to deathstar mpi4/K2) |
| Trajectory | `~/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4/nccl4/train/trajectory.jls` (148521848 bytes) |
| `conditional_graph_timing_check.jl` SHA-256 | `1116b1ed15b10c2a13bc3ae1b7fcfc0db793cc8d49cdbacc714757377451b35e` |
| `conditional_graph_timing.jl` SHA-256 | `145c21b3165a15564146ec7c61196b6d2cfe12886e72ec185418710c3572c862` |
| `conditional_graph/reference.jl` SHA-256 | `324c735ca9c86cae55e3faeeeefe5759c3c3680c57166928ff82921716604e3a` |
| `flux-integration/src/Flux.jl` SHA-256 | `78710f9b398971609ac86824cb9ce6ab2ce3ab4672139df2142d24d44bacd0ce` |
| `flux-integration/src/distributed/public_api.jl` SHA-256 | `2984210b84e136da9bc9948c3b5928877f1f6c10f870f8db841ad2d4ab6136f0` |

No `scripts/remote/precompile.sh leonardo` and no GPU-node precompile helper
were needed: every console log reports 0 `Precompiling packages` lines.
`flux-integration` was not modified, and no forward sync ran after the
pull-back.

## Interpretation and caveats

- One 120-step repeat of this small model takes well under a second, so 5
  repeats give a median with visible variance (baseline repeat 3 is 1.89× the
  min total, driven by a 0.54 s validation pass). Treat the headline as
  diagnostic.
- The baseline evaluates all 4 shards sequentially per global update; DDP
  computes 1 shard per rank concurrently and the allreduce cost is small, so
  DDP is faster here despite communication overhead. This is a property of the
  tiny per-step compute, not a general scaling claim.
- The live stream arrived in a late buffered burst (documented deviation); the
  viewer PNG is the live viewer's last frame and the pulled `ddp/records.tsv`
  is the authoritative complete record.

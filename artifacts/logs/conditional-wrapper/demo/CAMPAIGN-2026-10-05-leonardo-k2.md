# Phase K Leonardo K1/K2 campaign — leonardo — 2026-10-05 (FINAL)

Status: **COMPLETE on leonardo. All cells PASS.** No Flux source changed this session; no precompile occurred. Correctness claims apply only to the stored roots below.
K1 ran the toy CUDA/NCCL driver at 4 ranks on node `lrdn3394`; K2 ran MNIST/Fashion-MNIST CUDA/NCCL at 4 ranks on node `lrdn0436`. Driver exit 0 for both roots.
Failure and resolution record: `wiki/failures/2026-10-05-leonardo-k1-k2-nccl4.md`.
Devlog: `wiki/devlog/2026-10-05-leonardo-k1-k2-4rank-nccl.md`.

## Final cell table

All exit codes are driver exit codes. "Maxima" are the maximum per-console `peer_param_maxerr` / `peer_mom1_maxerr` / `peer_mom2_maxerr` values; K1 and K2 each report the same maxima in train and resume. Collective counts are per-console occurrences `collectives=5; collectives=7`.

| Configuration | Stage | Root | Exit | Elapsed | DEMO_STEP records (rank-rows) | Maxima (param / mom1 / mom2) | Collectives | resume_match |
|---|---|---|---|---|---|---|---|---|
| K1 toy nccl4 (16 steps) | prepared | `k1-2026-10-05-leonardo-nccl4` | 0 | 12.716 s | — | — | — | — |
| K1 toy nccl4 | train | same | 0 | 183.574 s | 64 | 2.980e-08 / 2.980e-08 / 1.164e-10 | 24×5 + 40×7 | n/a |
| K1 toy nccl4 | resume | same | 0 | 67.707 s | 56 | 2.980e-08 / 2.980e-08 / 1.164e-10 | 24×5 + 32×7 | 56 / 56 |
| K2 nccl4 (120 steps, reused prepared) | train | `k2-2026-10-05-leonardo-nccl4` | 0 | 184.657 s | 480 | 5.960e-08 / 4.657e-09 / 1.455e-11 | 160×5 + 320×7 | n/a |
| K2 nccl4 | resume | same | 0 | 77.235 s | 472 | 5.960e-08 / 4.657e-09 / 1.455e-11 | 160×5 + 312×7 | 472 / 472 |

Driver summaries:

```text
PHASE_K_SUMMARY dataset=toy backend=nccl ranks=4 PASS root=.../k1-2026-10-05-leonardo-nccl4
PHASE_K_SUMMARY dataset=mnist-fashion backend=nccl ranks=4 PASS root=.../k2-2026-10-05-leonardo-nccl4
```

Driver testsets: K1 `Phase K toy campaign (nccl) | 8 8 4m27.5s`; K2 `Phase K mnist-fashion campaign (nccl) | 6 6 4m25.6s`. Both driver runs exited 0.

The nonzero 4-rank maxima are Float32 accumulation differences within the declared tolerances (`atol=2.0e-6`, `rtol=1.0e-5`, unchanged); the 2-rank cells reported exactly zero.

## Evidence checklist

| Check | Result |
|---|---|
| `summary.txt` with exit codes in each root | present; status PASS |
| `PHASE_K_SUMMARY ... backend=nccl ranks=4 PASS` for both datasets | present |
| Driver testset totals | K1 8/8, K2 6/6; driver exit 0 |
| `DEMO_SCHEMA_OK` markers | K1 train 17 / resume 15 validation steps; K2 train 121 / resume 119 |
| `DEMO_STEP` records | K1 64 train / 56 resume; K2 480 train / 472 resume (K2 records share 362/352 console lines; counts verified with `grep -o`) |
| `resume_match=true` | K1 56 / 56; K2 472 / 472 |
| `peer_beta_exact=false` count | 0 |
| GPU maps | rank 0 -> device 1, rank 1 -> device 2, rank 2 -> device 3, rank 3 -> device 0, `devices=[1, 2, 3, 0]`, NVIDIA A100-SXM-64GB; identical in train and resume |
| Maxima inside declared tolerances | yes (see table) |
| Per-child products | every train/resume directory has `metadata.txt`, `metrics-*.tsv`, `validation-*.tsv`, `learning-rate-*.tsv`, `predictions-*.tsv`; train also has `checkpoint-step2.jls` and `trajectory.jls`; `command.txt` and `console.log` at the job level |
| K1 prepared products | `prepared/toy_prepared.jls`, `prepared/schedule-ranks4.tsv`, `prepared/prepare_metadata.txt` |
| K2 `prepared/` | absent by design (`--prepared` reuse); metadata records the prepared path, dataset digest, and SHA-256 |
| No `Precompiling packages` line in any child console | confirmed |
| Flux source identity | demo and `flux-integration` source hashes match the deathstar campaign record; no Flux source changed this session |

## Exact commands

Prepared dataset transfer (SHA verified after transfer; no download on compute nodes):

```bash
ssh leonardo 'mkdir -p /leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05'
rsync -avz --partial artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-deathstar-mpi/prepared/ leonardo:/leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/
```

Targeted source sync (no `--delete`):

```bash
rsync -avzR ./scripts/checks/conditional_graph_demo_check.jl ./scripts/checks/conditional_graph_checkpoint_roundtrip.jl ./scripts/examples/conditional_graph_demo.jl ./scripts/examples/conditional_graph/ leonardo:~/projects/ddp-flux-thesis/
```

GPU check:

```bash
scripts/remote/run.sh leonardo "srun --mpi=pmi2 -n 1 -c 2 bash -lc '/usr/bin/nvidia-smi -L'"
```

K1 toy, 4-rank CUDA/NCCL:

```bash
scripts/remote/run.sh leonardo 'srun --mpi=pmi2 -n 1 -c 8 --chdir=$HOME/projects/ddp-flux-thesis julia --startup-file=no --project=$HOME/projects/flux-integration/test scripts/checks/conditional_graph_demo_check.jl --root artifacts/logs/conditional-wrapper/demo/k1-2026-10-05-leonardo-nccl4 --dataset toy --backend nccl --ranks 4 --steps 16 --timeout 1800 --label k1-leonardo-nccl4'
```

K2 MNIST/Fashion-MNIST, 4-rank CUDA/NCCL:

```bash
scripts/remote/run.sh leonardo 'srun --mpi=pmi2 -n 1 -c 8 --chdir=$HOME/projects/ddp-flux-thesis julia --startup-file=no --project=$HOME/projects/flux-integration/test scripts/checks/conditional_graph_demo_check.jl --root artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4 --dataset mnist-fashion --backend nccl --ranks 4 --steps 120 --prepared /leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls --timeout 3600 --label k2-leonardo-nccl4'
```

Pull-back (exit 0; no forward sync afterward):

```bash
rsync -avz leonardo:~/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/demo/k1-2026-10-05-leonardo-nccl4 leonardo:~/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4 artifacts/logs/conditional-wrapper/demo/
```

Figures:

```bash
uv run --offline --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_demo.py --run artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-deathstar-mpi --run artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-deathstar-nccl2 --run artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4 --run artifacts/logs/conditional-wrapper/demo/k1-regression-2026-10-05-deathstar-mpi2 --out reports/figures/phase-k-k2
```

Local driver logs: `artifacts/logs/leonardo/2026-10-05-k1-leonardo-nccl4.log`, `artifacts/logs/leonardo/2026-10-05-k2-leonardo-nccl4.log`.

## Environment and source identity

| Item | Value |
|---|---|
| Julia | 1.12.6 |
| Project | `~/projects/flux-integration/test` |
| Flux | 0.16.11, worktree `ddp/integration-m1-spike`, HEAD `55171e65` dirty; git fields read `unavailable` on the synced tree |
| Optimisers / Functors | 0.4.9 / 0.5.3 |
| MPI / Zygote | 0.20.27 (MPICH via `srun --mpi=pmi2`) / 0.7.13 |
| CUDA / NCCL / cuDNN | 6.2.2 / 0.1.3 / 6.2.2 |
| Driver | 535.274.02 |
| GPUs | 4x NVIDIA A100-SXM-64GB |
| Optimizer | native Adam, eta = 0.001 |
| Tolerances | atol = 2.0e-6, rtol = 1.0e-5 (Float32, unchanged) |
| Warning (not fatal) | `srun: Job step's --cpus-per-task value exceeds that of job (8 > 4)` |
| `scripts/examples/conditional_graph/checkpoint.jl` SHA-256 | `d494550fbdabdfe43808d4ced48fa860da9c9412616f7be19f45f9d7eb287b4e` |
| `scripts/checks/conditional_graph_demo_check.jl` SHA-256 | `2241cb6b1c37ce8ce6f61099797e55bea5b422bcbdbb1b017d4d58843a5da488` |
| `flux-integration/src/distributed/public_api.jl` SHA-256 | `2984210b84e136da9bc9948c3b5928877f1f6c10f870f8db841ad2d4ab6136f0` |
| `ext/FluxMPIExt/FluxMPIExt.jl` SHA-256 | `24dca3c3b9472b75579abffebe245a72026f376eb5c4d2d15c5208d03ae96766` |
| `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl` SHA-256 | `08c02c5aec7ac0349d3bc0f6bae87308a900b12129873d2791f91c11458b28b8` |

Demo source hashes match the deathstar campaign record (`CAMPAIGN-2026-10-05-deathstar-k2.md`). No Flux source changed this session.

## Prepared artifacts and digests

| Run | Prepared artifact | Dataset digest | Schedule digest | Exposures |
|---|---|---|---|---|
| K1 toy nccl4 | generated in-root (`prepared/toy_prepared.jls`; schedule in `prepared/schedule-ranks4.tsv`) | — | — | 16 steps |
| K2 nccl4 | reused `/leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls`, SHA-256 `7f0d38d997b14de3905ab9176cd3f914103363398c0bd9be6bf6360595373cd6`, 39804054 bytes | `658e3dd341ca4e4fa9e5e4a05eacf53c0bebca626a05b17a87e917062dec2971` | `ce6b41aa81c9eb9d3c4cc526a25f50e4e302738e47e660db0b4e5ad2aa78bc57` (identical to deathstar mpi4) | A 6400 / B 8960 |

K2 used local batch 32, 120 steps, and native Adam `eta=0.001`; the routed objective differs from the 2-rank runs by design (steps 1-2 route A to 1 of 4 ranks, B to 3 of 4). The prepared artifact was transferred from the deathstar campaign; loader and preprocessing are unchanged (see `CAMPAIGN-2026-10-05-deathstar-k2.md`).

## Artifacts and figures

Both roots were pulled back to `artifacts/logs/conditional-wrapper/demo/` (exit 0; no forward sync afterward). K1 has 22 files; K2 has 17 files.

The figure command exited 0 and produced 41 PNGs plus `figures-metadata.txt` in `reports/figures/phase-k-k2/`, including 8 new files:

- `k2-2026-10-05-leonardo-nccl4-nccl4-validation-loss.png`
- `k2-2026-10-05-leonardo-nccl4-nccl4-validation-accuracy.png`
- `k2-2026-10-05-leonardo-nccl4-nccl4-validation-difference.png`
- `k2-2026-10-05-leonardo-nccl4-nccl4-learning-rate.png`
- `k2-2026-10-05-leonardo-nccl4-nccl4-discrepancy.png`
- `k2-2026-10-05-leonardo-nccl4-nccl4-schedule.png`
- `k2-2026-10-05-leonardo-nccl4-nccl4-predictions-task-a.png`
- `k2-2026-10-05-leonardo-nccl4-nccl4-predictions-task-b.png`

## Deviations and resolved issues

- Direct `srun ... nvidia-smi -L` failed (`srun: fatal: Can not execute nvidia-smi`); fixed with `bash -lc` and the absolute `/usr/bin/nvidia-smi`.
- Remote rsync path `$WORK` was not shell-expanded; fixed with the literal path `/leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/`.
- `uv run --with` hit a PyPI connect timeout; fixed with `--offline`.
- A full-tree rsync would have uploaded ~650 MB; replaced by the targeted source sync (no `--delete`).
- Four ranks write one `console.log`, so some records share a line; record counts were verified with `grep -o` (K2 480/472 records on 362/352 lines; K1 64/56 records). Measurement note, not a defect.

Pointer: `wiki/failures/2026-10-05-leonardo-k1-k2-nccl4.md`; resolution devlog: `wiki/devlog/2026-10-05-leonardo-k1-k2-4rank-nccl.md`.

## Out of scope / unrun

- The optional post-Phase K live one-A100 versus four-rank A100 timing stretch.
- The deferred-behavior report.

The required Phase K acceptance matrix is now complete (deathstar CPU/MPI 2+4 ranks and CUDA/NCCL 2 ranks; Leonardo CUDA/NCCL 4 ranks). No blocker remains.

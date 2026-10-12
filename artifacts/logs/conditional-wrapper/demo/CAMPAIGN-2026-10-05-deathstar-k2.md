# Phase K K2 campaign — deathstar — 2026-10-05 (FINAL)

Status: **COMPLETE on deathstar. All cells PASS.** Flux and demo source were not edited by the execution owner.
First attempt (root `k2-smoke-2026-10-05-deathstar-mpi2`) was blocked at the smoke resume by the `checkpoint_digest` `Dict`-order defect; the implementation owner fixed it, and the campaign resumed from fresh roots.
Failure and resolution record: `wiki/failures/2026-10-05-phase-k-k2-campaign.md`.

## Final cell table

All exit codes are driver exit codes. "Maxima" are the maximum per-console `peer_param_maxerr` / `peer_mom1_maxerr` / `peer_mom2_maxerr` values. Collective counts are per-console occurrences `collectives=5; collectives=7`; the driver also verified the ordered per-rank trace every step.

| Configuration | Stage | Root suffix | Exit | Elapsed | Steps (rank-rows) | Maxima (param / mom1 / mom2) | Collectives | resume_match |
|---|---|---|---|---|---|---|---|---|
| K2 smoke2 mpi2 (6 steps, 640/100) | prepared | `k2-smoke2-2026-10-05-deathstar-mpi2` | 0 | 13.068 s | — | — | — | — |
| K2 smoke2 mpi2 | train | same | 0 | 48.169 s | 12 | 0.000e+00 / 0.000e+00 / 0.000e+00 | 4×5 + 8×7 | n/a |
| K2 smoke2 mpi2 | resume | same | 0 | 46.661 s | 8 | 0.000e+00 / 0.000e+00 / 0.000e+00 | 4×5 + 4×7 | 8 / 8 |
| K1 toy regression mpi2 (16 steps) | prepared | `k1-regression-2026-10-05-deathstar-mpi2` | 0 | 11.564 s | — | — | — | — |
| K1 toy regression mpi2 | train | same | 0 | 49.170 s | 32 | 0.000e+00 / 0.000e+00 / 0.000e+00 | 12×5 + 20×7 | n/a |
| K1 toy regression mpi2 | resume | same | 0 | 47.665 s | 28 | 0.000e+00 / 0.000e+00 / 0.000e+00 | 12×5 + 16×7 | 28 / 28 |
| K2 full mpi2 (120 steps, 5120/1000) | prepared | `k2-2026-10-05-deathstar-mpi` | 0 | 13.567 s | — | — | — | — |
| K2 full mpi2 | train | same | 0 | 53.684 s | 240 | 0.000e+00 / 0.000e+00 / 0.000e+00 | 80×5 + 160×7 | n/a |
| K2 full mpi2 | resume | same | 0 | 51.176 s | 236 | 0.000e+00 / 0.000e+00 / 0.000e+00 | 80×5 + 156×7 | 236 / 236 |
| K2 full mpi4 | train | same | 0 | 56.198 s | 480 | 7.451e-09 / 9.313e-10 / 9.095e-12 | 160×5 + 320×7 | n/a |
| K2 full mpi4 | resume | same | 0 | 56.699 s | 472 | 5.960e-08 / 7.451e-09 / 7.276e-12 | 160×5 + 312×7 | 472 / 472 |
| K2 NCCL2 (120 steps, reused MPI prepared) | train | `k2-2026-10-05-deathstar-nccl2` | 0 | 80.302 s | 240 | 0.000e+00 / 0.000e+00 / 0.000e+00 | 80×5 + 160×7 | n/a |
| K2 NCCL2 | resume | same | 0 | 78.773 s | 236 | 0.000e+00 / 0.000e+00 / 0.000e+00 | 80×5 + 156×7 | 236 / 236 |
| Phase K checkpoint roundtrip regression | single process | — | 0 | — | — | — | — | — |

Driver summaries:

```text
PHASE_K_SUMMARY dataset=mnist-fashion backend=mpi ranks=2 PASS root=.../k2-smoke2-2026-10-05-deathstar-mpi2
PHASE_K_SUMMARY dataset=toy backend=mpi ranks=2 PASS root=.../k1-regression-2026-10-05-deathstar-mpi2
PHASE_K_SUMMARY dataset=mnist-fashion backend=mpi ranks=2,4 PASS root=.../k2-2026-10-05-deathstar-mpi
PHASE_K_SUMMARY dataset=mnist-fashion backend=nccl ranks=2 PASS root=.../k2-2026-10-05-deathstar-nccl2
```

Driver testset totals: smoke2 8/8, K1 regression 8/8, full MPI 13/13, NCCL 6/6; all `DEMO_SCHEMA_OK` markers present for every train and resume cell.

The mpi4 non-zero maxima are Float32/4-rank accumulation differences within the declared tolerances (`atol=2.0e-6`, `rtol=1.0e-5`); all mpi2 and NCCL maxima are exactly zero. Tolerances were not changed.

## Roundtrip regression evidence (post-fix)

Command:

```bash
scripts/remote/run.sh deathstar "/home/picazio/.juliaup/bin/julia --startup-file=no scripts/checks/conditional_graph_checkpoint_roundtrip.jl"
```

Output (exit 0):

```text
CHECKPOINT_DIGEST 51cc18f34c48e59bee3538c5f404b8bea80c9e96437e80949c6291f9cdd47429
CHECKPOINT_ROUNDTRIP_STABLE roundtrips=3
CHECKPOINT_DICT_ORDER_CANONICAL entries=11
CHECKPOINT_NESTED_DICT_CANONICAL
CHECKPOINT_MUTATION_SENSITIVE
CHECKPOINT_ROUNDTRIP_OK
```

## GPU evidence (NCCL cell)

Pre-run `nvidia-smi`: all three TITAN RTX GPUs present, driver 555.42.02, 0% utilization, 6/6/16 MiB. No other Julia/MPI processes.

```text
DEMO_GPU_MAP rank=0 device=1 name=NVIDIA TITAN RTX devices=[1, 2]
DEMO_GPU_MAP rank=1 device=2 name=NVIDIA TITAN RTX devices=[1, 2]
```

Same map in train and resume. Ranks used distinct physical devices.

## Exact commands

Sync (step 0, after moving the cache out of the synced tree):

```bash
ssh deathstar 'mv ~/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/demo/k2-cache-2026-10-05 ~/k2-cache-2026-10-05'
scripts/remote/sync_code.sh deathstar
```

Cache retained remotely at `/home/picazio/k2-cache-2026-10-05` (8 gz files); excluded from all pulls.

Campaign driver commands as executed (all through `scripts/remote/run.sh deathstar` with `env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0`):

```bash
# Smoke2
/home/picazio/.juliaup/bin/julia --startup-file=no --project=/home/picazio/projects/flux-integration/test scripts/checks/conditional_graph_demo_check.jl \
  --root artifacts/logs/conditional-wrapper/demo/k2-smoke2-2026-10-05-deathstar-mpi2 \
  --dataset mnist-fashion --backend mpi --ranks 2 --steps 6 --train-per-task 640 --val-per-task 100 \
  --cache /home/picazio/k2-cache-2026-10-05 --timeout 900 --label k2-smoke2

# K1 toy regression
/home/picazio/.juliaup/bin/julia --startup-file=no --project=/home/picazio/projects/flux-integration/test scripts/checks/conditional_graph_demo_check.jl \
  --root artifacts/logs/conditional-wrapper/demo/k1-regression-2026-10-05-deathstar-mpi2 \
  --dataset toy --backend mpi --ranks 2 --steps 16 --timeout 1800 --label k1-regression

# K2 full CPU/MPI
/home/picazio/.juliaup/bin/julia --startup-file=no --project=/home/picazio/projects/flux-integration/test scripts/checks/conditional_graph_demo_check.jl \
  --root artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-deathstar-mpi \
  --dataset mnist-fashion --backend mpi --ranks 2,4 --steps 120 \
  --cache /home/picazio/k2-cache-2026-10-05 --timeout 5400 --label k2-deathstar-mpi

# K2 CUDA/NCCL (reused the MPI prepared artifact)
/home/picazio/.juliaup/bin/julia --startup-file=no --project=/home/picazio/projects/flux-integration/test scripts/checks/conditional_graph_demo_check.jl \
  --root artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-deathstar-nccl2 \
  --dataset mnist-fashion \
  --prepared artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-deathstar-mpi/prepared/mnist-fashion_prepared.jls \
  --backend nccl --ranks 2 --steps 120 --timeout 5400 --label k2-deathstar-nccl2
```

Per child the driver runs `prepare` once (single process) and `train`/`resume` through `MPI.mpiexec()` with the active test project; no outer MPI launcher was used. No `Precompiling packages...` line appeared in any child console.

## Environment and source identity

| Item | Value |
|---|---|
| Julia | 1.12.6 (`/home/picazio/.juliaup/bin/julia`) |
| Project | `/home/picazio/projects/flux-integration/test` |
| Flux | 0.16.11, `/home/picazio/projects/flux-integration/src/Flux.jl` |
| Optimisers / Functors | 0.4.9 / 0.5.3 |
| MPI / Zygote | 0.20.27 (MPICH 5.0.2) / 0.7.13 |
| CUDA / NCCL / cuDNN | 6.2.2 / 0.1.3 / 6.2.2 |
| Optimizer | native Adam, eta = 0.001, beta = (0.9, 0.999), epsilon = 1e-8 |
| Tolerances | atol = 2.0e-6, rtol = 1.0e-5 (Float32, unchanged) |
| `flux-integration/src/distributed/public_api.jl` SHA-256 | `2984210b84e136da9bc9948c3b5928877f1f6c10f870f8db841ad2d4ab6136f0` |
| remote Flux manifest SHA-256 | `4c4eaf77633c493d8463a1a57236197423a8995f92f92eff00879475fdce5837` |
| fixed `scripts/examples/conditional_graph/checkpoint.jl` SHA-256 | `d494550fbdabdfe43808d4ced48fa860da9c9412616f7be19f45f9d7eb287b4e` |
| `scripts/checks/conditional_graph_checkpoint_roundtrip.jl` SHA-256 | `a70bcf9e8de9c5859792230bf139380c8b42e5b5d50571f4d75f65790ef13772` |

Demo source hashes reported by the full-Run train metadata (identical to local before execution):

| Path | SHA-256 |
|---|---|
| `scripts/checks/conditional_graph_demo_check.jl` | `2241cb6b1c37ce8ce6f61099797e55bea5b422bcbdbb1b017d4d58843a5da488` |
| `scripts/examples/conditional_graph/checkpoint.jl` | `d494550fbdabdfe43808d4ced48fa860da9c9412616f7be19f45f9d7eb287b4e` |
| `scripts/examples/conditional_graph/common.jl` | `9dfa74ee90c29ccfcac2b102315255791f5ad15d11ba6622e9ed060dd7355f99` |
| `scripts/examples/conditional_graph/dataset.jl` | `1681c7b7a5059267dd805ae3a38f9e48226f9ab5b8c3de521897e4e9da6b4488` |
| `scripts/examples/conditional_graph/metrics.jl` | `74f25667fd48bce2e4e0872793cd54cfeda57878e72e1be55fa2ad34d915b67c` |
| `scripts/examples/conditional_graph/model.jl` | `07de4f0fb5bed36727a49cd89f1f21bc1c198c12515cb1aa0ac32fe9fa9b67f0` |
| `scripts/examples/conditional_graph/reference.jl` | `71ac4a90d3770835a43a14aa310bad840b486be9c1d7d73c7d944a70503f9ff1` |
| `scripts/examples/conditional_graph/schedule.jl` | `9e7bb59e8494ac9dbf26d84643ce144f91e61c22115f22429a699cdb59ce99e3` |
| `scripts/examples/conditional_graph_demo.jl` | `b6e6725d51577e4bde4ac900cfbb7ba59ce156bff0321bd7b94d5a26eb99ce7d` |
| `flux-integration/src/Flux.jl` | `78710f9b398971609ac86824cb9ce6ab2ce3ab4672139df2142d24d44bacd0ce` |
| `flux-integration/src/distributed/public_api.jl` | `2984210b84e136da9bc9948c3b5928877f1f6c10f870f8db841ad2d4ab6136f0` |

## Prepared artifacts and digests

| Run | Dataset digest | Schedule digest | Prepared SHA-256 | Pools |
|---|---|---|---|---|
| smoke2 (6 steps) | `b19e4b6354f2048dab98d9c49698fd57db7515171fc14931c0781244ff11d5d8` | `684e7430269d2a400709768798fa3e1d82e1db0b363532d5d48afef42feb05ee` | `3b1df1d5edaf99513162c7f8dbe4702ba8da75fb483e6f5a0c9a852365910130` | 640 / 100 |
| K1 toy regression | toy digest `55a8204c50f7f5fe03f046cb39490e9f641f262c6723c40cc90e1108eddf6ee0` | `517d03e040d2b7909766708fcdd11cc05b506a29e1a379048ab4e08844c0542b` | — | 512 / 128 |
| full mpi2 | `658e3dd341ca4e4fa9e5e4a05eacf53c0bebca626a05b17a87e917062dec2971` | `f27e2ef87e10d9dfe63f99f62fad82d58db5d54309c9b621bcdbb8496e539280` | `7f0d38d997b14de3905ab9176cd3f914103363398c0bd9be6bf6360595373cd6` | 5120 / 1000 |
| full mpi4 | `658e3dd341ca4e4fa9e5e4a05eacf53c0bebca626a05b17a87e917062dec2971` | `ce6b41aa81c9eb9d3c4cc526a25f50e4e302738e47e660db0b4e5ad2aa78bc57` | same prepared artifact | 5120 / 1000 |
| NCCL2 | reuses full mpi2 prepared artifact | `f27e2ef87e10d9dfe63f99f62fad82d58db5d54309c9b621bcdbb8496e539280` | `7f0d38d997b14de3905ab9176cd3f914103363398c0bd9be6bf6360595373cd6` | 5120 / 1000 |

Loader: `official-idx-gzip-v1; Julia Downloads; external gzip`; preprocessing `IDX row-major flatten; Float32(byte)/255; no augmentation`; seeds train 101, val 202, init 303, order 404, subset 505; task map `a = MNIST digits`, `b = Fashion-MNIST clothing`.

## Artifacts

All four roots were pulled back to `artifacts/logs/conditional-wrapper/demo/` (exit 0 each; raw cache excluded). The earlier failed root `k2-smoke-2026-10-05-deathstar-mpi2` remains as historical evidence. No forward sync was run after the pull.

- `k2-smoke2-2026-10-05-deathstar-mpi2/` (24 files)
- `k1-regression-2026-10-05-deathstar-mpi2/` (22 files; the toy run has no `sources.tsv`/`subset_indices.tsv` by design, generated dataset)
- `k2-2026-10-05-deathstar-mpi/` (41 files; includes `prepared/mnist-fashion_prepared.jls`, both rank-count schedules and all train/resume products)
- `k2-2026-10-05-deathstar-nccl2/` (17 files; no `prepared/` by design, `--prepared` reuse)

Every root contains `summary.txt` with exit codes; every train/resume directory contains `metadata.txt`, `metrics-*.tsv`, `validation-*.tsv`, `learning-rate-*.tsv`, `predictions-*.tsv`, and a console plus `command.txt`; each train also contains `checkpoint-step2.jls` and `trajectory.jls`. NCCL `metadata.txt` records CUDA 6.2.2, NCCL 0.1.3, and cuDNN 6.2.2.

## Out of scope / unrun

- Leonardo 4-rank/K1 and any Leonardo K2 cells (not part of this deathstar campaign).
- Plotting and the presentation report (plotting owner).
- The optional post–Phase K live Leonardo timing stretch.

No blocker remains for the deathstar bounded campaign.

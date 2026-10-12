# Leonardo Phase K: 4-rank CUDA/NCCL K1 and K2

Date: 2026-10-05
Host: `leonardo` (CINECA), 4× NVIDIA A100-SXM-64GB, driver 535.274.02;
K1 node `lrdn3394`, K2 node `lrdn0436`
Flux worktree: `~/projects/flux-integration` at `ddp/integration-m1-spike`
HEAD `55171e65` dirty (git fields read `unavailable` on the synced tree); no
Flux source changed this session
Julia: 1.12.6, project `~/projects/flux-integration/test`
Predecessor: [Leonardo bring-up: 4-rank native NCCL validation](2026-10-04-leonardo-bringup-4rank-nccl.md)
Campaign record: `artifacts/logs/conditional-wrapper/demo/CAMPAIGN-2026-10-05-leonardo-k2.md`
Failure record: `wiki/failures/2026-10-05-leonardo-k1-k2-nccl4.md`

## Outcome

The two remaining Phase K acceptance cells PASS on Leonardo at 4 ranks with
CUDA/NCCL, completing the required Phase K matrix (deathstar CPU/MPI 2 and 4
ranks and CUDA/NCCL 2 ranks; Leonardo CUDA/NCCL 4 ranks). Both drivers exited
0 and printed `PHASE_K_SUMMARY ... backend=nccl ranks=4 PASS`.

- K1 toy: `Phase K toy campaign (nccl) | 8 8 4m27.5s`.
- K2 MNIST/Fashion-MNIST: `Phase K mnist-fashion campaign (nccl) | 6 6 4m25.6s`.
- Tolerances unchanged (`atol=2.0e-6`, `rtol=1.0e-5`); the nonzero 4-rank
  maxima are Float32 accumulation differences inside them.
- No Flux source changed; no precompile occurred (0 `Precompiling packages`
  lines).

Evidence roots (pulled back, exit 0, no forward sync afterward):
`artifacts/logs/conditional-wrapper/demo/k1-2026-10-05-leonardo-nccl4` (22
files) and `.../k2-2026-10-05-leonardo-nccl4` (17 files). Local driver logs:
`artifacts/logs/leonardo/2026-10-05-k1-leonardo-nccl4.log` and
`artifacts/logs/leonardo/2026-10-05-k2-leonardo-nccl4.log`.

## Per-cell results

All exit codes are driver exit codes. "Maxima" are the maximum per-console
`peer_param_maxerr` / `peer_mom1_maxerr` / `peer_mom2_maxerr` values; K1 and
K2 report the same maxima in train and resume. Collective counts are
per-console occurrences of `collectives=5; collectives=7`.

| Cell | Stage | Root | Ranks | Exit | Elapsed | `DEMO_STEP` records | `DEMO_SCHEMA_OK` validation steps | Maxima (param / mom1 / mom2) | Collectives | `resume_match` |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 toy nccl4 (16 steps) | prepared | `k1-2026-10-05-leonardo-nccl4` | 1 | 0 | 12.7 s | — | — | — | — | — |
| K1 toy nccl4 | train | same | 4 | 0 | 183.6 s | 64 | 17 | 2.980e-08 / 2.980e-08 / 1.164e-10 | 24×5 + 40×7 | n/a |
| K1 toy nccl4 | resume | same | 4 | 0 | 67.7 s | 56 | 15 | 2.980e-08 / 2.980e-08 / 1.164e-10 | 24×5 + 32×7 | 56 / 56 |
| K2 nccl4 (120 steps, reused prepared) | train | `k2-2026-10-05-leonardo-nccl4` | 4 | 0 | 184.657 s | 480 | 121 | 5.960e-08 / 4.657e-09 / 1.455e-11 | 160×5 + 320×7 | n/a |
| K2 nccl4 | resume | same | 4 | 0 | 77.235 s | 472 | 119 | 5.960e-08 / 4.657e-09 / 1.455e-11 | 160×5 + 312×7 | 472 / 472 |

Driver summaries:

```text
PHASE_K_SUMMARY dataset=toy backend=nccl ranks=4 PASS root=.../k1-2026-10-05-leonardo-nccl4
PHASE_K_SUMMARY dataset=mnist-fashion backend=nccl ranks=4 PASS root=.../k2-2026-10-05-leonardo-nccl4
```

Every rank used a distinct A100 in both cells: rank 0→1, 1→2, 2→3, 3→0,
`devices=[1, 2, 3, 0]`, identical in train and resume. `peer_beta_exact=false`
count is 0 in both cells.

## Exact commands

K1 toy, 4-rank CUDA/NCCL:

```bash
scripts/remote/run.sh leonardo 'srun --mpi=pmi2 -n 1 -c 8 --chdir=$HOME/projects/ddp-flux-thesis julia --startup-file=no --project=$HOME/projects/flux-integration/test scripts/checks/conditional_graph_demo_check.jl --root artifacts/logs/conditional-wrapper/demo/k1-2026-10-05-leonardo-nccl4 --dataset toy --backend nccl --ranks 4 --steps 16 --timeout 1800 --label k1-leonardo-nccl4'
```

K2 MNIST/Fashion-MNIST, 4-rank CUDA/NCCL:

```bash
scripts/remote/run.sh leonardo 'srun --mpi=pmi2 -n 1 -c 8 --chdir=$HOME/projects/ddp-flux-thesis julia --startup-file=no --project=$HOME/projects/flux-integration/test scripts/checks/conditional_graph_demo_check.jl --root artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4 --dataset mnist-fashion --backend nccl --ranks 4 --steps 120 --prepared /leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls --timeout 3600 --label k2-leonardo-nccl4'
```

Sync was a targeted rsync of only the demo sources (no `--delete`), because a
full-tree sync would upload about 650 MB of prior artifacts:

```bash
rsync -avzR ./scripts/checks/conditional_graph_demo_check.jl ./scripts/checks/conditional_graph_checkpoint_roundtrip.jl ./scripts/examples/conditional_graph_demo.jl ./scripts/examples/conditional_graph/ leonardo:~/projects/ddp-flux-thesis/
```

Pull-back (exit 0; no forward sync afterward):

```bash
rsync -avz leonardo:~/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/demo/k1-2026-10-05-leonardo-nccl4 leonardo:~/projects/ddp-flux-thesis/artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4 artifacts/logs/conditional-wrapper/demo/
```

Figures (exit 0; regenerated 41 PNGs plus `figures-metadata.txt`, including 8
new `k2-2026-10-05-leonardo-nccl4-nccl4-*` files):

```bash
uv run --offline --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_demo.py --run artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-deathstar-mpi --run artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-deathstar-nccl2 --run artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4 --run artifacts/logs/conditional-wrapper/demo/k1-regression-2026-10-05-deathstar-mpi2 --out reports/figures/phase-k-k2
```

The prepared artifact was transferred from deathstar; it was not downloaded on
a compute node.

## Environment and source identity

| Item | Value |
|---|---|
| Julia | 1.12.6 |
| Project | `~/projects/flux-integration/test` |
| Flux | 0.16.11, worktree `ddp/integration-m1-spike` HEAD `55171e65` dirty; git fields read `unavailable` on the synced tree |
| Optimisers / Functors | 0.4.9 / 0.5.3 |
| MPI / Zygote | 0.20.27 (MPICH via `srun --mpi=pmi2`) / 0.7.13 |
| CUDA / NCCL / cuDNN | 6.2.2 / 0.1.3 / 6.2.2 |
| Driver | 535.274.02 |
| GPUs | 4× NVIDIA A100-SXM-64GB |
| Optimizer | native Adam, eta = 0.001 |
| Tolerances | atol = 2.0e-6, rtol = 1.0e-5 (Float32, unchanged) |
| `scripts/examples/conditional_graph/checkpoint.jl` SHA-256 | `d494550fbdabdfe43808d4ced48fa860da9c9412616f7be19f45f9d7eb287b4e` |
| `scripts/checks/conditional_graph_demo_check.jl` SHA-256 | `2241cb6b1c37ce8ce6f61099797e55bea5b422bcbdbb1b017d4d58843a5da488` |
| `flux-integration/src/distributed/public_api.jl` SHA-256 | `2984210b84e136da9bc9948c3b5928877f1f6c10f870f8db841ad2d4ab6136f0` |
| `ext/FluxMPIExt/FluxMPIExt.jl` SHA-256 | `24dca3c3b9472b75579abffebe245a72026f376eb5c4d2d15c5208d03ae96766` |
| `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl` SHA-256 | `08c02c5aec7ac0349d3bc0f6bae87308a900b12129873d2791f91c11458b28b8` |

No Flux source changed this session; the synced tree reports git fields as
`unavailable`.

## Prepared artifact and K2 schedule

K2 reused the prepared artifact transferred from the deathstar campaign (no
download on compute nodes):

- `/leonardo_work/IscrC_UNMASKED/epicazio/phase-k-prepared-2026-10-05/mnist-fashion_prepared.jls`,
  SHA-256 `7f0d38d997b14de3905ab9176cd3f914103363398c0bd9be6bf6360595373cd6`,
  39804054 bytes.
- Dataset digest `658e3dd341ca4e4fa9e5e4a05eacf53c0bebca626a05b17a87e917062dec2971`.
- Schedule digest `ce6b41aa81c9eb9d3c4cc526a25f50e4e302738e47e660db0b4e5ad2aa78bc57`
  (rank-4 schedule, identical to deathstar mpi4).
- Exposures a=6400 / b=8960, local batch 32, 120 steps, Adam eta=0.001.

K1 generated its prepared products in-root (`prepared/toy_prepared.jls`,
`prepared/schedule-ranks4.tsv`).

## Deviations

All resolved; details in the failure record
`wiki/failures/2026-10-05-leonardo-k1-k2-nccl4.md`:

- Direct `srun ... nvidia-smi -L` failed; rerun as
  `srun ... bash -lc '/usr/bin/nvidia-smi -L'`.
- The remote rsync path `$WORK` was not expanded by the remote shell; the
  literal `/leonardo_work/...` path was used.
- `uv run --with matplotlib --with numpy` hit a PyPI connect timeout; rerun
  with `--offline` against the warm uv cache (matplotlib 3.11.2, numpy 2.5.3).
- A full-tree rsync was aborted (about 650 MB) and replaced by the targeted
  sync above.
- Benign warning `srun: Job step's --cpus-per-task value exceeds that of job
  (8 > 4)`.
- Four ranks write one `console.log`, so some `DEMO_STEP` records share a line;
  counts were verified with `grep -o` (K2 480/472 records on 362/352 lines;
  K1 64/56 records). Measurement note, not a defect.

## Remaining cells

- Optional post-Phase K live one-A100 versus four-rank A100 timing stretch.
- Deferred-behavior report.

The required Phase K acceptance matrix is complete.

## Next action

Run the optional live one-A100 versus four-rank A100 timing stretch and write
the deferred-behavior report.

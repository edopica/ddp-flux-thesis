# Phase K K2: task-conditioned MNIST/Fashion-MNIST implementation and evidence

Date: 2026-10-05

Scope: implement and run the K2 task-conditioned MNIST/Fashion-MNIST
presentation (prompt section 8) and produce the required figures and supervisor
report. No Flux production or test source changed; all runs use the wrapper at
`55171e65` through `--project=../flux-integration/test` on `deathstar`. Code and
demo files were owned by the implementation and plotting owners; this session
changed documentation only.

## Delivered

- K2 mode in the existing harness
  (`scripts/examples/conditional_graph_demo.jl`,
  `scripts/examples/conditional_graph/{common,model,dataset,schedule,reference,checkpoint,metrics}.jl`,
  driver `scripts/checks/conditional_graph_demo_check.jl`): official IDX
  gzip loader `official-idx-gzip-v1`, balanced 5120/1000 subsets per task, the
  six-step routing cycle with globally absent heads, independent native Adam
  reference over the saved multi-rank schedule, per-step results, and data-only
  schema-2 Serialization checkpoints.
- Campaign evidence: `artifacts/logs/conditional-wrapper/demo/CAMPAIGN-2026-10-05-deathstar-k2.md`
  (cell table, driver summaries, exact commands, source hashes, GPU map).
- Run roots: `k2-smoke2-2026-10-05-deathstar-mpi2/`,
  `k1-regression-2026-10-05-deathstar-mpi2/`,
  `k2-2026-10-05-deathstar-mpi/` (mpi2 and mpi4),
  `k2-2026-10-05-deathstar-nccl2/`.
- Presentation: `scripts/plots/plot_phase_k_demo.py`, `scripts/plots/README.md`,
  `reports/figures/phase-k-k2/` (33 PNGs plus `figures-metadata.txt`),
  `reports/conditional-graph-phase-k.md`.

## Results (all deathstar, all summaries PASS)

- K2 full mpi2 (120 steps): train/resume exit 0, 53.684 s / 51.176 s; maxima
  param/mom1/mom2 exactly 0.0; `resume_match` 236/236.
- K2 full mpi4 (120 steps): exit 0, 56.198 s / 56.699 s; maxima train
  7.451e-09 / 9.313e-10 / 9.095e-12 and resume 5.960e-08 / 7.451e-09 /
  7.276e-12, inside the declared Float32 `atol=2e-6, rtol=1e-5`;
  `resume_match` 472/472.
- K2 NCCL 2 ranks on TITAN RTX (rank devices `[1, 2]`, driver 555.42.02),
  reusing the MPI prepared artifact: 80.302 s / 78.773 s; maxima exactly 0.0;
  `resume_match` 236/236.
- K1 toy regression mpi2 (16 steps): all cells exit 0, maxima 0.0,
  `resume_match` 28/28, toy validation accuracy 1.000/1.000 at the final step.
- Driver totals: smoke2 8/8, full MPI 13/13, K1 regression 8/8, NCCL 6/6.
- Final validation (initial digits 0.1170, clothing 0.0460): mpi2 and NCCL
  digits acc 0.8560 loss 0.6935, clothing acc 0.7270 loss 0.7968 (exposure
  7680/7680); mpi4 digits acc 0.8470 loss 0.7658 (exposure 6400), clothing acc
  0.7390 loss 0.7740 (exposure 8960). mpi2 and NCCL trajectories are
  identical; mpi4 differs because steps 1-2 route A to 1 of 4 ranks and B to
  3 of 4, as the prompt anticipated.
- Per-step checks every step: parameters, both Adam moments, decay products,
  rules/frozen/shape/dtype/device, local and global AD absence, equal update
  counts, rank-vs-reference and rank-vs-rank comparison; ordered collective
  traces per rank every step (5 collectives when one head is globally absent,
  7 otherwise); only rank 0 writes shared files; checkpoints are data-only and
  the loaded state is asserted in a fresh process before the first resumed
  update.

## Defect found and fixed

The first K2 smoke aborted the fresh-process resume:
`CGCheckpoint.checkpoint_digest` hashed a `Dict` whose `Serialization`
iteration order changed across a save/load roundtrip. The payload values were
correct (`DDP_SNAP_MATCH=true`, `REF_SNAP_MATCH=true`). The fix canonicalizes
`AbstractDict` (sorted by `string(key)`), `NamedTuple`, `Tuple`, and
`AbstractVector` before hashing; the field list and schema version are
unchanged. Regression
`scripts/checks/conditional_graph_checkpoint_roundtrip.jl` was RED pre-fix and
GREEN locally and on `deathstar` (exit 0, `CHECKPOINT_ROUNDTRIP_OK`). Pre-fix
checkpoints are intentionally incompatible. Failure/resolution record:
`wiki/failures/2026-10-05-phase-k-k2-campaign.md`.

## Figures

`scripts/plots/plot_phase_k_demo.py` validates the stored TSV schema strictly
and writes deterministic PNGs only from those TSVs. The output includes
reference-overlay validation loss/accuracy, difference panels, configured Adam
eta, log-scale discrepancy panels (`beta_exact` true), prediction panels,
schedule/presence/exposure panels with globally absent shading and the step-2
checkpoint boundary, and a cross-configuration comparison. Reproducible command:

```bash
uv run --with matplotlib --with numpy python3 scripts/plots/plot_phase_k_demo.py \
  --run artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-deathstar-mpi \
  --run artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-deathstar-nccl2 \
  --run artifacts/logs/conditional-wrapper/demo/k1-regression-2026-10-05-deathstar-mpi2 \
  --out reports/figures/phase-k-k2
```

## Environment

Julia 1.12.6; Flux 0.16.11 from `../flux-integration` (branch
`ddp/integration-m1-spike`, HEAD `55171e65` dirty) via
`--project=../flux-integration/test`; Optimisers 0.4.9, Functors 0.5.3,
MPI 0.20.27 (MPICH 5.0.2), Zygote 0.7.13, CUDA 6.2.2, NCCL 0.1.3, cuDNN 6.2.2.
Prepared logs record the resolved Flux path, package versions, and source
hashes; each run directory has `metadata.txt`. The demo enforces a Flux-root
guard. Exact commands and per-cell timings are in
`artifacts/logs/conditional-wrapper/demo/CAMPAIGN-2026-10-05-deathstar-k2.md`.

## Limits and unrun cells

- Leonardo 4-rank K1/K2 correctness cells: unrun (no authorization this
  session).
- Optional post-Phase K live one-process/single-A100 versus four-rank/four-A100
  timing comparison: unrun/pending.
- Deferred-behavior report: pending.
- The campaign timings are diagnostic, not a speedup claim; no four-GPU host
  was available.

## Exact next action

Request Leonardo authorization for the 4-rank K1/K2 correctness cells. When
authorized: sync the fixed demo sources, prefetch the datasets, precompile on
GPU nodes, and run `scripts/checks/conditional_graph_demo_check.jl` with fresh
roots at 4 ranks.

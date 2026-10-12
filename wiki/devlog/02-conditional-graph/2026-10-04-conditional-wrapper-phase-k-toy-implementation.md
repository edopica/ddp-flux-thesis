# Phase K toy case: implementation and CPU/NCCL evidence

Date: 2026-10-04

Scope: implement and validate the deterministic toy case (K1) from
`temp/conditional_graph_phase_k_prompt.md`. No Flux production or test source
changed. All runs use the wrapper at `55171e65` through the resolved workspace
environment `--project=../flux-integration/test` (root `[workspace]
projects = ["test", "docs"]`; no `test/Manifest.toml` is needed).

## Delivered files

- `scripts/examples/conditional_graph_demo.jl` — CLI: `prepare`, `train`,
  `resume`; `--backend mpi|nccl`, `--out`, `--prepared`, `--checkpoint`,
  `--expected`, `--save-at`, `--save-trajectory`, `--atol/--rtol`, `--label`.
- `scripts/examples/conditional_graph/common.jl` — tolerances, snapshot and
  comparison helpers, recording backend with local/global presence flags,
  ordered collective-trace checker, metadata helpers.
- `scripts/examples/conditional_graph/model.jl` — shared-trunk `TwoHeadModel`
  (`Dense(2=>16, tanh)`, `Dense(16=>2)` per head) and Float64→Float32/device
  conversion.
- `scripts/examples/conditional_graph/dataset.jl` — seeded two-task toy pools
  (512 train / 128 val per task), recorded centers, noise, and dataset digest.
- `scripts/examples/conditional_graph/schedule.jl` — saved schedule with the
  six-step routing cycle, equal local batch sizes, no padding, schedule digest.
- `scripts/examples/conditional_graph/reference.jl` — independent single-process
  native `Optimisers` global-batch reference; unused head absent from the AD
  graph; never calls wrapper internals.
- `scripts/examples/conditional_graph/checkpoint.jl` — versioned plain-data
  `Serialization` payloads with identity digest.
- `scripts/checks/conditional_graph_demo_check.jl` — bounded driver: per-child
  watchdog, exit codes, per-rank `DEMO_STEP`/`DEMO_RANK_DONE` counts, done
  markers, `resume_match=true` counts, `summary.txt`, nonzero exit on any
  failing cell.

## Contract values used

- Dataset digest `55a8204c50f7f5fe03f046cb39490e9f641f262c6723c40cc90e1108eddf6ee0`.
- Schedule digests: R=2 `517d03e040d2b7909766708fcdd11cc05b506a29e1a379048ab4e08844c0542b`;
  R=4 `d677a30a782429357a7d6f27b18e0d152902eeae9e610eb560fb7ba47d58cb19`.
- 16 updates; global batch 128; local 64 (R=2) / 32 (R=4); six-step cycle
  (1/2/5 mixed, 3 all-A, 4 all-B, 6 swapped mixed); expected wrapper
  collectives 7 when both heads are globally present, 5 otherwise.
- Tolerances: CPU Float64 `atol=1e-12 rtol=1e-10`; GPU Float32
  `atol=2e-6 rtol=1e-5`.
- Optimizer `Adam eta=1e-2, beta=(0.9, 0.999), epsilon=1e-8`.

## Decision: toy learning rate 1e-2 (documented deviation)

The prompt's numerical defaults are recommendations. With `eta=1e-3` the
prepared seed starts head B on a fully inverted boundary, and the 16-update
budget leaves B at accuracy 0.000 for both R=2 and R=4 (measured with an
independent single-process native run; loss decreases but the boundary never
flips). A measured sweep (16 updates, native reference) gave B accuracy
0.875/0.906 at `eta=5e-3` and 1.000/1.000 at `eta=1e-2` for R=2/R=4. The
default is now `1e-2`, which demonstrates learning for both heads inside the
bounded budget. This is a toy-hyperparameter choice, not a wrapper property.

## Decision: checksum convention

`checkpoint_digest` covers identity fields only
(`schema_version, step_next, world_size, backend_name, dtype,
model_definition_version, optimizer, schedule_digest, dataset_digest`).
Digests are therefore comparable only within the same backend/config: local
R=2 MPI `e97c726de6d2c19fbfa886792eb9d44c4f7c6c144319f41d119bc5b7702fbae2`,
deathstar R=2 NCCL `834b7671544e2b3b2ab45c5021051a3b4d2ba6a1d37372f16859dd9f411ad4d2`.
The digest is an integrity check, not a cross-backend identity claim.

## Commands

Local CPU/MPI matrix (final source):

```bash
julia --startup-file=no --project=../flux-integration/test \
  scripts/checks/conditional_graph_demo_check.jl \
  --root artifacts/logs/conditional-wrapper/demo/toy-2026-10-04-local-mpi \
  --backend mpi --ranks 2,4 --label toy-local-mpi
```

deathstar CUDA/NCCL (after targeted rsync of the demo scripts):

```bash
scripts/remote/run.sh deathstar \
  "julia --startup-file=no --project=/home/picazio/projects/flux-integration/test \
   scripts/checks/conditional_graph_demo_check.jl \
   --root artifacts/logs/conditional-wrapper/demo/toy-2026-10-04-deathstar-nccl2 \
   --backend nccl --ranks 2 --label toy-deathstar-nccl2"
```

GPU preflight: 3× TITAN RTX healthy (driver 555.42.02), warm test-env load
6.9 s with no precompilation; GPU mapping rank 0 → device 1, rank 1 → device 2.
The remote workspace root `Manifest.toml` differs from local
(md5 `3c13d7692bced27ede1527ef414515f2` vs `038810fd622670aaa62937c7a9505c9f`);
both resolve the identical root/test `Project.toml`
(`9a2957cd...` / `ca91c533...`) and wrapper sources
(`public_api.jl 9bfa9813...`, `FluxMPIExt ba022e51...`,
`FluxMPINCCLExt 330999ce...`).

## Results

Local matrix PASS, 9/9 driver assertions
(`artifacts/logs/conditional-wrapper/demo/toy-2026-10-04-local-mpi/`):
prepare 11.5 s; R=2 train 51.2 s / resume 48.6 s; R=4 train 64.7 s /
resume 61.7 s. Per-step comparison against the independent native reference:
maximum parameter error exactly 0.0 (R=2) and 1.11e-16 (R=4); maximum moment
errors ≤5.6e-17 (R=4); `beta_exact=true` on all 60 train/resume rows.
Continuation matched the saved trajectory on 100% of steps.

deathstar 2-rank NCCL PASS, 6/6 driver assertions
(`.../toy-2026-10-04-deathstar-nccl2/`): prepare 9.5 s, train 64.7 s,
resume 63.7 s. Per-step parameter, moment-1, and moment-2 errors are exactly
0.0 on all 16 train and 14 resume steps (the reference runs in Float32 on the
same device); `beta_exact=true` on all rows. Collective trace: 20 steps × 7
collectives and 12 steps × 5 collectives = 32 per rank, at exactly the
scheduled presence pattern. Task-A validation accuracy 1.000 throughout;
task-B accuracy 0.000 → 0.047 → 0.516 → 0.922 → 0.961 → 1.000 by step 11.
Fresh-process resume marked `resume_match=true` on all 28 step markers.

Environment: Julia 1.12.6, Flux 0.16.11, Optimisers 0.4.9, Functors 0.5.3,
MPI 0.20.27, Zygote 0.7.13; deathstar also CUDA 6.2.2, NCCL 0.1.3, cuDNN
functional (Phase J versions, warm cache re-verified today).

First NCCL attempt failed in GPU-only demo code
(`UndefVarError: CUDA not defined in Main.CGCommon`): recorded with cause and
fix in `wiki/failures/2026-10-04-phasek-nccl-cuda-ref.md`; the failed console
is kept under `toy-2026-10-04-deathstar-nccl2-attempt1/`.

## Limits

- K2 (task-conditioned MNIST/Fashion-MNIST presentation, 120 updates) is not
  implemented or run; the toy validates the harness and the wrapper protocol.
- Rank counts tested: CPU/MPI 2 and 4, NVIDIA/NCCL 2 (the accepted GPU
  configuration; 4 GPUs are unavailable on both hosts).
- The exact-zero NCCL agreement is a property of this tiny deterministic model
  and its same-device Float32 reference, not a general bitwise-reproducibility
  claim.

## Exact next action

Implement K2: extend the prepared dataset/model to the paired
MNIST/Fashion-MNIST presentation run (120 updates, rank-0 metrics panels,
fresh-process continuation) and run it through the same driver at the
validated configurations.

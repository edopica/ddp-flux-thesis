# Static-network GPU scaling: shared experiment specification (Phase 1)

## Phase 4 GPU rule and immediate timing scope — 2026-10-18

The user approved finite `abs(a-b)<1e-3` for Phase 4 FP32 comparisons, with no relative allowance.
Structural, hash, counter, shard, protocol, and rank consistency checks remain exact.
The explicit checker option is `--fp32-absolute-1e3`. CPU validation is deferred.
This rule supersedes the Phase 4 CUDA trajectory overrides, not the historical Phase 3 result record.
The [Phase 4 report](../../artifacts/logs/static-scaling/2026-10-18-phase4/RESULTS.md) records the actual acceptance evidence.

The immediate GPU timing pilot uses global batch 12, two alternating saved fixture batches, and W=1,2,4.
It retains the existing update-indexed learning-rate schedule and strict FP32 precision helper.
The pilot measures training-only duration, images/s, and update time after excluded warmup and state reset.
The larger-batch and full-dataset requirements in this specification remain subsequent campaign requirements.

## Original specification record

Date: 2026-10-15 (review findings R1-R4 corrected 2026-10-16; OD-1, OD-5, OD-6 recorded 2026-10-16; OD-10 recorded 2026-10-17; OD-2, OD-3, OD-4, OD-11 recorded 2026-10-18)
Status: architecture, initialization, normalization, augmentation, optimizer, schedule, cross-framework tolerance, Flux entry point, and OD-14 benchmark policy are recorded. OD-7, OD-8, OD-9, OD-12, and OD-13 remain proposed in [open-decisions.md](open-decisions.md).
Review: [Phase 1 implementation review](../../reports/static-scaling-phase1-review.md); R1 (classifier axis), R2 (kernel orientation), R3 (AdamW constructor), R4 (beta-power mapping) corrected in both companions.
Scope: define one workload that the Flux and PyTorch implementations can each reproduce.

Companions:

- Machine-readable config: [experiment.toml](experiment.toml).
- Pending decisions: [open-decisions.md](open-decisions.md).
- Plan: [implementation phases](../../temp/static-graph-gpu-scaling-implementation-phases.md) (section 3 is this phase).
- Proposal: [detailed proposal](../../reports/static-graph-gpu-scaling-proposal-detailed.md).
- Environment: [Leonardo setup devlog](../../wiki/devlog/03-static-scaling/2026-10-15-leonardo-benchmark-env-setup.md), [Leonardo context](../../wiki/context/leonardo.md).

No experiment has run. This document fixes rules and records sources; it states no result.

## 0. Phase 1 task map

| Phase 1 task | Section |
|---|---|
| 1. Architecture record | 4 |
| 2. AdamW hyperparameters | 6 |
| 3. Weight decay for biases and normalization parameters | 6.2 |
| 4. Learning-rate schedule by global update | 6.3 |
| 5. Mean cross-entropy and label conversion | 5 |
| 6. Preprocessing, augmentation, RGB decoding, validation | 3 |
| 7. Shared format for parameters, inputs, labels, gradients, optimizer state | 8 |
| 8. Tolerances | 11 |
| 9. Batch candidates and memory margin | 9 |
| 10. CPU budget, workers, threads, affinity | 7 |
| 11. Source pinning | 10 |
| 12. Flux entry point and guard behavior | 6.4, 10.4 |

## 1. Experiment matrix and fixed work

One Leonardo node, four A100-SXM-64GB GPUs, at one, two, and four GPUs for each framework.
Every configuration uses the same global batch, the same initial parameters, the same global sample order, and the same number of optimizer updates.

| Framework | GPUs | Processes | Local batch at global 512 | Local batch at global 256 |
|---|---:|---:|---:|---:|
| Flux | 1 | 1 | 512 | 256 |
| Flux | 2 | 2 | 256 | 128 |
| Flux | 4 | 4 | 128 | 64 |
| PyTorch | 1 | 1 | 512 | 256 |
| PyTorch | 2 | 2 | 256 | 128 |
| PyTorch | 4 | 4 | 128 | 64 |

The global batch is **proposed 512 with 256 as the Phase 5 fallback**. The schedule never scales with GPU count.

## 2. Data schedule invariants

- 100,000 training images; 200 classes; 500 images per class.
- Effective samples per epoch: 99,840. The same final 160 images of the saved epoch permutation are dropped in every configuration.
- Updates per epoch: 195 at global batch 512; 390 at global batch 256.
- One saved global permutation per epoch. Global batches are formed from it before rank sharding.
- Each global batch splits into equal, nonoverlapping contiguous shards; rank `r` receives shard `r`.
- Reconstructed global batches must match the saved order at one, two, and four ranks.
- Independent rank shuffling and sampler padding must never change the totals above.

## 3. Input pipeline

### 3.1 Decoding

Each JPEG is decoded exactly once during dataset preparation with one canonical decoder; the saved uint8 arrays are the shared input for both frameworks.
This removes PIL-versus-libjpeg differences from the comparison entirely and makes decoded-input equality testable.
The decoder identity and version are recorded with the prepared dataset; JPEG decode cost is outside both timing views (recorded limitation, [open decision OD-9](open-decisions.md)).

### 3.2 Normalization

Rule (recorded, OD-5, 2026-10-16): `x = (x_uint8 / 255 - mean_c) / std_c`, computed in float32, per channel, with the same constants applied to training and validation inputs.
`mean_c` and `std_c` are computed once from all 100,000 original training images using the canonical decoded uint8 data scaled to [0,1]: the per-channel RGB mean, and the population standard deviation (divide by N) over all training pixels of that channel, with float64 accumulation.
Statistics are computed before augmentation, padding, and epoch tail dropping; validation and test images are never used to estimate them.
The float32 constants and this calculation method are frozen in the fixture manifest (Phase 2).

### 3.3 Augmentation

Training transform set (recorded, OD-6, 2026-10-16): zero-padding of 4 pixels on each side (64x64 -> 72x72), then a uniformly sampled 64x64 crop, then a horizontal flip with probability 0.5.
`crop_top` and `crop_left` are zero-based integers in `0..8`, sampled independently per axis.
The order is fixed: padding, crop, and flip applied to the decoded uint8 image; then conversion to float32, division by 255, and normalization.
Validation applies conversion to float32, division by 255, and normalization only (no crop, no flip, no resize).

Critical rule: augmentation decisions are **saved data**, not fresh randomness.
One `(crop_top, crop_left, flip)` triple is drawn per (epoch, global sample index) and saved.
Both frameworks replay the same record; the record does not depend on rank or loader worker count.
A framework must never re-draw its own augmentation randomness.

### 3.4 Validation

The official validation split is evaluated with its flat layout and `val/val_annotations.txt` labels.
The validation procedure aggregates loss sums, correct-prediction counts, and sample counts over the 10,000 validation images; a sample is counted exactly once.
Validation runs outside timing windows.

## 4. Model architecture

Name (recorded, OD-1, 2026-10-16): **Tiny ImageNet ResNet-18, small-image stem, GroupNorm-32.**
It is an adaptation of the documented model, not a BatchNorm reproduction.

| Component | Specification | Output per sample |
|---|---|---|
| Input | RGB, native resolution | 64x64x3 |
| Stem | 3x3 conv, 64 channels, stride 1, padding 1, no bias | 64x64x64 |
| Stem normalization | GroupNorm(32), affine, eps 1e-5, then ReLU | 64x64x64 |
| Initial max-pool | none | - |
| Stage 1 | 2 BasicBlocks, 64 channels, stride 1, no projection | 64x64x64 |
| Stage 2 | 2 BasicBlocks, 128 channels; first block stride 2 with 1x1 projection | 32x32x128 |
| Stage 3 | 2 BasicBlocks, 256 channels; first block stride 2 with 1x1 projection | 16x16x256 |
| Stage 4 | 2 BasicBlocks, 512 channels; first block stride 2 with 1x1 projection | 8x8x512 |
| Pooling | global spatial average | 512 |
| Classifier | linear 512 -> 200, with bias | 200 logits |

BasicBlock: `conv3x3 -> GN -> ReLU -> conv3x3 -> GN -> residual add -> ReLU`.
All convolutions are bias-free. All 3x3 convolutions use padding 1; all 1x1 convolutions use padding 0.
Flux implements every convolution as `CrossCor` (cross-correlation), which matches the PyTorch `Conv2d` operator, and the classifier as `Dense(512 => 200)`, whose weight keeps shape `(out, in) = (200, 512)`. Section 8 defines the parameter conversion rules.
GroupNorm is affine, has 32 groups, eps 1e-5, `gamma = 1`, `beta = 0`, and is applied to the stem, both residual convolutions of every block, and every projection path.
Groups are contiguous channel blocks of size `channels / 32`; all stage widths are divisible by 32.

Parameter count and decomposition (verified by an independent count on 2026-10-15):

| Component | Parameters |
|---|---:|
| Stem 3x3 conv | 1,728 |
| Stage 1 (2 blocks, no projection) | 147,456 |
| Stage 2 (2 blocks, one 1x1 projection) | 524,288 |
| Stage 3 (2 blocks, one 1x1 projection) | 2,097,152 |
| Stage 4 (2 blocks, one 1x1 projection) | 8,388,608 |
| Convolution weights (total) | 11,159,232 |
| GroupNorm affine (20 layers x gamma/beta) | 9,600 |
| Classifier weight and bias | 102,600 |
| **Trainable total** | **11,271,432** |

The total also follows from TorchVision ResNet-18 (11,689,512 at 1,000 outputs) by the proposal's formula:
`11,689,512 - (512+1)(1000-200) - 64*3*(7^2-3^2) = 11,271,432`.
A different measured count requires an explained architecture correction before acceptance.

Trainable arrays for the collective plan: 20 convolution weights + 40 GroupNorm arrays + 2 classifier arrays = **62 unique arrays**.

### 4.1 Initial parameters

Recipe (recorded, OD-1, 2026-10-16): convolution weights Kaiming-normal with `fan_out` mode and ReLU gain; classifier weight and bias follow the PyTorch `Linear` defaults; GroupNorm `gamma = 1`, `beta = 0`.
One canonical bundle is generated once; the generator, its version, its seed, the recipe, and the bundle hash are recorded in the fixture manifest (Phase 2).
Both frameworks load this same bundle; framework default initializers are never used at load time.

## 5. Loss and labels

- Mean-reduced cross-entropy over the local batch; inputs are raw logits.
- Shared labels are int64, 0-based class indices; the class index is the position in `wnids.txt`.
- Julia conversion: add 1 at load, then build one-hot rows with `onehotbatch(y, 1:200)`.
- Python conversion: use the shared 0-based labels directly with `CrossEntropyLoss`.
- Reduction: sum over the local batch divided by the local batch size.
- With equal local batch sizes, the averaged local gradients equal the gradient of the global-batch mean loss.

## 6. Optimizer and schedule

### 6.1 AdamW equations (identical in both frameworks)

For update index `t`:

```
m     = b1 * m + (1 - b1) * g
v     = b2 * v + (1 - b2) * g^2
mhat  = m / (1 - b1^t)
vhat  = v / (1 - b2^t)
dx    = mhat / (sqrt(vhat) + eps)
x     = x - eta * (dx + lambda * x)
```

The weight decay is applied through the same learning-rate factor (`eta * lambda * x`). Epsilon sits outside the square root, added to `sqrt(vhat)`.
Bias correction is on. `amsgrad` is off. There is no gradient clipping and no maximization.

Library realization, verified against the pinned sources:

| Framework | Setting | Source |
|---|---|---|
| Flux | `Optimisers.AdamW(; eta, beta = (0.9, 0.999), lambda, epsilon, couple = true)`; the all-keyword form is required (the positional constructor accepts only `couple` as a keyword and raises `MethodError`); `couple = true` yields exactly the equations above | Optimisers 0.4.9 (manifest git-tree `b6a586b5`), `src/rules.jl`, `AdamW.apply!` |
| PyTorch | `torch.optim.AdamW(params, lr, betas, eps, weight_decay)` | PyTorch 2.9 docs |

Defaults together: `eta = 1e-3`, `beta = (0.9, 0.999)`, `eps = 1e-8`, `lambda = 1e-4` (recorded 2026-10-18 as OD-2; user-approved).
The Flux `Adam` rule alone is not used; `OptimiserChain(Adam, WeightDecay)` is not equivalent to `AdamW` and must not be substituted.

### 6.2 Parameter groups

| Group | Members | Weight decay |
|---|---|---|
| Decay | 20 convolution weights, classifier weight | `lambda = 1e-4` |
| No decay | 40 GroupNorm `gamma`/`beta` arrays, classifier bias | `lambda = 0` |

Recorded 2026-10-18 (OD-3; user-approved): the decay group has 21 leaves and the no-decay group 41 leaves, with `lambda = 1e-4` and `lambda = 0` respectively.

PyTorch realization: `weight_decay` applies only to parameters with `ndim >= 2` (equivalent for this architecture).
Flux realization: an Optimisers rule tree built with Functors-compatible traversal, assigning `AdamW(lambda = 0)` to the 1-D arrays.
Phase 3 must assert group membership and per-leaf `lambda`; library defaults alone do not establish this.

### 6.3 Schedule (function of the global update index)

Identical in both frameworks; it does not scale with GPU count.

- Warm-up: linear from 0 to `eta_max` over the first `W` updates: `eta_t = eta_max * t / W`.
- Decay: cosine from `eta_max` to 0 over updates `W+1 .. T`: `eta_t = eta_max * 0.5 * (1 + cos(pi * (t - W) / (T - W)))`.

Recorded values (2026-10-18, OD-4; user-approved): 5 warm-up epochs and a 50-epoch learning-curve budget, hence:

| Global batch | `W` (5 epochs) | `T` (50 epochs) |
|---|---:|---:|
| 512 | 975 | 9,750 |
| 256 | 1,950 | 19,500 |

Throughput runs execute updates `1..K` of the same schedule; `K` is fixed in Phase 6 so the fastest configuration has tens of seconds of work.

### 6.4 Gradient averaging

`g = sum_r(g_r) / world_size` for local mean losses, applied exactly once by the framework:

- MPI: `MPI.Allreduce!` with `+`, then division by `MPI.Comm_size` (`ext/FluxMPIExt/FluxMPIExt.jl`).
- NCCL: native `NCCL.avg` (`ext/FluxMPINCCLExt/FluxMPINCCLExt.jl`).
- PyTorch: native DDP averaging.

No user-side loss scaling by `world_size` is allowed; a double division silently changes the objective.

## 7. Resource policy

Proposed and identical for all six configurations:

| Item | Policy |
|---|---|
| Total CPU budget | 32 physical cores per job |
| CPUs per task | 32 (one GPU), 16 (two GPUs), 8 (four GPUs) |
| Compute threads | 1 per rank: `JULIA_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)` |
| Data-loader workers | 4 per rank |
| Affinity | Slurm default binding for `--cpus-per-task`; no manual pinning; Phase 6 records `srun --cpu-bind=verbose` |
| GPU allocation | `--gres=gpu:4` for every configuration; all four GPUs visible |
| Rank-to-device | Flux `mod1(r + 2, ndevices)`, observed `[1, 2, 3, 0]`; PyTorch mirrors the same mapping; single-GPU runs use device 0 |
| Launch | `srun --mpi=pmi2` (MPICH over PMI2); `salloc` stays on the login node; use `srun --chdir` instead of `cd && srun` |

## 8. Shared tensor exchange format

Container: one `.npz` bundle plus one JSON manifest per fixture set (proposed).
Julia reads TOML with the standard library and NPZ with NPZ.jl; Python reads TOML with `tomllib` and NPZ with numpy (both installed in the pinned environments).

Array rules:

- Floats in float32; raw decoded images in uint8; labels and step counters in int64.
- Canonical layout is PyTorch-native: images `NCHW`; convolution kernels `(out, in, kh, kw)`; linear weights `(out, in)`; all C-order.
- Flux conversion, images: `permutedims(x, (4, 3, 2, 1))` gives `WHCN`; the inverse export uses the same permutation (it is self-inverse).
- Flux conversion, convolution kernels: `permutedims(w, (4, 3, 2, 1))` gives `(kw, kh, in, out)`, used with `Flux.CrossCor` (cross-correlation, `F=true`), which reproduces the PyTorch `Conv2d` operator. A plain `Flux.Conv` computes true convolution and would additionally require reversing both spatial kernel axes; that variant is not used.
- Flux conversion, linear weights: none. `Flux.Dense` already stores `(out, in)` and computes `weight * input` (`flux-integration/src/layers/basic.jl:189-199`), so the classifier weight is copied unchanged.
- Gradients and optimizer moments follow the layout of the array they belong to and use the same rules; under the `CrossCor` rule no kernel reversal is required in either direction.
- Parameter names follow the model path in both frameworks; the manifest maps shared names to both paths.
- Phase 2 asserts every conversion against saved fixtures, including one rectangular linear layer and one asymmetric kernel with an asymmetric input; shape checks alone cannot expose an orientation defect.

Required array set per correctness fixture: decoded input batch, labels, initial parameters, input gradients, updated parameters, optimizer moments `m` and `v`, and the completed-update counter.
Optimizer-state mapping: the shared counter is the int64 number of completed optimizer updates `s`; PyTorch `state['step']` equals `s`, and `s` is compared exactly, separately from float tolerances. Flux `m`/`v` correspond to PyTorch `exp_avg`/`exp_avg_sq`.
Flux stores no separate counter: its third state array is the beta-power pair, initialized to `beta` before the first update and evolved by float32 multiplication, so after `s` completed updates it equals `beta^(s+1)` (the exponent used by the next update's bias correction). Verification checks `beta_power ≈ beta .^ (s + 1)` within the cross-framework float tolerance; bitwise equality with a freshly computed power is not required. Continuation from an exchanged state reconstructs the pair by starting from `beta` and multiplying by `beta` exactly `s` times in float32 (the bit-exact route).

Manifest fields: producer, creation date, source hashes, and one entry per array with name, dtype, shape, layout, and SHA-256 of the raw C-order bytes.
Integrity is checked per array; the `.npz` archive bytes themselves need not be reproducible.

## 9. Memory policy

- Capacity per GPU: 64 GiB (A100-SXM-64GB).
- Required margin (proposed): peak device memory used by the process stays at or below 80% of capacity (at least 12.8 GiB free).
- Counters are recorded separately: peak live tensor memory, allocator-reserved memory, device-level process memory.
  PyTorch: `torch.cuda.max_memory_allocated`, `torch.cuda.max_memory_reserved`, `mem_get_info`/nvidia-smi delta.
  Julia: `CUDA.used_memory`, `CUDA.cached_memory`, `CUDA.pool_status`, nvidia-smi delta.
- An unavailable counter is recorded as unavailable; a different counter must never silently replace it.
- Device staging memory is measured and reported separately from model and optimizer memory.
- Pilot sequence: batch 512 on one GPU first; then two and four GPUs; on margin failure the whole matrix repeats at batch 256.

## 10. Source pinning

### 10.1 Flux

| Item | Value |
|---|---|
| Worktree | `/home/kurapica/Projects/ddp_flux/flux-integration` |
| Branch | `ddp/integration-m1-spike` (not pushed) |
| HEAD | `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e` |
| State | clean: no tracked modifications, no untracked files |
| Patch identity | `git diff HEAD | sha256sum` = `e3b0c442...` (empty); untracked-list sha256 = `e3b0c442...` (empty) |
| Guard content | `public_api.jl` and `FluxMPINCCLExt.jl` match the ADR-0009 corrected hash table; all four guard files are pinned by sha256 in experiment.toml and re-checked before runs |

The commit `f7b4520c` implements the ADR-0009 post-reduction gradient guard; its parent `314aef20` implements the ADR-0008 loss guard. Both are committed; the tree is no longer dirty. The patch-identity commands remain part of the run record because a dirty tree can appear again.

### 10.2 Julia environment

| Item | Value |
|---|---|
| Julia | 1.12.6 on `$WORK/epicazio/julia-1.12.6`; depot `$WORK/epicazio/julia-depot` |
| Project | `flux-integration/test/Project.toml`; workspace root `flux-integration/Manifest.toml` |
| Remote manifest | sha256 `4c4eaf77633c493d8463a1a57236197423a8995f92f92eff00879475fdce5837`, `julia_version` 1.12.6, project hash `08d133fc...` |
| Local manifest | sha256 `4f0582cf...`; stale (no CUDA/NCCL/cuDNN entries); must not be used for benchmark runs |

### 10.3 Python environment

| Item | Value |
|---|---|
| Venv | `$WORK/epicazio/venvs/pytorch-scaling`, CPython 3.11.17 |
| Freeze | `$VENV/requirements-freeze.txt` (28 lines); local record `artifacts/logs/leonardo/2026-10-15-pytorch-env-freeze.txt`, sha256 `cd39f282...` |
| Pins | torch 2.9.1+cu126, torchvision 0.24.1+cu126, numpy 2.4.6, pillow 12.3.0, nccl 2.27.5, cudnn 9.10.2.21, triton 3.5.1 |

### 10.4 Flux training entry point and guard behavior

Selected entry point (recorded 2026-10-18, OD-11; user-approved): explicit two-phase training — `Zygote.withgradient` for differentiation, then `Optimisers.update!` on the distributed state. All communication happens inside the wrapper after AD; no collective runs inside the differentiated region.

Actual guard behavior on this path (verified against source):

- `_sync_gradients` (`flux-integration/src/distributed/public_api.jl:745-802`) runs: one presence allreduce of host Float32 flags; one `avg` allreduce per globally present buffer in ascending model order; then a post-reduction finiteness scan and one all-rank validity vote; only then the gradient tree is rebuilt and the inner optimizer runs.
- Any rank that finds a non-finite reduced buffer makes every rank throw `DomainError` with the `_NONFINITE_GRADIENT_ERROR` message; parameters and optimizer state are unchanged; no rank updates early.
- The ADR-0008 loss vote (`Train._loss_update_allowed`, `public_api.jl:739-743`, called from `src/train.jl:385-389`) runs only on the eager `Flux.train!`/`trainstep!` path. The explicit path does not invoke it; a non-finite loss is expected to surface as a non-finite gradient, where the gradient guard rejects it.
- The NCCL extension overrides the device-buffer scan with a stream synchronization and device reduction (`ext/FluxMPINCCLExt/FluxMPINCCLExt.jl:87-96`); guard metadata stays on the host by design.
- Collective count per explicit update: 1 presence + 62 gradient allreduces + 1 gradient vote = **64**. The eager path would add one loss vote (65). All guards stay inside the measured workload.

PyTorch side: native DDP with `find_unused_parameters = False`, `static_graph = False`, default bucketing and overlap, no extra rejection contract. This semantic difference is recorded, not removed; the final report must attribute any performance gap to the complete implementations, not to NCCL alone.

All ranks must call every collective in the same order; the wrapper enforces a fixed ascending order and the tests pin the protocol.

### 10.5 Remote state warning

The Leonardo copy of `flux-integration` is a **pre-guard snapshot** (verified 2026-10-15: `src/train.jl` has no `_loss_update_allowed`, the NCCL extension is the pre-guard hash, no non-finite test files).
Before any Phase 3 or later remote run:

1. Sync the local clean `f7b4520c` tree (do not use `scripts/remote/sync_code.sh` unchanged: it points at `ddp_flux`).
2. Resolve and instantiate on the login node.
3. Precompile on a compute node (never the login node).
4. Re-record the patch identity and manifest hash in the run metadata.

### 10.6 Precision policy (accepted 2026-10-18)

FP32 is used with TF32 disabled. The reusable policy in
`scripts/static_scaling/precision.jl` requests explicit `CUDNN_FMA_MATH` and
installs the FMA-aware `math_mode` mapping and convolution chooser methods.
Strict FMA is fail-closed: a non-FMA descriptor or the absence of a successful
FMA candidate raises `PrecisionPolicyError`; there is no non-FMA fallback.

Supported package versions are CUDA.jl `6.2.2` and cuDNN.jl `6.2.2`. A loaded
cuDNN library version is recorded separately from the Julia package versions.
`configure_precision!` must run before convolution work in every process;
running a convolution before it is configured raises `PrecisionPolicyError`.
Unsupported package versions are rejected unless the clearly marked diagnostic
`allow_unsupported_versions=true` override is used, which warns and records
`versions_supported=false`. The version and library policy evidence is in the
[R1/R2 acceptance-review response](../../reports/static-scaling-phase3-acceptance-review-response.md).

## 11. Acceptance tolerances

These rules cover short-horizon correctness. They do not define long-run benchmark equivalence.
Phase 4 GPU acceptance subsequently adopted finite `abs(a-b) < 1e-3`, with exact structural and rank checks.
That Phase 4 rule supersedes the historical same-framework default for its accepted GPU fixture only.

Rule: elementwise `|a - b| <= atol + rtol * max(|a|, |b|)`.

| Comparison class | atol | rtol | Basis |
|---|---:|---:|---|
| Same framework: DDP vs single-GPU reference; rank replicas | 2.0e-6 | 1.0e-5 | existing project standard (Phase K, C8 matrix) |
| Cross framework: Flux vs PyTorch on shared fixtures (base rule) | 1.0e-5 | 1.0e-4 | recorded 2026-10-17 (OD-10); applies to every category except the three overrides below |
| Cross framework, `phase3.grads.u1` (frozen override) | 1.0e-3 | 1.0e-4 | PyTorch CPU-vs-GPU same-framework control fails the base rule at 3.021732e-4 (14,365 violations); four-comparison envelope max 4.126653e-4; >=2.42x margin |
| Cross framework, `phase3.grads.u3` (frozen override) | 2.0e-3 | 1.0e-4 | same-framework control 4.099458e-4 (128,170 violations); envelope max 7.480253e-4; >=2.67x margin |
| Cross framework, `phase3.m.u3` (frozen override) | 2.0e-4 | 1.0e-4 | same-framework control 5.057361e-5 (1,137 violations); envelope max 9.655068e-5; >=2.07x margin |

OD-10 was recorded on 2026-10-17. The three overrides apply only to
`phase3.grads.u1`, `phase3.grads.u3`, and `phase3.m.u3`; all other
categories remain at the base `1e-5` / `1e-4`. The calibration evidence is
[`artifacts/logs/static-scaling/2026-10-17-phase3-numerics/RESULTS.md`](../../artifacts/logs/static-scaling/2026-10-17-phase3-numerics/RESULTS.md).
The decisive control is the same-framework PyTorch CPU-vs-GPU comparison
with identical generator, fixture, bundle, updates, and provenance: it
fails the base rule on exactly these three backward categories, with the
same worst leaves. The maximum Flux residual is within 1.63x of that
same-framework envelope, and each new absolute floor has at least 2x margin
over the observed envelope. The residual is consistent with FP32
backend/algorithm differences between valid backends, including possible
sensitivity near nonlinear boundaries. The same-framework control supports
that explanation but does not isolate every residual to backward-kernel
rounding. Per-leaf relative L2 stays a supplementary diagnostic and does not
replace the elementwise rule; the zero/nonzero check compares whole-leaf L2
norms, not individual elements.

Categories covered: logits, loss, gradients, updated parameters, optimizer moments `m` and `v`, step counters.
Integer counters are compared exactly; the tolerances above apply only to floating-point arrays.
A tolerance revision requires a documented numerical cause (kernel or reduction-order difference) recorded in [experiment.toml](experiment.toml) before the acceptance evidence is generated.
Acceptance evidence includes multiple optimizer updates, covering nonzero moments and the schedule.

### 11.1. OD-14 benchmark acceptance — recorded 2026-10-19

The normative decision is [OD-14: benchmark policy and numerical claims](OD-14-benchmark-acceptance.md).
The primary claim is qualified descriptive throughput and strong scaling under explicit policies.
Flux retains FMA-only candidate selection and its timed gradient guard.
PyTorch retains deterministic cuDNN, disabled benchmarking, and its native DDP guard behavior.
Both frameworks retain FP32 with TF32 disabled.

Short-horizon correctness gates remain separate from long-run diagnostics.
Long-run diagnostics require matched work, finite complete state, exact replica agreement, and consistent update state.
They report same-framework and cross-framework discrepancies separately, with no numerical-equivalence threshold.
The report retains `NUMERICAL_EQUIVALENCE_CLAIM none`.
Neither the 300-update envelope nor the short-horizon tolerances become long-run acceptance bounds.

Numerical-equivalence and comparable-training-quality claims require prospective criteria and separate acceptance evidence.
The final Phase 6 cross-framework report requires a deterministic-mode sensitivity in both frameworks.
Qualified descriptive results and the Phase 5 capacity pilot can precede that sensitivity.
Every campaign records loaded runtime identity, source and input hashes, algorithm policies, guard placement, resources, and timer semantics.
Recorded source drift requires relevant correctness evidence before acceptance transfers to that source.

## 12. Open decisions and change control

The open decisions are in [open-decisions.md](open-decisions.md): final batch (OD-7), memory margin (OD-8), JPEG-decode exclusion (OD-9), update count (OD-12), and CPU/thread/worker policy (OD-13).
OD-1, OD-5, and OD-6 were recorded on 2026-10-16; OD-10 on 2026-10-17; OD-2, OD-3, OD-4, and OD-11 on 2026-10-18. Their values appear in sections 3, 4, 6, 10.4, and 11 and in the companion config.
OD-14 was recorded on 2026-10-19 in section 11.1. OD-7, OD-8, OD-9, OD-12, and OD-13 remain open.

Change rule: a change to any value in [experiment.toml](experiment.toml) that can affect gradients, sample exposure, or measured work requires an update of this specification and, where acceptance evidence already exists, renewed correctness checks.

## Sources

1. [Detailed proposal](../../reports/static-graph-gpu-scaling-proposal-detailed.md) (architecture, batch, memory estimates, CPU policy).
2. [Implementation phases](../../temp/static-graph-gpu-scaling-implementation-phases.md).
3. [Leonardo setup devlog](../../wiki/devlog/03-static-scaling/2026-10-15-leonardo-benchmark-env-setup.md) (dataset, Python pins, paths).
4. [Leonardo context](../../wiki/context/leonardo.md) (hardware, MPI, GPU mapping, storage).
5. [ADR-0008](../../wiki/decisions/ADR-0008-coordinated-nonfinite-loss-rejection.md), [ADR-0009](../../wiki/decisions/ADR-0009-post-reduction-nonfinite-gradient-rejection.md) (guards).
6. Pinned sources: Optimisers 0.4.9 `src/rules.jl` (AdamW, Adam, WeightDecay); `flux-integration/src/layers/normalise.jl` + NNlib `src/normalization.jl` (GroupNorm grouping, eps placement); CUDA.jl 6.x `CUDACore` math modes (`math_mode!`, `PEDANTIC_MATH`) with cuBLAS/cuDNN mappings.
7. PyTorch 2.9 documentation: [AdamW](https://docs.pytorch.org/docs/2.9/generated/torch.optim.AdamW.html), [GroupNorm](https://docs.pytorch.org/docs/2.9/generated/torch.nn.GroupNorm.html), [CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html), [DistributedDataParallel](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html).
# Campaign batch decision — 2026-10-19

OD-7 now fixes the global batch at 256 for the full-dataset campaign.
This decision supersedes earlier proposed batch-512 defaults in this document.
Local batches are 256, 128, and 64 at W=1, 2, and 4.
One measured epoch contains 390 updates and 99,840 images.
The learning-rate schedule uses 1,950 warmup updates and 19,500 total updates, as recorded in OD-4.
The nine-configuration memory preflight remains required before timing.

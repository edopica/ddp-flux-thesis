# Open decisions: static-network GPU scaling (Phase 1)

Date: 2026-10-15 (OD-1, OD-5, OD-6 recorded 2026-10-16)
Status: OD-1, OD-5, and OD-6 are recorded (see the decision record); the remaining items still need a recorded decision before the phase they block.
Proposed defaults live in [experiment.toml](experiment.toml) and [specification.md](specification.md); this file tracks what is not yet frozen.

## Decision table

| ID | Decision | Proposed default | Why it matters | Blocks |
|---|---|---|---|---|
| OD-1 | Architecture confirmation — **recorded 2026-10-16** | Small-image ResNet-18, 3x3 stride-1 stem, no max-pool, GroupNorm-32 eps 1e-5, biased 200-class classifier, 11,271,432 parameters | Any change alters gradients, memory, and measured work | Phase 2, 3 |
| OD-2 | AdamW hyperparameters | `lr = 1e-3`, `wd = 1e-4`, `betas = (0.9, 0.999)`, `eps = 1e-8` | Sets the learning dynamics for all six configurations | Phase 3, 8 |
| OD-3 | Weight-decay exclusions | No decay for GroupNorm `gamma`/`beta` and classifier bias; decay only for conv and classifier weights | Couples to optimization quality and to the PyTorch/Flux group implementations | Phase 3 |
| OD-4 | LR schedule and learning-curve budget | Linear warm-up over 5 epochs, cosine to 0 over a 50-epoch budget (`T` = 9,750 at batch 512, 19,500 at 256) | Fixes the training budget and every schedule position | Phase 3, 8 |
| OD-5 | Input normalization constants — **recorded 2026-10-16** | Per-channel RGB population mean and standard deviation from all 100,000 original training images (decoded uint8 scaled to [0,1], float64 accumulation), computed before augmentation, padding, and tail dropping; float32 constants frozen in the fixture manifest | Changes effective conditioning of the first layers | Phase 2 |
| OD-6 | Augmentation set — **recorded 2026-10-16** | Zero-padding of 4 per side, uniformly sampled 64x64 crop (zero-based offsets 0..8, independent), horizontal flip p=0.5; `(crop_top, crop_left, flip)` saved per (epoch, sample_id) | Affects accuracy trajectory and end-to-end timing; the saved-decision mechanism itself is fixed | Phase 2 |
| OD-7 | Final global batch | 512, fallback 256 if the memory pilot fails the margin | Batch is the fixed-work unit of the whole comparison | Phase 5 |
| OD-8 | Memory margin | Peak process memory at most 80% of 64 GiB (at least 12.8 GiB free) | Defines pass/fail for the memory pilot | Phase 5 |
| OD-9 | End-to-end timing excludes JPEG decode | Decoded uint8 arrays are prepared once with one canonical decoder and consumed by both frameworks | Removes decoder differences but narrows the meaning of "end-to-end" | Phase 6 |
| OD-10 | Cross-framework tolerance | Provisional `atol = 1e-5`, `rtol = 1e-4`, calibrated on frozen fixtures in Phase 3 | Defines cross-framework acceptance; too loose hides defects, too tight blocks acceptance | Phase 3 |
| OD-11 | Flux entry point | Explicit `Zygote.withgradient` + `Optimisers.update!` (64 collectives/update; the eager path adds a loss vote, 65) | Changes the measured work and which guard runs | Phase 3, 6 |
| OD-12 | Throughput update count `K` | Chosen in Phase 6: tens of seconds of work for the fastest configuration | Must be identical in all configurations | Phase 6 |
| OD-13 | CPU/thread/worker policy | 32 CPUs per job total, 1 compute thread per rank, 4 loader workers per rank, Slurm default affinity | Required for interpreting end-to-end scaling and avoiding oversubscription | Phase 6, 7 |

## Notes per decision

**OD-1.** The architecture follows the detailed proposal and is fully specified in the spec. The parameter count 11,271,432 was independently recomputed (convolution 11,159,232 + GroupNorm 9,600 + classifier 102,600) and equals the proposal's formula applied to TorchVision ResNet-18. **Recorded 2026-10-16:** GroupNorm-32 and the removed max-pool are confirmed, and the initial-parameter recipe is confirmed (Kaiming-normal `fan_out` with ReLU gain, PyTorch `Linear` defaults for the classifier, GN `gamma = 1`/`beta = 0`). One canonical bundle is generated once with a recorded generator, generator version, and seed, and both frameworks load that same bundle; the bundle hash and recipe are recorded in the Phase 2 fixture manifest.

**OD-2/OD-3.** The proposal's original TorchVision recipe (SGD momentum 0.9, lr 0.2, wd 1e-4) does not transfer to AdamW. The proposed AdamW values are a new, documented choice; both frameworks use the identical rule. Verified library facts: Optimisers 0.4.9 `AdamW` with the default `couple = true` implements `x = x - eta*(dx + lambda*x)` exactly like `torch.optim.AdamW`; `OptimiserChain(Adam, WeightDecay)` is **not** equivalent and must not be substituted. Weight-decay exclusions follow standard practice and map to `ndim >= 2` on the PyTorch side.

**OD-4.** The schedule is defined by global update index (equations in the spec), not by wall-clock or rank. The 50-epoch budget is a proposed learning-curve budget; the published BatchNorm/SGD numbers of the reference recipe are context only and are not acceptance thresholds for GroupNorm/AdamW.

**OD-5/OD-6.** The proposal requires "matching preprocessing and augmentation decisions, indexed by sample and epoch rather than rank". The mechanism is fixed now: one saved decision record per (epoch, global sample index), replayed by both frameworks. **Recorded 2026-10-16:** the training transform set is zero-padding of 4 per side, then a uniformly sampled 64x64 crop (zero-based `crop_top`/`crop_left` in `0..8`, sampled independently), then a horizontal flip at probability 0.5; the `(crop_top, crop_left, flip)` triple is saved per (epoch, sample_id). The normalization constants are per-channel RGB population mean and standard deviation computed once from all 100,000 original training images, on the canonical decoded uint8 data scaled to [0,1], before augmentation, padding, and epoch tail dropping, with float64 accumulation; validation and test images are never used. The constants are frozen as float32 values in the fixture manifest and applied identically by both frameworks. A different seed does not produce matching augmentation across frameworks; the saved record does.

**OD-7/OD-8.** The proposal's activation estimates are planning assumptions, not measured limits. The pilot measures forward, backward, communication, guard, and optimizer memory at one, two, and four GPUs, and the margin decides pass/fail. If both candidate batches fail, the experiment requires a documented revision before the campaign.

**OD-9.** JPEG decode once, at preparation time, removes PIL-versus-Julia decoder differences and allows exact decoded-input equality tests. The cost is that the end-to-end view measures augmentation and host-to-device staging but not raw JPEG decode. Both frameworks are treated identically, and the limitation must be stated in the final report.

**OD-10.** Same-framework tolerances stay at the project standard (2e-6 / 1e-5). Cross-framework values must be calibrated on the frozen fixtures before any Phase 3 acceptance run and then frozen in the config; changing a tolerance later requires a documented numerical cause.

**OD-11.** The explicit path is the project's established benchmark path (Phase K driver), keeps communication outside AD, and keeps the ADR-0009 gradient guard inside the timed workload. The eager `Flux.train!` path additionally runs the ADR-0008 loss vote; if the project prefers the fullest guard contract in the benchmark, choose the eager path instead and record 65 collectives per update.

**OD-12/OD-13.** The phases document requires one fixed update count for timing and a fixed total CPU budget with recorded worker and thread counts. The proposed policy keeps total CPU at 32 cores for every configuration and one compute thread per rank; the pilot and the timing dry run record the effective thread and affinity settings.

## Decision record

When a decision is made, add a row here and update the affected files.

| Date | Decision | Value | Recorded by | Files updated |
|---|---|---|---|---|
| 2026-10-16 | OD-1 | Small-image ResNet-18 as specified (3x3 stride-1 stem, no initial max-pool, affine GroupNorm-32 eps 1e-5, biased 200-class classifier) frozen at 11,271,432 trainable parameters across 62 arrays; convolution weights Kaiming-normal `fan_out` with ReLU gain; classifier weight and bias use the PyTorch `Linear` defaults; GN `gamma = 1`, `beta = 0`; one canonical bundle with recorded generator, version, and seed loaded by both frameworks | supervisor | specification.md, experiment.toml |
| 2026-10-16 | OD-5 | Per-channel RGB population mean and standard deviation from all 100,000 original training images, canonical decoded uint8 scaled to [0,1], float64 accumulation, computed before augmentation, padding, and epoch tail dropping; float32 constants and the calculation method frozen in the fixture manifest; validation and test images excluded; the same rule applies to training and validation | supervisor | specification.md, experiment.toml |
| 2026-10-16 | OD-6 | Zero-padding of 4 per side, then a uniformly sampled 64x64 crop (zero-based `crop_top`/`crop_left` in `0..8`, sampled independently), then a horizontal flip with probability 0.5; `(crop_top, crop_left, flip)` saved per (epoch, sample_id), independent of rank and loader worker; padding/crop/flip applied to decoded uint8, then float32 conversion, division by 255, and normalization; validation uses conversion and normalization only | supervisor | specification.md, experiment.toml |

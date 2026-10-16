# OD-1, OD-5, OD-6 recorded (Phase 2 unblocked)

Date: 2026-10-16
Scope: static-network GPU scaling, Phase 1 decisions
Status: three decisions recorded; Phase 2 can start

## Decisions recorded

The supervisor recorded the three decisions that blocked Phase 2. Recorded
values, as entered in `open-decisions.md`:

- **OD-1 (architecture and initialization).** Small-image ResNet-18: 3x3
  stride-1 stem, no initial max-pool, affine GroupNorm with 32 groups and eps
  1e-5, biased 200-class classifier; frozen at 11,271,432 trainable parameters
  across 62 arrays. Convolution weights use Kaiming-normal initialization with
  `fan_out` mode and ReLU gain; classifier weight and bias use the
  PyTorch/TorchVision `Linear` defaults; GroupNorm `gamma = 1`, `beta = 0`. One
  canonical initial-parameter bundle is generated once, and its generator,
  generator version, and seed are recorded; both frameworks load that same
  bundle.
- **OD-5 (normalization statistics).** Per-channel RGB mean and standard
  deviation computed once from all 100,000 original training images, using the
  canonical decoded uint8 data scaled to [0,1]. Computed before augmentation,
  padding, and epoch tail dropping. Population standard deviation over all
  training pixels of each channel, with float64 accumulation. Validation and
  test images are never used. The constants and the calculation method are
  frozen in the fixture manifest; both frameworks apply the same float32
  constants to training and validation inputs.
- **OD-6 (augmentation set).** Zero-padding of 4 pixels on each side
  (64x64 -> 72x72), then a uniformly sampled 64x64 crop, then a horizontal
  flip with probability 0.5. `crop_top` and `crop_left` are zero-based
  integers in `0..8`, sampled independently. The `(crop_top, crop_left, flip)`
  triple is saved per (epoch, sample_id), independent of rank and loader
  worker. Padding, crop, and flip apply to the decoded uint8 image; then
  conversion to float32, division by 255, and normalization. Validation uses
  conversion and normalization only, without augmentation or resizing.

## Files updated

| File | Change |
|---|---|
| `experiments/static-scaling/open-decisions.md` | Header status; recorded markers and values in the decision table; OD-1 and OD-5/OD-6 notes; three rows added to the decision record |
| `experiments/static-scaling/specification.md` | Header; section 3.2 normalization rule; section 3.3 augmentation set and record format; section 4 name; section 4.1 initialization recipe; section 12 remaining decisions |
| `experiments/static-scaling/experiment.toml` | `[model]` status and `initial_parameters`; `[data_pipeline]` normalization status/rule/constants and augmentation status/sets/record/application |

## Verification

```bash
python3 -c "import tomllib; tomllib.load(open('experiments/static-scaling/experiment.toml','rb')); print('TOML OK')"
```

Result: `TOML OK`.

Consistency checks: the augmentation triple, the zero-based `0..8` offsets, the
transform order, the population-statistics method, the sample counts, and the
bundle recipe read identically in `open-decisions.md`, `specification.md`, and
`experiment.toml`. No field for OD-1, OD-5, or OD-6 still carries a
`proposed`/`pending` status. All companion links resolve.

## Notes and limits

- The numeric normalization constants are not computed in this session: they
  require the staged dataset and are generated and frozen in the Phase 2
  fixture manifest together with the initial bundle.
- OD-2/OD-3/OD-4/OD-10/OD-11 are still open and must be recorded before the
  phases that depend on them (Phase 3 acceptance and later); they do not block
  Phase 2.
- No GPU, remote, or training run; no commit.

## Exact next action

Implement Phase 2 (plan section 4): dataset adapters for both frameworks, the
saved epoch permutations and saved `(crop_top, crop_left, flip)` augmentation
records, the normalization statistics and canonical initial-parameter bundle
with recorded generator/version/seed, and the shared correctness fixtures in
the `.npz` plus manifest format, including the rectangular-linear and
asymmetric-kernel fixtures from the Phase 1 review. PyTorch-side fixture
generation needs the pinned Leonardo venv, so this phase includes a
compute-node run there (sync `f7b4520c` and precompile first).

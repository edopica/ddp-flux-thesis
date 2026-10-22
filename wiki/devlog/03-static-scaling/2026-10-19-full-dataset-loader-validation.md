# Full-dataset loader built and validated — 2026-10-19

**Status: PASS for loader validation only.** The minimal full-dataset path now
exists for PyTorch, Flux current `f7b4520c`, and Flux historical `8add0e42`.
The full benchmark campaign was **not** launched. No throughput, scaling,
numerical-equivalence, training-quality, or OD-8 memory-margin claim is made.

## Scope and implementation

New files under `scripts/static_scaling/`:

- `full_dataset.jl` — read-only `.npy` mmap loader. Parses the v1/v2 header,
  reverses dimensions so Julia's column-major layout matches NumPy C-order,
  verifies manifest metadata, and exposes epoch ids, global batches, rank
  shards, the frozen transform, and order/hash metadata.
- `full_dataset_timing.jl` — full-dataset timing path. Opens the mmap once per
  process, runs 20 discarded warmup updates, resets the model/optimizer from
  the bundle, and times updates `1..K`. Batch assembly, transform, one-hot, H2D,
  forward, backward, and the optimizer step are inside the timed region.
- `check_full_dataset.jl` — Julia contract checker against the Python reference
  at W=1,2,4.
- `full_dataset_common.py` — shared NumPy-only hashes, manifest/file checks,
  bundle verification, and loader helpers.
- `full_dataset_reference.py` — writes the reference NPZ/JSON for one epoch and
  update-1 batch.
- `check_full_dataset.py` — independent Python checker and optional Julia-JSON
  cross-check.
- `scripts/remote/leonardo/run-static-full-validation.sh` — the single
  validation driver (precompile → CPU checks → fixture smoke → dry runs).
- `scripts/remote/leonardo/run-static-full-campaign.sh` — **template only**,
  refuses to run without `CONFIRM_CAMPAIGN=1`; not executed.

Modified files:

- `timing_flux.jl` — includes the new files and dispatches to the full-dataset
  path on `--dataset-dir`; the fixture path below the dispatch is unchanged.
- `timing_torch.py` — adds `--dataset-dir`, `--epoch`, `--batch-size`,
  `--bundle`, source flags, and a full-dataset path; the fixture path retains
  its schema, defaults, and JSON keys.

The fixture timing path was smoke-tested after the edits: Flux and PyTorch
3-update records retain `schema=static-timing-pilot-v1`, `global_batch=12`,
`workload=alternating-two-fixture-batches`, and `timing=training-only`.

| File | SHA-256 |
|---|---|
| `full_dataset.jl` | `c3e0bf08debca81c847b405fda0da498f465b29df72ea1da2b015331775386bd` |
| `full_dataset_timing.jl` | `a1b5d206f81d7e654db5cad98d8c07f51333bcc68244ff89ceb4bbfd5cd074a5` |
| `check_full_dataset.jl` | `d2943b651aef7c26ef74069b7d4bcb6bb69cd685656608e210a1b00bb0ae43c3` |
| `full_dataset_common.py` | `4c3c4e0492728418417bc3bf3c1a7d2bb7741190897c7620e5888ab226a0581b` |
| `full_dataset_reference.py` | `aa8793517ad3b813e70c0c1afbe5c01db9555d73affae24ee8f54af1667283a2` |
| `check_full_dataset.py` | `d6b9dd4bb8d1d0ed23eeac023f9ffe6efa8d93855745a351b86d6784b903420b` |
| `timing_flux.jl` | `35ebc5788e64aa386acae523799306718904093a7924e12f025c6ef6208622ed` |
| `timing_torch.py` | `5cf1d28311afa175e2bad38ee298116ec3ee44a00ba24c657718df1cd2e28925` |
| `run-static-full-validation.sh` | `de88f950c1ebffbdea5d618fa2c25d771a781c811bc98abf00e4ab817e54abea` |
| `run-static-full-campaign.sh` (template) | `a204ba9a18589fa04ed9d1477ec6fa1beeeff2ac21407fdf89a1f72cf044be4f` |

## Frozen contract implementation

- Epoch 0 permutation, drop the final 160 ids → 99,840; global batches formed
  before sharding; rank `r` receives contiguous shard `r` in rank order.
- 195 updates at batch 512; 390 at batch 256.
- Transform order: zero-pad 4, crop at zero-based `(top,left)`, flip W when
  `flip==1`, `uint8 -> Float32`, divide by 255, subtract mean, divide by std.
- Flux images are moved to WHCN only after gathering; PyTorch remains NCHW.
- Flux labels use `Flux.onehotbatch(y .+ 1, 1:200)`; PyTorch uses the int64
  class ids directly.
- K defaults to 195 in the full-dataset path (300 remains the fixture default).
- Warmup is 20 discarded updates at `t=1..20`; the model and optimizer are reset
  from the bundle, and the measured region restarts at update 1 and data batch 1.
- The loader uses no DataLoader workers. PyTorch uses a synchronous NumPy
  loader with `torch.set_num_threads(1)`.

## Validation result (job 61450014)

| Item | Value |
|---|---|
| Job | `61450014`, `lrdn0667.leonardo.local`, `boost_qos_dbg` |
| State | `COMPLETED 0:0`, elapsed `00:14:09` |
| Precompile | `FLUX_PRECOMPILE_OK` for both Flux worktrees |
| Python reference | `FULL_DATASET_REFERENCE PASS` |
| Python independent check | `FULL_DATASET_PYTHON_CHECK PASS checks=169` |
| Julia current | `FULL_DATASET_JULIA_CHECK PASS label=flux-current checks=111` |
| Julia historical | `FULL_DATASET_JULIA_CHECK PASS label=flux-historical checks=108` |
| Python combined | `FULL_DATASET_PYTHON_CHECK PASS checks=177` for each revision |
| Fixture path | `FIXTURE_TIMING_PATH PASS` |
| Dry runs | `FULL_DATASET_DRYRUN PASS` for Flux current, Flux historical, PyTorch |
| Driver | `FULL_VALIDATION PASS` |

The checked properties include the exact epoch ids, rank sequences at W=1,2,4,
shard concatenation in rank order, no duplicated/missing ids, the 160-id tail,
195/390 update counts, one-hot mapping, the 62-array/11,271,432-parameter
bundle, dataset file hashes, and the transformed update-1 tensor. The Julia
transformed tensor matched the Python reference with
`max_abs_diff=0.0` at W=1, 2, and 4.

Frozen hashes:

- sample order `f773fcec7d5b5bd9f7b0f98c3fdf0364be5233ffb15cfa05f97ad06c3d865335`
- update-1 order `23c183e8e1b9b777a70013926c30590851cda4ae2c7fbb18ec8a0aa21515a61a`
- transform spec `e82e0703f40b15e2cf7f95c404a21e0ca8ae3efd403fa54da8c30a19cd5825fa`
- transformed update-1 sample `ab5730ec5bd7f2e144355de91ba889e7220937ac9ef812f4b6dfa6b4ef0f4cf0`
- `bundle_hash` `a7632ca90c9c51cd022b3c82e0dc81391608f317c1afbd17889deeb2d1dcce23`
- bundle file `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317`

## Dry-run memory (no performance claim)

Each dry run used epoch 0, batch 512, W=1, 20 warmup updates, and one measured
update. Final loss was `5.492467880249023` in all three configurations and was
finite. The identical loss is a shared-input cross-check, not a
training-quality claim.

| Framework | Validation seconds | Device memory |
|---|---:|---|
| Flux current | 0.50566162 | used `58,796,081,152` / `68,099,571,712` bytes (`86.34%`) |
| Flux historical | 0.512345482 | used `56,615,043,072` / `68,099,571,712` bytes (`83.14%`) |
| PyTorch | 0.5795233901590109 | max allocated `15,394,088,960` (`22.61%`), max reserved `24,540,872,704` (`36.04%`) |

`device used = total - free` from `CUDA.memory_info()`; it includes the CUDA
context and allocator caches. **The proposed OD-8 80% margin is not
established.** Flux device-used is already 83–86% at batch 512/W=1, so the
Phase 5 larger-batch memory pilot must measure peak process memory and decide
OD-7/OD-8. Batch 512 did not OOM; no silent fallback to 256 occurred.

## Failure history

- Job `61446924` failed before any workload: `local name=$1 ... $name` expanded
  an unset variable under `set -u`. Split the declarations.
- Job `61447030` passed all checks but recorded `host_maxrss_bytes` 1024× too
  large. Julia 1.12 `Sys.maxrss()` already returns bytes, not kilobytes. The
  fix was applied and job `61450014` supersedes it.
- A login-node `python -c "import numpy, torch"` probe timed out after 120 s;
  no process remained. All environment work was kept on the compute node.

## Campaign command (documented, not executed)

```bash
ssh leonardo 'CONFIRM_CAMPAIGN=1 sbatch --parsable -A iscrc_unmasked \
  -p boost_usr_prod --qos=boost_qos_lprod -N1 -n1 --cpus-per-task=32 \
  --gres=gpu:4 --time=04:00:00 --job-name=static-full-campaign \
  --output=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-campaign-%j.log \
  "$HOME/projects/ddp-flux-thesis/scripts/remote/leonardo/run-static-full-campaign.sh"'
```

The campaign template runs current/historical Flux and PyTorch at W=1,2,4 with
epoch 0, batch 512, K=195, and one repeat by default. It is not acceptance
evidence and needs its own preflight, repeat policy, and provenance review.

## Evidence

- [RESULTS](../../artifacts/logs/static-scaling/2026-10-19-full-loader-validation-61450014/RESULTS.md)
- [COMMANDS](../../artifacts/logs/static-scaling/2026-10-19-full-loader-validation-61450014/COMMANDS.md)
- [SUMMARY](../../artifacts/logs/static-scaling/2026-10-19-full-loader-validation-61450014/SUMMARY.json)
- [Slurm log](../../artifacts/logs/leonardo/2026-10-19-static-full-loader-validation-61450014.log)
- Superseded metadata-only run: `artifacts/logs/static-scaling/2026-10-19-full-loader-validation-61447030/`

## Exact next action

Run the Phase 5 larger-batch memory pilot under OD-14, measure peak process
memory at batch 512/W=1 (and the W=2/4 matrix), and record OD-7/OD-8 before any
full-dataset campaign. Do not use the validation dry-run timings as benchmark
results.

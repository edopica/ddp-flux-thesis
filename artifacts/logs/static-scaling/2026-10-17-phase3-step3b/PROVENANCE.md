# Provenance — static-scaling Phase 3 Step 3b (FMA algorithm chooser)

Date: 2026-10-17 (UTC run window 20:22–20:46).
Status: **precision regression PASS**, **full comparison FAIL**.
No Flux source change, no commit, no tolerance or elementwise-rule change,
no model-math or optimizer change. `atol=1e-5`, `rtol=1e-4` unchanged.

## 1. Source identity

| Item | Value |
|---|---|
| Thesis `git rev-parse HEAD` | `3088f61020137912af32d9c969e87ea9ae2a71c3` |
| Thesis source-state hash | `4f2da8a364b277839b4dc77f1f642866d900e9b5e025de17791a0bf28cfc3c1e` |
| Flux worktree commit | `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e` |
| Flux worktree status | clean |
| Flux source-state hash | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty-diff hash) |

Source-state recipe (unchanged from Phase 2/3):

```bash
STATE=$( { git diff HEAD; git ls-files --others --exclude-standard \
  | grep -v '^artifacts/' | sort | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1 )
```

Changed/added source files in this correction:

| File | SHA-256 |
|---|---|
| `scripts/static_scaling/precision.jl` (run-time) | `7d47c687bf003eb9cb2357e6950d95a44263afa88707e35fa26b213bf8bb9bf8` |
| `scripts/static_scaling/check_phase3_flux.jl` | `dd0489e38ed2f76ae04aa9e4b16dd648eeff55dc66924b64ab9e8caa4d7c54fa` |
| `scripts/static_scaling/check_precision_math.jl` | `13afc6a13d4dd65f62ef2d1333b6024d525d719695c8dc82d6f465363876e24f` |
| `scripts/remote/leonardo/run-phase3-precision-regression.sh` | `8bfccb66395f13542330cb4c975c8419c5085a83f3d4d25b8fcc58bd3b47c90d` |
| `scripts/static_scaling/precision.jl` (current local, post-run) | `60e003ced5cb50fd546edddc2976bc403ee3d855977274f0bb09d7f34946916d` |

Remote thesis copies matched the run-time hashes after sync (checked
with `sha256sum` on `leonardo`). The only post-run source change is the
`@warn` metadata syntax in the fallback branch of `precision.jl`
(`kind` -> `kind = kind`); that branch did not execute in either case
and the algorithm-selection logic is unchanged. The remote tree was not
re-synced after that fix, so the remote code is the run-time snapshot.

## 2. Remote environment and run identity

| Item | Value |
|---|---|
| Cluster / partition / QoS | Leonardo, `boost_usr_prod`, `boost_qos_dbg` |
| Precompile job / node | `60909956` / `lrdn0034` (`FLUX_PRECOMPILE_OK`) |
| Precision regression job / node | `60910394` / `lrdn0365` (`PHASE3_PRECISION_REGRESSION PASS`) |
| Full driver job / node | `60910786` / `lrdn0001` (exit 1) |
| GPU | NVIDIA A100-SXM-64GB (one GPU allocated) |
| Julia | 1.12.6 |
| CUDA.jl / cuDNN.jl | 6.2.2 / 6.2.2 (`$WORK/epicazio/julia-depot/packages/cuDNN/0SFve`) |
| cuDNN library | 9.26.0 (`cuDNN.version()` prints `92.60.0`) |
| NNlib / Flux | 0.9.45 / 0.16.11 |
| PyTorch reference env | torch 2.9.1+cu126, CUDA 12.6, cuDNN 9.10.2.21 |
| Driver tolerances | `atol=1e-5`, `rtol=1e-4` (defaults, no override) |

Remote run roots:

```text
/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3b/
  reference/phase3_reference.npz
  reference/phase3_reference_manifest.json
  logs/remote-check.log
  precision/logs/fma.log
  precision/logs/current.log
```

## 3. Sync

The two preserved-evidence directories were excluded exactly as in
Step 3. Two additional excludes were necessary because the local copies
of the failed Step 3 evidence (286 MB baseline + logs) are not inputs to
this run and the link to Leonardo was transferring at roughly
0.1–0.3 MB/s; the first full-sync attempt was stopped after it spent
15 minutes uploading that local copy. The remote fixtures under
`artifacts/baselines/static-scaling/2026-10-16-phase2/` were already
present and were not modified.

```bash
rsync -avz --delete --exclude-from='.rsyncignore' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-attempt3-preserved/' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-step3/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-step3/' \
  . leonardo:projects/ddp-flux-thesis/
```

The final transfer was 81,535 bytes; the remote source tree now matches
the local source hashes above. `scripts/remote/sync_code.sh` was not
used.

## 4. Reference equality and independent check

New reference SHA-256 (remote == copied local):

```text
97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783  phase3_reference.npz
d9985934156c91404a7d02563de63e07b631fcf028859202b3ee6104882effca  phase3_reference_manifest.json
```

The reference is byte-identical to the preserved attempt-3 reference
(`97e5007c…`). Independent comparison: 337/337 arrays exact-equal, zero
differing arrays, maximum absolute difference `0.0`. The independent
NumPy checker was run locally on the copied reference/manifest:
`PHASE3_REFERENCE_CHECK PASS arrays=337 atol=1e-05 rtol=0.0001`, exit 0.

Because the remote new `.npz` hash equals the already-local preserved
copy, the local Step 3 reference file was copied byte-for-byte into this
evidence directory instead of downloading 286 MB again.

## 5. Algorithm evidence

All 32 full-model algorithm choices are `CUDNN_FMA_MATH(3)` with
`fallback=false`. Every recorded candidate math type is `3`; no
tensor-op/TF32 candidate was present. Chosen algorithms:

- forward: algo 0 (7), algo 6 (4)
- backward data: algo 0 (7), algo 4 (3)
- backward filter: algo 0 (2), algo 3 (6), algo 5 (3)

Per-choice lines with shapes are in `algorithm-choices.txt`. The
precision regression's autotuned `Flux.CrossCor` item chose
`algo=0, CUDNN_FMA_MATH(3), fallback=false` with
`rel_l2=4.248559e-7`.

## 6. Outcome

`PHASE3_REFERENCE PASS`, `PHASE3_REFERENCE_CHECK PASS`,
`PHASE3_FLUX_CHECK FAIL` (`phase3.grads.u1`, `phase3.grads.u3`,
`phase3.m.u3`), `PHASE3_REMOTE_CHECK FAIL`, driver exit 1.

The residual is FP32 backward-algorithm rounding (FFT/Winograd/direct),
not TF32: the forward now passes and all candidate math types are FMA.
The next decision belongs to the user (see `RESULTS.md`).

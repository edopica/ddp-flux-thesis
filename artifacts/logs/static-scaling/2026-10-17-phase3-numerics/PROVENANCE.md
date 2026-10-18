# Provenance — Phase 3 numerical calibration (2026-10-17)

Status: diagnostic only. No tolerance, elementwise rule, model math, optimizer,
checker, or Flux source change. No commit.

## 1. Source identity

The CPU reference intentionally reuses the preserved GPU reference provenance,
so the two archives differ only by backend and by the diagnostics they
contained at generation time.

| Item | Value |
|---|---|
| Thesis `git rev-parse HEAD` | `3088f61020137912af32d9c969e87ea9ae2a71c3` |
| Source-state hash recorded in both manifests | `82623bbb516a981d6c795792d1433521d25a8ee3bbcc44a2b66f6f4631a452b9` |
| Flux worktree commit | `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e` |
| Flux source-state hash | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty-diff hash) |

Run-time script hashes (local and remote matched before the runs):

| File | SHA-256 |
|---|---|
| `scripts/static_scaling/check_phase3_flux.jl` (unchanged) | `dd0489e38ed2f76ae04aa9e4b16dd648eeff55dc66924b64ab9e8caa4d7c54fa` |
| `scripts/static_scaling/make_phase3_reference.py` (unchanged) | `0a9a2645f2bf0bd66e94e1351d3538e050a8261dee451c77dd4b7b282cdab21d` |
| `scripts/static_scaling/model_torch.py` (unchanged) | `77ac48e815b58948ab64b16938d988cae35f4719f084c25432703f6ffadfdf76` |
| `scripts/static_scaling/compare_reference_arrays.py` (new) | `27edd256f917cc8101c3d881d16949695c71f4fac7265eaccb1a8498c44885e5` |
| `scripts/remote/leonardo/run-phase3-numerics-reference.sh` (new) | `1696a698022c9b63fb5b9b0dd6cf91e139d38160f39361a13974d55071b15c3a` |
| `scripts/remote/leonardo/run-phase3-numerics-flux-cpu.sh` (new) | `3528c286cdc82c6515d0bf89889bf913ca7973b1f24369946eb45c486e4eedce` |

The remote Julia files were not re-synced; their hashes match the Step 3b
run-time snapshot. Only Python and shell files were rsynced. No precompile was
run and no checker log contains `Precompiling packages...`.

## 2. Remote environment and jobs

| Item | Value |
|---|---|
| Cluster / partition / QoS | Leonardo, `boost_usr_prod`, `boost_qos_dbg` |
| CPU reference job / node | `60912525` / `lrdn3450` |
| Flux CPU job / node | `60912664` / `lrdn3450` |
| GPUs allocated | one A100 visible (`GRES=gpu:1`) but unused by both jobs |
| Python | `$WORK/epicazio/venvs/pytorch-scaling/bin/python`, 3.11.17 |
| PyTorch / CUDA / cuDNN (PyTorch) | 2.9.1+cu126 / 12.6 / 9.10.2.21 (`cudnn_version=91002`) |
| NumPy (remote) | 2.4.6 |
| Julia | 1.12.6, `JULIA_NUM_THREADS=1` |
| Julia packages | CUDA.jl 6.2.2, cuDNN.jl 6.2.2, cuDNN library 9.26.0, NNlib 0.9.45, Flux 0.16.11 (from Step 3b) |
| Threads | PyTorch `torch.set_num_threads(1)`; one Julia thread |

Remote run roots:

```text
/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-numerics/
  torch_cpu_reference/phase3_reference.npz
  torch_cpu_reference/phase3_reference_manifest.json
  logs/torch-cpu-reference.log
  logs/torch-cpu-reference-check.log
  logs/torch-cpu-vs-gpu-reference.log
  logs/torch-cpu-vs-gpu-reference.json
  logs/flux-cpu-vs-torch-gpu.log
  logs/flux-cpu-vs-torch-cpu.log
  logs/SHA256SUMS
  logs/SHA256SUMS-flux-cpu
```

Preserved inputs (read-only from this run):

```text
/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-attempt3/reference/
  phase3_reference.npz
  phase3_reference_manifest.json
```

## 3. Reference identity

| Item | Value |
|---|---|
| CPU reference arrays | 337 |
| CPU reference manifest `device` | `"cpu"` |
| CPU reference manifest `created` | `2026-10-17T20:54:36Z` |
| CPU reference npz SHA-256 | `a7ff467b0e1d216a6d006e8ae576a97694e4a654d481bdec86b06e269a1217b3` |
| CPU reference manifest SHA-256 | `2ff9241c172a8f42fdffac81bc6fcb6907b3de9985c4d835f4ab17b52e6894ea` |
| Preserved GPU reference npz SHA-256 | `97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783` |
| Preserved GPU manifest SHA-256 | `2416e5557eb696e499a553fd1866d49c1c529a5fb778541df2f009825689f64d` |
| Fixture `data_fixture.npz` SHA-256 | `6614a00ba7cac588b331b65cef4240f27624944c7a9df0383fdbeee3124066b3` |
| Bundle `initial_bundle.npz` SHA-256 | `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317` |
| Fixture manifest SHA-256 | `51448d5905c5dfd9a47b945d96102d19ea55073fbfe9f249fd268cca03455a65` |

The CPU manifest records `--device cpu`, the same 62 parameter arrays /
11,271,432 parameters, the same 337-array contract, and the same source/flux
provenance as the preserved GPU manifest.

## 4. Transfers and verification

The CPU npz (299,701,312 bytes) and manifest were copied home with rsync; the
local SHA-256 values match the remote values in `logs/SHA256SUMS`. The remote
checker and comparison logs were copied with rsync; their SHA-256 values match
`logs/SHA256SUMS` and `logs/SHA256SUMS-flux-cpu`. The two Leonardo console logs
were copied into this directory and hashed locally.

`sync_code.sh` was not used. The explicit exclusions from Steps 2/3 were not
needed because only three individual files were transferred; no evidence
directory was uploaded or deleted.

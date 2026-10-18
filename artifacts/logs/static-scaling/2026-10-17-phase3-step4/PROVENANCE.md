# Provenance — static-scaling Phase 3 Step 4 optimizer probe

Date: 2026-10-17 (UTC run start 2026-10-18T00:33:04Z).
Status: **PASS**. Probe-only evidence; no training-path, tolerance, schedule,
model-math, acceptance-checker, or Flux source change.

## 1. Source identity

| Item | Value |
|---|---|
| Thesis `git rev-parse HEAD` | `3088f61020137912af32d9c969e87ea9ae2a71c3` |
| Thesis source-state hash | `39b1051f27077581b30ff49ae09aada8d58086f75d446a8a278718c992106164` |
| Flux worktree commit | `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e` |
| Flux worktree state | clean; empty-diff source-state hash below |
| Flux source-state hash | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Source-state recipe (unchanged):

```bash
STATE=$( { git diff HEAD; git ls-files --others --exclude-standard \
  | grep -v '^artifacts/' | sort | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1 )
```

New files in this revision:

| File | SHA-256 |
|---|---|
| `experiments/static-scaling/probes/probe_optimizer_contract_cross.jl` | `243e69e827854141f6fbec05b6c21d99a94d8c6e6f29f8cc49bf86480ec273c9` |
| `experiments/static-scaling/probes/probe_optimizer_contract_cross.py` | `7e3a58f93224e8adbe3c4cd0bbce1e0387c76dd61fdff5eb754b999e9a9844f3` |
| `experiments/static-scaling/probes/compare_optimizer_contract.py` | `56c7e043c6b8d6c28edc6ffc5261ec0dd32bf929c29d96e2bd62dca5b5d2e889` |
| `scripts/remote/leonardo/run-phase3-optimizer-probe.sh` | `4a70f3acc0205c975e242370ef36895d76fbaf0f5d20bb81ce2ac0b0078c4e6d` |

Remote copies of all four files matched these hashes after sync (checked with
`sha256sum` over SSH).

The existing `experiments/static-scaling/probes/probe_optimizer_contract.jl`
was included unchanged by the Julia probe; its constructor and beta-power
checks still run and print their R3/R4 lines.

## 2. Inputs

| Item | Value |
|---|---|
| Bundle | `artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz` |
| Bundle SHA-256 | `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317` |
| Arrays / elements | 62 / 11,271,432 |
| Model source (Flux) | `scripts/static_scaling/model_flux.jl` (`build_optimizer_state`, `schedule_eta`, constants) |
| Model source (PyTorch) | `scripts/static_scaling/model_torch.py` (`build_optimizer`, `schedule_eta`, `parameter_groups`, constants) |
| Spec / config | `experiments/static-scaling/specification.md`, `experiment.toml` |

## 3. Remote environment and run identity

| Item | Value |
|---|---|
| Cluster / partition / QoS | Leonardo, `boost_usr_prod`, `boost_qos_dbg` |
| Allocation | 1 node, 1 task, 8 CPUs, `--gres=none` (CPU-only), 30 min cap |
| Job / node | `60948431` / `lrdn0171` |
| UTC start | `2026-10-18T00:33:04Z` |
| Python | 3.11.17 |
| numpy | 2.4.6 |
| torch | 2.9.1+cu126 (`cuda_available=False`) |
| Julia | 1.12.6 |
| Julia depot | `/leonardo_work/IscrC_UNMASKED/epicazio/julia-depot` |
| Flux load path | thesis `scripts/static_scaling/julia-env` + `~/projects/flux-integration/test` + stdlib |
| Python venv | `/leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling` |

Remote run root:

```text
/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step4/
  probe_flux.json
  probe_flux_arrays.npz
  probe_torch.json
  probe_torch_arrays.npz
  comparison.json
  logs/probe-flux.log
  logs/probe-torch.log
  logs/comparator.log
  logs/SHA256SUMS
```

The evidence root name `2026-10-17-phase3-step4` is new; no previous run
directory was reused or overwritten.

## 4. Sync

`scripts/remote/sync_code.sh` was not used. The explicit Step 3c exclusion
rsync was repeated with the newer local-only Step 3c evidence directories
also excluded; the exact command is in `COMMANDS.md` section 4.

## 5. Run

The exact allocation and environment-prefix command is in `COMMANDS.md`
section 5. The driver ran three fresh processes in order (Flux probe,
PyTorch probe, comparator) and printed
`PHASE3_OPTIMIZER_PROBE PASS` and `PHASE3_OPTIMIZER_PROBE_DRIVER PASS`,
exit 0.

## 6. Evidence hashes

| File | SHA-256 |
|---|---|
| `comparison.json` | `6d3716517903a3c9dc49327bae5c6510fe5e2fc7166323c92e7fc53bd3ac7431` |
| `probe_flux.json` | `a0a08e0822a50e2bd866fc448f5727ab044169c5c07b2fa0dd7cfb63f6acd3de` |
| `probe_flux_arrays.npz` | `9f8df0f98323919dc18af5aa5b609d360b9205b26f586ccf052ba5cc57887c58` |
| `probe_torch.json` | `1ca4455560806a3a31ba6c928b428493ec683ec423d1b2b8ba2cb7f5e013332e` |
| `probe_torch_arrays.npz` | `77e195bc4332954f5a0ef1336b4eb1d7a0bebec3651dcab6913fbb5b402c19b0` |
| `logs/comparator.log` | `3783c3442dcd129c0ca9d745bfd0387aebe53cffb7aff66547b604bb3cfef1c0` |
| `logs/probe-flux.log` | `97d8adab45218aa597df8992ad2f22ce7a8661dfb772d02f08a2f2d707e4d9b1` |
| `logs/probe-torch.log` | `0074dbcec0d01b93a93165b1e7bdedac49f725976872d79e67a7df60a1184afe` |
| `2026-10-17-phase3-step4.log` | `6703b3236ac25c0a2faa77e3757ade588dafd4b599fe197a5fc6064529560a4d` |

These values match the remote `logs/SHA256SUMS` for the copied artifacts.
`SHA256SUMS` in this directory adds the local console log and the source
files.

## 7. Scope

The probe is independent arithmetic evidence for the library optimizer
contract and the schedule boundaries. It does not establish training
convergence, distributed collective behavior, GPU kernel numerics, or a new
Phase 3 acceptance rule. The probe's `eta=1e-2` is a sensitivity setting,
not a training-schedule change.

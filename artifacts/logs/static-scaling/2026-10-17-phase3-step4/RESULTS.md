# Step 4 results — cross-framework optimizer-contract and schedule probe PASS

Date: 2026-10-17 (UTC run start 2026-10-18T00:33:04Z).
Source: `2026-10-17-phase3-step4.log`, job `60948431`, node `lrdn0171`.
Status: **PASS**. `PHASE3_OPTIMIZER_PROBE PASS`, 69 checks, 34/34
negative controls rejected, driver exit 0.

This is a compact sensitivity probe of library configuration, not a custom
optimizer for training. Training continues to use `Optimisers.AdamW` and
`torch.optim.AdamW`. No training code, Phase 3 tolerance, acceptance
checker, schedule value, model math, or Flux source changed.

## Probe design

Three fresh processes on one CPU-only compute node:

1. `experiments/static-scaling/probes/probe_optimizer_contract_cross.jl`
   (Flux + Optimisers.jl). It includes the existing
   `probe_optimizer_contract.jl`, so the constructor and beta-power checks
   still run unchanged, then writes `probe_flux.json` and
   `probe_flux_arrays.npz`.
2. `experiments/static-scaling/probes/probe_optimizer_contract_cross.py`
   (PyTorch + `torch.optim.AdamW`), writing `probe_torch.json` and
   `probe_torch_arrays.npz`.
3. `experiments/static-scaling/probes/compare_optimizer_contract.py`
   (independent Float64 arithmetic oracle, cross-framework comparison,
   schedule boundaries, and negative controls), writing `comparison.json`.

The driver is
`scripts/remote/leonardo/run-phase3-optimizer-probe.sh`.

### Probe-specific comparison rule

```text
|a - b| <= 1e-7 + 5e-7 * max(|a|, |b|)
```

This is **not** the Phase 3 acceptance rule. Float32 has ~1.19e-7 relative
spacing near `|x| = 1`; the observed library-vs-Float64-oracle residuals are
at most 0.270 of this bound (Flux) and 0.103 (PyTorch), so the bound keeps
~3.7x margin over observed library rounding. The probe's no-decay leaf has
`|x| >= 2`, so an incorrect decay changes `x` by `eta*lambda*|x| =
1e-6*|x| >= 2e-6`; the weakest decay controls exceed the bound by 5.1x
(`omitted_decay`) and 5.4x (`decay_on_nodecay`). Wrong epsilon placement
changes the tiny-gradient update by ~1e-3, i.e. ~2,858x the bound.

Moments are supplementary and compared with a relative-only
`|a-b| <= 5e-5*max(|a|,|b|)` because the library carries `beta` in Float32
(`1-beta2` differs from the Float64 value by ~1.29e-5 relative). The
nonzero-moment requirement is checked directly.

## Training-path configuration assertions

Both frameworks built and inspected the real full-model configuration
without running an 11M-parameter update:

| Item | Flux | PyTorch |
|---|---|---|
| Parameters / arrays | 11,271,432 / 62 | 11,271,432 / 62 |
| Decay leaves | 21, lambda `1e-4` | group 0: 21 parameters, `weight_decay=1e-4` |
| No-decay leaves | 41, lambda `0` | group 1: 41 parameters, `weight_decay=0.0` |
| Rule metadata | `beta=(0.9,0.999)`, `epsilon=1e-8`, `couple=true` | `betas=(0.9,0.999)`, `eps=1e-8`, `lr=1e-3` |
| Bundle | same `initial_bundle.npz` (62 arrays) | same `initial_bundle.npz` (62 arrays) |

The Flux per-leaf `lambda` assignment comes from
`build_optimizer_state`; the PyTorch groups come from `build_optimizer`.
The legacy constructor check still prints `mixed_constructor=MethodError`
and the beta-power continuation is exact.

## Controlled update cases

Probe learning rate `eta=1e-2` (sensitivity probe; the training schedule
still uses `eta_max=1e-3`). Each case runs three updates with controlled
nonzero gradients; `m` and `v` are nonzero after the two updates that precede
the measured third update. Both leaves start near `|x| = 1..2`.

Residuals against the Float64 oracle under the probe rule:

| Framework | Case | Leaf | `x_final` max abs | ratio | x_steps max abs | m ratio | v ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Flux | main | decay | 6.116890e-08 | 0.270 | 6.116890e-08 | 0.005 | 0.259 |
| Flux | main | no-decay | 1.073242e-07 | 0.233 | 1.073242e-07 | 0.006 | 0.259 |
| Flux | tiny | decay | 5.680083e-08 | 0.095 | 5.680083e-08 | 0.016 | 0.259 |
| Flux | tiny | no-decay | 1.071318e-07 | 0.097 | 1.071318e-07 | 0.016 | 0.259 |
| PyTorch | main | decay | 6.067518e-08 | 0.102 | 6.067518e-08 | 0.002 | 0.002 |
| PyTorch | main | no-decay | 5.382743e-08 | 0.064 | 5.382743e-08 | 0.001 | 0.002 |
| PyTorch | tiny | decay | 6.031653e-08 | 0.103 | 6.031653e-08 | 0.004 | 0.001 |
| PyTorch | tiny | no-decay | 1.071484e-07 | 0.097 | 1.071484e-07 | 0.004 | 0.001 |

Flux vs PyTorch final values (same rule): maximum absolute differences are
`5.96e-08` (main decay), `1.19e-07` (main no-decay), `5.96e-08` (tiny
decay), and `0.0` (tiny no-decay); all zero violations.

Full values and the moment ratios are in `controlled-cases.txt`. The
`tiny` case uses gradients of `1e-9` and `1e-12` so `epsilon=1e-8`
dominates the denominator.

## Negative controls

All controls use the measured library value as the candidate and a wrong
oracle as the reference. `REJECTED` means the probe rule counts at least one
element violation. 34 of 34 were rejected; the weakest ratio is 5.10.

| Control | Runs | Weakest ratio | Max abs difference |
|---|---:|---:|---:|
| omitted update | 8 | 1,204.8 | 2.77e-02 |
| wrong decay scale (`lambda=1e-3`) | 4 | 44.9 | 2.70e-05 |
| omitted decay (`lambda=0`) | 4 | 5.10 | 3.06e-06 |
| decay on no-decay leaf | 4 | 5.36 | 6.00e-06 |
| wrong epsilon placement (`sqrt(vhat+eps)`) | 4 | 1,204.7 | 1.33e-03 |
| wrong bias correction | 8 | 996.0 | 8.52e-02 |
| schedule: warm-up formula at `W+1` | 2 | 10.2 | 1.03e-06 |

Every control printed an explicit
`NEGATIVE_CONTROL ... REJECTED` line in `logs/comparator.log`.

## Schedule boundaries

`eta_max=1e-3`, `W=975`, `T=9750`. Flux and PyTorch agree with the Float64
formula and with each other exactly:

| `t` | Expected | Flux | PyTorch |
|---|---:|---:|---:|
| 1 | `eta_max/W` = `1.0256410256410257e-06` | same | same |
| `W` | `eta_max` = `1.0e-3` | same | same |
| `W+1` | `9.9999996795609018e-04` | same | same |
| `T` | `0.0` | same | same |

Both frameworks reject `t=0` and `t=T+1` (Flux `ArgumentError`, PyTorch
`RuntimeError`). The off-by-one control (warm-up formula at `W+1`)
produced `1.001025641025641e-03`, a difference of `1.025673e-06`, and was
rejected with ratio 10.2.

## Run identity

| Item | Value |
|---|---|
| Cluster / partition / QoS | Leonardo, `boost_usr_prod`, `boost_qos_dbg` |
| Allocation | 1 node, 1 task, 8 CPUs, `--gres=none` (CPU-only) |
| Job / node | `60948431` / `lrdn0171` |
| UTC start | `2026-10-18T00:33:04Z` |
| Python / numpy / torch | 3.11.17 / 2.4.6 / 2.9.1+cu126 (`cuda_available=False`) |
| Julia | 1.12.6 |
| Thesis HEAD | `3088f61020137912af32d9c969e87ea9ae2a71c3` |
| Thesis source-state SHA-256 | `39b1051f27077581b30ff49ae09aada8d58086f75d446a8a278718c992106164` |
| Flux HEAD | `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e` |
| Flux source-state SHA-256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Bundle SHA-256 | `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317` |

## Evidence hashes

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
| probe Julia source | `243e69e827854141f6fbec05b6c21d99a94d8c6e6f29f8cc49bf86480ec273c9` |
| probe Python source | `7e3a58f93224e8adbe3c4cd0bbce1e0387c76dd61fdff5eb754b999e9a9844f3` |
| comparator source | `56c7e043c6b8d6c28edc6ffc5261ec0dd32bf929c29d96e2bd62dca5b5d2e889` |
| driver source | `4a70f3acc0205c975e242370ef36895d76fbaf0f5d20bb81ce2ac0b0078c4e6d` |

The copied artifacts match the remote `logs/SHA256SUMS` byte for byte.

## Scope and limitations

- The probe tests the optimizer contract and schedule boundaries; it does
  not test training convergence, distributed collectives, or GPU kernels.
- `eta=1e-2` is deliberately larger than the training `eta_max=1e-3` so the
  decoupled decay term is visible at Float32 scale. The equation is the same;
  the schedule is unchanged.
- `m`/`v` are compared with a supplementary relative bound because the
  library stores `beta` in Float32.
- The probe is CPU-only and allocates no GPU. It ran on a compute node per
  Leonardo policy.

## Outcome

`PHASE3_OPTIMIZER_PROBE PASS`; 69 checks, 34/34 negative controls rejected;
driver `PHASE3_OPTIMIZER_PROBE_DRIVER PASS`, exit 0. This closes the Step 4
optimizer/schedule sensitivity gap (review R3) without changing the training
path or any Phase 3 acceptance rule.

# Phase 3 Step 4: cross-framework optimizer-contract probe PASS

Date: 2026-10-17 (UTC run start 2026-10-18T00:33:04Z)
Status: **PASS**. `PHASE3_OPTIMIZER_PROBE PASS`, 69 checks, 34/34 negative
controls rejected, driver exit 0 (job `60948431`, node `lrdn0171`).

## Why

Review R3 found that the three Phase 3 updates use warm-up rates whose total
is below the parameter tolerance, so final-parameter agreement is weak
evidence for the optimizer update and the decoupled decay. Step 4 adds a
compact sensitivity probe for the numerical optimizer contract and the
untested cosine-schedule branch, without changing the training path or the
Phase 3 acceptance rule.

## What was added

| File | Purpose |
|---|---|
| `experiments/static-scaling/probes/probe_optimizer_contract_cross.jl` | Flux probe; includes the existing `probe_optimizer_contract.jl` unchanged, asserts the real 21/41 rule tree, runs controlled leaves, writes JSON + NPZ |
| `experiments/static-scaling/probes/probe_optimizer_contract_cross.py` | PyTorch probe; asserts the real two-group AdamW configuration, runs the same controlled leaves, writes JSON + NPZ |
| `experiments/static-scaling/probes/compare_optimizer_contract.py` | Independent Float64 arithmetic oracle, Flux/PyTorch comparison, schedule boundaries, negative controls |
| `scripts/remote/leonardo/run-phase3-optimizer-probe.sh` | CPU-only compute-node driver: three fresh processes, then summary |

The probe uses the library optimizers only. The Float64 oracle is confined to
the comparator. No custom optimizer enters the training path.

## Training-path configuration

Both frameworks built the real full-model configuration without an
11M-parameter update:

- Flux: 62 leaves, 21 decay (`lambda=1e-4`), 41 no-decay (`lambda=0`),
  `beta=(0.9,0.999)`, `epsilon=1e-8`, `couple=true`.
- PyTorch: two parameter groups, 21 parameters at `weight_decay=1e-4` and 41
  at `0.0`, `betas=(0.9,0.999)`, `eps=1e-8`, `lr=1e-3`.

The legacy constructor and beta-power checks still run in-process; the
beta-power continuation is exact.

## Controlled cases and probe rule

Three updates per case, controlled nonzero gradients; `m` and `v` are
nonzero after the two updates before the measured third update. One decay
leaf and one no-decay leaf per framework. Probe `eta=1e-2` (sensitivity
setting; training stays at `eta_max=1e-3`).

Probe-specific rule, separate from Phase 3 acceptance:

```text
|a - b| <= 1e-7 + 5e-7 * max(|a|, |b|)
```

Observed maxima: Flux `x_final` ratio 0.270, PyTorch 0.103; all zero
violations. Flux vs PyTorch maximum difference `1.19e-07`. The `tiny` case
uses `1e-9` and `1e-12` gradients; the wrong `sqrt(vhat+eps)` differs by
`1.33e-03` and is rejected at 1,205x the bound.

The moment check uses a supplementary `5e-5` relative bound because the
library stores `beta` in Float32.

## Negative controls

34/34 rejected, weakest ratio 5.10:

| Control | Runs | Weakest ratio |
|---|---:|---:|
| omitted update | 8 | 1,204.8 |
| wrong decay scale (`lambda=1e-3`) | 4 | 44.9 |
| omitted decay | 4 | 5.10 |
| decay on no-decay leaf | 4 | 5.36 |
| wrong epsilon placement | 4 | 1,204.7 |
| wrong bias correction | 8 | 996.0 |
| schedule warm-up formula at `W+1` | 2 | 10.2 |

## Schedule boundaries

`eta_max=1e-3`, `W=975`, `T=9750`; Flux and PyTorch agree exactly with the
Float64 formula:

| `t` | Value |
|---|---|
| 1 | `1.0256410256410257e-06` |
| W | `1.0e-3` |
| W+1 | `9.9999996795609018e-04` |
| T | `0.0` |

Both reject `t=0` and `t=T+1`. The off-by-one warm-up value at `W+1`
(`1.001025641025641e-03`, delta `1.025673e-06`) is rejected.

## Run identity

- Allocation: Leonardo `boost_usr_prod`, `boost_qos_dbg`, 1 node, 1 task,
  8 CPUs, `--gres=none` (CPU-only), 30 min cap.
- Job `60948431`, node `lrdn0171`, UTC start `2026-10-18T00:33:04Z`.
- Python 3.11.17, numpy 2.4.6, torch 2.9.1+cu126 (`cuda_available=False`);
  Julia 1.12.6.
- Thesis HEAD `3088f610...`, source-state
  `39b1051f27077581b30ff49ae09aada8d58086f75d446a8a278718c992106164`;
  Flux HEAD `f7b4520c...`, clean (empty-diff hash).
- Bundle `a790b03c...`.

## Evidence

- [Results](../../artifacts/logs/static-scaling/2026-10-17-phase3-step4/RESULTS.md)
- [Commands](../../artifacts/logs/static-scaling/2026-10-17-phase3-step4/COMMANDS.md)
- [Provenance](../../artifacts/logs/static-scaling/2026-10-17-phase3-step4/PROVENANCE.md)
- [Controlled cases](../../artifacts/logs/static-scaling/2026-10-17-phase3-step4/controlled-cases.txt)
- Console log:
  `artifacts/logs/static-scaling/2026-10-17-phase3-step4/2026-10-17-phase3-step4.log`
- Source hashes: probe Julia `243e69e8...`, probe Python `7e3a58f9...`,
  comparator `56c7e043...`, driver `4a70f3ac...`.

## Scope

No training code, Phase 3 tolerance, acceptance checker, schedule value,
model math, or Flux source changed. No commit. Steps 5–6 were not run.

**Exact next action:** start Step 5 of the Phase 3 handoff: locate evidence
for the hardened Phase 2 remote re-check against the real `$PREP`; if it is
absent, run `scripts/remote/leonardo/run-phase2-check.sh` with the corrected
checkers. Then finish the decision record for OD-2, OD-3, OD-4, and OD-11
(OD-10 is already recorded) in `open-decisions.md`, `specification.md`, and
`experiment.toml`, asking the user where a decision still requires authority.
Do not start Step 6.

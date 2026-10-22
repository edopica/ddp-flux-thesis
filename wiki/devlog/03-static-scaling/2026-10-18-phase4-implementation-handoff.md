# Phase 4 implementation handoff — 2026-10-18

Status: **incomplete; handoff written for a separate session.**
No Phase 4 acceptance claim. No remote run completed this session.

## What was implemented

- `scripts/static_scaling/phase4_common.jl` and `phase4_flux.jl`: Flux
  reference, DDP, and split-control modes; recording backend for the
  64-collective wrapper protocol; NPZ/JSON artifacts; rank-0-only shared
  writes.
- `scripts/static_scaling/check_phase4.py`: independent numpy checker with
  fixture/reference integrity, Phase 2 shard replay, rank replicas, offline
  synchronized gradients, an independent float64 decoupled-AdamW oracle,
  same/cross-framework categories, protocol checks, and split-control
  diagnostics.
- `scripts/remote/sync_flux_integration.sh`,
  `scripts/remote/leonardo/run-phase4-check.sh`,
  `scripts/remote/deathstar/run-phase4-cpu.sh`: sync/precompile-aware remote
  drivers for GPU/NCCL and CPU/MPI.
- `temp/phase4-implementation-contract.md`: frozen Phase 4 interface/schema.

## Evidence that passed

- Flux reference: `PHASE4_FLUX_REFERENCE PASS arrays=337`
  (`/tmp/opencode/phase4-reference-1t.log`).
- Flux DDP CPU/MPI world 2 and 4: all ranks PASS, 64 collectives per update,
  exact rank-replica hashes
  (`/tmp/opencode/phase4-ddp-w2.log`, `phase4-ddp-w4.log`).
- Checker on real Flux artifacts: replicas, oracle, `grads.u1`, `v.u3`,
  beta power, eta, loss, logits, protocol, shards, and update count PASS
  (`/tmp/opencode/phase4-flux/check/phase4_report.txt`).
- Checker self-test: `PHASE4_SELFTEST PASS checks=28`; synthetic end-to-end
  `PHASE4_CHECK PASS categories=190`; negative mutations rejected.
- Split-control: full vs half-mean gradient max abs `2.53e-7` at u1/u3 on
  the initial state.

## Open issues

1. `scripts/static_scaling/phase4_torch.py` does not exist. This blocks the
   PyTorch DDP cells and the cross-framework acceptance categories.
2. Flux `same.flux.w2/w4.grads.u3` fails at max abs `~4e-4` (and `m.u3`
   `~4e-5`), while `grads.u1` is `~2.5e-7`, the exactly-once oracle passes,
   and rank replicas are exact. Hypothesis: two updates amplify tiny
   parameter differences through ReLU/GroupNorm. The recommended proof is a
   single-process shard-arithmetic reference (same local batch sizes,
   averaged gradients) compared to DDP under `2e-6/1e-5`, plus a documented
   full-batch envelope. Do not widen tolerances without that evidence.
3. Remote access failed: Leonardo publickey denied; deathstar DNS
   unresolved. The drivers are untested on real nodes. Nested
   `srun --overlap`, GPU visibility, and torchrun ports must be confirmed on
   the first Leonardo run.
4. A later multi-threaded CPU run failed the accepted Phase 3 path check on
   2 activation elements (1.61e-5 vs 1e-5). Pin `JULIA_NUM_THREADS=1` and
   `OPENBLAS_NUM_THREADS=1` for CPU validation; the earlier single-threaded
   run passed all categories.

## Handoff

See [`temp/phase4-handoff.md`](../../temp/phase4-handoff.md) for the complete
file inventory, commands, decisions, and exact next actions.

## Next action

Implement `scripts/static_scaling/phase4_torch.py`, add the shard-arithmetic
reference mode, resolve the Flux `u3` comparison, then run deathstar CPU/MPI
smoke and Leonardo GPU/NCCL acceptance.

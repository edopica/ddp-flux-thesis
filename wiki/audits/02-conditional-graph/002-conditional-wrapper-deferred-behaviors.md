# 002 — Conditional-wrapper deferred behaviors

Date: 2026-10-06
Scope: behavioral limitations of the `DistributedOptimizerState` conditional-graph
wrapper (Flux branch `ddp/integration-m1-spike`), evidence-based assessment and
deferral recommendations.
Status: investigation complete. No production, test, or Flux source changed.

Source under test: `/home/kurapica/Projects/ddp_flux/flux-integration`,
HEAD `55171e659b2f425da980cbc83177bf5c38f0a7a1` plus the uncommitted Phase H/J
changes (see `artifacts/logs/conditional-wrapper/deferred/deferred-metadata.md`
for the exact blob hashes). Runtime: Julia 1.12.6, Flux 0.16.11, MPI 0.20.27,
Optimisers 0.4.9, MPICH, 2 ranks on CPU.

---

## 1. Supervisor summary

The conditional-graph wrapper is **correct on its supported path**: all finite,
rank-consistent runs probed here match an independent native reference exactly,
and the Phase J/K GPU campaigns passed their full correctness matrices. This
investigation characterized the boundaries **outside** that path. Three
runtime-confirmed failure modes matter, all originating in code the wrapper does
not own (`src/train.jl` control flow plus blocking collectives), not in the
gradient-synchronization algorithm:

1. **Finite loss, non-finite gradient silently corrupts every replica.**
   `_eager_step!` checks only the scalar loss (`flux-integration/src/train.jl:377`).
   A NaN/Inf gradient passes the guard, is averaged by the wrapper, and
   propagates to parameters and Adam moments on all ranks with no exception.
   Runtime-confirmed: `grad-nonfinite` (rank 1 has a NaN `w` gradient while its
   loss is `0.0`; after one update both ranks have `w=NaN` and `mt=NaN`,
   `exc=none`). This is the highest-impact finding because it is silent.
2. **One rank's non-finite loss deadlocks the job under `Flux.train!`.**
   The non-finite rank skips its update (and therefore all collectives) and
   throws `DomainError`; every other rank blocks in the update's first
   `allreduce!` with no timeout. Runtime-confirmed: `nan-loss-hang` (rank 0
   blocked inside presence collective #7; rank 1 held after `DomainError`).
3. **A manual `trainstep!` loop with one skipped update silently shifts
   gradient pairing and then hangs.** `trainstep!` does not throw on a
   non-finite loss, so the skipping rank continues one step ahead; later
   collectives pair rank A's step *k* with rank B's step *k+1*. Runtime-confirmed:
   `step-desync` (rank 1's final state matches the shifted-pairing reference
   exactly, `delta_vs_shifted=0.0`, and differs from the correct reference,
   `delta_vs_correct=0.0499`; rank 0 then blocks at the last step).

All three require a non-finite numerical event (or a rank-local control-flow
decision); none occurred in the finite K1/K2 demonstration campaigns. Rank-local
optimizer configuration (learning rate, frozen flags) was also confirmed to
diverge replicas **silently without hanging**, but that is the accepted
rank-consistent-configuration precondition, already documented in the
`synchronize!!`/`DistributedOptimizer` docstrings and NEWS.

**Recommendation:** the demonstration may proceed on the documented supported
path (finite, rank-consistent data and losses; identical optimizer
configuration on every rank; `DistributedDataContainer` for sharding; no
per-rank early stop). Record findings DB-1/DB-3/DB-4 as prominent limitations
and defer code changes to the post-1.0 batch (backlog in section 7). This report
is a recommendation; accepting the limitations is the user's decision.

---

## 2. Evidence classes and reachability key

Evidence class (per finding):

- **runtime-confirmed** — reproduced in this investigation with a bounded
  2-rank CPU/MPI probe and machine-readable markers.
- **source-only** — established by code trace; no probe was written (reason
  stated).
- **documented/tested** — pinned by existing tests or an accepted contract.
- **untested** — no evidence available in the current environment.

Reachability (per the investigation prompt):

- **Normal supported workflow** — a documented API used as intended.
- **Plausible numerical or data-dependent event** — a normal API plus a
  numerical condition that real training can produce.
- **Explicit advanced configuration or unsupported workflow** — requires a
  deliberate out-of-contract action or misuse.
- **Unknown** — not established.

The probes inject NaN/Inf deliberately. A finite toy/MNIST run does not measure
how often real workloads produce non-finite values; no incidence estimate is
made. Every mechanism below is supported by a source trace or a reproduction,
not by speculation.

---

## 3. Behavior matrix

| ID | Trigger | API path | Ranks affected | Consequence | Evidence | Reachability |
|---|---|---|---|---|---|---|
| DB-1 | One rank's scalar loss becomes NaN/Inf at a step | `Flux.train!` | 1 stops, others hang | Job deadlock; offender throws `DomainError`, others block in the presence `allreduce!`; no corruption | runtime-confirmed | Plausible numerical event, normal workflow |
| DB-2 | Every rank's loss is NaN/Inf at the same step | `Flux.train!` | all | Coordinated clean stop; both skip the update and throw; model/state unchanged | runtime-confirmed | Plausible numerical event, benign |
| DB-3 | Gradient becomes NaN/Inf while the loss stays finite | `train!`, `trainstep!`, explicit `withgradient`+`update!` | all (silent) | NaN parameters and Adam moments on every rank; no exception | runtime-confirmed | Plausible numerical event, normal workflow |
| DB-4 | Manual `trainstep!` loop; one rank's loss is NaN at one step | `Flux.trainstep!` loop | all (silent shift), then hang | Later gradients pair across different steps; wrong trajectory then deadlock; no exception | runtime-confirmed | Plausible numerical event in a documented manual-loop workflow |
| DB-5 | Rank-local early stop / loop break / unequal custom iteration counts | custom loop | 1 stops, others hang | Deadlock at the first unmatched update | runtime-confirmed | Unsupported custom-loop misuse (built-in sharding pads) |
| DB-6 | Rank-local learning-rate schedule (`adjust!` on a subset) | wrapper `update!`/`update` | all (silent) | Replicas diverge without error or hang; collective schedule unchanged | runtime-confirmed | Explicit advanced configuration (accepted precondition) |
| DB-7 | Rank-local frozen flag (`freeze!` on a subset) | wrapper `update!`/`update` | all (silent) | Frozen rank stays put, others move; no error or hang; frozen leaves still reduce | runtime-confirmed | Explicit advanced configuration (accepted precondition) |
| DB-8 | Rule/frozen config or backend expected from `synchronize!!`; root-only checkpoint restore | `synchronize!!`, checkpoint load | — | Contract excludes rules, frozen flags, stored backend, unsupported types; root-only restore gives a valid-looking but inconsistent state | documented/tested (Phase H tests) | Explicit advanced/ops workflow |
| DB-9 | Model structure/ties changed after `setup`; custom optimizer state; broken tied identity | any update path | — | New leaves are not synchronized; changed ties can double-count; custom state is not synchronized | source-only | Unsupported workflow (no probe; see section 4.9) |
| DB-10 | Non-finite behaviors on native CUDA/NCCL; listed GPU-deferred cases | GPU wrapper paths | — | Same mechanisms expected, not exercised; MPI+CuArray, device movement, partial device freeze, non-square device ties untested | untested | Unknown until a GPU probe runs |
| DB-11 | Loss returns aux values with a finite scalar but non-finite aux | `train!`, `trainstep!` | all | Guard/stop look only at `first(v)`; non-finite aux does not stop training or corrupt state | documented/tested + source | Normal workflow, low impact |

Runtime totals: 8 modes, 83 PASS assertions, 0 failures, driver exit 0
(`artifacts/logs/conditional-wrapper/deferred/check-summary-2026-10-06.log`).

---

## 4. Findings

Source references are relative to the implementation worktree
`/home/kurapica/Projects/ddp_flux/flux-integration`.

### 4.1 DB-1 — One-rank non-finite loss: asymmetric participation and deadlock

**Mechanism.** `_eager_step!` computes the gradient first and guards only the
update (`src/train.jl:375-379`):

```julia
v, gs = Flux.withgradient(m -> loss(m, batch...), adtype, model)
isfinite(_loss_scalar(v)) && _update!(opt_state, model, gs[1])
return v, gs[1]
```

The non-finite rank never calls `_update!`, so it never enters
`_sync_gradients` and issues **no collective** for that step
(`src/distributed/public_api.jl:698-727`). `train!` throws afterwards, per rank
(`src/train.jl:240-245`). Every other rank enters `_sync_gradients` and blocks
in the first blocking `allreduce!`. No timeout, abort, or rank agreement exists.

**Runtime evidence.**
`artifacts/logs/conditional-wrapper/deferred/raw-nan-loss-hang-mpi2-2026-10-06.log`:
rank 0 completed steps 1-2 (6 collectives), began presence collective #7 and
never completed it; rank 1 logged `LOSS_NAN rank=1 i=3`,
`NONFINITE_STOP rank=1 item=3 exc=DomainError`, `HOLDING rank=1`. The driver's
watchdog fired only after the startup readiness markers and terminated the job
(exit 15). A timeout alone was not accepted as proof: the driver asserts the
unmatched `COLLECTIVE_BEGIN` plus the stop/hold markers.

**Impact.** High: production-style runs stall until an external kill; no
parameter corruption on the stopping rank because the update was skipped.

**Reachability.** Plausible numerical or data-dependent event in a normal
workflow. `NaN`/`Inf` losses occur with divergent training, bad data, or
overflow. The finite K1/K2 campaigns never produced one.

**Recommendation: document and defer.** The demonstration uses a fixed finite
schedule; the reference-comparison driver would flag a corrupted trajectory.
Deferral conditions: (a) the limitation is prominent in the user-facing notes;
(b) demonstration data/losses are finite on every rank (K2 validation already
shows finite losses and reference matches); (c) a future fix adds a
rank-consistent non-finite decision (backlog A1). Workaround for any user run:
do not rely on `train!`'s per-rank stop; pre-screen data or add an
all-rank loss check.

> **Correction (2026-10-13): DB-1 resolved.** `Flux.train!` alone was **not** a
> workaround for asymmetric non-finite losses: before ADR-0008 it hung exactly
> like the manual loop, because the offending rank skipped its update and
> issued no collective. ADR-0008 replaces the per-rank skip with a coordinated
> all-rank loss vote; every rank now raises `DomainError` before any optimizer
> collective, and the pre-screening workaround above is obsolete. The original
> recommendation is preserved as the pre-fix characterization.

### 4.2 DB-2 — All ranks non-finite: coordinated clean stop

**Mechanism.** Every rank skips its update at the same step, so no rank enters
a collective; every rank then throws `DomainError` from `train!`
(`src/train.jl:244-245`). This is the benign boundary of DB-1.

**Runtime evidence.** `raw-nan-loss-all-mpi2-2026-10-06.log`: both ranks
`NONFINITE_STOP rank=r item=3`; `STOP_STATE_MATCH rank=r caught=true
maxdiff=0.0 state_ok=true`; 6 collectives per rank (steps 1-2 only); exit 0.

**Impact / reachability.** Same trigger as DB-1; consequence is a clean stop,
not a hang. **Recommendation: no action.** Positive characterization.

### 4.3 DB-3 — Finite loss with non-finite gradient: silent corruption

**Mechanism.** No finiteness check exists on gradients anywhere in
`src/distributed/` or the MPI/NCCL extensions; a source-wide scan for
`isfinite`/`isnan`/`isinf` finds only the loss guards at
`src/train.jl:245,377`. The
gradient buffers built by `_walk!` are allreduced unconditionally
(`src/distributed/public_api.jl:723-727`), so one rank's NaN/Inf contaminates
every rank's averaged gradient and, through Adam, the parameters and moments.

**Runtime evidence.** `raw-grad-nonfinite-mpi2-2026-10-06.log`. Loss:
`sqrt(sum(abs2, m.w .- xr))` with residual 1 on rank 0 and residual 0 on rank 1
(forward `0.0`, backward `NaN` at the zero residual). Markers:
`GRAD rank=0 ... wgrad_finite=true`,
`GRAD rank=1 loss=0.0 ... wgrad_finite=false`, then on both ranks
`CORRUPTION rank=r exc=none w_finite=false v_finite=true wmom_finite=false
w1=NaN`. Three collectives (one presence, two gradients), all matched.

**Impact.** High: silent model/optimizer corruption with no exception; the
run continues on NaN parameters until some later loss or metric reveals it.
A finite scalar loss does **not** imply finite gradients.

**Reachability.** Plausible numerical event in a normal workflow (e.g.
zero-norm normalizations, `sqrt`/`log` at a boundary, overflow in the backward
pass). Not observed in any finite campaign.

**Recommendation: document and defer.** This is the finding that most deserves
user-visible wording. Deferral conditions: (a) documented prominently
("finite-loss guard does not check gradients"); (b) the demonstration keeps
losses finite and reference comparisons catch divergence within the declared
tolerances; (c) backlog A2 adds a rank-consistent gradient check, tested with
this probe. Workaround: validate losses/gradients for the intended workload or
add an application-level finite check before `update!`.

> **Resolution (2026-10-14): DB-3 closed.** Implemented and gate-accepted by
> [ADR-0009](../../decisions/ADR-0009-post-reduction-nonfinite-gradient-rejection.md).
> `_sync_gradients` scans every globally present reduced buffer with
> `_reduced_buf_finite` and casts one all-rank validity vote through the
> existing `_any_rank_invalid` helper, after the complete reduction loop and
> before `_build!`; the NCCL extension synchronizes the CUDA stream before the
> device reduction. Any non-finite synchronized gradient raises `DomainError`
> on every rank before the inner optimizer, and a rejected step leaves
> parameters and optimizer state unchanged. Evidence:
> [fix devlog](../../devlog/02-conditional-graph/2026-10-14-db3-gradient-rejection.md)
> and [COMMANDS.md, DB-3 campaign](../../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#db-3-campaign-2026-10-14-deathstar).
> The original recommendation and workaround above are preserved as the pre-fix
> characterization.

### 4.4 DB-4 — Manual `trainstep!` loop: shifted gradient pairing, then deadlock

**Mechanism.** `trainstep!` returns without throwing on a non-finite loss
(`src/train.jl:323-339` through `_eager_step!`). In a manual loop (the
documented alternative to `train!`), the offending rank skips that update and
immediately proceeds to the next iteration, while the other ranks finish the
current one. The next collectives then pair different steps. Because the
skipping rank's replicas stay at the same state, the result is not immediately
detectable: it becomes an Adam step with `mean(g_step_k, g_step_k+1)`. The loop
finally deadlocks when the update counts differ.

**Runtime evidence.** `raw-step-desync-mpi2-2026-10-06.log`. Rank 1's loss is
NaN at step 3 and its loop continues; rank 1's final state matches the
independently computed shifted-pairing reference exactly
(`DESYNC_RESULT rank=1 delta_vs_correct=0.04994879241581662
delta_vs_shifted=0.0`); rank 0 completed 12 collectives and blocked at presence
collective #13 (step 5); rank 1 completed 12 and held.

**Impact.** High: silent wrong trajectory (not just a stop), then a hang. The
exact shifted-pairing match is direct mechanistic proof, not a timeout.

**Reachability.** A plausible numerical event inside a supported manual-loop
workflow. **Recommendation: document and defer.** Deferral conditions: (a)
document that manual loops must check the returned loss on every rank and stop
consistently; (b) the demonstration drivers use `train!`/explicit fixed loops
with finite data; (c) backlog A3 adds an offset-detection or coordinated-stop
guard.

> **Correction (2026-10-13): DB-4 resolved; inspecting the returned loss is too
> late.** In a manual `trainstep!` loop, checking the returned loss cannot
> prevent the pairing shift: the skipping rank has already returned and moved on
> while its peers are inside the skipped step's update collectives, so the
> mismatch exists before any caller can inspect the value. ADR-0008 rejects at
> the `_eager_step!` boundary: the coordinated vote raises `DomainError` on
> every rank at the injected step, before the next batch runs. The original
> recommendation is preserved as the pre-fix characterization.

### 4.5 DB-5 — Unequal rank participation: custom-loop early stop hangs

**Mechanism.** Any rank-local decision that stops calling the wrapper update
before the others (early stop, `break`, exception before the next update, or a
custom data loader with unequal shard lengths) leaves the remaining ranks
blocked in the first unmatched collective. `DistributedDataContainer` prevents
the data-length variant by padding each shard to `cld(N, workers)` with
`mod1`-style cyclic indices (`src/distributed/public_api.jl:309-313,353-365`;
tests `test/ext_distributed/data_distributedtest.jl:21`), and an empty dataset
throws `ArgumentError` (`:350-352`). Process-failure recovery is not supported.

**Runtime evidence.** `raw-early-stop-mpi2-2026-10-06.log`: rank 1 logged
`LOOP_STOP rank=1 i=4 reason=rank-local` and held; rank 0 completed 9
collectives and blocked at presence collective #10 (step 4).

**Impact.** High (job stall). **Reachability:** unsupported custom-loop misuse;
the built-in sharding path is safe. **Recommendation: document and defer.**
Deferral conditions: (a) demonstration loops have a fixed step count and no
rank-local breaks; (b) data uses `DistributedDataContainer`; (c) backlog A4
adds an explicit guard or documented pattern.

### 4.6 DB-6 — Rank-local learning-rate schedule: silent divergence, no hang

**Mechanism.** `_sync_gradients` averages gradients only; each rank applies its
local rule afterwards (`src/distributed/public_api.jl:461-473`). A rank-local
`Optimisers.adjust!` changes the update without changing the collective
schedule, so no rank hangs and no error is raised; replicas diverge silently.
`synchronize!!` does not repair rules (`src/distributed/public_api.jl:244-269,
401-422`), and the Phase H test pins that contract
(`test/ext_distributed/synchronized_distributedtest.jl:162-201`).

**Runtime evidence.** `raw-config-eta-mpi2-2026-10-06.log`: rank 1 called
`Optimisers.adjust!(state, 0.1)` at step 2; each rank matched its own rule
reference exactly (`self_maxdiff=0.0`), and after 4 steps the replicas differed
by `cross_delta=0.14991810934487604`; 12 collectives per rank, no error, exit 0.

**Impact.** Medium-high (silent divergence); **reachability:** explicit
advanced configuration, and the accepted precondition says configuration must
match across ranks. **Recommendation: document (already in the docstrings and
NEWS) and defer.** Deferral conditions: the demonstration applies identical
rules/schedules on every rank; any schedule change is applied identically.
Backlog A5 is optional (a consistency assertion).

### 4.7 DB-7 — Rank-local frozen flag: silent divergence, schedule unchanged

**Mechanism.** `_walk!` selects a child by `t isa Optimisers.Leaf` and never
inspects `Leaf.frozen` (`src/distributed/public_api.jl:609`), so frozen
parameters still take part in the presence and gradient collectives. A
rank-local `freeze!` therefore changes no collective and is not detected:
the frozen rank keeps its parameters while the others update.

**Runtime evidence.** `raw-config-frozen-mpi2-2026-10-06.log`: rank 1
`LOCAL_FREEZE`, `unchanged=true`, rank 0 moved; `cross_delta=0.20012763731269412`;
12 collectives per rank, no error, exit 0. Existing CPU coverage:
`test/ext_distributed/conditional_distributedtest.jl:1408-1450` pins that the
collective protocol ignores the frozen flag.

**Impact / reachability / recommendation.** Same as DB-6: explicit advanced
configuration within the accepted precondition; document and defer. The
frozen-mismatch case is additionally useful evidence that "more localized
computation" does not change the collective schedule.

### 4.8 DB-8 — State, topology, and checkpoint boundaries (accepted contract)

These are documented restrictions, not newly discovered defects; they are
included because the prompt asked for an explicit boundary assessment. Evidence:
`wiki/audits/02-conditional-graph/001-conditional-wrapper-compatibility.md`, the working-tree
docstrings (`src/distributed/public_api.jl:244-269,401-422,739-750`),
`temp/phase_h/03-rule-frozen-contract.md`, `temp/phase_h/04-state-shape-migration.md`,
and Phase H tests.

- `synchronize!!` synchronizes supported values inside `Leaf.state` (arrays and
  isbits scalars, recursively), preserves leaf identity, and leaves `Leaf.rule`,
  `Leaf.frozen`, and the stored backend unchanged; unsupported non-isbits types
  are returned unchanged (`:296-299`). It performs no precondition checks.
- It is **not** a checkpoint restore. Checkpoints are data-only (Serialization
  or JLD2), loaded on every rank, with a fresh process-local backend attached;
  migration uses explicit `.tree` paths. Root-only load plus `synchronize!!`
  does not restore rules or frozen flags (docstring, audit 001).
- `synchronize!!` on a wrapper uses the caller-passed backend, not `s.backend`
  (`:746-750`, source-only).
- Structural gradient pairing is deliberate; the state tree, not the model, is
  the source of truth for which leaves are trainable
  (`src/distributed/public_api.jl:531-546`).
- Composition is rejected (`ArgumentError`) for direct and nested
  `DistributedOptimizer` (`:435-452`); higher-order gradients are rejected
  (`:454-456`); Enzyme `Duplicated` updates are rejected (`:458-459`), while
  `setup` unwraps `Duplicated` (`:445-448`). `Flux.train!` also unwraps
  `Duplicated` before the update (`src/train.jl:427-430`), so the wrapper's
  rejection is bypassed on that path (source-only; the tested behavior is
  setup/rejection in `test/ext_distributed/conditional_distributedtest.jl:1280`).

**Recommendation: document and defer (already accepted).** No change required
for the demonstration; keep the documented loading recipe.

### 4.9 DB-9 — Post-setup model changes, broken ties, custom state (source-only)

No probe was written because these are outside the supported contract; the
assessment is a source trace and is labeled source-only. The prompt requires
flagging such cases rather than silently omitting them.

- **Post-setup model changes.** The update pairs the *old* state tree with the
  current model each call. A child present in the model but absent from the
  tree is treated as non-trainable (`_child(t,k) === nothing`, `_walk!` returns;
  `src/distributed/public_api.jl:570-573,643`), and `Optimisers.update!` is then
  invoked on a model whose structure no longer mirrors the tree. Structural
  changes after `setup` are unsupported by Optimisers itself; re-run `setup`.
- **Broken tied identity.** Caching is keyed by model-array identity
  (`seen::IdDict`, `:611-634`) while `Leaf` identity encodes sharing
  (`mutable struct Leaf` comment, Optimisers `interface.jl:18`). Replacing one
  occurrence of a tied parameter after setup makes the wrapper build two
  buffers while the inner optimizer still accumulates both into the one shared
  leaf, double-counting that parameter. Keep ties stable after setup.
- **Unsupported custom optimizer state.** A custom rule whose state is not a
  supported container is left unchanged by `synchronize!!` (`:296-299`). With
  identical averaged gradients such state evolves identically on every rank,
  but rank-local numerical differences are not reconciled. Documented
  restriction; no unexpected behavior found inside the supported contract.
- **Unknown gradient containers** are treated as absent (`_child(xc,k) =
  nothing`, `:573`); a tied non-array gradient throws a clear `ArgumentError`
  (`:614-617`); a plan-consumption mismatch throws an internal `ArgumentError`
  (`:731-735`). All source-only, defensive errors rather than silent paths.

**Recommendation: document; investigate only if custom optimizers or dynamic
model structure enter scope.** Backlog A6 (optional): detect a structure
mismatch at update time and throw a targeted error.

### 4.10 DB-10 — GPU assessment (native evidence elsewhere, non-finite paths untested)

A separate assessment is required for the CPU prototype and the native GPU
configurations. **No new GPU runs were performed in this investigation.** The
classification below uses the existing Phase J/K evidence:

- Native correctness passes: focused NCCL regression **217/217 assertions per
  rank** at 2 ranks on `deathstar` (2× TITAN RTX) and 4 ranks on `leonardo`
  (4× A100), plus the full dedicated suite **14/14 under MPI and NCCL** at both
  rank counts (`artifacts/logs/conditional-wrapper/phaseJ-metadata.md`,
  `wiki/devlog/02-conditional-graph/2026-10-04-conditional-wrapper-phase-j-native-gpu.md`,
  `wiki/devlog/02-conditional-graph/2026-10-04-leonardo-bringup-4rank-nccl.md`). The K1/K2
  demonstration campaigns pass per-step reference comparison, ordered
  collectives, and fresh-process continuation on CUDA/NCCL at 2 and 4 ranks
  (`README.md` Phase K rows).
- GPU failure records are **harness/environment only**, no product defect:
  cuDNN extension not loaded (`MLDataDevices.functional(CUDADevice)`), a
  Float32 1-ULP tolerance below network-average precision, and a `CUDA`
  reference in a device-generic demo module
  (`wiki/failures/2026-10-04-phaseJ-nccl-harness-failures.md`,
  `wiki/failures/2026-10-04-phasek-nccl-cuda-ref.md`).
- Still untested on GPU (`phaseJ-metadata.md` "Untested / deferred"): MPI with
  `CuArray` parameters; device movement with preserved state; partial
  freeze/thaw on device arrays; transpose of non-square tied device arrays;
  no meaningful isbits GPU analogue. Fresh-process continuation with explicit
  device restoration was later confirmed by the Phase K campaigns for the
  supported path.
- **DB-1/2/3/4/5 were not exercised under NCCL.** Their mechanisms are
  backend-independent (the loss guard is `src/train.jl` control flow; the hang
  is a blocking `allreduce!`, and the presence vector always goes over MPI even
  under NCCL, `src/distributed/public_api.jl:706-720`), so the same behavior is
  expected but **not runtime-confirmed on GPU**. The host presence metadata and
  the non-CUDA-aware MPI staging are documented, not silent parameter
  transfers.

**Recommendation: document and defer to the next GPU batch.** Conditions: the
demonstration uses finite, rank-consistent data (true for K1/K2); backlog A7
ports the non-finite probes to 2-rank NCCL with `allowscalar(false)`.

### 4.11 DB-11 — Auxiliary outputs (minor)

`_loss_scalar` takes only `first(v)` for the guard and the `train!` stop
(`src/train.jl:244,384-385`), while the whole `(loss, aux...)` is returned.
Non-finite aux values neither stop training nor corrupt state; only metrics can
be misleading. Existing coverage: `test/train.jl` aux tests and
`test/ext_distributed/conditional_distributedtest.jl:1317` (`Flux.train!` with
the wrapper). **Recommendation: document only.**

---

## 5. Corrections that must not be reopened

The following were corrected in Phases D-H and are re-confirmed by the current
source and tests; this report found no new evidence against them:

- Scalar optimizer state (Adam `βt`) is synchronized in place, preserving leaf
  identity (`NEWS.md`; `src/distributed/public_api.jl:739-744`;
  `test/ext_distributed/synchronized_distributedtest.jl:268-308`).
- Native tree synchronization no longer loses tied-leaf identity (same tests).
- Array containers keep their shape during gradient rebuild
  (`CartesianIndices` mapping, `src/distributed/public_api.jl:680-696`).
- Trainable selection is read from the setup state tree and is stable across
  updates (`src/distributed/public_api.jl:531-546`).

---

## 6. Deferral assumptions and required workarounds

Consolidated conditions for accepting the recommendation above for the 1.0
demonstration:

| Finding | Assumption for deferral | Concrete workaround |
|---|---|---|
| DB-1 | All ranks' losses stay finite; fixed step schedule | Pre-screen data; if a non-finite event is possible, add an all-rank loss check before `update!` |
| DB-3 | Demonstrations avoid boundary-singular backward paths; reference comparison catches divergence | Check gradients for finiteness in the application before `update!` |
| DB-4 | Manual loops, if used, stop consistently on every rank | Prefer `train!`; in manual loops check each returned loss on every rank before the next update |
| DB-5 | No rank-local early stop; equal iteration counts | Use `DistributedDataContainer` (pads to `cld(N, workers)`); fixed iteration counts |
| DB-6/DB-7 | Identical rules, schedules, and frozen flags on every rank | Apply `adjust!`/`freeze!`/`thaw!` identically on all ranks; keep `.tree` state rank-consistent |
| DB-8 | Data-only checkpoints loaded on every rank with a fresh backend | Keep the documented `.tree` migration recipe; do not root-only restore |
| DB-9 | Model structure and ties are fixed after `setup`; no custom optimizer state | Re-run `setup` after structural changes; keep ties stable; prefer Optimisers' built-in rules |
| DB-10 | GPU untested cases stay out of scope for the demonstration | Use the validated finite K1/K2 configurations |

> **Corrections (2026-10-13).** The DB-1 and DB-4 rows above record pre-fix
> workarounds and are superseded by ADR-0008:
>
> - DB-1: the all-rank loss check is now built into `train!` and the other eager
>   distributed APIs. Note that `Flux.train!` alone was **not** a workaround
>   before the fix; it suffered the same asymmetric hang as the manual loop.
> - DB-4: "Prefer `train!`" was not safe before the fix either (DB-1 affects
>   `train!`), and checking each returned loss is too late: the non-finite rank
>   proceeds to the next iteration while its peers block in the skipped step.
>   The manual loop now raises `DomainError` on the injected step.
>
> **Correction (2026-10-14).** The DB-3 row above records a pre-fix workaround
> and is superseded by
> [ADR-0009](../../decisions/ADR-0009-post-reduction-nonfinite-gradient-rejection.md):
> the distributed wrapper now rejects a non-finite synchronized gradient on
> every rank before the inner optimizer, so the application-level finiteness
> check before `update!` is no longer required for the supported distributed
> paths.

These are conditions for deferral, not an acceptance decision. The user must
approve which limitations the demonstration carries.

---

## 7. Follow-up backlog and acceptance tests

Suggested order; each item is RED-first with the named acceptance test.

- **A1 (DB-1/DB-2, high).** Rank-consistent non-finite-loss policy in
  `train!`/`trainstep!`. Acceptance: a 2-rank probe where one rank's loss is
  NaN either stops all ranks with one error or continues all ranks, with no
  unmatched collective; the current `nan-loss-*` probe is the RED case.
  **Addressed (2026-10-13):** [ADR-0008](../../decisions/ADR-0008-coordinated-nonfinite-loss-rejection.md),
  resolution above, and the [fix devlog](../../devlog/02-conditional-graph/2026-10-13-db1-db4-coordinated-nonfinite-loss.md).
- **A2 (DB-3, high).** Rank-consistent non-finite-gradient handling. Acceptance:
  the `grad-nonfinite` probe must not produce non-finite parameters or state on
  any rank.
  **Addressed (2026-10-14):**
  [ADR-0009](../../decisions/ADR-0009-post-reduction-nonfinite-gradient-rejection.md)
  implements the post-reduction scan and the all-rank vote; the new gradient
  harness and `test/ext_distributed/nonfinite_gradient_distributedtest.jl`
  establish rejection with parameters and optimizer state unchanged
  ([devlog](../../devlog/02-conditional-graph/2026-10-14-db3-gradient-rejection.md),
  [COMMANDS.md](../../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#db-3-campaign-2026-10-14-deathstar)).
- **A3 (DB-4, high).** Manual-loop step-pairing protection or a documented
  coordinated-stop helper. Acceptance: the `step-desync` probe must not match
  the shifted-pairing reference and must not hang.
  **Addressed (2026-10-13):** manual loops reject on every rank at the injected
  step ([ADR-0008](../../decisions/ADR-0008-coordinated-nonfinite-loss-rejection.md));
  harness `loop-one`, `loop-withgradient-one`, and `recover-one` PASS and
  `nonfinite_loss_distributedtest.jl` testset 3 covers the shift. The historical
  `step-desync` probe was not rerun against the fixed source.
- **A4 (DB-5, medium).** Explicit unequal-participation guard or documented
  pattern. Acceptance: the `early-stop` probe produces a clear error (or a
  coordinated stop), not a hang.
- **A5 (DB-6/DB-7, low/optional).** Optional configuration-consistency check at
  update entry. Acceptance: a rank-local `adjust!` or `freeze!` mismatch is
  detected or documented by test.
- **A6 (DB-9, optional).** Structure-mismatch detection at update time.
  Acceptance: a post-setup structural change throws a targeted `ArgumentError`.
- **A7 (DB-10, medium).** GPU port of DB-1..DB-5 probes at 2-rank NCCL with
  `CUDA.allowscalar(false)`, following the Phase J focused-file pattern.
  **Partially addressed (2026-10-13):** DB-1/DB-4 rejection now passes under
  native CUDA/NCCL at 2 ranks (harness `green-deathstar-loss-nccl2-r2`, 12 PASS
  + 1 UNTESTED; suite 16/16). DB-3 and the Duplicated eager route under NCCL
  remain untested.
- **A8 (DB-11, low).** Document the aux-value guard scope in the `train!`
  docstring. **Still open after the 2026-10-13 fix:** the aux-value guard scope
  remains undocumented in the `train!` docstring; DB-11's status is unchanged.

Performance/metadata work remains separate (sections 21-23 of the
implementation plan) and is out of scope for this behavioral report.

---

## 8. Reproduction

Probe and driver (new, thesis repository):

- `scripts/checks/conditional_graph_deferred_probe.jl`
- `scripts/checks/conditional_graph_deferred_check.jl`

Exact command:

```bash
julia --startup-file=no --project=../flux-integration/test \
    scripts/checks/conditional_graph_deferred_check.jl
```

Per-mode raw logs and the summary are under
`artifacts/logs/conditional-wrapper/deferred/`; source/script hashes, versions,
timeouts, and the defect iterations are in
`artifacts/logs/conditional-wrapper/deferred/deferred-metadata.md`.

Key source anchors:

- Loss guard and per-rank stop: `flux-integration/src/train.jl:240-245,375-385`.
- Presence + gradient collectives: `flux-integration/src/distributed/public_api.jl:698-737`.
- Frozen flag ignored by the walk: `.../public_api.jl:609`; test
  `test/ext_distributed/conditional_distributedtest.jl:1408-1450`.
- `synchronize!!` contract: `.../public_api.jl:244-269,739-750`; tests
  `test/ext_distributed/synchronized_distributedtest.jl:162-308`.
- Data padding: `.../public_api.jl:302-366`; tests
  `test/ext_distributed/data_distributedtest.jl:21-66`.
- MPI/NCCL collective implementations:
  `ext/FluxMPIExt/FluxMPIExt.jl:100-143`,
  `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl:63-85`.

---

## 9. Limits of this investigation

- Probes are synthetic NaN/Inf injections at 2 CPU/MPI ranks; they establish
  mechanism and consequence, not real-workload incidence.
- DB-1..DB-7 have new runtime evidence from this investigation; DB-8 is
  documented/tested; DB-9 is source-only; DB-10's non-finite behavior on GPU is
  untested; DB-11 rests on existing single-process/CPU tests.
- No remote/GPU command was run; the GPU section classifies stored evidence
  only.
- The report does not modify Flux, does not commit anything, and does not accept
  limitations on the user's behalf.

---

## Resolution (2026-10-13)

DB-1 and DB-4 are resolved by
[ADR-0008](../../decisions/ADR-0008-coordinated-nonfinite-loss-rejection.md)
(accepted 2026-10-13). Every eager distributed step now casts one all-rank
loss-validity vote before the presence exchange and the gradient reductions
(`_loss_update_allowed` in `src/train.jl` with the distributed specialization
and `_any_rank_invalid` in `src/distributed/public_api.jl`). Any non-finite
scalar loss raises `DomainError` on every rank, the rejected step performs no
optimizer update, and the collective traces of rejected steps are matched.

Evidence:

- [Fix devlog](../../devlog/02-conditional-graph/2026-10-13-db1-db4-coordinated-nonfinite-loss.md)
  and [COMMANDS.md](../../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md):
  exact commands, versions, hashes, and results.
- Harness: `green-deathstar-loss-mpi2/` 13/13 PASS,
  `green-deathstar-loss-mpi4/` 8/8 PASS (one invalid non-root rank of four
  included), `green-deathstar-loss-nccl2-r2/` 12 PASS + 1 UNTESTED
  (`dup-train-one`).
- Dedicated suite: `suite-deathstar/suite-mpi2-r2.log` 8/8,
  `suite-mpi4.log` 8/8, `suite-nccl2.log` 16/16 (MPI + NCCL passes);
  single-process `train` 137/137; `detect_ambiguities` 0.
- New regression: `test/ext_distributed/nonfinite_loss_distributedtest.jl`
  (device-generic, 8 testsets) with single-process compatibility in
  `test/train.jl`.

DB-2 remains a clean coordinated stop, now with a matched loss vote (harness
`train-all` PASS at 2 MPI ranks and 2 NCCL ranks; the 4-rank matrix covers a
single invalid rank of four).

DB-3 remains open under
[conditional-graph-db3-fix-plan.md](../../../reports/conditional-graph-db3-fix-plan.md);
`--group gradient|all` prints `GROUP_UNAVAILABLE` and exits 2, so Phase D
combined acceptance is pending. DB-10 is now partially covered: the DB-1/DB-4
rejection is exercised under native CUDA/NCCL at 2 ranks, while the Duplicated
eager route under NCCL remains untested and DB-3 GPU behavior remains open.

> **DB-3 resolution (2026-10-14).** DB-3 is closed by
> [ADR-0009](../../decisions/ADR-0009-post-reduction-nonfinite-gradient-rejection.md):
> the post-reduction gradient scan and the all-rank vote reject a non-finite
> synchronized gradient on every rank before the inner optimizer, and the
> rejected step leaves parameters and optimizer state unchanged. The section 4.3
> resolution note and the A2 backlog item above record the details and evidence
> ([devlog](../../devlog/02-conditional-graph/2026-10-14-db3-gradient-rejection.md),
> [COMMANDS.md](../../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#db-3-campaign-2026-10-14-deathstar)).
> DB-3 GPU behavior is now exercised under native CUDA/NCCL at 2 ranks: the two
> native overflow fixtures are UNTESTED on NCCL (MPI-only), and the controlled
> injection covers the post-reduction device path.

> **Review-response note (2026-10-13, later the same day).** The harness was
> hardened after external review
> ([report](../../../reports/conditional-graph-db1-db4-review.md)): cleanup is
> now campaign-token-scoped (`reap_campaign`, `--campaign TOKEN`,
> `--selftest-cleanup`) instead of filename-wide, and the `healthy` gate runs a
> four-step rank-distinct conditional eager `train!` trajectory with per-step
> parameter and full Adam-leaf-state comparison against a native reference
> (`STEP_STATE_OK`, `ABSENCE_FROZEN`, `REACTIVATED`). The dedicated test file
> also gained a NamedTuple-loss testset (9 testsets). Re-gated with no Flux
> source change: harness `green2-*` 13/13 (mpi2), 8/8 (mpi4), 12 PASS + 1
> UNTESTED (nccl2); suite v2 8/8, 8/8, 16/16 (MPI + NCCL); 0
> `Precompiling packages...` lines. Evidence:
> [COMMANDS.md v2](../../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#v2-review-response-campaign-2026-10-13),
> [fix devlog, review response](../../devlog/02-conditional-graph/2026-10-13-db1-db4-coordinated-nonfinite-loss.md#review-response-second-pass),
> [failure note](../../failures/2026-10-13-db1-db4-healthy-probe-fielderror.md).
> DB-3 and the Duplicated NCCL route remain as stated above.
>
> **R1 round-2 note (2026-10-14).** The v2 cleanup was additionally hardened to
> exact adjacent-argv matching (`campaign_pids` scans `/proc/<pid>/cmdline` and
> requires the probe filename plus the exact pair `("--campaign", token)`, with
> no regex), re-gated by the extended
> [cleanup selftest](../../../artifacts/logs/conditional-wrapper/numerical-guards/selftest-cleanup-deathstar-v2.log),
> the `green3-*` harness artifacts, and a fresh
> [concurrency check](../../../artifacts/logs/conditional-wrapper/numerical-guards/cleanup-concurrency-deathstar/)
> ([COMMANDS.md v3](../../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#v3-review-response-round-2-2026-10-13)).

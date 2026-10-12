# How Conditional Graph Support Was Implemented in Flux.jl DDP

**Report date:** 2026-10-06
**Prepared for:** supervisor progress meeting
**Implementation branch:** `ddp/integration-m1-spike`
**Implementation worktree:** `/home/kurapica/Projects/ddp_flux/flux-integration`
**Base commit:** `fee2c310` (head of PR #2694)
**Latest commit:** `55171e65`. The Phase H and Phase J corrections are uncommitted on top of it.
**Status:** implemented and tested at two and four ranks on CPU/MPI and CUDA/NCCL. Demonstrated on MNIST/Fashion-MNIST.

This report explains how conditional graph support was implemented. It starts with the problem, then explains each technical component and its global role. It lists the production code changes, the defects found during development, the test evidence, and the current limitations.

---

## Executive summary

A conditional graph uses only some model parameters on some training steps. The base `DistributedOptimizer` in Flux.jl communicated gradients inside the per-leaf `Optimisers.apply!` callback. That callback runs only when a local gradient exists. As a result, the collective schedule followed the local control flow of each rank. Two failure modes followed:

1. **Silent corruption.** Two ranks skipped different branches with the same number of collectives. MPI paired the collectives by call order. Branch A received the average of branch B gradients.
2. **Deadlock.** Two ranks skipped different numbers of branches. One rank waited forever for an unmatched collective.

The implementation corrects both modes with one design change: **synchronize the whole gradient tree once per update, before the inner optimizer runs**. Every rank makes one global presence decision. Every rank then runs the inner optimizer on the same averaged gradients. Globally absent parameters stay absent for the inner optimizer, so the native skip semantics of Optimisers.jl are preserved.

The implementation lives in one production file, `src/distributed/public_api.jl` (+384/−13 lines in the working tree). A focused test file of 1,598 lines pins the behavior at 367/367 assertions per rank at two and four ranks. The external detector now passes 210/210 assertions as a success regression. A CUDA/NCCL regression passes 217/217 assertions per rank at two ranks on `deathstar` and four ranks on Leonardo. The Phase K MNIST/Fashion-MNIST demonstration matches an independent native reference exactly at two ranks, and within declared Float32 tolerances at four ranks.

---

## 1. The problem

**Global role of this section.** This section defines the failure that the implementation must correct. The correct definition of the failure determines the correct shape of the fix.

### 1.1 What a conditional graph is

A conditional graph is a model whose forward pass uses a subset of the parameters on a given step. A common example is a two-head model:

```julia
struct TwoHeadModel
    trunk::Dense
    head_a::Dense
    head_b::Dense
end

(m::TwoHeadModel)(x, head::Symbol) =
    head === :a ? m.head_a(m.trunk(x)) : m.head_b(m.trunk(x))
```

A batch for task `:a` uses `head_a` only. The gradient for `head_b` is absent on that step. Zygote reports the absent gradient as `nothing`. ChainRules reports it as an `AbstractZero` value, for example `ZeroTangent`. Optimisers.jl treats both as "no gradient for this leaf" and skips the update for that leaf.

### 1.2 The failure of the base implementation

The base implementation at `fee2c310` allreduced one gradient inside `Optimisers.apply!`:

```julia
# fee2c310:src/distributed/public_api.jl
function Optimisers.apply!(opt::DistributedOptimizer, state, x, y)
    y_avg = DistributedUtils.allreduce!(opt.backend, y, DistributedUtils.avg)
    return Optimisers.apply!(opt.opt, state, x, y_avg)
end
```

`Optimisers.update!` calls `apply!` only for parameters that have a local gradient. Therefore the number of `allreduce!` calls and their order were a function of the local control flow. Consider a model with branches `a` and `b`:

| Mode | Rank 0 | Rank 1 | Result |
|---|---|---|---|
| Uniform | gradient for `a` | gradient for `a` | correct |
| Silent corruption | gradient for `a` | gradient for `b` | `a` and `b` cross-reduce |
| Deadlock | gradients for `a` and `b` | gradient for `b` | rank 0 waits on an unmatched collective |

MPI matches collectives by call order, not by tag. With equal counts, no error occurs. The wrong values spread to every rank. This is the more dangerous mode, because training continues without a warning.

### 1.3 Why zero-filling is not a solution

An earlier helper, `resolve_unused_parameters!!`, replaced absent gradients with zero gradients. That approach is not correct. Zero is a value. Absent is the lack of a value. Optimisers.jl skips `apply!` for an absent gradient. With a zero gradient, Adam still updates the moment estimates, still advances the decay product, and can take a nonzero step. The zero-fill helper also lacked a global presence decision, so it cannot distinguish "absent on this rank" from "absent on every rank". The project rejected the helper and deleted its public use.

---

## 2. Design of the correction

**Global role of this section.** This section gives the central idea. All code components below implement this idea.

The design separates two concerns:

- **The collective schedule.** One small, fixed sequence of collectives per update. Every rank calls the same collectives in the same order.
- **The per-leaf optimizer walk.** The native Optimisers.jl walk. It stays local and unchanged.

The new protocol per update is:

1. Walk the model and the local gradient tree in canonical model order.
2. Collect one writable buffer per unique trainable parameter.
3. Reduce one presence vector of `Float32` flags with `avg`. A value above zero means "a gradient exists somewhere".
4. Reduce each globally present buffer with `avg`, in ascending buffer index.
5. Rebuild a gradient tree that mirrors the model. Use the averaged buffer for globally present parameters. Use `ZeroTangent` for globally absent parameters and for repeated occurrences of tied parameters.
6. Call the native `Optimisers.update!` or `Optimisers.update` with the rebuilt tree.

The collective count is one presence reduction plus one reduction per globally present buffer. The count does not depend on the local branch of any rank.

The communication stays outside automatic differentiation (AD). The project rule requires this. Section 3.6 gives the code evidence.

---

## 3. Production code changes

**Global role of this section.** All production changes are in one file: `src/distributed/public_api.jl`. The file also contains the older distributed utilities. This section walks through the added components in dependency order. Line numbers refer to the implementation worktree at `55171e65` plus the uncommitted Phase H/J changes.

### 3.1 The state wrapper

**Global role.** The wrapper is the anchor of the design. It carries the distribution backend together with the inner optimizer state. The specialized `update` and `update!` methods dispatch on this type. The communication boundary can move to the whole-model level only because this type exists.

```julia
# src/distributed/public_api.jl:423-428
struct DistributedOptimizerState{B <: AbstractFluxDistributedBackend, T}
    backend::B
    tree::T
end

@functor DistributedOptimizerState (tree,)
```

Only `tree` is a functor child. Therefore `Flux.cpu(state)` and other device transformations reach the inner optimizer state and never touch the backend object. The test `Functor device adaptation keeps the backend` pins this behavior.

### 3.2 Setup and the composition guard

**Global role.** `setup` is the entry point that creates the wrapper. It must also stop invalid rule nesting. If a user nests a `DistributedOptimizer` inside another rule, synchronization stops silently.

```julia
# src/distributed/public_api.jl:435-452
const _COMPOSE_ERROR = "DistributedOptimizer cannot be composed inside OptimiserChain " *
                       "or any nested rule (setup would silently disable gradient " *
                       "synchronization). Wrap the whole rule instead: " *
                       "DistributedOptimizer(backend, OptimiserChain(...))."

function Optimisers.setup(opt::DistributedOptimizer, model)
    opt.opt isa DistributedOptimizer && throw(ArgumentError(_COMPOSE_ERROR))
    return DistributedOptimizerState(opt.backend, Optimisers.setup(opt.opt, model))
end

function Optimisers.setup(opt::DistributedOptimizer, model::EnzymeCore.Duplicated)
    opt.opt isa DistributedOptimizer && throw(ArgumentError(_COMPOSE_ERROR))
    return DistributedOptimizerState(opt.backend, Optimisers.setup(opt.opt, model.val))
end

Optimisers.init(opt::DistributedOptimizer, x::AbstractArray) = throw(ArgumentError(_COMPOSE_ERROR))

Optimisers.apply!(opt::DistributedOptimizer, state, x, y) = throw(ArgumentError(_COMPOSE_ERROR))
```

`Flux.setup` calls `Optimisers.setup`, so the wrapper appears through the public Flux API. The two direct-nesting checks reject `DistributedOptimizer(backend, DistributedOptimizer(...))`. The `init` and `apply!` guards reject nesting inside `OptimiserChain`. The valid form is `DistributedOptimizer(backend, OptimiserChain(...))`: the distributed rule wraps the whole chain.

The inner setup call constructs all leaf state. Optimisers.jl therefore keeps ownership of tied-parameter caching, isbits parameters, frozen flags, and rule-specific state.

### 3.3 The update boundary

**Global role.** `update` and `update!` are the communication boundary. They run exactly once per training step on every rank. They synchronize the gradients first, then call the native optimizer.

```julia
# src/distributed/public_api.jl:461-473
function Optimisers.update!(s::DistributedOptimizerState, model, grads, higher...)
    isempty(higher) || throw(ArgumentError(_HIGHER_ORDER_ERROR))
    grads2 = _sync_gradients(s.backend, s.tree, model, grads)
    _, model2 = Optimisers.update!(s.tree, model, grads2)
    return (s, model2)
end

function Optimisers.update(s::DistributedOptimizerState, model, grads, higher...)
    isempty(higher) || throw(ArgumentError(_HIGHER_ORDER_ERROR))
    grads2 = _sync_gradients(s.backend, s.tree, model, grads)
    tree2, model2 = Optimisers.update(s.tree, model, grads2)
    return (DistributedOptimizerState(s.backend, tree2), model2)
end
```

Three properties are important:

1. **Higher-order gradients are rejected.** Their extra gradient arguments cannot be synchronized. The error message states this.
2. **`update!` mutates the inner tree.** It returns the same wrapper and the inner model result.
3. **`update` returns a fresh wrapper.** It re-wraps the new inner tree with the same backend object.

Enzyme `Duplicated` models are rejected in `update` and `update!`. `Flux.train!` unwraps those models first, so the normal path still works.

### 3.4 Gradient synchronization

**Global role of this subsection.** `_sync_gradients` is the core of the implementation. It replaces the local per-leaf communication with a whole-model, order-independent protocol. The function has three phases: collect, reduce, rebuild.

```julia
# src/distributed/public_api.jl:698-737 (condensed)
function _sync_gradients(backend::AbstractFluxDistributedBackend, tree, model, grads)
    bufs = _SyncBuf[]
    seen = IdDict{Any, Int}()  # unique mutable parameter => index into `bufs`
    plan = _SyncOccurrence[]   # one entry per numeric occurrence, in model order
    _walk!(bufs, seen, plan, tree, model, grads)
    np = length(bufs)
    np == 0 && return grads  # no trainable parameters: no collectives, grads unchanged
    ...
end
```

A model without trainable parameters returns the input gradients unchanged and runs zero collectives. This keeps the empty-model case valid.

#### 3.4.1 Structural traversal

**Global role.** The walk must select exactly the leaves that `Optimisers.setup` selected. It must also pair each model node with its gradient node without KeyPath indexing. Structural pairing handles `Transpose`, `Adjoint`, missing `NamedTuple` fields, and absent gradient subtrees correctly.

```julia
# src/distributed/public_api.jl:553-573
_base(dx) = dx
_base(dx::CRC.Tangent) = CRC.backing(CRC.canonicalize(dx))

function _grad_children(x, x̄)
    (x̄ === nothing || x̄ isa CRC.AbstractZero) && return nothing
    x̄b = _base(x̄)
    return try
        functor(typeof(x), x̄b)[1]
    catch e
        e isa MethodError || rethrow()
        functor(x̄b)[1]  # grad node without a model-type pairing method: use its own structure
    end
end

_child(xc::Nothing, k) = nothing  # whole gradient subtree absent at this container
_child(xc::NamedTuple, k) = haskey(xc, k) ? xc[k] : nothing
_child(xc::Tuple, k::Integer) = 1 <= k <= length(xc) ? xc[k] : nothing
_child(xc::AbstractArray, k::Union{Integer, CartesianIndex}) =
    checkbounds(Bool, xc, k) ? xc[k] : nothing
_child(xc::AbstractDict, k) = haskey(xc, k) ? xc[k] : nothing
_child(xc, k) = nothing  # unknown container: treat the child as absent
```

The state tree is the source of truth for trainability. The code comment at lines 542-546 states the reason: the selection from `Optimisers.setup` is stable across updates. A recomputation from the model breaks legacy no-names tuple selections. After the first update, two equal-valued fields can differ. A field can then silently drop out of synchronization although its `Leaf` remains in the tree.

#### 3.4.2 Buffers and local presence

**Global role.** The buffers hold one gradient contribution per trainable parameter. They isolate the caller gradients from the communication. They also carry the local presence flag that the global presence decision uses.

```julia
# src/distributed/public_api.jl:575-578
mutable struct _SyncBuf{B}
    buf::B      # fresh writable buffer (never aliases a caller gradient)
    present::Bool
end
```

Pass 1 collects the buffers in `_walk!`:

```julia
# src/distributed/public_api.jl:606-636 (condensed)
function _walk!(bufs, seen, plan, t, x, x̄)
    if Optimisers.isnumeric(x)
        t isa Optimisers.Leaf || return  # not selected by `Optimisers.setup`
        g = x̄
        ...
        buf = isbits(x) ? Array(x) : similar(x)
        present = g !== nothing && !(g isa CRC.AbstractZero)
        if present
            copyto!(buf, g)
        else
            fill!(buf, 0)
        end
        push!(bufs, _SyncBuf(buf, present))
        ...
    end
    ...
end
```

Key details:

- `similar(x)` gives a buffer of the same array type and device as the parameter. A CUDA parameter stays on the GPU. An isbits parameter uses a host `Array(x)`, because an isbits value has no device.
- A locally absent gradient fills the buffer with zeros. The buffer still participates in the global reduction if another rank has the gradient.
- The gradient is copied into a fresh buffer. The caller gradient is never mutated. The tests pin this with a read-only gradient wrapper.

#### 3.4.3 Tied parameters and the occurrence plan

**Global role.** A tied (shared) mutable parameter occurs several times in one model. Optimisers.jl accumulates all occurrences into one shared `Leaf`. The synchronizer must reduce one buffer once, and must not emit the global gradient twice. The occurrence plan makes this mapping explicit.

```julia
# src/distributed/public_api.jl:588-591
struct _SyncOccurrence
    buffer_index::Int
    emit::Bool
end
```

`_walk!` pushes one plan entry per numeric occurrence. The `emit` flag is true for the first occurrence of a mutable parameter and for each isbits occurrence. Later occurrences of a tied parameter accumulate into the existing buffer and emit `ZeroTangent` during the rebuild. The inner optimizer walk therefore adds the global gradient once.

#### 3.4.4 The presence protocol

**Global role.** One collective decides, identically on all ranks, which parameters have a gradient anywhere. This decision drives the rest of the update. Every rank must reach the same decision before any rank skips a reduction.

```julia
# src/distributed/public_api.jl:706-727
# Presence protocol: one Float32 `avg` allreduce of 0/1 flags decides — identically
# on every rank — which parameters have a gradient anywhere. The threshold is
# `pres[i] > 0`, not `> 0.5` (wrong for more than two ranks). The flags are host
# metadata, so they are allocated on the CPU, never from a parameter buffer.
pres = zeros(Float32, np)
for i in 1:np
    pres[i] = bufs[i].present ? 1f0 : 0f0
end
DistributedUtils.allreduce!(backend, pres, DistributedUtils.avg)
present = pres .> 0

# Allreduce each present buffer in strictly ascending index on every rank:
# every collective must be called by every rank in the same order.
for i in 1:np
    present[i] && DistributedUtils.allreduce!(backend, bufs[i].buf, DistributedUtils.avg)
end
```

Three points matter:

- The presence vector is host memory (`Vector{Float32}`). A device allocation needs scalar device writes and device synchronization. CUDA rejects those operations when `CUDA.allowscalar(false)` is active. The NCCL backend falls back to MPI for CPU buffers.
- The threshold is `> 0`. A threshold of `> 0.5` fails at more than two ranks, because the averaged flag is `1/R`, `2/R`, and so on.
- The gradient loop uses the shared `present` vector. Every rank therefore calls the same reductions in the same ascending order. A globally absent buffer gets no collective on any rank.

#### 3.4.5 Rebuild and skip semantics

**Global role.** The rebuild pass returns a gradient tree that the native Optimisers.jl walk can consume. It must preserve the native skip semantics: a globally absent parameter must look absent, not zero.

```julia
# src/distributed/public_api.jl:657-673
function _build!(t, x, bufs, present, plan, cursor)
    if Optimisers.isnumeric(x)
        t isa Optimisers.Leaf || return nothing  # not selected by `Optimisers.setup`
        cursor[] += 1
        occ = plan[cursor[]]
        return occ.emit && present[occ.buffer_index] ?
            bufs[occ.buffer_index].buf : CRC.ZeroTangent()
    end
    ch = functor(x)[1]
    isempty(ch) && return nothing
    t === () && return nothing  # empty tree node: no trainable descendant
    return _buildmap(ch) do k, c
        tk = _child(t, k)
        tk === nothing ? nothing : _build!(tk, c, bufs, present, plan, cursor)
    end
end
```

The allreduced buffer appears only at an `emit` occurrence of a globally present parameter. All other numeric occurrences emit `ZeroTangent`. The inner optimizer skips them and leaves the corresponding optimizer state untouched.

The rebuild mirrors the model container structure through `_buildmap`:

```julia
# src/distributed/public_api.jl:680-686
_buildmap(f, ch::NamedTuple) = NamedTuple{keys(ch)}(map(k -> f(k, ch[k]), keys(ch)))
_buildmap(f, ch::Tuple) = Tuple(f(k, ch[k]) for k in 1:length(ch))
# `CartesianIndices` keeps the container's shape and column-major index order:
# mapping over `eachindex` of a multidimensional `Array` would flatten it to a
# vector and the rebuilt gradient would not mirror the model container.
_buildmap(f, ch::AbstractArray) = map(k -> f(k, ch[k]), CartesianIndices(ch))
_buildmap(f, ch::AbstractDict) = Dict(k => f(k, v) for (k, v) in pairs(ch))
```

A final consistency check throws an `ArgumentError` if the rebuild pass does not consume every plan entry. This check detects inconsistent traversal between the two passes.

### 3.5 Whole-model state synchronization

**Global role.** `synchronize!!` is a user-facing utility. It broadcasts optimizer state values from the root rank. It is not part of the training update path.

```julia
# src/distributed/public_api.jl:739-750
function DistributedUtils.synchronize!!(
        backend::AbstractFluxDistributedBackend, ps::Leaf; root::Int=0)
    # Retain scalar replacements in the original leaf, whose identity encodes sharing.
    ps.state = DistributedUtils.synchronize!!(backend, ps.state; root)
    return ps
end

function DistributedUtils.synchronize!!(
        backend::AbstractFluxDistributedBackend, s::DistributedOptimizerState; root::Int=0)
    DistributedUtils.synchronize!!(backend, s.tree; root)
    return s
end
```

The first method writes into `ps.state` directly. This keeps the `Leaf` object identity, which encodes tied-parameter sharing. The second method applies the same operation to the inner state tree. Rules, frozen flags, and the stored backend are not synchronized. These values must match across ranks before training. This is an accepted contract precondition.

### 3.6 Communication stays outside AD

**Global role of this section.** The project requires that no collective runs inside automatic differentiation. This rule keeps the gradient computation and the communication separate. It also prevents silent GPU-to-CPU transfers and AD through communication.

The training loop runs the gradient computation to completion first. Only then does the update run:

```julia
# src/train.jl:375-379
function _eager_step!(loss, adtype, model, batch::Tuple, opt_state)
    v, gs = Flux.withgradient(m -> loss(m, batch...), adtype, model)
    isfinite(_loss_scalar(v)) && _update!(opt_state, model, gs[1])
    return v, gs[1]
end
```

`Flux.withgradient` returns before `_update!` runs. `_update!` calls `Optimisers.update!`, which dispatches to the wrapper method and then to `_sync_gradients`. All collectives live inside `_sync_gradients` (lines 720 and 726).

The distributed primitives are marked non-differentiable:

```julia
# src/distributed/public_api.jl:201
CRC.@non_differentiable allreduce!(::Any...)
```

The same marker covers `bcast!`, `reduce!`, `get_distributed_backend`, `local_rank`, and `total_workers`. Even if user code calls a collective under Zygote, no gradient can flow through it.

The old in-`apply!` communication path is disabled. `Optimisers.init` and `Optimisers.apply!` on the distributed rule throw an `ArgumentError`. Synchronization can therefore happen only in the whole-model pass.

---

## 4. Defects found and corrected during implementation

**Global role of this section.** This section records the defects that the regression gates found. It separates production defects from test-harness defects. The table shows that each defect has a named fix and a regression test.

| Phase | Type | Defect | Root cause | Fix |
|---|---|---|---|---|
| B | Production | Silent branch cross-reduction | Per-leaf `allreduce!` inside `apply!` | Whole-model `_sync_gradients` (`109aad9c`) |
| C review | Production | Direct nesting bypassed the guards | `setup` delegated to the inner rule | Nesting check in both `setup` methods (`f5b87975`) |
| D1 | Production | Reversed named `trainable` reordered collectives | `_walk!` used selection order, `_build!` used model order | Canonical `_foreach_child` traversal in both passes |
| D2 | Production | Wrong buffer for mixed tied/isbits parameters | Rebuild counted occurrences, not buffers | Explicit `_SyncOccurrence` plan and cursor (`cbdd3edc`) |
| D3 | Production | Presence flags landed on the parameter device | `similar(parameter)` allocation | Host `zeros(Float32, np)` (`cbdd3edc`) |
| D4 | Production | Legacy tuple `trainable` failed, then drifted | Selection recomputed from the model each update | Selection read from the state tree (`d6046103`) |
| D5 | Production | Multidimensional array containers flattened | `eachindex` mapping | `CartesianIndices` mapping (`d6046103`) |
| H | Production | Scalar leaf state discarded and tied leaf identity lost | `@set!` rebuilt the `Leaf` | Direct `ps.state` assignment (uncommitted) |
| Onboarding | Test harness | Empty detector trace | Same log path opened twice | One `IO` handle for both streams |
| G | Test harness | Trace parser errors | MPI interleaved the message and the newline | Atomic record writes plus an `END` terminator |
| J | Test harness | `force_cuda` error | cuDNN resolvable but not loaded | `import cuDNN` in the NCCL test setup |

Two production defects deserve more detail.

**D4: legacy tuple `trainable`.** The first correction recomputed the trainable selection from the model on every update with value membership (`c in tr`). At setup, two fields had equal values. After one update, they differed. The selection then changed between steps. The wrapper stopped synchronizing one field, while native Optimisers kept its `Leaf`. The corrected design takes the selection from the state tree produced by `Optimisers.setup`. The selection is stable across updates. The two-pass regression shows the old design diverged at step 2 (`model.b` `[0.55, ...]` against native `[-0.605, ...]`) and the new design matches.

**H: scalar leaf state and tied identity.** The synchronization method rebuilt each leaf with `@set!`. The wrapper discarded the returned tree. Mutable Adam moment arrays synchronized, but Adam scalar decay products did not. Rank 0 and rank 1 then applied different bias corrections. The corrected method assigns `ps.state` in place. This change also keeps the `Leaf` identity that encodes tied-parameter sharing.

---

## 5. Test and demonstration evidence

**Global role of this section.** This section shows how we know the implementation is correct. The project rule is "do not treat loss decrease as proof of correctness". Every claim below compares against an independent reference.

### 5.1 Focused CPU/MPI regression file

`test/ext_distributed/conditional_distributedtest.jl` has 30 testsets and 1,598 lines. It covers:

- Equal-shaped conditional branches at two and four ranks, with analytic global means and rank weighting.
- Canonical traversal order, occurrence mapping, CPU presence metadata, and legacy tuple `trainable`.
- BatchNorm running-statistic exclusion, transpose/adjoint pairing, caller-gradient preservation, globally absent Adam state, and empty models.
- Real-AD multi-step averaging, tied parameters, isbits `FillArrays` parameters, and higher-order rejection.
- Collective protocol counting, `Flux.train!`, nested models, `synchronize!!`, `adjust`/`adjust!`, `freeze!`/`thaw!`, and Functor device adaptation.

Result: **367/367 assertions per rank at two ranks and at four ranks** on `deathstar`, exit 0. The four-rank test proves the analytical weighting: rank 0 uses branch A with weight 1/4, ranks 1-3 use branch B with weight 3/4. Every rank asserts both updated values.

The composition regression file `test/ext_distributed/optimizer_distributedtest.jl` passes **6/6 per rank**. The complete dedicated suite passes **6/6** child files at two ranks and four ranks.

### 5.2 The external detector

The detector has a probe under `mpiexec` and a single-process driver:

- `scripts/checks/conditional_deadlock_probe.jl`
- `scripts/checks/conditional_deadlock_check.jl`
- Make target: `make verify-conditional-deadlock FLUX_REPO_PATH=../flux-integration`

The driver checks an independent analytical trace for each mode. The trace contains the presence reduction, then one gradient reduction per globally present branch. The driver checks exact kinds, data types, operations, inputs, and outputs. The expected parameter values are:

- `deadlock` mode: `(trunk, a, b, c) = (1.0, 1.1, 1.4, 1.2)`
- `corruption` mode: `(1.0, 1.1, 1.4, 1.0)`
- `control` mode: `(1.0, 1.2, 1.8, 1.4)`

Before the fix, the detector characterized the bug and treated a hang as a pass. Phase G converted it into a success regression. Result: **210/210 assertions, exit 0, 1 m 7.1 s**. The closing line is now `DETECTOR SUMMARY: conditional-graph support works (CPU/MPI, two ranks)`.

### 5.3 Native CUDA/NCCL regression

`test/ext_distributed/cuda_conditional_distributedtest.jl` uses real `CuArray` parameters with `CUDA.allowscalar(false)`. The file is device-generic: the same body runs on host `Array` under MPI. It covers rank-to-device mapping, six-step conditional branches, globally absent branches with nonzero Adam history, tied device parameters, excluded device arrays, transpose parent-space pairing, and freeze/thaw.

Result: **217/217 assertions per rank**, 6 testsets, exit 0, at two ranks on `deathstar` (TITAN RTX) and four ranks on Leonardo (A100). The dedicated suite passes **14/14** child runs under both MPI and NCCL at both rank counts. No production source changed for this phase.

### 5.4 Phase K dataset demonstration

The Phase K harness compares every update against an independent single-process native Adam reference. The reference consumes the same saved multi-rank schedule. It never reads wrapper output. The campaign checks:

- Every trainable parameter, both Adam moments, decay products, rule configuration, frozen flags, shape, dtype, and device.
- Actual local gradient absence against reference global absence.
- Ordered collective traces on every rank and every step: 7 collectives when both heads are present, 5 when one head is globally absent.
- Data-only checkpoints with fresh-process continuation.
- Rank-0-only output files.

Results on MNIST/Fashion-MNIST, 120 updates, global batch 128:

| Configuration | Train/resume exit | Parameter/moment maxima | `resume_match` | Driver |
|---|---|---|---|---|
| CPU/MPI, 2 ranks | 0/0 | 0.0 / 0.0 / 0.0 | 236/236 | 13/13 |
| CPU/MPI, 4 ranks | 0/0 | 7.451e-09 / 9.313e-10 / 9.095e-12 (train) | 472/472 | 13/13 |
| CUDA/NCCL, 2 ranks | 0/0 | 0.0 / 0.0 / 0.0 | 236/236 | 6/6 |
| CUDA/NCCL, 4 ranks (Leonardo) | 0/0 | 5.960e-08 / 4.657e-09 / 1.455e-11 | 472/472 | 6/6 |

Declared tolerances: Float32 `atol=2e-6`, `rtol=1e-5`. Float64 tolerances are `atol=1e-12`, `rtol=1e-10`. Tolerances were not changed during the campaign. The nonzero four-rank maxima are Float32 accumulation differences inside the declared tolerances. At two ranks the maxima are exactly zero. K1 toy results pass 9/9 driver cells on CPU/MPI, 6/6 on NCCL at two ranks, and 8/8 on NCCL at four ranks.

Learning-curve figures (41 PNGs plus metadata) live under `reports/figures/phase-k-k2/`. The curves illustrate the run. The per-step reference agreement is the correctness evidence.

### 5.5 Optional timing stretch

A separate Leonardo campaign compares one A100 against four A100 ranks on the same fixed work. The corrected ratio-of-medians results are:

- Training-only speedup: `0.263440 / 0.139490 = 1.888594`
- Total speedup: `0.438678 / 0.313939 = 1.397335`

These values are diagnostic for a small model. They are not a production speedup claim. An earlier instrumented value, 1.480203, is superseded. That run took asymmetric host copies in the timed baseline.

---

## 6. Code change map and reproduction

**Global role of this section.** This section lists the changed files and the exact commands. It gives the supervisor a map from claims to files.

Production file:

| File | Change | Role |
|---|---|---|
| `src/distributed/public_api.jl` | +384/−13 lines (working tree) | Wrapper, setup/update boundary, synchronization passes, guards, `synchronize!!` |

Test and documentation files in the implementation worktree:

| File | Change | Role |
|---|---|---|
| `test/ext_distributed/conditional_distributedtest.jl` | +1,598 lines, 30 testsets | Main focused regression |
| `test/ext_distributed/optimizer_distributedtest.jl` | +38/−3 lines | Composition and wrapper-equivalence checks |
| `test/ext_distributed/synchronized_distributedtest.jl` | +223 lines | Scalar state and tied-leaf synchronization |
| `test/ext_distributed/cuda_conditional_distributedtest.jl` | 451 lines (new, untracked) | Native CUDA/NCCL regression |
| `test/Project.toml`, `test/ext_distributed/distributed_setup.jl`, `test/ext_distributed/runtests.jl` | small edits | CUDA/NCCL test dependencies and routing |
| `NEWS.md` | +9 lines | User-facing note |

Commit history on `ddp/integration-m1-spike` after the base:

```text
55171e65 Add Phase E conditional wrapper tests and restore M1.1 API checks
d6046103 Read trainable selection from setup state and keep array containers shaped
cbdd3edc Correct distributed gradient traversal and CPU presence metadata
f5b87975 Reject directly nested DistributedOptimizer at setup
a18392dd Add nested DistributedOptimizer composition regression (RED)
109aad9c Synchronize whole-model gradients before optimizer updates
8add0e42 Strengthen conditional-graph RED test preconditions
6065ce8f Add conditional-graph distributed corruption regression (RED)
```

Exact commands:

```bash
# Focused conditional file, two ranks
FLUX_TEST_DISTRIBUTED_BACKEND=mpi mpiexecjl --project=test -n 2 \
  julia --startup-file=no --project=test \
  test/ext_distributed/conditional_distributedtest.jl

# Complete dedicated suite, two or four ranks (thesis repository)
make c8-mpi   FLUX_REPO_PATH=../flux-integration
make c8-mpi-4 FLUX_REPO_PATH=../flux-integration
make c8-nccl  FLUX_REPO_PATH=../flux-integration

# External detector (thesis repository)
make verify-conditional-deadlock FLUX_REPO_PATH=../flux-integration

# Bounded deferred-behavior probe (thesis repository)
julia --startup-file=no --project=../flux-integration/test \
  scripts/checks/conditional_graph_deferred_check.jl
```

---

## 7. Current limitations and deferred behaviors

**Global role of this section.** This section separates the supported path from the deferred path. The supervisor must know which failures remain open.

Audit 002 (`wiki/audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md`) records the deferred behaviors. The most important points are:

- **DB-1: one rank with a non-finite loss under `Flux.train!` hangs.** The offending rank skips its update and throws. The other ranks block in the presence collective. The loss guard is in `src/train.jl:377`, outside the wrapper.
- **DB-3: a finite loss with a non-finite gradient corrupts every replica silently.** The guard checks only the scalar loss. A NaN gradient is averaged and propagates to parameters and Adam moments. No exception occurs.
- **DB-4: a manual `trainstep!` loop with one skipped update shifts the step pairing and then hangs.** `trainstep!` does not throw. The skipping rank runs one step ahead.
- **DB-5: a rank-local loop stop hangs the remaining ranks.**
- **DB-6 and DB-7: rank-local learning-rate changes or frozen-flag changes diverge replicas silently.** This is an accepted configuration precondition. All ranks must apply `adjust!` and `freeze!` together.
- **DB-8: `synchronize!!` is not a checkpoint restore.** It does not synchronize rules, frozen flags, or the backend. Checkpoints use data-only payloads with explicit `.tree` paths and a fresh backend on every rank.
- **DB-10: the non-finite behaviors are not exercised under CUDA/NCCL.** The mechanisms are backend-independent, but runtime confirmation is pending.

The supported path requires finite losses, rank-consistent configuration, equal step counts, and a fixed model structure after `setup`. The demonstration runs meet these conditions. Backlog items A1 to A8 in the audit define the follow-up work. The user must decide which limitations the demonstration carries.

One further scope limit: the GPU protocol uses host presence metadata through MPI and device gradients through NCCL. It is not an entirely NCCL-only protocol.

---

## 8. Recommended next steps

1. Review the deferred findings and decide the demonstration preconditions (audit 002, section 6).
2. Implement `notebook/conditional_graph_benchmark.ipynb` from the corrected `timing/stretch-2026-10-06-leonardo/` artifacts. This is the remaining notebook deliverable.
3. Discuss the wrapper with the Optimisers.jl maintainers as the path to a shared Flux/Lux solution.
4. Keep the production changes on `ddp/integration-m1-spike` until a separate review accepts an upstream integration plan.

---

## Appendix A: commit references

| Commit | Description |
|---|---|
| `fee2c310` | Base: PR #2694 head, without conditional graph support |
| `6065ce8f` | Add conditional-graph distributed corruption regression (RED) |
| `8add0e42` | Strengthen conditional-graph RED test preconditions |
| `109aad9c` | Synchronize whole-model gradients before optimizer updates |
| `a18392dd` | Add nested `DistributedOptimizer` composition regression (RED) |
| `f5b87975` | Reject directly nested `DistributedOptimizer` at setup |
| `cbdd3edc` | Correct distributed gradient traversal and CPU presence metadata |
| `d6046103` | Read trainable selection from setup state and keep array containers shaped |
| `55171e65` | Add Phase E conditional wrapper tests and restore M1.1 API checks |

## Appendix B: terms

| Term | Meaning |
|---|---|
| Conditional graph | A model that uses only some parameters on some steps |
| Absent gradient | `nothing` or `AbstractZero` returned for an unused parameter |
| Presence decision | The global result of the averaged `Float32` flag reduction |
| Presence collective | The single `allreduce` that carries the presence flags |
| Occurrence plan | The per-occurrence list that maps numeric occurrences to buffers |
| Emit occurrence | The occurrence that emits the synchronized buffer. Tied repeats do not emit it |
| State tree | The inner optimizer state returned by `Optimisers.setup` |
| Wrapper | `DistributedOptimizerState`, which holds the backend and the state tree |

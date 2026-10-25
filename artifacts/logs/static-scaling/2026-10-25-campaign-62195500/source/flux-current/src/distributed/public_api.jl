# ==============================================
#          inspired by avik-pal's Lux.jl
# ==============================================

module DistributedUtils

using ChainRulesCore: ChainRulesCore
using ..Flux: AbstractFluxDistributedBackend, MPIBackend, NCCLBackend, Chain, EnzymeCore
import ..Flux: Train
using MLDataDevices: get_device, AbstractDevice
using Functors: fmap, functor, @functor
using MLUtils: MLUtils, numobs
using Optimisers: Optimisers, AbstractRule, Leaf
using Random: Random
using Setfield: @set!

const CRC = ChainRulesCore

const NCCL_Initialized = Ref(false)
const MPI_Initialized = Ref(false)

"""
    initialized(backend::Type{<:AbstractFluxDistributedBackend})

Check if the given backend is initialized.
"""
initialized(::Type{<:MPIBackend}) = MPI_Initialized[]
initialized(::Type{<:NCCLBackend}) = NCCL_Initialized[]

"""
    initialize(backend::Type{<:AbstractFluxDistributedBackend}; kwargs...)

Initialize the given backend. Users can supply `cuda_devices` and `amdgpu_devices` to
initialize the backend with the given devices. These can be set to `missing` to prevent
initialization of the given device type. If set to `nothing`, and the backend is functional
we assign GPUs in a round-robin fashion. Finally, a list of integers can be supplied to
initialize the backend with the given devices. A guardrail runs before `MPI.Init()` —
including the `MPI.Init()` performed while bootstrapping `NCCLBackend`: it errors when
the job was started by a PMIx/OpenMPI launcher (i.e. `PMIX_RANK` or
`OMPI_COMM_WORLD_RANK` is set in the environment) while MPICH is the loaded MPI
library, a known unsafe combination that aborts `MPI.Init()` at the native level.
System OpenMPI launched with PMIx passes the check. Pass `force=true` to the same
`initialize(...)` call (for either `MPIBackend` or `NCCLBackend`) to bypass the
guardrail entirely (expert use only).

Possible values for `backend` are:

  - `MPIBackend`: MPI backend for distributed training. Requires `MPI.jl` to be installed.
  - `NCCLBackend`: NCCL backend for CUDA distributed training. Requires `CUDA.jl`,
    `MPI.jl`, and `NCCL.jl` to be installed. This also wraps `MPI` backend for non-CUDA
    communications.
"""
function initialize(backend::Type{<:AbstractFluxDistributedBackend}; kwargs...)
    # initialized(backend) && return
    __initialize(backend; kwargs...)
    return
end

function __initialize end

"""
    __check_launcher_compat(env; library, force=false, mpi_initialized=false)

Internal helper. Users should call `initialize` instead.

Pure, MPI-free compatibility check between the environment a process was launched
in and the name of the loaded MPI library (as reported by `MPI.MPI_LIBRARY`, e.g.
`"MPICH"`, `"OpenMPI"`, `"MPIwrapper"`). Used by the MPI backend guardrail,
which runs only before `MPI.Init()`.

Returns `nothing` when the launch environment is compatible with the loaded
library, or a non-empty `String` error message when a known unsafe mismatch is
detected. The only combination judged unsafe is MPICH loaded while a PMIx/OpenMPI
launcher started the job (`PMIX_RANK` and/or `OMPI_COMM_WORLD_RANK` set). System
OpenMPI with PMIx passes, non-MPICH non-OpenMPI libraries are never judged, and
PMI2/SLURM/PMI1-era variables alone never trigger an error.

Passing `force=true` (expert bypass) or `mpi_initialized=true` (the guard only
runs before `MPI.Init()`) always returns `nothing`.
"""
function __check_launcher_compat(env::AbstractDict;
        library::AbstractString, force::Bool=false, mpi_initialized::Bool=false)
    (force || mpi_initialized) && return nothing
    library == "MPICH" || return nothing
    trigger = [v for v in ("PMIX_RANK", "OMPI_COMM_WORLD_RANK") if haskey(env, v)]
    isempty(trigger) && return nothing

    vars = join(trigger, " and ")
    return string(
        "Distributed initialization failed: found ", vars, " in the environment, ",
        "but the loaded MPI library is ", library, ". PMIx/OpenMPI launchers set ",
        "these variables, and the loaded ", library,
        " build was not started by such a launcher, so `MPI.Init()` will likely ",
        "abort or hang at the native level. Relaunch with a launcher matching the ",
        "loaded MPI library (e.g. `mpiexecjl` for MPI.jl's default MPICH_jll, or ",
        "`srun --mpi=pmi2` on SLURM), configure MPI.jl to use your system MPI via ",
        "`MPIPreferences.use_system_binary()`, or bypass this check by passing ",
        "`force=true` to the same `initialize` call.")
end

"""
    get_distributed_backend(backend::Type{<:AbstractFluxDistributedBackend})

Get the distributed backend for the given backend type. Possible values are:

  - `MPIBackend`: MPI backend for distributed training. Requires `MPI.jl` to be installed.
  - `NCCLBackend`: NCCL backend for CUDA distributed training. Requires `CUDA.jl`,
    `MPI.jl`, and `NCCL.jl` to be installed. This also wraps `MPI` backend for non-CUDA
    communications.

!!! danger

    `initialize(backend; kwargs...)` must be called before calling this function.
"""
function get_distributed_backend(backend::Type{<:AbstractFluxDistributedBackend})
    initialized(backend) ||
        error("Backend `$(backend)` is not initialized. Call `DistributedUtils.initialize` first.")
    return __get_distributed_backend(backend)
end

function __get_distributed_backend end

CRC.@non_differentiable get_distributed_backend(::Any...)

"""
    local_rank(backend::AbstractFluxDistributedBackend)

Get the local rank for the given backend.
"""
function local_rank end

CRC.@non_differentiable local_rank(::Any...)

"""
    total_workers(backend::AbstractFluxDistributedBackend)

Get the total number of workers for the given backend.
"""
function total_workers end

CRC.@non_differentiable total_workers(::Any...)

"""
    bcast!(backend::AbstractFluxDistributedBackend, sendrecvbuf; root::Int=0)
    bcast!(backend::AbstractFluxDistributedBackend, sendbuf, recvbuf; root::Int=0)

Backend Agnostic API to broadcast the given buffer `sendrecvbuf` or `sendbuf` to all
workers into `recvbuf`. The value at `root` will be broadcasted to all other workers.
"""
function bcast!(backend::AbstractFluxDistributedBackend, sendrecvbuf; root::Int=0)
    return __bcast!(backend, sendrecvbuf, get_device(sendrecvbuf); root)
end

function bcast!(backend::AbstractFluxDistributedBackend, sendbuf, recvbuf; root::Int=0)
    send_dev = get_device(sendbuf)
    recv_dev = get_device(recvbuf)
    if send_dev == recv_dev
        return __bcast!(backend, sendbuf, recvbuf, send_dev; root)
    else
        sendbuf_ = sendbuf |> recv_dev
        @warn "`sendbuf` and `recvbuf` are on different devices." maxlog=1
        return __bcast!(backend, sendbuf_, recvbuf, recv_dev; root)
    end
end

function __bcast! end

CRC.@non_differentiable bcast!(::Any...)

function avg end

"""
    allreduce!(backend::AbstractFluxDistributedBackend, sendrecvbuf, op)
    allreduce!(backend::AbstractFluxDistributedBackend, sendbuf, recvbuf, op)

Backend Agnostic API to perform an allreduce operation on the given buffer `sendrecvbuf` or
`sendbuf` and store the result in `recvbuf`.

`op` allows a special `DistributedUtils.avg` operation that averages the result across all
workers.
"""
function allreduce!(backend::AbstractFluxDistributedBackend, sendrecvbuf, op::F) where {F}
    return __allreduce!(backend, sendrecvbuf, op, get_device(sendrecvbuf))
end

function allreduce!(
        backend::AbstractFluxDistributedBackend, sendbuf, recvbuf, op::F) where {F}
        send_dev = get_device(sendbuf)
    recv_dev = get_device(recvbuf)
    if send_dev == recv_dev
        __allreduce!(backend, sendbuf, recvbuf, op, send_dev)
    else
        sendbuf_ = sendbuf |> recv_dev
        @warn "`sendbuf` and `recvbuf` are on different devices." maxlog=1
        __allreduce!(backend, sendbuf_, recvbuf, op, recv_dev)
    end
    return
end

function __allreduce! end

CRC.@non_differentiable allreduce!(::Any...)

"""
    reduce!(backend::AbstractFluxDistributedBackend, sendrecvbuf, op; root::Int=0)
    reduce!(backend::AbstractFluxDistributedBackend, sendbuf, recvbuf, op; root::Int=0)

Backend Agnostic API to perform a reduce operation on the given buffer `sendrecvbuf` or
`sendbuf` and store the result in `recvbuf`.

`op` allows a special `DistributedUtils.avg` operation that averages the result across all
workers.
"""
function reduce!(
        backend::AbstractFluxDistributedBackend, sendrecvbuf, op::F; root::Int=0) where {F}
    return __reduce!(backend, sendrecvbuf, op, get_device(sendrecvbuf); root)
end

function reduce!(backend::AbstractFluxDistributedBackend,
        sendbuf, recvbuf, op::F; root::Int=0) where {F}
    send_dev = get_device(sendbuf)
    recv_dev = get_device(recvbuf)
    if send_dev == recv_dev
        __reduce!(backend, sendbuf, recvbuf, op, send_dev; root)
    else
        sendbuf_ = sendbuf |> recv_dev
        @warn "`sendbuf` and `recvbuf` are on different devices." maxlog=1
        __reduce!(backend, sendbuf_, recvbuf, op, recv_dev; root)
    end
    return
end

function __reduce! end

CRC.@non_differentiable reduce!(::Any...)

## As Flux model is an arbitrary type it's not possible to dispatch `synchronize!!`
## end user needs to wrap Flux model into `FluxDistributedModel`
## e.g. model = DistributedUtils.synchronize!!(backend, FluxDistributedModel(model); root=0)
struct FluxDistributedModel{M}
    model::M
end

# synchronize!
"""
    synchronize!!(backend::AbstractFluxDistributedBackend, ps; root::Int=0)

Synchronize supported values in `ps` from `root` using the given backend.
Retain the returned value because immutable values and containers can require replacement.
Tuples, named tuples, and arrays recurse into their elements. Arrays with isbits
elements and isbits scalar values use backend broadcasts.
Other types without a synchronization method remain unchanged.

For an `Optimisers.Leaf`, only supported values within `Leaf.state` synchronize.
The original leaf identity, including shared leaves for tied parameters, is preserved.
The rule configuration (`Leaf.rule`) and frozen flag (`Leaf.frozen`) remain local.
The same contract applies to the inner tree of a `DistributedOptimizerState`.
Its stored backend remains unchanged.

All ranks must supply compatible structure, types, array shapes, and sharing topology.
Every rank must call the same collectives in the same order, with the same `root`.
Before distributed updates, model values, optimizer state, rule configuration, and frozen
flags must agree across ranks. This function does not check these preconditions.
Apply `Optimisers.adjust!`, `freeze!`, and `thaw!` consistently on every rank.
Different local configuration can produce different parameters despite identical averaged gradients.

This function is not a complete checkpoint restore operation.
For optimizer continuation, load the same model and inner optimizer tree on every rank,
then attach a fresh process-local backend. Root-only loading followed by this function
does not restore rule configuration or frozen flags. Model values require separate synchronization.
"""
function synchronize!!(backend::AbstractFluxDistributedBackend, model::FluxDistributedModel; root::Int=0)
    return fmap(x -> synchronize!!(backend, x; root), model.model)
end

function synchronize!!(backend::AbstractFluxDistributedBackend, ps::Tuple; root::Int=0)
    length(ps) == 0 && return ps
    return map(x -> synchronize!!(backend, x; root), ps)
end

function synchronize!!(backend::AbstractFluxDistributedBackend,
        ps::NamedTuple{fields}; root::Int=0) where {fields}
    length(ps) == 0 && return ps
    return NamedTuple{fields}(map(x -> synchronize!!(backend, x; root), values(ps)))
end

function synchronize!!(
        backend::AbstractFluxDistributedBackend, ps::AbstractArray{T}; root::Int=0) where {T}
    if isbitstype(T)
        bcast!(backend, ps; root)
        return ps
    end
    return map(x -> synchronize!!(backend, x; root), ps)
end

# if no method for a given type, just return the value
function synchronize!!(backend::AbstractFluxDistributedBackend, ps::T; root::Int=0) where {T}
    isbitstype(T) && return bcast!(backend, [ps]; root)[]
    return ps
end

# data container
"""
    DistributedDataContainer(backend::AbstractFluxDistributedBackend, data)

`data` must be compatible with `MLUtils` interface. The returned container is compatible
with `MLUtils` interface and is used to partition the dataset across the available
processes.

The container does not modify the underlying dataset. It only pads its internal index
sequence so that every process receives a shard of the same length. When the number of
observations `N = numobs(data)` is not divisible by the number of processes, the padding
is drawn by repeating indices cyclically from the start of the dataset (`mod1`-style),
keeping every index in `1:N`. Each shard therefore has length `cld(N, workers)`.

Because the padding duplicates observations, those duplicated observations are included
in training batches and therefore receive extra weight in the averaged gradients, which
changes the effective training objective. They also affect epoch accounting and any
metric computed over the sharded or padded data (for example epoch counts, losses, or
sums). Metrics meant to describe the original dataset must exclude or otherwise account
for the duplicates.

An empty dataset (`numobs(data) == 0`) is rejected with an `ArgumentError`.

!!! danger

    `MLUtils.jl` must be installed and loaded before using this.
"""
struct DistributedDataContainer
    data
    idxs
end

function DistributedDataContainer(backend::AbstractFluxDistributedBackend, data)
    return __construct_distributed_data_container(backend, data)
end

Base.length(ddc::DistributedDataContainer) = length(ddc.idxs)

Base.getindex(ddc::DistributedDataContainer, i) = getindex(ddc.data, ddc.idxs[i])

function MLUtils.getobs(dc::DistributedDataContainer, idx)
    return MLUtils.getobs(dc.data, dc.idxs[idx])
end

function __construct_distributed_data_container(
        backend::AbstractFluxDistributedBackend, data)
    total_size = numobs(data)
    split_across = total_workers(backend)

    total_size == 0 &&
        throw(ArgumentError("cannot build a DistributedDataContainer for an empty " *
                            "dataset (numobs(data) == 0)"))
    size_per_worker = cld(total_size, split_across)

    # Pad the index list so that it is evenly divisible by the number of workers.
    # Repeats are drawn cyclically from the start of the dataset, keeping every
    # index in `1:total_size` and the original indices `1:total_size` before any
    # repeats.
    total_padded = size_per_worker * split_across
    indices = [mod1(i, total_size) for i in 1:total_padded]

    partitions = collect(Iterators.partition(indices, size_per_worker))
    idxs = collect(partitions[local_rank(backend) + 1])

    return DistributedDataContainer(data, idxs)
end

# Distributed Optimizer
"""
    DistributedOptimizer(backend::AbstractFluxDistributedBackend, optimizer)

Wrap the `optimizer` in a `DistributedOptimizer`. When set up with `Optimisers.setup` (or
`Flux.setup`), this returns a [`DistributedOptimizerState`](@ref
DistributedUtils.DistributedOptimizerState). Every `update`/`update!` of that state
averages the gradients across all processes — one `allreduce` per globally-present
unique trainable parameter, plus one presence `allreduce` per step, then one all-rank
gradient-validity vote — immediately before the wrapped optimizer is applied locally,
so that every rank takes the same step. The vote runs for every update with at least
one trainable parameter, including updates where every gradient is globally absent;
only a model with no trainable parameters issues no collectives. When any rank reports
a non-finite synchronized gradient, every rank raises a `DomainError` before the
wrapped optimizer runs, so parameters and optimizer state are unchanged.

## Restrictions

  - `DistributedOptimizer` must be the outermost rule: composing it inside
    `OptimiserChain` or any other nested rule is not supported, and throws an
    `ArgumentError` at `setup` (it would otherwise silently disable gradient
    synchronization).
  - Model values, optimizer state, rule configuration, and frozen flags must agree
    across ranks before updates. Configuration changes must occur consistently on
    every rank. `synchronize!!` does not synchronize rules or frozen flags.
  - Only first-order gradients are supported: higher-order gradient arguments to
    `update`/`update!` are rejected with an `ArgumentError`.

## Arguments

  - `optimizer`: An Optimizer compatible with the Optimisers.jl package

"""
struct DistributedOptimizer{B <: AbstractFluxDistributedBackend} <: AbstractRule
    backend::B
    opt
end

"""
    DistributedOptimizerState(backend, tree)

State returned by `Optimisers.setup(rule::DistributedOptimizer, model)` and hence by
`Flux.setup`. It wraps the state `tree` of the inner rule together with the distributed
`backend`; the inner state is accessible as `state.tree`.

Gradient synchronization mirrors Optimisers' trainable-parameter semantics: the
model is paired with the state `tree` produced by `Optimisers.setup`, whose leaves
record the parameters selected at setup (down to `Optimisers.isnumeric` leaves), so
the selection cannot change between updates. Tied (shared, mutable) parameters are
reduced exactly once, while isbits parameters are reduced once per occurrence. On
each `Optimisers.update`/`update!` the gradients of the model are first averaged
across all processes — one `allreduce` per globally-present unique trainable
parameter, in the order of the model structure, plus one presence `allreduce` per
step — and then every globally present reduced buffer is scanned for finiteness. One
all-rank gradient-validity vote follows before the result is applied by the inner
rule. The vote runs for every update with at least one canonical trainable buffer,
including updates where every gradient is globally absent. When any rank reports a
non-finite synchronized gradient, every rank raises a `DomainError` before the inner
rule runs, so parameters and optimizer state are unchanged.
[`synchronize!!`](@ref DistributedUtils.synchronize!!) broadcasts supported values
within each leaf's numerical state. It preserves shared leaf identity, but does not
synchronize rule configuration, frozen flags, or the stored backend.
All ranks must have matching model values, optimizer state, rule configuration, and
frozen flags before updates. Synchronization alone is not a complete checkpoint restore.
"""
struct DistributedOptimizerState{B <: AbstractFluxDistributedBackend, T}
    backend::B
    tree::T
end

@functor DistributedOptimizerState (tree,)

# Guards against composing DistributedOptimizer inside any other rule. Direct
# nesting bypasses the per-leaf `init`/`apply!` guards below, because `setup`
# delegates straight to the inner rule, so it is rejected here before the inner
# setup runs. Nesting inside another rule (e.g. OptimiserChain) reaches `init`,
# which throws the same diagnostic.
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

const _HIGHER_ORDER_ERROR = "DistributedOptimizer supports only first-order gradients; " *
                            "got higher-order gradients (higher...), which cannot be " *
                            "synchronized."

const _DUP_MODEL_ERROR = "DistributedOptimizer does not support Enzyme Duplicated models " *
                         "in update; pass the plain model (Flux.train! does this automatically)."

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

function Optimisers.update!(s::DistributedOptimizerState, model::EnzymeCore.Duplicated,
        grads, higher...)
    throw(ArgumentError(_DUP_MODEL_ERROR))
end

function Optimisers.update(s::DistributedOptimizerState, model::EnzymeCore.Duplicated,
        grads, higher...)
    throw(ArgumentError(_DUP_MODEL_ERROR))
end

# An easy error to make is to pass the result of explicit `gradient(...)`, not
# `gradient(...)[1]`. These mirror the warning in Flux's layers/basic.jl and resolve
# the dispatch ambiguity with the `Chain` methods defined there.
function Optimisers.update!(s::DistributedOptimizerState, model::Chain, grads::Tuple)
    # Zygote will make a NamedTuple{(:layers,)} for the gradient of Chain, Diffractor a Tangent
    @warn """explicit `update!(opt, model, grad)` wants the gradient for the model alone,
    not the whole tuple from `gradient(m -> loss(m, x, y), model)`. You probably want `grads[1]`."""
    return Optimisers.update!(s, model, grads[1])
end

function Optimisers.update(s::DistributedOptimizerState, model::Chain, grads::Tuple)
    @warn """explicit `update(opt, model, grad)` wants the gradient for the model alone,
    not the whole tuple from `gradient(m -> loss(m, x, y), model)`. You probably want `grads[1]`."""
    return Optimisers.update(s, model, grads[1])
end

# Adjusting the rule happens through the public API, on the rule itself or on the
# wrapped state.
Optimisers.adjust(r::DistributedOptimizer, eta::Real) =
    DistributedOptimizer(r.backend, Optimisers.adjust(r.opt, eta))

Optimisers.adjust(r::DistributedOptimizer; kw...) =
    DistributedOptimizer(r.backend, Optimisers.adjust(r.opt; kw...))

Optimisers.adjust(s::DistributedOptimizerState, eta::Real) =
    DistributedOptimizerState(s.backend, Optimisers.adjust(s.tree, eta))

Optimisers.adjust(s::DistributedOptimizerState; kw...) =
    DistributedOptimizerState(s.backend, Optimisers.adjust(s.tree; kw...))

Optimisers.adjust(s::DistributedOptimizerState, r::AbstractRule) =
    DistributedOptimizerState(s.backend, Optimisers.adjust(s.tree, r))

Optimisers.adjust!(s::DistributedOptimizerState, eta::Real) =
    (Optimisers.adjust!(s.tree, eta); s)

Optimisers.adjust!(s::DistributedOptimizerState; kw...) =
    (Optimisers.adjust!(s.tree; kw...); s)

Optimisers.adjust!(s::DistributedOptimizerState, r::AbstractRule) =
    (Optimisers.adjust!(s.tree, r); s)

Optimisers.freeze!(s::DistributedOptimizerState) = (Optimisers.freeze!(s.tree); s)

Optimisers.thaw!(s::DistributedOptimizerState) = (Optimisers.thaw!(s.tree); s)

# --- gradient synchronization ---
#
# Mirrors Optimisers' trainable-parameter semantics: `Optimisers.setup` walks the model
# through `Optimisers.trainable`, stopping at `Optimisers.isnumeric` leaves, and returns
# a state tree whose leaves are exactly the selected parameters. Mutable parameters
# occurring several times in the model share one `Leaf` through an `IdDict` keyed by
# identity, while isbits parameters are *not* cached — one `Leaf` per occurrence
# (Optimisers interface.jl:45-58). `_sync_gradients` pairs the model with that state
# tree and reproduces the same walk so that the averaged gradients can be consumed by
# `Optimisers.update!`/`update` exactly as if they had been produced locally.
#
# The state tree is the source of truth for *which* children are trainable: it records
# the selection made at setup and is stable across updates. Recomputing the selection
# from the model would break legacy no-names tuple selections (`c in tr`): once the
# first update makes two equal-valued fields differ, a field would silently stop being
# synchronized even though its `Leaf` remains in the tree.

# The same pairing Optimisers uses to align a gradient with a model node
# (Optimisers interface.jl:3-4 and 129-135): gradients are never indexed by the model's
# KeyPath, but paired structurally, which also handles Transpose/Adjoint wrappers.
# `_child` is that structural child lookup, shared by gradient nodes and by the
# optimizer state tree, which mirrors the model's children.
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

mutable struct _SyncBuf{B}
    buf::B      # fresh writable buffer (never aliases a caller gradient)
    present::Bool
end

# One explicit occurrence-plan entry per numeric trainable occurrence, in canonical
# model order. `buffer_index` is the synchronization buffer this occurrence contributes
# to; `emit` is true only for the occurrence that emits the synchronized buffer: the
# first occurrence of a mutable parameter, or the single occurrence of an isbits
# parameter. Later occurrences of a tied mutable parameter accumulate locally but emit
# `ZeroTangent`, so the inner Optimisers walk does not add the same global gradient
# twice. The rebuild pass must not infer buffer indices by counting occurrences: a tied
# parameter contributes several occurrences but only one buffer.
struct _SyncOccurrence
    buffer_index::Int
    emit::Bool
end

# Pass 1: collect gradients into one fresh buffer per unique mutable parameter (or per
# occurrence, for isbits parameters), in model order, accumulating the contribution of
# every occurrence — exactly as Optimisers' `_grads!` accumulates per `Leaf`. Gradients
# are copied through a fresh `similar(x)` temporary, so caller gradients are never
# mutated and a gradient object shared by several occurrences of a tied parameter
# contributes once per occurrence (matching Optimisers). Alongside, push one explicit
# occurrence-plan entry per numeric occurrence for the rebuild pass.
#
# Both passes walk the model's functor children in canonical order, paired with the
# corresponding node of the optimizer state tree. A child is trainable iff its tree node
# is a `Leaf`, so the selection is exactly the one `Optimisers.setup` made, in the order
# `Optimisers.setup`/`Optimisers.update` use; occurrence-to-buffer mapping stays aligned
# with the inner optimizer walk.
function _walk!(bufs::Vector, seen::IdDict{Any, Int}, plan::Vector{_SyncOccurrence},
        t, x, x̄)
    if Optimisers.isnumeric(x)
        t isa Optimisers.Leaf || return  # not selected by `Optimisers.setup`
        g = x̄
        if !isbits(x) && haskey(seen, x)
            i = seen[x]
            push!(plan, _SyncOccurrence(i, false))
            if g !== nothing && !(g isa CRC.AbstractZero)
                g isa AbstractArray || throw(ArgumentError(
                    "DistributedOptimizer: gradient for tied parameter $(typeof(x)) " *
                    "must be an AbstractArray, got $(typeof(g))"))
                tmp = similar(x)
                copyto!(tmp, g)
                bufs[i].buf .+= tmp
                bufs[i].present = true
            end
            return
        end
        buf = isbits(x) ? Array(x) : similar(x)
        present = g !== nothing && !(g isa CRC.AbstractZero)
        if present
            copyto!(buf, g)
        else
            fill!(buf, 0)
        end
        push!(bufs, _SyncBuf(buf, present))
        i = length(bufs)
        !isbits(x) && (seen[x] = i)
        push!(plan, _SyncOccurrence(i, true))
        return
    end
    t === () && return  # empty tree node: setup selected no leaf in this subtree
    ch = functor(x)[1]
    xc = _grad_children(x, x̄)
    _foreach_child(ch) do k, c
        tk = _child(t, k)
        tk === nothing && return
        _walk!(bufs, seen, plan, tk, c, _child(xc, k))
    end
    return
end

# Pass 2: rebuild the synchronized gradient structure. Containers mirror the model's
# structural (functor) children, with `nothing` (like Zygote) at non-trainable children
# and at nodes without trainable descendants. Each numeric occurrence consumes the next
# occurrence-plan entry: the allreduced buffer appears only at entries with `emit` (the
# first occurrence of a tied parameter), so Optimisers' per-`Leaf` accumulation does not
# double-count; tied repeats and globally absent parameters emit `ZeroTangent`, which
# the inner update skips. The tree node decides which children are trainable, exactly as
# in pass 1, so the two passes consume the plan in lockstep.
function _build!(t, x, bufs::Vector, present::AbstractVector{Bool},
        plan::Vector{_SyncOccurrence}, cursor::Base.RefValue{Int})
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

# Selection is read from the optimizer state tree (see `_walk!`); the tree methods that
# mirrored `Optimisers._trainable` are gone. No warning is emitted here: native setup
# already emits the legacy `trainable` warning (with `maxlog=3`); `_sync_gradients` must
# not repeat it on every distributed update.

_buildmap(f, ch::NamedTuple) = NamedTuple{keys(ch)}(map(k -> f(k, ch[k]), keys(ch)))
_buildmap(f, ch::Tuple) = Tuple(f(k, ch[k]) for k in 1:length(ch))
# `CartesianIndices` keeps the container's shape and column-major index order:
# mapping over `eachindex` of a multidimensional `Array` would flatten it to a
# vector and the rebuilt gradient would not mirror the model container.
_buildmap(f, ch::AbstractArray) = map(k -> f(k, ch[k]), CartesianIndices(ch))
_buildmap(f, ch::AbstractDict) = Dict(k => f(k, v) for (k, v) in pairs(ch))

# `_walk!` and `_build!` must traverse the model's children in the same canonical
# order; `_foreach_child` mirrors `_buildmap` (and must stay in sync with it).
# Iterating the tree's or the selection's own field order instead would let a reversed
# named `trainable` selection reorder collectives and scramble the occurrence-to-buffer
# mapping, which `Optimisers.update` does not follow.
_foreach_child(f, ch::NamedTuple) = (foreach(k -> f(k, ch[k]), keys(ch)); nothing)
_foreach_child(f, ch::Tuple) = (foreach(k -> f(k, ch[k]), 1:length(ch)); nothing)
_foreach_child(f, ch::AbstractArray) = (foreach(k -> f(k, ch[k]), CartesianIndices(ch)); nothing)
_foreach_child(f, ch::AbstractDict) = (foreach(p -> f(first(p), last(p)), pairs(ch)); nothing)

# Coordinated validity vote for distributed metadata: one all-rank reduction of
# a host Float32 flag. invalid is 1f0, valid is 0f0; `avg` of the flags is
# positive exactly when at least one rank reported invalid. The threshold is
# `> 0`, not `> 0.5`: one invalid rank out of four must reject.
function _any_rank_invalid(backend::AbstractFluxDistributedBackend, invalid::Bool)
    flag = Float32[invalid ? 1f0 : 0f0]
    DistributedUtils.allreduce!(backend, flag, DistributedUtils.avg)
    return flag[1] > 0
end

# Post-reduction validity of one reduced gradient buffer. The generic method
# performs a framework reduction over a host or device buffer. The NCCL
# extension overrides it for `CuArray`, because `NCCL.Allreduce!` returns after
# enqueueing the collective on the task-local CUDA stream.
_reduced_buf_finite(buf::AbstractArray) = all(isfinite, buf)

const _NONFINITE_GRADIENT_ERROR = "DistributedOptimizer rejected the step: at least one rank " *
    "reduced a non-finite synchronized gradient. No optimizer update occurred: parameters and " *
    "optimizer state are unchanged. The local gradient is not necessarily the offending value."

const _NONFINITE_LOSS_ERROR = "DistributedOptimizer rejected the step: at least one rank reported " *
    "a non-finite scalar loss. No optimizer update occurred: parameters and optimizer state are " *
    "unchanged. The local scalar loss is not necessarily the offending value."

# Extension of the `Flux.Train._loss_update_allowed` hook (see `src/train.jl`):
# one all-rank loss-validity vote before the presence exchange and the gradient
# reductions. The flag is host metadata, never derived from a parameter buffer,
# so no device-to-host transfer is involved. Any invalid rank makes every rank
# throw, so no rank enters the optimizer update and the collectives of the
# rejected step cannot desynchronize.
function Train._loss_update_allowed(s::DistributedOptimizerState, scalar_loss)
    _any_rank_invalid(s.backend, !isfinite(scalar_loss)) &&
        throw(DomainError(scalar_loss, _NONFINITE_LOSS_ERROR))
    return true
end

function _sync_gradients(backend::AbstractFluxDistributedBackend, tree, model, grads)
    bufs = _SyncBuf[]
    seen = IdDict{Any, Int}()  # unique mutable parameter => index into `bufs`
    plan = _SyncOccurrence[]   # one entry per numeric occurrence, in model order
    _walk!(bufs, seen, plan, tree, model, grads)
    np = length(bufs)
    np == 0 && return grads  # no trainable parameters: no collectives, grads unchanged

    # Presence protocol: one Float32 `avg` allreduce of 0/1 flags decides — identically
    # on every rank — which parameters have a gradient anywhere. The threshold is
    # `pres[i] > 0`, not `> 0.5` (wrong for more than two ranks). The flags are host
    # metadata, so they are allocated on the CPU, never from a parameter buffer:
    # deriving them with `similar(parameter)` would place them on the parameter device,
    # where the scalar flag writes and the globally present decision would be device
    # operations driven by host control flow (rejected by CUDA scalar-indexing checks)
    # and would force a device synchronization. `avg` works on every backend: the NCCL
    # backend already falls back to MPI for CPU buffers. Parameter gradients stay on
    # their own device and use the selected backend in the allreduces below.
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

    # Post-reduction validity check, after the reductions and before any
    # rebuild or optimizer update. A buffer can be non-finite after averaging
    # even when every local contribution was finite (for example +Inf on one
    # rank and -Inf on another, or a sum that overflows), and a rank can
    # silently reduce a locally non-finite buffer. Every rank inspects the
    # reduced buffers it holds, then one all-rank validity vote makes every
    # rank throw when any rank found an invalid buffer, so no rank enters the
    # optimizer update. The scan runs only when `np > 0` (see the early return
    # above): with no trainable parameters there are no reduced buffers and no
    # collective.
    invalid = false
    for i in 1:np
        present[i] || continue
        invalid |= !_reduced_buf_finite(bufs[i].buf)
    end
    _any_rank_invalid(backend, invalid) &&
        throw(DomainError(nothing, _NONFINITE_GRADIENT_ERROR))

    cursor = Ref(0)
    rebuilt = _build!(tree, model, bufs, present, plan, cursor)
    cursor[] == length(plan) || throw(ArgumentError(
        "DistributedOptimizer internal error: the rebuild pass consumed $(cursor[]) " *
        "of $(length(plan)) trainable occurrences; the occurrence plan must be fully " *
        "consumed. This indicates inconsistent traversal between the collection and " *
        "reconstruction passes."))
    return rebuilt
end

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

end

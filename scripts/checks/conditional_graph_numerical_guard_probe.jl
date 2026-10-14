# ==============================================================================
# DB-1/DB-4 numerical-guard probe (child process, runs under mpiexec)
# ==============================================================================
#
# RED-first success regression for coordinated non-finite-loss rejection on the
# eager distributed training paths (`train!`, `trainstep!`,
# `trainstep_withgradient!`). The expected behavior is defined by
# `reports/conditional-graph-db1-db4-fix-plan.md` sections 2-4. This probe fails
# on the unmodified source and must pass after the DB-1/DB-4 fix.
#
# The probe never writes files: every marker goes to stdout (flushed) and the
# parent driver (`conditional_graph_numerical_guard_check.jl`) owns all shared
# output.
#
# CLI:
#   --case NAME            scenario to run (required)
#   --backend mpi|nccl     backend selection (default mpi)
#   --ranks R              expected MPI world size (default 2)
#   --fault-rank K         rank that injects the non-finite loss (default none)
#   --fault-value nan|inf|ninf   injected non-finite scalar (default nan)
#   --campaign TOKEN       ownership marker for the parent's orphan cleanup;
#                          accepted and ignored by the probe
#
# Collective trace protocol (consumed by the parent driver):
#   COLLECTIVE_BEGIN rank=<r> number=<n> op=<op> eltype=<t> len=<l> buftype=<host|device|other>
#   COLLECTIVE_END rank=<r> number=<n>
#
# Result protocol:
#   VERSIONS julia=... flux=... mpi=... optimisers=...
#   GUARD_PROBE_READY rank=<r>
#   CASE_RESULT rank=<r> case=<name> status=PASS|FAIL|UNTESTED [reason="..."]
#   STEP_STATE_OK rank=<r> step=<i> ok=<bool>
#   STEP_STATE_DETAIL rank=<r> step=<i> param_maxdiff=... state_maxdiff=... counters_ok=<bool>
#   ABSENCE_FROZEN rank=<r> ok=<bool>
#   REACTIVATED rank=<r> ok=<bool>
#
# The `healthy` case trains a conditional model through the eager `train!`
# path. The rank-distinct branch schedule is: step 1 every rank uses U and V;
# step 2 even ranks use U only, odd ranks use V only (U and V are each
# globally present); step 3 every rank uses U only (V is globally absent after
# a nonzero Adam history); step 4 every rank uses U and V again. After every
# step each rank compares parameters and the complete Adam leaf state (m1,
# m2, counters) against a native host reference updated with the global
# averaged gradient, and checks that V's leaf state is exactly frozen at the
# absent step and changed by the reactivation.
#
# Run directly with the selected Flux environment:
#   mpiexec -n 2 julia --project=../flux-integration/test \
#       scripts/checks/conditional_graph_numerical_guard_probe.jl \
#       --case train-one --backend mpi --ranks 2 --fault-rank 1
# ==============================================================================

using Flux
using Optimisers
using Functors

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

function parse_cli(args)
    opts = Dict{String, String}()
    i = 1
    while i <= length(args)
        a = args[i]
        startswith(a, "--") || error("unexpected argument $(repr(a))")
        key = a[3:end]
        key in ("case", "backend", "ranks", "fault-rank", "fault-value", "campaign") ||
            error("unknown option $(repr(a))")
        i += 1
        i <= length(args) || error("missing value for $(repr(a))")
        opts[key] = args[i]
        i += 1
    end
    return opts
end

const OPTS = parse_cli(ARGS)
const CASE = get(OPTS, "case", "")
isempty(CASE) && error("--case is required")
const BACKEND_NAME = get(OPTS, "backend", "mpi")
const RANKS_ARG = parse(Int, get(OPTS, "ranks", "2"))
const FAULT_RANK = parse(Int, get(OPTS, "fault-rank", "-1"))
const FAULT_VALUE_NAME = get(OPTS, "fault-value", "nan")

BACKEND_NAME in ("mpi", "nccl") ||
    error("--backend must be \"mpi\" or \"nccl\"; got $(repr(BACKEND_NAME))")
FAULT_VALUE_NAME in ("nan", "inf", "ninf") ||
    error("--fault-value must be nan, inf or ninf; got $(repr(FAULT_VALUE_NAME))")
FAULT_RANK == -1 || 0 <= FAULT_RANK < RANKS_ARG ||
    error("--fault-rank must satisfy 0 <= fault-rank < --ranks=$(RANKS_ARG); got $(FAULT_RANK)")
RANKS_ARG >= 1 || error("--ranks must be >= 1")

# ------------------------------------------------------------------
# Backend initialization, exactly like test/ext_distributed/distributed_setup.jl.
# ------------------------------------------------------------------

if BACKEND_NAME == "mpi"
    import MPI
    const backend_type = Flux.MPIBackend
else
    # `cuDNN` must be loaded for `MLDataDevices.functional(CUDADevice)`, which
    # `FluxMPIExt` checks during NCCL initialization.
    import MPI, NCCL, CUDA, cuDNN
    CUDA.allowscalar(false)
    const backend_type = Flux.NCCLBackend
end

Flux.DistributedUtils.initialize(backend_type)
const BACKEND = Flux.DistributedUtils.get_distributed_backend(backend_type)
const RANK = Flux.DistributedUtils.local_rank(BACKEND)
const WORLD = Flux.DistributedUtils.total_workers(BACKEND)

WORLD == RANKS_ARG ||
    error("--ranks=$(RANKS_ARG) does not match the real MPI world size $(WORLD)")

# The barrier always uses the MPI communicator: for NCCL the backend's own
# communicator is an NCCL.Communicator and the MPI backend is the fallback.
const COMM = BACKEND_NAME == "nccl" ? BACKEND.mpi_backend.comm : BACKEND.comm

marker(message) = (write(stdout, message * "\n"); flush(stdout))

# ------------------------------------------------------------------
# Model and data
#
# `GuardModel.a` and `GuardModel.b` are the same mutable array (tied, one
# canonical gradient buffer); `c` is independent. The canonical wrapper protocol
# therefore has np = 2 gradient buffers for this model. Data is identical on
# every rank, so the averaged global gradient equals the local gradient.
# ------------------------------------------------------------------

const ARR_T = BACKEND_NAME == "nccl" ? Float32 : Float64
const ETA = 0.05
const A_LEN = 2
const C_LEN = 3

makearr(x) = BACKEND_NAME == "nccl" ? CUDA.cu(ARR_T.(x)) : Vector{ARR_T}(x)
todev(x) = BACKEND_NAME == "nccl" ? CUDA.cu(ARR_T.(x)) : Vector{ARR_T}(x)

struct GuardModel{T}
    a::T
    b::T
    c::T
end
Functors.@functor GuardModel

struct EmptyModel end
Functors.@functor EmptyModel

const C0 = makearr([1.0, 1.0, 1.0])

function fresh_model()
    a = makearr([1.0, 2.0])
    return GuardModel(a, a, makearr([-1.0, 0.0, 2.0]))
end

step_data(i) = [0.1 * i, -0.05 * i]
data_range(range) = [(i, step_data(i)) for i in range]

loss_guard(m, x) = (xd = todev(x);
    sum(abs2, m.a .- xd) + sum(abs2, m.b .- (xd .+ 1)) + sum(abs2, m.c .- C0))

scalar_fault() = FAULT_VALUE_NAME == "nan" ? ARR_T(NaN) :
                 FAULT_VALUE_NAME == "inf" ? ARR_T(Inf) : ARR_T(-Inf)

# ------------------------------------------------------------------
# Conditional model for the `healthy` case. `w` (len 2) is always used, `u`
# (len 3, branch U) and `v` (len 4, branch V) are rank-scheduled. Distinct
# lengths let the parent classify each gradient reduction unambiguously.
# ------------------------------------------------------------------

struct CondModel{T}
    w::T
    u::T
    v::T
end
Functors.@functor CondModel

function fresh_cond_model()
    return CondModel(makearr([1.0, 2.0]), makearr([-1.0, 0.5, 1.5]),
        makearr([0.5, -0.5, 1.0, 2.0]))
end

fresh_cond_ref() = CondModel(Vector{ARR_T}([1.0, 2.0]), Vector{ARR_T}([-1.0, 0.5, 1.5]),
    Vector{ARR_T}([0.5, -0.5, 1.0, 2.0]))

w_target(x) = x
u_target(x) = [x[1], x[2], x[1] + x[2]]
v_target(x) = [x[1], x[2], x[1] - x[2], 0.5 * (x[1] + x[2])]

# Host target converted to the parameter's device: the measured model is
# device under NCCL, the native reference stays on the host.
devlike(p, t) = BACKEND_NAME == "nccl" && p isa CUDA.CuArray ? CUDA.cu(ARR_T.(t)) :
                Vector{ARR_T}(t)

wl(m, x) = (xt = devlike(m.w, w_target(x)); sum(abs2, m.w .- xt))
ul(m, x) = (xt = devlike(m.u, u_target(x)); sum(abs2, m.u .- xt))
vl(m, x) = (xt = devlike(m.v, v_target(x)); sum(abs2, m.v .- xt))

function step_branches(r, i)
    i == 1 && return (; u=true, v=true)
    i == 2 && return iseven(r) ? (; u=true, v=false) : (; u=false, v=true)
    i == 3 && return (; u=true, v=false)
    return (; u=true, v=true)
end

# The measured loss uses this rank's branches; the native reference sums the
# local loss of every rank and averages, matching the wrapper's zero-filled
# `avg` reduction of globally present buffers.
function cond_local_loss(m, r, i, x)
    br = step_branches(r, i)
    total = wl(m, x)
    br.u && (total += ul(m, x))
    br.v && (total += vl(m, x))
    return total
end

cond_loss(m, i, x) = cond_local_loss(m, RANK, i, x)

# ------------------------------------------------------------------
# Snapshot helper: parameters, complete Adam leaf state (m, v, beta
# counters), and tied identity. Host copies only; device buffers are
# never read element-by-element.
# ------------------------------------------------------------------

function leaf_state_snapshot(leaf)
    return (copy(Array(leaf.state[1])), copy(Array(leaf.state[2])), collect(leaf.state[3]))
end

# Accepts either the wrapper state or a bare Optimisers tree (native shadow).
tree_of(state) = state.tree
tree_of(state::NamedTuple) = state

function snapshot(model, state)
    tree = tree_of(state)
    return (;
        a=copy(Array(model.a)),
        b=copy(Array(model.b)),
        c=copy(Array(model.c)),
        sa=leaf_state_snapshot(tree.a),
        sc=leaf_state_snapshot(tree.c),
        tied_model=model.a === model.b,
        tied_tree=tree.a === tree.b,
    )
end

function leaf_snap_equal(x, y)
    length(x) == length(y) || return false
    return all(i -> isequal(x[i], y[i]), 1:length(x))
end

function snap_equal(s1, s2)
    return s1.tied_model == s2.tied_model &&
           s1.tied_tree == s2.tied_tree &&
           isequal(s1.a, s2.a) && isequal(s1.b, s2.b) && isequal(s1.c, s2.c) &&
           leaf_snap_equal(s1.sa, s2.sa) &&
           leaf_snap_equal(s1.sc, s2.sc)
end

moments_nonzero(snap) = any(!iszero, snap.sa[1]) && any(!iszero, snap.sc[1])

# Host-side diff for the healthy per-step reference comparison. Device
# buffers are converted once; nothing is read element-by-element on device.
function maxdiff(x, y)
    size(x) == size(y) || return Inf
    return maximum(abs, Array(x) .- Array(y))
end

function leaf_state_maxdiff(s1, s2)
    return max(maxdiff(s1[1], s2[1]), maxdiff(s1[2], s2[2]))
end

function cond_state_diff(model, state, ref, ref_state)
    tree = tree_of(state)
    param_d = max(maxdiff(model.w, ref.w), maxdiff(model.u, ref.u), maxdiff(model.v, ref.v))
    state_d = max(leaf_state_maxdiff(leaf_state_snapshot(tree.w),
                                     leaf_state_snapshot(ref_state.w)),
                  leaf_state_maxdiff(leaf_state_snapshot(tree.u),
                                     leaf_state_snapshot(ref_state.u)),
                  leaf_state_maxdiff(leaf_state_snapshot(tree.v),
                                     leaf_state_snapshot(ref_state.v)))
    counters_ok = isequal(leaf_state_snapshot(tree.w)[3], leaf_state_snapshot(ref_state.w)[3]) &&
                  isequal(leaf_state_snapshot(tree.u)[3], leaf_state_snapshot(ref_state.u)[3]) &&
                  isequal(leaf_state_snapshot(tree.v)[3], leaf_state_snapshot(ref_state.v)[3])
    return (; param_d, state_d, counters_ok)
end

# ------------------------------------------------------------------
# Tracing backend
# ------------------------------------------------------------------

buftype(::Array) = "host"
buftype(::AbstractArray) = "other"
if BACKEND_NAME == "nccl"
    buftype(::CUDA.CuArray) = "device"
end

struct TraceBackend{B} <: Flux.AbstractFluxDistributedBackend
    inner::B
    rank::Int
    counter::Base.RefValue{Int}
    quiet::Bool
end

function Flux.DistributedUtils.allreduce!(b::TraceBackend, buf, op)
    b.counter[] += 1
    n = b.counter[]
    if !b.quiet
        marker("COLLECTIVE_BEGIN rank=$(b.rank) number=$n op=$(op) eltype=$(eltype(buf)) " *
               "len=$(length(buf)) buftype=$(buftype(buf))")
    end
    result = Flux.DistributedUtils.allreduce!(b.inner, buf, op)
    b.quiet || marker("COLLECTIVE_END rank=$(b.rank) number=$n")
    return result
end

quiet_backend() = TraceBackend(BACKEND, RANK, Ref(0), true)
trace_backend() = TraceBackend(BACKEND, RANK, Ref(0), false)

function make_state(model; backend=trace_backend())
    return Flux.setup(
        Flux.DistributedUtils.DistributedOptimizer(backend, Optimisers.Adam(ETA)), model)
end

# ------------------------------------------------------------------
# Case context and loss closures
#
# The loss functions are top-level so every case in this process shares one
# compiled specialization; `warmup` exercises the same closure type and the
# quiet trace backend, so the measured phase only pays measured work.
# ------------------------------------------------------------------

mutable struct CaseCtx{S, F, A}
    state::S
    fault_rank::Int
    fault_value::F
    aux_value::A
    snap::Any
    evals::Int
    last_item::Int
    loss_invalid::Bool
end

CaseCtx(state; fault_rank=-1, fault_value=0.0, aux_value=0.0) =
    CaseCtx(state, fault_rank, fault_value, aux_value, nothing, 0, 0, false)

function snapshot_here(ctx, m, i)
    i == 3 && ctx.snap === nothing && (ctx.snap = snapshot(m, ctx.state))
    return nothing
end

function case_loss(ctx, m, i, x)
    ctx.evals += 1
    ctx.last_item = i
    if i == 3
        snapshot_here(ctx, m, i)
        if RANK == ctx.fault_rank
            ctx.loss_invalid = true
            marker("LOSS_INVALID rank=$RANK item=$i value=$(ctx.fault_value)")
            return ctx.fault_value
        end
    end
    return loss_guard(m, x)
end

function aux_case_loss(ctx, m, i, x)
    ctx.evals += 1
    ctx.last_item = i
    if i == 3
        snapshot_here(ctx, m, i)
        if RANK == ctx.fault_rank
            ctx.loss_invalid = true
            marker("LOSS_INVALID rank=$RANK item=$i value=$(ctx.fault_value)")
            return (ctx.fault_value, (; acc=ctx.aux_value))
        end
    end
    return (loss_guard(m, x), (; acc=ctx.aux_value))
end

function empty_case_loss(ctx, m, i, x)
    ctx.evals += 1
    ctx.last_item = i
    if i == 3 && RANK == ctx.fault_rank
        ctx.loss_invalid = true
        marker("LOSS_INVALID rank=$RANK item=$i value=$(ctx.fault_value)")
        return ctx.fault_value
    end
    return sum(abs2, todev(x))
end

plain_loss(ctx) = (m, i, x) -> case_loss(ctx, m, i, x)
aux_loss(ctx) = (m, i, x) -> aux_case_loss(ctx, m, i, x)
empty_loss(ctx) = (m, i, x) -> empty_case_loss(ctx, m, i, x)

# Plain loss adapter (no fault injection) shared by the healthy case and the
# Duplicated availability probe.
plain_train_loss(m, i, x) = loss_guard(m, x)

# ------------------------------------------------------------------
# Result helpers
# ------------------------------------------------------------------

function case_result(status; reason="")
    extra = isempty(reason) ? "" : " reason=\"$reason\""
    marker("CASE_RESULT rank=$RANK case=$CASE status=$status$extra")
    return nothing
end

function finish_case(status; reason="")
    case_result(status; reason)
    MPI.Barrier(COMM)
    MPI.Finalize()
    exit(status == "FAIL" ? 1 : 0)
end

# ------------------------------------------------------------------
# Shared measured-phase pieces
# ------------------------------------------------------------------

function run_train_rejection(; fault_rank, fault_value, aux_nan_first=false)
    model = fresh_model()
    state = make_state(model)
    ctx = CaseCtx(state; fault_rank, fault_value,
        aux_value=aux_nan_first ? ARR_T(1) : ARR_T(NaN))
    loss = aux_nan_first ? aux_loss(ctx) : plain_loss(ctx)

    caught = false
    try
        Flux.train!(loss, model, data_range(1:5), state)
        marker("TRAIN_RETURNED rank=$RANK")
    catch e
        e isa DomainError || rethrow()
        caught = true
    end

    marker("REJECTED rank=$RANK caught=$(caught ? "DomainError" : "none")")
    completed = ctx.evals - ((caught || ctx.loss_invalid) ? 1 : 0)
    marker("STEPS_EXECUTED rank=$RANK completed=$completed")
    snap_ok = ctx.snap !== nothing && snap_equal(ctx.snap, snapshot(model, state))
    moments_ok = ctx.snap !== nothing && moments_nonzero(ctx.snap)
    tied_ok = model.a === model.b && state.tree.a === state.tree.b
    marker("SNAPSHOT_OK rank=$RANK ok=$snap_ok moments_nonzero=$moments_ok tied=$tied_ok")

    ok = caught && snap_ok && moments_ok && tied_ok && completed == 2 && ctx.last_item == 3
    reason = ok ? "" : "caught=$caught snapshot=$snap_ok moments=$moments_ok tied=$tied_ok completed=$completed"
    return finish_case(ok ? "PASS" : "FAIL"; reason)
end

function run_train_all()
    model = fresh_model()
    state = make_state(model)
    # Every rank injects the same non-finite loss at item 3.
    ctx = CaseCtx(state; fault_rank=RANK, fault_value=ARR_T(NaN))
    loss = plain_loss(ctx)

    caught = false
    try
        Flux.train!(loss, model, data_range(1:5), state)
        marker("TRAIN_RETURNED rank=$RANK")
    catch e
        e isa DomainError || rethrow()
        caught = true
    end
    marker("REJECTED rank=$RANK caught=$(caught ? "DomainError" : "none")")
    completed = ctx.evals - ((caught || ctx.loss_invalid) ? 1 : 0)
    marker("STEPS_EXECUTED rank=$RANK completed=$completed")
    snap_ok = ctx.snap !== nothing && snap_equal(ctx.snap, snapshot(model, state))
    marker("SNAPSHOT_OK rank=$RANK ok=$snap_ok")

    shadow = fresh_model()
    sstate = Optimisers.setup(Optimisers.Adam(ETA), shadow)
    for i in 1:2
        g = Flux.gradient(mm -> loss_guard(mm, step_data(i)), shadow)[1]
        Optimisers.update!(sstate, shadow, g)
    end
    shadow_ok = isequal(Array(model.a), Array(shadow.a)) &&
                isequal(Array(model.c), Array(shadow.c)) &&
                leaf_snap_equal(snapshot(model, state).sa, snapshot(shadow, sstate).sa) &&
                leaf_snap_equal(snapshot(model, state).sc, snapshot(shadow, sstate).sc)
    marker("SHADOW_MATCH rank=$RANK ok=$shadow_ok")

    ok = caught && completed == 2 && snap_ok && shadow_ok
    reason = ok ? "" : "caught=$caught completed=$completed snapshot=$snap_ok shadow=$shadow_ok"
    return finish_case(ok ? "PASS" : "FAIL"; reason)
end

function run_loop(; withgradient::Bool, fault_rank, fault_value)
    model = fresh_model()
    state = make_state(model)
    ctx = CaseCtx(state; fault_rank, fault_value)
    loss = plain_loss(ctx)

    caught = false
    returned_at_fault = Ref{Any}(nothing)
    for i in 1:5
        try
            if withgradient
                r = Flux.trainstep_withgradient!(loss, model, (i, step_data(i)), state)
                i == 3 && (returned_at_fault[] = r)
            else
                r = Flux.trainstep!(loss, model, (i, step_data(i)), state)
                i == 3 && (returned_at_fault[] = r)
            end
            marker("STEP rank=$RANK i=$i")
        catch e
            e isa DomainError || rethrow()
            caught = true
            marker("REJECTED rank=$RANK caught=DomainError item=$i")
            break
        end
        if i == 3 && returned_at_fault[] !== nothing
            marker("RETURNED_AT_FAULT rank=$RANK")
            break
        end
    end

    completed = ctx.evals - ((caught || ctx.loss_invalid) ? 1 : 0)
    marker("STEPS_EXECUTED rank=$RANK completed=$completed")
    snap_ok = ctx.snap !== nothing && snap_equal(ctx.snap, snapshot(model, state))
    moments_ok = ctx.snap !== nothing && moments_nonzero(ctx.snap)
    marker("SNAPSHOT_OK rank=$RANK ok=$snap_ok moments_nonzero=$moments_ok")
    withgradient && marker("RETURNED=$(returned_at_fault[] !== nothing)")

    ok = caught && snap_ok && moments_ok && completed == 2 &&
         returned_at_fault[] === nothing
    reason = ok ? "" : "caught=$caught snapshot=$snap_ok moments=$moments_ok " *
                       "completed=$completed returned=$(returned_at_fault[] !== nothing)"
    return finish_case(ok ? "PASS" : "FAIL"; reason)
end

function run_aux_nonfinite_ok()
    model = fresh_model()
    state = make_state(model)
    ctx = CaseCtx(state; fault_rank=-1, fault_value=ARR_T(NaN), aux_value=ARR_T(NaN))
    loss = aux_loss(ctx)

    ok = true
    for i in 1:3
        v = Flux.trainstep!(loss, model, (i, step_data(i)), state)
        good = v isa Tuple && length(v) == 2 && v[2] isa NamedTuple &&
               haskey(v[2], :acc) && v[2].acc != v[2].acc && isfinite(v[1])
        ok &= good
        marker("AUX_STEP rank=$RANK i=$i ok=$good finite=$(isfinite(v[1])) " *
               "aux_nan=$(v[2].acc != v[2].acc)")
    end
    marker("AUX_OK rank=$RANK ok=$ok")
    return finish_case(ok ? "PASS" : "FAIL";
        reason=ok ? "" : "aux return structure or finite-scalar contract broken")
end

function run_no_trainable(; fault_rank, fault_value)
    model = EmptyModel()
    state = make_state(model)
    empty_state = isempty(state.tree)
    marker("STATE_EMPTY=$(empty_state) rank=$RANK")
    ctx = CaseCtx(state; fault_rank, fault_value)
    loss = empty_loss(ctx)

    caught = false
    try
        Flux.train!(loss, model, data_range(1:5), state)
        marker("TRAIN_RETURNED rank=$RANK")
    catch e
        e isa DomainError || rethrow()
        caught = true
    end
    marker("REJECTED rank=$RANK caught=$(caught ? "DomainError" : "none")")
    completed = ctx.evals - ((caught || ctx.loss_invalid) ? 1 : 0)
    marker("STEPS_EXECUTED rank=$RANK completed=$completed")

    ok = caught && empty_state && completed == 2 && ctx.last_item == 3
    reason = ok ? "" : "caught=$caught empty=$empty_state completed=$completed"
    return finish_case(ok ? "PASS" : "FAIL"; reason)
end

function run_recover_one(; fault_rank, fault_value)
    model = fresh_model()
    state = make_state(model)
    ctx = CaseCtx(state; fault_rank, fault_value)
    loss = plain_loss(ctx)

    caught = false
    for i in 1:5
        try
            Flux.trainstep!(loss, model, (i, step_data(i)), state)
            marker("STEP rank=$RANK i=$i")
        catch e
            e isa DomainError || rethrow()
            caught = true
            marker("REJECTED rank=$RANK caught=DomainError item=$i")
            break
        end
    end
    completed = ctx.evals - ((caught || ctx.loss_invalid) ? 1 : 0)
    marker("STEPS_EXECUTED rank=$RANK completed=$completed")
    snap_ok = ctx.snap !== nothing && snap_equal(ctx.snap, snapshot(model, state))
    marker("SNAPSHOT_OK rank=$RANK ok=$snap_ok")

    # One known-good step on item 4 after the rejection.
    recovery_val = Flux.trainstep!(loss, model, (4, step_data(4)), state)
    marker("RECOVERY_STEP rank=$RANK item=4 loss_finite=$(isfinite(recovery_val))")

    shadow = fresh_model()
    sstate = Optimisers.setup(Optimisers.Adam(ETA), shadow)
    for i in (1, 2, 4)
        g = Flux.gradient(mm -> loss_guard(mm, step_data(i)), shadow)[1]
        Optimisers.update!(sstate, shadow, g)
    end
    recovery_ok = isequal(Array(model.a), Array(shadow.a)) &&
                  isequal(Array(model.c), Array(shadow.c)) &&
                  leaf_snap_equal(snapshot(model, state).sa, snapshot(shadow, sstate).sa) &&
                  leaf_snap_equal(snapshot(model, state).sc, snapshot(shadow, sstate).sc)
    marker("RECOVERY_MATCH rank=$RANK ok=$recovery_ok")

    ok = caught && snap_ok && recovery_ok && completed == 2
    reason = ok ? "" : "caught=$caught snapshot=$snap_ok recovery=$recovery_ok completed=$completed"
    return finish_case(ok ? "PASS" : "FAIL"; reason)
end

# Finite conditional training through the eager `train!` path. The
# rank-distinct branch schedule creates a full step, a partially present step,
# a globally absent step with nonzero Adam history for `v`, and a
# reactivation. After every step the measured parameters and complete Adam
# leaf state are compared against a native host reference that is updated with
# the global gradient the wrapper's zero-filled `avg` reduction produces.
function run_healthy()
    model = fresh_cond_model()
    state = make_state(model)
    ref = fresh_cond_ref()
    ref_state = Optimisers.setup(Optimisers.Adam(ETA), ref)
    tol = BACKEND_NAME == "nccl" ? 1e-6 : 1e-8

    all_ok = true
    v_after2 = nothing
    v_after3 = nothing
    for i in 1:4
        Flux.train!(cond_loss, model, [(i, step_data(i))], state)

        g = Flux.gradient(mm -> ARR_T(1 / RANKS_ARG) * sum(
            cond_local_loss(mm, r, i, step_data(i)) for r in 0:(RANKS_ARG - 1)), ref)[1]
        Optimisers.update!(ref_state, ref, g)

        cmp = cond_state_diff(model, state, ref, ref_state)
        step_ok = cmp.param_d <= tol && cmp.state_d <= tol && cmp.counters_ok
        all_ok &= step_ok
        marker("STEP_STATE_OK rank=$RANK step=$i ok=$step_ok")
        marker("STEP_STATE_DETAIL rank=$RANK step=$i param_maxdiff=$(cmp.param_d) " *
               "state_maxdiff=$(cmp.state_d) counters_ok=$(cmp.counters_ok)")

        if i == 2
            v_after2 = leaf_state_snapshot(tree_of(state).v)
        elseif i == 3
            frozen = leaf_snap_equal(leaf_state_snapshot(tree_of(state).v), v_after2)
            marker("ABSENCE_FROZEN rank=$RANK ok=$frozen")
            all_ok &= frozen
            v_after3 = leaf_state_snapshot(tree_of(state).v)
        elseif i == 4
            reactivated = !leaf_snap_equal(leaf_state_snapshot(tree_of(state).v), v_after3)
            marker("REACTIVATED rank=$RANK ok=$reactivated")
            all_ok &= reactivated
        end
    end

    return finish_case(all_ok ? "PASS" : "FAIL";
        reason=all_ok ? "" : "conditional trajectory diverged from the native reference")
end

# ------------------------------------------------------------------
# Duplicated (`Enzyme.Duplicated`) optional route
# ------------------------------------------------------------------

# The Duplicated/Enzyme eager route is exercised under MPI only. Under
# CUDA/NCCL the availability check (`Enzyme.Duplicated` on `CuArray`
# parameters) did not complete within the startup watchdog in the first GREEN
# attempt, so the route is reported UNTESTED there, never silently skipped.
const ENZYME_LOADED = CASE == "dup-train-one" && BACKEND_NAME == "mpi" && try
    @eval import Enzyme
    true
catch
    false
end

# Filled by the warmup phase (before GUARD_PROBE_READY): the Duplicated route
# availability and its compilation must not consume the execution watchdog.
const DUP_STATUS = Ref{Any}(nothing)

function dup_route_check()
    if BACKEND_NAME == "nccl"
        return (false, "Duplicated/Enzyme eager route is not exercised under CUDA/NCCL: " *
                       "the availability check exceeded the startup watchdog (420 s) in the " *
                       "green-deathstar-loss-nccl2 attempt; MPI exercises this route")
    end
    if !ENZYME_LOADED
        return (false, "Enzyme.jl is not loadable in this environment")
    end
    try
        m = fresh_model()
        dup = Enzyme.Duplicated(m, Enzyme.make_zero(m))
        st = make_state(m; backend=BACKEND)
        Flux.trainstep!(plain_train_loss, dup, (1, step_data(1)), st)
        return (true, "")
    catch e
        return (false, sprint(showerror, e))
    end
end

function run_dup_train_one(; fault_rank, fault_value)
    avail, err = DUP_STATUS[] === nothing ? (false, "availability not evaluated") : DUP_STATUS[]
    # The agreement vote uses the MPI communicator; under NCCL `BACKEND.comm`
    # is an NCCL communicator and is not valid for MPI collectives.
    all_avail = MPI.Allreduce(avail ? 1 : 0, +, COMM) == WORLD
    if !all_avail
        marker("UNTESTED rank=$RANK reason=\"Duplicated route unavailable: $(replace(err, '"' => '\''))\"")
        return finish_case("UNTESTED"; reason="Duplicated route unavailable: $err")
    end

    model = fresh_model()
    dup = Enzyme.Duplicated(model, Enzyme.make_zero(model))
    state = make_state(dup)
    ctx = CaseCtx(state; fault_rank, fault_value)
    loss = plain_loss(ctx)

    caught = false
    try
        Flux.train!(loss, dup, data_range(1:5), state)
        marker("TRAIN_RETURNED rank=$RANK")
    catch e
        e isa DomainError || rethrow()
        caught = true
    end
    marker("REJECTED rank=$RANK caught=$(caught ? "DomainError" : "none")")
    completed = ctx.evals - ((caught || ctx.loss_invalid) ? 1 : 0)
    marker("STEPS_EXECUTED rank=$RANK completed=$completed")
    snap_ok = ctx.snap !== nothing && snap_equal(ctx.snap, snapshot(model, state))
    moments_ok = ctx.snap !== nothing && moments_nonzero(ctx.snap)
    marker("SNAPSHOT_OK rank=$RANK ok=$snap_ok moments_nonzero=$moments_ok")

    ok = caught && snap_ok && moments_ok && completed == 2 && ctx.last_item == 3
    reason = ok ? "" : "caught=$caught snapshot=$snap_ok moments=$moments_ok completed=$completed"
    return finish_case(ok ? "PASS" : "FAIL"; reason)
end

# ------------------------------------------------------------------
# Warmup: a throwaway wrapped step with the quiet trace backend and the case's
# own loss closure type, before READY. The execution watchdog therefore only
# covers measured work.
# ------------------------------------------------------------------

function warmup()
    warm = data_range(1:2)
    push!(warm, (3, step_data(3)))
    if CASE == "no-trainable"
        model = EmptyModel()
        ctx = CaseCtx(make_state(model; backend=quiet_backend()); fault_rank=-99,
            fault_value=ARR_T(NaN))
        Flux.train!(empty_loss(ctx), model, warm, ctx.state)
    elseif CASE == "aux-nonfinite-ok" || CASE == "aux-nan-first"
        model = fresh_model()
        ctx = CaseCtx(make_state(model; backend=quiet_backend()); fault_rank=-99,
            fault_value=ARR_T(NaN), aux_value=ARR_T(NaN))
        Flux.train!(aux_loss(ctx), model, warm, ctx.state)
        snap_equal(snapshot(model, ctx.state), snapshot(model, ctx.state))
    elseif CASE == "healthy"
        model = fresh_cond_model()
        state = make_state(model; backend=quiet_backend())
        Flux.train!(cond_loss, model, [(i, step_data(i)) for i in 1:4], state)
        ref = fresh_cond_ref()
        ref_state = Optimisers.setup(Optimisers.Adam(ETA), ref)
        for i in 1:4
            g = Flux.gradient(mm -> ARR_T(1 / RANKS_ARG) * sum(
                cond_local_loss(mm, r, i, step_data(i)) for r in 0:(RANKS_ARG - 1)), ref)[1]
            Optimisers.update!(ref_state, ref, g)
        end
        cond_state_diff(model, state, ref, ref_state)
        leaf_state_snapshot(tree_of(state).v)
    elseif CASE == "dup-train-one"
        model = fresh_model()
        ctx = CaseCtx(make_state(model; backend=quiet_backend()); fault_rank=-99,
            fault_value=ARR_T(NaN))
        Flux.train!(plain_loss(ctx), model, warm, ctx.state)
        snap_equal(snapshot(model, ctx.state), snapshot(model, ctx.state))
        DUP_STATUS[] = dup_route_check()
    else
        model = fresh_model()
        ctx = CaseCtx(make_state(model; backend=quiet_backend()); fault_rank=-99,
            fault_value=ARR_T(NaN))
        Flux.train!(plain_loss(ctx), model, warm, ctx.state)
        snap_equal(snapshot(model, ctx.state), snapshot(model, ctx.state))
    end
    return nothing
end

# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------

function run_case()
    if CASE == "train-one" || CASE == "train-one-inf" || CASE == "train-one-ninf"
        return run_train_rejection(; fault_rank=FAULT_RANK, fault_value=scalar_fault())
    elseif CASE == "train-all"
        return run_train_all()
    elseif CASE == "loop-one"
        return run_loop(; withgradient=false, fault_rank=FAULT_RANK, fault_value=scalar_fault())
    elseif CASE == "loop-withgradient-one"
        return run_loop(; withgradient=true, fault_rank=FAULT_RANK, fault_value=scalar_fault())
    elseif CASE == "aux-nonfinite-ok"
        return run_aux_nonfinite_ok()
    elseif CASE == "aux-nan-first"
        return run_train_rejection(; fault_rank=FAULT_RANK, fault_value=scalar_fault(),
            aux_nan_first=true)
    elseif CASE == "no-trainable"
        return run_no_trainable(; fault_rank=FAULT_RANK, fault_value=scalar_fault())
    elseif CASE == "recover-one"
        return run_recover_one(; fault_rank=FAULT_RANK, fault_value=scalar_fault())
    elseif CASE == "healthy"
        return run_healthy()
    elseif CASE == "dup-train-one"
        return run_dup_train_one(; fault_rank=FAULT_RANK, fault_value=scalar_fault())
    else
        error("unknown --case $(repr(CASE))")
    end
end

# ------------------------------------------------------------------

try
    warmup()
    marker("VERSIONS julia=$(VERSION) flux=$(pkgversion(Flux)) mpi=$(pkgversion(MPI)) " *
           "optimisers=$(pkgversion(Optimisers))")
    marker("GUARD_PROBE_READY rank=$RANK")
    run_case()
catch e
    msg = replace(sprint(showerror, e), '"' => '\'', '\n' => ' ')
    marker("CASE_RESULT rank=$RANK case=$CASE status=FAIL error=\"$msg\"")
    Base.showerror(stderr, e, catch_backtrace())
    flush(stderr)
    MPI.Abort(MPI.COMM_WORLD, 1)
end

# ==============================================================================
# Conditional-graph deferred-behavior probe (child process, runs under mpiexec)
# ==============================================================================
#
# Characterizes remaining behavioral boundaries of `DistributedOptimizerState`.
# Every mode runs exactly 2 MPI ranks on CPU. Modes:
#
#   control         all finite, `train!` + wrapper vs a native Adam shadow
#   nan-loss-hang   `train!`, rank 1 loss NaN at step 3, rank 0 finite
#   nan-loss-all    `train!`, loss NaN at step 3 on every rank
#   grad-nonfinite  finite loss, NaN gradient on rank 1, explicit update
#   step-desync     manual `trainstep!` loop, rank 1 loss NaN at step 3
#   early-stop      manual `trainstep!` loop, rank 1 stops after step 3
#   config-eta      rank-local `adjust!`, no hang, silent replica divergence
#   config-frozen   rank-local `freeze!`, no hang, silent replica divergence
#
# Markers are machine-readable and consumed by
# `scripts/checks/conditional_graph_deferred_check.jl`. Marker format:
#
#   VERSIONS julia=... flux=... mpi=... optimisers=...
#   PROBE_READY rank=<r>
#   COLLECTIVE_BEGIN rank=<r> number=<n> kind=<presence|gradient> eltype=<t> len=<l>
#   COLLECTIVE_END rank=<r> number=<n> kind=<k> value=<v>
#   STEP / STEP_DONE / LOSS_NAN / NONFINITE_STOP / HOLDING / CONTROL_MATCH
#   GRAD / CORRUPTION / DESYNC_RESULT / LOOP_STOP / LOCAL_ADJUST / LOCAL_FREEZE
#   CONFIG_ETA / CONFIG_FROZEN / STOP_STATE_MATCH / MODE_RESULT
#
# Run directly with the selected Flux environment:
#   mpiexec -n 2 julia --project=../flux-integration/test \
#       scripts/checks/conditional_graph_deferred_probe.jl control
# ==============================================================================

using MPI
using Flux
using Optimisers
using Functors

const MODE = isempty(ARGS) ? "control" : ARGS[1]

Flux.DistributedUtils.initialize(Flux.MPIBackend)
const BACKEND = Flux.DistributedUtils.get_distributed_backend(Flux.MPIBackend)
const RANK = Flux.DistributedUtils.local_rank(BACKEND)
const WORLD = Flux.DistributedUtils.total_workers(BACKEND)
WORLD == 2 || error("probe needs exactly 2 ranks; got $WORLD")

marker(message) = (write(stdout, message * "\n"); flush(stdout))

# ------------------------------------------------------------------
# Model: two independent parameters so gradients and Adam moments can
# be inspected separately. Same data on both ranks unless a mode says
# otherwise, so the global-mean gradient equals the local gradient.
# ------------------------------------------------------------------

struct Tiny
    w::Vector{Float64}
    v::Vector{Float64}
end
Functors.@functor Tiny

const ETA = 0.05
const W0 = [1.0, 2.0]
const V0 = [0.5, -0.5]

fresh_model() = Tiny(copy(W0), copy(V0))
step_data(i) = [0.1 * i, -0.05 * i]
loss_quad(m, x) = sum(abs2, m.w .- x) + sum(abs2, m.v .- V0)

make_state(model; backend = BACKEND) = Optimisers.setup(
    Flux.DistributedUtils.DistributedOptimizer(backend, Optimisers.Adam(ETA)), model)

shadow_grad(m, x) = Flux.gradient(mm -> loss_quad(mm, x), m)[1]
combine(g0, g1) = (; w = (g0.w .+ g1.w) ./ 2, v = (g0.v .+ g1.v) ./ 2)
maxdiff(a, b) = max(maximum(abs, a .- b), maximum(abs, b .- a))

# ------------------------------------------------------------------
# Backend wrapper that traces every collective before it is entered.
# An unmatched `COLLECTIVE_BEGIN` is the evidence that a rank is
# blocked inside that collective.
# ------------------------------------------------------------------

struct LoggingBackend{B} <: Flux.AbstractFluxDistributedBackend
    inner::B
    rank::Int
    counter::Base.RefValue{Int}
end

function Flux.DistributedUtils.allreduce!(b::LoggingBackend, buf, op)
    b.counter[] += 1
    kind = eltype(buf) === Float32 ? "presence" : "gradient"
    marker("COLLECTIVE_BEGIN rank=$(b.rank) number=$(b.counter[]) kind=$(kind) " *
           "eltype=$(eltype(buf)) len=$(length(buf))")
    result = Flux.DistributedUtils.allreduce!(b.inner, buf, op)
    marker("COLLECTIVE_END rank=$(b.rank) number=$(b.counter[]) kind=$(kind) value=$(first(buf))")
    return result
end

function mode_result(mode, ok; extra = "")
    marker("MODE_RESULT mode=$mode rank=$RANK ok=$ok $extra")
    MPI.Barrier(BACKEND.comm)
    MPI.Finalize()
    exit(ok ? 0 : 1)
end

# ------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------

function run_control()
    model = fresh_model()
    logb = LoggingBackend(BACKEND, RANK, Ref(0))
    state = make_state(model; backend = logb)
    shadow = fresh_model()
    sstate = Optimisers.setup(Optimisers.Adam(ETA), shadow)
    for i in 1:4
        x = step_data(i)
        l = Flux.trainstep!(loss_quad, model, x, state)
        sstate, shadow = Optimisers.update(sstate, shadow, shadow_grad(shadow, x))
        marker("STEP rank=$RANK i=$i loss=$l")
    end
    d = max(maxdiff(model.w, shadow.w), maxdiff(model.v, shadow.v))
    marker("CONTROL_MATCH rank=$RANK ok=$(d <= 1e-13) maxdiff=$d")
    mode_result("control", d <= 1e-13; extra = "maxdiff=$d")
end

function run_nan_loss_hang()
    model = fresh_model()
    logb = LoggingBackend(BACKEND, RANK, Ref(0))
    state = make_state(model; backend = logb)
    item = Ref(0)
    function loss_nan(m, i)
        item[] = i
        l = loss_quad(m, step_data(i))
        RANK == 1 && i == 3 || return l
        marker("LOSS_NAN rank=$RANK i=$i value=$l")
        return NaN
    end
    try
        Flux.train!(loss_nan, model, 1:5, state)
        marker("TRAIN_RETURNED rank=$RANK")
    catch e
        e isa DomainError || rethrow()
        marker("NONFINITE_STOP rank=$RANK item=$(item[]) exc=DomainError")
    end
    RANK == 1 || error("rank 0 unexpectedly returned from train!")
    marker("HOLDING rank=$RANK after=DomainError")
    while true
        sleep(1)
    end
end

function run_nan_loss_all()
    model = fresh_model()
    logb = LoggingBackend(BACKEND, RANK, Ref(0))
    state = make_state(model; backend = logb)
    shadow = fresh_model()
    sstate = Optimisers.setup(Optimisers.Adam(ETA), shadow)
    for i in 1:2
        sstate, shadow = Optimisers.update(sstate, shadow, shadow_grad(shadow, step_data(i)))
    end
    function loss_nan(m, i)
        l = loss_quad(m, step_data(i))
        i == 3 ? NaN : l
    end
    caught = false
    try
        Flux.train!(loss_nan, model, 1:5, state)
    catch e
        e isa DomainError || rethrow()
        caught = true
        marker("NONFINITE_STOP rank=$RANK item=3")
    end
    d = max(maxdiff(model.w, shadow.w), maxdiff(model.v, shadow.v))
    t, st = state.tree, sstate
    state_ok = t.w.state[1] == st.w.state[1] && t.w.state[2] == st.w.state[2] &&
               t.w.state[3] == st.w.state[3] &&
               t.v.state[1] == st.v.state[1] && t.v.state[2] == st.v.state[2] &&
               t.v.state[3] == st.v.state[3]
    ok = caught && d <= 1e-13 && state_ok
    marker("STOP_STATE_MATCH rank=$RANK caught=$caught maxdiff=$d state_ok=$state_ok")
    mode_result("nan-loss-all", ok)
end

function run_grad_nonfinite()
    model = fresh_model()
    logb = LoggingBackend(BACKEND, RANK, Ref(0))
    state = make_state(model; backend = logb)
    xr = RANK == 0 ? (copy(W0) .- 1.0) : copy(W0)
    loss_sqrt(m) = sqrt(sum(abs2, m.w .- xr)) + sum(abs2, m.v .- V0)
    val, gs = Flux.withgradient(loss_sqrt, model)
    g = gs[1]
    marker("GRAD rank=$RANK loss=$val loss_finite=$(isfinite(val)) " *
           "wgrad_finite=$(all(isfinite, g.w)) vgrad_finite=$(all(isfinite, g.v))")
    exc = "none"
    try
        state, model = Optimisers.update!(state, model, g)
    catch e
        exc = string(typeof(e))
        marker("UPDATE_EXCEPTION rank=$RANK exc=$exc")
    end
    wf = all(isfinite, model.w)
    vf = all(isfinite, model.v)
    mf = all(isfinite, state.tree.w.state[1]) && all(isfinite, state.tree.w.state[2])
    marker("CORRUPTION rank=$RANK exc=$exc w_finite=$wf v_finite=$vf wmom_finite=$mf " *
           "w1=$(model.w[1])")
    mode_result("grad-nonfinite", exc == "none" && !wf && !mf;
                extra = "w_finite=$wf wmom_finite=$mf")
end

function run_step_desync()
    model = fresh_model()
    logb = LoggingBackend(BACKEND, RANK, Ref(0))
    state = make_state(model; backend = logb)

    # Correct 5-step reference.
    correct = fresh_model()
    cstate = Optimisers.setup(Optimisers.Adam(ETA), correct)
    for i in 1:5
        cstate, correct = Optimisers.update(cstate, correct, shadow_grad(correct, step_data(i)))
    end
    # Shifted reference: rank 1 skips step 3, then its step 4 pairs with rank 0's
    # step 3 and its step 5 with rank 0's step 4. Every pair is evaluated from the
    # same replica state, so the actual sequence is:
    #   s1 = Adam(s0, g1), s2 = Adam(s1, g2),
    #   s3 = Adam(s2, (g3+g4)/2), s4 = Adam(s3, (g4+g5)/2).
    shifted = fresh_model()
    shstate = Optimisers.setup(Optimisers.Adam(ETA), shifted)
    for i in 1:2
        shstate, shifted = Optimisers.update(shstate, shifted, shadow_grad(shifted, step_data(i)))
    end
    g3 = shadow_grad(shifted, step_data(3))
    g4a = shadow_grad(shifted, step_data(4))
    shstate, shifted = Optimisers.update(shstate, shifted, combine(g3, g4a))
    g4b = shadow_grad(shifted, step_data(4))
    g5 = shadow_grad(shifted, step_data(5))
    shstate, shifted = Optimisers.update(shstate, shifted, combine(g4b, g5))

    function loss_step(m, i)
        l = loss_quad(m, step_data(i))
        RANK == 1 && i == 3 || return l
        marker("LOSS_NAN rank=$RANK i=$i")
        return NaN
    end
    for i in 1:5
        l = Flux.trainstep!(loss_step, model, i, state)
        marker("STEP_DONE rank=$RANK i=$i loss=$l")
    end
    dc = max(maxdiff(model.w, correct.w), maxdiff(model.v, correct.v))
    ds = max(maxdiff(model.w, shifted.w), maxdiff(model.v, shifted.v))
    marker("DESYNC_RESULT rank=$RANK delta_vs_correct=$dc delta_vs_shifted=$ds")
    RANK == 1 || error("rank 0 unexpectedly finished the loop")
    marker("HOLDING rank=$RANK after=step-desync")
    while true
        sleep(1)
    end
end

function run_early_stop()
    model = fresh_model()
    logb = LoggingBackend(BACKEND, RANK, Ref(0))
    state = make_state(model; backend = logb)
    for i in 1:5
        if RANK == 1 && i == 4
            marker("LOOP_STOP rank=$RANK i=$i reason=rank-local")
            break
        end
        l = Flux.trainstep!(loss_quad, model, step_data(i), state)
        marker("STEP_DONE rank=$RANK i=$i loss=$l")
    end
    RANK == 1 || error("rank 0 unexpectedly finished the loop")
    marker("HOLDING rank=$RANK after=early-stop")
    while true
        sleep(1)
    end
end

function run_config_eta()
    model = fresh_model()
    logb = LoggingBackend(BACKEND, RANK, Ref(0))
    state = make_state(model; backend = logb)
    shadow = fresh_model()
    sstate = Optimisers.setup(Optimisers.Adam(ETA), shadow)
    for i in 1:4
        if i == 2 && RANK == 1
            Optimisers.adjust!(state, 2 * ETA)
            Optimisers.adjust!(sstate, 2 * ETA)
            marker("LOCAL_ADJUST rank=$RANK i=$i eta=$(2 * ETA)")
        end
        x = step_data(i)
        gm = shadow_grad(model, x)
        gs = shadow_grad(shadow, x)
        # The wrapper applies the global-mean gradient with each rank's local rule.
        # Replicate that mean for the shadow's own (possibly different) rule.
        gw = copy(gs.w)
        gv = copy(gs.v)
        MPI.Allreduce!(gw, +, BACKEND.comm)
        MPI.Allreduce!(gv, +, BACKEND.comm)
        gw ./= WORLD
        gv ./= WORLD
        state, model = Optimisers.update!(state, model, gm)
        sstate, shadow = Optimisers.update(sstate, shadow, (; w = gw, v = gv))
        d = max(maxdiff(model.w, shadow.w), maxdiff(model.v, shadow.v))
        marker("CONFIG_STEP rank=$RANK i=$i param_d=$d")
    end
    d = max(maxdiff(model.w, shadow.w), maxdiff(model.v, shadow.v))
    allw = MPI.Allgather(model.w, BACKEND.comm)
    allv = MPI.Allgather(model.v, BACKEND.comm)
    cross = max(maximum(abs, allw[3:4] .- allw[1:2]), maximum(abs, allv[3:4] .- allv[1:2]))
    marker("CONFIG_ETA rank=$RANK self_maxdiff=$d cross_delta=$cross w1=$(model.w[1])")
    mode_result("config-eta", d <= 1e-13 && cross > 1e-6;
                extra = "self_maxdiff=$d cross_delta=$cross")
end

function run_config_frozen()
    model = fresh_model()
    logb = LoggingBackend(BACKEND, RANK, Ref(0))
    state = make_state(model; backend = logb)
    if RANK == 1
        Optimisers.freeze!(state)
        marker("LOCAL_FREEZE rank=$RANK")
    end
    for i in 1:4
        x = step_data(i)
        state, model = Optimisers.update!(state, model, shadow_grad(model, x))
        marker("STEP_DONE rank=$RANK i=$i w1=$(model.w[1])")
    end
    allw = MPI.Allgather(model.w, BACKEND.comm)
    cross = maximum(abs, allw[3:4] .- allw[1:2])
    unchanged = model.w == W0 && model.v == V0
    ok = RANK == 0 ? (cross > 1e-6) : (unchanged && cross > 1e-6)
    marker("CONFIG_FROZEN rank=$RANK cross_delta=$cross unchanged=$unchanged w1=$(model.w[1])")
    mode_result("config-frozen", ok; extra = "cross_delta=$cross")
end

# ------------------------------------------------------------------
# Warm up compilation before the hang budget starts (both ranks run the
# same wrapper path with a throwaway model), then run the requested mode.
# ------------------------------------------------------------------

function warmup()
    model = fresh_model()
    state = make_state(model)
    Flux.train!(loss_quad, model, [step_data(1)], state)
end

warmup()
marker("VERSIONS julia=$(VERSION) flux=$(pkgversion(Flux)) mpi=$(pkgversion(MPI)) " *
       "optimisers=$(pkgversion(Optimisers))")
marker("PROBE_READY rank=$RANK")

if MODE == "control"
    run_control()
elseif MODE == "nan-loss-hang"
    run_nan_loss_hang()
elseif MODE == "nan-loss-all"
    run_nan_loss_all()
elseif MODE == "grad-nonfinite"
    run_grad_nonfinite()
elseif MODE == "step-desync"
    run_step_desync()
elseif MODE == "early-stop"
    run_early_stop()
elseif MODE == "config-eta"
    run_config_eta()
elseif MODE == "config-frozen"
    run_config_frozen()
else
    error("unknown mode $(repr(MODE))")
end

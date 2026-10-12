# Shared utilities for the Phase K conditional-graph demo.
#
# Everything in this file is device-generic and free of wrapper-internal API:
# the recording backend wraps the real backend only around wrapper collectives,
# and every snapshot/comparison is an explicit device-to-host copy outside the
# training communication path.
module CGCommon

using Flux, Optimisers, Functors, MPI, SHA, Serialization, Printf, Zygote

export Tolerances, tolerances_for, RecordingBackend, diagnostic_comm, real_backend,
    leaf_arrays, snapshot, compare_snapshots, format_comparison, head_snapshot,
    check_leaf_config,
    expected_collective_count, check_collective_trace, ensure,
    package_versions, source_hash, git_head, host_name, write_text

# Canonical leaf order of `CGModel.TwoHeadModel` (functor order).
const LEAF_ORDER = (
    (:trunk, :weight), (:trunk, :bias),
    (:head_a, :weight), (:head_a, :bias),
    (:head_b, :weight), (:head_b, :bias),
)

struct Tolerances
    atol::Float64
    rtol::Float64
end

Base.show(io::IO, t::Tolerances) = print(io, "atol=", t.atol, " rtol=", t.rtol)

# CPU Float64 is the diagnostic dtype; Float32/device comparisons need a
# Float32 rounding margin (1 ULP near unit scale is about 1.5e-8).
tolerances_for(dtype) = dtype === Float64 ?
    Tolerances(1.0e-12, 1.0e-10) : Tolerances(2.0e-6, 1.0e-5)

ensure(cond::Bool, msg::AbstractString) = cond || error(msg)

leaf_label(sub, field) = string(sub, "_", field)

# One (param, mom1, mom2, beta) tuple per trainable leaf, in canonical order.
function leaf_arrays(model, tree)
    entries = Any[]
    for (sub, field) in LEAF_ORDER
        p = getfield(getfield(model, sub), field)
        leaf = getfield(getfield(tree, sub), field)
        push!(entries, (; name = leaf_label(sub, field), p, m = leaf.state[1],
            v = leaf.state[2], beta = leaf.state[3]))
    end
    return entries
end

# Flatten param, first moment, second moment, and beta products to one
# `Vector{Float64}` with a per-leaf segment layout. Device arrays are copied to
# the host here explicitly; Float32 values convert losslessly.
function snapshot(model, tree)
    values = Float64[]
    layout = NamedTuple[]
    for entry in leaf_arrays(model, tree)
        start = length(values)
        append!(values, vec(Array(entry.p)))
        append!(values, vec(Array(entry.m)))
        append!(values, vec(Array(entry.v)))
        append!(values, Float64.(entry.beta))
        np = length(entry.p)
        nm = length(entry.m)
        nv = length(entry.v)
        push!(layout, (; name = entry.name,
            param = (start + 1):(start + np),
            mom1 = (start + np + 1):(start + np + nm),
            mom2 = (start + np + nm + 1):(start + np + nm + nv),
            beta = (start + np + nm + nv + 1):(start + np + nm + nv + 2)))
    end
    return (; values, layout)
end

# Elementwise combined absolute/relative criterion:
#     |a - b| <= atol + rtol * |b|
# Returns the maximum absolute error and the worst scaled error (> 1 is a fail).
function segment_error(a, b, r, tol::Tolerances)
    maxabs = 0.0
    worst = 0.0
    @inbounds for i in r
        e = abs(a[i] - b[i])
        maxabs = max(maxabs, e)
        worst = max(worst, e / (tol.atol + tol.rtol * abs(b[i])))
    end
    return (; maxabs, worst)
end

# Compare two snapshots (host Float64 vectors with the same layout).
function compare_snapshots(a, b, layout, tol::Tolerances; context="snapshot comparison")
    length(a) == length(b) || error("snapshot length mismatch")
    param = (maxabs = 0.0, worst = 0.0)
    mom1 = (maxabs = 0.0, worst = 0.0)
    mom2 = (maxabs = 0.0, worst = 0.0)
    beta_exact = true
    for seg in layout
        e = segment_error(a, b, seg.param, tol)
        param = (maxabs = max(param.maxabs, e.maxabs), worst = max(param.worst, e.worst))
        e = segment_error(a, b, seg.mom1, tol)
        mom1 = (maxabs = max(mom1.maxabs, e.maxabs), worst = max(mom1.worst, e.worst))
        e = segment_error(a, b, seg.mom2, tol)
        mom2 = (maxabs = max(mom2.maxabs, e.maxabs), worst = max(mom2.worst, e.worst))
        for i in seg.beta
            a[i] == b[i] || (beta_exact = false)
        end
        for (kind, range) in ((:parameter, seg.param), (:moment1, seg.mom1), (:moment2, seg.mom2))
            err = segment_error(a, b, range, tol)
            ensure(err.worst <= 1, "$context: first divergent leaf $(seg.name) $kind: maxabs=$(err.maxabs) scaled=$(err.worst)")
        end
    end
    return (; param, mom1, mom2, beta_exact)
end

function format_comparison(c)
    return @sprintf("param=(%.3e,%.3f) mom1=(%.3e,%.3f) mom2=(%.3e,%.3f) beta_exact=%s",
        c.param.maxabs, c.param.worst, c.mom1.maxabs, c.mom1.worst,
        c.mom2.maxabs, c.mom2.worst, c.beta_exact)
end

# Full snapshot of one head's parameters and complete native leaf history.
function head_snapshot(model, tree, head::Symbol)
    sub = head === :a ? :head_a : :head_b
    layer = getfield(model, sub)
    leaf_w = getfield(getfield(tree, sub), :weight)
    leaf_b = getfield(getfield(tree, sub), :bias)
    return (; w = Array(layer.weight), b = Array(layer.bias),
        mw = Array(leaf_w.state[1]), vw = Array(leaf_w.state[2]), bw = leaf_w.state[3],
        mb = Array(leaf_b.state[1]), vb = Array(leaf_b.state[2]), bb = leaf_b.state[3])
end

# Rule configuration and frozen flags are rank-local in the wrapper contract;
# a continuation must restore identical values on every rank.
function check_leaf_config(saved, live; context::AbstractString)
    for (sub, field) in LEAF_ORDER
        s = getfield(getfield(saved, sub), field)
        l = getfield(getfield(live, sub), field)
        ensure(l.frozen == s.frozen,
            "$context: frozen flag for $(leaf_label(sub, field)) is $(l.frozen), " *
            "expected $(s.frozen)")
        for p in (:eta, :beta, :epsilon)
            ensure(getproperty(l.rule, p) == getproperty(s.rule, p),
                "$context: rule field $p for $(leaf_label(sub, field)) is " *
                "$(getproperty(l.rule, p)), expected $(getproperty(s.rule, p))")
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Recording backend: wraps the real backend only for wrapper collectives.
# Diagnostics always use the real backend (`real_backend`) so the wrapper
# trace contains exactly the wrapper protocol.
# ---------------------------------------------------------------------------

struct RecordingBackend{B} <: Flux.AbstractFluxDistributedBackend
    inner::B
    log::Vector{Any}
end

function Flux.DistributedUtils.allreduce!(rec::RecordingBackend, buf, op)
    # The presence vector is the only host Float32 buffer the wrapper reduces.
    # Record the local contribution and the reduced result separately: the
    # reduced flags drive the global-presence decision on every rank.
    # CPU Float32 bias gradients are vectors too. Only the first call is presence.
    presence = isempty(rec.log)
    local_flags = presence && buf isa Vector{Float32} ? copy(buf) : nothing
    result = Flux.DistributedUtils.allreduce!(rec.inner, buf, op)
    global_flags = presence && buf isa Vector{Float32} ? copy(buf) : nothing
    push!(rec.log, (; buftype = typeof(buf), size = size(buf),
        avg = op === Flux.DistributedUtils.avg, local_flags, global_flags))
    return result
end

real_backend(rec::RecordingBackend) = rec.inner

diagnostic_comm(backend::Flux.MPIBackend) = backend.comm
diagnostic_comm(backend::Flux.NCCLBackend) = backend.mpi_backend.comm

# ---------------------------------------------------------------------------
# Expected wrapper protocol for the three-Dense two-head model.
# ---------------------------------------------------------------------------

function expected_collective_count(present_a::Bool, present_b::Bool)
    return 1 + 2 + 2 * present_a + 2 * present_b
end

function expected_buffer_shapes(model, present_a::Bool, present_b::Bool)
    shapes = Any[]
    for (sub, field) in LEAF_ORDER
        sub === :head_a && !present_a && continue
        sub === :head_b && !present_b && continue
        p = getfield(getfield(model, sub), field)
        push!(shapes, (leaf_label(sub, field), size(p), typeof(p)))
    end
    return shapes
end

function check_collective_trace(step_log, model, step::Int, rank::Int,
        local_head::Symbol, present::NamedTuple, counts::NamedTuple, R::Int,
        on_gpu::Bool)
    n_expected = expected_collective_count(present.a, present.b)
    ensure(length(step_log) == n_expected,
        "step $step rank $rank: wrapper issued $(length(step_log)) collectives, " *
        "expected $n_expected (presence a=$(present.a) b=$(present.b))")

    first_entry = step_log[1]
    ensure(first_entry.buftype === Vector{Float32},
        "step $step rank $rank: presence buffer type $(first_entry.buftype) is not " *
        "host Vector{Float32}")
    ensure(first_entry.size == (6,),
        "step $step rank $rank: presence buffer size $(first_entry.size) != (6,)")
    ensure(first_entry.avg, "step $step rank $rank: presence reduction is not avg")
    expected_local = Float32[1, 1, local_head === :a ? 1 : 0, local_head === :a ? 1 : 0,
        local_head === :b ? 1 : 0, local_head === :b ? 1 : 0]
    ensure(first_entry.local_flags == expected_local,
        "step $step rank $rank: local presence flags $(first_entry.local_flags) != " *
        "$expected_local")
    expected_global = Float32[1, 1, counts.a / R, counts.a / R, counts.b / R, counts.b / R]
    ensure(first_entry.global_flags == expected_global,
        "step $step rank $rank: reduced presence flags $(first_entry.global_flags) != " *
        "$expected_global (counts a=$(counts.a) b=$(counts.b) over R=$R)")

    expected = expected_buffer_shapes(model, present.a, present.b)
    for (i, (label, shape, array_type)) in enumerate(expected)
        entry = step_log[i + 1]
        ensure(entry.buftype === array_type,
            "step $step rank $rank: buffer type for $label differs from model parameter")
        ensure(entry.size == shape,
            "step $step rank $rank: gradient collective $i for $label has size " *
            "$(entry.size), expected $shape (canonical order)")
        ensure(entry.avg, "step $step rank $rank: gradient collective $i for $label " *
                          "is not avg")
        # The module stays CUDA-free so CPU runs never load the CUDA stack:
        # a device buffer is any array that is not a host `Array`.
        is_host = entry.buftype <: Array
        ensure(on_gpu ? !is_host : is_host,
            "step $step rank $rank: gradient collective $i for $label " *
            "has buffer type $(entry.buftype), expected a " *
            (on_gpu ? "device array" : "host Array"))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

function package_versions()
    versions = [
    (name = "Flux", version = string(pkgversion(Flux)), path = pathof(Flux)),
    (name = "Optimisers", version = string(pkgversion(Optimisers)), path = pathof(Optimisers)),
    (name = "Functors", version = string(pkgversion(Functors)), path = pathof(Functors)),
    (name = "MPI", version = string(pkgversion(MPI)), path = pathof(MPI)),
    (name = "Zygote", version = string(pkgversion(Zygote)), path = pathof(Zygote)),
]
    for name in (:CUDA, :NCCL, :cuDNN)
        if isdefined(Main, name)
            m = getfield(Main, name)
            push!(versions, (name=string(name), version=string(pkgversion(m)), path=pathof(m)))
        end
    end
    return versions
end

source_hash(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))

function git_head(path::AbstractString)
    try
        return strip(read(pipeline(`git -C $path rev-parse HEAD`, stderr = devnull), String))
    catch
        return "unavailable"
    end
end

host_name() = gethostname()

function write_text(path::AbstractString, text::AbstractString)
    open(path, "w") do io
        write(io, text)
    end
    return path
end

end # module

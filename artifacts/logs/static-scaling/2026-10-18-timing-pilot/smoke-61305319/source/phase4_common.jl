"""
Shared implementation for static-scaling Phase 4 (distributed correctness).

This file is included at runtime by `phase4_flux.jl` after `replay.jl`,
`model_flux.jl`, and `precision.jl`. It owns:

* the recording distributed backend that observes every wrapper `allreduce!`;
* the distributed optimizer state with the frozen per-leaf `lambda` contract;
* fixture sharding, NPZ/JSON artifact writing, package/source identity, and
  state hashes;
* array capture for the reference (337-array Phase 3 schema), DDP (per-rank),
  and split-control artifacts.

The training math stays in `StaticScalingFlux`; this file only drives it.
No loss or gradient scaling is applied here: the wrapper averages once as
`g = sum_r(g_r) / world_size` inside `Optimisers.update!`.
"""
module StaticScalingPhase4

using Flux
using Optimisers
using Functors
using Zygote
using NPZ
using SHA
using Printf
using Dates
using LinearAlgebra: BLAS

using ..Main: read_npz_fields, rank_shard, parse_cli_args, nchw_to_whcn, fmt_float,
    c_order_sha256
using ..StaticScalingFlux

export RecordingBackend, real_backend, build_distributed_state,
       capture_reference_arrays, capture_ddp_arrays,
       run_reference, run_shard_reference, run_ddp, run_split_control, parse_phase4_args

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_FIXTURE_DIR =
    joinpath(REPO_ROOT, "artifacts", "baselines", "static-scaling", "2026-10-16-phase2")
const EXPECTED_UPDATES = 3
const REFERENCE_ARRAY_COUNT = 337
const EXPECTED_MODEL_ARRAYS = 62
const EXPECTED_PARAMETER_COUNT = 11_271_432

const MPI_UUID = "da04e1cc-30fd-572f-bb4f-1f8673147195"
const NCCL_UUID = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
const CUDA_UUID = "052768ef-5323-5732-b1bb-66c8b64840ba"
const CUDNN_UUID = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

const PACKAGE_UUIDS = [
    ("Flux", "587475ba-b771-5e3f-ad9e-33799f191a9c"),
    ("Optimisers", "3bd65402-5787-11e9-1adc-39752487f4e2"),
    ("Functors", "d9f16b24-f501-4c13-a1f2-28368ffc5196"),
    ("Zygote", "e88e6eb3-aa80-5325-afca-941959d7151f"),
    ("NPZ", "15e1cf62-19b3-5cfa-8e77-841668bca605"),
    ("MPI", MPI_UUID),
    ("NCCL", NCCL_UUID),
    ("CUDA", CUDA_UUID),
    ("cuDNN", CUDNN_UUID),
]

const REF_ACTIVATIONS = [
    ("act.stem.gn", :stem_gn),
    ("act.stem", :stem),
    ("act.stage1.block0", :stage1_block0),
    ("act.stage1.block1", :stage1_block1),
    ("act.stage2.block0", :stage2_block0),
    ("act.stage2.block1", :stage2_block1),
    ("act.stage3.block0", :stage3_block0),
    ("act.stage3.block1", :stage3_block1),
    ("act.stage4.block0", :stage4_block0),
    ("act.stage4.block1", :stage4_block1),
    ("act.pooled", :pooled),
]

# ---------------------------------------------------------------------------
# Recording backend (wrapper collectives only; diagnostics use the real one)
# ---------------------------------------------------------------------------

"""
    RecordingBackend(inner)

Wrap a real distributed backend. Every `Flux.DistributedUtils.allreduce!`
issued by the training wrapper is appended to `log` as
`(buftype, size, avg)` and then delegated unchanged. Diagnostics, barriers,
and broadcasts use `real_backend(rec)` so the trace contains exactly the
wrapper protocol.
"""
struct RecordingBackend{B} <: Flux.AbstractFluxDistributedBackend
    inner::B
    log::Vector{Any}
end

RecordingBackend(inner) = RecordingBackend(inner, Any[])

function Flux.DistributedUtils.allreduce!(rec::RecordingBackend, sendrecvbuf, op)
    result = Flux.DistributedUtils.allreduce!(rec.inner, sendrecvbuf, op)
    push!(rec.log, (buftype = typeof(sendrecvbuf), size = size(sendrecvbuf),
                    avg = op === Flux.DistributedUtils.avg))
    return result
end

function Flux.DistributedUtils.allreduce!(rec::RecordingBackend, sendbuf, recvbuf, op)
    result = Flux.DistributedUtils.allreduce!(rec.inner, sendbuf, recvbuf, op)
    push!(rec.log, (buftype = typeof(recvbuf), size = size(recvbuf),
                    avg = op === Flux.DistributedUtils.avg))
    return result
end

Flux.DistributedUtils.local_rank(rec::RecordingBackend) =
    Flux.DistributedUtils.local_rank(rec.inner)
Flux.DistributedUtils.total_workers(rec::RecordingBackend) =
    Flux.DistributedUtils.total_workers(rec.inner)

real_backend(rec::RecordingBackend) = rec.inner

# ---------------------------------------------------------------------------
# Distributed optimizer state
# ---------------------------------------------------------------------------

"""
    build_distributed_state(model, backend; eta=ETA_MAX)

Build `Optimisers.setup(DistributedOptimizer(backend, AdamW(...)), model)` and
apply the same per-leaf contract as `build_optimizer_state`: 21 decay leaves
with `lambda=1e-4`, 41 no-decay leaves with `lambda=0`, zero initial moments,
and the frozen beta powers. The outer wrapper state is returned; leaf
traversal uses `state.tree`.
"""
function build_distributed_state(model, backend; eta::Real = ETA_MAX)
    rule = Optimisers.AdamW(; eta = eta, beta = BETA,
                            lambda = WEIGHT_DECAY, epsilon = EPSILON,
                            couple = true)
    dist_rule = Flux.DistributedUtils.DistributedOptimizer(backend, rule)
    state = Optimisers.setup(dist_rule, model)
    state isa Flux.DistributedUtils.DistributedOptimizerState ||
        error("distributed setup returned $(typeof(state)); expected DistributedOptimizerState")
    state.backend === backend ||
        error("distributed state backend $(typeof(state.backend)) is not the supplied backend")

    leaves = state_leaves(state.tree)
    length(leaves) == EXPECTED_MODEL_ARRAYS ||
        error("distributed optimizer state has $(length(leaves)) leaves; expected $(EXPECTED_MODEL_ARRAYS)")

    want_beta_power = (Float32(BETA[1]), Float32(BETA[2]))
    ndecay = 0
    for (name, leaf) in leaves
        leaf.rule isa Optimisers.AdamW ||
            error("distributed optimizer leaf $(name) uses $(typeof(leaf.rule)); expected AdamW")
        want_lambda = is_decay_param(name) ? WEIGHT_DECAY : 0.0
        Optimisers.adjust!(leaf; lambda = want_lambda)
        leaf.rule.lambda == want_lambda ||
            error("distributed optimizer leaf $(name) has lambda=$(leaf.rule.lambda); expected $(want_lambda)")
        leaf.rule.eta == float(eta) ||
            error("distributed optimizer leaf $(name) has eta=$(leaf.rule.eta); expected $(float(eta))")
        leaf.rule.beta == BETA ||
            error("distributed optimizer leaf $(name) has beta=$(leaf.rule.beta); expected $(BETA)")
        leaf.rule.epsilon == EPSILON ||
            error("distributed optimizer leaf $(name) has epsilon=$(leaf.rule.epsilon); expected $(EPSILON)")
        leaf.rule.couple === true ||
            error("distributed optimizer leaf $(name) has couple=$(leaf.rule.couple); expected true")
        m, v, beta_power = leaf.state
        all(iszero, m) ||
            error("distributed optimizer leaf $(name) has nonzero initial m")
        all(iszero, v) ||
            error("distributed optimizer leaf $(name) has nonzero initial v")
        beta_power == want_beta_power ||
            error("distributed optimizer leaf $(name) has beta_power=$(beta_power); expected $(want_beta_power)")
        ndecay += want_lambda != 0.0
    end
    ndecay == 21 ||
        error("distributed optimizer has $(ndecay) decay leaves; expected 21 (41 no-decay)")
    return state
end

_tree(state::Flux.DistributedUtils.DistributedOptimizerState) = state.tree
_tree(state) = state

function _state_leaf_map(state)
    return Dict(name => leaf for (name, leaf) in state_leaves(_tree(state)))
end

# ---------------------------------------------------------------------------
# C-order hashing, JSON, and provenance
# ---------------------------------------------------------------------------

function general_c_order_bytes(a::AbstractArray)
    isbitstype(eltype(a)) || error("general_c_order_bytes: unsupported eltype $(eltype(a))")
    b = ndims(a) <= 1 ? Array(a) : permutedims(a, reverse(1:ndims(a)))
    return Vector{UInt8}(reinterpret(UInt8, vec(b)))
end

_general_array(a::AbstractArray) = a
_general_array(a::Number) = reshape([a], ())

function general_c_order_sha256(a)
    arr = _general_array(a)
    T = eltype(arr)
    if T === Float32 || T === UInt8 || T === Int64
        return c_order_sha256(arr)
    end
    return bytes2hex(sha256(general_c_order_bytes(arr)))
end

function _concat_sha256(parts)
    ctx = SHA.SHA2_256_CTX()
    for p in parts
        SHA.update!(ctx, general_c_order_bytes(_general_array(p)))
    end
    return bytes2hex(SHA.digest!(ctx))
end

function _json_escape(s::AbstractString)
    io = IOBuffer()
    for c in s
        if c == '"'
            write(io, "\\\"")
        elseif c == '\\'
            write(io, "\\\\")
        elseif c == '\n'
            write(io, "\\n")
        elseif c == '\r'
            write(io, "\\r")
        elseif c == '\t'
            write(io, "\\t")
        elseif UInt32(c) < 0x20
            print(io, "\\u", lpad(string(UInt32(c); base = 16), 4, '0'))
        else
            write(io, c)
        end
    end
    return String(take!(io))
end

_json_key(k) = _json_escape(String(k))

function _json_write(io::IO, obj, indent::Int)
    pad = "  " ^ indent
    if obj === nothing || obj === missing
        write(io, "null")
    elseif obj isa Bool
        write(io, obj ? "true" : "false")
    elseif obj isa Integer
        print(io, obj)
    elseif obj isa AbstractFloat
        isfinite(obj) ? print(io, repr(Float64(obj))) : print(io, repr(obj))
    elseif obj isa AbstractString || obj isa Symbol
        write(io, '"', _json_escape(String(obj)), '"')
    elseif obj isa NamedTuple
        _json_write(io, Dict(string(k) => v for (k, v) in pairs(obj)), indent)
    elseif obj isa AbstractDict
        keys_sorted = sort!(collect(keys(obj)); by = string)
        write(io, "{")
        if !isempty(keys_sorted)
            write(io, "\n")
            for (i, k) in enumerate(keys_sorted)
                write(io, pad, "  ", '"', _json_key(k), '"', ": ")
                _json_write(io, obj[k], indent + 1)
                i < length(keys_sorted) && write(io, ",")
                write(io, "\n")
            end
            write(io, pad)
        end
        write(io, "}")
    elseif obj isa AbstractVector || obj isa Tuple
        write(io, "[")
        if !isempty(obj)
            write(io, "\n")
            for (i, v) in enumerate(obj)
                write(io, pad, "  ")
                _json_write(io, v, indent + 1)
                i < length(obj) && write(io, ",")
                write(io, "\n")
            end
            write(io, pad)
        end
        write(io, "]")
    else
        _json_write(io, string(obj), indent)
    end
    return nothing
end

function write_json(path::AbstractString, obj)
    mkpath(dirname(path))
    open(path, "w") do io
        _json_write(io, obj, 0)
        write(io, "\n")
    end
    return path
end

utc_now() = Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SSZ")

sha256_file(path::AbstractString) = open(path, "r") do io
    bytes2hex(sha256(io))
end

function _loaded_package(name::AbstractString, uuid::AbstractString)
    id = Base.PkgId(Base.UUID(uuid), name)
    return get(Base.loaded_modules, id, nothing)
end

function _require_package(name::AbstractString, uuid::AbstractString)
    id = Base.PkgId(Base.UUID(uuid), name)
    return Base.require(id)
end

function package_versions()
    out = Any[]
    for (name, uuid) in PACKAGE_UUIDS
        mod = _loaded_package(name, uuid)
        mod === nothing && continue
        version = try
            v = Base.pkgversion(mod)
            v === nothing ? "unknown" : string(v)
        catch
            "unknown"
        end
        path = try
            string(pathof(mod))
        catch
            ""
        end
        push!(out, Dict("name" => name, "version" => version, "path" => path))
    end
    return out
end

function _git_output(path::AbstractString, args::Vector{String})
    try
        return strip(read(`git -C $(path) $(args)`, String))
    catch
        return nothing
    end
end

git_head(path::AbstractString) = something(_git_output(path, ["rev-parse", "HEAD"]), "unavailable")

function _git_state_sha256(root::AbstractString; exclude_artifacts::Bool)
    filt = exclude_artifacts ? "| grep -v '^artifacts/'" : ""
    cmd = "cd \"$(root)\" && { git diff HEAD; " *
          "git ls-files --others --exclude-standard " *
          "| sort $(filt) | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1"
    try
        return strip(read(`bash -lc $(cmd)`, String))
    catch
        return "unavailable"
    end
end

function flux_repo_root()
    try
        return normpath(joinpath(dirname(dirname(pathof(Flux)))))
    catch
        return "unavailable"
    end
end

function source_identity(opts)
    thesis_commit = get(opts, "source-commit", get(ENV, "SOURCE_COMMIT", ""))
    thesis_state = get(opts, "source-state-sha256", get(ENV, "SOURCE_STATE_SHA256", ""))
    flux_commit = get(opts, "flux-commit", get(ENV, "FLUX_COMMIT", ""))
    flux_state = get(opts, "flux-state-sha256", get(ENV, "FLUX_STATE_SHA256", ""))
    isempty(thesis_commit) && (thesis_commit = git_head(REPO_ROOT))
    isempty(thesis_state) &&
        (thesis_state = _git_state_sha256(REPO_ROOT; exclude_artifacts = true))
    flux_root = flux_repo_root()
    isempty(flux_commit) && (flux_commit = git_head(flux_root))
    isempty(flux_state) &&
        (flux_state = _git_state_sha256(flux_root; exclude_artifacts = false))
    return Dict(
        "thesis_commit" => thesis_commit,
        "thesis_state_sha256" => thesis_state,
        "flux_commit" => flux_commit,
        "flux_state_sha256" => flux_state,
        "flux_repo_root" => flux_root,
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => try
            BLAS.get_num_threads()
        catch
            nothing
        end,
        "host" => gethostname(),
    )
end

function bundle_array_hash(bundle; names = bundle_parameter_names())
    return _concat_sha256([bundle[name] for name in names])
end

function model_array_hash(model; names = bundle_parameter_names())
    return _concat_sha256([canonical_param(name, host_array(param_by_name(model, name)))
                           for name in names])
end

function state_hashes(model, state; names = bundle_parameter_names())
    leaves = _state_leaf_map(state)
    param_arrays = AbstractArray[]
    m_arrays = AbstractArray[]
    v_arrays = AbstractArray[]
    bp_arrays = AbstractArray[]
    for name in names
        haskey(leaves, name) || error("distributed state is missing leaf $(name)")
        leaf = leaves[name]
        push!(param_arrays, canonical_param(name, host_array(param_by_name(model, name))))
        push!(m_arrays, canonical_param(name, host_array(leaf.state[1])))
        push!(v_arrays, canonical_param(name, host_array(leaf.state[2])))
        bp = leaf.state[3]
        push!(bp_arrays, Float32[Float32(bp[1]), Float32(bp[2])])
    end
    return Dict(
        "param" => _concat_sha256(param_arrays),
        "m" => _concat_sha256(m_arrays),
        "v" => _concat_sha256(v_arrays),
        "beta_power" => _concat_sha256(bp_arrays),
    )
end

# ---------------------------------------------------------------------------
# Fixture and batch helpers
# ---------------------------------------------------------------------------

function _fixture_paths(opts)
    fixture_dir = get(opts, "fixture-dir", "")
    isempty(fixture_dir) && (fixture_dir = DEFAULT_FIXTURE_DIR)
    fixture_dir = abspath(fixture_dir)
    isdir(fixture_dir) || error("--fixture-dir not found: $(fixture_dir)")
    data_path = joinpath(fixture_dir, "data_fixture.npz")
    isfile(data_path) || error("data fixture not found: $(data_path)")
    bundle_path = get(opts, "bundle", "")
    isempty(bundle_path) && (bundle_path = joinpath(fixture_dir, "initial_bundle.npz"))
    bundle_path = abspath(bundle_path)
    isfile(bundle_path) || error("bundle not found: $(bundle_path)")
    manifest_path = joinpath(fixture_dir, "fixture_manifest.json")
    return fixture_dir, data_path, bundle_path, manifest_path
end

function _global_batch(data, row::Int)
    refs = data["train_transformed_ref"]
    labels = vec(Int64.(data["train_labels"]))
    x_nchw = Array(refs[row, :, :, :, :])
    return x_nchw, labels
end

function _batch_for_update(data, t::Int)
    t in 1:3 || error("update index $(t) outside 1:3")
    row = t == 2 ? 2 : 1
    return _global_batch(data, row)
end

function shard_batch(x_nchw, labels, rank::Integer, world::Integer)
    ids = rank_shard(collect(1:size(x_nchw, 1)), rank, world)
    return x_nchw[ids, :, :, :], labels[ids], ids
end

function _to_training_batch(x_nchw, labels, to_device)
    x = to_device(nchw_to_whcn(x_nchw))
    y = to_device(Flux.onehotbatch(labels .+ 1, 1:200))
    return x, y
end

host_array(a::Array) = copy(a)
host_array(a::AbstractArray) = Array(a)
host_array(a::Number) = a

function loss32(loss)
    loss isa Number && return Float32(loss)
    a = host_array(loss)
    a isa AbstractArray || return Float32(a)
    length(a) == 1 && return Float32(a[])
    return Float32(sum(a) / length(a))
end

function _gradient_map(grads, names; canonical = true)
    out = Dict{String,Any}()
    for name in names
        g = gradient_for(grads, name)
        g === nothing && error("missing gradient for $(name)")
        a = host_array(g)
        out[name] = canonical ? canonical_param(name, a) : a
    end
    return out
end

# ---------------------------------------------------------------------------
# Array capture
# ---------------------------------------------------------------------------

activation_to_canonical(key::Symbol, a) =
    key === :pooled ? permutedims(host_array(a), (2, 1)) :
    permutedims(host_array(a), (4, 3, 2, 1))

function expected_reference_names()
    names = String[]
    for t in 1:3
        for prefix in ("input", "label", "eta", "loss", "logits")
            push!(names, "$(prefix).u$(t)")
        end
    end
    append!(names, first.(REF_ACTIVATIONS))
    for prefix in ("grad.u1", "grad.u3", "param.u3", "m.u3", "v.u3")
        for name in bundle_parameter_names()
            push!(names, "$(prefix).$(name)")
        end
    end
    push!(names, "step.u3")
    return names
end

function _activation_arrays(acts)
    return Pair{String,Any}[
        refname => activation_to_canonical(key, getproperty(acts, key))
        for (refname, key) in REF_ACTIVATIONS]
end

"""
    capture_reference_arrays(...)

Assemble the exact 337-array Phase 3 reference schema. Activations are
captured at u1, gradients at u1/u3, and parameters/moments after u3.
"""
function capture_reference_arrays(inputs, labels, etas, losses, logits,
                                  acts_u1, grads_u1, grads_u3, model, state)
    names = bundle_parameter_names()
    entries = Pair{String,Any}[]
    for t in 1:3
        push!(entries, "input.u$(t)" => inputs[t])
        push!(entries, "label.u$(t)" => labels[t])
        push!(entries, "eta.u$(t)" => Float64[etas[t]])
        push!(entries, "loss.u$(t)" => Float32[losses[t]])
        push!(entries, "logits.u$(t)" => permutedims(logits[t], (2, 1)))
    end
    append!(entries, _activation_arrays(acts_u1))

    g1 = _gradient_map(grads_u1, names)
    g3 = _gradient_map(grads_u3, names)
    for name in names
        push!(entries, "grad.u1.$(name)" => g1[name])
    end
    for name in names
        push!(entries, "grad.u3.$(name)" => g3[name])
    end
    for (prefix, arrays) in (
            ("param.u3", Dict(name => canonical_param(name, host_array(param_by_name(model, name)))
                              for name in names)),
            ("m.u3", Dict(name => canonical_param(name, host_array(leaf.state[1]))
                          for (name, leaf) in state_leaves(_tree(state)))),
            ("v.u3", Dict(name => canonical_param(name, host_array(leaf.state[2]))
                          for (name, leaf) in state_leaves(_tree(state)))))
        for name in names
            push!(entries, "$(prefix).$(name)" => arrays[name])
        end
    end
    push!(entries, "step.u3" => Int64[EXPECTED_UPDATES])

    expected = expected_reference_names()
    actual = first.(entries)
    length(actual) == REFERENCE_ARRAY_COUNT ||
        error("reference capture produced $(length(actual)) arrays; expected $(REFERENCE_ARRAY_COUNT)")
    actual == expected ||
        error("reference capture name order mismatch: $(first(setdiff(expected, actual), 3))")
    return entries
end

function _beta_power_map(state; names = bundle_parameter_names())
    leaves = _state_leaf_map(state)
    out = Dict{String,Any}()
    for name in names
        bp = leaves[name].state[3]
        out[name] = Float32[Float32(bp[1]), Float32(bp[2])]
    end
    return out
end

"""
    capture_ddp_arrays(...)

Assemble one rank's DDP NPZ entries (contract schema). All arrays are host
copies in canonical layout.
"""
function capture_ddp_arrays(rank::Integer, world::Integer, device_index::Integer,
                            inputs, labels, etas, losses, logits,
                            localgrads, model, state)
    names = bundle_parameter_names()
    entries = Pair{String,Any}[]
    push!(entries, "meta.rank" => Int64(rank))
    push!(entries, "meta.world_size" => Int64(world))
    push!(entries, "meta.device_index" => Int64(device_index))
    push!(entries, "meta.updates" => Int64(EXPECTED_UPDATES))
    push!(entries, "input.u1" => inputs[1])
    push!(entries, "input.u3" => inputs[3])
    push!(entries, "label.u1" => labels[1])
    push!(entries, "label.u3" => labels[3])
    for t in 1:3
        push!(entries, "eta.u$(t)" => Float64[etas[t]])
    end
    for t in 1:3
        push!(entries, "loss.u$(t)" => Float32[losses[t]])
    end
    push!(entries, "logits.u1" => permutedims(logits[1], (2, 1)))
    push!(entries, "logits.u3" => permutedims(logits[3], (2, 1)))
    for t in 1:3
        g = localgrads[t]
        for name in names
            push!(entries, "localgrad.u$(t).$(name)" => g[name])
        end
    end
    for (prefix, which) in (("param.u3", 0), ("m.u3", 1), ("v.u3", 2))
        if which == 0
            arrays = Dict(name => canonical_param(name, host_array(param_by_name(model, name)))
                          for name in names)
        else
            arrays = Dict(name => canonical_param(name, host_array(leaf.state[which]))
                          for (name, leaf) in state_leaves(_tree(state)))
        end
        for name in names
            push!(entries, "$(prefix).$(name)" => arrays[name])
        end
    end
    bp = _beta_power_map(state)
    for name in names
        push!(entries, "beta_power.u3.$(name)" => bp[name])
    end
    return entries
end

# ---------------------------------------------------------------------------
# Metadata and artifact writing
# ---------------------------------------------------------------------------

function _dtype_name(T::Type)
    T === Float32 && return "float32"
    T === Float64 && return "float64"
    T === Int64 && return "int64"
    T === UInt8 && return "uint8"
    return lowercase(string(T))
end

function _is_conv_weight_name(name::AbstractString)
    return endswith(name, ".conv.weight") ||
           endswith(name, ".conv1.weight") ||
           endswith(name, ".conv2.weight") ||
           endswith(name, ".proj.weight")
end

function _parameter_layout(name::AbstractString)
    _is_conv_weight_name(name) && return "conv_kernel_out_in_kh_kw"
    name == "classifier.weight" && return "linear_out_in"
    return "vector"
end

function _parameter_name_from_array(name::AbstractString)
    for prefix in ("localgrad.u1.", "localgrad.u2.", "localgrad.u3.",
                   "param.u3.", "m.u3.", "v.u3.", "beta_power.u3.",
                   "grad.u1.", "grad.u3.")
        startswith(name, prefix) && return name[(length(prefix) + 1):end]
    end
    m = match(r"^grad\.u[13]\.(full|half1|half2|halfavg)\.(.*)$", name)
    m === nothing || return String(m.captures[2])
    return nothing
end

function _layout_for(name::AbstractString)
    startswith(name, "meta.") && return "scalar_int64"
    startswith(name, "input.") && return "nchw_float32_normalized"
    startswith(name, "label.") && return "int64_vector"
    startswith(name, "eta.") && return "scalar_float64"
    startswith(name, "loss.") && return "scalar_float32"
    name == "step.u3" && return "scalar_int64"
    startswith(name, "logits.") && return "logits_batch_classes"
    startswith(name, "act.") &&
        return name == "act.pooled" ? "features_batch_channels" : "nchw_float32"
    param = _parameter_name_from_array(name)
    param === nothing && return "unknown"
    return _parameter_layout(param)
end

function array_metadata(entries)
    out = Any[]
    for (name, a) in entries
        arr = _general_array(a)
        push!(out, Dict(
            "name" => name,
            "dtype" => _dtype_name(eltype(arr)),
            "shape" => collect(Int, size(arr)),
            "layout" => _layout_for(name),
            "sha256" => general_c_order_sha256(arr),
        ))
    end
    return out
end

function _write_npz(path::AbstractString, entries)
    mkpath(dirname(path))
    arrays = Dict{String,Any}(String(name) => host_array(a) for (name, a) in entries)
    NPZ.npzwrite(path, arrays)
    return path
end

function _verify_npz(path::AbstractString, entries)
    stored = read_npz_fields(path)
    expected_names = Set(String(name) for (name, _) in entries)
    Set(keys(stored)) == expected_names ||
        error("NPZ $(path) name-set mismatch after write")
    for (name, a) in entries
        arr = _general_array(a)
        got = stored[String(name)]
        got_arr = _general_array(got)
        (eltype(got_arr) === eltype(arr) && size(got_arr) == size(arr)) ||
            error("NPZ $(path) array $(name) metadata mismatch after write")
        general_c_order_sha256(got_arr) == general_c_order_sha256(arr) ||
            error("NPZ $(path) array $(name) hash mismatch after write")
    end
    return length(entries)
end

function _refuse_existing(paths; force::Bool)
    force && return nothing
    existing = [p for p in paths if isfile(p)]
    isempty(existing) ||
        error("refusing to overwrite existing output(s): $(join(existing, ", ")); use --force")
    return nothing
end

function _fixture_hashes(fixture_dir, data_path, bundle_path, manifest_path)
    out = Dict("data_fixture.npz" => sha256_file(data_path),
               "initial_bundle.npz" => sha256_file(bundle_path))
    isfile(manifest_path) && (out["fixture_manifest.json"] = sha256_file(manifest_path))
    for name in ("data_fixture.npz", "initial_bundle.npz", "fixture_manifest.json")
        path = joinpath(fixture_dir, name)
        isfile(path) && (out[name] = sha256_file(path))
    end
    return out
end

function _training_metadata()
    return Dict(
        "updates" => EXPECTED_UPDATES,
        "batch_size" => 12,
        "tags" => ["u1", "u2", "u3"],
        "inputs" => Dict("u1" => "train_transformed_ref[0]",
                         "u2" => "train_transformed_ref[1]",
                         "u3" => "train_transformed_ref[0]"),
        "labels" => "train_labels (int64, 0-based)",
        "loss" => "mean cross-entropy on raw logits",
        "captured_activations" => "u1",
        "captured_gradients" => ["u1", "u3"],
        "captured_optimizer_state" => "u3",
        "loss_scaling" => "none",
        "gradient_average" => "g = sum_r(g_r) / world_size applied once by the wrapper",
    )
end

function _optimizer_metadata()
    return Dict(
        "name" => "Optimisers.AdamW",
        "lr" => ETA_MAX,
        "betas" => [BETA[1], BETA[2]],
        "eps" => EPSILON,
        "weight_decay" => WEIGHT_DECAY,
        "couple" => true,
        "group_count" => 2,
        "decay_group" => Dict(
            "weight_decay" => WEIGHT_DECAY,
            "parameter_count" => 21,
            "parameters" => [name for name in bundle_parameter_names() if is_decay_param(name)],
        ),
        "no_decay_group" => Dict(
            "weight_decay" => 0.0,
            "parameter_count" => 41,
            "parameters" => [name for name in bundle_parameter_names() if !is_decay_param(name)],
        ),
    )
end

function _model_metadata()
    return Dict(
        "name" => "Tiny ImageNet ResNet-18, small-image stem, GroupNorm-32",
        "parameter_count" => EXPECTED_PARAMETER_COUNT,
        "parameter_array_count" => EXPECTED_MODEL_ARRAYS,
        "parameter_names" => bundle_parameter_names(),
        "input_layout" => "NCHW",
        "activation_names" => first.(REF_ACTIVATIONS),
    )
end

function _schedule_metadata()
    return Dict(
        "eta_max" => ETA_MAX,
        "betas" => [BETA[1], BETA[2]],
        "eps" => EPSILON,
        "warmup_updates" => WARMUP_UPDATES,
        "total_updates" => TOTAL_UPDATES,
        "rule" => "eta_t = eta_max * t / W for t <= W; cosine from eta_max to 0 for W < t <= T",
        "eta" => Dict("u$(t)" => schedule_eta(t) for t in 1:3),
        "updates" => [1, 2, 3],
    )
end

function _precision_metadata(cfg)
    cfg === nothing && return Dict("configured" => false, "device" => "cpu")
    return Dict(
        "configured" => true,
        "fma" => cfg.fma,
        "cublas_math_mode" => string(cfg.cublas_math_mode),
        "cudnn_requested" => string(cfg.cudnn_requested),
        "cudnn_effective" => string(cfg.cudnn_effective),
        "cudnn_version" => string(cfg.cudnn_version),
        "cuda_package_version" => string(cfg.cuda_package_version),
        "cudnn_package_version" => string(cfg.cudnn_package_version),
        "cudnn_library_version" => string(cfg.cudnn_library_version),
        "versions_supported" => cfg.versions_supported,
        "caches_cleared" => cfg.caches_cleared,
    )
end

function _configure_precision!()
    isdefined(Main, :StaticScalingPrecision) ||
        error("precision.jl was not loaded; start a fresh process with --device cuda")
    return Main.StaticScalingPrecision.configure_precision!(fma = true)
end

function _pin_cpu_threads!(device::AbstractString)
    device == "cpu" || return Threads.nthreads()
    BLAS.set_num_threads(1)
    return BLAS.get_num_threads()
end

function _prepare_device(device::AbstractString)
    if device == "cpu"
        return (to_device = identity, precision = nothing, dev = nothing)
    elseif device == "cuda"
        cfg = _configure_precision!()
        dev = Flux.gpu_device(; force = true)
        return (to_device = dev, precision = cfg, dev = dev)
    else
        error("--device must be cpu or cuda, got $(device)")
    end
end

# ---------------------------------------------------------------------------
# Collective trace verification
# ---------------------------------------------------------------------------

function expected_gradient_buffers(model, state)
    out = Any[]
    for (name, _leaf) in state_leaves(_tree(state))
        a = param_by_name(model, name)
        push!(out, (name = name, buftype = typeof(a), size = size(a)))
    end
    return out
end

"""
    verify_collective_trace(trace, expected)

Enforce the wrapper protocol: one host `Vector{Float32}` presence reduction
sized to the number of trainable buffers, one `avg` reduction per gradient
buffer in wrapper/model order with the parameter type and shape, and one
host `Vector{Float32}` length-1 gradient-validity vote.
"""
function verify_collective_trace(trace, expected)
    length(expected) == EXPECTED_MODEL_ARRAYS ||
        error("expected $(length(expected)) gradient buffers; expected $(EXPECTED_MODEL_ARRAYS)")
    expected_count = 1 + length(expected) + 1
    length(trace) == expected_count ||
        error("collective trace has $(length(trace)) entries; expected $(expected_count)")

    presence = trace[1]
    presence.avg === true || error("presence allreduce did not use avg")
    presence.buftype === Vector{Float32} ||
        error("presence buffer type $(presence.buftype); expected Vector{Float32}")
    presence.size == (length(expected),) ||
        error("presence buffer size $(presence.size); expected $((length(expected),))")

    for (i, exp) in enumerate(expected)
        entry = trace[i + 1]
        entry.avg === true || error("gradient allreduce $(exp.name) did not use avg")
        entry.buftype === exp.buftype ||
            error("gradient buffer $(exp.name) type $(entry.buftype); expected $(exp.buftype)")
        entry.size == exp.size ||
            error("gradient buffer $(exp.name) size $(entry.size); expected $(exp.size)")
    end

    vote = trace[end]
    vote.avg === true || error("gradient-validity vote did not use avg")
    vote.buftype === Vector{Float32} ||
        error("vote buffer type $(vote.buftype); expected Vector{Float32}")
    vote.size == (1,) || error("vote buffer size $(vote.size); expected (1,)")
    return true
end

function trace_sha256(trace)
    ctx = SHA.SHA2_256_CTX()
    for e in trace
        SHA.update!(ctx, codeunits(string(e.buftype)))
        SHA.update!(ctx, UInt8[0x00])
        SHA.update!(ctx, codeunits(join(string.(e.size), "x")))
        SHA.update!(ctx, UInt8[0x00])
        SHA.update!(ctx, UInt8[e.avg ? 0x01 : 0x00])
        SHA.update!(ctx, UInt8[0x0a])
    end
    return bytes2hex(SHA.digest!(ctx))
end

function _canonical_size(name, flux_size)
    if _is_conv_weight_name(name)
        return collect(Int, reverse(flux_size))
    end
    return collect(Int, flux_size)
end

function trace_entries(trace, expected)
    out = Any[]
    for (i, e) in enumerate(trace)
        kind = i == 1 ? "presence" : (i == length(trace) ? "vote" : "gradient")
        name = kind == "gradient" ? expected[i - 1].name : nothing
        push!(out, Dict(
            "index" => i,
            "kind" => kind,
            "name" => name,
            "buftype" => string(e.buftype),
            "size" => collect(Int, e.size),
            "canonical_size" => name === nothing ? nothing : _canonical_size(name, e.size),
            "layout" => name === nothing ? nothing : _parameter_layout(name),
            "avg" => e.avg,
        ))
    end
    return out
end

"""
    canonical_trace_entries(entries, names)

Reorder a recorded trace from the wrapper/model functor call order into the
canonical Phase 3 bundle order expected by `check_phase4.py`, keeping the
presence reduction first and the validity vote last. The unmodified call
order remains available in the summary as `call_order_entries`.
"""
function canonical_trace_entries(entries, names)
    gradients = Dict{String,Any}()
    for entry in entries
        entry["kind"] == "gradient" || continue
        gradients[entry["name"]] = entry
    end
    length(gradients) == length(names) ||
        error("trace has $(length(gradients)) gradient buffers; expected $(length(names))")
    ordered = Any[entries[1]]
    for name in names
        haskey(gradients, name) ||
            error("trace is missing gradient buffer $(name)")
        push!(ordered, gradients[name])
    end
    push!(ordered, entries[end])
    return ordered
end

# ---------------------------------------------------------------------------
# Distributed diagnostics (always on the real backend)
# ---------------------------------------------------------------------------

function backend_comm(backend::Flux.MPIBackend)
    return backend.comm
end

function backend_comm(backend::Flux.NCCLBackend)
    return backend.mpi_backend.comm
end

backend_comm(backend) = error("unsupported backend $(typeof(backend))")

function phase4_barrier(mpi_mod, backend)
    backend === nothing && return nothing
    mpi_mod === nothing && return nothing
    mpi_mod.Barrier(backend_comm(backend))
    return nothing
end

function all_ranks_true(backend, ok::Bool)
    flags = Int32[ok ? 1 : 0]
    Flux.DistributedUtils.allreduce!(backend, flags, +)
    return flags[1] == Flux.DistributedUtils.total_workers(backend)
end

function broadcast_hash(mpi_mod, backend, local_hash::AbstractString, label::AbstractString)
    backend === nothing && return String(local_hash)
    buf = Vector{UInt8}(codeunits(String(local_hash)))
    local_copy = copy(buf)
    Flux.DistributedUtils.bcast!(backend, buf; root = 0)
    root_hash = String(buf)
    String(local_copy) == root_hash ||
        error("$(label) differs between ranks: local=$(String(local_copy)) root=$(root_hash)")
    return root_hash
end

function _ensure_mpi()
    mod = try
        _require_package("MPI", MPI_UUID)
    catch err
        error("MPI.jl is required for --backend mpi: $(sprint(showerror, err))")
    end
    Base.get_extension(Flux, :FluxMPIExt) === nothing &&
        error("FluxMPIExt is not loaded; load MPI.jl before initializing the MPI backend")
    return mod
end

function _ensure_nccl()
    _require_package("CUDA", CUDA_UUID)
    nccl = try
        _require_package("NCCL", NCCL_UUID)
    catch err
        error("NCCL.jl is required for --backend nccl: $(sprint(showerror, err))")
    end
    Base.get_extension(Flux, :FluxMPINCCLExt) === nothing &&
        error("FluxMPINCCLExt is not loaded; load CUDA.jl and NCCL.jl before initializing")
    return nccl
end

function _backend_metadata(backend_name, mpi_mod, real, rank, world, device_index)
    if backend_name == "mpi"
        library = try
            String(mpi_mod.MPI_LIBRARY)
        catch
            "unknown"
        end
        version = try
            String(mpi_mod.MPI_VERSION)
        catch
            "unknown"
        end
        return Dict("backend" => "mpi", "library" => library, "version" => version,
                    "world_size" => world, "rank" => rank, "device_index" => device_index)
    else
        nccl_mod = _loaded_package("NCCL", NCCL_UUID)
        version = nccl_mod === nothing ? "unknown" :
                  try
                      string(Base.pkgversion(nccl_mod))
                  catch
                      "unknown"
                  end
        return Dict("backend" => "nccl", "nccl_version" => version,
                    "communicator_size" => world, "communicator_rank" => rank,
                    "device_index" => device_index)
    end
end

# ---------------------------------------------------------------------------
# CLI parsing and shared validation
# ---------------------------------------------------------------------------

const ALLOWED_OPTS = Set([
    "mode", "device", "backend", "world-size", "fixture-dir", "bundle",
    "out", "rank-out", "force", "source-commit", "source-state-sha256",
    "flux-commit", "flux-state-sha256",
])

function parse_phase4_args(argv)
    opts = parse_cli_args(argv)
    for key in keys(opts)
        key in ALLOWED_OPTS || error("unknown option --$(key)")
    end
    mode = lowercase(get(opts, "mode", ""))
    mode in ("reference", "shard-reference", "ddp", "split-control") ||
        error("--mode must be reference, shard-reference, ddp, or split-control; got $(repr(mode))")
    device = lowercase(get(opts, "device", "cpu"))
    device in ("cpu", "cuda") || error("--device must be cpu or cuda; got $(repr(device))")
    backend = lowercase(get(opts, "backend", "mpi"))
    backend in ("mpi", "nccl") || error("--backend must be mpi or nccl; got $(repr(backend))")
    backend == "nccl" && device != "cuda" &&
        error("--backend nccl requires --device cuda")
    out = get(opts, "out", "")
    isempty(out) && error("--out is required")
    opts["out"] = abspath(out)
    opts["mode"] = mode
    opts["device"] = device
    opts["backend"] = backend
    get(opts, "force", "false") in ("true", "1", "yes") || delete!(opts, "force")
    return opts
end

function _world_size_opt(opts)
    s = get(opts, "world-size", "")
    isempty(s) && return nothing
    v = tryparse(Int, s)
    v === nothing && error("--world-size must be a positive integer, got $(s)")
    v > 0 || error("--world-size must be a positive integer, got $(s)")
    return v
end

# ---------------------------------------------------------------------------
# Reference mode
# ---------------------------------------------------------------------------

function run_reference(opts)
    device = opts["device"]
    _pin_cpu_threads!(device)
    fixture_dir, data_path, bundle_path, fixture_manifest_path = _fixture_paths(opts)
    world_opt = _world_size_opt(opts)
    world_opt === nothing || world_opt == 1 ||
        error("reference mode is single-process; got --world-size $(world_opt)")

    data = read_npz_fields(data_path)
    bundle = read_npz_fields(bundle_path)
    source = source_identity(opts)

    prep = _prepare_device(device)
    model = make_model(bundle)
    model_array_hash(model) == bundle_array_hash(bundle) ||
        error("model parameter hash does not match the loaded bundle")
    if prep.dev !== nothing
        model = prep.dev(model)
    end
    to_device = prep.to_device

    out_dir = joinpath(opts["out"], "reference", "flux", device)
    npz_path = joinpath(out_dir, "phase4_reference.npz")
    manifest_path = joinpath(out_dir, "phase4_reference.json")
    manifest_alias_path = joinpath(out_dir, "phase4_reference_manifest.json")
    _refuse_existing([npz_path, manifest_path, manifest_alias_path]; force = haskey(opts, "force"))

    opt_state = build_optimizer_state(model; eta = schedule_eta(1))
    inputs = Dict{Int,Any}()
    labels = Dict{Int,Any}()
    etas = Dict{Int,Float64}()
    losses = Dict{Int,Float32}()
    logits = Dict{Int,Any}()
    acts_u1 = nothing
    grads_u1 = nothing
    grads_u3 = nothing
    for t in 1:3
        x_nchw, y_np = _batch_for_update(data, t)
        x, y = _to_training_batch(x_nchw, y_np, to_device)
        eta = schedule_eta(t)
        res = train_step!(model, opt_state, x, y; eta = eta,
                          capture_activations = (t == 1))
        model = res.model
        opt_state = res.opt_state
        inputs[t] = x_nchw
        labels[t] = y_np
        etas[t] = eta
        losses[t] = loss32(res.loss)
        logits[t] = host_array(res.logits)
        t == 1 && (acts_u1 = res.activations; grads_u1 = res.grads)
        t == 3 && (grads_u3 = res.grads)
    end
    acts_u1 === nothing && error("u1 activations were not captured")
    grads_u3 === nothing && error("u3 gradients were not captured")

    entries = capture_reference_arrays(inputs, labels, etas, losses, logits,
                                       acts_u1, grads_u1, grads_u3, model, opt_state)
    meta = array_metadata(entries)

    _write_npz(npz_path, entries)
    _verify_npz(npz_path, entries)

    manifest = Dict(
        "producer" => "scripts/static_scaling/phase4_flux.jl",
        "created" => utc_now(),
        "synthetic" => false,
        "schema" => "phase4-reference-flux-v1; arrays entries use raw npz key names; " *
                    "sha256 = sha256 of C-order bytes",
        "mode" => "reference",
        "framework" => "flux",
        "device" => device,
        "source" => source,
        "package_versions" => package_versions(),
        "precision" => _precision_metadata(prep.precision),
        "fixture_dir" => fixture_dir,
        "fixture_sha256" => _fixture_hashes(fixture_dir, data_path, bundle_path,
                                            fixture_manifest_path),
        "data_fixture_sha256" => sha256_file(data_path),
        "bundle_path" => bundle_path,
        "bundle_sha256" => sha256_file(bundle_path),
        "initial_bundle_sha256" => sha256_file(bundle_path),
        "bundle_array_hash" => bundle_array_hash(bundle),
        "model_parameter_count" => EXPECTED_PARAMETER_COUNT,
        "model_array_count" => EXPECTED_MODEL_ARRAYS,
        "optimizer" => _optimizer_metadata(),
        "schedule" => _schedule_metadata(),
        "model" => _model_metadata(),
        "training" => _training_metadata(),
        "reference_file" => Dict(
            "path" => basename(npz_path),
            "bytes" => filesize(npz_path),
            "sha256" => sha256_file(npz_path),
            "array_count" => length(entries),
        ),
        "arrays" => meta,
    )
    write_json(manifest_path, manifest)
    write_json(manifest_alias_path, manifest)
    println("PHASE4_FLUX_REFERENCE PASS arrays=$(length(entries)) " *
            "npz=$(npz_path) manifest=$(manifest_path) " *
            "npz_sha256=$(manifest["reference_file"]["sha256"])")
    return 0
end

# ---------------------------------------------------------------------------
# Shard reference mode
# ---------------------------------------------------------------------------

function run_shard_reference(opts)
    device = opts["device"]
    _pin_cpu_threads!(device)
    fixture_dir, data_path, bundle_path, fixture_manifest_path = _fixture_paths(opts)
    world_opt = _world_size_opt(opts)
    W = world_opt === nothing ? 1 : world_opt
    W in (1, 2, 4) ||
        error("shard-reference mode requires --world-size in (1, 2, 4); got $(repr(world_opt))")

    data = read_npz_fields(data_path)
    bundle = read_npz_fields(bundle_path)
    source = source_identity(opts)

    prep = _prepare_device(device)
    model = make_model(bundle)
    model_array_hash(model) == bundle_array_hash(bundle) ||
        error("model parameter hash does not match the loaded bundle")
    if prep.dev !== nothing
        model = prep.dev(model)
    end
    to_device = prep.to_device

    out_dir = joinpath(opts["out"], "shard_reference", "flux", device, "w$(W)")
    npz_path = joinpath(out_dir, "phase4_reference.npz")
    manifest_path = joinpath(out_dir, "phase4_reference.json")
    manifest_alias_path = joinpath(out_dir, "phase4_reference_manifest.json")
    _refuse_existing([npz_path, manifest_path, manifest_alias_path]; force = haskey(opts, "force"))

    opt_state = build_optimizer_state(model; eta = schedule_eta(1))
    inputs = Dict{Int,Any}()
    labels = Dict{Int,Any}()
    etas = Dict{Int,Float64}()
    losses = Dict{Int,Float32}()
    logits = Dict{Int,Any}()
    acts_u1 = nothing
    grads_u1 = nothing
    grads_u3 = nothing
    for t in 1:3
        x_nchw, y_np = _batch_for_update(data, t)
        eta = schedule_eta(t)
        losses_r = Float32[]
        logits_r = Any[]
        acts_r = Any[]
        grads_list = Any[]
        for r in 1:W
            x_local, y_local, _ = shard_batch(x_nchw, y_np, r, W)
            x, y = _to_training_batch(x_local, y_local, to_device)
            res = train_step!(model, opt_state, x, y; eta = eta,
                              capture_activations = (t == 1),
                              apply_update = false)
            push!(losses_r, loss32(res.loss))
            push!(logits_r, host_array(res.logits))
            t == 1 && push!(acts_r, res.activations)
            push!(grads_list, res.grads)
        end
        g_avg = Functors.fmap(
            (gs...) -> gs[1] === nothing ? nothing : (.+(gs...)) ./ Float32(W),
            grads_list...,
        )
        Optimisers.adjust!(opt_state, eta)
        opt_state, model = Optimisers.update!(opt_state, model, g_avg)

        inputs[t] = x_nchw
        labels[t] = y_np
        etas[t] = eta
        losses[t] = Float32(sum(losses_r) / Float32(W))
        logits[t] = hcat(logits_r...)
        if t == 1
            act_keys = keys(acts_r[1])
            acts_u1 = NamedTuple{act_keys}(Tuple(
                cat([host_array(getproperty(a, k)) for a in acts_r]...;
                    dims = ndims(getproperty(acts_r[1], k)))
                for k in act_keys
            ))
            grads_u1 = g_avg
        end
        t == 3 && (grads_u3 = g_avg)
    end
    acts_u1 === nothing && error("u1 activations were not captured")
    grads_u3 === nothing && error("u3 gradients were not captured")

    entries = capture_reference_arrays(inputs, labels, etas, losses, logits,
                                       acts_u1, grads_u1, grads_u3, model, opt_state)
    meta = array_metadata(entries)

    _write_npz(npz_path, entries)
    _verify_npz(npz_path, entries)

    manifest = Dict(
        "producer" => "scripts/static_scaling/phase4_flux.jl",
        "created" => utc_now(),
        "synthetic" => false,
        "schema" => "phase4-reference-flux-v1; arrays entries use raw npz key names; " *
                    "sha256 = sha256 of C-order bytes",
        "mode" => "shard-reference",
        "world_size" => W,
        "framework" => "flux",
        "device" => device,
        "source" => source,
        "package_versions" => package_versions(),
        "precision" => _precision_metadata(prep.precision),
        "fixture_dir" => fixture_dir,
        "fixture_sha256" => _fixture_hashes(fixture_dir, data_path, bundle_path,
                                            fixture_manifest_path),
        "data_fixture_sha256" => sha256_file(data_path),
        "bundle_path" => bundle_path,
        "bundle_sha256" => sha256_file(bundle_path),
        "initial_bundle_sha256" => sha256_file(bundle_path),
        "bundle_array_hash" => bundle_array_hash(bundle),
        "model_parameter_count" => EXPECTED_PARAMETER_COUNT,
        "model_array_count" => EXPECTED_MODEL_ARRAYS,
        "optimizer" => _optimizer_metadata(),
        "schedule" => _schedule_metadata(),
        "model" => _model_metadata(),
        "training" => _training_metadata(),
        "reference_file" => Dict(
            "path" => basename(npz_path),
            "bytes" => filesize(npz_path),
            "sha256" => sha256_file(npz_path),
            "array_count" => length(entries),
        ),
        "arrays" => meta,
    )
    write_json(manifest_path, manifest)
    write_json(manifest_alias_path, manifest)
    println("PHASE4_FLUX_SHARD_REFERENCE PASS world_size=$(W) arrays=$(length(entries)) " *
            "npz=$(npz_path) manifest=$(manifest_path) " *
            "npz_sha256=$(manifest["reference_file"]["sha256"])")
    return 0
end

# ---------------------------------------------------------------------------
# Split control mode
# ---------------------------------------------------------------------------

function _gradient_without_update(model, opt_state, x_nchw, y_np, eta, to_device)
    x, y = _to_training_batch(x_nchw, y_np, to_device)
    res = train_step!(model, opt_state, x, y; eta = eta, capture_activations = false,
                      apply_update = false)
    return res.grads
end

function _half_batches(x_nchw, labels)
    n = size(x_nchw, 1)
    iseven(n) || error("split-control requires an even global batch, got $(n)")
    h = n ÷ 2
    return (x_nchw[1:h, :, :, :], labels[1:h]),
           (x_nchw[(h + 1):n, :, :, :], labels[(h + 1):n])
end

function run_split_control(opts)
    device = opts["device"]
    _pin_cpu_threads!(device)
    fixture_dir, data_path, bundle_path, fixture_manifest_path = _fixture_paths(opts)
    world_opt = _world_size_opt(opts)
    world_opt === nothing || world_opt == 1 ||
        error("split-control is single-process; got --world-size $(world_opt)")

    data = read_npz_fields(data_path)
    bundle = read_npz_fields(bundle_path)
    source = source_identity(opts)
    prep = _prepare_device(device)
    model = make_model(bundle)
    prep.dev !== nothing && (model = prep.dev(model))
    to_device = prep.to_device
    opt_state = build_optimizer_state(model; eta = schedule_eta(1))
    names = bundle_parameter_names()

    out_dir = joinpath(opts["out"], "split", "flux", device)
    npz_path = joinpath(out_dir, "split_control.npz")
    json_path = joinpath(out_dir, "split_control.json")
    _refuse_existing([npz_path, json_path]; force = haskey(opts, "force"))

    entries = Pair{String,Any}[]
    diagnostics = Dict{String,Any}()
    for t in (1, 3)
        tag = "u$(t)"
        x_nchw, labels = _batch_for_update(data, t)
        eta = schedule_eta(t)
        (xa, ya), (xb, yb) = _half_batches(x_nchw, labels)
        g_full = _gradient_map(
            _gradient_without_update(model, opt_state, x_nchw, labels, eta, to_device), names)
        g_a = _gradient_map(
            _gradient_without_update(model, opt_state, xa, ya, eta, to_device), names)
        g_b = _gradient_map(
            _gradient_without_update(model, opt_state, xb, yb, eta, to_device), names)
        g_avg = Dict(name => (g_a[name] .+ g_b[name]) ./ 2.0f0 for name in names)
        max_abs = 0.0
        worst = "-"
        max_l2 = 0.0
        for name in names
            d = maximum(abs.(Float64.(g_full[name]) .- Float64.(g_avg[name])); init = 0.0)
            if d > max_abs
                max_abs = d
                worst = name
            end
            max_l2 = max(max_l2, sqrt(sum(abs2, Float64.(g_full[name]) .- Float64.(g_avg[name]))))
        end
        push!(entries, "input.$(tag)" => x_nchw)
        push!(entries, "label.$(tag)" => labels)
        push!(entries, "eta.$(tag)" => Float64[eta])
        for (kind, gradients) in (("full", g_full), ("half1", g_a),
                                  ("half2", g_b), ("halfavg", g_avg))
            for name in names
                push!(entries, "grad.$(tag).$(kind).$(name)" => gradients[name])
            end
        end
        diagnostics[tag] = Dict(
            "full_vs_half_average_max_abs" => max_abs,
            "full_vs_half_average_max_l2" => max_l2,
            "worst_leaf" => worst,
            "half_batch_size" => size(xa, 1),
            "gradient_average_rule" => "(g_half1 + g_half2) / 2",
        )
        println("PHASE4_FLUX_SPLIT_CONTROL_UPDATE $(tag) " *
                "max_abs=$(fmt_float(max_abs)) max_l2=$(fmt_float(max_l2)) worst=$(worst)")
    end

    _write_npz(npz_path, entries)
    _verify_npz(npz_path, entries)
    record = Dict(
        "producer" => "scripts/static_scaling/phase4_flux.jl",
        "created" => utc_now(),
        "mode" => "split-control",
        "framework" => "flux",
        "device" => device,
        "source" => source,
        "package_versions" => package_versions(),
        "precision" => _precision_metadata(prep.precision),
        "fixture_dir" => fixture_dir,
        "fixture_sha256" => _fixture_hashes(fixture_dir, data_path, bundle_path,
                                            fixture_manifest_path),
        "data_fixture_sha256" => sha256_file(data_path),
        "bundle_path" => bundle_path,
        "bundle_sha256" => sha256_file(bundle_path),
        "updates" => 3,
        "input_updates" => [1, 3],
        "diagnostics" => diagnostics,
        "npz_file" => Dict(
            "path" => basename(npz_path),
            "bytes" => filesize(npz_path),
            "sha256" => sha256_file(npz_path),
            "array_count" => length(entries),
        ),
        "arrays" => array_metadata(entries),
    )
    write_json(json_path, record)
    println("PHASE4_FLUX_SPLIT_CONTROL PASS npz=$(npz_path) json=$(json_path)")
    return 0
end

# ---------------------------------------------------------------------------
# DDP mode
# ---------------------------------------------------------------------------

function _cuda_device_index(rank::Integer, world::Integer)
    cuda = _loaded_package("CUDA", CUDA_UUID)
    cuda === nothing && error("CUDA.jl is not loaded for --device cuda")
    count = cuda.ndevices()
    count > 0 || error("no CUDA devices are visible")
    expected = mod(rank + 1, count)
    actual = Int(cuda.deviceid(cuda.device()))
    actual == expected ||
        error("rank $(rank) is on CUDA device $(actual); expected $(expected) " *
              "from (rank + 1) % $(count)")
    return actual
end

function run_ddp(opts)
    device = opts["device"]
    backend_name = opts["backend"]
    _pin_cpu_threads!(device)
    requested_world = _world_size_opt(opts)
    fixture_dir, data_path, bundle_path, fixture_manifest_path = _fixture_paths(opts)

    mpi_mod = _ensure_mpi()
    backend_name == "nccl" && _ensure_nccl()
    backend_type = backend_name == "mpi" ? Flux.MPIBackend : Flux.NCCLBackend
    Flux.DistributedUtils.initialize(backend_type)
    real = Flux.DistributedUtils.get_distributed_backend(backend_type)
    rank = Flux.DistributedUtils.local_rank(real)
    world = Flux.DistributedUtils.total_workers(real)
    requested_world === nothing || requested_world == world ||
        error("--world-size $(requested_world) does not match backend world size $(world)")
    device_index = device == "cuda" ? _cuda_device_index(rank, world) : 0
    precision_cfg = device == "cuda" ? _configure_precision!() : nothing

    rec = RecordingBackend(real)
    source = source_identity(opts)
    data = read_npz_fields(data_path)
    bundle = read_npz_fields(bundle_path)

    # Every rank must load exactly the same values.
    local_bundle_hash = bundle_array_hash(bundle)
    root_bundle_hash = broadcast_hash(mpi_mod, real, local_bundle_hash, "initial bundle")
    root_bundle_hash == local_bundle_hash ||
        error("rank $(rank) bundle hash $(local_bundle_hash) differs from root $(root_bundle_hash)")

    model = make_model(bundle)
    model_array_hash(model) == local_bundle_hash ||
        error("rank $(rank) model parameter hash does not match the bundle")
    to_device = identity
    if device == "cuda"
        dev = Flux.gpu_device(; force = true)
        model = dev(model)
        to_device = dev
    end
    state = build_distributed_state(model, rec; eta = schedule_eta(1))

    rank_dir = get(opts, "rank-out", "")
    isempty(rank_dir) &&
        (rank_dir = joinpath(opts["out"], "ddp", "flux", "w$(world)"))
    rank_dir = abspath(rank_dir)
    mkpath(rank_dir)
    rank_tag = @sprintf("rank%d", rank)
    npz_path = joinpath(rank_dir, rank_tag * ".npz")
    json_path = joinpath(rank_dir, rank_tag * ".json")
    summary_path = joinpath(rank_dir, "summary.json")

    local_ok = haskey(opts, "force") ||
               (!isfile(npz_path) && !isfile(json_path) &&
                (rank != 0 || !isfile(summary_path)))
    all_ranks_true(real, local_ok) ||
        error("refusing to overwrite existing rank outputs in $(rank_dir); use --force")

    names = bundle_parameter_names()
    global_ids = collect(1:12)
    local_ids = rank_shard(global_ids, rank + 1, world)
    inputs = Dict{Int,Any}()
    labels = Dict{Int,Any}()
    etas = Dict{Int,Float64}()
    losses = Dict{Int,Float32}()
    logits = Dict{Int,Any}()
    localgrads = Dict{Int,Any}()
    state_hash_history = Dict{String,Any}()
    trace_records = Dict{String,Any}()
    trace_full = Dict{String,Any}()
    trace_call_order = Dict{String,Any}()

    for t in 1:3
        tag = "u$(t)"
        x_nchw, y_np = _batch_for_update(data, t)
        x_local, y_local, _ = shard_batch(x_nchw, y_np, rank + 1, world)
        x, y = _to_training_batch(x_local, y_local, to_device)
        eta = schedule_eta(t)

        expected = expected_gradient_buffers(model, state)
        empty!(rec.log)
        res = train_step!(model, state, x, y; eta = eta, capture_activations = false)
        trace = copy(rec.log)
        verify_collective_trace(trace, expected)
        localgrads[t] = _gradient_map(res.grads, names)

        model = res.model
        state = res.opt_state
        inputs[t] = x_local
        labels[t] = y_local
        etas[t] = eta
        losses[t] = loss32(res.loss)
        logits[t] = host_array(res.logits)
        state_hash_history[tag] = state_hashes(model, state; names = names)
        trace_records[tag] = Dict(
            "count" => length(trace),
            "sha256" => trace_sha256(trace),
            "presence_size" => collect(Int, trace[1].size),
            "gradient_buffers" => length(expected),
            "vote_size" => collect(Int, trace[end].size),
        )
        rank == 0 && begin
            call_order = trace_entries(trace, expected)
            trace_call_order[tag] = call_order
            trace_full[tag] = canonical_trace_entries(call_order, names)
        end
        println("PHASE4_FLUX_DDP_UPDATE rank=$(rank) tag=$(tag) " *
                "collectives=$(length(trace)) loss=$(fmt_float(losses[t]))")
    end

    entries = capture_ddp_arrays(rank, world, device_index, inputs, labels, etas,
                                 losses, logits, localgrads, model, state)
    _write_npz(npz_path, entries)
    array_count = _verify_npz(npz_path, entries)

    ddp_config = Dict(
        "framework" => "flux",
        "entry_point" => "Zygote.withgradient + Optimisers.update! on DistributedOptimizerState",
        "communication" => "outside automatic differentiation, inside Optimisers.update!",
        "gradient_average" => "g = sum_r(g_r) / world_size applied exactly once",
        "loss_scaling" => "none",
        "wrapper" => "RecordingBackend over the real $(backend_name) backend",
        "collective_protocol" => "1 presence Float32(62) + 62 gradient + 1 vote Float32(1) = 64",
        "updates" => EXPECTED_UPDATES,
    )
    rank_record = Dict(
        "producer" => "scripts/static_scaling/phase4_flux.jl",
        "created" => utc_now(),
        "mode" => "ddp",
        "framework" => "flux",
        "device" => device,
        "backend" => backend_name,
        "world_size" => world,
        "rank" => rank,
        "device_index" => device_index,
        "updates" => EXPECTED_UPDATES,
        "shard_rule" => "rank_shard(global 1:12, Julia rank $(rank + 1), world $(world))",
        "shard_ids" => Dict("u1" => local_ids, "u3" => local_ids,
                            "u1_0based" => local_ids .- 1,
                            "u3_0based" => local_ids .- 1),
        "source" => source,
        "package_versions" => package_versions(),
        "precision" => _precision_metadata(precision_cfg),
        "ddp_configuration" => ddp_config,
        "backend_configuration" => _backend_metadata(
            backend_name, mpi_mod, real, rank, world, device_index),
        "fixture_dir" => fixture_dir,
        "fixture_sha256" => _fixture_hashes(fixture_dir, data_path, bundle_path,
                                            fixture_manifest_path),
        "data_fixture_sha256" => sha256_file(data_path),
        "bundle_path" => bundle_path,
        "bundle_sha256" => sha256_file(bundle_path),
        "bundle_array_hash" => local_bundle_hash,
        "model_parameter_count" => EXPECTED_PARAMETER_COUNT,
        "model_array_count" => EXPECTED_MODEL_ARRAYS,
        "optimizer" => _optimizer_metadata(),
        "schedule" => _schedule_metadata(),
        "model" => _model_metadata(),
        "training" => _training_metadata(),
        "state_hashes" => state_hash_history,
        "collective_trace" => trace_records,
        "npz_file" => Dict(
            "path" => basename(npz_path),
            "bytes" => filesize(npz_path),
            "sha256" => sha256_file(npz_path),
            "array_count" => array_count,
        ),
        "arrays" => array_metadata(entries),
    )
    write_json(json_path, rank_record)

    phase4_barrier(mpi_mod, real)
    if rank == 0
        summary = Dict(
            "producer" => "scripts/static_scaling/phase4_flux.jl",
            "created" => utc_now(),
            "mode" => "ddp",
            "framework" => "flux",
            "device" => device,
            "backend" => backend_name,
            "world_size" => world,
            "updates" => EXPECTED_UPDATES,
            "rank_files" => [@sprintf("rank%d.json", r) for r in 0:(world - 1)],
            "collective_trace" => Dict(
                tag => Dict(
                    "count" => trace_records[tag]["count"],
                    "sha256" => trace_records[tag]["sha256"],
                    "gradient_order" => "entries: canonical Phase 3 bundle order; " *
                                        "call_order_entries: actual wrapper/model functor order",
                    "entries" => trace_full[tag],
                    "call_order_entries" => trace_call_order[tag],
                ) for tag in ("u1", "u2", "u3")
            ),
            "source" => source,
            "bundle_array_hash" => local_bundle_hash,
        )
        write_json(summary_path, summary)
        println("PHASE4_FLUX_DDP PASS rank=0 world=$(world) " *
                "summary=$(summary_path)")
    else
        println("PHASE4_FLUX_DDP PASS rank=$(rank) world=$(world) " *
                "npz=$(npz_path)")
    end
    return 0
end

end # module StaticScalingPhase4
